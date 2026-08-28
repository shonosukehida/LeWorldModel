"""JEPA Implementation"""

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn


def detach_clone(v):
    return v.detach().clone() if torch.is_tensor(v) else v


class JEPA(nn.Module):

    def __init__(
        self,
        overhead_encoder,
        wrist_encoder,
        predictor,
        action_encoder,
        prop_encoder,
        overhead_projector=None,
        wrist_projector=None,
        pred_proj=None,
        # action_space="",
    ):
        super().__init__()

        self.overhead_encoder = overhead_encoder
        self.wrist_encoder = wrist_encoder

        self.predictor = predictor
        self.action_encoder = action_encoder
        self.prop_encoder = prop_encoder

        self.overhead_projector = (
            overhead_projector or nn.Identity()
        )
        self.wrist_projector = (
            wrist_projector or nn.Identity()
        )

        self.pred_proj = pred_proj or nn.Identity()

        # self.action_space = action_space

    def encode(self, info):
        """Encode multiview pixels, proprioception, and actions.

        Expected inputs:
            pixels:
                Overhead camera images.
                Shape: (B, T, C, H, W)

            wrist_pixels:
                Wrist camera images.
                Shape: (B, T, C, H, W)

            proprio:
                Shape: (B, T, 8)

            action/action_joint/action_cartesian:
                Shape: (B, T, action_dim)

        Outputs:
            overhead_img_emb:
                Shape: (B, T, overhead_embed_dim)

            wrist_img_emb:
                Shape: (B, T, wrist_embed_dim)

            prop_emb:
                Shape: (B, T, prop_embed_dim)

            emb:
                Shape:
                (
                    B,
                    T,
                    overhead_embed_dim
                    + wrist_embed_dim
                    + prop_embed_dim
                )

            act_emb:
                Shape: (B, T, state_embed_dim),
                if action exists
        """

        if "pixels" not in info:
            raise KeyError("pixels not found in info")

        if "wrist_pixels" not in info:
            raise KeyError("wrist_pixels not found in info")

        if "proprio" not in info:
            raise KeyError("proprio not found in info")

        overhead_pixels = info["pixels"].float()
        wrist_pixels = info["wrist_pixels"].float()
        proprio = info["proprio"].float()

        #
        # Shape validation
        #

        if overhead_pixels.ndim != 5:
            raise ValueError(
                "pixels must have shape (B, T, C, H, W), "
                f"got {overhead_pixels.shape}"
            )

        if wrist_pixels.ndim != 5:
            raise ValueError(
                "wrist_pixels must have shape (B, T, C, H, W), "
                f"got {wrist_pixels.shape}"
            )

        if proprio.ndim != 3:
            raise ValueError(
                "proprio must have shape (B, T, D), "
                f"got {proprio.shape}"
            )

        if overhead_pixels.shape[:2] != wrist_pixels.shape[:2]:
            raise ValueError(
                "pixels and wrist_pixels must have the same "
                "batch and time dimensions, "
                f"got pixels={overhead_pixels.shape[:2]} and "
                f"wrist_pixels={wrist_pixels.shape[:2]}"
            )

        if overhead_pixels.shape[:2] != proprio.shape[:2]:
            raise ValueError(
                "pixels and proprio must have the same "
                "batch and time dimensions, "
                f"got pixels={overhead_pixels.shape[:2]} and "
                f"proprio={proprio.shape[:2]}"
            )

        batch_size = overhead_pixels.size(0)

        #
        # Flatten B,T
        #

        # (B, T, C, H, W) -> (B*T, C, H, W)
        overhead_pixels_flat = rearrange(
            overhead_pixels,
            "b t ... -> (b t) ...",
        )

        wrist_pixels_flat = rearrange(
            wrist_pixels,
            "b t ... -> (b t) ...",
        )

        # (B, T, 8) -> (B*T, 8)
        proprio_flat = rearrange(
            proprio,
            "b t d -> (b t) d",
        )

        #
        # Overhead image latent
        #

        overhead_encoder_output = self.overhead_encoder(
            overhead_pixels_flat,
            interpolate_pos_encoding=True,
        )

        overhead_features = (
            overhead_encoder_output.last_hidden_state[:, 0]
        )

        z_overhead_flat = self.overhead_projector(
            overhead_features
        )

        #
        # Wrist image latent
        #

        wrist_encoder_output = self.wrist_encoder(
            wrist_pixels_flat,
            interpolate_pos_encoding=True,
        )

        wrist_features = (
            wrist_encoder_output.last_hidden_state[:, 0]
        )

        z_wrist_flat = self.wrist_projector(
            wrist_features
        )

        #
        # Proprio latent
        #

        z_prop_flat = self.prop_encoder(
            proprio_flat
        )

        #
        # Concatenate multimodal state
        #
        # [overhead | wrist | proprio]
        #

        emb_flat = torch.cat(
            [
                z_overhead_flat,
                z_wrist_flat,
                z_prop_flat,
            ],
            dim=-1,
        )

        #
        # Restore B,T
        #

        z_overhead = rearrange(
            z_overhead_flat,
            "(b t) d -> b t d",
            b=batch_size,
        )

        z_wrist = rearrange(
            z_wrist_flat,
            "(b t) d -> b t d",
            b=batch_size,
        )

        z_prop = rearrange(
            z_prop_flat,
            "(b t) d -> b t d",
            b=batch_size,
        )

        emb = rearrange(
            emb_flat,
            "(b t) d -> b t d",
            b=batch_size,
        )

        info["overhead_img_emb"] = z_overhead
        info["wrist_img_emb"] = z_wrist
        info["prop_emb"] = z_prop
        info["emb"] = emb

        #
        # Action embedding
        #

        if "action" in info:
            info["act_emb"] = self.action_encoder(
                info["action"]
            )

        elif "action_joint" in info:
            info["act_emb"] = self.action_encoder(
                info["action_joint"]
            )

        elif "action_cartesian" in info:
            info["act_emb"] = self.action_encoder(
                info["action_cartesian"]
            )

        return info

    def predict(self, emb, act_emb):
        """Predict next state embedding.

        emb:
            (B, T, D)

        act_emb:
            (B, T, A_emb)
        """

        preds = self.predictor(
            emb,
            act_emb,
        )

        preds = self.pred_proj(
            rearrange(
                preds,
                "b t d -> (b t) d",
            )
        )

        preds = rearrange(
            preds,
            "(b t) d -> b t d",
            b=emb.size(0),
        )

        return preds

    ####################
    ## Inference only ##
    ####################

    def rollout(
        self,
        info,
        action_sequence,
        history_size: int = 3,
    ):
        """Rollout the model given an initial info dict and action sequence.

        pixels:
            (B, S, T, C, H, W)

        wrist_pixels:
            (B, S, T, C, H, W)

        proprio:
            (B, S, T, D)

        action_sequence:
            (B, S, T, action_dim)

        S:
            number of action plan samples

        T:
            planning horizon
        """

        assert "pixels" in info, "pixels not in info_dict"
        assert (
            "wrist_pixels" in info
        ), "wrist_pixels not in info_dict"

        H = info["pixels"].size(2)

        if info["wrist_pixels"].size(2) != H:
            raise ValueError(
                "pixels and wrist_pixels must have the same "
                "history length in rollout, "
                f"got {H} and {info['wrist_pixels'].size(2)}"
            )

        B, S, T = action_sequence.shape[:3]

        act_0, act_future = torch.split(
            action_sequence,
            [H, T - H],
            dim=2,
        )

        info["action"] = act_0

        n_steps = T - H

        #
        # Copy and encode initial observations
        #

        _init = {
            k: v[:, 0]
            for k, v in info.items()
            if torch.is_tensor(v)
        }

        _init = self.encode(_init)

        emb = (
            _init["emb"]
            .unsqueeze(1)
            .expand(B, S, -1, -1)
        )

        info["emb"] = emb

        _init = {
            k: detach_clone(v)
            for k, v in _init.items()
        }

        #
        # Flatten batch and CEM sample dimensions
        #

        emb = rearrange(
            emb,
            "b s ... -> (b s) ...",
        ).clone()

        act = rearrange(
            act_0,
            "b s ... -> (b s) ...",
        )

        act_future = rearrange(
            act_future,
            "b s ... -> (b s) ...",
        )

        #
        # Autoregressive rollout
        #

        HS = history_size

        for t in range(n_steps):
            act_emb = self.action_encoder(act)

            emb_trunc = emb[:, -HS:]
            act_trunc = act_emb[:, -HS:]

            pred_emb = self.predict(
                emb_trunc,
                act_trunc,
            )[:, -1:]

            emb = torch.cat(
                [emb, pred_emb],
                dim=1,
            )

            next_act = act_future[
                :,
                t : t + 1,
                :,
            ]

            act = torch.cat(
                [act, next_act],
                dim=1,
            )

        #
        # Predict last state
        #

        act_emb = self.action_encoder(act)

        emb_trunc = emb[:, -HS:]
        act_trunc = act_emb[:, -HS:]

        pred_emb = self.predict(
            emb_trunc,
            act_trunc,
        )[:, -1:]

        emb = torch.cat(
            [emb, pred_emb],
            dim=1,
        )

        #
        # Restore batch / CEM sample dimensions
        #

        pred_rollout = rearrange(
            emb,
            "(b s) ... -> b s ...",
            b=B,
            s=S,
        )

        info["predicted_emb"] = pred_rollout

        return info

    def criterion(self, info_dict: dict):
        """Compute cost between predicted embeddings and goal embeddings."""

        pred_emb = info_dict["predicted_emb"]
        goal_emb = info_dict["goal_emb"]

        goal_emb = (
            goal_emb[..., -1:, :]
            .expand_as(pred_emb)
        )

        cost = F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction="none",
        ).sum(
            dim=tuple(
                range(
                    2,
                    pred_emb.ndim,
                )
            )
        )

        return cost

    def get_cost(
        self,
        info_dict: dict,
        action_candidates: torch.Tensor,
    ):
        """Compute cost of action candidates."""

        assert "goal" in info_dict, "goal not in info_dict"
        assert (
            "goal_wrist_pixels" in info_dict
        ), "goal_wrist_pixels not in info_dict"

        device = next(
            self.parameters()
        ).device

        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(
                    device
                )

        goal = {
            k: v[:, 0]
            for k, v in info_dict.items()
            if torch.is_tensor(v)
        }

        #
        # Overhead goal image
        #

        goal["pixels"] = goal["goal"]

        #
        # Convert:
        # goal_wrist_pixels -> wrist_pixels
        # goal_proprio      -> proprio
        #

        for k in list(goal.keys()):
            if k.startswith("goal_"):
                new_key = k[len("goal_") :]
                goal[new_key] = goal.pop(k)

        #
        # Actions are not required for goal encoding
        #

        for k in [
            "action",
            "action_joint",
            "action_cartesian",
        ]:
            goal.pop(k, None)

        goal = self.encode(goal)

        info_dict["goal_emb"] = goal["emb"]

        info_dict = self.rollout(
            info_dict,
            action_candidates,
        )

        cost = self.criterion(
            info_dict
        )

        return cost