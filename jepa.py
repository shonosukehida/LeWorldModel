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
        encoder,
        predictor,
        action_encoder,
        prop_encoder,
        projector=None,
        pred_proj=None,
        # action_space="",
    ):
        super().__init__()

        self.encoder = encoder
        self.predictor = predictor
        self.action_encoder = action_encoder
        self.prop_encoder = prop_encoder
        self.projector = projector or nn.Identity()
        self.pred_proj = pred_proj or nn.Identity()
        # self.action_space = action_space


    def encode(self, info):
        """Encode pixels, proprioception, and actions.

        Expected inputs:
            pixels:
                (B, T, C, H, W)

            proprio:
                (B, T, 8)

            action/action_joint/action_cartesian:
                (B, T, action_dim)

        Outputs:
            img_emb:
                (B, T, img_embed_dim)

            prop_emb:
                (B, T, prop_embed_dim)

            emb:
                (B, T, img_embed_dim + prop_embed_dim)

            act_emb:
                (B, T, state_embed_dim), if action exists
        """
        if "pixels" not in info:
            raise KeyError("pixels not found in info")

        if "proprio" not in info:
            raise KeyError("proprio not found in info")

        pixels = info["pixels"].float()
        proprio = info["proprio"].float()

        if pixels.ndim != 5:
            raise ValueError(
                "pixels must have shape (B, T, C, H, W), "
                f"got {pixels.shape}"
            )

        if proprio.ndim != 3:
            raise ValueError(
                "proprio must have shape (B, T, D), "
                f"got {proprio.shape}"
            )

        if pixels.shape[:2] != proprio.shape[:2]:
            raise ValueError(
                "pixels and proprio must have the same "
                "batch and time dimensions, "
                f"got pixels={pixels.shape[:2]} and "
                f"proprio={proprio.shape[:2]}"
            )

        batch_size = pixels.size(0)

        # (B, T, C, H, W) -> (B*T, C, H, W)
        pixels_flat = rearrange(
            pixels,
            "b t ... -> (b t) ...",
        )

        # (B, T, 8) -> (B*T, 8)
        proprio_flat = rearrange(
            proprio,
            "b t d -> (b t) d",
        )

        # 画像潜在
        encoder_output = self.encoder(
            pixels_flat,
            interpolate_pos_encoding=True,
        )

        pixels_features = (
            encoder_output.last_hidden_state[:, 0]
        )

        z_img_flat = self.projector(
            pixels_features
        )

        # Proprio潜在
        z_prop_flat = self.prop_encoder(
            proprio_flat
        )
        # print("z_prop_flat.shape: ", z_prop_flat.shape)

        emb_flat = torch.cat(
            [z_img_flat, z_prop_flat],
            dim=-1,
        )

        z_img = rearrange(
            z_img_flat,
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

        info["img_emb"] = z_img
        info["prop_emb"] = z_prop
        info["emb"] = emb

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
        """Predict next state embedding
        emb: (B, T, D)
        act_emb: (B, T, A_emb)
        """
        preds = self.predictor(emb, act_emb)
        preds = self.pred_proj(rearrange(preds, "b t d -> (b t) d"))
        preds = rearrange(preds, "(b t) d -> b t d", b=emb.size(0))
        return preds

    ####################
    ## Inference only ##
    ####################

    def rollout(self, info, action_sequence, history_size: int = 3):
        """Rollout the model given an initial info dict and action sequence.
        pixels: (B, S, T, C, H, W)
        action_sequence: (B, S, T, action_dim)
         - S is the number of action plan samples
         - T is the time horizon
        """
        # print("(jepa.py) info.keys(): ", info.keys())
        # print("(jepa.py) info(action): ", info["action"])

        assert "pixels" in info, "pixels not in info_dict"
        H = info["pixels"].size(2)
        B, S, T = action_sequence.shape[:3]
        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)
        info["action"] = act_0 #コメントアウトしても動作する --> 使われてない, info['action] が計算に使われていないぽいので、正常範囲内
        # print("(jepa.py)(after input act_0) info(action): ", info["action"])
        n_steps = T - H

        # copy and encode initial info dict
        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}
        _init = self.encode(_init)
        emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)
        _init = {k: detach_clone(v) for k, v in _init.items()}

        # flatten batch and sample dimensions for rollout
        emb = rearrange(emb, "b s ... -> (b s) ...").clone()
        act = rearrange(act_0, "b s ... -> (b s) ...")
        act_future = rearrange(act_future, "b s ... -> (b s) ...")

        # rollout predictor autoregressively for n_steps
        HS = history_size
        for t in range(n_steps):
            act_emb = self.action_encoder(act)
            emb_trunc = emb[:, -HS:]  # (BS, HS, D)
            act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
            pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
            emb = torch.cat([emb, pred_emb], dim=1)  # (BS, T+1, D)

            next_act = act_future[:, t : t + 1, :]  # (BS, 1, action_dim)
            act = torch.cat([act, next_act], dim=1)  # (BS, T+1, action_dim)

        # predict the last state
        act_emb = self.action_encoder(act)  # (BS, T, A_emb)
        emb_trunc = emb[:, -HS:]  # (BS, HS, D)
        act_trunc = act_emb[:, -HS:]  # (BS, HS, A_emb)
        pred_emb = self.predict(emb_trunc, act_trunc)[:, -1:]  # (BS, 1, D)
        emb = torch.cat([emb, pred_emb], dim=1)

        # unflatten batch and sample dimensions
        pred_rollout = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)
        info["predicted_emb"] = pred_rollout

        return info

    def criterion(self, info_dict: dict):
        """Compute the cost between predicted embeddings and goal embeddings."""
        pred_emb = info_dict["predicted_emb"]  # (B,S, T-1, dim)
        goal_emb = info_dict["goal_emb"]  # (B, S, T, dim)
        
        #pred_emb
        # print("pred_emb.shape:", pred_emb.shape)
        # print("pred_emb.min:", pred_emb.min())
        # print("pred_emb.max:", pred_emb.max())
        # print("pred_emb.mean:", pred_emb.mean())
        
        #goal_emb
        # print("goal_emb.shape:", goal_emb.shape)
        # print("goal_emb.min:", goal_emb.min())
        # print("goal_emb.max:", goal_emb.max())
        # print("goal_emb.mean:", goal_emb.mean())
        

        goal_emb = goal_emb[..., -1:, :].expand_as(pred_emb)

        # return last-step cost per action candidate
        cost = F.mse_loss(
            pred_emb[..., -1:, :],
            goal_emb[..., -1:, :].detach(),
            reduction="none",
        ).sum(dim=tuple(range(2, pred_emb.ndim)))  # (B, S)

        return cost

    def get_cost(self, info_dict: dict, action_candidates: torch.Tensor):
        """ Compute the cost of action candidates given an info dict with goal and initial state."""

        assert "goal" in info_dict, "goal not in info_dict"

        device = next(self.parameters()).device
        for k in list(info_dict.keys()):
            if torch.is_tensor(info_dict[k]):
                info_dict[k] = info_dict[k].to(device)

        goal = {k: v[:, 0] for k, v in info_dict.items() if torch.is_tensor(v)}
        goal["pixels"] = goal["goal"]

        for k in info_dict:
            if k.startswith("goal_"):
                goal[k[len("goal_") :]] = goal.pop(k)

        for k in ["action", "action_joint", "action_cartesian"]:
            goal.pop(k, None)
            
            
        goal = self.encode(goal)

        info_dict["goal_emb"] = goal["emb"]
        info_dict = self.rollout(info_dict, action_candidates)

        cost = self.criterion(info_dict)
        
        return cost
