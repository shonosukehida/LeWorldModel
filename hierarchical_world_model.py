"""Inference-only composition of independently trained lower and upper JEPAs.

Public inputs are RAW observations and PHYSICAL Cartesian actions, before GPC's
_prepare_info/action normalization. No GPC adapter or combined checkpoint format
is provided here. Each layer rolls out exactly once per get_cost call.
"""
from pathlib import Path
import math

import numpy as np
import torch
from torch import nn

from jepa import JEPA
from upper_jepa import UpperJEPA


class _ObservationTransform(nn.Module):
    """Training image transform and a layer's own proprio statistics."""
    def __init__(self, process, img_size):
        super().__init__()
        from utils import get_img_preprocessor
        self.images = {key: get_img_preprocessor(key, key, img_size)
                       for key in ("pixels", "wrist_pixels")}
        self.register_buffer("mean", torch.as_tensor(process["proprio"].mean_, dtype=torch.float32).reshape(8).clone())
        self.register_buffer("scale", torch.as_tensor(process["proprio"].scale_, dtype=torch.float32).reshape(8).clone())

    def forward(self, info, device):
        # Copies also protect callers from in-place dictionary transforms/models.
        result = {key: info[key].to(device).clone() for key in ("pixels", "wrist_pixels", "proprio")}
        for key, transform in self.images.items():
            result = transform(result)
        prop = result["proprio"].float()
        result["proprio"] = (prop - self.mean.to(prop)) / self.scale.to(prop)
        return result


def _validate_process(process, keys, layer):
    for key in keys:
        if key not in process:
            raise ValueError(f"{layer}: missing required normalization processor {key}")
        for name in ("mean_", "scale_"):
            value = np.asarray(getattr(process[key], name, None))
            if value.shape not in ((8,), (1, 8)) or not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
                raise ValueError(f"{layer}: {key}.{name} must contain eight finite values")
            if name == "scale_" and (value <= 0).any():
                raise ValueError(f"{layer}: {key}.scale_ must be strictly positive")


def _number(value, name, positive=False):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(value) or (positive and value <= 0):
        raise ValueError(f"{name} must be finite" + (" and strictly positive" if positive else ""))
    return value


class HierarchicalWorldModel(nn.Module):
    """Compose inference; do not copy, retrain, or freeze either model's weights.

    info keys: pixels, wrist_pixels=(B,1,3,Y,X), proprio=(B,1,8).
    get_cost additionally requires goal, goal_wrist_pixels=(B,1,3,Y,X),
    goal_proprio=(B,1,8). Images are raw uint8 or unnormalized floats in the
    training-supported ranges. No candidate axis and no preprocessed features
    are accepted. actions=(B,K,T,8) are physical xyzw Cartesian targets.

    rollout returns lower_rollout and upper_rollout, the original model result
    dictionaries. get_cost additionally returns lower_cost, upper_cost,
    normalized_lower_cost, normalized_upper_cost, total_cost, all (B,K).
    """
    def __init__(self, lower_world_model, upper_world_model, *, lower_process,
                 upper_process, lower_img_size, upper_img_size, lower_history_size=3,
                 lower_weight=1.0, upper_weight=1.0, lower_cost_scale=1.0,
                 upper_cost_scale=1.0):
        super().__init__()
        if not isinstance(lower_world_model, JEPA) or isinstance(lower_world_model, UpperJEPA):
            raise TypeError("lower_world_model must be a Cartesian single-step JEPA")
        if not isinstance(upper_world_model, UpperJEPA):
            raise TypeError("upper_world_model must be an UpperJEPA")
        # Reject shared tensors instead of silently changing learned weights.
        def storage_ids(model):
            return {(tensor.device, tensor.untyped_storage().data_ptr())
                    for tensor in (*model.parameters(), *model.buffers()) if tensor.numel()}
        if storage_ids(lower_world_model) & storage_ids(upper_world_model):
            raise ValueError("Lower and Upper models must not share parameters or buffers")
        if isinstance(lower_history_size, bool) or not isinstance(lower_history_size, int) or not 1 <= lower_history_size <= 3:
            raise ValueError("lower_history_size must be an integer from 1 to 3")
        capacity = getattr(lower_world_model.predictor, "pos_embedding", None)
        if capacity is not None and capacity.shape[1] < lower_history_size:
            raise ValueError("lower_history_size exceeds the trained lower predictor capacity")
        patch = getattr(lower_world_model.action_encoder, "patch_embed", None)
        if patch is not None and patch.in_channels != 8:
            raise ValueError("Lower action encoder must accept 8D Cartesian actions (frameskip=1)")
        if upper_world_model.actions_normalized:
            raise ValueError("UpperJEPA must accept physical actions: actions_normalized=False")
        _validate_process(lower_process, ("action_cartesian", "proprio"), "lower")
        upper_keys = ("action_cartesian", "proprio") + (("action_delta",) if upper_world_model.use_action_delta else ())
        _validate_process(upper_process, upper_keys, "upper")
        upper_world_model._validate_statistics()
        # Upper action statistics live inside the model; prevent a mismatched
        # external observation/statistics bundle from silently being selected.
        pairs = [("action_cartesian", "action_mean", "action_scale")]
        if upper_world_model.use_action_delta:
            pairs.append(("action_delta", "delta_mean", "delta_scale"))
        for key, mean, scale in pairs:
            for attr, source in ((mean, "mean_"), (scale, "scale_")):
                actual = getattr(upper_world_model, attr)
                expected = torch.as_tensor(getattr(upper_process[key], source)).reshape(8).to(actual)
                if not torch.equal(actual.reshape(8), expected):
                    raise ValueError(f"upper: {key}.{source} does not match UpperJEPA checkpoint buffers")
        self.lower_world_model = lower_world_model
        self.upper_world_model = upper_world_model
        self.lower_history_size = lower_history_size
        self.lower_weight = _number(lower_weight, "lower_weight")
        self.upper_weight = _number(upper_weight, "upper_weight")
        self.lower_cost_scale = _number(lower_cost_scale, "lower_cost_scale", positive=True)
        self.upper_cost_scale = _number(upper_cost_scale, "upper_cost_scale", positive=True)
        self.lower_observation_transform = _ObservationTransform(lower_process, lower_img_size)
        self.upper_observation_transform = _ObservationTransform(upper_process, upper_img_size)
        self.register_buffer("lower_action_mean", torch.as_tensor(lower_process["action_cartesian"].mean_, dtype=torch.float32).reshape(8).clone())
        self.register_buffer("lower_action_scale", torch.as_tensor(lower_process["action_cartesian"].scale_, dtype=torch.float32).reshape(8).clone())
        self.eval()

    def _device(self):
        def device(model):
            tensor = next(model.parameters(), None)
            return tensor.device if tensor is not None else self.lower_action_mean.device
        lower, upper = device(self.lower_world_model), device(self.upper_world_model)
        if lower != upper:
            raise ValueError("Move both models to the same device with hierarchical_model.to(device)")
        return lower

    @staticmethod
    def _observations(info, batch, goal=False):
        mapping = {"pixels": "goal", "wrist_pixels": "goal_wrist_pixels", "proprio": "goal_proprio"} if goal else {
            key: key for key in ("pixels", "wrist_pixels", "proprio")}
        result = {}
        for target, key in mapping.items():
            if key not in info:
                raise ValueError(f"Missing required raw observation: {key}")
            value = info[key]
            if not torch.is_tensor(value):
                raise ValueError(f"{key} must be a Tensor")
            expected = (batch, 1, 8) if target == "proprio" else None
            if expected is not None:
                valid = value.shape == expected
            else:
                valid = value.ndim == 5 and value.shape[:3] == (batch, 1, 3) and min(value.shape[-2:]) > 0
            if not valid:
                shape = "(B,1,8)" if target == "proprio" else "(B,1,3,Y,X)"
                raise ValueError(f"{key} must have shape {shape}; provide one raw observation without a candidate axis")
            if not torch.isfinite(value).all():
                raise ValueError(f"{key} contains NaN/Inf")
            result[target] = value
        return result

    def rollout(self, info, action_sequences):
        """One rollout per layer; both terminal latents represent time t+T."""
        if not torch.is_tensor(action_sequences) or action_sequences.ndim != 4 or action_sequences.shape[-1] != 8:
            raise ValueError("action_sequences must have shape (B,K,T,8)")
        b, k, t, _ = action_sequences.shape
        if min(b, k, t) < 1:
            raise ValueError("B, K and T must be positive")
        if t % self.upper_world_model.horizon:
            raise ValueError("T must be divisible by upper_world_model.horizon; no truncation or padding")
        if not torch.isfinite(action_sequences).all():
            raise ValueError("action_sequences contains NaN/Inf")
        current = self._observations(info, b)
        self.eval()  # Preserve eval inference even if a parent has called train().
        device = self._device()
        physical = action_sequences.to(device).clone()
        if not physical.is_floating_point():
            physical = physical.float()
        lower_actions = (physical - self.lower_action_mean.to(physical)) / self.lower_action_scale.to(physical)
        def layer_info(transform):
            transformed = transform(current, device)
            return {key: value[:, None].expand(b, k, *value.shape[1:]).clone()
                    for key, value in transformed.items()}
        lower = self.lower_world_model.rollout(layer_info(self.lower_observation_transform),
                                               lower_actions, history_size=self.lower_history_size)
        upper = self.upper_world_model.rollout(layer_info(self.upper_observation_transform),
                                               physical.clone(), history_size=min(3, self.upper_world_model.history_size))
        return {"lower_rollout": lower, "upper_rollout": upper}

    def get_cost(self, info, action_sequences):
        """Return raw/normalized/weighted costs without a second rollout."""
        if not torch.is_tensor(action_sequences) or action_sequences.ndim != 4:
            raise ValueError("action_sequences must have shape (B,K,T,8)")
        b, k = action_sequences.shape[:2]
        goal = self._observations(info, b, goal=True)
        result = self.rollout(info, action_sequences)
        device = self._device()
        for layer in ("lower", "upper"):
            model = getattr(self, f"{layer}_world_model")
            transform = getattr(self, f"{layer}_observation_transform")
            goal_emb = model.encode(transform(goal, device))["emb"]
            # criterion expects an explicit candidate axis; (B,1,D) alone
            # cannot broadcast safely to (B,K,L,D) when B != K.
            output = result[f"{layer}_rollout"]
            output["goal_emb"] = goal_emb[:, None]
            cost = model.criterion(output)
            if cost.shape != (b, k) or not torch.isfinite(cost).all():
                raise ValueError(f"{layer}_cost must be finite with shape (B,K)")
            result[f"{layer}_cost"] = cost
            scale = _number(getattr(self, f"{layer}_cost_scale"), f"{layer}_cost_scale", positive=True)
            normalized = cost / scale
            if not torch.isfinite(normalized).all():
                raise ValueError(f"normalized_{layer}_cost contains NaN/Inf")
            result[f"normalized_{layer}_cost"] = normalized
        total = (_number(self.lower_weight, "lower_weight") * result["normalized_lower_cost"] +
                 _number(self.upper_weight, "upper_weight") * result["normalized_upper_cost"])
        if not torch.isfinite(total).all():
            raise ValueError("total_cost contains NaN/Inf")
        result["total_cost"] = total
        return result

    @classmethod
    def from_checkpoints(cls, lower_checkpoint, upper_checkpoint, lower_stats_path,
                         lower_config_path=None, device="cpu", **cost_config):
        """Use the existing lower object-loader and upper Lightning loader.

        Lower must be a *_object.ckpt file or run directory containing one,
        with config.yaml (or an explicit lower_config_path) describing training.
        Lower statistics are not embedded: supply the corresponding saved NPZ.
        Upper restores its own embedded statistics and architecture.
        """
        from omegaconf import OmegaConf
        from normalization_stats import load_normalization_process
        from upper_training import load_upper_checkpoint
        from stable_worldmodel.policy import AutoCostModel
        for name, value in (("lower_checkpoint", lower_checkpoint), ("upper_checkpoint", upper_checkpoint),
                            ("lower_stats_path", lower_stats_path)):
            if value is None:
                raise ValueError(f"{name} is required")
            if not Path(value).expanduser().exists():
                raise FileNotFoundError(f"{name}: {value}")
        lower_path = Path(lower_checkpoint).expanduser().resolve()
        if lower_path.is_dir():
            if not list(lower_path.glob("*_object.ckpt")):
                raise FileNotFoundError(f"No *_object.ckpt in {lower_path}")
            run = str(lower_path)
            config_path = lower_path / "config.yaml"
        else:
            if not lower_path.name.endswith("_object.ckpt"):
                raise ValueError("Lower loader requires *_object.ckpt, not a weights-only checkpoint; use ModelObjectCallBack output")
            # AutoCostModel appends _object.ckpt to file prefixes; passing a
            # complete existing filename would otherwise append it a second time.
            run = str(lower_path)[:-len("_object.ckpt")]
            config_path = lower_path.parent / "config.yaml"
        config_path = Path(lower_config_path).expanduser() if lower_config_path is not None else config_path
        if not config_path.is_file():
            raise FileNotFoundError(f"Lower training config not found: {config_path}; provide lower_config_path")
        cfg = OmegaConf.load(config_path)
        try:
            img_size = int(cfg.img_size)
            history = int(cfg.wm.history_size)
            single_step = int(cfg.wm.num_preds) == 1 and int(cfg.data.dataset.frameskip) == 1
            cartesian = "action_cartesian" in cfg.data.dataset.keys_to_load
        except (AttributeError, KeyError, TypeError, ValueError) as exc:
            raise ValueError("Lower config must specify img_size, wm.history_size/num_preds and Cartesian dataset frameskip/keys_to_load") from exc
        if not single_step or not cartesian or history < 1:
            raise ValueError("Lower checkpoint must be trained with action_cartesian, frameskip=1 and num_preds=1; use a matching single-step model")
        lower = AutoCostModel(run)
        upper = load_upper_checkpoint(Path(upper_checkpoint).expanduser(), map_location="cpu")
        lower_process, _ = load_normalization_process(Path(lower_stats_path).expanduser())
        result = cls(lower, upper.model, lower_process=lower_process, upper_process=upper.process,
                     lower_img_size=img_size, upper_img_size=upper.config["img_size"],
                     lower_history_size=min(3, history), **cost_config)
        return result.to(device).eval()
