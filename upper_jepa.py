"""Independent upper-level JEPA; one transition spans ``horizon`` base steps.

Observation preprocessing is unchanged. Action inputs are physical Cartesian
xyzw actions by default. For standardized inputs, set actions_normalized=True
and supply the *existing pipeline's* action_mean/action_scale (safe std).
Absolute features use the existing action_cartesian statistics; deltas use
separate externally fitted statistics. No statistics are fitted or defaulted
here. Use build_macro_action in training as well as inference.
"""
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn

from action_features import action_delta
from jepa import JEPA
from module import ARPredictor, Embedder, MLP


class UpperJEPA(JEPA):
    """Same 192+192+32 encoder layout as JEPA, with independent modules.

    Optional supplied modules are copied, including buffers, so a lower model's
    modules can safely serve as architecture templates. Defaults match lewm.yaml.
    encode accepts already-built macro features as ``action`` (B,K,macro_dim),
    or observations alone. Raw actions must go through build_macro_action first.

    Supply action_mean/action_scale from the existing action_cartesian scaler,
    and delta_mean/delta_scale fitted on physical deltas using this quaternion
    convention. Scales are the pipeline's safe standard deviations (no further
    epsilon/clamping is applied here). Optionally normalization_stats_path loads
    either process_stats.npz (action_cartesian and action_delta processors) or
    a non-pickled NPZ with those exact keys. Do not mix file and array inputs.
    Missing statistics are permitted at construction for observation-only use,
    but every macro build and encode with actions validates required buffers.
    """

    def __init__(self, horizon, use_action_delta=True, *, history_size=3,
                 actions_normalized=False, action_mean=None, action_scale=None,
                 delta_mean=None, delta_scale=None, normalization_stats_path=None,
                 overhead_encoder=None, wrist_encoder=None, prop_encoder=None,
                 overhead_projector=None, wrist_projector=None, predictor=None,
                 pred_proj=None):
        if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
            raise ValueError("horizon must be a positive integer")
        if isinstance(history_size, bool) or not isinstance(history_size, int) or history_size < 1:
            raise ValueError("history_size must be a positive integer")
        macro_dim = 8 * (horizon + (horizon - 1 if use_action_delta else 0))

        def image_encoder(template):
            if template is not None:
                return deepcopy(template)
            import stable_pretraining as spt
            return spt.backbone.utils.vit_hf(
                "tiny", patch_size=14, image_size=224,
                pretrained=False, use_mask_token=False)

        def own(template, factory):
            return deepcopy(template) if template is not None else factory()

        def projector(output_dim):
            return MLP(192, 2048, output_dim, norm_fn=nn.BatchNorm1d)

        super().__init__(
            overhead_encoder=image_encoder(overhead_encoder),
            wrist_encoder=image_encoder(wrist_encoder),
            prop_encoder=own(prop_encoder, lambda: MLP(8, 64, 32)),
            overhead_projector=own(overhead_projector, lambda: projector(192)),
            wrist_projector=own(wrist_projector, lambda: projector(192)),
            predictor=own(predictor, lambda: ARPredictor(
                num_frames=history_size, input_dim=416, hidden_dim=192,
                output_dim=192, depth=6, heads=16, mlp_dim=2048,
                dim_head=64, dropout=0.1, emb_dropout=0.0)),
            pred_proj=own(pred_proj, lambda: projector(416)),
            action_encoder=Embedder(input_dim=macro_dim, emb_dim=416),
        )
        self.horizon = horizon
        self.history_size = history_size
        self.use_action_delta = use_action_delta
        self.macro_action_dim = macro_dim
        self.actions_normalized = actions_normalized
        values = dict(action_mean=action_mean, action_scale=action_scale,
                      delta_mean=delta_mean, delta_scale=delta_scale)
        if normalization_stats_path is not None:
            path = Path(normalization_stats_path).expanduser()
            if not path.is_file():
                raise FileNotFoundError(f"Normalization statistics file not found: {path}")
            if any(value is not None for value in values.values()):
                raise ValueError("provide normalization statistics as arrays or a file, not both")
            required = self._required_statistics()
            try:
                with np.load(path, allow_pickle=False) as stats:
                    is_process = "metadata" in stats
                    if not is_process:
                        missing = [name for name in required if name not in stats]
                        if missing:
                            raise ValueError(f"Missing normalization statistics: {', '.join(missing)}")
                        values = {name: stats[name].copy() if name in stats else None for name in values}
                if is_process:
                    from normalization_stats import load_normalization_process
                    process, _ = load_normalization_process(path)
                    for prefix, key in (("action", "action_cartesian"), ("delta", "action_delta")):
                        if prefix == "delta" and not self.use_action_delta:
                            continue
                        if key not in process:
                            raise ValueError(f"Missing required normalization processor: {key}")
                        values[f"{prefix}_mean"] = process[key].mean_
                        values[f"{prefix}_scale"] = process[key].scale_
            except (OSError, KeyError, TypeError) as exc:
                raise ValueError(f"Cannot read normalization statistics from {path}: {exc}") from exc
        for name, value in values.items():
            try:
                tensor = None if value is None else torch.as_tensor(value, dtype=torch.float32).detach().clone()
            except (TypeError, ValueError, RuntimeError) as exc:
                raise ValueError(f"{name} must contain eight finite numeric values") from exc
            if tensor is not None:
                self._validate_statistic(name, tensor)
                tensor = tensor.reshape(8)
            self.register_buffer(name, tensor)

    def _required_statistics(self):
        names = ["action_mean", "action_scale"]
        if self.use_action_delta:
            names += ["delta_mean", "delta_scale"]
        return names

    @staticmethod
    def _validate_statistic(name, value):
        if value.shape not in ((8,), (1, 8)) or not torch.isfinite(value).all():
            raise ValueError(f"{name} must have shape (8,) or (1, 8) and contain finite values")
        if name.endswith("scale") and (value <= 0).any():
            raise ValueError(f"{name} must contain strictly positive standard deviations")

    def _validate_statistics(self):
        for name in self._required_statistics():
            value = getattr(self, name)
            if value is None:
                raise ValueError(f"Missing required normalization statistic: {name}")
            self._validate_statistic(name, value)

    def encode(self, info):
        """Reuse JEPA encoding; action features must already be normalized macros."""
        if any(key in info for key in ("action", "action_joint", "action_cartesian")):
            self._validate_statistics()
        return super().encode(info)

    def build_macro_action(self, actions):
        """(..., H, 8) -> (..., macro_dim), including (B,H,8).

        Delta orientation is the local rotation inv(q_t) * q_(t+1), xyzw.
        Unit relative quaternions have a deterministic sign (first nonzero
        component in w,x,y,z order positive), including exactly 180 degrees.
        Absolute actions, including raw quaternion magnitude/sign, use the
        existing action_cartesian standardization without preprocessing.
        Unit normalization and sign canonicalization apply only to deltas.
        """
        self._validate_statistics()
        if actions.ndim < 3 or actions.shape[-2:] != (self.horizon, 8):
            raise ValueError("actions must have shape (..., horizon, 8)")
        actions = actions if actions.is_floating_point() else actions.float()
        if self.actions_normalized:
            actions = actions * self.action_scale.to(actions) + self.action_mean.to(actions)
        if not torch.isfinite(actions).all():
            raise ValueError("actions must be finite")
        delta = None
        if self.use_action_delta and self.horizon > 1:
            delta = action_delta(actions)
        absolute = ((actions - self.action_mean.to(actions)) / self.action_scale.to(actions)).flatten(-2)
        if delta is None:
            return absolute
        delta = (delta - self.delta_mean.to(actions)) / self.delta_scale.to(actions)
        return torch.cat((absolute, delta.flatten(-2)), -1)

    def rollout(self, info, action_sequence, history_size=None):
        """Roll out direct H-step transitions using inherited encode/predict.

        Start from one current observation (K=1): the first macro predicts
        z_(t0+H), then predicted states grow the history up to history_size.
        Observations: (B,S,K,...) at t0,t0+H,...,t0+(K-1)H.
        Actions: (B,S,N*H,8), starting at t0, N >= K. Historical
        actions must be included, just as in JEPA.rollout. All S observations
        must represent the same history; candidate actions may differ.
        Output: (B,S,N+1,416), history followed by predictions through t0+N*H.
        The final macro always produces a prediction. For K=1, H=4 and
        eight base actions, the output contains [z_t0, z_(t0+4), z_(t0+8)].
        A partial final macro is rejected instead of silently truncating it.
        """
        hs = self.history_size if history_size is None else history_size
        if not isinstance(hs, int) or isinstance(hs, bool) or not 1 <= hs <= self.history_size:
            raise ValueError("history_size must be within the configured history capacity")
        if action_sequence.ndim != 4 or action_sequence.shape[-1] != 8:
            raise ValueError("action_sequence must have shape (B,S,N*H,8)")
        b, s, steps, _ = action_sequence.shape
        if steps % self.horizon:
            raise ValueError("action_sequence length must be divisible by horizon")
        k = info["pixels"].shape[2]
        n = steps // self.horizon
        if k < 1 or n < k:
            raise ValueError("at least one macro-action is required per history observation")
        for key, ndim in (("pixels", 6), ("wrist_pixels", 6), ("proprio", 4)):
            value = info[key]
            if value.ndim != ndim or value.shape[:3] != (b, s, k):
                raise ValueError(f"{key} has incompatible batch/sample/history dimensions")
        initial = self.encode({key: info[key][:, 0] for key in
                               ("pixels", "wrist_pixels", "proprio")})["emb"]
        emb = initial[:, None].expand(b, s, k, -1)
        info["emb"] = emb
        emb = emb.reshape(b * s, k, -1).clone()
        macros = self.build_macro_action(action_sequence.reshape(b * s, n, self.horizon, 8))
        act_emb = self.action_encoder(macros)
        for index in range(k - 1, n):
            # emb ends at z_(t0+index*H). Pair each available state with the
            # macro starting at that same time; never include future macros.
            context_length = min(emb.size(1), hs)
            state_history = emb[:, -context_length:]
            action_history = act_emb[:, index + 1 - context_length:index + 1]
            pred = self.predict(state_history, action_history)[:, -1:]
            emb = torch.cat((emb, pred), dim=1)
        info["predicted_emb"] = emb.reshape(b, s, n + 1, -1)
        return info
