"""IMLE-only checkpoint selection and Cartesian interface validation."""

import numpy as np


def validate_checkpoint_selection(cfg):
    kind = cfg.get("policy_type", "world_model")
    gpc = cfg.get("gpc", {})
    imle = gpc.get("imle_checkpoint")
    if isinstance(imle, str) and imle.strip().lower() in ("", "none", "null"):
        raise ValueError(
            "gpc.imle_checkpoint must be a path or YAML null, not a null-like string"
        )
    dp = gpc.get("diffusion_checkpoint")
    if kind == "gpc":
        if (dp is not None) == (imle is not None):
            raise ValueError(
                "GPC requires exactly one of diffusion_checkpoint and imle_checkpoint"
            )
        if (
            imle is not None
            and isinstance(dp, str)
            and dp.strip().lower() in ("none", "null", "")
        ):
            raise ValueError("Use YAML null for gpc.diffusion_checkpoint with IMLE-GPC")
    elif kind == "imle":
        if imle is None or dp is not None:
            raise ValueError(
                "IMLE requires imle_checkpoint and diffusion_checkpoint: null"
            )
    # Legacy diffusion's missing-value interpretation is deliberately unchanged.
    return imle


def validate_imle_action(action, env, current_ee):
    """Check raw values and verify the unchanged environment safety clip result.

    Layout and units are validated at checkpoint load. The environment remains
    authoritative for gripper clipping, workspace, position and rotation limits.
    """
    raw = np.asarray(action).reshape(-1)
    if raw.shape != (8,) or not np.isfinite(raw).all():
        raise ValueError("IMLE Cartesian action must contain eight finite values")
    if np.linalg.norm(raw[3:7]) < 1e-6:
        raise ValueError("IMLE Cartesian quaternion has near-zero norm")
    safe = np.asarray(env.clip_cartesian_action(raw, current_ee=current_ee)).reshape(-1)
    bounds = np.asarray(env.cfg.workspace_bounds_m)
    if safe.shape != (8,) or not np.isfinite(safe).all() or not 0 <= safe[7] <= 1:
        raise ValueError("Invalid IMLE Cartesian action after existing safety clipping")
    if np.any(safe[:3] < bounds[:, 0] - 1e-6) or np.any(safe[:3] > bounds[:, 1] + 1e-6):
        raise ValueError("IMLE action violates workspace bounds after clipping")
    # Workspace clipping can legitimately exceed the delta bound if the robot is
    # already outside the workspace: reject that IMLE command explicitly.
    if np.any(
        np.abs(safe[:3] - np.asarray(current_ee)[:3])
        > float(env.cfg.max_cartesian_delta_m) + 1e-6
    ):
        raise ValueError(
            "IMLE action violates Cartesian delta after workspace clipping"
        )
