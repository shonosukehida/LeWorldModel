"""Physical Cartesian action features shared by UpperJEPA and stats creation.

Inputs use [x,y,z,qx,qy,qz,qw,gripper]. These functions do not standardize
features or fit statistics. All computation stays in torch for differentiability.
"""
import torch


def canonical_quaternion(q):
    """Unit xyzw quaternion; first nonzero in w,x,y,z order is positive."""
    norms = q.norm(dim=-1, keepdim=True)
    if (norms < 1e-8).any() or not torch.isfinite(norms).all():
        raise ValueError("cannot normalize a zero or nonfinite quaternion")
    q = q / norms
    ordered = q[..., [3, 0, 1, 2]]
    first = (ordered != 0).to(torch.int64).argmax(-1, keepdim=True)
    sign = torch.where(ordered.gather(-1, first) < 0, -1, 1)
    return q * sign


def action_delta(actions):
    """(..., T, 8) physical actions -> (..., T-1, 8) physical deltas.

    Position/gripper use subtraction. Orientation is inv(q_t)*q_(t+1),
    normalized and sign-canonicalized using canonical_quaternion.
    Callers must split episodes and filter discontinuous pairs before calling.
    """
    if actions.ndim < 2 or actions.shape[-1] != 8 or actions.shape[-2] < 2:
        raise ValueError("action_delta requires shape (..., T, 8) with T >= 2")
    actions = actions if actions.is_floating_point() else actions.float()
    if not torch.isfinite(actions).all():
        raise ValueError("action_delta requires finite physical actions")
    q = canonical_quaternion(actions[..., 3:7])
    v0, w0 = q[..., :-1, :3], q[..., :-1, 3:]
    v1, w1 = q[..., 1:, :3], q[..., 1:, 3:]
    relative = torch.cat((w0 * v1 - w1 * v0 - torch.linalg.cross(v0, v1),
                          w0 * w1 + (v0 * v1).sum(-1, keepdim=True)), -1)
    return torch.cat((actions[..., 1:, :3] - actions[..., :-1, :3],
                      canonical_quaternion(relative),
                      actions[..., 1:, 7:] - actions[..., :-1, 7:]), -1)
