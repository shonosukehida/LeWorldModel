"""GPC-RANK using physical DP proposals and HierarchicalWorldModel costs."""
import numpy as np
import torch
from stable_worldmodel.policy import GPCPolicy


COST_KEYS = ('lower_cost', 'upper_cost', 'normalized_lower_cost', 'normalized_upper_cost', 'total_cost')


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


class HierarchicalGPCPolicy(GPCPolicy):
    """Reuse GPC's BasePolicy/env interface, override candidate evaluation only.

    get_action returns physical numpy (B,8), exactly one step per call.
    dp_info_dict is required and contains raw DP observation history (NHWC,
    matching the existing DiffusionPolicy interface). info_dict contains one raw
    current observation plus raw goals. raw_image_layout explicitly specifies
    NHWC or NCHW; conversion changes layout/type only, never normalization.
    last_diagnostics holds detached CPU cost arrays, best_idx, selected_action
    and a monotonically increasing selection_id; it never holds large rollouts.
    """
    def __init__(self, diffusion_policy, hierarchical_world_model, num_candidates=50,
                 action_horizon=None, raw_image_layout='NHWC'):
        _positive_int(num_candidates, 'num_candidates')
        for name in ('obs_horizon', 'pred_horizon', 'action_horizon', 'action_dim'):
            _positive_int(getattr(diffusion_policy, name, None), f'diffusion_policy.{name}')
        if diffusion_policy.action_dim != 8:
            raise ValueError('DP must generate 8D Cartesian actions')
        if diffusion_policy.obs_horizon > diffusion_policy.pred_horizon:
            raise ValueError('DP obs_horizon exceeds pred_horizon')
        action_horizon = diffusion_policy.action_horizon if action_horizon is None else action_horizon
        _positive_int(action_horizon, 'action_horizon')
        _positive_int(hierarchical_world_model.upper_world_model.horizon, 'Upper horizon')
        if raw_image_layout not in ('NHWC', 'NCHW'):
            raise ValueError('raw_image_layout must explicitly be NHWC or NCHW')
        super().__init__(diffusion_policy=diffusion_policy, world_model=hierarchical_world_model,
                         reward_fn=None, num_candidates=num_candidates, action_horizon=action_horizon)
        self.hierarchical_world_model = hierarchical_world_model
        self.raw_image_layout = raw_image_layout
        self.type = 'hierarchical_gpc'
        self.last_diagnostics = None
        self._selection_id = 0

    def _raw_wm_info(self, info):
        result = {}
        for key in ('pixels', 'wrist_pixels', 'goal', 'goal_wrist_pixels', 'proprio', 'goal_proprio'):
            if key not in info:
                raise ValueError(f'Missing raw HWM observation: {key}')
            value = torch.as_tensor(info[key]).clone()
            if key in ('proprio', 'goal_proprio'):
                if value.ndim != 3 or value.shape[1:] != (1, 8):
                    raise ValueError(f'{key} must have shape (B,1,8)')
            else:
                channel_axis = -1 if self.raw_image_layout == 'NHWC' else 2
                if value.ndim != 5 or value.shape[1] != 1 or value.shape[channel_axis] != 3:
                    raise ValueError(f'{key} must contain one raw RGB frame in {self.raw_image_layout}')
                if self.raw_image_layout == 'NHWC':
                    value = value.permute(0, 1, 4, 2, 3).contiguous()
            if not torch.isfinite(value).all():
                raise ValueError(f'{key} contains NaN/Inf')
            result[key] = value
        b = result['pixels'].shape[0]
        if b < 1 or any(value.shape[0] != b for value in result.values()):
            raise ValueError('Raw HWM observations must have matching nonempty batch dimensions')
        return result

    @torch.no_grad()
    def get_action(self, info_dict, *, dp_info_dict=None, **kwargs):
        self.last_diagnostics = None
        if dp_info_dict is None:
            raise ValueError('dp_info_dict is required: keep DP history separate from HWM current observation')
        raw_wm_info = self._raw_wm_info(info_dict)
        b = raw_wm_info['pixels'].shape[0]
        for key in ('pixels', 'wrist_pixels'):
            if key not in dp_info_dict or dp_info_dict[key].ndim != 5 or tuple(dp_info_dict[key].shape[:2]) != (b, self.diffusion_policy.obs_horizon) or dp_info_dict[key].shape[-1] != 3:
                raise ValueError(f'DP {key} must contain exactly obs_horizon raw frames for every batch')
            if not torch.isfinite(torch.as_tensor(dp_info_dict[key])).all():
                raise ValueError(f'DP {key} contains NaN/Inf')
        proposals = self.diffusion_policy.sample_action_sequences(
            dp_info_dict, num_samples=self.num_candidates, denormalize=True)
        expected = (b, self.num_candidates, self.diffusion_policy.pred_horizon, 8)
        if not torch.is_tensor(proposals) or proposals.shape != expected:
            raise ValueError(f'DP candidate actions must have shape {expected}')
        if not torch.isfinite(proposals).all():
            raise ValueError('DP candidate actions contain NaN/Inf')
        start = self.diffusion_policy.obs_horizon - 1
        end = min(start + self.action_horizon, proposals.shape[2])
        candidates = proposals[:, :, start:end]
        horizon = self.hierarchical_world_model.upper_world_model.horizon
        if end <= start or candidates.shape[2] % horizon:
            raise ValueError('Actual future candidate length must be positive and divisible by Upper horizon; no truncation or padding')
        # Do NOT call _prepare_info or normalize_action_for_world_model here.
        results = self.hierarchical_world_model.get_cost(raw_wm_info, candidates)
        if not isinstance(results, dict):
            raise ValueError('HWM get_cost must return a cost dictionary')
        for key in COST_KEYS:
            value = results.get(key)
            if not torch.is_tensor(value) or value.shape != (b, self.num_candidates) or not torch.isfinite(value).all():
                raise ValueError(f'{key} must be a finite Tensor with shape (B,K)')
        rollout_steps = {}
        for layer, steps in (('lower', candidates.shape[2]), ('upper', candidates.shape[2]//horizon)):
            key = f'{layer}_rollout'
            if key in results:
                predicted = results[key]['predicted_emb']
                if predicted.ndim != 4 or predicted.shape[:3] != (b, self.num_candidates, steps+1):
                    raise ValueError(f'{layer} rollout has incorrect terminal time/shape')
                rollout_steps[f'{layer}_steps'] = predicted.shape[2]-1
        best_idx = results['total_cost'].argmin(dim=1)
        rows = torch.arange(b, device=candidates.device)
        action = candidates[rows, best_idx.to(candidates.device), 0]
        self._selection_id += 1
        self.last_diagnostics = {key: results[key].detach().cpu().clone() for key in COST_KEYS}
        self.last_diagnostics.update(best_idx=best_idx.detach().cpu().clone(),
                                     selected_action=action.detach().cpu().clone(),
                                     selection_id=self._selection_id, evaluated_steps=candidates.shape[2])
        self.last_diagnostics.update(rollout_steps)
        return action.detach().cpu().numpy()


def hierarchical_step_record(policy, previous_selection_id, step_idx, commanded,
                             gripper_policy, gripper_actual, reinforcement_active):
    """Sidecar diagnostics; held reinforcement commands never reuse stale costs."""
    diagnostics = policy.last_diagnostics
    fresh = diagnostics is not None and diagnostics['selection_id'] != previous_selection_id
    record = dict(step_idx=int(step_idx), selection_id=diagnostics['selection_id'] if fresh else None,
                  inference_performed=fresh, reinforcement_active=bool(reinforcement_active),
                  executed_action=np.asarray(commanded).reshape(-1).tolist(),
                  gripper_policy=float(gripper_policy) if np.isfinite(gripper_policy) else None,
                  gripper_command=float(np.asarray(commanded).reshape(-1)[7]),
                  gripper_actual=float(gripper_actual) if np.isfinite(gripper_actual) else None,
                  best_idx=None, selected_costs=None, candidate_costs=None, policy_action=None)
    if fresh:
        best = diagnostics['best_idx']
        rows = torch.arange(best.numel())
        record.update(best_idx=best.tolist(), policy_action=diagnostics['selected_action'].tolist(),
                      selected_costs={key: diagnostics[key][rows,best].tolist() for key in COST_KEYS},
                      candidate_costs={key: diagnostics[key].tolist() for key in COST_KEYS})
    return record
