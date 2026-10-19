"""Explicit checkpoint factory and hardware-free offline H-JEPA GPC verification."""
from pathlib import Path

import h5py
import numpy as np
import torch

from diffusion_checkpoint import load_diffusion_policy
from hierarchical_gpc_policy import HierarchicalGPCPolicy
from hierarchical_world_model import HierarchicalWorldModel
from utils import get_eval_img_preprocessor


def require_file(value, name):
    if value is None:
        raise ValueError(f'{name} must be explicitly specified')
    path = Path(value).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f'{name}: {path}')
    return path


def load_hierarchical_gpc(config):
    """No robot imports, guessed checkpoint paths, or statistics recomputation."""
    dp_path = require_file(config.diffusion_checkpoint, 'hjepa.diffusion_checkpoint')
    wm = dict(config.world_model)
    require_file(wm.get('lower_checkpoint'), 'hjepa.world_model.lower_checkpoint')
    require_file(wm.get('upper_checkpoint'), 'hjepa.world_model.upper_checkpoint')
    require_file(wm.get('lower_stats_path'), 'hjepa.world_model.lower_stats_path')
    # The existing DP loader accepts an image transform. Select it using the
    # saved training config, not the evaluation image-size setting.
    checkpoint = torch.load(dp_path, map_location='cpu', weights_only=False)
    for key in ('config', 'action_key', 'action_dim', 'action_mean', 'action_std',
                'obs_horizon', 'pred_horizon', 'action_horizon'):
        if key not in checkpoint:
            raise ValueError(f'DP checkpoint missing {key}')
    if checkpoint['action_key'] != 'action_cartesian' or checkpoint['action_dim'] != 8:
        raise ValueError('H-JEPA GPC requires a Cartesian 8D DP checkpoint')
    for key in ('action_mean','action_std'):
        value = torch.as_tensor(checkpoint[key])
        if value.shape not in ((8,), (1,8)) or not torch.isfinite(value).all():
            raise ValueError(f'DP {key} must contain eight finite values')
        if key == 'action_std' and (value <= 0).any():
            raise ValueError('DP action_std must be strictly positive')
    if 'img_size' not in checkpoint['config']:
        raise ValueError('DP checkpoint config missing training img_size')
    image_size = checkpoint['config']['img_size']
    if isinstance(image_size, bool) or not isinstance(image_size, int) or image_size <= 0:
        raise ValueError('DP training img_size must be a positive integer')
    del checkpoint
    dp = load_diffusion_policy(dp_path, get_eval_img_preprocessor(image_size), device=config.device)
    hwm = HierarchicalWorldModel.from_checkpoints(**wm, device=config.device)
    if torch.device(dp.device) != hwm._device():
        raise ValueError('DP and both world models must use the same device')
    return HierarchicalGPCPolicy(dp, hwm, num_candidates=config.num_candidates,
                                  action_horizon=config.action_horizon, raw_image_layout=config.raw_image_layout)


def offline_observations(path, current_index, goal_index, obs_horizon):
    """Read an actual contiguous DP history ending at current, never pad it."""
    path = require_file(path, 'dataset_path')
    for name, index in (('current_index',current_index),('goal_index',goal_index)):
        if isinstance(index, bool) or not isinstance(index, int) or index < 0:
            raise ValueError(f'{name} must be an explicitly selected nonnegative row index')
    with h5py.File(path,'r') as file:
        for key in ('pixels','wrist_pixels','proprio','episode_idx'):
            if key not in file:
                raise ValueError(f'Missing offline dataset column: {key}')
        n = len(file['episode_idx'])
        rows = np.arange(current_index-obs_horizon+1, current_index+1)
        if rows[0] < 0 or max(current_index,goal_index) >= n:
            raise ValueError('Requested observation/history indices are outside the dataset')
        ep = file['episode_idx'][rows].reshape(-1)
        if not np.issubdtype(ep.dtype,np.integer) or (ep<0).any() or not (ep == ep[-1]).all():
            raise ValueError('DP history crosses an episode boundary or has invalid episode_idx')
        if 'step_idx' in file:
            steps = file['step_idx'][rows].reshape(-1)
            if not np.issubdtype(steps.dtype,np.integer) or (steps<0).any() or not (steps[1:]-steps[:-1] == 1).all():
                raise ValueError('DP history crosses missing/invalid step_idx')
        info, dp_info = {}, {}
        # HDF5 layout is explicit via the real Flip Mug conversion attributes
        # or shape; require NHWC here, matching the real camera interface.
        for key, goal_key in (('pixels','goal'),('wrist_pixels','goal_wrist_pixels')):
            shape = file[key].shape
            if len(shape)!=4 or shape[-1]!=3 or shape[0]!=n:
                raise ValueError(f'Offline {key} must be raw NHWC RGB; convert the dataset explicitly')
            dp_info[key] = file[key][rows][None]
            info[key] = file[key][current_index][None,None]
            info[goal_key] = file[key][goal_index][None,None]
        if file['proprio'].shape != (n,8):
            raise ValueError('Offline proprio must have shape (N,8)')
        info['proprio'] = file['proprio'][current_index][None,None]
        info['goal_proprio'] = file['proprio'][goal_index][None,None]
    return info, dp_info
