"""Dataset-only DP evaluation. No robot or control dependencies."""
from collections import deque
import json
from pathlib import Path

import h5py
import numpy as np


def load_episode(path, episode_id):
    """Select by stored ID, never by ordinal episode number.

    HDF5Dataset.load_episode assumes ep_offset/ep_len and ordinal IDs. Read
    selected rows directly instead, retaining HWC layout for DP._prepare_info.
    """
    with h5py.File(Path(path).expanduser(), 'r') as f:
        required = ('pixels', 'wrist_pixels', 'action_cartesian', 'episode_idx')
        for key in required:
            if key not in f:
                raise ValueError(f'Missing dataset column: {key}')
        ids = np.asarray(f['episode_idx']).reshape(-1)
        rows = np.flatnonzero(ids == episode_id)
        if not len(rows):
            raise ValueError(f'Episode ID {episode_id} not found')
        if any(len(f[k]) != len(ids) for k in required):
            raise ValueError('Dataset column lengths disagree')
        if 'step_idx' in f:
            if len(f['step_idx']) != len(ids):
                raise ValueError('step_idx length disagrees')
            steps = np.asarray(f['step_idx'][rows]).reshape(-1)
            if (not np.isfinite(steps).all() or not np.equal(steps, np.floor(steps)).all()
                    or np.any(steps < 0) or np.any(np.diff(steps) != 1)):
                raise ValueError('step_idx must be consecutive increasing integers in storage order')
            order_source = 'step_idx (validated consecutive storage order)'
        else:
            # Without timestamps, require the writer\'s episode layout metadata.
            if not {'ep_offset', 'ep_len'} <= set(f):
                raise ValueError('No step_idx: ep_offset/ep_len required to verify storage order')
            offsets, lengths = np.asarray(f['ep_offset']), np.asarray(f['ep_len'])
            matches = [int(n) for o, n in zip(offsets, lengths)
                       if o == rows[0] and n == len(rows)]
            if len(matches) != 1 or np.any(np.diff(rows) != 1):
                raise ValueError('Episode storage order does not match ep_offset/ep_len')
            steps = np.arange(len(rows))
            order_source = 'ep_offset/ep_len contiguous writer order; steps synthesized'
        data = {k: f[k][rows] for k in required if k != 'episode_idx'}
    for key in ('pixels', 'wrist_pixels'):
        images = data[key]
        if images.ndim != 4 or images.shape[-1] != 3 or images.dtype != np.uint8:
            raise ValueError(f'{key}: expected uint8 (T,H,W,3), got {images.shape}/{images.dtype}')
    gt = data['action_cartesian']
    if gt.shape != (len(rows), 8) or not np.isfinite(gt).all():
        raise ValueError('action_cartesian must be finite (T,8)')
    return data, steps.astype(np.int64), order_source


def compute_metrics(predicted, ground_truth, threshold=0.5):
    p, g = np.asarray(predicted, dtype=np.float64), np.asarray(ground_truth, dtype=np.float64)
    if p.shape != g.shape or p.ndim != 2 or p.shape[1] != 8 or not len(p):
        raise ValueError('Expected matching nonempty (T,8) actions')
    if not np.isfinite(p).all() or not np.isfinite(g).all():
        raise ValueError('Actions contain NaN/Inf')
    delta = p[:, :3] - g[:, :3]
    distance = np.linalg.norm(delta, axis=1)
    pq, gq = p[:, 3:7], g[:, 3:7]
    pn, gn = np.linalg.norm(pq, axis=1), np.linalg.norm(gq, axis=1)
    if np.any(pn < 1e-12) or np.any(gn < 1e-12):
        raise ValueError('Cannot evaluate zero-norm quaternion')
    dot = np.sum((pq / pn[:, None]) * (gq / gn[:, None]), axis=1)
    angle = np.degrees(2 * np.arccos(np.clip(np.abs(dot), 0, 1)))
    grip = p[:, 7] - g[:, 7]
    return {
        'position': {'mae_xyz_m': np.abs(delta).mean(0).tolist(),
                     'rmse_xyz_m': np.sqrt((delta ** 2).mean(0)).tolist(),
                     'mean_error_3d_m': float(distance.mean()),
                     'rmse_3d_m': float(np.sqrt(np.mean(distance ** 2)))},
        'orientation': {'mean_angle_error_deg': float(angle.mean()),
                        'rmse_angle_error_deg': float(np.sqrt(np.mean(angle ** 2)))},
        'gripper': {'mae': float(np.abs(grip).mean()),
                    'rmse': float(np.sqrt(np.mean(grip ** 2))),
                    'state_accuracy': float(np.mean((p[:, 7] >= threshold) == (g[:, 7] >= threshold))),
                    'closed_threshold': threshold, 'closed_rule': 'value >= threshold'},
    }


def evaluate_episode(policy, dataset_path, episode_id, output_dir, metadata):
    if policy.action_dim != 8 or 'action_cartesian' not in policy.process:
        raise ValueError('Offline evaluation requires an 8D action_cartesian checkpoint')
    if policy.obs_horizon < 1 or policy.action_horizon < 1 or policy.pred_horizon < policy.obs_horizon:
        raise ValueError('Invalid checkpoint horizons')
    data, steps, order_source = load_episode(dataset_path, episode_id)
    histories = {k: deque(maxlen=policy.obs_horizon) for k in ('pixels', 'wrist_pixels')}
    predicted = []
    for t in range(len(steps)):
        for key, history in histories.items():
            if not history:
                history.extend([data[key][t]] * policy.obs_horizon)
            else:
                history.append(data[key][t])
        info = {k: np.stack(v)[None] for k, v in histories.items()}
        # get_action selects predict_action[:, obs_horizon-1], i.e. current
        # observation's action in train_policy's same-row observation/action windows.
        # It also inverse-transforms using checkpoint statistics: do not do it twice.
        action = np.asarray(policy.get_action(info))
        if action.shape != (1, 8) or not np.isfinite(action).all():
            raise ValueError(f'Invalid DP output at step {steps[t]}: expected finite (1,8)')
        predicted.append(action[0].copy())
    predicted = np.stack(predicted)
    gt = data['action_cartesian']
    metrics = compute_metrics(predicted, gt)
    metadata = dict(metadata, dataset_path=str(Path(dataset_path).expanduser().resolve()),
                    episode_idx=int(episode_id), num_steps=len(steps), temporal_order=order_source,
                    alignment='observation t -> action_cartesian t; get_action uses obs_horizon-1',
                    model={k: int(getattr(policy, k)) for k in
                           ('pred_horizon', 'obs_horizon', 'action_horizon', 'action_dim', 'num_inference_steps')})
    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    with h5py.File(out / 'offline_rollout.h5', 'w') as f:
        for k, v in dict(predicted_action=predicted, ground_truth_action=gt, step_idx=steps,
                         episode_idx=np.full(len(steps), episode_id, dtype=np.int64)).items():
            f.create_dataset(k, data=v)
        f.attrs['metadata'] = json.dumps(metadata, allow_nan=False)
    (out / 'metrics.json').write_text(json.dumps(dict(metrics, metadata=metadata), indent=2, allow_nan=False) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for name, dims, labels, ylabel in (
        ('position', range(3), ('x', 'y', 'z'), 'Position [m]'),
        ('orientation', range(3, 7), ('qx', 'qy', 'qz', 'qw'), 'Quaternion component'),
        ('gripper', (7,), ('gripper',), 'Gripper (0=open, 1=closed)'),
    ):
        fig, axes = plt.subplots(len(labels), 1, sharex=True, squeeze=False, figsize=(10, 2.6 * len(labels)))
        for ax, dim, label in zip(axes[:, 0], dims, labels):
            ax.plot(steps, predicted[:, dim], color='tab:blue', label='Predicted')
            ax.plot(steps, gt[:, dim], color='tab:orange', label='Ground Truth')
            ax.set_ylabel(ylabel)
            ax.set_title(label)
            ax.grid(alpha=0.3)
            ax.legend()
        axes[-1, 0].set_xlabel('Step')
        fig.tight_layout()
        fig.savefig(out / f'predicted_vs_gt_{name}.png')
        plt.close(fig)
    return metrics


def run_offline_dp(cfg, loader, image_transform):
    """Load only the configured checkpoint; bypass all robot initialization."""
    robot = cfg.eval.real_robot
    policy = loader(checkpoint_path=cfg.gpc.diffusion_checkpoint,
                    image_transform=image_transform, device='cuda')
    metadata = dict(policy.offline_checkpoint_metadata,
                    image_size=cfg.eval.img_size,
                    preprocessing='existing dp_img_transform (ImageNet normalization)')
    return evaluate_episode(policy, robot.dry_run_dataset_path, int(robot.dry_run_episode_idx),
                            Path(robot.output_dir) / 'offline_dp', metadata)
