"""Dataset-only DP evaluation. No robot or control dependencies."""
from collections import deque
import json
from pathlib import Path

import h5py
import numpy as np


def dataset_images_to_hwc(images, key):
    """Validate raw pixels, then adapt only layout to DP._prepare_info's HWC API.

    Preserve float32 values (including fractional pixels). Scaling, ImageNet
    normalization and resize belong exclusively to the offline image transform,
    which uses utils.get_img_preprocessor just like train_policy.py.
    """
    images = np.asarray(images)
    if images.ndim != 4 or images.dtype not in (np.dtype('float32'), np.dtype('uint8')):
        raise ValueError(f'{key}: expected float32/uint8 (T,3,H,W) or (T,H,W,3), got {images.shape}/{images.dtype}')
    chw, hwc = images.shape[1] == 3, images.shape[-1] == 3
    if chw == hwc or any(size <= 0 for size in images.shape):
        raise ValueError(f'{key}: invalid or ambiguous RGB image shape {images.shape}')
    if not np.isfinite(images).all() or images.min() < 0 or images.max() > 255:
        raise ValueError(f'{key}: expected finite raw image values in [0,255]')
    return np.moveaxis(images, 1, -1) if chw else images


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
        data[key] = dataset_images_to_hwc(data[key], key)
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


def evaluate_episode(policy, dataset_path, episode_id, output_dir, metadata, dp_execute_steps=1):
    if policy.action_dim != 8 or 'action_cartesian' not in policy.process:
        raise ValueError('Offline evaluation requires an 8D action_cartesian checkpoint')
    if policy.obs_horizon < 1 or policy.action_horizon < 1 or policy.pred_horizon < policy.obs_horizon:
        raise ValueError('Invalid checkpoint horizons')
    data, steps, order_source = load_episode(dataset_path, episode_id)
    if (isinstance(dp_execute_steps, bool) or not isinstance(dp_execute_steps, int)
            or not 1 <= dp_execute_steps <= policy.action_horizon):
        raise ValueError('Require integer 1 <= dp_execute_steps <= policy.action_horizon')
    if dp_execute_steps == 1:
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
    else:
        predicted = []
        chunk_rows = range(0, len(steps), dp_execute_steps)
        for row in chunk_rows:
            info = build_observation_history(data, row, policy.obs_horizon)
            # get_action_chunk already returns physical actions.
            chunk = np.asarray(policy.get_action_chunk(info))
            if (chunk.ndim != 3 or chunk.shape[0] != 1
                    or chunk.shape[1] < dp_execute_steps or chunk.shape[2] != 8
                    or not np.issubdtype(chunk.dtype, np.number)
                    or np.iscomplexobj(chunk) or not np.isfinite(chunk).all()):
                raise ValueError(f'Invalid DP chunk at step {steps[row]}: '
                                 'expected finite numeric (1,Ta,8), Ta >= dp_execute_steps')
            n = min(dp_execute_steps, len(steps) - row)
            predicted.extend(chunk[0, :n].copy())
    predicted = np.stack(predicted)
    gt = data['action_cartesian']
    metrics = compute_metrics(predicted, gt)
    metadata = dict(metadata, dataset_path=str(Path(dataset_path).expanduser().resolve()),
                    episode_idx=int(episode_id), num_steps=len(steps), temporal_order=order_source,
                    alignment='observation t -> action_cartesian t; get_action uses obs_horizon-1',
                    model={k: int(getattr(policy, k)) for k in
                           ('pred_horizon', 'obs_horizon', 'action_horizon', 'action_dim', 'num_inference_steps')})
    if dp_execute_steps > 1:
        metadata.update(dp_execute_steps=dp_execute_steps,
                        num_inference_calls=len(chunk_rows),
                        chunk_start_steps=steps[list(chunk_rows)].tolist(),
                        alignment='observation t -> action_cartesian t; get_action_chunk uses obs_horizon-1')
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
    # for name, dims, labels, ylabel in (
    #     ('position', range(3), ('x', 'y', 'z'), 'Position [m]'),
    #     ('orientation', range(3, 7), ('qx', 'qy', 'qz', 'qw'), 'Quaternion component'),
    #     ('gripper', (7,), ('gripper',), 'Gripper (0=open, 1=closed)'),
    # ):
    #     fig, axes = plt.subplots(len(labels), 1, sharex=True, squeeze=False, figsize=(10, 2.6 * len(labels)))
    #     for ax, dim, label in zip(axes[:, 0], dims, labels):
    #         ax.plot(steps, predicted[:, dim], color='tab:blue', label='Predicted')
    #         ax.plot(steps, gt[:, dim], color='tab:orange', label='Ground Truth')
    #         ax.set_ylabel(ylabel)
    #         ax.set_title(label)
    #         ax.grid(alpha=0.3)
    #         ax.legend()
    #     axes[-1, 0].set_xlabel('Step')
    #     fig.tight_layout()
    #     fig.savefig(out / f'predicted_vs_gt_{name}.png')
    #     plt.close(fig)


    # Checkpointと同じ正規化方式・統計を使用
    processor = policy.process["action_cartesian"]
    predicted_norm = processor.transform(predicted.copy())
    gt_norm = processor.transform(gt.copy())

    plot_specs = (
        ("position", range(3), ("x", "y", "z"), "Position [m]"),
        ("orientation", range(3, 7), ("qx", "qy", "qz", "qw"), "Quaternion component"),
        ("gripper", (7,), ("gripper",), "Gripper (0=open, 1=closed)"),
    )

    for normalized, pred_values, gt_values in (
        (False, predicted, gt),
        (True, predicted_norm, gt_norm),
    ):
        for name, dims, labels, ylabel in plot_specs:
            fig, axes = plt.subplots(
                len(labels), 1, sharex=True, squeeze=False,
                figsize=(10, 2.6 * len(labels))
            )

            for ax, dim, label in zip(axes[:, 0], dims, labels):
                ax.plot(
                    steps, pred_values[:, dim],
                    color="tab:blue", label="Predicted"
                )
                ax.plot(
                    steps, gt_values[:, dim],
                    color="tab:orange", label="Ground Truth"
                )

                if normalized:
                    ax.axhline(
                        -1.0, color="red", linestyle="--",
                        linewidth=1.0, alpha=0.25,
                        label="Normalized -1"
                    )
                    ax.axhline(
                        1.0, color="red", linestyle="--",
                        linewidth=1.0, alpha=0.25,
                        label="Normalized +1"
                    )

                ax.set_ylabel(
                    "Normalized action value" if normalized else ylabel
                )
                ax.set_title(label)
                ax.grid(alpha=0.3)
                ax.legend()

            axes[-1, 0].set_xlabel("Step")
            fig.tight_layout()

            suffix = "_normalized" if normalized else ""
            fig.savefig(out / f"predicted_vs_gt_{name}{suffix}.png")
            plt.close(fig)



    return metrics


def run_offline_dp(cfg, loader, image_transform):
    """Load only the configured checkpoint; bypass all robot initialization."""
    robot = cfg.eval.real_robot
    policy = loader(checkpoint_path=cfg.gpc.diffusion_checkpoint,
                    image_transform=image_transform, device='cuda')
    metadata = dict(policy.offline_checkpoint_metadata,
                    image_size=cfg.eval.img_size,
                    preprocessing='training get_img_preprocessor (range scaling, ImageNet normalization, resize)')

    output_dir = Path(robot.output_dir) / "offline_dp"

    result = evaluate_episode(
        policy,
        robot.dry_run_dataset_path,
        int(robot.dry_run_episode_idx),
        output_dir,
        metadata,
        dp_execute_steps=getattr(robot, "dp_execute_steps", 1),
    )

    print(f"[Dry Run] Results saved to: {output_dir.resolve()}")

    return result

def _integer(value, name, minimum=0):
    from numbers import Integral
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def build_observation_history(data, row, obs_horizon):
    """Match episode-mode left padding, retaining raw HWC image values."""
    obs_horizon = _integer(obs_horizon, 'obs_horizon', 1)
    rows = np.maximum(np.arange(row - obs_horizon + 1, row + 1), 0)
    return {key: data[key][rows][None] for key in ('pixels', 'wrist_pixels')}


def sample_candidates(policy, observation, num_candidates, seed):
    """Use DP's complete trajectories and its single inverse transform.

    fork_rng restores CPU and all CUDA RNG states even on failure. Independent
    batch elements in conditional_sample's torch.randn supply independent noise.
    No sampling or scheduler implementation is duplicated here.
    """
    import torch
    num_candidates = _integer(num_candidates, 'dry_run_num_candidates', 1)
    seed = _integer(seed, 'dry_run_seed')
    if seed > 2**63 - 1:
        raise ValueError('dry_run_seed must be <= 2**63 - 1')
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        with torch.backends.cudnn.flags(benchmark=False, deterministic=True), torch.no_grad():
            sampled = policy.sample_action_sequences(
                observation, num_samples=num_candidates, denormalize=True)
    if not torch.is_tensor(sampled):
        raise ValueError('sample_action_sequences must return a tensor (B,K,H,D)')
    expected = (1, num_candidates, policy.pred_horizon, policy.action_dim)
    if tuple(sampled.shape) != expected:
        raise ValueError(f'Invalid candidate shape: expected {expected}, got {tuple(sampled.shape)}')
    predicted = sampled[0].detach().cpu().numpy()
    if not np.isfinite(predicted).all():
        raise ValueError('Predicted candidates contain NaN/Inf')
    return predicted


def compute_multi_sample_metrics(predicted, ground_truth, valid_mask):
    """Error summaries use valid GT only; diversity covers the full horizon."""
    p, g = np.asarray(predicted, dtype=np.float64), np.asarray(ground_truth, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    if (p.ndim != 3 or not p.shape[0] or p.shape[2] != 8
            or g.shape != p.shape[1:] or valid.shape != (p.shape[1],) or not valid.any()):
        raise ValueError('Expected candidates (K,H,8), GT (H,8) and nonempty valid mask (H,)')
    if not np.isfinite(p).all() or not np.isfinite(g[valid]).all():
        raise ValueError('Valid actions contain NaN/Inf')
    per_candidate = [compute_metrics(candidate[valid], g[valid]) for candidate in p]
    position_rmse = [m['position']['rmse_3d_m'] for m in per_candidate]
    gripper_mae = [m['gripper']['mae'] for m in per_candidate]
    pq, gq = p[:, valid, 3:7], g[valid, 3:7]
    # compute_metrics above validates nonzero quaternion norms and q/-q equivalence.
    pq = pq / np.linalg.norm(pq, axis=-1, keepdims=True)
    gq = gq / np.linalg.norm(gq, axis=-1, keepdims=True)
    angles = np.degrees(2 * np.arccos(np.clip(np.abs(np.sum(pq * gq[None], axis=-1)), 0, 1)))
    angle_series = [[None] * p.shape[1] for _ in range(p.shape[0])]
    for k in range(p.shape[0]):
        for j, angle in zip(np.flatnonzero(valid), angles[k]):
            angle_series[k][int(j)] = float(angle)

    def summary(values):
        return {'mean': float(np.mean(values)), 'min': float(np.min(values)), 'max': float(np.max(values))}

    return {
        'num_valid_steps': int(valid.sum()),
        'position': {'candidate_rmse_3d_m': position_rmse,
                     'rmse_3d_m': summary(position_rmse),
                     'std_xyz_m_per_step': p[:, :, :3].std(axis=0).tolist()},
        'orientation': {'candidate_angle_error_deg_per_step': angle_series,
                        'candidate_mean_angle_error_deg': angles.mean(axis=1).tolist(),
                        'angle_error_deg': summary(angles)},
        'gripper': {'candidate_mae': gripper_mae, 'mae': summary(gripper_mae),
                    'std_per_step': p[:, :, 7].std(axis=0).tolist()},
        'diversity_definition': 'Population standard deviation (ddof=0), all predicted steps; no GT required',
        'minimum_error_note': 'Best candidate error is diagnostic only; it does not imply the policy can select that candidate.',
    }


def plot_multi_sample(predicted, ground_truth, selected_index, output_dir, episode_id, timestep):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    indices = np.arange(predicted.shape[1])
    for name, dims, labels, ylabel in (
        ('position', range(3), ('x', 'y', 'z'), 'Position [m]'),
        ('orientation', range(3, 7), ('qx', 'qy', 'qz', 'qw'), 'Quaternion component'),
        ('gripper', (7,), ('gripper',), 'Gripper (0=open, 1=closed)'),
    ):
        fig, axes = plt.subplots(len(labels), 1, sharex=True, squeeze=False,
                                 figsize=(10, 2.6 * len(labels)))
        for ax, dim, label in zip(axes[:, 0], dims, labels):
            for k, candidate in enumerate(predicted):
                ax.plot(indices, candidate[:, dim], color='gray', alpha=0.3, linewidth=1,
                        label='Predicted candidates' if k == 0 else None)
            ax.plot(indices, ground_truth[:, dim], color='red', linewidth=2.5, label='Ground Truth')
            ax.axvline(selected_index, color='black', linestyle='--', label='Selected action index')
            ax.set_title(label)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.3)
            ax.legend()
        axes[-1, 0].set_xlabel('Prediction sequence index')
        fig.suptitle(f'Episode {episode_id}, step {timestep}, candidates {len(predicted)}')
        fig.tight_layout()
        fig.savefig(Path(output_dir) / f'multi_sample_{name}.png')
        plt.close(fig)


def evaluate_multi_sample(policy, dataset_path, episode_id, timestep, num_candidates,
                          seed, output_dir, metadata):
    """Evaluate full trajectories at one stored step_idx, without robot control."""
    if policy.action_dim != 8 or 'action_cartesian' not in policy.process:
        raise ValueError('Offline evaluation requires an 8D action_cartesian checkpoint')
    horizon = _integer(policy.pred_horizon, 'pred_horizon', 1)
    obs_horizon = _integer(policy.obs_horizon, 'obs_horizon', 1)
    if horizon < obs_horizon:
        raise ValueError('pred_horizon must be >= obs_horizon')
    timestep = _integer(timestep, 'dry_run_timestep')
    num_candidates = _integer(num_candidates, 'dry_run_num_candidates', 1)
    seed = _integer(seed, 'dry_run_seed')
    data, steps, order_source = load_episode(dataset_path, episode_id)
    matches = np.flatnonzero(steps == timestep)
    if len(matches) != 1:
        raise ValueError(f'dry_run_timestep {timestep} not found in episode {episode_id} step_idx')
    observation = build_observation_history(data, int(matches[0]), obs_horizon)
    predicted = sample_candidates(policy, observation, num_candidates, seed)
    selected_index = obs_horizon - 1
    # Training uses observation/action windows from the same dataset row and
    # frameskip=1. predict_action selects To-1 from that complete trajectory.
    prediction_steps = timestep - selected_index + np.arange(horizon, dtype=np.int64)
    valid = (prediction_steps >= steps[0]) & (prediction_steps <= steps[-1])
    gt = np.full((horizon, policy.action_dim), np.nan, dtype=np.float64)
    gt[valid] = data['action_cartesian'][np.searchsorted(steps, prediction_steps[valid])]
    metrics = compute_multi_sample_metrics(predicted, gt, valid)
    metadata = dict(metadata, dataset_path=str(Path(dataset_path).expanduser().resolve()),
                    episode_idx=int(episode_id), timestep=timestep, num_candidates=num_candidates,
                    seed=seed, selected_action_index=selected_index, temporal_order=order_source,
                    alignment='gt_step = timestep - (obs_horizon - 1) + prediction_index',
                    model={k: int(getattr(policy, k)) for k in
                           ('pred_horizon', 'obs_horizon', 'action_horizon', 'action_dim', 'num_inference_steps')})
    # Atomic unique directory creation prevents overwrites even for concurrent runs.
    import tempfile
    parent = Path(output_dir).expanduser()
    parent.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix=f'episode_{episode_id}_step_{timestep}_', dir=parent))
    metadata.update(run_id=out.name, output_dir=str(out.resolve()))
    with h5py.File(out / 'multi_sample_rollout.h5', 'w') as f:
        for key, value in dict(predicted_actions=predicted, ground_truth_actions=gt,
                               ground_truth_valid_mask=valid, prediction_step_idx=prediction_steps,
                               selected_action_index=selected_index).items():
            f.create_dataset(key, data=value)
        f.attrs['metadata'] = json.dumps(metadata, allow_nan=False)
    (out / 'multi_sample_metrics.json').write_text(
        json.dumps(dict(metrics, metadata=metadata), indent=2, allow_nan=False) + '\n')
    plot_multi_sample(predicted, gt, selected_index, out, episode_id, timestep)
    print(f'Multi-sample DP evaluation saved to: {out}')
    return dict(metrics, metadata=metadata)


def run_offline_dp_multi_sample(cfg, loader, image_transform):
    robot = cfg.eval.real_robot
    policy = loader(checkpoint_path=cfg.gpc.diffusion_checkpoint,
                    image_transform=image_transform, device='cuda')
    metadata = dict(policy.offline_checkpoint_metadata, image_size=cfg.eval.img_size,
                    preprocessing='training get_img_preprocessor (range scaling, ImageNet normalization, resize)')
    return evaluate_multi_sample(
        policy, robot.dry_run_dataset_path, int(robot.dry_run_episode_idx),
        robot.dry_run_timestep, robot.dry_run_num_candidates, robot.dry_run_seed,
        Path(robot.output_dir) / 'offline_dp' / 'multi_sample', metadata)


def compute_chunk_rollout_metrics(predicted_chunks, valid_mask, representative_index=0):
    """Measure raw command differences, excluding padding and separating boundaries."""
    p = np.asarray(predicted_chunks, dtype=np.float64)
    valid = np.asarray(valid_mask, dtype=bool)
    representative_index = _integer(representative_index, 'representative_index')
    if (p.ndim != 4 or min(p.shape[:3]) < 1 or p.shape[-1] != 8
            or valid.shape != (p.shape[0], p.shape[2])
            or representative_index >= p.shape[1]
            or not valid[:, 0].all() or np.any(np.diff(valid.astype(int), axis=1) > 0)):
        raise ValueError('Expected chunks (C,K,A,8) and nonempty prefix valid masks (C,A)')
    chunks = [chunk[:, mask] for chunk, mask in zip(p, valid)]
    for chunk in chunks:
        if not np.isfinite(chunk).all():
            raise ValueError('Valid predicted actions contain NaN/Inf')
        if np.any(np.linalg.norm(chunk[..., 3:7], axis=-1) < 1e-12):
            raise ValueError('Cannot evaluate zero-norm quaternion in predicted chunks')
    representatives = [chunk[representative_index] for chunk in chunks]

    def differences(a, b):
        position = np.linalg.norm(b[:, :3] - a[:, :3], axis=-1)
        qa, qb = a[:, 3:7], b[:, 3:7]
        qa = qa / np.linalg.norm(qa, axis=-1, keepdims=True)
        qb = qb / np.linalg.norm(qb, axis=-1, keepdims=True)
        angle = np.degrees(2 * np.arccos(np.clip(np.abs(np.sum(qa * qb, axis=-1)), 0, 1)))
        return position, angle

    def summary(values):
        return dict(count=len(values), mean=float(np.mean(values)) if len(values) else None,
                    max=float(np.max(values)) if len(values) else None,
                    p95=float(np.percentile(values, 95)) if len(values) else None)

    inside = differences(np.concatenate([c[:-1] for c in representatives]),
                         np.concatenate([c[1:] for c in representatives]))
    boundary = differences(np.asarray([c[-1] for c in representatives[:-1]]).reshape(-1, 8),
                           np.asarray([c[0] for c in representatives[1:]]).reshape(-1, 8))
    return dict(within_chunk=dict(position_m_per_step=summary(inside[0]),
                                 orientation_deg_per_step=summary(inside[1])),
                boundary=dict(position_m_per_step=summary(boundary[0]),
                              orientation_deg_per_step=summary(boundary[1])),
                note='Command differences are not measured robot velocity or acceleration and do not guarantee hardware safety.')


def plot_chunk_rollout(predicted_chunks, ground_truth, steps, chunk_steps, valid_mask,
                       representative_index, output_dir):
    """Each candidate in each chunk gets its own Line2D; never bridge chunks."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for name, dims, labels, ylabel in (
        ('position', range(3), ('x', 'y', 'z'), 'Position [m]'),
        ('orientation', range(3, 7), ('qx', 'qy', 'qz', 'qw'), 'Quaternion component'),
        ('gripper', (7,), ('gripper',), 'Gripper (0=open, 1=closed)'),
    ):
        fig, axes = plt.subplots(len(labels), 1, sharex=True, squeeze=False,
                                 figsize=(14, 2.6 * len(labels)))
        for ax, dim, label in zip(axes[:, 0], dims, labels):
            ax.plot(steps, ground_truth[:, dim], color='black', label='Ground Truth')
            for i, (chunk, mask) in enumerate(zip(predicted_chunks, valid_mask)):
                x = chunk_steps[i, mask]
                for k, candidate in enumerate(chunk):
                    if k != representative_index:
                        ax.plot(x, candidate[mask, dim], color='lightgray', alpha=0.4, linewidth=0.8,
                                label='Predicted candidates' if i == 0 and k == (1 if representative_index == 0 else 0) else None)
                ax.plot(x, chunk[representative_index, mask, dim], color='blue', linewidth=1.3,
                        label='Representative candidate' if i == 0 else None)
                ax.axvline(x[0], color='gray', linestyle='--', alpha=0.25, linewidth=0.8)
            ax.set_title(label)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.2)
            ax.legend()
        axes[-1, 0].set_xlabel('Episode step (step_idx)')
        fig.tight_layout()
        fig.savefig(Path(output_dir) / f'chunk_rollout_{name}.png')
        plt.close(fig)


def evaluate_chunk_rollout(policy, dataset_path, episode_id, num_candidates, seed,
                           output_dir, metadata):
    from tqdm.auto import tqdm
    if policy.action_dim != 8 or 'action_cartesian' not in policy.process:
        raise ValueError('Offline evaluation requires an 8D action_cartesian checkpoint')
    horizon = _integer(policy.pred_horizon, 'pred_horizon', 1)
    obs = _integer(policy.obs_horizon, 'obs_horizon', 1)
    action = _integer(policy.action_horizon, 'action_horizon', 1)
    if horizon < obs - 1 + action:
        raise ValueError('pred_horizon must be >= obs_horizon - 1 + action_horizon')
    num_candidates = _integer(num_candidates, 'dry_run_num_candidates', 1)
    seed = _integer(seed, 'dry_run_seed')
    data, steps, order_source = load_episode(dataset_path, episode_id)
    rows = np.arange(0, len(steps), action)
    if seed > 2**63 - 1 - (len(rows) - 1):
        raise ValueError('dry_run_seed + chunk_index must be <= 2**63 - 1')
    predicted = np.full((len(rows), num_candidates, action, 8), np.nan)
    valid = np.zeros((len(rows), action), dtype=bool)
    chunk_steps = np.full((len(rows), action), -1, dtype=np.int64)
    for i, row in enumerate(tqdm(rows, total=len(rows), desc='DP chunk rollout', unit='chunk')):
        observation = build_observation_history(data, int(row), obs)
        candidates = sample_candidates(policy, observation, num_candidates, seed + i)
        length = min(action, len(steps) - row)
        predicted[i, :, :length] = candidates[:, obs - 1:obs - 1 + length]
        valid[i, :length] = True
        chunk_steps[i, :length] = steps[row:row + length]
    representative_index = 0  # Fixed policy choice, independent of ground truth.
    metrics = compute_chunk_rollout_metrics(predicted, valid, representative_index)
    metadata = dict(metadata, dataset_path=str(Path(dataset_path).expanduser().resolve()),
                    episode_idx=int(episode_id), pred_horizon=horizon, obs_horizon=obs,
                    action_horizon=action, num_candidates=num_candidates, seed=seed,
                    representative_index=representative_index, temporal_order=order_source,
                    chunk_seed_rule='dry_run_seed + chunk_index (zero based)',
                    chunk_seeds=[seed + i for i in range(len(rows))],
                    chunk_step_idx=[chunk_steps[i, mask].tolist() for i, mask in enumerate(valid)],
                    alignment='prediction index obs_horizon - 1 + j -> stored step at chunk row + j',
                    action_values='physical actions; sample_candidates denormalize=True exactly once')
    import tempfile
    parent = Path(output_dir).expanduser()
    parent.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix=f'episode_{episode_id}_', dir=parent))
    metadata.update(run_id=out.name, output_dir=str(out.resolve()))
    with h5py.File(out / 'chunk_rollout.h5', 'w') as f:
        for key, value in dict(predicted_chunks=predicted, ground_truth_actions=data['action_cartesian'],
                               step_idx=steps, chunk_start_steps=steps[rows], chunk_valid_mask=valid,
                               chunk_step_idx=chunk_steps, representative_index=representative_index).items():
            f.create_dataset(key, data=value)
        f.attrs['metadata'] = json.dumps(metadata, allow_nan=False)
    result = dict(metrics, metadata=metadata)
    (out / 'chunk_rollout_metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    plot_chunk_rollout(predicted, data['action_cartesian'], steps, chunk_steps, valid,
                       representative_index, out)
    print(f'Chunk rollout DP evaluation saved to: {out}')
    return result


def run_offline_dp_chunk_rollout(cfg, loader, image_transform):
    robot = cfg.eval.real_robot
    policy = loader(checkpoint_path=cfg.gpc.diffusion_checkpoint,
                    image_transform=image_transform, device='cuda')
    metadata = dict(policy.offline_checkpoint_metadata, image_size=cfg.eval.img_size,
                    preprocessing='training get_img_preprocessor (range scaling, ImageNet normalization, resize)')
    return evaluate_chunk_rollout(
        policy, robot.dry_run_dataset_path, int(robot.dry_run_episode_idx),
        robot.dry_run_num_candidates, robot.dry_run_seed,
        Path(robot.output_dir) / 'offline_dp' / 'chunk_rollout', metadata)
