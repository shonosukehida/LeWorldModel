"""IMLE-only evaluation loops derived from the multiview DP baseline.

Retain action artifacts and plotting semantics without modifying DP evaluation.
Shared read-only metrics/plotters stay in offline_dp_eval; only these loops accept
IMLE's additional aligned observation column.
"""
from pathlib import Path
from collections import deque
import json
import h5py
import numpy as np
from offline_dp_eval import (
    load_episode, compute_metrics, _integer, sample_candidates,
    compute_multi_sample_metrics, plot_multi_sample,
    compute_chunk_rollout_metrics, plot_chunk_rollout,
)

def evaluate_episode(policy, dataset_path, episode_id, output_dir, metadata, dp_execute_steps=1, *,
                     episode_loader=load_episode, history_keys=('pixels', 'wrist_pixels')):
    if policy.action_dim != 8 or 'action_cartesian' not in policy.process:
        raise ValueError('Offline evaluation requires an 8D action_cartesian checkpoint')
    if policy.obs_horizon < 1 or policy.action_horizon < 1 or policy.pred_horizon < policy.obs_horizon:
        raise ValueError('Invalid checkpoint horizons')
    data, steps, order_source = episode_loader(dataset_path, episode_id)
    if (isinstance(dp_execute_steps, bool) or not isinstance(dp_execute_steps, int)
            or not 1 <= dp_execute_steps <= policy.action_horizon):
        raise ValueError('Require integer 1 <= dp_execute_steps <= policy.action_horizon')
    if dp_execute_steps == 1:
        histories = {k: deque(maxlen=policy.obs_horizon) for k in history_keys}
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
            info = build_observation_history(data, row, policy.obs_horizon, history_keys)
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


def build_observation_history(data, row, obs_horizon, keys=('pixels', 'wrist_pixels')):
    """Match episode-mode left padding, retaining raw HWC image values."""
    obs_horizon = _integer(obs_horizon, 'obs_horizon', 1)
    rows = np.maximum(np.arange(row - obs_horizon + 1, row + 1), 0)
    return {key: data[key][rows][None] for key in keys}


def evaluate_multi_sample(policy, dataset_path, episode_id, timestep, num_candidates,
                          seed, output_dir, metadata, *,
                          episode_loader=load_episode, history_keys=('pixels', 'wrist_pixels')):
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
    data, steps, order_source = episode_loader(dataset_path, episode_id)
    matches = np.flatnonzero(steps == timestep)
    if len(matches) != 1:
        raise ValueError(f'dry_run_timestep {timestep} not found in episode {episode_id} step_idx')
    observation = build_observation_history(data, int(matches[0]), obs_horizon, history_keys)
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


def evaluate_chunk_rollout(policy, dataset_path, episode_id, num_candidates, seed,
                           output_dir, metadata, *,
                           episode_loader=load_episode, history_keys=('pixels', 'wrist_pixels')):
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
    data, steps, order_source = episode_loader(dataset_path, episode_id)
    rows = np.arange(0, len(steps), action)
    if seed > 2**63 - 1 - (len(rows) - 1):
        raise ValueError('dry_run_seed + chunk_index must be <= 2**63 - 1')
    predicted = np.full((len(rows), num_candidates, action, 8), np.nan)
    valid = np.zeros((len(rows), action), dtype=bool)
    chunk_steps = np.full((len(rows), action), -1, dtype=np.int64)
    for i, row in enumerate(tqdm(rows, total=len(rows), desc='DP chunk rollout', unit='chunk')):
        observation = build_observation_history(data, int(row), obs, history_keys)
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
