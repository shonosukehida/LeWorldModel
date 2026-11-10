"""Dataset-only GPC diagnostics using the unmodified GPC inference method."""
from copy import copy
from functools import wraps
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from offline_dp_eval import evaluate_multi_sample, load_episode


def isolated_rng(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        state = np.random.get_state()
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        try:
            with torch.random.fork_rng(devices=devices):
                return function(*args, **kwargs)
        finally:
            np.random.set_state(state)
    return wrapped


def preserve_gpc_offline_rng(function):
    @wraps(function)
    def wrapped(cfg):
        if cfg.get('policy_type') != 'gpc':
            return function(cfg)
        robot = getattr(cfg.eval, 'real_robot', None)
        if not (robot is not None and getattr(robot, 'dry_run', False)
                and getattr(robot, 'dry_run_mode', 'episode') == 'multi_sample'):
            return function(cfg)
        if cfg.get('policy', 'random') == 'random':
            raise ValueError('Offline GPC requires a trained world model')
        return isolated_rng(function)(cfg)
    return wrapped


def select_candidates(policy, candidates, wm_observation):
    """Replay physical proposals through the existing GPC API, without sampling.

    Copies only the policy containers; models and processors remain shared and
    untouched. Capturing reward_fn preserves its exact computation and selection.
    """
    replay = copy(policy)
    replay.diffusion_policy = copy(policy.diffusion_policy)
    replay.num_candidates = len(candidates)
    proposals = torch.as_tensor(candidates, device=policy.device)[None]
    replay.diffusion_policy.sample_action_sequences = lambda *a, **kw: proposals.clone()
    captured = {}

    def reward(output, info):
        rewards = policy.reward_fn(output, info)
        captured['rewards'] = rewards
        return rewards

    replay.reward_fn = reward
    with torch.no_grad():
        chunk = replay.get_action_chunk(wm_observation)
    rewards = captured['rewards'].detach().clone()
    if rewards.shape != (1, len(candidates)) or not torch.isfinite(rewards).all():
        raise ValueError('Expected finite candidate rewards (1,K)')
    index = int(torch.argmax(rewards, dim=1).item())
    start = policy.diffusion_policy.obs_horizon - 1
    end = min(start + policy.action_horizon, candidates.shape[1])
    if not np.array_equal(chunk[0], candidates[index, start:end]):
        raise ValueError('GPC returned actions differing from saved candidate')
    return index, rewards.cpu().numpy(), (start, end)


def plot_gpc(predicted, gt, index, interval, out, episode, timestep):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x = np.arange(predicted.shape[1])
    for name, dims, labels in (
        ('position', range(3), ('x', 'y', 'z')),
        ('orientation', range(3, 7), ('qx', 'qy', 'qz', 'qw')),
        ('gripper', (7,), ('gripper',)),
    ):
        fig, axes = plt.subplots(len(labels), 1, sharex=True, squeeze=False,
                                 figsize=(10, 2.6 * len(labels)))
        for ax, dim, label in zip(axes[:, 0], dims, labels):
            for k, candidate in enumerate(predicted):
                ax.plot(x, candidate[:, dim], color='gray', alpha=.3, linewidth=1,
                        label='DP candidates' if k == 0 else None)
            ax.axvspan(interval[0] - .5, interval[1] - .5, color='gold', alpha=.15,
                       label='WM evaluated interval [start, end)')
            ax.plot(x, gt[:, dim], color='blue', linewidth=2.5, label='Ground Truth')
            ax.plot(x, predicted[index, :, dim], color='red', linewidth=3, zorder=5,
                    label=f'GPC selected candidate {index}')
            ax.axvline(interval[0], color='black', linestyle='--', label='Selected action index')
            ax.set_title(label)
            ax.set_ylabel('Position [m]' if dim < 3 else
                          'Quaternion component' if dim < 7 else 'Gripper (0=open, 1=closed)')
            ax.grid(alpha=.3)
            ax.legend()
        axes[-1, 0].set_xlabel('Prediction sequence index')
        fig.suptitle(f'Episode {episode}, step {timestep}, candidates {len(predicted)}')
        fig.tight_layout()
        fig.savefig(Path(out) / f'multi_sample_{name}.png')
        plt.close(fig)


@isolated_rng
def evaluate_gpc_multi_sample(policy, dataset_path, episode, timestep, count, seed,
                              output_dir, metadata, observation_builder):
    """Keep DP artifact generation unchanged; add GPC-specific artifacts separately."""
    data, steps, _ = load_episode(dataset_path, episode)
    rows = np.flatnonzero(steps == timestep)
    if len(rows) != 1:
        raise ValueError('dry_run_timestep not found uniquely')
    # Measured proprio must come from the dataset, never action_cartesian (GT).
    with h5py.File(Path(dataset_path).expanduser()) as f:
        if 'proprio' not in f or len(f['proprio']) != len(f['episode_idx']):
            raise ValueError('GPC requires measured dataset proprio, aligned with episode_idx')
        episode_rows = np.flatnonzero(np.asarray(f['episode_idx']).reshape(-1) == episode)
        proprio = np.asarray(f['proprio'][int(episode_rows[rows[0]])], dtype=np.float32)
    if proprio.shape != (8,) or not np.isfinite(proprio).all():
        raise ValueError('Expected finite measured proprio (8,)')
    wm_info = observation_builder(data, int(rows[0]), proprio)
    # Retain only the current WM frame while the shared DP evaluator loads the
    # episode. Array views would otherwise keep both full image copies alive.
    wm_info = {key: value.copy() if isinstance(value, np.ndarray) else value
               for key, value in wm_info.items()}
    del data
    captured = {}
    dp = policy.diffusion_policy
    class RecordingDP:
        def __getattr__(self, name):
            return getattr(dp, name)

        def sample_action_sequences(self, *args, **kwargs):
            sampled = dp.sample_action_sequences(*args, **kwargs)
            candidates = sampled[0].detach().cpu().numpy().copy()
            index, rewards, interval = select_candidates(policy, candidates, wm_info)
            captured.update(index=index, rewards=rewards, interval=interval)
            return sampled

    result = evaluate_multi_sample(RecordingDP(), dataset_path, episode, timestep,
                                   count, seed, output_dir, metadata)
    out = Path(result['metadata']['output_dir'])
    index, rewards, interval = captured['index'], captured['rewards'], captured['interval']
    result.update(selected_candidate_index=index, candidate_rewards=rewards.tolist())
    result['metadata'].update(wm_evaluation_interval=list(interval),
                              wm_interval_convention='prediction indices [start, end); full candidate plotted')
    with h5py.File(out / 'multi_sample_rollout.h5', 'a') as f:
        candidates = f['predicted_actions'][:]
        gt = f['ground_truth_actions'][:]
        f['selected_candidate_index'] = index
        f['candidate_rewards'] = rewards
        f['selected_candidate_actions'] = candidates[index]
        f['wm_evaluation_interval'] = interval
        f.attrs['metadata'] = json.dumps(result['metadata'], allow_nan=False)
    (out / 'multi_sample_metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    plot_gpc(candidates, gt, index, interval, out, episode, timestep)
    print(f'Multi-sample GPC evaluation saved to: {out}')
    return result


def run_offline_gpc_multi_sample(cfg, policy, goal_loader, observation_builder):
    robot = cfg.eval.real_robot
    if not robot.goal_image_path or not robot.goal_wrist_image_path:
        raise ValueError('Offline GPC requires saved goal images; capture is unavailable')
    goal, wrist, goal_proprio = goal_loader(None, robot)

    def observation(data, row, proprio):
        return observation_builder(data['pixels'][row], data['wrist_pixels'][row],
                                   goal, wrist, proprio[:7], proprio[7], goal_proprio,
                                   robot.dry_run_timestep, policy.process)

    from omegaconf import OmegaConf
    import stable_worldmodel as swm
    # AutoCostModel resolves directories to their newest *_object.ckpt by ctime.
    # Record the concrete file as well as the configured run name.
    run_path = Path(str(cfg.policy))
    if not run_path.exists():
        run_path = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints'), str(cfg.policy))
    wm_path = (max(run_path.glob('*_object.ckpt'), key=lambda p: p.stat().st_ctime)
               if run_path.is_dir() else Path(f'{run_path}_object.ckpt'))
    wm_stat = wm_path.stat()
    metadata = dict(policy.diffusion_policy.offline_checkpoint_metadata,
                    policy_type='gpc', world_model=str(cfg.policy),
                    world_model_checkpoint=dict(path=str(wm_path.resolve()),
                                                size_bytes=wm_stat.st_size,
                                                mtime_ns=wm_stat.st_mtime_ns),
                    # Unused legacy interpolations need not resolve for offline evaluation.
                    config=OmegaConf.to_container(cfg, resolve=False),
                    normalization_stats_path=str(Path(cfg.eval.normalization_stats_path).expanduser().resolve()),
                    goal_image_path=str(Path(robot.goal_image_path).expanduser().resolve()),
                    goal_wrist_image_path=str(Path(robot.goal_wrist_image_path).expanduser().resolve()),
                    image_size=cfg.eval.img_size,
                    preprocessing='DP training image transform; WM real-robot transform and process')
    return evaluate_gpc_multi_sample(
        policy, robot.dry_run_dataset_path, int(robot.dry_run_episode_idx),
        robot.dry_run_timestep, getattr(robot, 'dry_run_num_candidates', 20),
        getattr(robot, 'dry_run_seed', 42),
        Path(robot.output_dir) / 'offline_gpc' / 'multi_sample', metadata, observation)
