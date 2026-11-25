"""Dataset-only GPC diagnostics using the unmodified GPC inference method."""
from copy import copy
from functools import wraps
import json
from pathlib import Path

import h5py
import numpy as np
import torch

from offline_dp_eval import evaluate_multi_sample, load_episode, compute_metrics
from stable_worldmodel.smoothness import MeasuredHistory
from stable_worldmodel.reward import latent_reward_metadata, latent_goal_costs, latent_goal_modality_costs


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
                and getattr(robot, 'dry_run_mode', 'episode') in ('multi_sample', 'chunk_rollout', 'latent_dist')):
            return function(cfg)
        if cfg.get('policy', 'random') == 'random':
            raise ValueError('Offline GPC requires a trained world model')
        return isolated_rng(function)(cfg)
    return wrapped


def candidate_costs(reward_fn, output, rewards):
    """Optional diagnostics, independent of selection and its reduction order."""
    boundary = getattr(reward_fn, 'image_dim', None)
    if boundary is None:
        return {}
    overhead_dim = getattr(reward_fn, 'overhead_dim', None)
    costs = {}
    if overhead_dim is not None:
        overhead, wrist, prop = latent_goal_modality_costs(output, overhead_dim, boundary)
        image = overhead + wrist
        costs.update(overhead_cost=overhead.detach().cpu().numpy()[0],
                     wrist_cost=wrist.detach().cpu().numpy()[0])
    else:
        image, prop = latent_goal_costs(output, boundary)
    costs.update(image_cost=image.detach().cpu().numpy()[0],
                 prop_cost=prop.detach().cpu().numpy()[0],
                 weighted_total_cost=-rewards.detach().cpu().numpy()[0])
    return costs


def select_candidates(policy, candidates, wm_observation, diagnostics=None):
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
        if diagnostics is not None:
            diagnostics.update(candidate_costs(policy.reward_fn, output, rewards))
        return rewards

    replay.reward_fn = reward
    with torch.no_grad():
        chunk = replay.get_action_chunk(wm_observation)
    rewards = captured['rewards'].detach().clone()
    if rewards.shape != (1, len(candidates)) or not torch.isfinite(rewards).all():
        raise ValueError('Expected finite candidate rewards (1,K)')
    best_idx = int(torch.argmax(rewards, dim=1).item())
    index = int(replay.selected_candidate_index.item())
    if diagnostics is not None:
        diagnostics.update(best_idx=best_idx, selected_candidate_index=index)
        if policy.goal_mode == 'latent_smoothness':
            diagnostics.update(wm_observation['smoothness_history'].diagnostics(
                -rewards[0].cpu().numpy(), best_idx, index))
    start = policy.diffusion_policy.obs_horizon - 1
    end = min(start + policy.action_horizon, candidates.shape[1])
    if not np.array_equal(chunk[0], candidates[index, start:end]):
        raise ValueError('GPC returned actions differing from saved candidate')
    return index, rewards.cpu().numpy(), (start, end)


def plot_gpc(predicted, gt, index, interval, out, episode, timestep, smoothness=False):
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
            if smoothness:
                ax.plot(x, predicted[0, :, dim], color='blue', label='DP candidate 0')
            ax.plot(x, gt[:, dim], color='black' if smoothness else 'blue', linewidth=2.5, label='Ground Truth')
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
    metadata = dict(metadata, **latent_reward_metadata(policy),
                    goal_mode=policy.goal_mode,
                    execute_first_candidate=policy.execute_first_candidate)
    data, steps, _ = load_episode(dataset_path, episode)
    rows = np.flatnonzero(steps == timestep)
    if len(rows) != 1:
        raise ValueError('dry_run_timestep not found uniquely')
    # Measured proprio must come from the dataset, never action_cartesian (GT).
    with h5py.File(Path(dataset_path).expanduser()) as f:
        if 'proprio' not in f or len(f['proprio']) != len(f['episode_idx']):
            raise ValueError('GPC requires measured dataset proprio, aligned with episode_idx')
        episode_rows = np.flatnonzero(np.asarray(f['episode_idx']).reshape(-1) == episode)
        row = int(rows[0])
        proprio = np.asarray(f['proprio'][int(episode_rows[row])], dtype=np.float32)
        if policy.goal_mode == 'latent_smoothness':
            first = max(0, row + 1 - policy.smoothness_horizon)
            window = slice(first, row + 1)
            measured = np.asarray(f['proprio'][episode_rows[window]], dtype=np.float32)
            frames = {k: data[k][window].copy() for k in ('pixels', 'wrist_pixels')}
            history = MeasuredHistory.from_episode(policy.smoothness_horizon, frames, measured,
                                                    steps[window], row - first)
            del frames
    if proprio.shape != (8,) or not np.isfinite(proprio).all():
        raise ValueError('Expected finite measured proprio (8,)')
    wm_info = observation_builder(data, int(rows[0]), proprio)
    # Retain only the current WM frame while the shared DP evaluator loads the
    # episode. Array views would otherwise keep both full image copies alive.
    wm_info = {key: value.copy() if isinstance(value, np.ndarray) else value
               for key, value in wm_info.items()}
    if policy.goal_mode == 'latent_smoothness':
        wm_info['smoothness_history'] = history
    del data
    captured = {}
    dp = policy.diffusion_policy
    class RecordingDP:
        def __getattr__(self, name):
            return getattr(dp, name)

        def sample_action_sequences(self, *args, **kwargs):
            sampled = dp.sample_action_sequences(*args, **kwargs)
            candidates = sampled[0].detach().cpu().numpy().copy()
            index, rewards, interval = select_candidates(policy, candidates, wm_info, captured)
            captured.update(index=index, rewards=rewards, interval=interval)
            return sampled

    evaluate = evaluate_multi_sample
    options = {}
    if getattr(dp, 'type', None) == 'imle' and dp.use_proprio:
        from offline_imle_core import evaluate_multi_sample as evaluate
        from offline_imle_eval import load_proprio_episode
        options = dict(episode_loader=load_proprio_episode, history_keys=('pixels', 'wrist_pixels', 'proprio'))
    result = evaluate(RecordingDP(), dataset_path, episode, timestep, count, seed, output_dir, metadata, **options)
    out = Path(result['metadata']['output_dir'])
    index, rewards, interval = captured['index'], captured['rewards'], captured['interval']
    result.update(best_idx=captured['best_idx'], selected_candidate_index=index, candidate_rewards=rewards.tolist())
    result['metadata'].update(wm_evaluation_interval=list(interval),
                              wm_interval_convention='prediction indices [start, end); full candidate plotted')
    with h5py.File(out / 'multi_sample_rollout.h5', 'a') as f:
        candidates = f['predicted_actions'][:]
        gt = f['ground_truth_actions'][:]
        f['best_idx'] = captured['best_idx']
        f['selected_candidate_index'] = index
        f['candidate_rewards'] = rewards
        for key in ('image_cost', 'overhead_cost', 'wrist_cost', 'prop_cost', 'weighted_total_cost'):
            if key in captured:
                f[key] = captured[key]
        if policy.goal_mode == 'latent_smoothness':
            for key in ('smoothness_cost', 'observation_steps', 'history_padded'):
                result[key] = captured[key]
                f[key] = captured[key]
        f['selected_candidate_actions'] = candidates[index]
        f['wm_evaluation_interval'] = interval
        f.attrs['metadata'] = json.dumps(result['metadata'], allow_nan=False)
    (out / 'multi_sample_metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    plot_gpc(candidates, gt, index, interval, out, episode, timestep, policy.goal_mode == "latent_smoothness")
    print(f'Multi-sample GPC evaluation (goal_mode={policy.goal_mode}) saved to: {out}')
    return result


def ranking_observation_builder(policy, robot, goal_loader, observation_builder, *, multi_sample=False):
    if policy.goal_mode == 'latent_smoothness':
        def measured(data, row, proprio):
            return dict(pixels=data['pixels'][None, row:row+1], wrist_pixels=data['wrist_pixels'][None, row:row+1],
                        proprio=proprio[None, None])
        return measured
    if not robot.goal_image_path or not robot.goal_wrist_image_path:
        raise ValueError('Offline GPC requires saved goal images; capture is unavailable')
    goal, wrist, goal_proprio = goal_loader(None, robot)

    def observation(data, row, proprio):
        step = robot.dry_run_timestep if multi_sample else int(data['offline_step_idx'][row])
        return observation_builder(data['pixels'][row], data['wrist_pixels'][row],
                                   goal, wrist, proprio[:7], proprio[7], goal_proprio,
                                   step, policy.process)
    return observation


def run_offline_gpc_multi_sample(cfg, policy, goal_loader, observation_builder):
    robot = cfg.eval.real_robot
    observation = ranking_observation_builder(policy, robot, goal_loader, observation_builder, multi_sample=True)

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
                    goal_image_path=str(Path(robot.goal_image_path).expanduser().resolve()) if robot.goal_image_path else None,
                    goal_wrist_image_path=str(Path(robot.goal_wrist_image_path).expanduser().resolve()) if robot.goal_wrist_image_path else None,
                    image_size=cfg.eval.img_size,
                    preprocessing='DP training image transform; WM real-robot transform and process')
    return evaluate_gpc_multi_sample(
        policy, robot.dry_run_dataset_path, int(robot.dry_run_episode_idx),
        robot.dry_run_timestep, getattr(robot, 'dry_run_num_candidates', 20),
        getattr(robot, 'dry_run_seed', 42),
        Path(robot.output_dir) / 'offline_gpc' / 'multi_sample', metadata, observation)


def run_offline_gpc_chunk_rollout(cfg, policy, goal_loader, observation_builder):
    robot = cfg.eval.real_robot
    observation = ranking_observation_builder(policy, robot, goal_loader, observation_builder)

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
                    goal_image_path=str(Path(robot.goal_image_path).expanduser().resolve()) if robot.goal_image_path else None,
                    goal_wrist_image_path=str(Path(robot.goal_wrist_image_path).expanduser().resolve()) if robot.goal_wrist_image_path else None,
                    image_size=cfg.eval.img_size,
                    preprocessing='DP training image transform; WM real-robot transform and process')
    execute_steps = getattr(robot, 'dp_execute_steps', 1)
    metadata['normalization_statistics'] = {
        group: {key: {name: np.asarray(getattr(processor, name)).tolist()
                      for name in ('mean_', 'scale_', 'low', 'span')
                      if getattr(processor, name, None) is not None}
                for key, processor in processors.items()}
        for group, processors in (('dp', policy.diffusion_policy.process), ('wm', policy.process))
    }
    return evaluate_gpc_chunk_rollout(
        policy, robot.dry_run_dataset_path, int(robot.dry_run_episode_idx),
        getattr(robot, 'dry_run_num_candidates', 20),
        getattr(robot, 'dry_run_seed', 42),
        Path(robot.output_dir) / 'offline_gpc' / 'chunk_rollout', metadata, observation,
        dp_execute_steps=execute_steps)


def plot_gpc_chunk_rollout(predicted, gt, steps, chunk_steps, valid, selected, output_dir):
    """Draw each chunk separately, with the selected candidate above all others."""
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
            ax.plot(steps, gt[:, dim], color='black', label='Ground Truth')
            for i, (chunk, mask) in enumerate(zip(predicted, valid)):
                x = chunk_steps[i, mask]
                for k, candidate in enumerate(chunk):
                    ax.plot(x, candidate[mask, dim], color='lightgray', alpha=.4, linewidth=.8,
                            label='DP candidates' if i == k == 0 else None)
                ax.plot(x, chunk[0, mask, dim], color='blue', linewidth=1.3,
                        label='Representative candidate (0)' if i == 0 else None)
                ax.plot(x, chunk[selected[i], mask, dim], color='red', linewidth=3, zorder=10,
                        marker='.' if len(x) == 1 else None,
                        label='GPC selected candidate' if i == 0 else None)
                ax.axvline(x[0], color='gray', linestyle='--', alpha=.25, linewidth=.8)
            ax.set_title(label)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=.2)
            ax.legend()
        axes[-1, 0].set_xlabel('Episode step (step_idx)')
        fig.tight_layout()
        fig.savefig(Path(output_dir) / f'chunk_rollout_{name}.png')
        plt.close(fig)


@isolated_rng
def evaluate_gpc_chunk_rollout(policy, dataset_path, episode_id, num_candidates, seed,
                               output_dir, metadata, observation_builder, dp_execute_steps=1):
    from tqdm.auto import tqdm
    from offline_dp_eval import (_integer, build_observation_history, sample_candidates,
                                 compute_chunk_rollout_metrics)
    import tempfile
    dp = policy.diffusion_policy
    if dp.action_dim != 8 or 'action_cartesian' not in dp.process:
        raise ValueError('Offline evaluation requires an 8D action_cartesian checkpoint')
    horizon = _integer(dp.pred_horizon, 'pred_horizon', 1)
    obs = _integer(dp.obs_horizon, 'obs_horizon', 1)
    action = _integer(dp.action_horizon, 'action_horizon', 1)
    execute_steps = _integer(dp_execute_steps, 'dp_execute_steps', 1)
    if execute_steps > action:
        raise ValueError('dp_execute_steps must be <= action_horizon')
    if horizon < obs - 1 + action:
        raise ValueError('pred_horizon must be >= obs_horizon - 1 + action_horizon')
    count = _integer(num_candidates, 'dry_run_num_candidates', 1)
    seed = _integer(seed, 'dry_run_seed')
    data, steps, order = load_episode(dataset_path, episode_id)
    data['offline_step_idx'] = steps
    with h5py.File(Path(dataset_path).expanduser()) as f:
        if 'proprio' not in f or len(f['proprio']) != len(f['episode_idx']):
            raise ValueError('GPC requires measured dataset proprio, aligned with episode_idx')
        episode_rows = np.flatnonzero(np.asarray(f['episode_idx']).reshape(-1) == episode_id)
        proprio = np.asarray(f['proprio'][episode_rows], dtype=np.float32)
    if proprio.shape != (len(steps), 8) or not np.isfinite(proprio).all():
        raise ValueError('Expected finite measured proprio (T,8)')
    rows = np.arange(0, len(steps), execute_steps)
    if seed > 2**63 - 1 - (len(rows) - 1):
        raise ValueError('dry_run_seed + chunk_index must be <= 2**63 - 1')
    predicted = np.full((len(rows), count, execute_steps, 8), np.nan)
    valid = np.zeros((len(rows), execute_steps), dtype=bool)
    chunk_steps = np.full((len(rows), execute_steps), -1, dtype=np.int64)
    indices, rewards, intervals = [], [], []
    costs = []
    for i, row in enumerate(tqdm(rows, total=len(rows), desc='GPC chunk rollout', unit='chunk')):
        proposal_info = build_observation_history(data, int(row), obs)
        if getattr(dp, 'type', None) == 'imle' and dp.use_proprio:
            from offline_imle_core import build_observation_history as imle_history
            proposal_info.update(imle_history({'proprio': proprio}, int(row), obs, keys=('proprio',)))
        candidates = sample_candidates(dp, proposal_info, count, seed + i)
        info = observation_builder(data, int(row), proprio[row])
        if policy.goal_mode == 'latent_smoothness':
            info['smoothness_history'] = MeasuredHistory.from_episode(
                policy.smoothness_horizon, data, proprio, steps, row)
        diagnostics = {}
        index, reward, interval = select_candidates(policy, candidates, info, diagnostics)
        costs.append(diagnostics)
        length = min(execute_steps, len(steps) - row)
        if interval[1] - interval[0] < length:
            raise ValueError('WM evaluation interval must cover the visualized chunk')
        predicted[i, :, :length] = candidates[:, obs - 1:obs - 1 + length]
        valid[i, :length] = True
        chunk_steps[i, :length] = steps[row:row + length]
        indices.append(index)
        rewards.append(reward[0])
        intervals.append(interval)
    indices, rewards = np.asarray(indices), np.asarray(rewards)
    selected = predicted[np.arange(len(rows)), indices].copy()
    metadata = dict(metadata, **latent_reward_metadata(policy),
                    goal_mode=policy.goal_mode,
                    execute_first_candidate=policy.execute_first_candidate)
    metadata = dict(metadata, policy_type='gpc', dry_run_mode='chunk_rollout',
                    dataset_path=str(Path(dataset_path).expanduser().resolve()), episode_idx=int(episode_id),
                    num_candidates=count, seed=seed, pred_horizon=horizon, obs_horizon=obs,
                    action_horizon=action, dp_execute_steps=execute_steps, chunk_width=execute_steps, representative_index=0,
                    temporal_order=order, chunk_seed_rule='dry_run_seed + chunk_index (zero based)',
                    chunk_seeds=[seed + i for i in range(len(rows))],
                    wm_evaluation_intervals=[list(v) for v in intervals],
                    wm_interval_convention='prediction indices [start, end); evaluated even beyond episode tail',
                    visualization_interval=[obs - 1, obs - 1 + execute_steps],
                    execution_note='Offline stride and visualized width are dp_execute_steps; WM evaluates the full GPC action_horizon',
                    invalid_padding='NaN only outside chunk_valid_mask; never evaluated or plotted',
                    alignment='prediction index obs_horizon - 1 + j -> stored step at chunk row + j',
                    action_values='physical actions; sample_candidates denormalize=True exactly once')
    parent = Path(output_dir).expanduser()
    parent.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix=f'episode_{episode_id}_', dir=parent))
    metadata.update(run_id=out.name, output_dir=str(out.resolve()))
    result = dict(compute_chunk_rollout_metrics(predicted, valid, 0), metadata=metadata,
                  best_idx=rewards.argmax(axis=1).tolist(),
                  selected_candidate_indices=indices.tolist(), candidate_rewards=rewards.tolist())
    with h5py.File(out / 'chunk_rollout.h5', 'w') as f:
        for key, value in dict(predicted_chunks=predicted, ground_truth_actions=data['action_cartesian'],
                               step_idx=steps, chunk_start_steps=steps[rows], chunk_step_idx=chunk_steps,
                               chunk_valid_mask=valid, representative_index=0,
                               best_idx=rewards.argmax(axis=1),
                               selected_candidate_indices=indices, candidate_rewards=rewards,
                               selected_chunks=selected, wm_evaluation_intervals=intervals).items():
            f.create_dataset(key, data=value)
        if policy.goal_mode == 'latent_smoothness':
            for key in ('observation_steps', 'history_padded', 'smoothness_cost', 'selected_candidate_index'):
                result[key] = [value[key] for value in costs]
                f[key] = result[key]
            gt = data['action_cartesian']
            result['selected_vs_gt'] = compute_metrics(selected[valid], gt)
            result['candidate_0_vs_gt'] = compute_metrics(predicted[:, 0][valid], gt)
        for key in ('image_cost', 'overhead_cost', 'wrist_cost', 'prop_cost', 'weighted_total_cost'):
            if costs and key in costs[0]:
                f[key] = np.stack([value[key] for value in costs])
        f.attrs['metadata'] = json.dumps(metadata, allow_nan=False)
    (out / 'chunk_rollout_metrics.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    plot_gpc_chunk_rollout(predicted, data['action_cartesian'], steps, chunk_steps, valid, indices, out)
    print(f'Chunk rollout GPC evaluation (goal_mode={policy.goal_mode}) saved to: {out}')
    return result
