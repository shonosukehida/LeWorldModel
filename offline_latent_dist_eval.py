"""Offline latent-distance diagnostics; no changes to model or control APIs."""
from copy import copy
import json
from pathlib import Path
import tempfile

import h5py
import numpy as np
import torch
from tqdm.auto import tqdm

from offline_dp_eval import (_integer, load_episode, build_observation_history,
                             sample_candidates)
from offline_gpc_eval import isolated_rng, select_candidates
from stable_worldmodel.reward import latent_goal_reward, latent_reward_metadata


def _latent(value, name):
    if (not torch.is_tensor(value) or value.ndim != 3 or value.shape[:2] != (1, 1)
            or value.shape[-1] < 1 or not torch.isfinite(value).all()):
        raise ValueError(f'{name} must be finite (1,1,D); no pooling or flattening is supported')
    return value


def _terminal(output, count, dimension):
    value = output['predicted_emb']
    if (value.ndim != 4 or value.shape[:2] != (1, count)
            or value.shape[2] < 1 or value.shape[-1] != dimension
            or not torch.isfinite(value).all()):
        raise ValueError('Expected finite predicted_emb (1,K,T,D) matching goal latent')
    return value[0, :, -1].detach().cpu()


def _l2(a, b):
    if a.shape != b.shape:
        raise ValueError('Latent dimensions must match exactly')
    distance = torch.linalg.vector_norm(a - b, dim=-1)
    if not torch.isfinite(distance).all():
        raise ValueError('Nonfinite latent L2 distance')
    return distance.numpy()


def plot_latent_dist(values, horizon, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    t = values['rollout_start_steps']
    specs = (
        ('observed_latent_distance', values['observed_step_idx'],
         [('observed_latent_distance', 'black', 'Observed latent distance')],
         'Episode step', 'Latent L2 distance to Goal'),
        ('rollout_latent_distance', t,
         [('current_observed_distance', 'black', 'Observed at start t'),
          ('gt_rollout_terminal_distance', 'blue', f'GT action rollout ({horizon}-step terminal)'),
          ('gpc_rollout_terminal_distance', 'red', f'GPC selected rollout ({horizon}-step terminal)')],
         'Rollout start step t', 'Predicted terminal latent L2 distance to Goal'),
        ('latent_distance_delta', t,
         [('gt_distance_delta', 'blue', f'GT rollout ({horizon}-step)'),
          ('gpc_distance_delta', 'red', f'GPC rollout ({horizon}-step)')],
         'Rollout start step t', 'Predicted terminal distance − Current distance'),
        ('wm_prediction_error', t,
         [('wm_prediction_error', 'purple', f'GT prediction vs observed at t+{horizon}')],
         'Rollout start step t', 'Latent L2 prediction error'),
    )
    for name, x, curves, xlabel, ylabel in specs:
        fig, ax = plt.subplots(figsize=(12, 4))
        for key, color, label in curves:
            ax.plot(x, values[key], color=color, label=label, marker='.' if len(x) == 1 else None)
        if name == 'latent_distance_delta':
            ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(Path(out) / f'{name}.png')
        plt.close(fig)


@isolated_rng
@torch.no_grad()
def evaluate_latent_dist(policy, dataset_path, episode_id, num_candidates, seed,
                         output_dir, metadata, observation_builder, dp_execute_steps=1):
    dp = policy.diffusion_policy
    horizon = _integer(policy.action_horizon, 'action_horizon', 1)
    obs = _integer(dp.obs_horizon, 'obs_horizon', 1)
    execute = _integer(dp_execute_steps, 'dp_execute_steps', 1)
    if execute > horizon:
        raise ValueError('dp_execute_steps must be <= action_horizon')
    count = _integer(num_candidates, 'dry_run_num_candidates', 1)
    seed = _integer(seed, 'dry_run_seed')
    if dp.action_dim != 8 or dp.pred_horizon < obs - 1 + horizon:
        raise ValueError('Expected 8D DP with pred_horizon >= obs_horizon - 1 + action_horizon')
    data, steps, order = load_episode(dataset_path, episode_id)
    data['offline_step_idx'] = steps
    with h5py.File(Path(dataset_path).expanduser()) as f:
        if 'proprio' not in f or len(f['proprio']) != len(f['episode_idx']):
            raise ValueError('GPC requires measured dataset proprio, aligned with episode_idx')
        rows = np.flatnonzero(np.asarray(f['episode_idx']).reshape(-1) == episode_id)
        proprio = np.asarray(f['proprio'][rows], dtype=np.float32)
    if proprio.shape != (len(steps), 8) or not np.isfinite(proprio).all():
        raise ValueError('Expected finite measured proprio (T,8)')
    starts = np.arange(0, len(steps), execute)
    valid_rows = starts[starts + horizon < len(steps)]
    if seed > 2**63 - 1 - max(len(starts) - 1, 0):
        raise ValueError('dry_run_seed + evaluation_index must be <= 2**63 - 1')
    # Reject training mode rather than modifying the caller's shared model state.
    if policy.world_model.training:
        raise ValueError('latent_dist requires world_model.eval()')
    def observation(row):
        return observation_builder(data, int(row), proprio[row])
    goal = _latent(policy.encode_goal(observation(0)), 'goal_emb').detach()
    goal_cpu = goal[0, -1].cpu()
    dimension = goal.shape[-1]
    cached = copy(policy)
    cached.encode_goal = lambda info: goal
    observed = []
    wm_device = next(policy.world_model.parameters()).device
    for row in tqdm(range(len(steps)), total=len(steps), desc='Encoding observed latents', unit='step'):
        info = {k: v for k, v in observation(row).items()
                if k in ('pixels', 'wrist_pixels', 'proprio', 'step_idx')}
        prepared = policy._prepare_info(info)
        prepared = {k: v.to(wm_device) if torch.is_tensor(v) else v for k, v in prepared.items()}
        emb = _latent(policy.world_model.encode(prepared)['emb'], 'observed_emb')
        if emb.shape != goal.shape:
            raise ValueError('Observed and goal latent shapes must match exactly')
        observed.append(emb[0, -1].detach().cpu())
    observed = torch.stack(observed)
    observed_dist = _l2(observed, goal_cpu.expand_as(observed))
    gt_latents, selected_latents, indices, rewards = [], [], [], []
    candidate_costs = []
    reward_config = latent_reward_metadata(policy)
    for i, row in enumerate(tqdm(valid_rows, total=len(valid_rows),
                                desc='Evaluating latent distances', unit='rollout')):
        info = observation(row)
        candidates = sample_candidates(dp, build_observation_history(data, int(row), obs), count, seed + i)
        captured = {}
        recording = copy(cached)
        def rollout(**kwargs):
            output = cached.rollout(**kwargs)
            captured['terminal'] = _terminal(output, count, dimension)
            return output
        recording.rollout = rollout
        diagnostics = {}
        index, reward, interval = select_candidates(recording, candidates, info, diagnostics)
        candidate_costs.append(diagnostics)
        if interval != (obs - 1, obs - 1 + horizon):
            raise ValueError('GPC evaluation interval differs from the complete action_horizon')
        terminal = captured['terminal']
        # Validate the configured reward separately from ordinary L2 diagnostics.
        expected = latent_goal_reward(
            {'predicted_emb': terminal[None, :, None], 'goal_emb': goal_cpu[None, None]}, {},
            lambda_prop=reward_config['lambda_prop'], image_dim=reward_config['image_latent_dimension'])
        if not torch.allclose(torch.as_tensor(reward[0]), expected[0], rtol=1e-5, atol=1e-5):
            raise ValueError('latent_dist candidate rewards differ from the configured latent_goal_reward')
        selected_latents.append(terminal[index])
        indices.append(index)
        rewards.append(reward[0])
        # GT is used only after independent candidate selection. No DP offset here.
        gt = torch.as_tensor(data['action_cartesian'][row:row + horizon], device=policy.device)[None, None]
        normalized = cached.normalize_action_for_world_model(gt)
        output = cached.rollout(info_dict=info, action_sequences=normalized)
        gt_latents.append(_terminal(output, 1, dimension)[0])
    # Empty rollout sets remain well shaped when an episode is shorter than H+1.
    gt_latents = torch.stack(gt_latents) if gt_latents else torch.empty((0, dimension), dtype=goal_cpu.dtype)
    selected_latents = torch.stack(selected_latents) if selected_latents else torch.empty_like(gt_latents)
    gt_dist = _l2(gt_latents, goal_cpu.expand_as(gt_latents))
    gpc_dist = _l2(selected_latents, goal_cpu.expand_as(selected_latents))
    values = dict(observed_step_idx=steps, observed_latent_distance=observed_dist,
                  rollout_start_steps=steps[valid_rows], rollout_future_steps=steps[valid_rows + horizon],
                  current_observed_distance=observed_dist[valid_rows],
                  future_observed_distance=observed_dist[valid_rows + horizon],
                  gt_rollout_terminal_distance=gt_dist, gpc_rollout_terminal_distance=gpc_dist,
                  gt_distance_delta=gt_dist - observed_dist[valid_rows],
                  gpc_distance_delta=gpc_dist - observed_dist[valid_rows],
                  wm_prediction_error=_l2(gt_latents, observed[valid_rows + horizon]),
                  selected_candidate_indices=np.asarray(indices, dtype=np.int64),
                  candidate_rewards=np.asarray(rewards).reshape(len(valid_rows), count),
                  goal_latent=goal_cpu.numpy(), observed_latents=observed.numpy(),
                  gt_terminal_latents=gt_latents.numpy(), gpc_terminal_latents=selected_latents.numpy())
    for key in ('image_cost', 'prop_cost', 'weighted_total_cost'):
        if candidate_costs and key in candidate_costs[0]:
            values[key] = np.stack([cost[key] for cost in candidate_costs])
    metadata = dict(metadata, **reward_config)
    metadata = dict(metadata, policy_type='gpc', dry_run_mode='latent_dist', episode_idx=int(episode_id),
                    dataset_path=str(Path(dataset_path).expanduser().resolve()),
                    action_horizon=horizon, dp_execute_steps=execute, dry_run_num_candidates=count,
                    dry_run_seed=seed, latent_dimension=dimension, distance='L2 (not squared)',
                    temporal_order=order, rollout_start_rule='stored episode row 0,S,2S,...; observed evaluated every row',
                    alignment='observation row t, physical action_cartesian[t:t+H], future observation row t+H; DP candidates [obs_horizon-1:obs_horizon-1+H]',
                    temporal_assumption='robot/utils/data_conversion.py calculate_action_cartesian stores target[t+1] as action_cartesian[t] (last target repeated); consume stored actions without an extra shift; external datasets must follow this convention',
                    skipped_steps=[dict(step=int(steps[r]), reason='Incomplete H-step GT actions or missing observation at t+H')
                                   for r in starts if r + horizon >= len(steps)],
                    chunk_seed_rule='dry_run_seed + evaluation_index (zero based)',
                    evaluation_seeds=[seed + i for i in range(len(valid_rows))],
                    obs_horizon=obs, pred_horizon=int(dp.pred_horizon), goal_encoded_once=True,
                    preprocessing='existing GPC _prepare_info / encode_goal; no pooling, flatten, concat or extra normalization')
    parent = Path(output_dir).expanduser()
    parent.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix=f'episode_{episode_id}_', dir=parent))
    metadata.update(output_dir=str(out.resolve()), run_id=out.name)
    # These diagnostics alone do not establish success, task phases, or causality.
    result = dict(metadata=metadata, num_observations=len(steps), num_rollouts=len(valid_rows),
                  observed_increase_count=int(np.count_nonzero(np.diff(observed_dist) > 0)))
    encoded = json.dumps(result, indent=2, allow_nan=False)
    with h5py.File(out / 'latent_distance.h5', 'w') as f:
        for key, value in values.items():
            f.create_dataset(key, data=value)
        f.attrs['metadata'] = json.dumps(metadata, allow_nan=False)
    (out / 'latent_distance_metrics.json').write_text(encoded + '\n')
    plot_latent_dist(values, horizon, out)
    print(f'Latent-distance GPC diagnostics saved to: {out}')
    return result


def run_offline_latent_dist(cfg, policy, goal_loader, observation_builder):
    robot = cfg.eval.real_robot
    if not robot.goal_image_path or not robot.goal_wrist_image_path:
        raise ValueError('Offline GPC requires saved goal images; capture is unavailable')
    goal, wrist, goal_proprio = goal_loader(None, robot)

    def observation(data, row, proprio):
        return observation_builder(data['pixels'][row], data['wrist_pixels'][row],
                                   goal, wrist, proprio[:7], proprio[7], goal_proprio,
                                   int(data['offline_step_idx'][row]), policy.process)

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
    metadata['goal_proprio'] = np.asarray(goal_proprio).tolist()
    execute_steps = getattr(robot, 'dp_execute_steps', 1)
    metadata['normalization_statistics'] = {
        group: {key: {name: np.asarray(getattr(processor, name)).tolist()
                      for name in ('mean_', 'scale_', 'low', 'span')
                      if getattr(processor, name, None) is not None}
                for key, processor in processors.items()}
        for group, processors in (('dp', policy.diffusion_policy.process), ('wm', policy.process))
    }
    return evaluate_latent_dist(
        policy, robot.dry_run_dataset_path, int(robot.dry_run_episode_idx),
        getattr(robot, 'dry_run_num_candidates', 20),
        getattr(robot, 'dry_run_seed', 42),
        Path(robot.output_dir) / 'offline_gpc' / 'latent_dist', metadata, observation,
        dp_execute_steps=execute_steps)


