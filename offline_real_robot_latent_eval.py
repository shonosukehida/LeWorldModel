"""Read-only JEPA diagnostics for a saved real-robot rollout; no hardware imports."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import torch

from normalization_stats import load_normalization_process
from offline_gpc_eval import isolated_rng
from offline_latent_dist_eval import _latent, _l2

DEFAULT_STATS = '~/.stable_worldmodel/datasets/flip_mug/ep100_tm300_multiview_play_normal_brightness/process_stats.npz'
IMAGE_KEYS = ('pixels', 'wrist_pixels', 'goal', 'goal_wrist_pixels')


def fingerprint(path):
    path = Path(path).expanduser().resolve(strict=True)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return dict(path=str(path), sha256=digest.hexdigest(), size_bytes=path.stat().st_size)


def load_wm_process(path):
    # Never compute statistics or fall back to DP statistics.
    process, action_key = load_normalization_process(path)
    required = {'action_cartesian', 'proprio', 'goal_proprio'}
    if action_key != 'action_cartesian' or not required <= process.keys():
        raise ValueError('WM statistics require action_cartesian, proprio and goal_proprio')
    for key in required:
        scaler = process[key]
        if (np.asarray(scaler.mean_).shape != (1, 8)
                or np.asarray(scaler.scale_).shape != (1, 8)
                or not np.isfinite(scaler.mean_).all()
                or not np.isfinite(scaler.scale_).all() or np.any(scaler.scale_ <= 0)):
            raise ValueError(f'Invalid WM normalization statistics for {key}: expected finite (1,8)')
    if not (np.array_equal(process['proprio'].mean_, process['goal_proprio'].mean_)
            and np.array_equal(process['proprio'].scale_, process['goal_proprio'].scale_)):
        raise ValueError('Goal and observation proprio statistics must match GPC encode_goal remapping')
    return process


def validate_rollout(file):
    for key in (*IMAGE_KEYS, 'proprio', 'goal_proprio'):
        if key not in file:
            raise ValueError(f'rollout.h5 is missing required dataset: {key}')
    count = len(file['proprio'])
    if count < 1 or file['proprio'].shape != (count, 8) or file['goal_proprio'].shape != (8,):
        raise ValueError('Expected nonempty proprio (T,8) and fixed goal_proprio (8,)')
    for current, goal in (('pixels', 'goal'), ('wrist_pixels', 'goal_wrist_pixels')):
        shape = file[current].shape
        if (len(shape) != 4 or shape[0] != count or shape[-1] != 3
                or min(shape[1:3]) < 1):
            raise ValueError(f'Expected aligned {current} (T,H,W,3); got {shape}, T={count}')
        goal_shape = file[goal].shape
        if len(goal_shape) != 3 or goal_shape[-1] != 3 or min(goal_shape[:2]) < 1:
            raise ValueError(f'Expected fixed {goal} (H,W,3); got {goal_shape}')


def _input(value, name):
    value = np.asarray(value)
    if not np.issubdtype(value.dtype, np.number) or not np.isfinite(value).all():
        raise ValueError(f'Nonfinite or nonnumeric rollout input: {name}')
    return value[None, None]


@isolated_rng
@torch.inference_mode()
def evaluate_rollout(policy, rollout_path, output_dir=None, metadata=None, max_steps=None):
    """Encode a fixed goal once and each recorded observation once, using GPC APIs."""
    source = Path(rollout_path).expanduser().resolve(strict=True)
    out = Path(output_dir).expanduser().resolve() if output_dir else source.parent / 'latent_eval'
    # Keep evaluation outputs separate from the source data directory.
    if out == source or out in source.parents:
        raise ValueError('Output directory must not contain the source rollout')
    if policy.world_model.training or any(m.training for m in policy.world_model.modules()):
        raise ValueError('Expected world_model.eval() including all submodules')
    if max_steps is not None and (isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps < 1):
        raise ValueError('max_steps must be a positive integer')
    with h5py.File(source, 'r') as file:
        validate_rollout(file)
        count = min(len(file['proprio']), max_steps) if max_steps else len(file['proprio'])
        goal_info = {key: _input(file[key][()], key)
                     for key in ('goal', 'goal_wrist_pixels', 'goal_proprio')}
        goal = _latent(policy.encode_goal(goal_info), 'goal_emb').detach()
        device = next(policy.world_model.parameters()).device
        observed = []
        for row in range(count):
            info = {key: _input(file[key][row], key) for key in ('pixels', 'wrist_pixels', 'proprio')}
            info['step_idx'] = np.asarray([[row]], dtype=np.int64)
            info = policy._prepare_info(info)
            info = {key: value.to(device) if torch.is_tensor(value) else value for key, value in info.items()}
            latent = _latent(policy.world_model.encode(info)['emb'], 'observed_emb')
            if latent.shape != goal.shape:
                raise ValueError('Observed and goal latent shapes must match exactly')
            observed.append(latent[0, -1].detach().cpu())
        recorded_config = file.attrs.get('config', '')
        input_shapes = {key: list(file[key].shape) for key in (*IMAGE_KEYS, 'proprio', 'goal_proprio')}
    observed = torch.stack(observed)
    goal = goal[0, -1].cpu()
    values = dict(step_idx=np.arange(count, dtype=np.int64),
                  observed_latent_distance=_l2(observed, goal.expand_as(observed)),
                  observed_latents=observed.numpy(), goal_latent=goal.numpy())
    metrics = dict(metadata or {}, input_rollout=fingerprint(source), evaluation_steps=count,
                   latent_dimension=goal.numel(), distance='L2 (not squared): ||z_t - z_goal||_2',
                   executed_at=datetime.now(timezone.utc).isoformat(), goal_encoded_once=True,
                   preprocessing='GPCPolicy._prepare_info / encode_goal; get_eval_img_preprocessor; final time latent (1,1,D); no additional transforms',
                   recorded_rollout_config=str(recorded_config), input_shapes=input_shapes,
                   prediction_comparison='Not computed: rollout has no recorded selected prediction latents with execution alignment')
    encoded = json.dumps(metrics, indent=2, allow_nan=False)
    out.mkdir(parents=True, exist_ok=True)
    # Render a complete new result before replacing only these owned outputs.
    # Replacing files also avoids modifying any symlink/hardlink targets in-place.
    names = ('latent_distance.h5', 'latent_distance_metrics.json',
             'observed_latent_distance.png')
    with tempfile.TemporaryDirectory(prefix='.latent_eval_', dir=out) as temporary:
        staged = Path(temporary)
        with h5py.File(staged / names[0], 'x') as file:
            for key, value in values.items():
                file.create_dataset(key, data=value)
            file.attrs['metadata'] = encoded
        (staged / names[1]).write_text(encoded + '\n')
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 4))
        try:
            ax.plot(values['step_idx'], values['observed_latent_distance'], label='Observed latent distance', marker='.' if count == 1 else None)
            ax.set_xlabel('Observation step')
            ax.set_ylabel('Latent L2 distance to Goal')
            ax.grid(alpha=.25)
            ax.legend()
            fig.tight_layout()
            fig.savefig(staged / names[2])
        finally:
            plt.close(fig)
        for name in names:
            os.replace(staged / name, out / name)
    return out


def validate_recorded_config(source, checkpoint, stats, image_size):
    """Fail on configuration mismatch rather than silently evaluating another WM."""
    from omegaconf import OmegaConf
    import stable_worldmodel as swm
    with h5py.File(source, 'r') as file:
        if 'config' not in file.attrs:
            raise ValueError('rollout.h5 has no recorded config; WM/preprocessing consistency cannot be verified')
        cfg = OmegaConf.create(file.attrs['config'])
    if OmegaConf.select(cfg, 'eval.img_size') != image_size:
        raise ValueError('image-size differs from recorded eval.img_size')
    recorded_stats = OmegaConf.select(cfg, 'eval.normalization_stats_path')
    if not recorded_stats:
        raise ValueError('Recorded WM normalization_stats_path is missing')
    recorded_stats = Path(recorded_stats).expanduser().resolve()
    supplied_stats = Path(stats).expanduser().resolve()
    if recorded_stats != supplied_stats:
        if not recorded_stats.is_file() or fingerprint(recorded_stats)['sha256'] != fingerprint(supplied_stats)['sha256']:
            raise ValueError('WM statistics differ from recorded GPC statistics (or recorded statistics cannot be verified)')
    recorded_model = OmegaConf.select(cfg, 'policy')
    if not recorded_model or recorded_model == 'random':
        raise ValueError('Recorded config has no trained WM policy')
    run = Path(str(recorded_model)).expanduser()
    if not run.exists():
        run = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints')) / run
    checkpoint = Path(checkpoint).resolve()
    expected = run.resolve()
    if not (checkpoint.parent == expected if run.is_dir() else checkpoint == Path(str(expected) + '_object.ckpt')):
        raise ValueError('WM checkpoint differs from the recorded policy run')
    if OmegaConf.select(cfg, 'eval.probing.use_random_encoder', default=False):
        raise ValueError('Recorded run used a random encoder; checkpoint encoder does not reproduce it')


@isolated_rng
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('rollout', type=Path)
    parser.add_argument('--wm-checkpoint', required=True, type=Path, help='Exact *_object.ckpt used during inference')
    parser.add_argument('--wm-stats', default=DEFAULT_STATS, type=Path, help='WM training statistics; never DP statistics')
    parser.add_argument('--image-size', required=True, type=int, help='Same eval.img_size as real GPC')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--max-steps', type=int)
    args = parser.parse_args()
    process = load_wm_process(args.wm_stats)
    checkpoint = args.wm_checkpoint.expanduser().resolve(strict=True)
    if not checkpoint.name.endswith('_object.ckpt'):
        raise ValueError('Provide the exact WM *_object.ckpt file')
    if args.image_size < 1:
        raise ValueError('image-size must be positive')
    validate_recorded_config(args.rollout, checkpoint, args.wm_stats, args.image_size)
    # No eval_real_robot import: no robot, camera, GELLO, or DP checkpoint needed.
    from stable_worldmodel.policy import AutoCostModel, GPCPolicy
    from utils import get_eval_img_preprocessor
    model = AutoCostModel(str(checkpoint)[:-len('_object.ckpt')]).to(args.device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    transform = get_eval_img_preprocessor(args.image_size)
    policy = GPCPolicy(SimpleNamespace(action_horizon=1), model, reward_fn=None,
                       process=process, transform={key: transform for key in IMAGE_KEYS})
    metadata = dict(world_model_checkpoint=fingerprint(checkpoint),
                    normalization_stats=fingerprint(args.wm_stats), image_size=args.image_size,
                    device=str(args.device), torch_version=str(torch.__version__),
                    numpy_version=np.__version__)
    print(evaluate_rollout(policy, args.rollout, args.output_dir, metadata, args.max_steps))


if __name__ == '__main__':
    main()
