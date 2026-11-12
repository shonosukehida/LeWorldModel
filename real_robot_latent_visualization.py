"""Optional post-save diagnostics. This module never controls robot hardware."""
from copy import deepcopy
from pathlib import Path
import random
import traceback
from types import SimpleNamespace


def enabled(cfg):
    from omegaconf import OmegaConf
    return (OmegaConf.select(cfg, 'policy_type') == 'gpc'
            and OmegaConf.select(cfg, 'eval.real_robot.execute', default=False) is True
            and OmegaConf.select(cfg, 'eval.real_robot.dry_run', default=False) is False
            and OmegaConf.select(cfg, 'eval.real_robot.latent_visualization.enabled', default=False) is True)


def checkpoint_path(reference):
    """Resolve exactly as AutoCostModel does, without loading a second model."""
    import stable_worldmodel as swm
    run = Path(str(reference)).expanduser()
    if not run.exists():
        run = Path(swm.data.utils.get_cache_dir(sub_folder='checkpoints')) / run
    if run.is_dir():
        files = list(run.glob('*_object.ckpt'))
        if not files:
            raise FileNotFoundError(f'No WM object checkpoint in {run}')
        return max(files, key=lambda p: p.stat().st_ctime).resolve()
    return Path(str(run) + '_object.ckpt').resolve(strict=True)


def capture_metadata(cfg, process, checkpoint):
    from offline_real_robot_latent_eval import fingerprint, load_wm_process
    import numpy as np
    stats = Path(cfg.eval.normalization_stats_path).expanduser()
    saved = load_wm_process(stats)
    for key in ('action_cartesian', 'proprio', 'goal_proprio'):
        for name in ('mean_', 'scale_'):
            if not np.array_equal(getattr(saved[key], name), getattr(process[key], name)):
                raise ValueError(f'Loaded WM processor differs from saved statistics: {key}.{name}')
    return dict(world_model_checkpoint=fingerprint(checkpoint),
                normalization_stats=fingerprint(stats), image_size=int(cfg.eval.img_size),
                evaluation_mode='private copy of loaded WM and GPC preprocessing')


def run_after_save(cfg, policy, run_dir):
    """Catch only auxiliary errors; an active control exception propagates unchanged."""
    if not enabled(cfg):
        return None
    state = random.getstate()
    try:
        return _evaluate_saved(cfg, policy, run_dir)
    except Exception as error:
        print(f'[Latent visualization error] {type(error).__name__}: {error}', flush=True)
        traceback.print_exc()
        return None
    finally:
        random.setstate(state)


def _evaluate_saved(cfg, policy, run_dir):
    from offline_gpc_eval import isolated_rng

    @isolated_rng
    def evaluate():
        import torch
        from stable_worldmodel.policy import GPCPolicy
        from offline_real_robot_latent_eval import evaluate_rollout, fingerprint
        metadata = getattr(policy, '_latent_visualization_metadata', None)
        if not metadata:
            raise ValueError('WM checkpoint/statistics provenance was not captured at loading time')
        if 'error' in metadata:
            raise ValueError(metadata['error'])
        if fingerprint(cfg.eval.normalization_stats_path) != metadata['normalization_stats']:
            raise ValueError('WM statistics changed after loading')
        # Isolate potential encoder caches/buffers from the live policy. Never copy
        # policy.env or the DP, and never call eval()/to() on the original model.
        model = deepcopy(policy.world_model)
        if model.training or any(module.training for module in model.modules()):
            raise ValueError('Loaded WM must already be in eval mode')
        private = GPCPolicy(SimpleNamespace(action_horizon=1), model, None,
                            process=deepcopy(policy.process), transform=deepcopy(policy.transform))
        from stable_worldmodel.reward import latent_reward_metadata
        details = dict(metadata, lambda_prop=latent_reward_metadata(policy)['lambda_prop'],
                       device=str(next(model.parameters()).device),
                       torch_version=str(torch.__version__))
        output = evaluate_rollout(private, Path(run_dir) / 'rollout.h5', metadata=details)
        print(f'Latent visualization saved to: {output}', flush=True)
        return output

    return evaluate()
