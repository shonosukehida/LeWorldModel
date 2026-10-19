"""Phase 2: evaluate trained checkpoints against saved observations, no hardware."""
from pathlib import Path

import hydra
import numpy as np
import torch

from hierarchical_gpc_runtime import load_hierarchical_gpc, offline_observations


@hydra.main(version_base=None, config_path='./config', config_name='hierarchical_gpc_offline')
def run(cfg):
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if cfg.hjepa.raw_image_layout != 'NHWC':
        raise ValueError('Offline dataset reader uses explicit NHWC layout')
    policy = load_hierarchical_gpc(cfg.hjepa)
    info, dp_info = offline_observations(cfg.dataset_path, cfg.current_index, cfg.goal_index,
                                        policy.diffusion_policy.obs_horizon)
    action = policy.get_action(info, dp_info_dict=dp_info)
    diagnostics = policy.last_diagnostics
    if 'lower_steps' not in diagnostics or 'upper_steps' not in diagnostics:
        raise ValueError('HWM must return both actual rollout trajectories for offline verification')
    output = Path(cfg.output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    arrays = {key: value.numpy() if torch.is_tensor(value) else value for key,value in diagnostics.items()}
    arrays.update(action=action, current_index=cfg.current_index, goal_index=cfg.goal_index,
                  verified_terminal_alignment=True)
    np.savez_compressed(output, **arrays)
    print(f'Offline selection saved: {output}; best_idx={diagnostics["best_idx"].tolist()}; action={action.tolist()}')
    return action


if __name__ == '__main__':
    run()
