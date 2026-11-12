"""Offline-only asset smoke test. Never imports the robot evaluation entrypoint.

Without --imle-checkpoint, explicitly creates an UNTRAINED diagnostic checkpoint;
this mode cannot establish compatibility with a previously trained IMLE policy.
"""
import argparse
import hashlib
import importlib.abc
import inspect
import json
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
LWM = Path(__file__).resolve().parents[1]
SWM = LWM.parent / 'stable-worldmodel'
sys.path[:0] = [str(LWM), str(SWM), str(LWM.parent / 'test-deps')]
os.environ.setdefault('MPLCONFIGDIR', '/tmp/imle-followup-mpl')
os.environ['HF_HUB_OFFLINE'] = '1'
os.environ['TRANSFORMERS_OFFLINE'] = '1'


class NoHardwareImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'robopy', 'xarm', 'pyrealsense2', 'robot', 'env', 'eval_real_robot'}:
            raise RuntimeError(f'Hardware/entrypoint import prohibited: {fullname}')
        return None


sys.meta_path.insert(0, NoHardwareImports())
import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
import stable_worldmodel as swm
from stable_worldmodel.policy import GPCPolicy
from stable_worldmodel.reward import LatentGoalReward
from imle_checkpoint import build_policy, save_checkpoint, load_imle_policy, parameters
from normalization_stats import load_normalization_process
from offline_dp_eval import load_episode, build_observation_history, evaluate_episode
from offline_imle_eval import run_offline_imle_gpc_from_config
from utils import get_eval_img_preprocessor


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def audit_origins():
    origins = {}
    local = {'jepa', 'module', 'utils', 'imle_checkpoint', 'imle_runtime',
             'offline_imle_eval', 'offline_dp_eval', 'normalization_stats', 'action_normalization'}
    for name, module in list(sys.modules.items()):
        path = getattr(module, '__file__', None)
        if path and (name.split('.')[0] in local or name.split('.')[0] == 'stable_worldmodel'):
            path = Path(path).resolve()
            expected = SWM if name.split('.')[0] == 'stable_worldmodel' else LWM
            assert path.is_relative_to(expected), (name, str(path), str(expected))
            origins[name] = dict(path=str(path), sha256=digest(path))
    return origins


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    parser.add_argument('--world-model', type=Path, required=True, help='prefix before _object.ckpt')
    parser.add_argument('--stats', type=Path, required=True)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--imle-checkpoint', type=Path)
    parser.add_argument('--rows', type=int, default=16)
    parser.add_argument('--candidates', type=int, default=20)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(41)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    if args.device == 'cuda':
        assert torch.cuda.is_available(), 'CUDA is unavailable; no CPU fallback'
        torch.cuda.reset_peak_memory_stats()
    stat_hash = digest(args.stats)
    process, _ = load_normalization_process(args.stats)
    sample = args.output / 'real_observation_slice.h5'
    with h5py.File(args.dataset, 'r') as f, h5py.File(sample, 'w') as out:
        for key in ('pixels', 'wrist_pixels', 'proprio', 'action_cartesian', 'episode_idx', 'step_idx'):
            out[key] = f[key][:args.rows]
        out.attrs['source_path'] = str(args.dataset.resolve())
        out.attrs['scope'] = 'prefix only; stored images/actions/proprio are unchanged'
    checkpoint = args.imle_checkpoint
    kind = 'existing_checkpoint'
    if checkpoint is None:
        kind = 'UNTRAINED_DIAGNOSTIC_ONLY'
        cfg = OmegaConf.load(LWM / 'config/train/policy/imle_policy.yaml')
        cfg.validation_only = True
        proposal = build_policy(cfg)
        scaler = process['action_cartesian']
        action_stats = dict(action_normalization='min_max',
                            action_min=torch.as_tensor(scaler.raw_min_),
                            action_max=torch.as_tensor(scaler.raw_max_))
        checkpoint = args.output / 'UNTRAINED_DIAGNOSTIC_ONLY.ckpt'
        save_checkpoint(checkpoint, proposal, torch.optim.AdamW(parameters(proposal)),
                        -1, float('nan'), cfg, action_stats)
        del proposal
    state = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if state['config'].get('validation_only') or checkpoint.name == 'UNTRAINED_DIAGNOSTIC_ONLY.ckpt':
        kind = 'UNTRAINED_DIAGNOSTIC_ONLY'
    checkpoint_info = dict(path=str(checkpoint.resolve()), sha256=digest(checkpoint),
                           kind=kind, epoch=state.get('epoch'),
                           optimizer_state_count=len(state.get('optimizer_state_dict', {}).get('state', {})),
                           encoder_architecture=state.get('encoder_architecture'))
    del state
    transform = get_eval_img_preprocessor(224)
    policy = load_imle_policy(checkpoint, transform, args.device)
    data, steps, _ = load_episode(sample, 0)
    history = build_observation_history(data, 0, policy.obs_horizon)
    with h5py.File(sample) as f:
        proprio = np.asarray(f['proprio'])
    info = dict(pixels=data['pixels'][0][None, None], wrist_pixels=data['wrist_pixels'][0][None, None],
                proprio=proprio[0][None, None], goal=data['pixels'][-1][None, None],
                goal_wrist_pixels=data['wrist_pixels'][-1][None, None], goal_proprio=proprio[-1][None, None])
    world = swm.policy.AutoCostModel(str(args.world_model)).to(args.device).eval()
    world.requires_grad_(False)
    world.interpolate_pos_encoding = True
    assert Path(inspect.getfile(type(world))).resolve().is_relative_to(LWM)
    gpc = GPCPolicy(action_policy=policy, world_model=world,
                    reward_fn=LatentGoalReward(world, 1.0), num_candidates=args.candidates,
                    process=process, transform={k: transform for k in ('pixels', 'wrist_pixels', 'goal', 'goal_wrist_pixels')})
    with torch.no_grad():
        torch.manual_seed(42)
        standalone = policy.get_action_chunk(history)
        assert np.isfinite(standalone).all()
        torch.manual_seed(42)
        proposals = policy.sample_action_sequences(history, args.candidates, True)
        assert proposals.device.type == args.device
        start = policy.obs_horizon - 1
        candidates = proposals[:, :, start:start + policy.action_horizon]
        actions = gpc.normalize_action_for_world_model(candidates)
        rollout = gpc.rollout(info, actions)
        rewards = gpc.reward_fn(rollout, info)
        assert torch.isfinite(rollout['predicted_emb']).all() and torch.isfinite(rewards).all()
        best = rewards.argmax(1).item()
        torch.manual_seed(42)
        selected = gpc.get_action_chunk(info, dp_info_dict=history)
        np.testing.assert_array_equal(selected, candidates[:, best].cpu().numpy())
        np.savez(args.output / 'selection.npz', candidates=candidates.cpu().numpy(),
                 rewards=rewards.cpu().numpy(), selected=selected, best_index=best,
                 standalone=standalone)
    evaluate_episode(policy, sample, 0, args.output / 'standalone',
                     dict(policy.offline_checkpoint_metadata, checkpoint_kind=kind),
                     dp_execute_steps=policy.action_horizon)
    cfg = OmegaConf.load(LWM / 'config/eval/flip_mug.yaml')
    cfg.policy_type = 'gpc'
    cfg.policy = str(args.world_model)
    cfg.gpc.diffusion_checkpoint = None
    cfg.gpc.imle_checkpoint = str(checkpoint)
    cfg.gpc.imle_device = args.device
    cfg.gpc.num_candidates = args.candidates
    cfg.eval.normalization_stats_path = str(args.stats)
    robot = cfg.eval.real_robot
    robot.dry_run = True
    robot.execute = False
    robot.dry_run_mode = 'episode'
    robot.dry_run_dataset_path = str(sample)
    robot.dry_run_episode_idx = 0
    robot.dp_execute_steps = policy.action_horizon
    robot.output_dir = str(args.output / 'from_config')
    # Real config loader/normalization/recorder, reusing the already loaded WM.
    predicted = run_offline_imle_gpc_from_config(cfg, transform, lambda path: world)
    assert predicted.shape == (len(steps), 8) and np.isfinite(predicted).all()
    assert digest(args.stats) == stat_hash
    device_comparison = None
    if args.device == 'cuda':
        # Controlled diagnostic: common Gaussian inputs distinguish sampling-stream
        # differences from floating-point differences. Only these test instances
        # are instrumented; production source remains untouched.
        import copy
        cpu_policy = load_imle_policy(checkpoint, transform, 'cpu')
        torch.manual_seed(42)
        common_noise = cpu_policy._noise(args.candidates)
        torch.manual_seed(42)
        cuda_noise = policy._noise(args.candidates)
        cpu_noise_method, cuda_noise_method = cpu_policy._noise, policy._noise
        try:
            cpu_policy._noise = lambda batch: common_noise.clone()
            policy._noise = lambda batch: common_noise.to(args.device)
            with torch.no_grad():
                common_cpu = cpu_policy.sample_action_sequences(history, args.candidates, True)
                common_cuda = policy.sample_action_sequences(history, args.candidates, True)
        finally:
            cpu_policy._noise, policy._noise = cpu_noise_method, cuda_noise_method
        cpu_world = copy.deepcopy(world).cpu()
        cpu_gpc = GPCPolicy(action_policy=cpu_policy, world_model=cpu_world,
                            reward_fn=LatentGoalReward(cpu_world, 1.0), num_candidates=args.candidates,
                            process=process, transform=gpc.transform)
        fixed = common_cpu[:, :, start:start + policy.action_horizon]
        with torch.no_grad():
            fixed_cpu_rewards = cpu_gpc.reward_fn(
                cpu_gpc.rollout(info, cpu_gpc.normalize_action_for_world_model(fixed)), info)
            fixed_cuda_rewards = gpc.reward_fn(
                gpc.rollout(info, gpc.normalize_action_for_world_model(fixed.to(args.device))), info)
        cpu_best = fixed_cpu_rewards.argmax(1).item()
        cuda_best = fixed_cuda_rewards.argmax(1).item()
        assert cpu_best == cuda_best, 'Identical physical candidates changed ranking across devices'
        device_comparison = dict(
            same_seed_device_noise_equal=torch.equal(common_noise, cuda_noise.cpu()),
            common_noise_proposals_bitwise_equal=torch.equal(common_cpu, common_cuda.cpu()),
            common_noise_proposals_max_abs_error=float((common_cpu - common_cuda.cpu()).abs().max()),
            fixed_candidates_rewards_bitwise_equal=torch.equal(fixed_cpu_rewards, fixed_cuda_rewards.cpu()),
            fixed_candidates_rewards_max_abs_error=float((fixed_cpu_rewards - fixed_cuda_rewards.cpu()).abs().max()),
            fixed_candidates_cpu_best=cpu_best, fixed_candidates_cuda_best=cuda_best,
            note='No cross-device bitwise equality or tolerance is assumed; exact argmax equality checked.')
        np.savez(args.output / 'controlled_device_comparison.npz',
                 cpu_noise=common_noise.cpu().numpy(), cuda_noise=cuda_noise.cpu().numpy(),
                 common_noise_cpu_proposals=common_cpu.cpu().numpy(),
                 common_noise_cuda_proposals=common_cuda.cpu().numpy(),
                 fixed_candidates_cpu_rewards=fixed_cpu_rewards.cpu().numpy(),
                 fixed_candidates_cuda_rewards=fixed_cuda_rewards.cpu().numpy())
    result = dict(device=args.device, torch=torch.__version__, cuda_runtime=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(0) if args.device == 'cuda' else None,
                  checkpoint=checkpoint_info, world_model=dict(path=str(args.world_model) + '_object.ckpt',
                      sha256=digest(str(args.world_model) + '_object.ckpt'), class_name=str(type(world)),
                      parameter_count=sum(p.numel() for p in world.parameters())),
                  stats=dict(path=str(args.stats), sha256=stat_hash, unchanged=True),
                  dataset_source=str(args.dataset), rows=len(steps), candidates=args.candidates,
                  standalone_shape=list(standalone.shape), predicted_embedding_shape=list(rollout['predicted_emb'].shape),
                  selected_index=best, selection_exact=True, offline_output_shape=list(predicted.shape),
                  cuda_peak_bytes=torch.cuda.max_memory_allocated() if args.device == 'cuda' else None,
                  device_comparison=device_comparison,
                  origins=audit_origins(), hardware_imports=[])
    assert not any(n.split('.')[0] in {'robopy', 'xarm', 'pyrealsense2', 'robot', 'env', 'eval_real_robot'} for n in sys.modules)
    (args.output / 'result.json').write_text(json.dumps(result, indent=2))
    print(json.dumps({k:v for k,v in result.items() if k != 'origins'}, indent=2))


if __name__ == '__main__':
    main()
