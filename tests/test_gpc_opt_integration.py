"""Gradient and bitwise regression checks through the public GPC chunk entry point."""
from pathlib import Path
from types import SimpleNamespace
import subprocess
from unittest.mock import patch

import numpy as np
import pytest
import torch
from torch import nn

from action_normalization import restore_action_processor
from jepa import JEPA
from stable_worldmodel import policy as policies
from stable_worldmodel.reward import gpc_scoring_options
from stable_worldmodel.smoothness import MeasuredHistory
from test_offline_gpc_eval import observation


class Proposals(nn.Module):
    obs_horizon = 2
    action_horizon = 3
    device = torch.device('cpu')

    def __init__(self):
        super().__init__()
        self.offset = nn.Parameter(torch.tensor(.1))

    def sample_action_sequences(self, info, num_samples, denormalize):
        assert denormalize
        self.actions = torch.randn(1, num_samples, 4, 8) * .02 + self.offset
        self.actions[..., 3:7] = torch.tensor([0., 0., 0., 1.])
        self.actions[..., 7] = .5
        return self.actions


class Encoder(nn.Module):
    def forward(self, image, **kwargs):
        return SimpleNamespace(last_hidden_state=image.mean((-1, -2))[:, None])


class Predictor(nn.Module):
    def forward(self, emb, actions):
        return emb * .8 + actions


def components():
    torch.manual_seed(101)
    wm = JEPA(Encoder(), Encoder(), Predictor(), nn.Linear(8, 8), nn.Linear(8, 2))
    return Proposals().eval(), wm.eval().requires_grad_(False)


def scaler(mode):
    a = torch.arange(8, dtype=torch.float32) / 10
    b = a + 1
    stats = {'action_mean': a, 'action_std': b} if mode == 'zscore' else {'action_min': a, 'action_max': b}
    return restore_action_processor(dict(stats, action_normalization=mode), 8)


def make(policy_cls, dp, wm, cfg, first=False, count=3):
    return policy_cls(dp, wm, num_candidates=count, execute_first_candidate=first,
                      process={'action_cartesian': scaler('zscore')},
                      transform={key: lambda x: x.float() / 255 for key in
                                 ('pixels', 'wrist_pixels', 'goal', 'goal_wrist_pixels')},
                      **gpc_scoring_options(cfg, wm))


def info():
    obs = observation()
    obs['goal_wrist_pixels'] = obs['wrist_pixels'].copy()
    history = MeasuredHistory(3)
    history.append(0, obs)
    obs['smoothness_history'] = history
    return obs


@pytest.fixture(scope='module')
def previous():
    repo = Path(__file__).resolve().parents[2] / 'stable-worldmodel'
    src = subprocess.check_output(['git', 'show', '36c313a:stable_worldmodel/policy.py'], cwd=repo)
    namespace = vars(policies).copy()
    exec(compile(src, '<before GPC-OPT>', 'exec'), namespace)
    return namespace['GPCPolicy']


@pytest.mark.parametrize('explicit', [False, True])
@pytest.mark.parametrize('first', [False, True])
@pytest.mark.parametrize('mode', ['final_goal', 'current_observation', 'latent_smoothness'])
def test_previous_chunk_and_intermediates_exact(previous, explicit, first, mode):
    dp, wm = components()
    cfg = dict(goal_mode=mode)
    old = make(previous, dp, wm, cfg, first)
    if explicit:
        cfg.update(rank={'enabled': True}, opt={'enabled': False})
    new = make(policies.GPCPolicy, dp, wm, cfg, first)

    def trace(policy):
        torch.manual_seed(891)
        with patch.object(policy, 'rollout', wraps=policy.rollout) as rollout:
            with patch.object(policy, 'reward_fn', wraps=policy.reward_fn) as reward:
                chunk = policy.get_action_chunk(info())
                output, obs = reward.call_args.args
                return [dp.actions, rollout.call_args.kwargs['action_sequences'], output['predicted_emb'],
                        reward(output, obs), policy.best_idx, policy.selected_candidate_index,
                        torch.from_numpy(chunk), torch.get_rng_state()]

    assert all(torch.equal(a, b) for a, b in zip(trace(old), trace(new)))


@pytest.mark.parametrize('mode', ['min_max', 'zscore'])
def test_normalization_exact_inverse_and_grad(mode):
    dp, wm = components()
    policy = make(policies.GPCPolicy, dp, wm, {})
    processor = scaler(mode)
    policy.process = {'action_cartesian': processor}
    x = torch.randn(2, 3, 4, 8, requires_grad=True)
    y = policy.normalize_action_for_optimization(x)
    assert y.dtype == x.dtype and y.device == x.device
    assert torch.equal(y, policy.normalize_action_for_world_model(x))
    np.testing.assert_allclose(processor.inverse_transform(y.detach().numpy()), x.detach().numpy(), atol=3e-7)
    grad, = torch.autograd.grad(y.sum(), x)
    slope = 2 / processor.span if mode == 'min_max' else 1 / processor.scale_
    torch.testing.assert_close(grad, torch.as_tensor(slope).to(x).expand_as(x))
    policy.process = {'action_cartesian': object()}
    with pytest.raises(TypeError, match='transform_tensor'):
        policy.normalize_action_for_optimization(x)


@pytest.mark.parametrize('mode', ['final_goal', 'current_observation', 'latent_smoothness'])
@pytest.mark.parametrize('rank,first,count', [(False, False, 5), (True, False, 1), (True, False, 3), (True, True, 3)])
def test_refine_every_candidate_and_select_after_final_rollout(mode, rank, first, count):
    dp, wm = components()
    policy = make(policies.GPCPolicy, dp, wm, dict(goal_mode=mode, rank={'enabled': rank},
                  opt={'enabled': True, 'num_steps': 2, 'step_size': .0001}), first, count)
    params = list(dp.parameters()) + list(wm.parameters())
    snapshots = [v.detach().clone() for v in params]
    with torch.inference_mode():
        chunk = policy.get_action_chunk(info())
    trace = policy.opt_trace
    initial, final = trace['initial_actions'], trace['optimized_actions']
    assert final.shape[1] == (count if rank else 1)
    assert torch.all((final - initial).abs().sum((-1, -2)) > 0)
    assert torch.all(trace['gradient_norm'] > 0)
    obs = info()
    output = policy.rollout(obs, policy.normalize_action_for_world_model(final))
    reward = policy.reward_fn(output, obs)
    assert torch.all(-reward < trace['initial_cost'])
    assert torch.equal(policy.best_idx, reward.argmax(1))
    selected = 0 if first else reward.argmax(1).item()
    assert policy.selected_candidate_index.item() == selected
    np.testing.assert_array_equal(chunk, final[:, selected].numpy())
    assert all(torch.equal(v, before) and v.grad is None for v, before in zip(params, snapshots))


def test_direct_proposal_skips_wm():
    dp, wm = components()
    policy = make(policies.GPCPolicy, dp, wm, dict(rank={'enabled': False}))
    with patch.object(wm, 'rollout', side_effect=AssertionError), patch.object(wm, 'encode', side_effect=AssertionError):
        chunk = policy.get_action_chunk({})
    np.testing.assert_array_equal(chunk, dp.actions[:, 0, 1:].numpy())


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_gpu_pretrained_checkpoints_chunk_dry_run(tmp_path):
    import h5py
    import sys
    from test_offline_gpc_eval import ROOT, REAL_DP, REAL_WM, REAL_DATA

    if not all(path.is_file() for path in (REAL_DATA, REAL_DP, Path(f'{REAL_WM}_object.ckpt'))):
        pytest.skip('Local production Flip Mug dataset and checkpoints required')
    # Map recorded rows without copying full-resolution frames onto disk.
    dataset = tmp_path / 'recorded.h5'
    with h5py.File(REAL_DATA) as source, h5py.File(dataset, 'w', libver='latest') as dest:
        for key in ('pixels', 'wrist_pixels', 'proprio', 'action_cartesian', 'episode_idx', 'step_idx'):
            data = h5py.VirtualSource(source[key])
            layout = h5py.VirtualLayout(shape=(2, *data.shape[1:]), dtype=source[key].dtype)
            layout[:] = data[:2]
            dest.create_virtual_dataset(key, layout)
    subprocess.run([sys.executable, 'eval_real_robot.py', '--config-name', 'flip_mug',
                    f'policy={REAL_WM}', f'gpc.diffusion_checkpoint={REAL_DP}',
                    'gpc.goal_mode=final_goal', 'gpc.opt.enabled=true', 'gpc.rank.enabled=true',
                    'gpc.opt.step_size=0.000001', 'gpc.opt.optimize_orientation=true',
                    'gpc.opt.optimize_gripper=true', 'eval.real_robot.execute=false',
                    'eval.real_robot.dry_run=true', 'eval.real_robot.dry_run_mode=chunk_rollout',
                    'eval.real_robot.dry_run_num_candidates=3', 'eval.probing.exe_probe=false',
                    f'eval.real_robot.dry_run_dataset_path={dataset}',
                    f'eval.real_robot.output_dir={tmp_path}', f'hydra.run.dir={tmp_path}/hydra'],
                   cwd=ROOT, check=True)
    with h5py.File(next(tmp_path.rglob('chunk_rollout.h5'))) as f:
        trace = f['opt']
        assert np.all(trace['gradient_norm'][:] > 0)
        assert np.all(trace['final_cost'][:] < trace['cost'][:, 0, 0])
        final = trace['optimized_actions'][:, 0]
        best = f['best_idx'][:]
        np.testing.assert_array_equal(best, np.argmin(trace['final_cost'], axis=1))
        np.testing.assert_array_equal(f['selected_chunks'][0, :2], final[0, best[0], :2])


@pytest.mark.parametrize('first', [False, True])
def test_winner_changes_after_opt_and_recorder_keeps_both_sequences(first, tmp_path):
    from test_gpc_visualization import vis
    dp, wm = components()
    dp.process = {'action_cartesian': scaler('zscore')}
    cfg = {'opt': {'enabled': True, 'step_size': .25, 'num_steps': 1, 'max_position_delta_m': 1.}}
    policy = make(policies.GPCPolicy, dp, wm, cfg, first, 2)
    # x^4 with eta=.25 maps x to x-x^3. The second candidate overtakes the first.
    base = torch.zeros(1, 2, 4, 8)
    base[..., 6] = 1
    base[0, :, :, 0] = torch.tensor([.6, .9])[:, None]
    policy.rollout = lambda info_dict, action_sequences: {'x': action_sequences[..., 0]}
    policy.reward_fn = lambda output, info: -output['x'].pow(4).sum(-1)
    with patch.object(dp, 'sample_action_sequences', return_value=base), vis.GPCRecorder(policy) as recorder:
        recorder.begin_step(0)
        action = policy.get_action_chunk({})
    assert policy.opt_trace['initial_cost'].argmin(1).item() == 0
    assert policy.best_idx.item() == 1
    executed = 0 if first else 1
    assert policy.selected_candidate_index.item() == executed
    np.testing.assert_array_equal(action, policy.opt_trace['optimized_actions'][:, executed].numpy())
    assert len(recorder.records) == 1
    recorder.save(tmp_path)
    with np.load(tmp_path / 'gpc/data/candidates.npz') as data:
        np.testing.assert_array_equal(data['record_0000_candidate_actions_physical'], base[0, :, 1:].numpy())
        np.testing.assert_array_equal(data['record_0000_opt_optimized_actions'], policy.opt_trace['optimized_actions'])
