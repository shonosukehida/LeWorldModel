"""Proposal proprio contracts through the actual GPC offline evaluators."""
from copy import deepcopy
from pathlib import Path
import subprocess
import types

import h5py
import numpy as np
import pytest
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

import offline_dp_eval as dp_eval
import offline_gpc_eval as gpc_eval
from action_normalization import restore_action_processor
from imle_checkpoint import build_policy
from jepa_integration_baseline import integration_source
from stable_worldmodel.diffusion import ConditionalUnet1D, create_obs_encoder
from stable_worldmodel.diffusion.jepa import EmptyWristEncoder, JEPADiffusionPolicy
from stable_worldmodel.imle.proprio import fit_proprio_stats
from stable_worldmodel.policy import DiffusionPolicy, GPCPolicy
from test_gpc_opt_integration import components, make
from test_imle_integration import config as imle_config, stats as action_stats
from test_jepa_dp import synthetic

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def measured_dataset(tmp_path):
    """Interleaved episodes, nonzero step IDs, raw state distinct from GT."""
    path = tmp_path/'measured.h5'
    values = np.arange(10*8, dtype=np.float32).reshape(10,8)*1.7 + 100
    with h5py.File(path, 'w') as f:
        f['episode_idx'] = np.tile([99,7], 5)
        f['step_idx'] = np.repeat(np.arange(10,15), 2)
        f['proprio'] = values
        for key in ('pixels', 'wrist_pixels'):
            f[key] = np.random.default_rng(42).integers(256, size=(10,16,16,3), dtype=np.uint8)
        actions = np.zeros((10,8), dtype=np.float32)
        actions[:, :3] = [.5, 0, .2]
        actions[:,6] = 1
        actions[:,7] = .5
        f['action_cartesian'] = actions
    return path, values[1::2]


def proposal(kind, synthetic):
    torch.manual_seed(123)
    if kind.startswith('imle'):
        cfg = imle_config()
        cfg.policy.obs_horizon = 3
        cfg.policy.action_horizon = 2
        cfg.imle.use_proprio = kind == 'imle_state'
        prop_stats = fit_proprio_stats(torch.arange(80).reshape(10,8).float()) if cfg.imle.use_proprio else None
        policy = build_policy(cfg, proprio_stats=prop_stats)
    else:
        if kind == 'jepa':
            (encoder, _), _ = synthetic
            wrist, cls = EmptyWristEncoder(), JEPADiffusionPolicy
        else:
            encoder = create_obs_encoder('conv', {'feature_dim':8})
            wrist, cls = deepcopy(encoder), DiffusionPolicy
        width = encoder.output_shape()[0] + wrist.output_shape()[0]
        model = ConditionalUnet1D(8, global_cond_dim=width*3,
                                 down_dims=[16,32], diffusion_step_embed_dim=16, n_groups=8)
        policy = cls(model, encoder, DDPMScheduler(num_train_timesteps=4),
                     8,3,2,8,num_inference_steps=2,wrist_obs_encoder=wrist)
    for module in (policy.model, policy.obs_encoder, policy.wrist_obs_encoder):
        module.eval()
    policy.process = {'action_cartesian': restore_action_processor(action_stats(), 8)}
    policy.transform = {key:lambda image:image.float()/255 for key in ('pixels','wrist_pixels')}
    return policy


def policy_for(proposal, rank=True, opt=False):
    _, world = components()
    # Orientation optimization supports the non-unit quaternion proposals from
    # a small untrained generator using the existing optimizer's normalization.
    return make(GPCPolicy, proposal, world, dict(goal_mode='current_observation', rank={'enabled':rank},
                opt={'enabled':opt, 'num_steps':1, 'optimize_orientation':True}), count=2)


def observation(data, row, proprio):
    return dict(pixels=data['pixels'][None,row:row+1],
                wrist_pixels=data['wrist_pixels'][None,row:row+1],
                proprio=proprio[None,None])


def no_plots(monkeypatch, module=gpc_eval):
    monkeypatch.setattr(module, 'plot_gpc_chunk_rollout', lambda *a,**kw:None)
    monkeypatch.setattr(module, 'plot_gpc', lambda *a,**kw:None)
    monkeypatch.setattr(dp_eval, 'plot_multi_sample', lambda *a,**kw:None)


@pytest.mark.parametrize('kind', ['dp','imle_image','imle_state','jepa'])
@pytest.mark.parametrize('mode', ['chunk_rollout','multi_sample'])
def test_proposal_raw_history_and_real_sampling(kind, mode, synthetic, measured_dataset, tmp_path, monkeypatch):
    no_plots(monkeypatch)
    path, measured = measured_dataset
    p = proposal(kind, synthetic)
    policy = policy_for(p)
    seen, encoded = [], []
    sample = p.sample_action_sequences
    def record(info, **kwargs):
        seen.append({k:np.asarray(v).copy() for k,v in info.items()})
        result = sample(info, **kwargs)
        assert result.shape == (1,2,p.pred_horizon,8)
        assert torch.isfinite(result).all()
        return result
    p.sample_action_sequences = record
    if kind == 'jepa':
        # Inspect JEPA.encode's actual proprio input, after exactly one saved
        # normalization, independently of the raw observation-history assertion.
        original = p.obs_encoder.world_model.encode
        def encode(info):
            encoded.append(info['proprio'].detach().clone())
            return original(info)
        monkeypatch.setattr(p.obs_encoder.world_model, 'encode', encode)
    if mode == 'chunk_rollout':
        result = gpc_eval.evaluate_gpc_chunk_rollout(policy,path,7,2,42,tmp_path/'out',{},observation,dp_execute_steps=2)
        rows = [0,2,4]
    else:
        result = gpc_eval.evaluate_gpc_multi_sample(policy,path,7,11,2,42,tmp_path/'out',{},observation)
        rows = [1]
    assert len(seen) == len(rows)
    for info,row in zip(seen, rows):
        assert set(info) == {'pixels','wrist_pixels'} | ({'proprio'} if kind in ('imle_state','jepa') else set())
        if 'proprio' in info:
            expected = measured[np.maximum(np.arange(row-2,row+1),0)][None]
            assert info['proprio'].shape == (1,3,8)
            np.testing.assert_array_equal(info['proprio'], expected)
    if kind == 'jepa':
        assert len(encoded) == len(rows)
        for info,value in zip(seen,encoded):
            expected = (torch.from_numpy(info['proprio'])-p.obs_encoder.proprio_mean)/p.obs_encoder.proprio_std
            assert torch.equal(value,expected)
    assert len(list(Path(result['metadata']['output_dir']).glob('*.h5'))) == 1


@pytest.mark.parametrize('rank,opt', [(False,False),(True,False),(False,True),(True,True)])
def test_jepa_candidates_rank_and_opt(rank, opt, synthetic, measured_dataset, tmp_path, monkeypatch):
    no_plots(monkeypatch)
    p = proposal('jepa', synthetic)
    policy = policy_for(p,rank,opt)
    path, _ = measured_dataset
    result = gpc_eval.evaluate_gpc_chunk_rollout(policy,path,7,2,42,tmp_path/'out',{},observation,dp_execute_steps=2)
    with h5py.File(Path(result['metadata']['output_dir'])/'chunk_rollout.h5') as f:
        count = 2 if rank else 1
        assert f['predicted_chunks'].shape == (3,count,2,8)
        mask = f['chunk_valid_mask'][:]
        assert np.isfinite(f['selected_chunks'][:][mask]).all()
        if rank or opt:
            rewards = f['candidate_rewards'][:]
            assert np.isfinite(rewards).all()
            np.testing.assert_array_equal(f['selected_candidate_indices'],rewards.argmax(1))
        if opt:
            assert 'opt' in f
            assert np.isfinite(f['opt/final_cost'][:]).all()
            assert f['opt/gradient_norm'].shape[1] == 1


@pytest.mark.parametrize('kind', ['dp','imle_image','imle_state'])
def test_existing_proposals_rewards_selection_and_rng_exact(kind, synthetic, measured_dataset, tmp_path, monkeypatch):
    before = types.ModuleType('before_proprio_fix')
    source = subprocess.check_output(['git','show','037e4d17f1333ab53f4e169df620d948a4b116fa:offline_gpc_eval.py'],cwd=ROOT,text=True)
    exec(compile(source,'<before proprio fix>','exec'),before.__dict__)
    no_plots(monkeypatch, before)
    no_plots(monkeypatch)
    path, _ = measured_dataset
    original = proposal(kind, synthetic)
    outputs, states, candidates = [], [], []
    for index,module in enumerate((before,gpc_eval)):
        policy = policy_for(deepcopy(original))
        sample = policy.diffusion_policy.sample_action_sequences
        def record(info, sample=sample, **kwargs):
            value = sample(info, **kwargs)
            candidates.append(value.detach().clone())
            return value
        policy.diffusion_policy.sample_action_sequences = record
        torch.manual_seed(99)
        outputs.append(module.evaluate_gpc_chunk_rollout(policy,path,7,2,42,tmp_path/str(index),{},observation,dp_execute_steps=2))
        states.append(torch.get_rng_state())
    assert torch.equal(*states)
    assert len(candidates) == 6
    for old,new in zip(candidates[:3],candidates[3:]):
        assert old.shape == (1,2,original.pred_horizon,8)
        assert torch.equal(old,new)
    with h5py.File(Path(outputs[0]['metadata']['output_dir'])/'chunk_rollout.h5') as a, \
         h5py.File(Path(outputs[1]['metadata']['output_dir'])/'chunk_rollout.h5') as b:
        assert set(a) == set(b)
        for key in a:
            np.testing.assert_array_equal(a[key][()],b[key][()])
    assert outputs[0]['candidate_rewards'] == outputs[1]['candidate_rewards']
    assert outputs[0]['selected_candidate_indices'] == outputs[1]['selected_candidate_indices']


def test_only_proprio_guard_is_allowed():
    baseline = integration_source(ROOT,'offline_gpc_eval.py')
    current = (ROOT/'offline_gpc_eval.py').read_text()
    assert baseline.count("if getattr(dp, 'type', None) == 'imle' and dp.use_proprio:") == 2
    # The multi-sample IMLE evaluator selection remains type-specific.
    assert current.count("if getattr(dp, 'type', None) == 'imle' and dp.use_proprio:") == 1
    with pytest.raises(AssertionError,match='Unreviewed integration source change'):
        integration_source(ROOT,'offline_gpc_eval.py',current.replace('start = policy.diffusion_policy.obs_horizon - 1','start = policy.diffusion_policy.obs_horizon'))
