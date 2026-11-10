"""GPC chunk rollout contracts and byte-for-byte DP regression."""
import json
from pathlib import Path
import subprocess
import types
from unittest.mock import Mock, patch

import h5py
import numpy as np
import pytest
import torch

import offline_dp_eval as dp
import offline_gpc_eval as gpc
import test_offline_gpc_eval as old
import test_offline_dp_eval as fixtures


def test_chunks_selection_rng_gt_independence(tmp_path):
    path = tmp_path/'data.h5'
    fixtures.OfflineTests().dataset(path)
    with h5py.File(path, 'a') as f:
        f['proprio'] = np.full((5,8),17,dtype=np.float32)
    policy = old.make_policy()
    policy.num_candidates = 5
    calls = Mock(wraps=policy.diffusion_policy.sample_action_sequences)
    policy.diffusion_policy.sample_action_sequences = calls
    builder = Mock(side_effect=lambda data,row,prop: old.observation())
    def reward(output, info):
        r = torch.zeros(1,20)
        r[0, (len(policy.world_model.calls)-1) % 2] = 1
        return r
    policy.reward_fn = reward
    before = torch.get_rng_state(), np.random.get_state()
    result = gpc.evaluate_gpc_chunk_rollout(policy,path,7,20,42,tmp_path,{},builder)
    assert torch.equal(before[0],torch.get_rng_state())
    assert all(np.array_equal(a,b) for a,b in zip(before[1],np.random.get_state()))
    assert calls.call_count == 2
    assert policy.num_candidates == 5
    assert [c.kwargs['num_samples'] for c in calls.call_args_list] == [20,20]
    assert [c.args[1] for c in builder.call_args_list] == [0,2]
    for c in builder.call_args_list:
        np.testing.assert_array_equal(c.args[2],np.full(8,17))
    with h5py.File(Path(result['metadata']['output_dir'])/'chunk_rollout.h5') as f:
        np.testing.assert_array_equal(f['selected_candidate_indices'],[0,1])
        np.testing.assert_array_equal(f['chunk_valid_mask'],[[1,1],[1,0]])
        np.testing.assert_array_equal(f['chunk_start_steps'],[0,2])
        np.testing.assert_array_equal(f['selected_candidate_indices'],f['candidate_rewards'][:].argmax(1))
        for i,k in enumerate(f['selected_candidate_indices']):
            np.testing.assert_array_equal(f['selected_chunks'][i],f['predicted_chunks'][i,k])
        assert f['candidate_rewards'].shape == (2,20)
        saved = f['predicted_chunks'][:]
    assert all(c[1].shape == (1,20,2,8) for c in policy.world_model.calls)
    with h5py.File(path,'a') as f:
        f['action_cartesian'][:] = 999
    policy.world_model.calls.clear()
    repeat = gpc.evaluate_gpc_chunk_rollout(policy,path,7,20,42,tmp_path,{},builder)
    assert repeat['candidate_rewards'] == result['candidate_rewards']
    with h5py.File(Path(repeat['metadata']['output_dir'])/'chunk_rollout.h5') as f:
        np.testing.assert_array_equal(saved,f['predicted_chunks'][:])


def test_plot_chunk_boundaries_and_colors(tmp_path):
    import matplotlib.pyplot as plt
    figures=[]
    original=plt.subplots
    def capture(*a,**kw):
        fig,axes=original(*a,**kw); figures.append(axes); return fig,axes
    with patch.object(plt,'subplots',side_effect=capture):
        gpc.plot_gpc_chunk_rollout(np.ones((2,3,2,8)),np.zeros((3,8)),np.arange(3),
            np.array([[0,1],[2,-1]]),np.array([[1,1],[1,0]],bool),[0,2],tmp_path)
    assert [len(a) for a in figures] == [3,4,1]
    for axes in figures:
        for ax in axes[:,0]:
            red=[line for line in ax.lines if line.get_color()=='red']
            assert len(red)==2
            assert [list(line.get_xdata()) for line in red]==[[0,1],[2]]
            assert all(line.get_zorder()==10 and line.get_linewidth()==3 for line in red)
            assert ax.lines[0].get_color()=='black'


def test_dp_chunk_artifacts_against_actual_starting_revision(tmp_path):
    source=subprocess.check_output(['git','show','437aec5d059b4000cdaf994fc597d6f32047a335:offline_dp_eval.py'],cwd=fixtures.ROOT,text=True)
    assert source==(fixtures.ROOT/'offline_dp_eval.py').read_text()
    baseline=types.ModuleType('baseline'); exec(compile(source,'<baseline>','exec'),baseline.__dict__)
    results=[]
    for i,module in enumerate((baseline,dp)):
        path,policy,_=fixtures.ChunkRolloutTests().setup_rollout(tmp_path)
        results.append(module.evaluate_chunk_rollout(policy,path,7,3,42,tmp_path/str(i),{}))
    outs=[Path(r['metadata']['output_dir']) for r in results]
    for r in results:
        for k in ('output_dir','run_id'): del r['metadata'][k]
    assert results[0]==results[1]
    with h5py.File(outs[0]/'chunk_rollout.h5') as a,h5py.File(outs[1]/'chunk_rollout.h5') as b:
        assert set(a)==set(b)
        for k in a: np.testing.assert_array_equal(a[k][()],b[k][()])
    for name in ('position','orientation','gripper'):
        assert (outs[0]/f'chunk_rollout_{name}.png').read_bytes()==(outs[1]/f'chunk_rollout_{name}.png').read_bytes()
    docs=[json.loads((o/'chunk_rollout_metrics.json').read_text()) for o in outs]
    for d in docs:
        for k in ('output_dir','run_id'): del d['metadata'][k]
    assert docs[0]==docs[1]


@pytest.mark.parametrize('mode',['multi_sample','chunk_rollout'])
def test_hardware_free_dispatch_without_gpc_count(mode):
    # Execute the existing comprehensive dispatcher fixture for both modes.
    import inspect
    source=inspect.getsource(old.test_run_routes_before_hardware_and_uses_offline_transform)
    source=source.replace('test_run_routes_before_hardware_and_uses_offline_transform','check')
    source=source.replace('multi_sample',mode)
    ns=dict(old.__dict__)
    exec(source,ns)
    ns['check']()


@pytest.mark.skipif(not all(p.exists() for p in (old.REAL_DP,Path(f'{old.REAL_WM}_object.ckpt'),old.REAL_DATA,old.REAL_STATS)),reason='Production files unavailable')
def test_real_checkpoints_cpu_chunk_rollout():
    import inspect
    source=inspect.getsource(old.test_production_checkpoints_cpu)
    source=source[source.index('def test_production'):]
    source=source[:source.index('    dp_result =')]
    source=source.replace('test_production_checkpoints_cpu','check').replace('multi_sample','chunk_rollout')
    source=source.replace("source[key][:5]", "source[key][:17]")
    source += "    assert result['metadata']['num_candidates'] == 20\n    print('Real chunk artifacts:', result['metadata']['output_dir'])\n"
    ns=dict(old.__dict__); exec(source,ns); ns['check']()


def test_robot_keeps_five_candidates_and_offline_failure_restores_rng():
    policy=old.make_policy()
    policy.num_candidates=5
    sampler=Mock(wraps=policy.diffusion_policy.sample_action_sequences)
    policy.diffusion_policy.sample_action_sequences=sampler
    selected=policy.get_action_chunk(old.observation())
    assert selected.shape==(1,2,8)
    assert sampler.call_args.kwargs['num_samples']==5
    from omegaconf import OmegaConf
    cfg=OmegaConf.create({'policy_type':'gpc','policy':'trained','eval':{'real_robot':{
        'dry_run':True,'dry_run_mode':'chunk_rollout'}}})
    @gpc.preserve_gpc_offline_rng
    def fail(cfg):
        torch.rand(5); np.random.rand(5)
        raise RuntimeError('loading')
    before=torch.get_rng_state(),np.random.get_state()
    with pytest.raises(RuntimeError): fail(cfg)
    assert torch.equal(before[0],torch.get_rng_state())
    assert all(np.array_equal(a,b) for a,b in zip(before[1],np.random.get_state()))


def test_dp_episode_artifacts_against_starting_revision(tmp_path):
    source=subprocess.check_output(['git','show','437aec5d059b4000cdaf994fc597d6f32047a335:offline_dp_eval.py'],cwd=fixtures.ROOT,text=True)
    baseline=types.ModuleType('baseline_episode'); exec(compile(source,'<baseline>','exec'),baseline.__dict__)
    path=tmp_path/'data.h5'; fixtures.OfflineTests().dataset(path)
    metrics=[]
    outs=[tmp_path/'before',tmp_path/'after']
    for module,out in zip((baseline,dp),outs):
        policy=fixtures.MultiSampleTests().policy()
        policy.get_action=lambda info: policy.sample_action_sequences(info,num_samples=1,denormalize=True)[0,:,policy.obs_horizon-1].numpy()
        with torch.random.fork_rng():
            torch.manual_seed(42)
            metrics.append(module.evaluate_episode(policy,path,7,out,{}))
    assert metrics[0]==metrics[1]
    assert (outs[0]/'metrics.json').read_bytes()==(outs[1]/'metrics.json').read_bytes()
    with h5py.File(outs[0]/'offline_rollout.h5') as a,h5py.File(outs[1]/'offline_rollout.h5') as b:
        assert dict(a.attrs)==dict(b.attrs)
        for k in a: np.testing.assert_array_equal(a[k][()],b[k][()])
    for path in outs[0].glob('*.png'):
        assert path.read_bytes()==(outs[1]/path.name).read_bytes()
