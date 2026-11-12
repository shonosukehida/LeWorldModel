"""Execution stride, full WM horizon, and real CLI progress contracts."""
from pathlib import Path
from unittest.mock import Mock, patch

import h5py
import numpy as np
import pytest
import torch
from tqdm.auto import tqdm

import offline_dp_eval as dp
import offline_gpc_eval as gpc
import test_offline_dp_eval as fixtures
import test_offline_gpc_eval as old


@pytest.mark.parametrize('execute,length', [(8,17), (4,17), (1,17), (4,300)])
def test_execution_stride_and_full_wm_horizon(tmp_path, capsys, execute, length):
    path, _, _ = fixtures.ChunkRolloutTests().setup_rollout(tmp_path, length)
    with h5py.File(path, 'a') as f:
        f['proprio'] = np.ones((length,8),np.float32)
    policy = old.make_policy()
    policy.diffusion_policy.obs_horizon = 8
    policy.diffusion_policy.pred_horizon = 16
    policy.diffusion_policy.action_horizon = policy.action_horizon = 8
    samples = []
    def sample(info, num_samples, denormalize):
        assert denormalize and num_samples == 20
        value = torch.rand(1, num_samples, 16, 8)
        samples.append(value[0].numpy().copy())
        return value
    policy.diffusion_policy.sample_action_sequences = Mock(side_effect=sample)
    policy.reward_fn = lambda output, info: torch.arange(20)[None].float()
    count = (length + execute - 1) // execute
    # Test real tqdm output; skip only expensive plots in this numerical contract.
    with patch.object(gpc,'plot_gpc_chunk_rollout'):
        result = gpc.evaluate_gpc_chunk_rollout(policy,path,7,20,42,tmp_path,{},
            lambda data,row,prop: old.observation(), dp_execute_steps=execute)
    output = capsys.readouterr().err
    assert 'GPC chunk rollout' in output and '100%' in output
    assert f'{count}/{count}' in output
    assert len(samples) == len(policy.world_model.calls) == count
    assert all(actions.shape == (1,20,8,8) for _,actions in policy.world_model.calls)
    meta = result['metadata']
    assert (meta['action_horizon'],meta['dp_execute_steps'],meta['chunk_width']) == (8,execute,execute)
    assert meta['visualization_interval'] == [7,7+execute]
    assert meta['wm_evaluation_intervals'] == [[7,15]]*count
    with h5py.File(Path(meta['output_dir'])/'chunk_rollout.h5') as f:
        assert f['predicted_chunks'].shape == (count,20,execute,8)
        assert f['selected_chunks'].shape == (count,execute,8)
        np.testing.assert_array_equal(f['chunk_start_steps'],np.arange(100,100+length,execute))
        np.testing.assert_array_equal(f['selected_candidate_indices'],f['candidate_rewards'][:].argmax(1))
        for i, row in enumerate(range(0,length,execute)):
            mask=f['chunk_valid_mask'][i]
            n=min(execute,length-row)
            assert mask.sum()==n
            np.testing.assert_array_equal(f['predicted_chunks'][i,:,mask],samples[i][:,7:7+n])
            np.testing.assert_array_equal(f['selected_chunks'][i,mask],samples[i][19,7:7+n])
            assert np.isnan(f['selected_chunks'][i,~mask]).all()
    if length==300:
        assert count==75


@pytest.mark.parametrize('execute',[9,0,-1,True,1.5])
def test_invalid_execution_stride(tmp_path,execute):
    policy=old.make_policy()
    policy.diffusion_policy.action_horizon=policy.action_horizon=8
    policy.diffusion_policy.pred_horizon=16
    with pytest.raises(ValueError,match='dp_execute_steps'):
        gpc.evaluate_gpc_chunk_rollout(policy,'unused.h5',7,20,42,tmp_path,{},Mock(),dp_execute_steps=execute)


def test_dp_progress_and_exception_propagation(tmp_path,capsys):
    path,policy,_=fixtures.ChunkRolloutTests().setup_rollout(tmp_path,18)
    dp.evaluate_chunk_rollout(policy,path,7,3,42,tmp_path,{})
    output=capsys.readouterr().err
    assert 'DP chunk rollout' in output and '100%' in output and '3/3' in output
    error=RuntimeError('sampler failed')
    policy.sample_action_sequences=Mock(side_effect=error)
    with pytest.raises(RuntimeError) as raised:
        dp.evaluate_chunk_rollout(policy,path,7,3,42,tmp_path,{})
    assert raised.value is error
    assert '100%' not in capsys.readouterr().err


@pytest.mark.parametrize('failure_index', [0, 1])
def test_gpc_progress_advances_after_wm_and_preserves_exception(tmp_path, failure_index):
    path=tmp_path/'data.h5'
    fixtures.OfflineTests().dataset(path)
    with h5py.File(path,'a') as f:
        f['proprio']=np.ones((5,8),np.float32)
    policy=old.make_policy()
    bars=[]
    def progress(*args,**kwargs):
        bar=tqdm(*args,**kwargs,mininterval=0)
        bars.append(bar)
        return bar
    error=RuntimeError('WM failed')
    def fail(output,info):
        completed = len(policy.world_model.calls) - 1
        assert bars[0].n == completed
        if completed == failure_index:
            raise error
        return torch.zeros(1,20)
    policy.reward_fn=fail
    before=torch.get_rng_state()
    with patch('tqdm.auto.tqdm',side_effect=progress),pytest.raises(RuntimeError) as raised:
        gpc.evaluate_gpc_chunk_rollout(policy,path,7,20,42,tmp_path,{},
            lambda *args:old.observation(),dp_execute_steps=1)
    assert raised.value is error
    assert bars[0].total==3 and bars[0].n==failure_index
    assert torch.equal(before,torch.get_rng_state())


def test_eight_step_artifacts_match_prechange_gpc(tmp_path):
    import subprocess
    import types
    source = subprocess.check_output(
        ['git','show','daadc29273f52e9ed3b37a8903f9bfbd219e3d0f:offline_gpc_eval.py'],
        cwd=fixtures.ROOT,text=True)
    baseline = types.ModuleType('baseline_gpc_stride')
    exec(compile(source,'<baseline GPC>','exec'),baseline.__dict__)
    path,_,_=fixtures.ChunkRolloutTests().setup_rollout(tmp_path,17)
    with h5py.File(path,'a') as f:
        f['proprio']=np.ones((17,8),np.float32)
    results=[]
    for i,module in enumerate((baseline,gpc)):
        policy=old.make_policy()
        policy.diffusion_policy.obs_horizon=8
        policy.diffusion_policy.pred_horizon=16
        policy.diffusion_policy.action_horizon=policy.action_horizon=8
        policy.diffusion_policy.sample_action_sequences=lambda info,num_samples,denormalize: torch.rand(1,num_samples,16,8)
        kwargs = {} if i==0 else {'dp_execute_steps':8}
        results.append(module.evaluate_gpc_chunk_rollout(policy,path,7,20,42,tmp_path/str(i),{},
            lambda *args:old.observation(),**kwargs))
    outs=[Path(r['metadata']['output_dir']) for r in results]
    with h5py.File(outs[0]/'chunk_rollout.h5') as a,h5py.File(outs[1]/'chunk_rollout.h5') as b:
        assert set(a)==set(b)
        for k in a: np.testing.assert_array_equal(a[k][()],b[k][()])
    for name in ('position','orientation','gripper'):
        assert (outs[0]/f'chunk_rollout_{name}.png').read_bytes()==(outs[1]/f'chunk_rollout_{name}.png').read_bytes()
    assert results[1]['metadata']['lambda_prop'] == 1.0
    assert results[1]['metadata']['image_latent_dimension'] is None
    for result in results:
        for key in ('run_id','output_dir','execution_note','dp_execute_steps',
                    'lambda_prop','image_latent_dimension','reward'):
            result['metadata'].pop(key,None)
    assert results[0]==results[1]
