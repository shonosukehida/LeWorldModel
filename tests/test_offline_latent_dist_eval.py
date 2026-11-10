"""Numerical, temporal and routing contracts using the actual GPC API."""
import inspect
import json
from pathlib import Path
from unittest.mock import Mock, patch

import h5py
import numpy as np
import pytest
import torch

import offline_latent_dist_eval as latent
import offline_dp_eval as dp_eval
import test_offline_dp_eval as fixtures
import test_offline_gpc_eval as old


class WorldModel(old.WorldModel):
    def __init__(self):
        super().__init__()
        self.encodings = []

    def encode(self, info):
        self.encodings.append({k:v.detach().clone() for k,v in info.items() if torch.is_tensor(v)})
        # Real GPC preprocessing supplies normalized proprio and both camera views.
        image = info['pixels'].float().mean(tuple(range(2,info['pixels'].ndim)))
        wrist = info['wrist_pixels'].float().mean(tuple(range(2,info['wrist_pixels'].ndim)))
        return {'emb': info['proprio'].float() + (image + wrist)[...,None] / 100}

    def rollout(self, info, actions):
        self.calls.append(({k:v.detach().clone() for k,v in info.items() if torch.is_tensor(v)}, actions.clone()))
        initial = info['proprio'].float()
        # One recorded current frame, never a future observation.
        assert initial.shape[2]==1
        return {'predicted_emb': initial + actions.cumsum(dim=2)}


def setup(tmp_path, length=17):
    path,_,_=fixtures.ChunkRolloutTests().setup_rollout(tmp_path,length)
    proprio = np.arange(length*8,dtype=np.float32).reshape(length,8)/20
    with h5py.File(path,'a') as f:
        f['proprio'] = proprio
        f['action_cartesian'][:] = np.arange(length*8).reshape(length,8)/100
    policy=old.make_policy()
    policy.world_model=WorldModel().eval()
    policy.diffusion_policy.obs_horizon=8
    policy.diffusion_policy.pred_horizon=16
    policy.diffusion_policy.action_horizon=policy.action_horizon=8
    samples=[]
    def sample(info,num_samples,denormalize):
        assert denormalize
        sampled=torch.rand(1,num_samples,16,8)
        samples.append(sampled[0].numpy().copy())
        return sampled
    policy.diffusion_policy.sample_action_sequences=Mock(side_effect=sample)
    def observation(data,row,prop):
        info=old.observation()
        info['pixels']=data['pixels'][row][None,None]
        info['wrist_pixels']=data['wrist_pixels'][row][None,None]
        info['proprio']=prop[None,None]
        info['goal_proprio']=proprio[0][None,None]
        info['goal_wrist_pixels']=info['goal'].copy()
        info['step_idx']=np.array([[data['offline_step_idx'][row]]])
        return info
    return path,policy,observation,samples


@pytest.mark.parametrize('execute',[8,4,1])
def test_distances_alignment_selection_and_encoding_once(tmp_path,capsys,execute):
    path,policy,builder,samples=setup(tmp_path)
    before=torch.get_rng_state(),np.random.get_state()
    result=latent.evaluate_latent_dist(policy,path,7,20,42,tmp_path,{},builder,execute)
    assert torch.equal(before[0],torch.get_rng_state())
    assert all(np.array_equal(a,b) for a,b in zip(before[1],np.random.get_state()))
    n=len(range(0,9,execute))
    assert len(samples)==n
    assert len(policy.world_model.calls)==2*n
    assert len(policy.world_model.encodings)==18 # Goal once + all 17 observations.
    assert result['num_observations']==17 and result['num_rollouts']==n
    output=capsys.readouterr().err
    assert 'Encoding observed latents' in output and '17/17' in output
    assert 'Evaluating latent distances' in output and f'{n}/{n}' in output and '100%' in output
    with h5py.File(Path(result['metadata']['output_dir'])/'latent_distance.h5') as f, h5py.File(path) as source:
        assert f['observed_latent_distance'][0]==pytest.approx(0)
        assert (f['observed_latent_distance'][:]>=0).all()
        starts=np.arange(0,9,execute)
        np.testing.assert_array_equal(f['rollout_start_steps'],100+starts)
        np.testing.assert_array_equal(f['rollout_future_steps'],108+starts)
        np.testing.assert_array_equal(f['selected_candidate_indices'],f['candidate_rewards'][:].argmax(1))
        assert f['candidate_rewards'].shape==(n,20)
        for i,row in enumerate(starts):
            candidates_call,gt_call=policy.world_model.calls[2*i:2*i+2]
            assert candidates_call[1].shape==(1,20,8,8)
            assert gt_call[1].shape==(1,1,8,8)
            np.testing.assert_array_equal(gt_call[1],policy.normalize_action_for_world_model(
                torch.tensor(source['action_cartesian'][row:row+8])[None,None]))
            np.testing.assert_array_equal(candidates_call[1],policy.normalize_action_for_world_model(
                torch.tensor(samples[i][:,7:15])[None]))
            for key in ('pixels','wrist_pixels','proprio','step_idx'):
                np.testing.assert_array_equal(candidates_call[0][key][:,0],gt_call[0][key][:,0])
            expected=gt_call[0]['proprio'][0,0,0]+gt_call[1][0,0].sum(0)
            np.testing.assert_allclose(f['gt_terminal_latents'][i],expected,rtol=1e-6,atol=1e-6)
        z=f['observed_latents'][:]; goal=f['goal_latent'][:]
        np.testing.assert_allclose(f['observed_latent_distance'],np.linalg.norm(z-goal,axis=-1),rtol=1e-6)
        gt=f['gt_terminal_latents'][:]; selected=f['gpc_terminal_latents'][:]
        for key,z_end in [('gt',gt),('gpc',selected)]:
            distance=np.linalg.norm(z_end-goal,axis=-1)
            np.testing.assert_allclose(f[f'{key}_rollout_terminal_distance'],distance,rtol=1e-6)
            np.testing.assert_allclose(f[f'{key}_distance_delta'],distance-f['current_observed_distance'][:],rtol=1e-6,atol=1e-6)
        np.testing.assert_allclose(f['wm_prediction_error'],np.linalg.norm(gt-z[starts+8],axis=-1),rtol=1e-6)
        np.testing.assert_allclose(f['candidate_rewards'][:].max(1),-f['gpc_rollout_terminal_distance'][:]**2,rtol=1e-6)
    assert policy.num_candidates==3
    assert all(p['step']>=100+9 for p in result['metadata']['skipped_steps'])
    for name in ('observed_latent_distance','rollout_latent_distance','latent_distance_delta','wm_prediction_error'):
        assert (Path(result['metadata']['output_dir'])/f'{name}.png').stat().st_size>0


def test_gt_independence_and_existing_gpc_numeric_selection(tmp_path):
    path,policy,builder,samples=setup(tmp_path)
    results=[]
    for run in range(2):
        with patch.object(latent,'plot_latent_dist'):
            result=latent.evaluate_latent_dist(policy,path,7,20,42,tmp_path,{},builder,8)
        with h5py.File(Path(result['metadata']['output_dir'])/'latent_distance.h5') as f:
            results.append((f['candidate_rewards'][:],f['selected_candidate_indices'][:],f['gt_terminal_latents'][:]))
        with h5py.File(path,'a') as f:
            f['action_cartesian'][:]=999
    for i in (0,1): np.testing.assert_array_equal(results[0][i],results[1][i])
    assert not np.array_equal(results[0][2],results[1][2])
    data,steps,_=dp_eval.load_episode(path,7); data['offline_step_idx']=steps
    with h5py.File(path) as f: prop=f['proprio'][0]
    info=builder(data,0,prop)
    with patch.object(policy,'num_candidates',20), patch.object(
            policy.diffusion_policy,'sample_action_sequences',return_value=torch.tensor(samples[0][None])):
        expected=policy.get_action_chunk(info)
    index=results[0][1][0]
    np.testing.assert_array_equal(expected[0],samples[0][index,7:15])


@pytest.mark.parametrize('execute',[0,-1,9,True,1.5])
def test_invalid_interval(tmp_path,execute):
    path,policy,builder,_=setup(tmp_path)
    with pytest.raises(ValueError,match='dp_execute_steps'):
        latent.evaluate_latent_dist(policy,path,7,20,42,tmp_path,{},builder,execute)


def test_short_episode_has_observed_only_and_missing_proprio_errors(tmp_path):
    path,policy,builder,_=setup(tmp_path,length=5)
    result=latent.evaluate_latent_dist(policy,path,7,20,42,tmp_path,{},builder,4)
    assert result['num_observations']==5 and result['num_rollouts']==0
    with h5py.File(Path(result['metadata']['output_dir'])/'latent_distance.h5') as f:
        assert f['candidate_rewards'].shape==(0,20)
        assert f['wm_prediction_error'].shape==(0,)
    with h5py.File(path,'a') as f: del f['proprio']
    with pytest.raises(ValueError,match='measured dataset proprio'):
        latent.evaluate_latent_dist(policy,path,7,20,42,tmp_path,{},builder,4)


def test_hardware_free_latent_dispatch_without_gpc_count():
    source=inspect.getsource(old.test_run_routes_before_hardware_and_uses_offline_transform)
    source=source.replace('test_run_routes_before_hardware_and_uses_offline_transform','check')
    source=source.replace('multi_sample','latent_dist').replace('offline_gpc_eval.run_offline_gpc_latent_dist',
                                                            'offline_latent_dist_eval.run_offline_latent_dist')
    ns=dict(old.__dict__); exec(source,ns); ns['check']()


def test_shape_validation():
    for value in (torch.zeros(1,2,8),torch.zeros(1,1,2,4),torch.full((1,1,8),float('nan'))):
        with pytest.raises(ValueError): latent._latent(value,'embedding')
    with pytest.raises(ValueError): latent._l2(torch.zeros(8),torch.zeros(9))


@pytest.mark.skipif(not all(p.exists() for p in (old.REAL_DP,Path(f'{old.REAL_WM}_object.ckpt'),old.REAL_DATA,old.REAL_STATS)),reason='Production files unavailable')
def test_real_checkpoints_cpu_latent_dist():
    source=inspect.getsource(old.test_production_checkpoints_cpu)
    source=source[source.index('def test_production'):source.index('    dp_result =')]
    source=source.replace('test_production_checkpoints_cpu','check').replace('multi_sample','latent_dist')
    source=source.replace('from offline_gpc_eval import run_offline_gpc_latent_dist',
                          'from offline_latent_dist_eval import run_offline_latent_dist')
    source=source.replace('run_offline_gpc_latent_dist','run_offline_latent_dist')
    source=source.replace('source[key][:5]','source[key][:10]')
    source=source.replace('robot.dry_run_timestep, robot.dry_run_num_candidates, robot.dry_run_seed = 2, 20, 42',
                          'robot.dry_run_timestep, robot.dry_run_num_candidates, robot.dry_run_seed = 2, 20, 42\n    robot.dp_execute_steps = 4')
    source=source.replace('robot.dry_run, robot.dry_run_mode = True,', 'robot.dry_run_episode_idx = 0\n    robot.dry_run, robot.dry_run_mode = True,')
    source += "    assert result['num_observations']==10 and result['num_rollouts']==1\n    print('Real latent artifacts:',result['metadata']['output_dir'])\n"
    ns=dict(old.__dict__); exec(source,ns); ns['check']()


@pytest.mark.parametrize('policy_type',['diffusion','world_model'])
def test_diffusion_latent_dist_rejected_before_loading(policy_type):
    import ast
    from omegaconf import OmegaConf
    tree=ast.parse((fixtures.ROOT/'eval_real_robot.py').read_text())
    run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
    run.decorator_list=[]
    loader=Mock(side_effect=AssertionError('must not load model'))
    ns=dict(DictConfig=object,load_diffusion_policy=loader,img_transform=loader)
    exec(compile(ast.Module(body=[run],type_ignores=[]),'<run>','exec'),ns)
    cfg=OmegaConf.create({'policy_type':policy_type,'eval':{'real_robot':{
        'dry_run':True,'dry_run_mode':'latent_dist'}}})
    with pytest.raises(ValueError,match='latent_dist requires policy_type=gpc'): ns['run'](cfg)
    loader.assert_not_called()


def test_latent_rng_wrapper_restores_loading_failure():
    from omegaconf import OmegaConf
    from offline_gpc_eval import preserve_gpc_offline_rng
    @preserve_gpc_offline_rng
    def fail(cfg):
        torch.rand(10); np.random.rand(10)
        raise RuntimeError('model failed')
    cfg=OmegaConf.create({'policy_type':'gpc','policy':'trained','eval':{'real_robot':{
        'dry_run':True,'dry_run_mode':'latent_dist'}}})
    before=torch.get_rng_state(),np.random.get_state()
    with pytest.raises(RuntimeError): fail(cfg)
    assert torch.equal(before[0],torch.get_rng_state())
    assert all(np.array_equal(a,b) for a,b in zip(before[1],np.random.get_state()))


def test_failure_and_eval_mode_validation(tmp_path):
    path,policy,builder,_=setup(tmp_path)
    policy.world_model.train()
    with pytest.raises(ValueError,match='eval'):
        latent.evaluate_latent_dist(policy,path,7,20,42,tmp_path,{},builder,4)
    policy.world_model.eval()
    policy.reward_fn=lambda output,info:torch.full((1,20),float('nan'))
    before=torch.get_rng_state()
    with pytest.raises(ValueError,match='finite candidate rewards'):
        latent.evaluate_latent_dist(policy,path,7,20,42,tmp_path,{},builder,4)
    assert torch.equal(before,torch.get_rng_state())


def test_episode_boundaries_exclude_foreign_actions_and_observations(tmp_path):
    path,policy,builder,_=setup(tmp_path,length=25)
    with h5py.File(path,'a') as f:
        f['episode_idx'][9:]=8
        f['step_idx'][9:]=np.arange(16)
        f['action_cartesian'][9:]=99999
        f['proprio'][9:]=99999
    with patch.object(latent,'plot_latent_dist'):
        result=latent.evaluate_latent_dist(policy,path,7,20,42,tmp_path,{},builder,4)
    assert result['num_observations']==9 and result['num_rollouts']==1
    with h5py.File(Path(result['metadata']['output_dir'])/'latent_distance.h5') as f:
        np.testing.assert_array_equal(f['rollout_start_steps'],[100])
        np.testing.assert_array_equal(f['rollout_future_steps'],[108])
    assert len(policy.world_model.calls)==2
    assert policy.world_model.calls[1][1].abs().max()<100


def test_existing_evaluators_and_robot_methods_unchanged():
    import ast
    import subprocess
    root=fixtures.ROOT
    rev='ac2a9381c6bc26ed040d120927d8c2727c73636d'
    def previous(name):
        return subprocess.check_output(['git','show',f'{rev}:{name}'],cwd=root,text=True)
    assert previous('offline_dp_eval.py')==(root/'offline_dp_eval.py').read_text()
    def definitions(source):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(source).body
                if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    before=definitions(previous('offline_gpc_eval.py'))
    after=definitions((root/'offline_gpc_eval.py').read_text())
    for name in before:
        if name!='preserve_gpc_offline_rng': assert before[name]==after[name]
    before=definitions(previous('eval_real_robot.py')); after=definitions((root/'eval_real_robot.py').read_text())
    for name in before:
        if name!='run': assert before[name]==after[name]
