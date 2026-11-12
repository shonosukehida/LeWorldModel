"""Hardware-free regression against the pre-change DP and actual GPC API."""
import json
from pathlib import Path
import subprocess
import tempfile
import types
from unittest.mock import Mock, patch

import h5py
import numpy as np
import pytest
import torch

import offline_dp_eval as dp_eval
from offline_gpc_eval import (evaluate_gpc_multi_sample, isolated_rng, plot_gpc,
                              preserve_gpc_offline_rng, select_candidates)
from stable_worldmodel.policy import GPCPolicy
from stable_worldmodel.reward import latent_goal_reward, LatentGoalReward
import test_offline_dp_eval as dp_tests
ROOT = dp_tests.ROOT


class WorldModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.calls = []

    def encode(self, info):
        return {'emb': torch.zeros(1, 1, 8)}

    def rollout(self, info, actions):
        self.calls.append((info, actions.detach().clone()))
        return {'predicted_emb': actions}


def make_policy():
    dp = dp_tests.MultiSampleTests().policy()
    wm = WorldModel()
    # Reuse the actual action processor and actual rollout / goal / reward methods.
    policy = GPCPolicy(dp, wm, latent_goal_reward, num_candidates=3,
                       process=dp.process, transform={})
    return policy


def observation():
    return {'pixels': np.zeros((1, 1, 4, 4, 3), np.uint8),
            'wrist_pixels': np.zeros((1, 1, 4, 4, 3), np.uint8),
            'proprio': np.ones((1, 1, 8), np.float32),
            'goal': np.zeros((1, 1, 4, 4, 3), np.uint8),
            'goal_proprio': np.zeros((1, 1, 8), np.float32)}


def test_existing_gpc_api_same_candidates_observation_goal_and_slice():
    policy = make_policy()
    candidates = dp_eval.sample_candidates(policy.diffusion_policy, observation(), 3, 42)
    original_sampler = policy.diffusion_policy.sample_action_sequences
    index, rewards, interval = select_candidates(policy, candidates, observation())
    assert rewards.shape == (1, 3)
    assert index == rewards.argmax(axis=1)[0]
    assert interval == (3, 5)
    assert policy.diffusion_policy.sample_action_sequences is original_sampler
    np.testing.assert_array_equal(policy.world_model.calls[-1][1].numpy(),
        policy.normalize_action_for_world_model(torch.from_numpy(candidates[None, :, 3:5])).numpy())
    with patch.object(policy.diffusion_policy, 'sample_action_sequences',
                      return_value=torch.from_numpy(candidates[None])) as sampler:
        expected = policy.get_action_chunk(observation(), dp_info_dict={'history': 'same'})
        sampler.assert_called_once()
        np.testing.assert_array_equal(expected[0], candidates[index, 3:5])
        np.testing.assert_array_equal(policy.get_action(observation())[0], expected[0, 0])
    assert policy.world_model.calls[-1][1].shape == (1, 3, 2, 8)


def test_artifacts_one_sampling_gt_independence_measured_proprio_and_rng():
    with tempfile.TemporaryDirectory() as tmp:
        dataset = Path(tmp) / 'data.h5'
        dp_tests.OfflineTests().dataset(dataset)
        with h5py.File(dataset, 'a') as f:
            f['proprio'] = np.full((5, 8), 17., np.float32)
        policy = make_policy()
        sampler = Mock(wraps=policy.diffusion_policy.sample_action_sequences)
        policy.diffusion_policy.sample_action_sequences = sampler
        builder = Mock(return_value=observation())
        # Include NumPy randomness in reward to verify isolation even for callbacks.
        reward = policy.reward_fn
        policy.reward_fn = lambda *a: (np.random.rand(), reward(*a))[1]
        before_torch, before_numpy = torch.get_rng_state(), np.random.get_state()
        result = evaluate_gpc_multi_sample(policy, dataset, 7, 1, 3, 42,
                                           Path(tmp)/'offline_gpc', {}, builder)
        assert torch.equal(before_torch, torch.get_rng_state())
        assert all(np.array_equal(a, b) for a, b in zip(before_numpy, np.random.get_state()))
        sampler.assert_called_once()
        np.testing.assert_array_equal(builder.call_args.args[2], np.full(8, 17.))
        out = Path(result['metadata']['output_dir'])
        with h5py.File(out/'multi_sample_rollout.h5') as f:
            index = f['selected_candidate_index'][()]
            np.testing.assert_array_equal(f['selected_candidate_actions'][:], f['predicted_actions'][index])
            assert f['candidate_rewards'].shape == (1, 3)
            assert index == f['candidate_rewards'][:].argmax(axis=1)[0]
            np.testing.assert_array_equal(f['wm_evaluation_interval'], [3, 5])
            saved_rewards = f['candidate_rewards'][:]
        assert json.loads((out/'multi_sample_metrics.json').read_text())['selected_candidate_index'] == index
        with h5py.File(dataset, 'a') as f:
            f['action_cartesian'][:, 0] = 12345
        changed = evaluate_gpc_multi_sample(policy, dataset, 7, 1, 3, 42,
                                            Path(tmp)/'offline_gpc', {}, builder)
        assert changed['selected_candidate_index'] == index
        np.testing.assert_array_equal(changed['candidate_rewards'], saved_rewards)
        with h5py.File(dataset, 'a') as f:
            del f['proprio']
        with pytest.raises(ValueError, match='measured dataset proprio'):
            evaluate_gpc_multi_sample(policy, dataset, 7, 1, 3, 42, tmp, {}, builder)


def test_diffusion_exact_prechange_artifacts_and_png():
    # Pin the actual pre-implementation revision, rather than a moving HEAD.
    source = subprocess.check_output(['git', 'show',
        '351dbd30365a14fe8fe1219752a3db0db9d4257c:offline_dp_eval.py'], cwd=ROOT, text=True)
    baseline = types.ModuleType('baseline_offline_dp')
    exec(compile(source, '<baseline>', 'exec'), baseline.__dict__)
    # Only the explicitly authorized progress wrapper may differ from baseline.
    current = (ROOT/'offline_dp_eval.py').read_text()
    current = current.replace('    from tqdm.auto import tqdm\n', '')
    current = current.replace(
        "enumerate(tqdm(rows, total=len(rows), desc='DP chunk rollout', unit='chunk'))",
        'enumerate(rows)')
    assert source == current
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp)/'data.h5'
        dp_tests.OfflineTests().dataset(path)
        results = [module.evaluate_multi_sample(dp_tests.MultiSampleTests().policy(), path, 7, 1,
                    3, 42, Path(tmp)/str(i), {'checkpoint_path': 'same.ckpt'})
                   for i, module in enumerate((baseline, dp_eval))]
        outputs = [Path(r['metadata']['output_dir']) for r in results]
        for result in results:
            for key in ('output_dir', 'run_id'):
                del result['metadata'][key]
        assert results[0] == results[1]
        with h5py.File(outputs[0]/'multi_sample_rollout.h5') as a, h5py.File(outputs[1]/'multi_sample_rollout.h5') as b:
            assert set(a) == set(b)
            for key in a:
                np.testing.assert_array_equal(a[key][:] if a[key].shape else a[key][()],
                                              b[key][:] if b[key].shape else b[key][()])
            metadata = [json.loads(f.attrs['metadata']) for f in (a,b)]
            for m in metadata:
                for key in ('output_dir','run_id'): del m[key]
            assert metadata[0] == metadata[1]
        docs = [json.loads((out/'multi_sample_metrics.json').read_text()) for out in outputs]
        for doc in docs:
            for key in ('output_dir','run_id'): del doc['metadata'][key]
        assert docs[0] == docs[1]
        for name in ('position', 'orientation', 'gripper'):
            assert (outputs[0]/f'multi_sample_{name}.png').read_bytes() == (outputs[1]/f'multi_sample_{name}.png').read_bytes()


def test_plot_all_components_and_execution_start_meaning():
    import matplotlib.pyplot as plt
    figures = []
    subplots = plt.subplots
    def capture(*args, **kwargs):
        fig, axes = subplots(*args, **kwargs)
        figures.append(axes)
        return fig, axes
    with tempfile.TemporaryDirectory() as tmp, patch.object(plt, 'subplots', side_effect=capture):
        candidates = np.arange(3*6*8).reshape(3, 6, 8)
        plot_gpc(candidates, np.zeros((6,8)), 1, (3,5), tmp, 7, 1)
    assert [len(a) for a in figures] == [3,4,1]
    for axes in figures:
        for ax in axes[:,0]:
            assert [line.get_color() for line in ax.lines] == ['gray']*3 + ['blue','red','black']
            assert ax.lines[-2].get_linewidth() > ax.lines[0].get_linewidth()
            assert ax.lines[-2].get_zorder() > ax.lines[0].get_zorder()
            assert list(ax.lines[-1].get_xdata()) == [3,3]
            assert len(ax.patches) == 1


def test_rng_restoration_on_failure_including_loading_scope():
    from omegaconf import OmegaConf
    cfg = OmegaConf.create({'policy_type':'gpc','policy':'trained','eval':{'real_robot':{
        'dry_run':True,'dry_run_mode':'multi_sample'}}})
    @preserve_gpc_offline_rng
    def fail(cfg):
        np.random.rand(9)
        torch.rand(9)
        raise RuntimeError('failed loading')
    before_torch, before_numpy = torch.get_rng_state(), np.random.get_state()
    with pytest.raises(RuntimeError): fail(cfg)
    assert torch.equal(before_torch, torch.get_rng_state())
    assert all(np.array_equal(a,b) for a,b in zip(before_numpy,np.random.get_state()))


def test_real_jepa_rollout_accepts_tensor_observations():
    from jepa import JEPA
    from types import SimpleNamespace
    class Encoder(torch.nn.Module):
        def forward(self, pixels, **kwargs):
            return SimpleNamespace(last_hidden_state=pixels.mean((-2,-1))[:,None])
    class Predictor(torch.nn.Module):
        def forward(self, emb, act):
            return emb + act
    wm = JEPA(Encoder(), Encoder(), Predictor(), torch.nn.Linear(8,8),
              torch.nn.Linear(8,2))
    policy = make_policy()
    policy.world_model = wm.eval()
    policy.transform = {k: lambda x: x.float()/255 for k in
                        ('pixels','wrist_pixels','goal','goal_wrist_pixels')}
    info = observation()
    info['goal_wrist_pixels'] = info['wrist_pixels'].copy()
    candidates = dp_eval.sample_candidates(policy.diffusion_policy, info, 3, 42)
    index, rewards, interval = select_candidates(policy, candidates, info)
    assert rewards.shape == (1,3)
    with patch.object(policy.diffusion_policy, 'sample_action_sequences',
                      return_value=torch.from_numpy(candidates[None])):
        expected = policy.get_action_chunk(info, dp_info_dict=info)
    np.testing.assert_array_equal(expected[0], candidates[index, interval[0]:interval[1]])


def test_run_routes_before_hardware_and_uses_offline_transform():
    import ast
    from types import SimpleNamespace
    from omegaconf import OmegaConf
    from offline_gpc_eval import preserve_gpc_offline_rng
    tree = ast.parse((ROOT/'eval_real_robot.py').read_text())
    run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
    run.decorator_list = []
    model = Mock()
    model.to.return_value = model
    model.eval.return_value = model
    loader = Mock(return_value=Mock())
    gpc = Mock(return_value=Mock())
    hardware = Mock(side_effect=AssertionError('hardware initialized'))
    ns = dict(DictConfig=object, Path=Path, OmegaConf=OmegaConf, LatentGoalReward=LatentGoalReward,
        swm=SimpleNamespace(data=SimpleNamespace(utils=SimpleNamespace(get_cache_dir=lambda: '/tmp')),
            policy=SimpleNamespace(AutoCostModel=Mock(return_value=model), GPCPolicy=gpc)),
        img_transform=lambda cfg:'offline_transform', dp_img_transform=lambda cfg:'robot_transform',
        load_normalization_process=lambda p: ({k:object() for k in ('action_cartesian','proprio','goal_proprio')},'action_cartesian'),
        load_diffusion_policy=loader, latent_goal_reward=latent_goal_reward,
        _load_or_capture_goal=Mock(), _policy_observation=Mock(),
        XArmInferenceEnv=hardware, run_xarm_task=hardware)
    exec(compile(ast.Module(body=[run],type_ignores=[]),'<run>', 'exec'), ns)
    with tempfile.NamedTemporaryFile() as stats:
        cfg = OmegaConf.create(dict(policy_type='gpc',policy='weights', cache_dir=None,
            eval=dict(dataset_name='missing',normalization_stats_path=stats.name,
                      probing=dict(use_random_encoder=False), real_robot=dict(
                        dry_run=True,dry_run_mode='multi_sample',execute=True)),
            gpc=dict(diffusion_checkpoint='dp.ckpt')))
        with patch('offline_gpc_eval.run_offline_gpc_multi_sample') as offline:
            ns['run'](cfg)
        offline.assert_called_once()
        hardware.assert_not_called()
        loader.assert_called_once_with(checkpoint_path='dp.ckpt',image_transform='offline_transform',device='cuda')
        assert gpc.call_args.kwargs['num_candidates'] == 20


def test_existing_apis_and_robot_implementation_unchanged():
    import ast
    before = subprocess.check_output(['git','show',
        '351dbd30365a14fe8fe1219752a3db0db9d4257c:eval_real_robot.py'],cwd=ROOT,text=True)
    def definitions(source):
        return {n.name: ast.dump(n,include_attributes=False) for n in ast.parse(source).body
                if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name not in {'run', 'run_xarm_task'}}  # Post-save hook has dedicated tests.
    from test_imle_multiview_regression import without_imle_execution_guard
    assert definitions(before) == definitions(without_imle_execution_guard((ROOT/'eval_real_robot.py').read_text()))
    wm_root = ROOT.parent/'stable-worldmodel'
    previous = subprocess.check_output(['git','show',
        '7e1b7b03f707a1b5648c818b84808904f5db64eb:stable_worldmodel/policy.py'],cwd=wm_root,text=True)
    previous = previous.replace(
        'wm_info[key] = (value[:, None].expand(B, K, *value.shape[1:],))',
        'wm_info[key] = value[:, None].expand(B, K, *value.shape[1:])')
    def legacy_policy_definitions(source):
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name == 'GPCPolicy':
                node.body = [n for n in node.body if getattr(n, 'name', None) != '__init__']
        return ast.dump(tree, include_attributes=False)
    assert legacy_policy_definitions(previous) == legacy_policy_definitions(
        (wm_root/'stable_worldmodel/policy.py').read_text())


def test_twenty_candidates_and_argmax_ties():
    policy = make_policy()
    candidates = dp_eval.sample_candidates(policy.diffusion_policy, observation(), 20, 42)
    policy.reward_fn = lambda output, info: torch.zeros(1,20)
    index, rewards, interval = select_candidates(policy,candidates,observation())
    assert index == 0
    assert rewards.shape == (1,20)
    assert policy.world_model.calls[-1][1].shape == (1,20,2,8)


REAL_DP = Path('/home/shonosukehida/.stable_worldmodel/checkpoints/flip_mug/policy/diffusion_policy/ep200_tm300_multiview_demo/seed_41/20261021_135400/latest.ckpt')
REAL_WM = Path('/home/shonosukehida/.stable_worldmodel/checkpoints/flip_mug/ep100_tm300_multiview_play_normal_brightness/lewm')
REAL_DATA = Path('/home/shonosukehida/.stable_worldmodel/datasets/flip_mug/ep200_tm300_multiview_demo/push.h5')
REAL_STATS = REAL_DATA.parent/'process_stats.npz'


@pytest.mark.skipif(not all(p.exists() for p in (REAL_DP,Path(f'{REAL_WM}_object.ckpt'),REAL_DATA,REAL_STATS)),
                    reason='Local production DP/WM/data/statistics unavailable')
def test_production_checkpoints_cpu():
    """A small real-data slice, real DP/JEPA weights; CPU path, twenty candidates."""
    import ast
    import cv2
    import stable_worldmodel as swm
    from omegaconf import OmegaConf
    from normalization_stats import load_normalization_process
    from test_dp_encoder_selection import loader
    from utils import get_eval_img_preprocessor
    from offline_gpc_eval import run_offline_gpc_multi_sample
    torch.set_num_threads(2)
    out = Path(tempfile.mkdtemp(prefix='gpc_real_checkpoint_',dir='/tmp'))
    dataset = out/'real_data_slice.h5'
    with h5py.File(REAL_DATA) as source, h5py.File(dataset,'w') as target:
        for key in ('pixels','wrist_pixels','proprio','action_cartesian','episode_idx','step_idx'):
            target[key] = source[key][:5]
    cfg = OmegaConf.load(ROOT/'config/eval/flip_mug.yaml')
    cfg.policy_type, cfg.policy = 'gpc', str(REAL_WM)
    cfg.gpc.diffusion_checkpoint = str(REAL_DP)
    robot = cfg.eval.real_robot
    robot.dry_run, robot.dry_run_mode = True, 'multi_sample'
    robot.dry_run_dataset_path, robot.output_dir = str(dataset), str(out)
    robot.dry_run_timestep, robot.dry_run_num_candidates, robot.dry_run_seed = 2, 20, 42
    robot.goal_image_path = str(ROOT/robot.goal_image_path)
    robot.goal_wrist_image_path = str(ROOT/robot.goal_wrist_image_path)
    process, _ = load_normalization_process(REAL_STATS)
    transform = get_eval_img_preprocessor(cfg.eval.img_size)
    dp = loader()(REAL_DP,transform,'cpu')
    wm = swm.policy.AutoCostModel(str(REAL_WM)).eval()
    wm.requires_grad_(False)
    wm.interpolate_pos_encoding = True
    policy = GPCPolicy(dp,wm,latent_goal_reward,num_candidates=20,process=process,
                       transform={k:transform for k in ('pixels','wrist_pixels','goal','goal_wrist_pixels')})
    nodes = [n for n in ast.parse((ROOT/'eval_real_robot.py').read_text()).body
             if isinstance(n,ast.FunctionDef) and n.name in ('_load_or_capture_goal','_policy_observation')]
    ns = dict(np=np,cv2=cv2)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),'<robot observation helpers>','exec'),ns)
    result = run_offline_gpc_multi_sample(cfg,policy,ns['_load_or_capture_goal'],ns['_policy_observation'])
    dp_result = dp_eval.evaluate_multi_sample(dp,dataset,0,2,20,42,out/'offline_dp',{})
    with h5py.File(Path(result['metadata']['output_dir'])/'multi_sample_rollout.h5') as gpc_file, h5py.File(Path(dp_result['metadata']['output_dir'])/'multi_sample_rollout.h5') as dp_file:
        for key in dp_file:
            np.testing.assert_array_equal(gpc_file[key][()],dp_file[key][()])
        candidates = gpc_file['predicted_actions'][:]
        index = gpc_file['selected_candidate_index'][()]
    data, _, _ = dp_eval.load_episode(dataset,0)
    with h5py.File(dataset) as f:
        proprio = f['proprio'][2]
    goal,wrist,goal_prop = ns['_load_or_capture_goal'](None,robot)
    info = ns['_policy_observation'](data['pixels'][2],data['wrist_pixels'][2],goal,wrist,
                                    proprio[:7],proprio[7],goal_prop,2,process)
    with patch.object(dp,'sample_action_sequences',return_value=torch.from_numpy(candidates[None])):
        selected = policy.get_action_chunk(info,dp_info_dict=dp_eval.build_observation_history(data,2,dp.obs_horizon))
    start = dp.obs_horizon - 1
    np.testing.assert_array_equal(selected[0],candidates[index,start:start+policy.action_horizon])
    assert result['metadata']['world_model_checkpoint']['path'] == str(Path(f'{REAL_WM}_object.ckpt').resolve())
    print('Real DP/WM CPU artifacts:',out)


@pytest.mark.parametrize('name',[
    'test_one_step_frozen_loop_equivalence',
    'test_multi_step_refill_history_and_discard',
    'test_real_cartesian_safety_in_loop'])
def test_existing_robot_loop_with_current_sdk_state_fixture(name):
    # Existing FakeEnv predates SDK joint-state recording. Supply that measured
    # state only in this test; neither robot code nor old test files are changed.
    from test_dp_action_chunks import LoopTests
    from test_gripper_reinforcement import FakeEnv
    with patch.object(FakeEnv,'_last_sdk_qpos',np.zeros(7),create=True):
        getattr(LoopTests(),name)()


def test_gpc_reward_validation_and_clipped_evaluation_interval():
    policy = make_policy()
    candidates = dp_eval.sample_candidates(policy.diffusion_policy,observation(),3,42)
    policy.action_horizon = 100
    index,rewards,interval = select_candidates(policy,candidates,observation())
    assert interval == (3,6)
    assert policy.world_model.calls[-1][1].shape == (1,3,3,8)
    for value,error in ((torch.zeros(3),ValueError),(np.zeros((1,3)),TypeError),
                        (torch.full((1,3),float('nan')),ValueError)):
        policy.reward_fn = lambda *args,value=value: value
        with pytest.raises(error): select_candidates(policy,candidates,observation())


def test_rng_wrapper_preserves_non_target_dispatch_without_new_settings():
    from omegaconf import OmegaConf
    fn = Mock(return_value='unchanged')
    wrapped = preserve_gpc_offline_rng(fn)
    for cfg in (OmegaConf.create({'policy_type':'diffusion'}),
                OmegaConf.create({'policy_type':'world_model'}),
                OmegaConf.create({'policy_type':'gpc','eval':{}}),
                OmegaConf.create({'policy_type':'gpc','eval':{'real_robot':{'dry_run':True}}})):
        assert wrapped(cfg) == 'unchanged'
    assert fn.call_count == 4
