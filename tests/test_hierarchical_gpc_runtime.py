"""Phase 2/3 verification uses synthetic files and the production control loop."""
import ast
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import h5py
import numpy as np
from omegaconf import OmegaConf
import torch
from stable_worldmodel.policy import GPCPolicy

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from hierarchical_gpc_policy import HierarchicalGPCPolicy, COST_KEYS
from hierarchical_gpc_runtime import offline_observations, load_hierarchical_gpc
from test_gripper_reinforcement import namespace, Clock, FakeEnv

ROOT = Path(__file__).resolve().parents[1]


class RuntimeTests(unittest.TestCase):
    def test_offline_and_robot_hydra_configs_share_explicit_model_paths(self):
        from hydra import initialize_config_dir, compose
        with initialize_config_dir(config_dir=str(ROOT/'config'),version_base=None):
            offline=compose(config_name='hierarchical_gpc_offline')
        with initialize_config_dir(config_dir=str(ROOT/'config/eval'),version_base=None):
            robot=compose(config_name='flip_mug_hjepa')
        self.assertEqual(robot.hjepa.action_horizon,8)
        self.assertEqual(offline.hjepa.world_model,robot.hjepa.world_model)
        self.assertIsNone(robot.hjepa.diffusion_checkpoint)
        self.assertIsNone(robot.hjepa.world_model.lower_checkpoint)
        self.assertIsNone(robot.hjepa.world_model.upper_checkpoint)
        self.assertFalse(robot.eval.real_robot.execute)
        self.assertTrue(robot.eval.real_robot.dry_run)

    def test_legacy_eval_uses_shared_loader_with_same_signature(self):
        import inspect
        from diffusion_checkpoint import load_diffusion_policy
        tree=ast.parse((ROOT/'eval_real_robot.py').read_text())
        imports=[node for node in tree.body if isinstance(node,ast.ImportFrom) and node.module=='diffusion_checkpoint']
        self.assertEqual([alias.name for node in imports for alias in node.names],['load_diffusion_policy'])
        signature=inspect.signature(load_diffusion_policy)
        self.assertEqual(list(signature.parameters),['checkpoint_path','image_transform','device'])
        self.assertEqual(signature.parameters['device'].default,'cuda')

    def test_offline_history_boundary_and_missing_steps(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'data.h5'
            with h5py.File(path,'w') as f:
                f['pixels']=np.broadcast_to(np.arange(12,dtype=np.uint8)[:,None,None,None],(12,6,7,3))
                f['wrist_pixels']=f['pixels'][:]+20
                f['proprio']=np.zeros((12,8),np.float32)
                f['episode_idx']=np.repeat([0,1],6)
                f['step_idx']=np.tile(np.arange(6),2)
            info,history=offline_observations(path,4,10,3)
            np.testing.assert_array_equal(history['pixels'][0,:,0,0,0],[2,3,4])
            self.assertEqual(info['pixels'][0,0,0,0,0],4)
            self.assertEqual(info['goal'][0,0,0,0,0],10)
            with self.assertRaisesRegex(ValueError,'episode'):
                offline_observations(path,6,10,3)
            with h5py.File(path,'a') as f:
                f['step_idx'][3]=9
            with self.assertRaisesRegex(ValueError,'step_idx'):
                offline_observations(path,4,10,3)
            with h5py.File(path,'a') as f:
                del f['step_idx']
            offline_observations(path,4,10,3)
            with self.assertRaises(ValueError):
                offline_observations(path,0,10,3)

    def test_checkpoint_paths_and_statistics_fail_without_recomputation(self):
        cfg=OmegaConf.load(ROOT/'config/eval/hjepa/gpc.yaml')
        with self.assertRaisesRegex(ValueError,'diffusion_checkpoint'):
            load_hierarchical_gpc(cfg)
        cfg.diffusion_checkpoint='/tmp/nonexistent-hjepa-dp.ckpt'
        with self.assertRaises(FileNotFoundError):
            load_hierarchical_gpc(cfg)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'stub.ckpt'
            path.touch()
            cfg.diffusion_checkpoint=str(path)
            for key in ('lower_checkpoint','upper_checkpoint','lower_stats_path'):
                cfg.world_model[key]=str(path)
            invalid={'config':{'img_size':16},'action_key':'action_cartesian','action_dim':8,
                     'action_mean':torch.zeros(8),'action_std':torch.zeros(8),
                     'obs_horizon':2,'pred_horizon':12,'action_horizon':8}
            with patch('hierarchical_gpc_runtime.torch.load',return_value=invalid), \
                 patch('hierarchical_gpc_runtime.load_diffusion_policy') as loader:
                with self.assertRaisesRegex(ValueError,'action_std'):
                    load_hierarchical_gpc(cfg)
                loader.assert_not_called()

    def test_factory_uses_dp_training_transform_and_explicit_layer_paths(self):
        cfg=OmegaConf.load(ROOT/'config/eval/hjepa/gpc.yaml')
        cfg.device='cpu'
        cfg.num_candidates=3
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'model.ckpt'
            path.touch()
            cfg.diffusion_checkpoint=str(path)
            for key in ('lower_checkpoint','upper_checkpoint','lower_stats_path'):
                cfg.world_model[key]=str(path)
            saved=dict(config=dict(img_size=16),action_key='action_cartesian',action_dim=8,
                       action_mean=torch.zeros(8),action_std=torch.ones(8),
                       obs_horizon=2,pred_horizon=12,action_horizon=8)
            dp=SimpleNamespace(obs_horizon=2,pred_horizon=12,action_horizon=8,action_dim=8,device=torch.device('cpu'))
            hwm=SimpleNamespace(upper_world_model=SimpleNamespace(horizon=4),_device=lambda:torch.device('cpu'))
            with patch('hierarchical_gpc_runtime.torch.load',return_value=saved), \
                 patch('hierarchical_gpc_runtime.get_eval_img_preprocessor',return_value='training-transform') as transform, \
                 patch('hierarchical_gpc_runtime.load_diffusion_policy',return_value=dp) as loader, \
                 patch('hierarchical_gpc_runtime.HierarchicalWorldModel.from_checkpoints',return_value=hwm) as wm_loader:
                policy=load_hierarchical_gpc(cfg)
            self.assertIsInstance(policy,HierarchicalGPCPolicy)
            transform.assert_called_once_with(16)
            loader.assert_called_once_with(path,'training-transform',device='cpu')
            wm_loader.assert_called_once_with(**dict(cfg.world_model),device='cpu')

    def test_hierarchical_run_branch_skips_legacy_stats_generation(self):
        node=next(n for n in ast.parse((ROOT/'eval_real_robot.py').read_text()).body
                  if isinstance(n,ast.FunctionDef) and n.name=='run')
        node.decorator_list=[]
        process={'proprio':object(),'goal_proprio':object()}
        policy=object()
        run_robot=Mock(side_effect=AssertionError('execute=false must not open robot'))
        ns=dict(DictConfig=object,Path=Path,time=SimpleNamespace(strftime=lambda x:'mock'),
                load_normalization_process=Mock(return_value=(process,'action_cartesian')),run_xarm_task=run_robot)
        exec(compile(ast.Module(body=[node],type_ignores=[]),'eval_real_robot.py','exec'),ns)
        cfg=OmegaConf.create(dict(policy_type='hierarchical_gpc',plan_config=dict(action_space='cartesian'),
                                 hjepa=dict(world_model=dict(lower_stats_path='/explicit/lower.npz')),
                                 eval=dict(probing=dict(exe_probe=False),real_robot=dict(execute=False,output_dir='/tmp/hjepa_mock'))))
        with patch('hierarchical_gpc_runtime.load_hierarchical_gpc',return_value=policy) as factory:
            self.assertIs(ns['run'](cfg),policy)
            factory.assert_called_once_with(cfg.hjepa)
        run_robot.assert_not_called()

    def test_actual_hierarchical_policy_with_existing_mock_robot_loop(self):
        commands=iter([.9,.1,.8])
        sampled=[]
        def sample(info,num_samples,denormalize):
            sampled.append(info)
            self.assertTrue(denormalize)
            self.assertEqual(info['pixels'].shape[1],2)
            out=torch.zeros(1,num_samples,10,8)
            out[...,0]=.7
            out[...,2]=.2
            out[...,6]=1
            out[...,7]=next(commands)
            return out
        dp=SimpleNamespace(obs_horizon=2,pred_horizon=10,action_horizon=8,action_dim=8,
                           device=torch.device('cpu'),sample_action_sequences=Mock(side_effect=sample))
        costs={key:torch.tensor([[3.,1.,2.]]) for key in COST_KEYS}
        hwm=SimpleNamespace(upper_world_model=SimpleNamespace(horizon=4),get_cost=Mock(return_value=costs))
        policy=HierarchicalGPCPolicy(dp,hwm,num_candidates=3)
        cfg=OmegaConf.create(dict(plan_config=dict(action_space='cartesian'),eval=dict(real_robot=dict(
            max_steps=6,control_hz=10,non_interactive=True,
            gripper=dict(open_position=84.,closed_position=0.,max_delta=1.),
            gripper_reinforcement=dict(enabled=True,threshold=.2,target_opening_mm=20.,duration_sec=.25)))))
        clock=Clock()
        ns=namespace(clock)
        ns['json']=json
        ns['swm'].policy.GPCPolicy=GPCPolicy
        env=FakeEnv()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            ns['_run_xarm_task_with_env'](cfg,policy,{'proprio':None,'goal_proprio':None},root,env)
            with h5py.File(root/'rollout.h5') as f:
                np.testing.assert_array_equal(f['gripper_reinforcement_active'][:],[1,1,1,0,1,1])
                self.assertEqual(f['commanded_action'].shape,(6,8))
                np.testing.assert_allclose(f['gripper_command'][:],f['commanded_action'][:,7])
            records=[json.loads(line) for line in (root/'hierarchical_costs.jsonl').read_text().splitlines()]
            self.assertEqual([r['inference_performed'] for r in records],[True,False,False,True,True,False])
            self.assertIsNone(records[1]['selected_costs'])
            self.assertEqual(records[0]['best_idx'],[1])
            self.assertEqual(records[0]['selected_costs']['total_cost'],[1.])
            np.testing.assert_allclose(records[0]['executed_action'],env.sent[0])
        self.assertEqual(dp.sample_action_sequences.call_count,3)
        self.assertEqual(hwm.get_cost.call_count,3)
        np.testing.assert_array_equal(sampled[1]['pixels'][0,:,0,0,0],[2,3])
        np.testing.assert_allclose(np.asarray(env.sent)[:3,:7],np.tile(env.sent[0][:7],(3,1)))
        self.assertEqual(len(env.sent),6)  # One 8D action per control period.
        env.close.assert_called_once()


if __name__=='__main__':
    unittest.main()
