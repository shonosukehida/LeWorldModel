"""Frozen-source comparisons; no robot or production checkpoint required."""
import ast
import contextlib
import copy
import importlib.util
import io
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace
from scipy.spatial.transform import Rotation
from typing import Any
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from sklearn.preprocessing import StandardScaler
from torchvision import tv_tensors
from omegaconf import OmegaConf
from test_gripper_reinforcement import ROOT, Clock, Diffusion, GPC, FakeEnv, namespace

POLICY = ROOT.parent/'stable-worldmodel/stable_worldmodel/policy.py'
BASE = ROOT/'tests/baseline_dp_chunks'


def classes(path):
    nodes = [n for n in ast.parse(path.read_text()).body
             if isinstance(n, ast.ClassDef) and n.name in {'BasePolicy','DiffusionPolicy','GPCPolicy'}]
    ns = dict(torch=torch, np=np, Any=Any, tv_tensors=tv_tensors)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
    return ns


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class NumericalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.old, cls.new = classes(BASE/'policy.py'), classes(POLICY)
        directory = POLICY.parent/'diffusion'
        cls.encoder = module('chunk_encoder', directory/'obs_encoder.py').ResNet18ObsEncoder
        cls.unet = module('chunk_unet', directory/'conditional_unet1d.py').ConditionalUnet1D

    def dp(self, versions):
        torch.manual_seed(41)
        encoder = self.encoder().eval()
        model = self.unet(8, global_cond_dim=1024, down_dims=(16,32),
                          diffusion_step_embed_dim=16, n_groups=8).eval()
        # Serialize and reload identical weights; this is a synthetic checkpoint,
        # explicitly not the unavailable trained robot checkpoint.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'weights.pt'
            torch.save({'model':model.state_dict(), 'encoder':encoder.state_dict()}, path)
            weights = torch.load(path, weights_only=True)
            model.load_state_dict(weights['model']); encoder.load_state_dict(weights['encoder'])
        scaler = StandardScaler().fit(np.arange(80).reshape(10,8)/20)
        return versions['DiffusionPolicy'](model, encoder, DDPMScheduler(num_train_timesteps=8),
            pred_horizon=8, obs_horizon=2, action_horizon=3, action_dim=8,
            num_inference_steps=4, process={'action_cartesian':scaler})

    def test_dp_real_resnet_unet_ddpm_rng_and_normalization(self):
        old, new = self.dp(self.old), self.dp(self.new)
        obs = {'pixels':torch.linspace(0,1,2*3*32*32).reshape(1,2,3,32,32)}
        for key in ('action_cartesian','action','action_joint', None):
            scaler = old.process.get('action_cartesian') or StandardScaler().fit(np.arange(80).reshape(10,8)/20)
            old.process = new.process = {key:scaler} if key else {}
            for seed in (0,42,99):
                torch.manual_seed(seed); rng = torch.get_rng_state()
                a = old.get_action(obs); after = torch.get_rng_state()
                torch.set_rng_state(rng); chunk = new.get_action_chunk(obs)
                np.testing.assert_array_equal(a, chunk[:,0])
                self.assertTrue(torch.equal(after, torch.get_rng_state()))
                torch.set_rng_state(rng)
                np.testing.assert_array_equal(a, new.get_action(obs))
                self.assertEqual(chunk.shape, (1,3,8))
        print('DP: 12 comparisons, max abs error=0; RNG states identical; all scaler keys and no scaler')

    def test_gpc_candidates_rollout_rewards_selection(self):
        obs = {'pixels':torch.ones(1,2,3,32,32)}
        traces = []
        policies = []
        for versions in (self.old, self.new):
            trace = {}; traces.append(trace)
            def rollout(info_dict, action_sequences, trace=trace):
                trace['wm_actions'] = action_sequences.clone()
                # Controlled world-model stand-in; actual JEPA checkpoint unavailable.
                trace['prediction'] = action_sequences.cumsum(dim=2)
                return {'prediction':trace['prediction']}
            def reward(output, info, trace=trace):
                trace['rewards'] = -((output['prediction'][:,:,-1]-info['goal'])**2).sum(-1)
                trace['best_idx'] = trace['rewards'].argmax(1)
                return trace['rewards']
            dp = self.dp(versions)
            sample = dp.sample_action_sequences
            def traced_sample(*args, sample=sample, trace=trace, **kwargs):
                value = sample(*args, **kwargs); trace['candidates'] = value.clone(); return value
            dp.sample_action_sequences = traced_sample
            gpc = versions['GPCPolicy'](dp, None, reward, num_candidates=4,
                                        process=copy.deepcopy(dp.process))
            gpc.rollout = rollout
            policies.append(gpc)
        for seed in (0,42,99):
            info = {'goal':torch.arange(8)/8}
            torch.manual_seed(seed); rng = torch.get_rng_state()
            old = policies[0].get_action(info, dp_info_dict=obs); after = torch.get_rng_state()
            torch.set_rng_state(rng)
            chunk = policies[1].get_action_chunk(info, dp_info_dict=obs)
            np.testing.assert_array_equal(old, chunk[:,0])
            self.assertTrue(torch.equal(after,torch.get_rng_state()))
            for key in traces[0]:
                torch.testing.assert_close(traces[0][key], traces[1][key], rtol=0, atol=0)
            torch.set_rng_state(rng)
            np.testing.assert_array_equal(old,policies[1].get_action(info, dp_info_dict=obs))
        print('GPC: 3 comparisons, max abs error=0 for proposals, WM inputs, mock predictions, rewards, indices, actions')


class LoopTests(unittest.TestCase):
    def run_loop(self, policy_class, source=None, steps=1, enabled=True, rearm=.1,
                 commands=(.9,.1,.8,.1,.1,.1), stop=None, fail=False, env_class=FakeEnv):
        clock = Clock(); ns = namespace(clock, source)
        buffers = []
        if '_DPActionBuffer' in ns:
            base = ns['_DPActionBuffer']
            class Tracked(base):
                def __init__(self, *args):
                    super().__init__(*args); buffers.append(self)
            ns['_DPActionBuffer'] = Tracked
        cfg = OmegaConf.create({'plan_config':{'action_space':'cartesian'}, 'eval':{'real_robot':{
            'max_steps':6,'control_hz':10,'non_interactive':True,'dp_execute_steps':steps,
            'gripper':{'open_position':84.,'closed_position':0.,'max_delta':.1},
            'gripper_reinforcement':{'enabled':enabled,'threshold':.2,'rearm_threshold':rearm,
                'target_opening_mm':20.,'duration_sec':.25}}}})
        env = env_class(stop_at=stop,max_delta=.1); policy = policy_class(commands)
        calls = []
        original = policy.get_action
        def record(*args, **kwargs):
            calls.append((env.step, clock.now)); return original(*args, **kwargs)
        policy.get_action = record
        if fail:
            env.execute = lambda *args: (_ for _ in ()).throw(RuntimeError('mock execute failed'))
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            if fail:
                with self.assertRaisesRegex(RuntimeError,'mock execute failed'):
                    ns['_run_xarm_task_with_env'](cfg,policy,{},Path(tmp),env)
                logs = {}
            else:
                ns['_run_xarm_task_with_env'](cfg,policy,{},Path(tmp),env)
                with h5py.File(Path(tmp)/'rollout.h5') as f:
                    logs = {k:f[k][:] for k in f if isinstance(f[k],h5py.Dataset)}
        for buffer in buffers:
            self.assertFalse(buffer.actions)
        env.close.assert_called_once()
        return env,policy,calls,logs

    def test_one_step_frozen_loop_equivalence(self):
        old_source = (BASE/'eval_real_robot.py').read_text()
        for policy in (Diffusion,GPC):
            for enabled,rearm,stop in ((False,.1,None),(True,1.,None),(True,.1,None),(True,.1,2)):
                with self.subTest(policy=policy.__name__,enabled=enabled,rearm=rearm,stop=stop):
                    old = self.run_loop(policy,old_source,enabled=enabled,rearm=rearm,stop=stop)
                    new = self.run_loop(policy,enabled=enabled,rearm=rearm,stop=stop)
                    self.assertEqual(old[2],new[2])
                    np.testing.assert_array_equal(old[0].sent,new[0].sent)
                    for a,b in zip(old[1].observations,new[1].observations):
                        for key in a: np.testing.assert_array_equal(a[key],b[key])
                    for key in old[3]: np.testing.assert_array_equal(old[3][key],new[3][key])
        print('Loop steps=1: 8 scenarios, exact equality of inference step/time, histories, robot mock commands and every HDF5 dataset')

    def test_multi_step_refill_history_and_discard(self):
        for policy in (Diffusion,GPC):
            env,p,calls,logs = self.run_loop(policy,steps=3,enabled=False,commands=(.1,.2))
            self.assertEqual([x[0] for x in calls],[0,3])
            np.testing.assert_allclose(logs['gripper_policy_command'] if policy is Diffusion else np.array(env.sent)[:,7], [.1]*3+[.2]*3)
            np.testing.assert_array_equal(p.observations[1]['pixels'][0,:,0,0,0],[2,3])
            env,p,calls,logs = self.run_loop(policy,steps=3)
            self.assertEqual([x[0] for x in calls],[0,3])
            np.testing.assert_array_equal(logs['gripper_reinforcement_active'],[1,1,1,0,0,0])
            np.testing.assert_allclose(np.array(env.sent)[:3,7],[.1,.2,.3])
            self.run_loop(policy,steps=3,enabled=False,commands=(.1,),stop=1)
            self.run_loop(policy,steps=3,enabled=False,commands=(.1,),fail=True)

    def test_real_cartesian_safety_in_loop(self):
        old_source = (BASE/'eval_real_robot.py').read_text()
        methods = []
        for source in (old_source, (ROOT/'eval_real_robot.py').read_text()):
            tree = ast.parse(source)
            cls = next(n for n in tree.body if isinstance(n,ast.ClassDef)
                       and any(isinstance(m,ast.FunctionDef) and m.name == 'clip_cartesian_action' for m in n.body))
            node = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name == 'clip_cartesian_action')
            ns = dict(np=np, Rotation=Rotation)
            exec(compile(ast.Module(body=[node],type_ignores=[]),'safety','exec'),ns)
            methods.append(ns['clip_cartesian_action'])
        def environment(clip):
            class SafetyEnv(FakeEnv):
                def execute(self,action,space):
                    self.cfg = SimpleNamespace(max_cartesian_delta_m=.01,
                        workspace_bounds_m=[[.2,.6],[-.4,.4],[.1,.5]],
                        max_orientation_delta_rad=.05, gripper=SimpleNamespace(max_delta=.1))
                    ee = np.array([.5+self.step*.001,0,.3,0,0,0,1])
                    command = clip(self,action,current_ee=ee)
                    self._last_gripper = command[7]
                    self.sent.append(command)
                    return command
            return SafetyEnv
        for policy in (Diffusion,GPC):
            old = self.run_loop(policy,old_source,enabled=False,env_class=environment(methods[0]))
            new = self.run_loop(policy,enabled=False,env_class=environment(methods[1]))
            np.testing.assert_array_equal(old[0].sent,new[0].sent)
            np.testing.assert_allclose(np.array(new[0].sent)[0], [.51,.01,.29,0,0,0,1,.1],atol=1e-7)
        print('Real Cartesian safety: old/new commands identical; first command=[.51,.01,.29,0,0,0,1,.1]')

    def test_buffer_order_validation_and_copy(self):
        ns = namespace(Clock()); Buffer = ns['_DPActionBuffer']
        class Policy:
            action_horizon = 3
            def get_action_chunk(self, obs): return chunk
        chunk = np.arange(24.).reshape(1,3,8); b = Buffer(Policy(),3)
        first = b.get_action({}); chunk[:] = -99
        np.testing.assert_array_equal(first,np.arange(8)[None])
        np.testing.assert_array_equal(b.get_action({}),np.arange(8,16)[None])
        np.testing.assert_array_equal(b.get_action({}),np.arange(16,24)[None])
        for bad in (0,4,-1,1.5,True,'1'):
            with self.assertRaises(ValueError): Buffer(Policy(),bad)
        for bad in (np.zeros((3,8)),np.zeros((2,3,8)),np.zeros((1,2,8))):
            chunk = bad
            with self.assertRaises(ValueError): Buffer(Policy(),3).get_action({})


if __name__ == '__main__': unittest.main()
