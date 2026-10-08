"""Hardware-free tests of the real rollout, including wall time and saved logs."""
import ast
from collections import deque
from pathlib import Path
import signal
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import cv2
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]


class Clock:
    now = 0.
    def monotonic(self):
        return self.now
    def sleep(self, duration):
        self.now += duration


class Diffusion:
    obs_horizon = 2
    def __init__(self, commands):
        self.commands = iter(commands)
        self.observations = []
    def set_env(self, env):
        pass
    def get_action(self, obs):
        self.observations.append(obs)
        return np.array([[.7, .1, .2, 0, 0, 0, 1, next(self.commands)]])


class GPC:
    def __init__(self, commands):
        self.diffusion_policy = Diffusion(commands)
        self.observations = self.diffusion_policy.observations
    def set_env(self, env):
        pass
    def get_action(self, obs, dp_info_dict, projection_state):
        return self.diffusion_policy.get_action(dp_info_dict)


class WorldModel:
    def __init__(self, commands):
        self.policy = Diffusion(commands)
        self.observations = self.policy.observations
        self.cfg = SimpleNamespace(receding_horizon=1)
        self.solver = Mock()
        self.action_processor = object()
        self.set_action_projector = Mock()
    def set_env(self, env):
        pass
    def get_action(self, obs, projection_state):
        return self.policy.get_action(obs)


class FakeEnv:
    def __init__(self, stop_at=None, max_delta=1.):
        self.num_envs = 1
        self.action_space = Mock()
        self.step = -1
        self.stop_at = stop_at
        self.max_delta = max_delta
        self.sent = []
        self._last_gripper = 0.
        self._last_actual_gripper = 0.
        self._last_target_qpos = np.zeros(7)
        self._last_safe_qpos = np.zeros(7)
        self.close = Mock()
    def get_image(self):
        self.step += 1
        if self.step == self.stop_at:
            signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
        image = np.full((16, 16, 3), self.step, np.uint8)
        return image
    def get_robot_state(self):
        self._last_actual_gripper = self._last_gripper
        return np.zeros(7), np.zeros(7), np.array([.5 + self.step*.001, 0, .3, 0, 0, 0, 1])
    def execute(self, action, space):
        assert space == 'cartesian'
        command = np.asarray(action).reshape(-1).copy()
        command[7] = np.clip(command[7], self._last_gripper-self.max_delta, self._last_gripper+self.max_delta)
        self._last_gripper = command[7]
        self.sent.append(command)
        return command


def namespace(clock):
    ns = dict(np=np, plt=plt, time=clock, signal=signal, cv2=cv2, h5py=h5py,
              OmegaConf=OmegaConf, deque=deque, subprocess=Mock(),
              swm=SimpleNamespace(policy=SimpleNamespace(DiffusionPolicy=Diffusion,
                  GPCPolicy=GPC, WorldModelPolicy=WorldModel)),
              _load_or_capture_goal=lambda env, cfg: (np.zeros((16,16,3), np.uint8), np.zeros(8)))
    names = {'run_xarm_task', '_policy_observation',
             '_reinforced_diffusion_action', 'plot_commanded_vs_actual_gripper'}
    nodes = [n for n in ast.parse((ROOT/'eval_real_robot.py').read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'eval_real_robot.py', 'exec'), ns)
    return ns


class ReinforcementTests(unittest.TestCase):
    def rollout(self, enabled=True, commands=(.9, .1, .8), stop_at=None, max_delta=1., policy_class=Diffusion):
        cfg = OmegaConf.create({'plan_config': {'action_space':'cartesian'}, 'eval': {
            'real_robot': {'max_steps':6, 'control_hz':10, 'non_interactive':True, 'use_action_projector':False,
                'gripper': {'open_position':84., 'closed_position':0., 'max_delta': max_delta},
                'gripper_reinforcement': {'enabled':enabled, 'threshold':.2,
                    'target_opening_mm':20., 'duration_sec':.25}}}})
        env, policy = FakeEnv(stop_at, max_delta), policy_class(commands)
        clock = Clock()
        handler = signal.getsignal(signal.SIGINT)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ns = namespace(clock)
            ns['XArmInferenceEnv'] = Mock(return_value=env)
            ns['run_xarm_task'](cfg, policy, {}, root)
            with h5py.File(root/'rollout.h5') as f:
                logs = {key: f[key][:] for key in f if isinstance(f[key], h5py.Dataset)}
            self.assertTrue((root/'commanded_vs_actual_gripper.png').exists())
        self.assertEqual(signal.getsignal(signal.SIGINT), handler)
        env.close.assert_called_once()
        return env, policy, logs

    def test_hold_resume_retrigger_history_and_logs(self):
        env, policy, logs = self.rollout()
        np.testing.assert_array_equal(logs['gripper_reinforcement_active'], [1,1,1,0,1,1])
        self.assertEqual(len(policy.observations), 3)
        np.testing.assert_array_equal(policy.observations[1]['pixels'][0,:,0,0,0], [2,3])
        self.assertTrue(all(set(obs) == {'pixels'} for obs in policy.observations))
        self.assertNotIn('wrist_pixels', logs)
        np.testing.assert_allclose(np.array(env.sent)[:3,:7], np.tile(env.sent[0][:7], (3,1)))
        np.testing.assert_allclose(env.sent[0][:7], [.5,0,.3,0,0,0,1])
        np.testing.assert_allclose(logs['gripper_command'], [64/84]*3+[.1]+[64/84]*2)
        np.testing.assert_allclose(logs['gripper_policy_command'], [.9,np.nan,np.nan,.1,.8,np.nan], equal_nan=True)
        np.testing.assert_allclose(logs['gripper_actual'][1:], logs['gripper_command'][:-1])
        np.testing.assert_allclose(logs['commanded_action'][:,7], logs['gripper_command'])
        np.testing.assert_allclose(logs['timestamp'], np.arange(6)*.1, atol=1e-8)
        self.assertEqual(logs['pixels'].shape[0], 6)

    def test_disabled_preserves_dp_actions(self):
        env, policy, logs = self.rollout(False, commands=[.9]*6)
        self.assertEqual(len(policy.observations),6)
        self.assertFalse(logs['gripper_reinforcement_active'].any())
        np.testing.assert_allclose(np.array(env.sent)[:,7], .9)

    def test_gpc_is_not_reinforced(self):
        env, policy, logs = self.rollout(commands=[.9]*6, policy_class=GPC)
        self.assertEqual(len(policy.observations),6)
        self.assertFalse(logs['gripper_reinforcement_active'].any())
        np.testing.assert_allclose(np.array(env.sent)[:,7], .9)

    def test_world_model_is_not_reinforced(self):
        env, policy, logs = self.rollout(commands=[.9]*6, policy_class=WorldModel)
        self.assertEqual(len(policy.observations), 6)
        policy.set_action_projector.assert_called_once_with(None)
        policy.solver.configure.assert_called_once()
        self.assertFalse(logs['gripper_reinforcement_active'].any())
        np.testing.assert_allclose(np.array(env.sent)[:,7], .9)
        self.assertTrue(np.isnan(logs['gripper_policy_command']).all())

    def test_invalid_config_rejected_before_robot_connection(self):
        for key, value in [('threshold', float('nan')), ('threshold', -.1),
                           ('threshold', 1.1), ('target_opening_mm', float('inf')),
                           ('target_opening_mm', -1), ('target_opening_mm', 85),
                           ('duration_sec', 0), ('duration_sec', -1),
                           ('duration_sec', float('inf'))]:
            with self.subTest(key=key, value=value):
                cfg = OmegaConf.create({'plan_config': {'action_space':'cartesian'}, 'eval': {
                    'real_robot': {'gripper': {'open_position':84., 'closed_position':0.},
                        'gripper_reinforcement': {'enabled':True, 'threshold':.2,
                            'target_opening_mm':0., 'duration_sec':5.}}}})
                cfg.eval.real_robot.gripper_reinforcement[key] = value
                ns = namespace(Clock())
                ns['XArmInferenceEnv'] = Mock()
                with self.assertRaisesRegex(ValueError, 'Invalid gripper reinforcement'):
                    ns['run_xarm_task'](cfg, Diffusion([.9]), {}, Path('/tmp'))
                ns['XArmInferenceEnv'].assert_not_called()

    def test_opening_conversion_and_single_view_dry_run(self):
        from unittest.mock import patch
        from scipy.spatial.transform import Rotation
        from test_xarm_admittance_config import inference_namespace
        cfg = OmegaConf.load(ROOT/'config/eval/flip_mug.yaml')
        real = cfg.eval.real_robot
        self.assertEqual(dict(real.gripper_reinforcement),
                         dict(enabled=True, threshold=.2, target_opening_mm=0., duration_sec=5.))
        cfg.plan_config.action_space = 'cartesian'
        real.dry_run = True
        real.dry_run_image_path = ''
        real.camera.width = real.camera.height = 16
        real.non_interactive = True
        real.max_steps = 4
        real.control_hz = 10
        real.gripper.max_delta = 1.
        real.gripper_reinforcement.duration_sec = .25
        ns = namespace(Clock())
        env_ns = inference_namespace()
        env_ns.update(cv2=cv2, Rotation=Rotation)
        ns['XArmInferenceEnv'] = env_ns['XArmInferenceEnv']
        policy = Diffusion([.9,.1])
        with tempfile.TemporaryDirectory() as tmp, patch('robopy.robots.xarm.XArmRobot') as owner:
            root = Path(tmp)
            ns['run_xarm_task'](cfg, policy, {}, root)
            owner.assert_not_called()
            with h5py.File(root/'rollout.h5') as f:
                np.testing.assert_allclose(f['gripper_command'][:3], 1.)
                np.testing.assert_allclose(f['commanded_action'][:3,:7],
                                           np.tile(f['ee_pos_quat'][0], (3,1)))
                self.assertTrue(np.isnan(f['actual_gripper'][:]).all())
                self.assertTrue(np.isnan(f['gripper_actual'][:]).all())
                self.assertNotIn('wrist_pixels', f)
        self.assertEqual(len(policy.observations), 2)
        self.assertTrue(all(set(obs) == {'pixels'} for obs in policy.observations))
        select = namespace(Clock())['_reinforced_diffusion_action']
        for opening, expected in [(0.,1.), (42.,.5), (84.,0.)]:
            action, active, _ = select(Diffusion([.9]), {}, np.array([.5,0,.3,0,0,0,1]), {},
                SimpleNamespace(threshold=.2,target_opening_mm=opening,duration_sec=.5), real.gripper)
            self.assertTrue(active)
            self.assertAlmostEqual(action[-1], expected)

    def test_threshold_is_strict(self):
        _, policy, logs = self.rollout(commands=[.1,.2,0,.2,.1,.2])
        self.assertEqual(len(policy.observations),6)
        self.assertFalse(logs['gripper_reinforcement_active'].any())

    def test_stop_during_reinforcement(self):
        env, policy, logs = self.rollout(stop_at=2)
        self.assertEqual(len(env.sent),2)
        self.assertEqual(len(policy.observations),1)
        np.testing.assert_array_equal(logs['gripper_reinforcement_active'], [1,1])

    def test_logs_use_safety_clipped_command(self):
        env, _, logs = self.rollout(max_delta=.1)
        self.assertAlmostEqual(logs['gripper_command'][0], .1)
        self.assertAlmostEqual(logs['gripper_command'][1], .2)
        np.testing.assert_allclose(logs['commanded_action'][:,7], logs['gripper_command'])

    def test_deadline_uses_elapsed_time(self):
        clock = Clock()
        select = namespace(clock)['_reinforced_diffusion_action']
        policy = Diffusion([.9, .1])
        cfg = SimpleNamespace(threshold=.2, target_opening_mm=20., duration_sec=.5)
        gripper = SimpleNamespace(open_position=84., closed_position=0.)
        state = {}
        pose = np.array([.5,0,.3,0,0,0,1])
        select(policy, {'latest':0}, pose, state, cfg, gripper)
        # Many iterations with no elapsed time must not expire the hold.
        for _ in range(20):
            self.assertTrue(select(policy, {'latest':1}, pose+.01, state, cfg, gripper)[1])
        clock.now = .499
        self.assertTrue(select(policy, {'latest':2}, pose, state, cfg, gripper)[1])
        clock.now = .5
        self.assertFalse(select(policy, {'latest':3}, pose, state, cfg, gripper)[1])
        self.assertEqual(policy.observations, [{'latest':0}, {'latest':3}])
        self.assertFalse(state)

    def test_plot_has_only_command_and_actual(self):
        from unittest.mock import patch
        commands = np.zeros((3,8))
        fig, ax = plt.subplots()
        with tempfile.TemporaryDirectory() as tmp, patch.object(plt, 'subplots', return_value=(fig,ax)):
            namespace(Clock())['plot_commanded_vs_actual_gripper'](
                commands, [0,0,0], Path(tmp)/'plot.png')
            self.assertGreater((Path(tmp)/'plot.png').stat().st_size, 0)
        self.assertEqual(len(ax.patches),0)
        self.assertEqual(len(ax.lines),2)
        self.assertEqual([t.get_text() for t in ax.get_legend().get_texts()],
                         ['Command (previous step)', 'Actual (G2 measurement)'])
        self.assertEqual(ax.get_title(), 'Commanded vs Actual Gripper')
        self.assertEqual(ax.get_xlabel(), 'Observation step')
        self.assertEqual(ax.get_ylabel(), 'Gripper (0=open, 1=closed)')


if __name__ == '__main__':
    unittest.main()
