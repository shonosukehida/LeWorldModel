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


class FakeEnv:
    def __init__(self, stop_at=None, max_delta=1.):
        self.step = -1
        self.stop_at = stop_at
        self.max_delta = max_delta
        self.sent = []
        self._last_gripper = 0.
        self._last_actual_gripper = 0.
        self._last_target_qpos = np.zeros(7)
        self._last_safe_qpos = np.zeros(7)
        self.close = Mock()
    def get_images(self):
        self.step += 1
        if self.step == self.stop_at:
            signal.getsignal(signal.SIGINT)(signal.SIGINT, None)
        image = np.full((16, 16, 3), self.step, np.uint8)
        return image, image + 10
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
                  GPCPolicy=GPC, WorldModelPolicy=type('WM', (), {}))),
              _load_or_capture_goal=lambda env, cfg: (np.zeros((16,16,3), np.uint8),
                  np.zeros((16,16,3), np.uint8), np.zeros(8)))
    names = {'_run_xarm_task_with_env', '_policy_observation',
             '_reinforced_diffusion_action', 'plot_commanded_vs_actual_gripper'}
    nodes = [n for n in ast.parse((ROOT/'eval_real_robot.py').read_text()).body
             if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'eval_real_robot.py', 'exec'), ns)
    return ns


class ReinforcementTests(unittest.TestCase):
    def rollout(self, enabled=True, commands=(.9, .1, .8), stop_at=None, max_delta=1., policy_class=Diffusion):
        cfg = OmegaConf.create({'plan_config': {'action_space':'cartesian'}, 'eval': {
            'real_robot': {'max_steps':6, 'control_hz':10, 'non_interactive':True,
                'gripper': {'open_position':84., 'closed_position':0., 'max_delta': max_delta},
                'gripper_reinforcement': {'enabled':enabled, 'threshold':.2,
                    'target_opening_mm':20., 'duration_sec':.25}}}})
        env, policy = FakeEnv(stop_at, max_delta), policy_class(commands)
        clock = Clock()
        handler = signal.getsignal(signal.SIGINT)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            namespace(clock)['_run_xarm_task_with_env'](cfg, policy, {}, root, env)
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
        np.testing.assert_array_equal(policy.observations[1]['wrist_pixels'][0,:,0,0,0], [12,13])
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
