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
    action_horizon = 3
    def __init__(self, commands):
        self.commands = iter(commands)
        self.observations = []
    def get_action_chunk(self, obs):
        return np.repeat(self.get_action(obs)[:, None], self.action_horizon, axis=1)
    def set_env(self, env):
        pass
    def get_action(self, obs):
        self.observations.append(obs)
        return np.array([[.7, .1, .2, 0, 0, 0, 1, next(self.commands)]])


class GPC:
    action_horizon = 3
    def get_action_chunk(self, obs, **kwargs):
        return np.repeat(self.get_action(obs, **kwargs)[:, None], self.action_horizon, axis=1)
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


def namespace(clock, source=None):
    ns = dict(np=np, plt=plt, time=clock, signal=signal, cv2=cv2, h5py=h5py,
              OmegaConf=OmegaConf, deque=deque, subprocess=Mock(),
              swm=SimpleNamespace(policy=SimpleNamespace(DiffusionPolicy=Diffusion,
                  GPCPolicy=GPC, WorldModelPolicy=type('WM', (), {}))),
              _load_or_capture_goal=lambda env, cfg: (np.zeros((16,16,3), np.uint8),
                  np.zeros((16,16,3), np.uint8), np.zeros(8)))
    names = {'_DPActionBuffer', '_validate_gripper_reinforcement', '_expire_gripper_reinforcement',
             '_gripper_reinforcement_allowed', '_run_xarm_task_with_env', '_policy_observation',
             '_reinforced_diffusion_action', '_reinforced_gpc_action', 'plot_commanded_vs_actual_gripper'}
    nodes = [n for n in ast.parse(source or (ROOT/'eval_real_robot.py').read_text()).body
             if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'eval_real_robot.py', 'exec'), ns)
    return ns


class ReinforcementTests(unittest.TestCase):
    def rollout(self, enabled=True, commands=(.9, .1, .8), stop_at=None, max_delta=1., policy_class=Diffusion, rearm=None):
        cfg = OmegaConf.create({'plan_config': {'action_space':'cartesian'}, 'eval': {
            'real_robot': {'max_steps':6, 'control_hz':10, 'non_interactive':True,
                'gripper': {'open_position':84., 'closed_position':0., 'max_delta': max_delta},
                'gripper_reinforcement': {'enabled':enabled, 'threshold':.2,
                    'target_opening_mm':20., 'duration_sec':.25}}}})
        if rearm is not None:
            cfg.eval.real_robot.gripper_reinforcement.rearm_threshold = rearm
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

    def test_gpc_disabled_preserves_actions(self):
        env, policy, logs = self.rollout(False, commands=[.9]*6, policy_class=GPC)
        self.assertEqual(len(policy.observations),6)
        self.assertFalse(logs['gripper_reinforcement_active'].any())
        np.testing.assert_allclose(np.array(env.sent)[:,7], .9)

    def test_gpc_hold_resume_history_and_logs(self):
        env, policy, logs = self.rollout(policy_class=GPC)
        _, _, dp_logs = self.rollout()
        self.assertEqual(set(logs), set(dp_logs))
        np.testing.assert_array_equal(logs['gripper_reinforcement_active'], [1,1,1,0,1,1])
        self.assertEqual(len(policy.observations), 3)
        np.testing.assert_array_equal(policy.observations[1]['pixels'][0,:,0,0,0], [2,3])
        np.testing.assert_allclose(np.array(env.sent)[:3,:7], np.tile(env.sent[0][:7], (3,1)))
        np.testing.assert_allclose(env.sent[0][:7], [.5,0,.3,0,0,0,1])
        np.testing.assert_allclose(logs['gripper_command'], [64/84]*3+[.1]+[64/84]*2)
        np.testing.assert_allclose(logs['gripper_policy_command'], [.9,np.nan,np.nan,.1,.8,np.nan], equal_nan=True)
        np.testing.assert_allclose(logs['commanded_action'][:,7], logs['gripper_command'])
        np.testing.assert_allclose(logs['gripper_actual'][1:], logs['gripper_command'][:-1])

    def test_gpc_threshold_is_strict(self):
        env, policy, logs = self.rollout(commands=[.1,.2,0,.2,.1,.2], policy_class=GPC)
        self.assertEqual(len(policy.observations), 6)
        self.assertFalse(logs['gripper_reinforcement_active'].any())
        np.testing.assert_allclose(np.array(env.sent)[:,7], [.1,.2,0,.2,.1,.2])

    def test_gpc_validation_and_monotonic_deadline(self):
        clock = Clock()
        select = namespace(clock)['_reinforced_gpc_action']
        cfg = SimpleNamespace(threshold=.2, target_opening_mm=0., duration_sec=5.)
        gripper = SimpleNamespace(open_position=84., closed_position=0.)
        pose = np.array([.5,0,.3,0,0,0,1])
        policy = Mock()
        result = np.array([[.7,.1,.2,0,0,0,1,.9]])
        policy.get_action.return_value = result
        state = {}
        obs, dp_obs, projection = {}, {'pixels': 1}, {'ee': pose}
        action, active, command = select(policy, obs, dp_obs, projection, pose, state, cfg, gripper)
        self.assertTrue(active)
        self.assertEqual(command, .9)
        self.assertEqual(action[7], 1.)
        policy.get_action.assert_called_once_with(obs, dp_info_dict=dp_obs, projection_state=projection)
        pose[0] = 99
        for now in [0., 1., 4.999]:
            clock.now = now
            held, active, command = select(policy, obs, dp_obs, projection, pose, state, cfg, gripper)
            np.testing.assert_array_equal(held, action)
            self.assertTrue(active)
            self.assertTrue(np.isnan(command))
        self.assertEqual(policy.get_action.call_count, 1)
        clock.now = 5.
        result[0,7] = .1
        resumed, active, _ = select(policy, obs, dp_obs, projection, pose, state, cfg, gripper)
        self.assertIs(resumed, result)
        self.assertFalse(active)
        self.assertFalse(state)
        self.assertEqual(policy.get_action.call_count, 2)
        for bad in [np.zeros(7), np.zeros((2,8)), np.zeros((2,4)), np.full((1,8), np.nan)]:
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                policy.get_action.return_value = bad
                select(policy, obs, dp_obs, projection, pose, {}, cfg, gripper)
        policy.get_action.return_value = np.array([[0,0,0,0,0,0,1,.9]])
        for bad_pose in [np.zeros(6), np.full(7, np.nan)]:
            with self.assertRaises(ValueError):
                select(policy, obs, dp_obs, projection, bad_pose, {}, cfg, gripper)
        for field, value in [('threshold', -1), ('threshold', float('nan')),
                             ('duration_sec', 0), ('duration_sec', float('inf')),
                             ('target_opening_mm', 85), ('target_opening_mm', -1)]:
            bad_cfg = SimpleNamespace(**vars(cfg))
            setattr(bad_cfg, field, value)
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                select(policy, obs, dp_obs, projection, pose, {}, bad_cfg, gripper)
        for bad_gripper in [SimpleNamespace(open_position=0., closed_position=0.),
                            SimpleNamespace(open_position=float('nan'), closed_position=0.)]:
            with self.assertRaises(ValueError):
                select(policy, obs, dp_obs, projection, pose, {}, cfg, bad_gripper)

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

    def test_exact_legacy_regression(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            'legacy', ROOT/'tests/legacy_gripper_reinforcement.py')
        legacy = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(legacy)
        for policy_type, name in [(Diffusion, '_reinforced_diffusion_action'),
                                  (GPC, '_reinforced_gpc_action')]:
            for rearm in [None, 1.0]:
                with self.subTest(policy=policy_type.__name__, rearm=rearm):
                    clocks = [Clock(), Clock()]
                    legacy.time = clocks[0]
                    functions = [getattr(legacy, name), namespace(clocks[1])[name]]
                    policies = [policy_type([.2, .3, .8, .1, .9, .1]) for _ in range(2)]
                    states = [{}, {}]
                    cfg = SimpleNamespace(threshold=.2, target_opening_mm=20., duration_sec=5.)
                    if rearm is not None:
                        cfg.rearm_threshold = rearm
                    gripper = SimpleNamespace(open_position=84., closed_position=0.)
                    for step, now in enumerate([0., 1., 1., 5.999, 6., 10.999, 11., 12., 17.]):
                        pose = np.array([.5 + step*.01, 0, .3, 0, 0, 0, 1])
                        outputs = []
                        for i in range(2):
                            clocks[i].now = now
                            args = (policies[i], {'step':step})
                            if policy_type is GPC:
                                args += ({'step':step}, {'ee':pose})
                            outputs.append(functions[i](*args, pose, states[i], cfg, gripper))
                        np.testing.assert_array_equal(outputs[0][0], outputs[1][0])
                        self.assertEqual(outputs[0][1], outputs[1][1])
                        np.testing.assert_equal(outputs[0][2], outputs[1][2])
                        self.assertEqual(states[0].keys(), states[1].keys())
                        for key in states[0]:
                            np.testing.assert_array_equal(states[0][key], states[1][key])
                        self.assertEqual(policies[0].observations, policies[1].observations)

    def test_hysteresis_both_policies(self):
        for policy_type, name in [(Diffusion, '_reinforced_diffusion_action'),
                                  (GPC, '_reinforced_gpc_action')]:
            with self.subTest(policy=policy_type.__name__):
                clock = Clock()
                select = namespace(clock)[name]
                policy = policy_type([0., .1, .2, .21, .8, .7, .1001, .1, .6])
                cfg = SimpleNamespace(threshold=.2, rearm_threshold=.1,
                                      target_opening_mm=0., duration_sec=5.)
                gripper = SimpleNamespace(open_position=84., closed_position=0.)
                state = {}
                for step, (now, active, calls) in enumerate([
                        (0, False, 1), (1, False, 2), (2, False, 3), (3, True, 4),
                        (3, True, 4), (7.999, True, 4), (8, False, 5),
                        (9, False, 6), (10, False, 7), (11, False, 8), (12, True, 9)]):
                    clock.now = now
                    pose = np.array([.5+step*.01, 0, .3, 0, 0, 0, 1])
                    args = (policy, {'step':step})
                    if policy_type is GPC:
                        args += ({'step':step}, {'ee':pose})
                    action, actual_active, raw = select(*args, pose, state, cfg, gripper)
                    self.assertEqual(actual_active, active)
                    self.assertEqual(len(policy.observations), calls)
                    if active:
                        np.testing.assert_array_equal(np.asarray(action)[:7], state['pose'])
                        self.assertEqual(action[7], 1.)
                    if 6 <= step <= 8:
                        self.assertTrue(state['disarmed'])
                        self.assertGreater(raw, cfg.rearm_threshold)
                        self.assertEqual(np.asarray(action).reshape(-1)[7], raw)
                    if step == 9:
                        self.assertFalse(state)

    def test_hysteresis_rollout_logs_and_disabled(self):
        for policy_type in [Diffusion, GPC]:
            with self.subTest(policy=policy_type.__name__):
                _, policy, logs = self.rollout(commands=[.9, .8, .7, .1],
                                              policy_class=policy_type, rearm=.1)
                np.testing.assert_array_equal(logs['gripper_reinforcement_active'], [1,1,1,0,0,0])
                np.testing.assert_allclose(logs['gripper_policy_command'],
                                           [.9,np.nan,np.nan,.8,.7,.1], equal_nan=True)
                np.testing.assert_allclose(logs['commanded_action'][:,7], logs['gripper_command'])
                self.assertEqual(len(policy.observations), 4)
                _, _, legacy_logs = self.rollout(policy_class=policy_type)
                self.assertEqual(set(logs), set(legacy_logs))
                for key in logs:
                    self.assertEqual(logs[key].dtype, legacy_logs[key].dtype)
                    self.assertEqual(logs[key].shape, legacy_logs[key].shape)
                _, policy, logs = self.rollout(False, commands=[.9]*6,
                                              policy_class=policy_type, rearm=float('nan'))
                self.assertEqual(len(policy.observations), 6)
                self.assertFalse(logs['gripper_reinforcement_active'].any())

    def test_threshold_one_only_allowed_in_hysteresis(self):
        validate = namespace(Clock())['_validate_gripper_reinforcement']
        cfg = SimpleNamespace(threshold=1., rearm_threshold=.1,
                              target_opening_mm=0., duration_sec=5.)
        gripper = SimpleNamespace(open_position=84., closed_position=0.)
        self.assertEqual(validate(cfg, gripper), .1)
        cfg.rearm_threshold = 1.
        with self.assertRaises(ValueError):
            validate(cfg, gripper)

    def test_invalid_hysteresis_configuration_both_policies(self):
        for name in ['_reinforced_diffusion_action', '_reinforced_gpc_action']:
            select = namespace(Clock())[name]
            for field, value in [('rearm_threshold', -.1), ('rearm_threshold', .2),
                                 ('rearm_threshold', .3), ('rearm_threshold', 1.01),
                                 ('rearm_threshold', float('nan')),
                                 ('rearm_threshold', float('inf')),
                                 ('rearm_threshold', None), ('rearm_threshold', 'invalid'),
                                 ('threshold', 1.01), ('threshold', -.1),
                                 ('threshold', float('nan')), ('threshold', float('inf')),
                                 ('duration_sec', 0), ('duration_sec', float('inf')),
                                 ('target_opening_mm', float('nan'))]:
                with self.subTest(name=name, field=field, value=value):
                    cfg = SimpleNamespace(threshold=.2, rearm_threshold=.1,
                                          target_opening_mm=0., duration_sec=5.)
                    setattr(cfg, field, value)
                    policy = Mock()
                    args = (policy, {}) + (({}, {}) if 'gpc' in name else ())
                    with self.assertRaisesRegex(ValueError, 'configuration'):
                        select(*args, np.zeros(7), {}, cfg,
                               SimpleNamespace(open_position=84., closed_position=0.))
                    policy.get_action.assert_not_called()

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
