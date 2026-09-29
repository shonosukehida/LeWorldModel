"""Hardware-free checks of YAML wiring and the production control lifecycle."""
import ast
import importlib.util
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, call, patch

import gymnasium as gym
import numpy as np
from omegaconf import OmegaConf
import yaml
from robopy.config.robot_config import XArmAdmittanceConfig

ROOT = Path(__file__).resolve().parents[1]
CUSTOM = dict(translational_mass=0.1, rotational_inertia_mass_ratio=0.02,
              position_stiffness=500., orientation_stiffness=8.,
              damping=[1., 2., 3., 4., 5., 6.], reference_frame=1,
              compliant_axis=[1, 0, 1, 0, 1, 0])


def inference_namespace():
    # Run the complete production class without loading unrelated ML models.
    path = ROOT / 'eval_real_robot.py'
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                and n.name == 'XArmInferenceEnv')
    ns = dict(np=np, gym=gym, XArm7IK=Mock(), XArm7FK=Mock())
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), ns)
    return ns


def rollout_enable(cfg, env):
    # Execute the actual rollout's enabled guard, not a copy of its logic.
    path = ROOT / 'eval_real_robot.py'
    tree = ast.parse(path.read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
              and n.name == 'run_xarm_task')
    body = next(n for n in fn.body if isinstance(n, ast.Try)).body
    start = next(i for i, n in enumerate(body) if isinstance(n, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == 'use_admittance'
                         for t in n.targets))
    exec(compile(ast.Module(body=body[start:start + 2], type_ignores=[]), str(path), 'exec'),
         dict(cfg=cfg, env=env, OmegaConf=OmegaConf))


class AdmittanceTests(unittest.TestCase):
    def assert_config(self, cfg):
        self.assertIsInstance(cfg.admittance, XArmAdmittanceConfig)
        for key, value in CUSTOM.items():
            self.assertEqual(getattr(cfg.admittance, key),
                             tuple(value) if isinstance(value, list) else value)

    def test_yaml_defaults(self):
        for file, keys, enabled in [
            ('config/robot/collect_data.yaml', ['robot'], True),
            ('config/eval/flip_mug.yaml', ['eval', 'real_robot'], False),
        ]:
            cfg = yaml.safe_load((ROOT / file).read_text())
            for key in keys:
                cfg = cfg[key]
            params = dict(cfg['admittance'])
            self.assertEqual(params.pop('enabled'), enabled)
            for key in ('damping', 'compliant_axis'):
                params[key] = tuple(params[key])
            self.assertEqual(XArmAdmittanceConfig(**params), XArmAdmittanceConfig())

    def test_collection_config_and_failure_cleanup(self):
        spec = importlib.util.spec_from_file_location('collection', ROOT / 'robot/teleop/collect_data.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for enabled in (True, False):
            with self.subTest(enabled=enabled), tempfile.TemporaryDirectory() as temp:
                cfg = module.load_robot_config()
                cfg.robot.admittance.update(CUSTOM)
                cfg.robot.admittance.enabled = enabled
                robot = Mock()
                follower = robot.robot_system.follower
                follower._control_lock = threading.Lock()
                robot.record_parallel.side_effect = RuntimeError('record failed')
                with patch.object(module, 'load_robot_config', return_value=cfg), \
                     patch('robopy.robots.xarm.XArmRobot', return_value=robot) as factory, \
                     patch.object(Path, 'home', return_value=Path(temp)), \
                     patch('builtins.input', return_value=''):
                    with self.assertRaisesRegex(RuntimeError, 'record failed'):
                        module.xarm_collect()
                self.assert_config(factory.call_args.args[0])
                expected = [call.connect()]
                if enabled:
                    expected += [call.robot_system.follower.enable_admittance_control(),
                                 call.robot_system.follower.resume_motion_commands()]
                expected += [call.record_parallel(max_frame=cfg.dataset.max_frames,
                             fps=cfg.dataset.fps, teleop_hz=cfg.dataset.teleop_hz)]
                if enabled:
                    expected += [call.robot_system.follower.disable_admittance_control()]
                    self.assertTrue(follower._motion_paused)
                self.assertEqual(robot.mock_calls, expected + [call.disconnect()])
                cameras = factory.call_args.args[0].sensors.cameras
                self.assertEqual([c.serial_no for c in cameras], cfg.robot.camera.serial_numbers)

    def eval_cfg(self):
        cfg = OmegaConf.load(ROOT / 'config/eval/flip_mug.yaml')
        cfg.eval.real_robot.admittance.update(CUSTOM)
        cfg.eval.real_robot.dry_run_image_path = ''
        return cfg

    def test_inference_config_enabled_guard_and_cleanup(self):
        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                cfg = self.eval_cfg()
                cfg.eval.real_robot.dry_run = False
                cfg.eval.real_robot.admittance.enabled = enabled
                follower = Mock()
                follower._control_lock = threading.Lock()
                with patch('robopy.robots.xarm.xarm_follower.XArmFollower', return_value=follower) as factory:
                    env = inference_namespace()['XArmInferenceEnv'](
                        cfg.eval.real_robot, cfg.plan_config, use_camera=False)
                self.assert_config(factory.call_args.args[0])
                fc = factory.call_args.args[0]
                self.assertEqual((fc.gripper_open, fc.gripper_close, fc.gripper_speed,
                                  fc.gripper_force), (84., 0., 200, 50))
                self.assertIsInstance(fc.gripper_open, float)
                follower.enable_admittance_control.assert_not_called()
                try:
                    rollout_enable(cfg, env)
                finally:
                    env.close()
                expected = [call.connect()]
                if enabled:
                    expected += [call.enable_admittance_control(), call.resume_motion_commands(),
                                 call.disable_admittance_control()]
                self.assertEqual(follower.method_calls, expected + [call.disconnect()])
                self.assertTrue(follower._motion_paused)

    def test_missing_parameters_fail_before_connection(self):
        for key in CUSTOM:
            with self.subTest(key=key):
                cfg = self.eval_cfg()
                cfg.eval.real_robot.dry_run = False
                del cfg.eval.real_robot.admittance[key]
                with patch('robopy.robots.xarm.xarm_follower.XArmFollower') as factory:
                    with self.assertRaisesRegex(AttributeError, key):
                        inference_namespace()['XArmInferenceEnv'](
                            cfg.eval.real_robot, cfg.plan_config, use_camera=False)
                factory.assert_not_called()

    def test_dry_run_does_not_construct_follower(self):
        cfg = self.eval_cfg()
        cfg.eval.real_robot.dry_run = True
        cfg.eval.real_robot.admittance.enabled = True
        with patch('robopy.robots.xarm.xarm_follower.XArmFollower') as factory:
            env = inference_namespace()['XArmInferenceEnv'](
                cfg.eval.real_robot, cfg.plan_config, use_camera=False)
            rollout_enable(cfg, env)
            env.disable_admittance()
            env.close()
        factory.assert_not_called()


if __name__ == '__main__':
    unittest.main()
