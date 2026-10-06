"""Hardware-free configuration and lifecycle checks (unittest, no model loading)."""
import ast
import importlib.util
from pathlib import Path
import sys
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
CUSTOM = dict(
    translational_mass=0.1,
    rotational_inertia_mass_ratio=0.02,
    position_stiffness=500.0,
    orientation_stiffness=8.0,
    damping=[1., 2., 3., 4., 5., 6.],
    reference_frame=1,
    compliant_axis=[1, 0, 1, 0, 1, 0],
)


def load_inference_class():
    # Execute the complete production class without importing policy/model stacks.
    path = ROOT / 'eval_real_robot.py'
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef)
                and n.name == 'XArmInferenceEnv')
    namespace = dict(np=np, gym=gym, XArm7IK=Mock(), XArm7FK=Mock())
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['XArmInferenceEnv']


class AdmittanceConfigTests(unittest.TestCase):
    def assert_custom(self, config):
        self.assertIsInstance(config.admittance, XArmAdmittanceConfig)
        for name, value in CUSTOM.items():
            if isinstance(value, list):
                value = tuple(value)
            self.assertEqual(getattr(config.admittance, name), value)

    def test_yaml_defaults(self):
        for path, keys in [('config/robot/collect_data.yaml', ('robot',)),
                           ('config/eval/flip_mug.yaml', ('eval', 'real_robot'))]:
            cfg = yaml.safe_load((ROOT / path).read_text())
            for key in keys:
                cfg = cfg[key]
            values = dict(cfg['admittance'])
            self.assertTrue(values.pop('enabled'))
            values['damping'] = tuple(values['damping'])
            values['compliant_axis'] = tuple(values['compliant_axis'])
            self.assertEqual(XArmAdmittanceConfig(**values), XArmAdmittanceConfig())

    def test_collection_config_and_finally(self):
        spec = importlib.util.spec_from_file_location('collect_under_test', ROOT / 'robot/teleop/collect_data.py')
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
                self.assert_custom(factory.call_args.args[0])
                expected = [call.connect()]
                if enabled:
                    expected += [call.robot_system.follower.enable_admittance_control(),
                                 call.robot_system.follower.resume_motion_commands()]
                expected += [call.record_parallel(max_frame=cfg.dataset.max_frames,
                                                  fps=cfg.dataset.fps,
                                                  teleop_hz=cfg.dataset.teleop_hz)]
                if enabled:
                    expected += [call.robot_system.follower.disable_admittance_control()]
                    self.assertTrue(follower._motion_paused)
                expected += [call.disconnect()]
                self.assertEqual(robot.mock_calls, expected)

    def eval_config(self):
        cfg = OmegaConf.load(ROOT / 'config/eval/flip_mug.yaml')
        robot_cfg = cfg.eval.real_robot
        robot_cfg.admittance.update(CUSTOM)
        robot_cfg.dry_run_image_path = ''
        robot_cfg.dry_run_wrist_image_path = ''
        return robot_cfg, cfg.plan_config

    def test_inference_config_and_cleanup(self):
        robot_cfg, plan_cfg = self.eval_config()
        robot_cfg.dry_run = False
        follower = Mock()
        follower._control_lock = threading.Lock()
        owner = Mock()
        owner.robot_system.follower = follower
        owner.sensors.cameras = [Mock(name=str(camera.serial)) for camera in
                                 (robot_cfg.cameras.overhead, robot_cfg.cameras.wrist)]
        for camera, config in zip(owner.sensors.cameras,
                                  (robot_cfg.cameras.overhead, robot_cfg.cameras.wrist)):
            camera.name = str(config.serial)
        with patch('robopy.robots.xarm.XArmRobot', return_value=owner) as factory:
            env = load_inference_class()(robot_cfg, plan_cfg)
        owner.connect.assert_called_once_with(connect_leader=False)
        config = factory.call_args.args[0]
        self.assert_custom(config)
        self.assertEqual((config.gripper_open, config.gripper_close,
                          config.gripper_speed, config.gripper_force),
                         (robot_cfg.gripper.open_position, robot_cfg.gripper.closed_position,
                          robot_cfg.gripper.speed, robot_cfg.gripper.force))
        self.assertIsInstance(config.gripper_open, float)
        self.assertIsInstance(config.gripper_close, float)
        follower.enable_admittance_control.assert_not_called()
        try:
            env.enable_admittance()
        finally:
            env.close()
        self.assertEqual(follower.method_calls, [call.enable_admittance_control(),
                         call.resume_motion_commands(), call.disable_admittance_control()])
        owner.disconnect.assert_called_once_with()
        env.close()
        owner.disconnect.assert_called_once_with()
        self.assertTrue(follower._motion_paused)

    def test_inference_missing_parameter_fails_before_connection(self):
        robot_cfg, plan_cfg = self.eval_config()
        robot_cfg.dry_run = False
        del robot_cfg.admittance.position_stiffness
        with patch('robopy.robots.xarm.XArmRobot') as factory, \
             patch.dict(sys.modules, {'pyrealsense2': Mock()}):
            with self.assertRaises(AttributeError):
                load_inference_class()(robot_cfg, plan_cfg)
        factory.assert_not_called()

    def test_inference_dry_run_never_constructs_follower(self):
        robot_cfg, plan_cfg = self.eval_config()
        robot_cfg.dry_run = True
        with patch('robopy.robots.xarm.XArmRobot') as factory:
            env = load_inference_class()(robot_cfg, plan_cfg)
            env.enable_admittance()
            env.disable_admittance()
            env.close()
        factory.assert_not_called()


if __name__ == '__main__':
    unittest.main()
