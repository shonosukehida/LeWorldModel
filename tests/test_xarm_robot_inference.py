"""Single-camera owner lifecycle and existing Proprio/dry-run contracts."""
import ast
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import Mock, patch

import cv2
import numpy as np
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation
from test_xarm_admittance_config import ROOT, inference_namespace


class RobotOwnerTests(unittest.TestCase):
    def cfg(self):
        return OmegaConf.load(ROOT / 'config/eval/flip_mug.yaml')

    def owner(self, cfg):
        owner = Mock()
        owner.robot_system.follower._control_lock = threading.Lock()
        camera = Mock()
        camera.name = str(cfg.eval.real_robot.camera.serial)
        owner.sensors.cameras = [camera]
        return owner, camera

    def test_single_camera_config_rgb_and_owner_cleanup(self):
        cfg = self.cfg()
        cfg.eval.real_robot.dry_run = False
        owner, camera = self.owner(cfg)
        # Channels have distinguishable values; clipping must precede uint8 conversion.
        camera.read.return_value = np.array([[[-1., 10.]], [[20., 30.]], [[300., 40.]]], np.float32)
        ns = inference_namespace()
        with patch('robopy.robots.xarm.XArmRobot', return_value=owner) as factory:
            env = ns['XArmInferenceEnv'](cfg.eval.real_robot, cfg.plan_config)
        owner.connect.assert_called_once_with(connect_leader=False)
        params = factory.call_args.args[0].sensors.cameras
        self.assertEqual(len(params), 1)
        expected = cfg.eval.real_robot.camera
        for key in ('width', 'height', 'fps', 'auto_exposure', 'exposure', 'auto_white_balance', 'white_balance'):
            self.assertEqual(getattr(params[0], key), expected[key])
        self.assertEqual(params[0].serial_no, str(expected.serial))
        self.assertEqual(params[0].name, str(expected.serial))
        self.assertIs(env._pipeline, camera)
        self.assertIs(env._robot, owner.robot_system.follower._robot)
        ns['XArm7IK'].assert_called_once()
        ns['XArm7FK'].assert_called_once()
        self.assertEqual(ns['XArm7IK'].call_args.kwargs, dict(
            tcp_offset=env._robot.tcp_offset, world_offset=env._robot.world_offset))
        image = env.get_image()
        self.assertEqual(image.shape, (1, 2, 3))
        self.assertEqual(image.dtype, np.uint8)
        np.testing.assert_array_equal(image, [[[0, 20, 255], [10, 30, 40]]])
        camera.read.assert_called_once_with(specific_color='rgb')
        owner.robot_system.follower.connect.assert_not_called()
        camera.connect.assert_not_called()
        camera.read.return_value = np.zeros((2, 3), np.float32)
        with self.assertRaisesRegex(RuntimeError, 'frame shape'):
            env.get_image()
        env.close()
        env.close()
        owner.disconnect.assert_called_once_with()
        owner.robot_system.follower.disconnect.assert_not_called()
        camera.disconnect.assert_not_called()
        camera.stop.assert_not_called()
        for key in ('_robot_owner', '_follower', '_robot', '_pipeline'):
            self.assertIsNone(getattr(env, key))

    def test_camera_disabled_and_leader_opt_in(self):
        cfg = self.cfg()
        cfg.eval.real_robot.dry_run = False
        cfg.eval.real_robot.connect_leader = True
        # No camera settings need to be accessed when use_camera=False.
        del cfg.eval.real_robot.camera
        owner = Mock()
        owner.robot_system.follower._control_lock = threading.Lock()
        with patch('robopy.robots.xarm.XArmRobot', return_value=owner) as factory:
            env = inference_namespace()['XArmInferenceEnv'](cfg.eval.real_robot, cfg.plan_config, use_camera=False)
        self.assertEqual(factory.call_args.args[0].sensors.cameras, [])
        owner.connect.assert_called_once_with(connect_leader=True)
        with self.assertRaisesRegex(RuntimeError, 'Camera is disabled'):
            env.get_image()
        env.close()

    def test_initialization_errors_preserved_even_when_cleanup_fails(self):
        for stage in ('connect', 'keyboard_interrupt', 'sdk', 'ik', 'camera_lookup'):
            with self.subTest(stage=stage):
                cfg = self.cfg()
                cfg.eval.real_robot.dry_run = False
                owner, camera = self.owner(cfg)
                ns = inference_namespace()
                error = RuntimeError(stage)
                if stage == 'connect':
                    owner.connect.side_effect = error
                elif stage == 'keyboard_interrupt':
                    error = KeyboardInterrupt()
                    owner.connect.side_effect = error
                elif stage == 'sdk':
                    owner.robot_system.follower._robot = None
                elif stage == 'ik':
                    ns['XArm7IK'].side_effect = error
                else:
                    owner.sensors.cameras = []
                owner.disconnect.side_effect = RuntimeError('cleanup failed')
                expected_type = KeyError if stage == 'camera_lookup' else type(error)
                with patch('robopy.robots.xarm.XArmRobot', return_value=owner):
                    with self.assertRaises(expected_type) as caught:
                        ns['XArmInferenceEnv'](cfg.eval.real_robot, cfg.plan_config)
                if stage not in ('sdk', 'camera_lookup'):
                    self.assertIs(caught.exception, error)
                self.assertNotIn('cleanup failed', str(caught.exception))
                owner.disconnect.assert_called_once_with()

    def test_failed_admittance_still_disconnects_owner(self):
        cfg = self.cfg()
        cfg.eval.real_robot.dry_run = False
        owner, camera = self.owner(cfg)
        with patch('robopy.robots.xarm.XArmRobot', return_value=owner):
            env = inference_namespace()['XArmInferenceEnv'](cfg.eval.real_robot, cfg.plan_config)
        follower = owner.robot_system.follower
        follower.enable_admittance_control.side_effect = RuntimeError('enable failed')
        with self.assertRaisesRegex(RuntimeError, 'enable failed'):
            env.enable_admittance()
        follower.disable_admittance_control.side_effect = RuntimeError('disable failed')
        with self.assertRaisesRegex(RuntimeError, 'disable failed'):
            env.close()
        follower.disable_admittance_control.assert_called_once_with()
        owner.disconnect.assert_called_once_with()
        self.assertIsNone(env._robot_owner)
        self.assertFalse(env._admittance_attempted)

    def test_single_view_dry_run_and_proprio_observation(self):
        cfg = self.cfg()
        cfg.eval.real_robot.dry_run = True
        ns = inference_namespace()
        ns.update(cv2=cv2, Rotation=Rotation)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'start.png'
            cv2.imwrite(str(path), np.full((4, 4, 3), [30, 20, 10], np.uint8))
            cfg.eval.real_robot.dry_run_image_path = str(path)
            with patch('robopy.robots.xarm.XArmRobot') as owner:
                env = ns['XArmInferenceEnv'](cfg.eval.real_robot, cfg.plan_config)
            owner.assert_not_called()
            image = env.get_image()
            self.assertEqual(image.shape, (cfg.eval.real_robot.camera.height, cfg.eval.real_robot.camera.width, 3))
            np.testing.assert_array_equal(image[0, 0], [10, 20, 30])
            image[0, 0] = 0
            np.testing.assert_array_equal(env.get_image()[0, 0], [10, 20, 30])
            qpos, qvel, ee = env.get_robot_state()
            self.assertEqual((qpos.shape, qvel.shape, ee.shape), ((7,), (7,), (7,)))
            tree = ast.parse((ROOT / 'eval_real_robot.py').read_text())
            function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_policy_observation')
            exec(compile(ast.Module(body=[function], type_ignores=[]), 'eval_real_robot.py', 'exec'), ns)
            obs = ns['_policy_observation'](image=env.get_image(), goal=env.get_image(), ee=ee,
                                            gripper=0.5, goal_proprio=np.arange(8), step_idx=0,
                                            process={'proprio': None, 'goal_proprio': None})
            self.assertEqual(obs['proprio'].shape, (1, 1, 8))
            self.assertEqual(obs['goal_proprio'].shape, (1, 1, 8))
            self.assertEqual(obs['proprio'][0, 0, -1], 0.5)
            self.assertNotIn('wrist_pixels', obs)
            env.close()


if __name__ == '__main__':
    unittest.main()
