"""Robot ownership, initialization rollback, and two-view inference contracts."""
import ast
from collections import deque
from pathlib import Path
import signal
import subprocess
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import cv2
import h5py
import numpy as np
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from test_xarm_admittance_config import load_inference_class, ROOT
from robopy.config.robot_config import XArmConfig, XArmSensorParams
from robopy.config.sensor_config.params_config import CameraParams
from robopy.robots.xarm import XArmRobot
from robopy.robots.xarm.xarm_pair_sys import XArmPairSys
from robopy.sensors.visual.realsense_camera import RealsenseCamera
from robopy.config.sensor_config.visual_config.camera_config import RealsenseCameraConfig


def load_functions(names, namespace):
    tree = ast.parse((ROOT / 'eval_real_robot.py').read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=functions, type_ignores=[]), 'eval_real_robot.py', 'exec'), namespace)
    return namespace


class RobotLifecycleTests(unittest.TestCase):
    def robot(self):
        cfg = XArmConfig(sensors=XArmSensorParams(cameras=[
            CameraParams(name=name, serial_no=name, width=640, height=480, fps=30)
            for name in ('937622072677', '044322070202')]))
        robot = XArmRobot(cfg)
        robot._pair_sys = Mock(is_connected=False)
        return robot

    def test_follower_only_and_default_collection_connect(self):
        for connect_leader in (False, True):
            pair = XArmPairSys(XArmConfig())
            pair._leader = Mock()
            pair._follower = Mock()
            with patch.object(pair, '_align_leader_follower') as align:
                if connect_leader:
                    pair.connect()  # collection default stays unchanged
                else:
                    pair.connect(connect_leader=False)
                pair._follower.connect.assert_called_once_with()
                self.assertEqual(pair._leader.connect.call_count, int(connect_leader))
                self.assertEqual(align.call_count, int(connect_leader))
                pair.disconnect()
                self.assertEqual(pair._leader.disconnect.call_count, int(connect_leader))
                pair._follower.disconnect.assert_called_once_with()

    def test_partial_camera_initialization_and_interrupt_rollback(self):
        for failure in (RuntimeError('second camera failed'), KeyboardInterrupt()):
            robot = self.robot()
            first, second = Mock(), Mock()
            second.connect.side_effect = failure
            with patch('robopy.robots.xarm.xarm_robot.RealsenseCamera', side_effect=[first, second]):
                with self.assertRaises(type(failure)) as caught:
                    robot.connect(connect_leader=False)
            self.assertIs(caught.exception, failure)
            first.disconnect.assert_called_once_with()
            second.disconnect.assert_called_once_with()
            robot._pair_sys.disconnect.assert_called_once_with()
            self.assertEqual(robot.sensors.cameras, [])
            robot.disconnect()
            first.disconnect.assert_called_once_with()

    def test_successful_robot_connect_owns_both_cameras_once(self):
        robot = self.robot()
        first, second = Mock(), Mock()
        first.name, second.name = '937622072677', '044322070202'
        with patch('robopy.robots.xarm.xarm_robot.RealsenseCamera', side_effect=[first, second]) as factory:
            robot.connect(connect_leader=False)
            robot._pair_sys.connect.assert_called_once_with(connect_leader=False)
            self.assertEqual(factory.call_count, 2)
            self.assertEqual(robot.sensors.cameras, [first, second])
            for camera in (first, second):
                camera.connect.assert_called_once_with()
            robot._pair_sys.is_connected = True
            robot.connect(connect_leader=False)
            self.assertEqual(factory.call_count, 2)
            robot.disconnect()
            for camera in (first, second):
                camera.disconnect.assert_called_once_with()

    def test_arm_failure_still_rolls_back(self):
        robot = self.robot()
        robot._pair_sys.connect.side_effect = RuntimeError('arm failed')
        with patch.object(robot, '_init_sensors') as sensors:
            with self.assertRaisesRegex(RuntimeError, 'arm failed'):
                robot.connect(connect_leader=False)
        sensors.assert_not_called()
        robot._pair_sys.disconnect.assert_called_once_with()

    def test_camera_failure_after_pipeline_start_stops_pipeline(self):
        rs = Mock()
        pipeline = rs.pipeline.return_value
        sensor = pipeline.start.return_value.get_device.return_value.first_color_sensor.return_value
        sensor.set_option.side_effect = RuntimeError('exposure failed')
        with patch('robopy.sensors.visual.realsense_camera.rs', rs):
            camera = RealsenseCamera(RealsenseCameraConfig(serial_no='044322070202'))
            with self.assertRaisesRegex(ConnectionError, 'exposure failed'):
                camera.connect(warmup=False)
        pipeline.stop.assert_called_once_with()
        self.assertIsNone(camera.rs_pipeline)
        self.assertFalse(camera.is_connected)


class InferenceTests(unittest.TestCase):
    def cfg(self):
        cfg = OmegaConf.load(ROOT / 'config/eval/flip_mug.yaml')
        cfg.eval.real_robot.dry_run = False
        return cfg

    def test_managed_camera_order_color_and_configuration(self):
        cfg = self.cfg()
        owner = Mock()
        owner.robot_system.follower._control_lock = threading.Lock()
        cameras = []
        for serial, color in [('044322070202', (4, 5, 6)), ('937622072677', (1, 2, 3))]:
            cam = Mock()
            cam.name = serial
            cam.read.return_value = np.broadcast_to(np.asarray(color, np.float32)[:, None, None], (3, 480, 640))
            cameras.append(cam)
        owner.sensors.cameras = cameras  # deliberately reversed
        with patch('robopy.robots.xarm.XArmRobot', return_value=owner) as factory:
            env = load_inference_class()(cfg.eval.real_robot, cfg.plan_config)
        overhead, wrist = env.get_images()
        np.testing.assert_array_equal(overhead[0, 0], [1, 2, 3])
        np.testing.assert_array_equal(wrist[0, 0], [4, 5, 6])
        self.assertEqual(overhead.shape, (480, 640, 3))
        self.assertEqual(overhead.dtype, np.uint8)
        params = factory.call_args.args[0].sensors.cameras
        self.assertEqual([(p.serial_no, p.exposure, p.white_balance) for p in params],
                         [('937622072677', 120, 3300), ('044322070202', 70, 2800)])
        owner.connect.assert_called_once_with(connect_leader=False)
        owner.robot_system.follower.connect.assert_not_called()
        for camera in cameras:
            camera.connect.assert_not_called()
            camera.read.assert_called_once_with(specific_color='rgb')
        env.close()
        owner.disconnect.assert_called_once_with()
        for camera in cameras:
            camera.disconnect.assert_not_called()

    def test_owner_failure_and_kinematics_failure_cleanup(self):
        for stage in ('connect', 'kinematics'):
            cfg = self.cfg()
            owner = Mock()
            owner.robot_system.follower._control_lock = threading.Lock()
            error = RuntimeError(stage)
            if stage == 'connect':
                owner.connect.side_effect = error
            cls = load_inference_class()
            if stage == 'kinematics':
                cls.__init__.__globals__['XArm7IK'] = Mock(side_effect=error)
            with patch('robopy.robots.xarm.XArmRobot', return_value=owner):
                with self.assertRaisesRegex(RuntimeError, stage):
                    cls(cfg.eval.real_robot, cfg.plan_config)
            owner.disconnect.assert_called_once_with()

    def test_policy_setup_failure_cleanup(self):
        env = Mock()
        policy = Mock()
        policy.set_env.side_effect = RuntimeError('policy setup failed')
        ns = load_functions({'run_xarm_task', '_run_xarm_task_with_env'}, dict(XArmInferenceEnv=Mock(return_value=env)))
        with self.assertRaisesRegex(RuntimeError, 'policy setup failed'):
            ns['run_xarm_task'](self.cfg(), policy, {}, Path('/tmp'))
        env.close.assert_called_once_with()

    def test_dry_run_two_views_history_and_rollout_files(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cfg = self.cfg()
            real = cfg.eval.real_robot
            real.dry_run = True
            real.max_steps = 1
            real.non_interactive = True
            # Store BGR on disk and verify the policy receives RGB exactly once.
            for name, bgr in [('overhead', (30, 20, 10)), ('wrist', (60, 50, 40))]:
                path = root / f'{name}.png'
                cv2.imwrite(str(path), np.full((8, 8, 3), bgr, np.uint8))
                real['dry_run_image_path' if name == 'overhead' else 'dry_run_wrist_image_path'] = str(path)
            real.cameras.overhead.width = real.cameras.wrist.width = 16
            real.cameras.overhead.height = real.cameras.wrist.height = 16
            cls = load_inference_class()
            cls.__init__.__globals__['cv2'] = cv2
            cls.__init__.__globals__['Rotation'] = Rotation
            class Diffusion:
                obs_horizon = 2
                def set_env(self, env):
                    self.env = env
                def get_action(self, info):
                    self.info = info
                    return np.array([[[0.5, 0., .3, 0., 0., 0., 1., 0.]]], np.float32)
            policy = Diffusion()
            ns = dict(XArmInferenceEnv=cls, swm=SimpleNamespace(policy=SimpleNamespace(
                WorldModelPolicy=type('WM', (), {}), DiffusionPolicy=Diffusion, GPCPolicy=type('GPC', (), {}))),
                np=np, cv2=cv2, h5py=h5py, time=time, signal=signal, OmegaConf=OmegaConf,
                deque=deque, subprocess=subprocess, _load_or_capture_goal=lambda env, cfg:
                (*env.get_images(), np.zeros(8, np.float32)))
            ns = load_functions({'run_xarm_task', '_run_xarm_task_with_env', '_policy_observation'}, ns)
            with patch('robopy.robots.xarm.XArmRobot') as owner:
                ns['run_xarm_task'](cfg, policy, {}, root)
            owner.assert_not_called()
            self.assertEqual(policy.info['pixels'].shape, (1, 2, 16, 16, 3))
            np.testing.assert_array_equal(policy.info['pixels'][0, 0, 0, 0], [10, 20, 30])
            np.testing.assert_array_equal(policy.info['wrist_pixels'][0, 0, 0, 0], [40, 50, 60])
            with h5py.File(root / 'rollout.h5', 'r') as file:
                self.assertEqual(file['pixels'].shape, (1, 16, 16, 3))
                self.assertEqual(file['wrist_pixels'].shape, (1, 16, 16, 3))
            for name in ('rollout_overhead.mp4', 'rollout_wrist.mp4'):
                self.assertGreater((root / name).stat().st_size, 0)
            # Exercise the actual rollout exception path, including Ctrl-C.
            for error in (RuntimeError('inference failed'), KeyboardInterrupt()):
                policy.get_action = Mock(side_effect=error)
                env = cls(real, cfg.plan_config)
                previous_handler = signal.getsignal(signal.SIGINT)
                with patch.object(env, 'close', wraps=env.close) as close:
                    with self.assertRaises(type(error)) as caught:
                        ns['_run_xarm_task_with_env'](cfg, policy, {}, root, env)
                    self.assertIs(caught.exception, error)
                    close.assert_called_once_with()
                self.assertEqual(signal.getsignal(signal.SIGINT), previous_handler)


if __name__ == '__main__':
    unittest.main()
