"""Hardware-free checks of evaluation config propagation to the G2 SDK."""
import threading
import unittest
from unittest.mock import Mock, patch

from omegaconf import OmegaConf
from robopy.robots.xarm.xarm_follower import XArmFollower

from test_xarm_admittance_config import ROOT, inference_namespace


class GripperWaitConfigTests(unittest.TestCase):
    def test_eval_wait_reaches_sdk(self):
        for setting in ('yaml', False, 'missing'):
            with self.subTest(setting=setting):
                cfg = OmegaConf.load(ROOT / 'config/eval/flip_mug.yaml')
                robot_cfg = cfg.eval.real_robot
                if setting == 'missing':
                    del robot_cfg.gripper.wait
                elif setting is False:
                    robot_cfg.gripper.wait = False
                expected = setting == 'yaml'
                if expected:
                    self.assertIs(robot_cfg.gripper.wait, True)
                robot_cfg.dry_run = False
                robot_cfg.dry_run_image_path = ''
                owner = Mock()
                owner.robot_system.follower._control_lock = threading.Lock()
                with patch('robopy.robots.xarm.XArmRobot', return_value=owner) as factory:
                    env = inference_namespace()['XArmInferenceEnv'](
                        robot_cfg, cfg.plan_config, use_camera=False,
                    )
                try:
                    follower_cfg = factory.call_args.args[0]
                    self.assertIs(follower_cfg.gripper_wait, expected)
                    follower = XArmFollower(follower_cfg)
                    follower._robot = Mock()
                    follower._set_gripper_position(42)
                    follower._robot.set_gripper_g2_position.assert_called_once_with(
                        42, speed=follower_cfg.gripper_speed,
                        force=follower_cfg.gripper_force, wait=expected,
                    )
                finally:
                    env.close()


if __name__ == '__main__':
    unittest.main()
