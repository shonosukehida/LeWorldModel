"""Hardware-free checks of G2 state reads and existing normalization."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
from scipy.spatial.transform import Rotation


def load_state_methods():
    path = Path(__file__).resolve().parents[1] / 'eval_real_robot.py'
    cls = next(n for n in ast.parse(path.read_text()).body
               if isinstance(n, ast.ClassDef) and n.name == 'XArmInferenceEnv')
    methods = [n for n in cls.body if isinstance(n, ast.FunctionDef)
               and n.name in ('get_robot_state', '_sdk_value')]
    node = ast.ClassDef(name='StateEnv', bases=[], keywords=[], body=methods,
                        decorator_list=[])
    ns = dict(np=np, Rotation=Rotation)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[node], type_ignores=[])),
                 str(path), 'exec'), ns)
    return ns['StateEnv']


class GripperG2StateTests(unittest.TestCase):
    def env(self):
        env = load_state_methods()()
        env.dry_run = False
        env.cfg = SimpleNamespace(gripper=SimpleNamespace(open_position=84., closed_position=0.))
        env._last_ee = np.array([0., 0., 0., 0., 0., 0., 1.], np.float32)
        env._follower = Mock()
        env._follower.get_joint_state.return_value = [0.] * 7 + [0.25]
        env._follower.get_ee_pos_quat.return_value = env._last_ee.copy()
        env._robot = Mock()
        env._robot.get_joint_states.return_value = (0, [[0.] * 7, [0.] * 7])
        env._robot.get_position.return_value = (0, [0.] * 6)
        return env

    def test_g2_read_normalization_and_clipping(self):
        for position, expected in ((84., 0.), (42., 0.5), (0., 1.), (100., 0.), (-10., 1.)):
            with self.subTest(position=position):
                env = self.env()
                env._robot.get_gripper_g2_position.return_value = (0, position)
                env.get_robot_state()
                self.assertEqual(env._last_gripper, expected)
                env._robot.get_gripper_g2_position.assert_called_once_with()
                env._robot.get_gripper_position.assert_not_called()

    def test_failed_reads_keep_follower_state(self):
        for result in ((1, 0.), AttributeError(), TypeError(), ValueError()):
            with self.subTest(result=result):
                env = self.env()
                if isinstance(result, Exception):
                    env._robot.get_gripper_g2_position.side_effect = result
                else:
                    env._robot.get_gripper_g2_position.return_value = result
                env.get_robot_state()
                self.assertEqual(env._last_gripper, 0.25)
                env._robot.get_gripper_position.assert_not_called()


if __name__ == '__main__':
    unittest.main()
