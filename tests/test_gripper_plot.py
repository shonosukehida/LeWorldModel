"""Hardware-free gripper plot generation and step alignment checks."""
import ast
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def load_plot():
    path = Path(__file__).resolve().parents[1] / 'eval_real_robot.py'
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'plot_commanded_vs_actual_gripper')
    ns = dict(np=np, plt=plt)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), ns)
    return ns[node.name]


class GripperPlotTests(unittest.TestCase):
    def test_alignment_nan_and_png(self):
        commands = np.zeros((4, 8))
        commands[:, 7] = [0., 1., .4, .8]
        actual = [0., .2, np.nan, .5]
        fig, ax = plt.subplots()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'commanded_vs_actual_gripper.png'
            with patch.object(plt, 'subplots', return_value=(fig, ax)):
                load_plot()(commands, actual, path)
            self.assertGreater(path.stat().st_size, 0)
            np.testing.assert_array_equal(ax.lines[0].get_xdata(), [1, 2, 3])
            np.testing.assert_allclose(ax.lines[0].get_ydata(), [0., 1., .4])
            np.testing.assert_allclose(ax.lines[1].get_ydata(), [.2, np.nan, .5], equal_nan=True)
            self.assertFalse(plt.fignum_exists(fig.number))

    def test_short_missing_and_joint_only_rollouts(self):
        with tempfile.TemporaryDirectory() as tmp:
            for n in (1, 3):
                path = Path(tmp) / f'{n}.png'
                load_plot()(np.zeros((n, 8)), np.full(n, np.nan), path)
                self.assertTrue(path.exists())
            path = Path(tmp) / 'skip.png'
            for actions, actual in (([], []), (np.zeros((2, 7)), [np.nan, np.nan])):
                load_plot()(actions, actual, path)
                self.assertFalse(path.exists())


if __name__ == '__main__':
    unittest.main()
