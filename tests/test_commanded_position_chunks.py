"""Exercise the real plotting blocks without importing robot dependencies."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


SOURCE = Path(__file__).resolve().parents[1] / "eval_real_robot.py"


def plotting_code():
    source = SOURCE.read_text()
    start = source.index('        # Only uninterrupted DP/GPC execution')
    end = source.index('        # RobopyとxArm SDK', start)
    import textwrap
    return compile(ast.parse(textwrap.dedent(source[start:end])), str(SOURCE), "exec")


class CommandedPositionChunkTests(unittest.TestCase):
    def test_lines_alignment_sources_and_fallbacks(self):
        for steps, buffer, dp_reinforcement, gpc_reinforcement, n in (
            (8, True, False, False, 25),
            (8, True, False, False, 18),  # singleton final chunk
            (1, True, False, False, 25),
            (8, True, True, False, 25),
            (8, True, False, True, 25),
            (8, False, False, False, 25),
        ):
            with self.subTest(steps=steps, buffer=buffer, dp=dp_reinforcement,
                              gpc=gpc_reinforcement, n=n):
                records = {
                    "commanded_action": np.arange(n * 8).reshape(n, 8),
                    "ee_pos_quat": np.arange(n * 7).reshape(n, 7) + 1000,
                    "safe_qpos": np.arange(n * 7).reshape(n, 7) + 2000,
                    "sdk_qpos": np.arange(n * 7).reshape(n, 7) + 3000,
                    "qpos": np.full((n, 7), -999),
                }
                ns = dict(np=np, plt=plt, records=records, run_dir=Path('/unused'),
                          real_cfg=SimpleNamespace(dp_execute_steps=steps),
                          action_buffer=object() if buffer else None,
                          reinforcement_enabled=dp_reinforcement,
                          gpc_reinforcement_enabled=gpc_reinforcement)
                figures = []
                try:
                    with patch.object(plt, 'close', side_effect=figures.append), \
                         patch('matplotlib.figure.Figure.savefig'), patch('builtins.print'):
                        exec(plotting_code(), ns)
                    self.assertEqual([len(fig.axes) for fig in figures], [3, 7])
                    split = steps > 1 and buffer and not dp_reinforcement and not gpc_reinforcement
                    size = steps if split else n - 1
                    for fig, command_key, actual_key in zip(
                        figures, ('commanded_action', 'safe_qpos'), ('ee_pos_quat', 'sdk_qpos')
                    ):
                        for i, ax in enumerate(fig.axes):
                            commands, actual = ax.lines[:-1], ax.lines[-1]
                            self.assertEqual(len(commands), (n - 2) // size + 1)
                            for line, start in zip(commands, range(0, n - 1, size)):
                                end = min(start + size, n - 1)
                                np.testing.assert_array_equal(line.get_xdata(), np.arange(start + 1, end + 1))
                                np.testing.assert_array_equal(line.get_ydata(), records[command_key][start:end, i])
                                self.assertEqual(line.get_color(), 'tab:blue')
                                if end - start == 1:
                                    self.assertEqual(line.get_marker(), '.')
                            np.testing.assert_array_equal(actual.get_xdata(), np.arange(1, n))
                            np.testing.assert_array_equal(actual.get_ydata(), records[actual_key][1:, i])
                            self.assertEqual(actual.get_color(), 'tab:orange')
                            self.assertEqual(len(ax.get_legend_handles_labels()[1]), 2)
                finally:
                    for fig in figures:
                        plt.close(fig)


if __name__ == '__main__':
    unittest.main()
