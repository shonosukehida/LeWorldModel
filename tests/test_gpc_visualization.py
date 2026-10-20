"""Run actual GPC selection source with mock DP/WM; no robot imports."""
import ast
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("gpc_visualization", ROOT / "utils/gpc_visualization.py")
vis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(vis)
from normalization_stats import SafeStandardScaler


def actual_gpc_class():
    # Locate installed source without importing its optional simulation stack.
    spec = importlib.util.find_spec("stable_worldmodel")
    path = Path(spec.origin).parent / "policy.py"
    node = next(n for n in ast.parse(path.read_text()).body
                if isinstance(n, ast.ClassDef) and n.name == "GPCPolicy")
    ns = dict(torch=torch, np=np, BasePolicy=object, DiffusionPolicy=object)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    return ns["GPCPolicy"]


def make_policy():
    scaler = SafeStandardScaler()
    scaler.mean_ = np.arange(8, dtype=np.float32)[None] + 2
    scaler.scale_ = np.arange(8, dtype=np.float32)[None] + .5
    physical = torch.arange(1*4*6*8, dtype=torch.float32).reshape(1,4,6,8) / 30
    dp = Mock(obs_horizon=3, action_horizon=8,
              process={"action_cartesian": scaler})
    dp.sample_action_sequences = Mock(return_value=physical)
    wm_scaler = SafeStandardScaler()
    wm_scaler.mean_, wm_scaler.scale_ = np.zeros((1,8)), np.full((1,8), 10.)
    reward = Mock(return_value=torch.tensor([[1., 4., 4., -2.]]))
    policy = actual_gpc_class()(dp, None, reward, num_candidates=4,
                               process={"action_cartesian": wm_scaler})
    policy.rollout = Mock(return_value={})
    return policy, physical


class VisualizationTests(unittest.TestCase):
    def test_capture_actual_selection_and_roundtrip(self):
        policy, physical = make_policy()
        baseline = policy.get_action({})
        policy.diffusion_policy.sample_action_sequences.reset_mock()
        policy.reward_fn.reset_mock()
        original_sample = policy.diffusion_policy.sample_action_sequences
        original_reward = policy.reward_fn
        with vis.GPCRecorder(policy) as recorder:
            recorder.begin_step(7)
            returned = policy.get_action({})
            # Reinforcement hold/no new inference: no duplicate record.
            recorder.begin_step(8)
            self.assertEqual(len(recorder.records), 1)
        self.assertIs(policy.reward_fn, original_reward)
        self.assertIs(policy.diffusion_policy.sample_action_sequences, original_sample)
        original_sample.assert_called_once_with({}, num_samples=4, denormalize=True)
        original_reward.assert_called_once()
        np.testing.assert_array_equal(returned, baseline)
        record = recorder.records[0]
        self.assertEqual(record['best_idx'], 1)
        self.assertEqual(record['candidate_actions_physical'].shape, (4,4,8))
        np.testing.assert_array_equal(record['candidate_actions_physical'][1,0], returned[0])
        np.testing.assert_array_equal(physical.numpy()[0,:,2:], record['candidate_actions_physical'])
        with tempfile.TemporaryDirectory() as tmp:
            recorder.save(tmp)
            with np.load(Path(tmp)/'gpc/data/candidates.npz', allow_pickle=False) as data:
                self.assertEqual(data['normalization_space'], 'DP training normalization')
                self.assertEqual(data['step_idx'].tolist(), [7])
                normalized = data['record_0000_candidate_actions']
                self.assertEqual(normalized.shape, (4,4,8))
                restored = recorder.processor.inverse_transform(normalized.reshape(-1,8))
                np.testing.assert_allclose(restored, physical.numpy()[0,:,2:].reshape(-1,8), atol=1e-6)
                self.assertEqual(data['record_0000_action_horizon'], 4)
            self.assertTrue((Path(tmp)/'gpc/gpc_actions/step_0007.png').exists())

    def test_plot_rank_colors_xyz_and_ties(self):
        actions = np.arange(4*5*8).reshape(4,5,8)
        rewards = np.array([1., 4., 4., -2.])
        np.testing.assert_array_equal(vis.evaluation_ranks(rewards), [3,1,2,4])
        fig, axes = vis.plot_candidates(actions, rewards, 1, 7)
        try:
            self.assertEqual(len(axes), 3)
            for i, ax in enumerate(axes):
                self.assertEqual(ax.get_ylabel(), f"{'XYZ'[i]} (normalized)")
                np.testing.assert_array_equal(ax.get_ylim(), [-1, 1])
                np.testing.assert_array_equal(ax.get_yticks(), [-1., -.5, 0., .5, 1.])
                self.assertEqual([t.get_text().replace("−", "-") for t in ax.get_yticklabels()],
                                 ["-1.0", "-0.5", "0.0", "0.5", "1.0"])
                selected = ax.lines[-1]
                self.assertEqual(selected.get_color(), '#8B0000')
                self.assertGreater(selected.get_linewidth(), ax.lines[0].get_linewidth())
                np.testing.assert_array_equal(selected.get_ydata(), actions[1,:,i])
                # Remaining candidates retain index order: 0,2,3.
                self.assertLess(ax.lines[1].get_color()[0], ax.lines[0].get_color()[0])
                self.assertLess(ax.lines[0].get_color()[0], ax.lines[2].get_color()[0])
                self.assertLess(ax.lines[2].get_color()[0], 1.)
            for cb_ax in fig.axes[3:]:
                self.assertEqual(cb_ax.get_ylabel(), 'WM Evaluation Rank')
        finally:
            plt.close(fig)

    def test_exception_restores_inherited_method_and_instance_reward(self):
        policy, _ = make_policy()
        dp = policy.diffusion_policy
        class DP:
            obs_horizon = dp.obs_horizon
            process = dp.process
            def sample_action_sequences(self, *args, **kwargs):
                raise RuntimeError('mock failure')
        policy.diffusion_policy = DP()
        reward = policy.reward_fn
        with self.assertRaisesRegex(RuntimeError, 'mock failure'):
            with vis.GPCRecorder(policy) as recorder:
                recorder.begin_step(0)
                policy.get_action({})
        self.assertNotIn('sample_action_sequences', vars(policy.diffusion_policy))
        self.assertIs(policy.reward_fn, reward)
        self.assertEqual(recorder.records, [])

    def test_rollout_enabled_disabled_and_reinforcement_hold(self):
        from test_gripper_reinforcement import namespace, FakeEnv, Clock, OmegaConf
        for enabled in (False, True):
            with self.subTest(enabled=enabled), tempfile.TemporaryDirectory() as tmp:
                policy, _ = make_policy()
                policy.set_env = Mock()
                original_sample = policy.diffusion_policy.sample_action_sequences
                original_reward = policy.reward_fn
                cfg = OmegaConf.create({
                    'gpc': {'visualization_enabled': enabled},
                    'plan_config': {'action_space': 'cartesian'},
                    'eval': {'real_robot': {'max_steps': 3, 'control_hz': 10,
                        'non_interactive': True,
                        'gripper': {'open_position': 84., 'closed_position': 0.},
                        'gripper_reinforcement': {'enabled': True, 'threshold': .2,
                            'target_opening_mm': 20., 'duration_sec': 1.}}}})
                env = FakeEnv()
                ns = namespace(Clock())
                ns.update(__file__=str(ROOT/'eval_real_robot.py'), Path=Path)
                ns['swm'].policy.GPCPolicy = type(policy)
                ns['_run_xarm_task_with_env'](cfg, policy, {}, Path(tmp), env)
                env.close.assert_called_once()
                self.assertIs(policy.diffusion_policy.sample_action_sequences, original_sample)
                self.assertIs(policy.reward_fn, original_reward)
                self.assertEqual(original_sample.call_count, 1)
                self.assertEqual(original_reward.call_count, 1)
                archive = Path(tmp)/'gpc/data/candidates.npz'
                self.assertEqual(archive.exists(), enabled)
                if enabled:
                    with np.load(archive) as data:
                        self.assertEqual(data['step_idx'].tolist(), [0])
                    vis.render_archive(archive, Path(tmp)/'replot')
                    self.assertTrue((Path(tmp)/'replot/step_0000.png').exists())

    def test_reward_failure_restores(self):
        policy, _ = make_policy()
        policy.reward_fn.side_effect = RuntimeError('reward failed')
        reward = policy.reward_fn
        with self.assertRaisesRegex(RuntimeError, 'reward failed'):
            with vis.GPCRecorder(policy) as recorder:
                recorder.begin_step(0)
                policy.get_action({})
        self.assertIs(policy.reward_fn, reward)
        self.assertEqual(recorder.records, [])


if __name__ == '__main__':
    unittest.main()
