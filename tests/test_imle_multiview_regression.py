"""Exact integration-vs-multiview comparisons, without hardware imports.

Git objects are pinned to the integration bases; both repositories must retain
these commits. Reward uses the same approved local prerequisite on both sides.
"""
import ast
import subprocess
import tempfile
from pathlib import Path
import unittest
import numpy as np
import torch
import test_dp_action_chunks as helper

ROOT = Path(__file__).resolve().parents[1]
SWM = ROOT.parent / 'stable-worldmodel'
LWM_BASE = 'cc06f7de7e1bc5dc6ee2440fca8131d39215a772'
SWM_BASE = 'ab3e681039977bcd978989082f0b7654a90ef07b'


def baseline(repo, commit, path):
    return subprocess.check_output(['git', '-C', str(repo), 'show', f'{commit}:{path}'], text=True)


class MultiviewRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.numerical = helper.NumericalTests()
        helper.NumericalTests.setUpClass()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'policy.py'
            path.write_text(baseline(SWM, SWM_BASE, 'stable_worldmodel/policy.py'))
            cls.numerical.old = helper.classes(path)
        cls.loop = helper.LoopTests()
        cls.old_source = baseline(ROOT, LWM_BASE, 'eval_real_robot.py')

    def test_dp_full_chunks_and_rng_exact(self):
        old, new = (self.numerical.dp(v) for v in (self.numerical.old, self.numerical.new))
        obs = {'pixels': torch.linspace(0, 1, 2*3*32*32).reshape(1, 2, 3, 32, 32)}
        for seed in (0, 42, 99):
            torch.manual_seed(seed)
            expected = old.get_action_chunk(obs)
            state = torch.get_rng_state()
            torch.manual_seed(seed)
            actual = new.get_action_chunk(obs)
            np.testing.assert_array_equal(actual, expected)
            self.assertEqual(actual.dtype, expected.dtype)
            self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_gpc_candidates_wm_inputs_rewards_selection_exact(self):
        self.numerical.test_gpc_candidates_rollout_rewards_selection()

    def test_chunks_gripper_histories_logs_exact(self):
        for policy in (helper.Diffusion, helper.GPC):
            for steps in (1, 3):
                for enabled in (False, True):
                    for stop in (None, 2):
                        with self.subTest(policy=policy.__name__, steps=steps, enabled=enabled, stop=stop):
                            args = dict(steps=steps, enabled=enabled, stop=stop)
                            old = self.loop.run_loop(policy, self.old_source, **args)
                            new = self.loop.run_loop(policy, **args)
                            self.assertEqual(old[2], new[2])
                            np.testing.assert_array_equal(old[0].sent, new[0].sent)
                            self.assertEqual(set(old[3]), set(new[3]))
                            for key in old[3]:
                                np.testing.assert_array_equal(old[3][key], new[3][key])
                            for a, b in zip(old[1].observations, new[1].observations):
                                for key in a:
                                    np.testing.assert_array_equal(a[key], b[key])

    def test_legacy_modules_and_dp_loader_unchanged(self):
        for path in ('normalization_stats.py', 'action_normalization.py',
                     'offline_dp_eval.py', 'offline_gpc_eval.py',
                     'offline_latent_dist_eval.py', 'offline_real_robot_latent_eval.py',
                     'real_robot_latent_visualization.py', 'utils/gpc_visualization.py'):
            self.assertEqual((ROOT / path).read_text(), baseline(ROOT, LWM_BASE, path), path)
        def definition(source, name):
            tree = ast.parse(source)
            node = next(n for n in tree.body if getattr(n, 'name', None) == name)
            return ast.dump(node, include_attributes=False)
        for name in ('load_diffusion_policy', '_DPActionBuffer',
                     '_reinforced_diffusion_action', '_reinforced_gpc_action'):
            self.assertEqual(definition(self.old_source, name), definition((ROOT / 'eval_real_robot.py').read_text(), name))
        old = ast.parse(baseline(SWM, SWM_BASE, 'stable_worldmodel/policy.py'))
        new = ast.parse((SWM / 'stable_worldmodel/policy.py').read_text())
        for name in ('BasePolicy', 'DiffusionPolicy'):
            a = next(n for n in old.body if getattr(n, 'name', None) == name)
            b = next(n for n in new.body if getattr(n, 'name', None) == name)
            self.assertEqual(ast.dump(a), ast.dump(b))


IMLE_EXECUTION_GUARD = """            imle_source = getattr(policy, "action_policy", policy)
            if getattr(imle_source, "type", None) == "imle":
                if str(cfg.plan_config.action_space) != "cartesian":
                    raise ValueError("IMLE requires Cartesian action execution")
                from imle_runtime import validate_imle_action
                validate_imle_action(action, env, ee)
"""


def without_imle_execution_guard(source):
    # Remove only the exact reviewed addition, retaining every legacy statement.
    assert source.count(IMLE_EXECUTION_GUARD) == 1
    return source.replace(IMLE_EXECUTION_GUARD, "")
