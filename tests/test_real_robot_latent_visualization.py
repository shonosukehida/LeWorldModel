"""Post-save hook tests without importing robot modules or contacting hardware."""
import ast
from copy import deepcopy
from pathlib import Path
import random
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import h5py
import numpy as np
from omegaconf import OmegaConf
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import real_robot_latent_visualization as auto
from offline_real_robot_latent_eval import fingerprint
import test_offline_real_robot_latent_eval as fixtures


def config():
    return OmegaConf.create(dict(policy_type='gpc', plan_config={}, eval=dict(real_robot=dict(
        execute=True, dry_run=False, latent_visualization=dict(enabled=True)))))


class AutomaticTests(unittest.TestCase):
    def test_gate(self):
        cfg = config()
        cases = [('policy_type', 'diffusion'), ('policy_type', 'world_model'),
                 ('eval.real_robot.execute', False), ('eval.real_robot.dry_run', True),
                 ('eval.real_robot.latent_visualization.enabled', False)]
        for key, value in cases:
            variant = deepcopy(cfg)
            OmegaConf.update(variant, key, value)
            with patch.object(auto, '_evaluate_saved') as evaluate:
                self.assertIsNone(auto.run_after_save(variant, Mock(), '/unused'))
                evaluate.assert_not_called()
        del cfg.eval.real_robot.latent_visualization
        with patch.object(auto, '_evaluate_saved') as evaluate:
            auto.run_after_save(cfg, Mock(), '/unused')
            evaluate.assert_not_called()
        with patch.object(auto, '_evaluate_saved', return_value='output') as evaluate:
            self.assertEqual(auto.run_after_save(config(), Mock(), '/saved'), 'output')
            evaluate.assert_called_once()

    def wrapper(self, cfg, control, close, after):
        tree = ast.parse((ROOT / 'eval_real_robot.py').read_text())
        node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run_xarm_task')
        namespace = dict(OmegaConf=OmegaConf, XArmInferenceEnv=lambda *_: SimpleNamespace(close=close),
                         _run_xarm_task_with_env=control)
        exec(compile(ast.Module(body=[node], type_ignores=[]), 'eval_real_robot.py', 'exec'), namespace)
        with patch.object(auto, 'run_after_save', side_effect=after):
            return namespace['run_xarm_task'](cfg, Mock(), {}, Path('/saved'))

    def test_hook_after_save_and_robot_close(self):
        events = []
        result = self.wrapper(config(), lambda *args: events.append('saved') or Path('/saved'),
                              lambda: events.append('closed'), lambda *args: events.append('latent'))
        self.assertEqual(events, ['saved', 'closed', 'latent'])
        self.assertEqual(result, Path('/saved'))

    def test_wrapper_disabled_or_missing_keeps_legacy_order(self):
        for missing in (False, True):
            cfg = config()
            if missing:
                del cfg.eval.real_robot.latent_visualization
            else:
                cfg.eval.real_robot.latent_visualization.enabled = False
            events = []
            self.wrapper(cfg, lambda *args: events.append('saved'),
                         lambda: events.append('closed'), lambda *args: events.append('latent'))
            self.assertEqual(events, ['saved', 'closed'])

    def test_auxiliary_exception_cannot_mask_control_exception(self):
        def control(*args):
            raise RuntimeError('original IK failure')
        def after(*args):
            raise ValueError('auxiliary failure')
        with patch('sys.stderr'), patch('builtins.print'):
            with self.assertRaisesRegex(RuntimeError, 'original IK failure'):
                self.wrapper(config(), control, lambda: None, after)

    def test_control_exception_propagates_after_postprocessing(self):
        events = []
        def control(*args):
            events.append('saved_partial')
            raise RuntimeError('IK control failure')
        with self.assertRaisesRegex(RuntimeError, 'IK control failure'):
            self.wrapper(config(), control, lambda: events.append('closed'),
                         lambda *args: events.append('latent'))
        self.assertEqual(events, ['saved_partial', 'closed', 'latent'])

    def test_failure_reports_traceback_and_preserves_files_and_rng(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'rollout.h5'
            source.write_bytes(b'existing saved data')
            before = fingerprint(source)
            state = random.getstate()
            def fail(*args):
                random.random()
                raise ValueError('bad goal shape')
            with patch.object(auto, '_evaluate_saved', side_effect=fail), patch('sys.stderr') as stderr, patch('builtins.print') as output:
                self.assertIsNone(auto.run_after_save(config(), Mock(), directory))
                self.assertIn('ValueError: bad goal shape', output.call_args_list[0].args[0])
                self.assertTrue(any('Traceback' in str(call) for call in output.call_args_list))
            self.assertEqual(before, fingerprint(source))
            self.assertEqual(state, random.getstate())

    def test_real_automatic_evaluation_isolates_loaded_model(self):
        fixture = fixtures.RolloutTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        class CachingModel(fixtures.Model):
            def __init__(self):
                super().__init__()
                self.register_buffer('encode_calls', torch.zeros(1))

            def encode(self, info):
                self.encode_calls.add_(1)
                return super().encode(info)

        fixture.model = CachingModel().eval()
        from stable_worldmodel.policy import GPCPolicy
        from utils import get_eval_img_preprocessor
        from normalization_stats import build_normalization_process, save_normalization_process
        stats_path = fixture.root / 'process_stats.npz'
        dataset = SimpleNamespace(get_col_data=lambda _: np.stack([np.zeros(8), np.ones(8)]).astype(np.float32))
        process, action_key = build_normalization_process(dataset, ['action_cartesian', 'proprio'])
        save_normalization_process(stats_path, process, action_key)
        transform = get_eval_img_preprocessor(8)
        policy = GPCPolicy(SimpleNamespace(action_horizon=1), fixture.model, None,
                           process=process, transform={key: transform for key in
                           ('pixels', 'wrist_pixels', 'goal', 'goal_wrist_pixels')})
        cfg = config()
        cfg.eval.normalization_stats_path = str(stats_path)
        cfg.eval.img_size = 8
        checkpoint = fixture.root / 'synthetic_object.ckpt'
        checkpoint.write_bytes(b'synthetic checkpoint provenance')
        self.assertEqual(auto.checkpoint_path(fixture.root / 'synthetic'), checkpoint)
        policy._latent_visualization_metadata = auto.capture_metadata(cfg, process, checkpoint)
        self.assertEqual(policy._latent_visualization_metadata['world_model_checkpoint'], fingerprint(checkpoint))
        params = {k: v.clone() for k, v in policy.world_model.state_dict().items()}
        rng = torch.random.get_rng_state().clone()
        before = fingerprint(fixture.source)
        output = auto.run_after_save(cfg, policy, fixture.root)
        self.assertIsNotNone(output)
        self.assertEqual(before, fingerprint(fixture.source))
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        for key, value in params.items():
            self.assertTrue(torch.equal(value, policy.world_model.state_dict()[key]))
        with h5py.File(output / 'latent_distance.h5') as file:
            self.assertEqual(file['observed_latent_distance'].shape, (3,))
            np.testing.assert_allclose(file['observed_latent_distance'][:],
                                       np.array([5, 0, 10]) / process['proprio'].scale_[0, 0], rtol=1e-6)
        # Repeated automatic evaluation refreshes the same output directory.
        self.assertEqual(auto.run_after_save(cfg, policy, fixture.root), output)
        self.assertEqual(before, fingerprint(fixture.source))


if __name__ == '__main__':
    unittest.main()
