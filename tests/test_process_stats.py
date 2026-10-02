"""Hardware-free CLI, serialization and evaluation control-flow tests."""
import ast
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from create_process_stats import create_process_stats
from normalization_stats import load_normalization_process


class Config(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


class ProcessStatsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = self.root / 'push.h5'
        self.output = self.root / 'nested/process_stats.npz'
        self.actions = np.array([[1., 8.], [3., 8.], [5., 8.], [np.nan, 999.]])
        with h5py.File(self.dataset, 'w') as file:
            file['action_cartesian'] = self.actions
            file['proprio'] = np.array([[2., 3.], [4., 5.], [6., 7.]])

    def test_cli_round_trip(self):
        result = subprocess.run([sys.executable, str(ROOT / 'create_process_stats.py'),
                                 '--dataset', str(self.dataset), '--output', str(self.output)],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        process, action_key = load_normalization_process(self.output)
        self.assertEqual(action_key, 'action_cartesian')
        self.assertEqual(set(process), {'action_cartesian', 'proprio', 'goal_proprio'})
        self.assertIs(process['proprio'], process['goal_proprio'])
        action = process[action_key]
        np.testing.assert_allclose(action.mean_, [[3., 8.]])
        np.testing.assert_allclose(action.scale_, [[2., 1.]])
        np.testing.assert_allclose(action.raw_min_, [[1., 8.]])
        np.testing.assert_allclose(action.raw_max_, [[5., 8.]])
        np.testing.assert_allclose(action.normed_min_, [[-1., 0.]])
        np.testing.assert_allclose(action.normed_max_, [[1., 0.]])
        np.testing.assert_allclose(action.inverse_transform(action.transform(self.actions[:3])), self.actions[:3])
        with np.load(self.output, allow_pickle=False) as stats:
            self.assertEqual(json.loads(stats['metadata'].item()),
                             dict(version=1, keys=['action_cartesian', 'proprio'], action_key='action_cartesian'))
            for index in range(2):
                for name in ('mean', 'scale', 'raw_min', 'raw_max', 'normed_min', 'normed_max', 'eps'):
                    self.assertIn(f'processor_{index}_{name}', stats)

    def test_errors_leave_no_output(self):
        with self.assertRaisesRegex(FileNotFoundError, str(self.root / 'missing.h5')):
            create_process_stats(self.root / 'missing.h5', self.output)
        with self.assertRaisesRegex(ValueError, str(self.dataset)):
            create_process_stats(self.dataset, self.dataset)
        for column in ('proprio', 'action_cartesian'):
            with h5py.File(self.dataset, 'a') as file:
                data = file[column][:]
                del file[column]
            with self.assertRaisesRegex(KeyError, column):
                create_process_stats(self.dataset, self.output)
            self.assertFalse(self.output.exists())
            with h5py.File(self.dataset, 'a') as file:
                file[column] = data
        for values, message in [(np.full((2, 2), np.nan), 'No valid samples'),
                                 (np.ones((1, 2)), 'two valid rows'),
                                 (np.full((2, 2), np.inf), 'infinite')]:
            with h5py.File(self.dataset, 'a') as file:
                del file['proprio']
                file['proprio'] = values
            with self.assertRaisesRegex(ValueError, message):
                create_process_stats(self.dataset, self.output)
            self.assertFalse(self.output.exists())

    def test_overwrite_and_hardlink(self):
        create_process_stats(self.dataset, self.output)
        before = self.output.read_bytes()
        with self.assertRaisesRegex(FileExistsError, str(self.output)):
            create_process_stats(self.dataset, self.output, overwrite=False)
        self.assertEqual(before, self.output.read_bytes())
        link = self.root / 'linked.h5'
        link.hardlink_to(self.dataset)
        with self.assertRaisesRegex(ValueError, str(self.dataset)):
            create_process_stats(self.dataset, link)

    def test_diffusion_evaluation_without_datasets_or_world_checkpoint(self):
        create_process_stats(self.dataset, self.output)
        # A present training dataset must also be ignored when statistics exist.
        (self.root / 'datasets').mkdir()
        (self.root / 'datasets/push.h5').hardlink_to(self.dataset)
        tree = ast.parse((ROOT / 'eval_real_robot.py').read_text())
        run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
        # Execute the actual normalization/policy/probing block with dependency mocks.
        start = next(i for i, n in enumerate(run.body) if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'dataset_name' for t in n.targets))
        cfg = Config(cache_dir=str(self.root), policy='nonexistent/world_checkpoint', policy_type='diffusion',
                     eval=Config(dataset_name='push', normalization_stats_path=str(self.output),
                                 real_robot=Config(execute=False), probing=Config(exe_probe=False)),
                     gpc=Config(diffusion_checkpoint='diffusion.ckpt'))
        get_dataset = Mock(side_effect=AssertionError('Dataset must not be loaded'))
        auto_model = Mock(side_effect=AssertionError('World checkpoint must not be loaded'))
        diffusion = Mock(return_value=object())
        namespace = dict(cfg=cfg, Path=Path, transform={}, results_path=self.root,
                         swm=Config(policy=Config(AutoCostModel=auto_model)),
                         get_dataset=get_dataset, load_normalization_process=load_normalization_process,
                         load_diffusion_policy=diffusion, img_transform=Mock(return_value='images'))
        exec(compile(ast.Module(body=run.body[start:], type_ignores=[]), 'eval_real_robot.py', 'exec'), namespace)
        get_dataset.assert_not_called()
        auto_model.assert_not_called()
        diffusion.assert_called_once_with(checkpoint_path='diffusion.ckpt', image_transform='images', device='cuda')

    def test_training_action_statistics_match(self):
        import torch
        from normalization_stats import build_normalization_process
        tree = ast.parse((ROOT / 'train_policy.py').read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                        and n.name == 'get_action_stats')
        namespace = dict(torch=torch, np=np)
        exec(compile(ast.Module(body=[function], type_ignores=[]), 'train_policy.py', 'exec'), namespace)
        dataset = Mock()
        dataset.get_col_data.return_value = self.actions.astype(np.float32)
        mean, std = namespace['get_action_stats'](dataset, 'action_cartesian')
        process, _ = build_normalization_process(dataset, ['action_cartesian'])
        np.testing.assert_allclose(process['action_cartesian'].mean_, mean.numpy(), rtol=1e-6)
        np.testing.assert_allclose(process['action_cartesian'].scale_, std.numpy(), rtol=1e-6)

    def test_checkpoint_action_statistics_are_still_used(self):
        source = (ROOT / 'eval_real_robot.py').read_text()
        tree = ast.parse(source)
        loader = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'load_diffusion_policy')
        assignments = {ast.unparse(n.targets[0]): ast.unparse(n.value)
                       for n in ast.walk(loader) if isinstance(n, ast.Assign)}
        self.assertIn("checkpoint['action_mean']", assignments['action_processor.mean_'])
        self.assertIn("checkpoint['action_std']", assignments['action_processor.scale_'])


if __name__ == '__main__':
    unittest.main()
