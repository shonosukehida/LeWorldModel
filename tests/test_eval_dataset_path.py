"""Hardware-free dataset resolution and normalization regression checks.

Run with pytest or unittest. Only temporary HDF5/statistics files are written.
"""
import ast
import copy
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'eval_real_robot.py'


def tree(baseline=False):
    source = subprocess.check_output(
        ['git', 'show', 'HEAD:eval_real_robot.py'], cwd=ROOT, text=True
    ) if baseline else SOURCE.read_text()
    return ast.parse(source)


def function(name, namespace, baseline=False):
    node = copy.deepcopy(next(n for n in tree(baseline).body
                              if isinstance(n, ast.FunctionDef) and n.name == name))
    node.decorator_list = []
    node.returns = None
    for arg in node.args.args:
        arg.annotation = None
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace[name]


def normalization(namespace, baseline=False):
    run = next(n for n in tree(baseline).body
               if isinstance(n, ast.FunctionDef) and n.name == 'run')
    start = next(i for i, n in enumerate(run.body)
                 if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'dataset_name')
    end = next(i for i, n in enumerate(run.body)
               if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == 'required_process_keys')
    exec(compile(ast.Module(body=run.body[start:end], type_ignores=[]), str(SOURCE), 'exec'), namespace)
    return namespace['dataset_path']


class DatasetPathTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.cfg = NS(cache_dir=str(self.root), dataset=NS(keys_to_cache=['action_cartesian', 'proprio']),
                      eval=NS(dataset_name='flip_mug/test/push',
                              normalization_stats_path=str(self.root / 'process_stats.npz')))

    def namespace(self, constructor=None):
        return dict(Path=Path, cfg=self.cfg,
                    swm=NS(data=NS(HDF5Dataset=constructor if constructor is not None else Mock(),
                                   utils=NS(get_cache_dir=lambda: self.root))),
                    load_normalization_process=Mock(return_value=({}, 'action_cartesian')),
                    build_normalization_process=Mock(return_value=({}, 'action_cartesian')),
                    save_normalization_process=Mock())

    def test_legacy_arguments_match_baseline(self):
        for cache in (str(self.root), None, '~/literal-cache'):
            self.cfg.cache_dir = cache
            calls = []
            for baseline in (True, False):
                ns = self.namespace()
                function('get_dataset', ns, baseline)(self.cfg, 'flip_mug/test/push', num_steps=3)
                calls.append(ns['swm'].data.HDF5Dataset.call_args)
            self.assertEqual(calls[0], calls[1])
            self.assertEqual(calls[1].args, ('flip_mug/test/push',))
            self.assertNotIn('path', calls[1].kwargs)

    def test_absolute_constructor_arguments(self):
        ns = self.namespace()
        function('get_dataset', ns)(self.cfg, '/tmp/test_dataset/push.h5', num_steps=2)
        ns['swm'].data.HDF5Dataset.assert_called_once_with(
            path='/tmp/test_dataset/push.h5', num_steps=2,
            keys_to_cache=self.cfg.dataset.keys_to_cache, cache_dir=self.root)

    def test_resolution_and_normalization_branches(self):
        for absolute in (False, True):
            expected = self.root / 'datasets/flip_mug/test/push.h5'
            expected.parent.mkdir(parents=True, exist_ok=True)
            self.cfg.eval.dataset_name = str(expected) if absolute else 'flip_mug/test/push'
            for saved_stats in (False, True):
                stats = Path(self.cfg.eval.normalization_stats_path)
                if saved_stats:
                    np.savez(stats, sentinel=np.array([42]))
                before = stats.read_bytes() if saved_stats else None
                expected.touch()
                for baseline in ((False,) if absolute else (True, False)):
                    ns = self.namespace()
                    ns['get_dataset'] = Mock(return_value='dataset')
                    self.assertEqual(normalization(ns, baseline), expected)
                    if saved_stats:
                        ns['load_normalization_process'].assert_called_once_with(stats)
                        ns['get_dataset'].assert_not_called()
                        ns['build_normalization_process'].assert_not_called()
                        ns['save_normalization_process'].assert_not_called()
                        self.assertEqual(stats.read_bytes(), before)
                    else:
                        ns['get_dataset'].assert_called_once_with(self.cfg, self.cfg.eval.dataset_name)
                        ns['build_normalization_process'].assert_called_once_with(
                            stats_dataset='dataset', keys_to_cache=self.cfg.dataset.keys_to_cache)
                        ns['save_normalization_process'].assert_called_once_with(
                            stats_path=stats, process={}, action_key='action_cartesian')
                if saved_stats:
                    stats.unlink()

    def test_missing_absolute_no_fallback(self):
        self.cfg.eval.dataset_name = str(self.root / 'missing.h5')
        ns = self.namespace()
        ns['get_dataset'] = Mock()
        with self.assertRaisesRegex(FileNotFoundError, self.cfg.eval.dataset_name):
            normalization(ns)
        ns['get_dataset'].assert_not_called()
        ns['save_normalization_process'].assert_not_called()

    def test_real_hdf5_name_and_path_read_identical_without_modification(self):
        from stable_worldmodel.data import HDF5Dataset
        path = self.root / 'datasets/flip_mug/test/push.h5'
        path.parent.mkdir(parents=True)
        data = dict(proprio=np.arange(20, dtype=np.float32).reshape(5, 4),
                    action_cartesian=np.arange(40, dtype=np.float32).reshape(5, 8),
                    pixels=np.arange(60, dtype=np.uint8).reshape(5, 2, 2, 3))
        with h5py.File(path, 'w') as f:
            f['ep_len'] = [3, 2]
            f['ep_offset'] = [0, 3]
            for key, values in data.items():
                f[key] = values
        before = path.read_bytes()
        datasets = []
        try:
            for name in ('flip_mug/test/push', str(path)):
                ds = function('get_dataset', self.namespace(HDF5Dataset))(self.cfg, name)
                datasets.append(ds)
                self.assertEqual(ds.h5_path, path)
                np.testing.assert_array_equal(ds.lengths, [3, 2])
                np.testing.assert_array_equal(ds.offsets, [0, 3])
                self.assertEqual(len(ds), 5)
                for key, values in data.items():
                    np.testing.assert_array_equal(ds.get_col_data(key), values)
                for ep, (start, end) in enumerate(((0, 3), (3, 5))):
                    episode = ds.load_episode(ep)
                    for key in ('proprio', 'action_cartesian'):
                        np.testing.assert_array_equal(episode[key].numpy(), data[key][start:end])
                    np.testing.assert_array_equal(episode['pixels'].numpy(),
                                                 data['pixels'][start:end].transpose(0, 3, 1, 2))
            with self.assertRaises(FileNotFoundError):
                function('get_dataset', self.namespace(HDF5Dataset))(self.cfg, str(self.root / 'missing.h5'))
        finally:
            for ds in datasets:
                if ds.h5_file is not None:
                    ds.h5_file.close()
        self.assertEqual(path.read_bytes(), before)

    def test_dry_run_dispatch_unchanged(self):
        for name in ('run',):
            old = next(n for n in tree(True).body if isinstance(n, ast.FunctionDef) and n.name == name)
            new = next(n for n in tree().body if isinstance(n, ast.FunctionDef) and n.name == name)
            # Early return dispatches both modes before dataset/statistics resolution.
            self.assertEqual(ast.dump(old.body[1]), ast.dump(new.body[1]))
            self.assertIsInstance(new.body[1], ast.If)
            self.assertIsInstance(new.body[1].body[-1], ast.Return)
        offline = ROOT / 'offline_dp_eval.py'
        baseline = subprocess.check_output(['git', 'show', 'HEAD:offline_dp_eval.py'], cwd=ROOT, text=True)
        self.assertEqual(offline.read_text(), baseline)
        for key in ('dry_run_dataset_path', 'dry_run_timestep', 'dry_run_episode_idx'):
            self.assertIn(key, baseline)


if __name__ == '__main__':
    unittest.main()
