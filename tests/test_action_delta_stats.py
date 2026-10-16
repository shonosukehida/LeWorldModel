"""CPU-only episode-safe action delta statistics and upper-model round trips."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from action_features import action_delta
from create_process_stats import create_process_stats
from normalization_stats import load_normalization_process
from test_upper_jepa import make_model

ROOT = Path(__file__).resolve().parents[1]


class ActionDeltaStatsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = self.root / 'data.h5'
        self.output = self.root / 'stats/process_stats.npz'
        self.raw = np.zeros((7, 8), dtype=np.float64)
        self.raw[:, 0] = [0, 1, 3, 100, 104, 900, 907]
        self.raw[:, 1] = [0, 2, 5, 100, 106, 900, 908]
        self.raw[:, 7] = [0, .2, .5, .9, .6, .4, .8]
        self.raw[:, 6] = 1
        self.episodes = np.array([0, 0, 0, 1, 1, 1, 1])
        self.steps = np.array([0, 1, 2, 0, 1, 3, 4])
        self.write()

    def write(self, raw=None, episodes=None, steps=None, include_steps=True):
        with h5py.File(self.dataset, 'w') as file:
            file['action_cartesian'] = self.raw if raw is None else raw
            file['proprio'] = np.arange(56.).reshape(7, 8)
            file['episode_idx'] = self.episodes if episodes is None else episodes
            if include_steps:
                file['step_idx'] = self.steps if steps is None else steps

    def generate(self):
        return create_process_stats(self.dataset, self.output.parent, include_action_delta=True)

    def test_episode_boundaries_step_gaps_and_round_trip(self):
        self.generate()
        process, key = load_normalization_process(self.output)
        self.assertEqual(key, 'action_cartesian')
        self.assertEqual(set(process), {'action_cartesian', 'proprio', 'goal_proprio', 'action_delta'})
        # Cross-episode 3->100 and step gap 104->900 must never be fitted.
        indices = np.array([0, 1, 3, 5])
        expected = np.zeros((4, 8))
        expected[:, :3] = self.raw[indices + 1, :3] - self.raw[indices, :3]
        expected[:, 6] = 1
        expected[:, 7] = self.raw[indices + 1, 7] - self.raw[indices, 7]
        delta = process['action_delta']
        np.testing.assert_allclose(delta.mean_, expected.mean(0, keepdims=True))
        std = expected.std(0, ddof=1, keepdims=True)
        np.testing.assert_allclose(delta.scale_, np.where(std < 1e-4, 1., std))
        np.testing.assert_allclose(delta.inverse_transform(delta.transform(expected)), expected)
        # Absolute/proprio statistics remain identical to the ordinary mode.
        baseline = create_process_stats(self.dataset, self.root / 'baseline')
        original, _ = load_normalization_process(baseline)
        for key in ('action_cartesian', 'proprio'):
            np.testing.assert_array_equal(process[key].mean_, original[key].mean_)
            np.testing.assert_array_equal(process[key].scale_, original[key].scale_)

    def test_optional_step_indices_use_stored_adjacency(self):
        self.write(include_steps=False)
        self.generate()
        process, _ = load_normalization_process(self.output)
        self.assertAlmostEqual(process['action_delta'].mean_[0, 0], (1+2+4+796+7)/5)

    def test_quaternion_matches_macro_and_shared_function(self):
        # Noncommuting Rx(90), Ry(90), identity. Nonunit/sign-flipped inputs.
        self.raw[:3, 3:7] = [[2, 0, 0, 2], [0, -3, 0, -3], [0, 0, 0, -4]]
        self.write()
        self.generate()
        process, _ = load_normalization_process(self.output)
        indices = np.array([0, 1, 3, 5])
        pairs = torch.tensor(np.stack((self.raw[indices], self.raw[indices+1]), 1))
        physical_delta = action_delta(pairs).squeeze(1)
        torch.testing.assert_close(physical_delta[0, 3:7], torch.tensor([-.5, .5, -.5, .5], dtype=torch.float64))
        model = make_model(2, normalization_stats_path=self.output,
                           action_mean=None, action_scale=None, delta_mean=None, delta_scale=None)
        macro = model.build_macro_action(pairs)
        np.testing.assert_allclose(macro[:, 16:].numpy(), process['action_delta'].transform(physical_delta.numpy()), atol=1e-6)
        normalized = make_model(2, normalization_stats_path=self.output, actions_normalized=True,
                                action_mean=None, action_scale=None, delta_mean=None, delta_scale=None)
        standardized = torch.tensor(process['action_cartesian'].transform(pairs.numpy()))
        torch.testing.assert_close(normalized.build_macro_action(standardized), macro)

    def test_invalid_boundaries_preserve_existing_file(self):
        self.generate()
        before = self.output.read_bytes()
        cases = [('missing', 'episode_idx'), ('short', 'episode_idx'),
                 ('float', 'episode_idx'), ('nan', 'episode_idx'), ('negative', 'episode_idx'),
                 ('repeated', 'episode_idx'), ('short', 'step_idx'),
                 ('duplicate', 'step_idx'), ('regression', 'step_idx'), ('nan', 'step_idx')]
        for case, key in cases:
            with self.subTest(case=case, key=key):
                self.write()
                with h5py.File(self.dataset, 'a') as file:
                    values = file[key][:]
                    del file[key]
                    if case == 'short':
                        values = values[:-1]
                    elif case == 'float':
                        values = values.astype(float)
                    elif case == 'nan':
                        values = values.astype(float)
                        values[1] = np.nan
                    elif case == 'negative':
                        values[1] = -1
                    elif case == 'repeated':
                        values[-1] = 0
                    elif case == 'duplicate':
                        values[1] = values[0]
                    elif case == 'regression':
                        values[:3] = [2, 1, 0]
                    if case != 'missing':
                        file[key] = values
                with self.assertRaisesRegex(ValueError, key):
                    self.generate()
                self.assertEqual(before, self.output.read_bytes())

    def test_nan_rows_never_bridge_and_insufficient_pairs_fail(self):
        self.raw[1] = np.nan
        self.write()
        self.generate()
        process, _ = load_normalization_process(self.output)
        self.assertAlmostEqual(process['action_delta'].mean_[0, 0], (4+7)/2)
        before = self.output.read_bytes()
        self.raw[4] = np.nan
        self.write()
        with self.assertRaisesRegex(ValueError, 'two valid continuous'):
            self.generate()
        self.assertEqual(before, self.output.read_bytes())

    def test_bad_actions_and_round_trip_failure_do_not_overwrite(self):
        self.generate()
        before = self.output.read_bytes()
        for raw in (self.raw[:, :7], np.full((7, 8), np.inf), np.zeros((7, 8))):
            with self.subTest(shape=raw.shape):
                self.write(raw=raw)
                with self.assertRaises(ValueError):
                    self.generate()
                self.assertEqual(before, self.output.read_bytes())
        self.write()
        with patch('create_process_stats.load_normalization_process', side_effect=ValueError('round trip failed')):
            with self.assertRaisesRegex(ValueError, 'round trip failed'):
                self.generate()
        self.assertEqual(before, self.output.read_bytes())
        self.assertEqual(list(self.output.parent.glob('.process_stats.*')), [])

    def test_missing_or_corrupt_saved_stats_rejected_by_upper(self):
        def load_model(path, **kwargs):
            return make_model(2, normalization_stats_path=path, action_mean=None,
                              action_scale=None, delta_mean=None, delta_scale=None, **kwargs)
        with self.assertRaises(FileNotFoundError):
            load_model(self.output)
        baseline = create_process_stats(self.dataset, self.output.parent)
        with self.assertRaisesRegex(ValueError, 'action_delta'):
            load_model(baseline)
        # The existing stats file still works for absolute-only upper actions.
        load_model(baseline, use_action_delta=False).build_macro_action(torch.tensor(self.raw[:2])[None])
        self.generate()
        with np.load(self.output, allow_pickle=False) as file:
            arrays = {key: file[key].copy() for key in file.files}
        meta = json.loads(str(arrays['metadata'].item()))
        action_index = meta['keys'].index('action_cartesian')
        delta_index = meta['keys'].index('action_delta')
        for key, bad in ((f'processor_{delta_index}_scale', np.zeros((1, 8))),
                         (f'processor_{delta_index}_mean', np.full((1, 8), np.nan)),
                         (f'processor_{action_index}_mean', np.zeros((1, 7)))):
            corrupted = self.root / 'bad.npz'
            np.savez(corrupted, **{**arrays, key: bad})
            with self.assertRaises(ValueError):
                load_model(corrupted)
        missing = dict(arrays)
        del missing[f'processor_{delta_index}_mean']
        np.savez(self.root / 'missing.npz', **missing)
        with self.assertRaises(ValueError):
            load_model(self.root / 'missing.npz')

    def test_cli_opt_in(self):
        result = subprocess.run([sys.executable, str(ROOT / 'create_process_stats.py'),
                                 '--dataset', str(self.dataset), '--output', str(self.output.parent),
                                 '--include-action-delta'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        process, _ = load_normalization_process(self.output)
        self.assertIn('action_delta', process)
        self.assertNotIn('goal_action_delta', process)


if __name__ == '__main__':
    unittest.main()
