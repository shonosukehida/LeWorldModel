"""IMLE-only column collection, exact legacy training, and 60k-row benchmark.

All HDF5 files/checkpoints are generated under pytest's temporary directory.
Run with -s to retain benchmark timings. No robot/camera imports or connections.
"""
from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import subprocess
from time import perf_counter
from unittest.mock import patch

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import random_split

import train_imle_policy as trainer
from imle_checkpoint import load_imle_policy
from stable_worldmodel.data import HDF5Dataset
from stable_worldmodel.imle.proprio import fit_proprio_stats, normalize_proprio
from test_imle_integration import config, dataset


def legacy_collect(raw, indices, obs_horizon):
    return torch.cat([
        torch.as_tensor(raw[i]['proprio'][:obs_horizon]) for i in indices
    ])


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_raw(path, lengths, offsets, frameskip=1, dtype='f4', cached=True,
             image_size=4):
    n = max(o + length for o, length in zip(offsets, lengths))
    rng = np.random.default_rng(15)
    values = rng.normal(size=(n, 8)).astype(dtype)
    values[:, 1] = 2  # Constant std fallback.
    values[:, 2] *= 1e-6  # Near-constant std fallback.
    values[:, 7] = np.arange(n) % 2
    with h5py.File(path, 'w') as f:
        f['ep_len'], f['ep_offset'] = lengths, offsets
        f['proprio'] = values
        f['action_cartesian'] = values
        for key in ('pixels', 'wrist_pixels'):
            f.create_dataset(key, data=rng.integers(0, 256, (n, image_size, image_size, 3), dtype='u1'),
                             chunks=(16, image_size, image_size, 3), compression='gzip')
    return HDF5Dataset(
        path=path, num_steps=16, frameskip=frameskip, transform=None,
        keys_to_load=['pixels', 'wrist_pixels', 'action_cartesian', 'proprio'],
        keys_to_cache=['action_cartesian', 'proprio'] if cached else [],
    )


@pytest.mark.parametrize('frameskip,obs_horizon', [(1, 1), (1, 8), (2, 8), (3, 16), (2, 20)])
@pytest.mark.parametrize('dtype,cached', [('f4', True), ('f8', False)])
def test_exact_split_order_weights_stats_and_no_image_io(tmp_path, frameskip, obs_horizon, dtype, cached):
    raw = make_raw(tmp_path / 'windows.h5', [64, 12, 71], [3, 70, 90],
                   frameskip, dtype, cached)
    try:
        train, val = random_split(raw, [int(len(raw) * .9), len(raw) - int(len(raw) * .9)],
                                  generator=torch.Generator().manual_seed(3072))
        expected = [legacy_collect(raw, split.indices, obs_horizon) for split in (train, val)]
        torch_rng, numpy_rng = torch.get_rng_state(), np.random.get_state()
        original = h5py.Dataset.__getitem__
        reads = []

        def read_column(column, key):
            reads.append(column.name)
            assert column.name == '/proprio'
            return original(column, key)

        with patch.object(HDF5Dataset, '_load_slice', side_effect=AssertionError('window I/O')), \
             patch.object(h5py.Dataset, '__getitem__', read_column), \
             patch.object(raw, 'get_col_data', wraps=raw.get_col_data) as get_col:
            actual = [trainer.collect_proprio_windows(raw, split.indices, obs_horizon)
                      for split in (train, val)]
        assert get_col.call_args_list == [(("proprio",),), (("proprio",),)]
        assert reads == ([] if cached else ['/proprio', '/proprio'])
        assert torch.equal(torch_rng, torch.get_rng_state())
        np.testing.assert_equal(numpy_rng, np.random.get_state())
        for old, new in zip(expected, actual):
            torch.testing.assert_close(old, new, rtol=0, atol=0)
            assert old.dtype == new.dtype
        old_stats, new_stats = fit_proprio_stats(expected[0]), fit_proprio_stats(actual[0])
        for key in ('proprio_mean', 'proprio_std'):
            torch.testing.assert_close(old_stats[key], new_stats[key], rtol=0, atol=0)
        for old, new in zip(expected, actual):
            torch.testing.assert_close(normalize_proprio(old, old_stats),
                                       normalize_proprio(new, new_stats), rtol=0, atol=0)
        # A global-column mean would silently lose the Train/window weighting.
        assert not torch.equal(new_stats['proprio_mean'], torch.from_numpy(raw.get_col_data('proprio')).mean(0).float())
    finally:
        if raw.h5_file is not None:
            raw.h5_file.close()


def assert_nested_equal(old, new):
    if torch.is_tensor(old):
        torch.testing.assert_close(old, new, rtol=0, atol=0)
    elif isinstance(old, dict):
        assert old.keys() == new.keys()
        for key in old:
            assert_nested_equal(old[key], new[key])
    elif isinstance(old, (list, tuple)):
        assert len(old) == len(new)
        for a, b in zip(old, new):
            assert_nested_equal(a, b)
    else:
        assert old == new


@pytest.mark.parametrize('use_proprio', [False, True])
def test_training_checkpoint_diagnostics_output_and_rng_match_original(tmp_path, use_proprio):
    # Pinned pre-fix entrypoint: compare the entire run, including initialization,
    # split/shuffle RNG, losses, optimizer updates, EMA, checkpoint, and inference.
    source = subprocess.check_output([
        'git', '-C', str(Path(trainer.__file__).resolve().parent), 'show',
        '21d1c5719c1871871b76a59800399c4ad10e3acb:train_imle_policy.py',
    ], text=True)
    original = dict(trainer.__dict__)
    exec(compile(source, trainer.__file__, 'exec'), original)
    path = tmp_path / 'data.h5'
    dataset(path)
    cfg = config()
    cfg.imle.use_proprio = use_proprio
    cfg.data.dataset.path = str(path)
    results = []
    for name, run, namespace in [('old', original['run'], original),
                                  ('new', trainer.run, trainer.__dict__)]:
        output = tmp_path / name
        log = StringIO()
        with patch.dict(namespace, checkpoint_directory=lambda _: output), redirect_stdout(log), \
             patch.object(trainer, 'collect_proprio_windows', wraps=trainer.collect_proprio_windows) as collect:
            run.__wrapped__(cfg)
            assert collect.call_count == (2 if use_proprio and name == 'new' else 0)
        state = torch.load(output / 'latest.ckpt', weights_only=False)
        rng = torch.get_rng_state()
        numpy_rng = np.random.get_state()
        diagnostics = json.loads((output / 'proprio_diagnostics.json').read_text()) if use_proprio else None
        policy = load_imle_policy(output / 'latest.ckpt', lambda x: x, 'cpu')
        policy.transform = {}
        info = dict(pixels=torch.zeros(1, 2, 3, 32, 32), wrist_pixels=torch.ones(1, 2, 3, 32, 32))
        if use_proprio:
            info['proprio'] = torch.zeros(1, 2, 8)
        torch.manual_seed(99)
        actions = policy.sample_action_sequences(info, 3)
        results.append((state, rng, diagnostics, actions, torch.get_rng_state(), numpy_rng))
        timing = '[IMLE preprocessing]'
        if name == 'new' and use_proprio:
            for stage in ('dataset initialization/cache', 'action statistics', 'transforms/split/loaders',
                          'Train proprio collection/statistics', 'policy/device/optimizer/EMA',
                          'proprio diagnostics/output directory', 'config/logging setup', 'total before first epoch'):
                assert f'{timing} {stage}:' in log.getvalue()
        else:
            assert timing not in log.getvalue()
    assert_nested_equal(results[0][:5], results[1][:5])
    np.testing.assert_equal(results[0][5], results[1][5])


def test_200_episode_300_step_preprocessing_benchmark(tmp_path):
    """Full window count and actual two-view HDF5 I/O; reduced 8x8 images.

    Relative timing plus strict I/O counts avoid a machine-specific time budget.
    This measures CPU preprocessing only, not production storage/GPU throughput.
    """
    raw = make_raw(tmp_path / 'scale.h5', [300] * 200, np.arange(200) * 300,
                   image_size=8)
    train, val = random_split(raw, [51300, 5700], generator=torch.Generator().manual_seed(3072))
    reads = []
    original = h5py.Dataset.__getitem__

    def read_column(column, key):
        if column.name in ('/pixels', '/wrist_pixels'):
            reads.append(column.name)
        return original(column, key)

    results = []
    try:
        for collect in (legacy_collect, trainer.collect_proprio_windows):
            reads.clear()
            started = perf_counter()
            with patch.object(h5py.Dataset, '__getitem__', read_column):
                values = [collect(raw, split.indices, 8) for split in (train, val)]
                stats = fit_proprio_stats(values[0])
                normalized = [normalize_proprio(x, stats) for x in values]
                diagnostics = [[x.double().mean(0), x.double().std(0, correction=0),
                                x.double().amin(0), x.double().amax(0)]
                               for x in values + normalized]
            results.append((perf_counter() - started, len(reads), values, stats, diagnostics))
        old, new = results
        assert old[1] == 114000
        assert new[1] == 0
        assert_nested_equal(old[2:], new[2:])
        assert new[0] < old[0]
        print(f'\nIMLE preprocessing benchmark: 200x300 rows, 57000 windows, '
              f'pred=16 obs=8, two 8x8 views; old={old[0]:.6f}s '
              f'new={new[0]:.6f}s speedup={old[0]/new[0]:.1f}x; '
              f'image_reads={old[1]}->{new[1]}; mean/std max_abs_diff=0')
    finally:
        if raw.h5_file is not None:
            raw.h5_file.close()
