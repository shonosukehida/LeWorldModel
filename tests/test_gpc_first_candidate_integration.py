"""Execution ablation: offline artifacts and real inference recorder, no hardware."""
import json
from pathlib import Path
from unittest.mock import Mock, patch

import h5py
import numpy as np
import pytest
import torch

import offline_gpc_eval as gpc
import offline_latent_dist_eval as latent
import test_offline_dp_eval as fixtures
import test_offline_gpc_eval as old
from test_gpc_visualization import make_policy, vis
from test_offline_latent_dist_eval import setup as latent_setup


@pytest.mark.parametrize('first', [False, True])
@pytest.mark.parametrize('mode', ['multi_sample', 'chunk_rollout'])
def test_offline_execution_and_saved_indices(tmp_path, first, mode):
    path = tmp_path / 'data.h5'
    fixtures.OfflineTests().dataset(path)
    with h5py.File(path, 'a') as f:
        f['proprio'] = np.full((5, 8), 17, dtype=np.float32)
    policy = old.make_policy()
    policy.execute_first_candidate = first
    policy.reward_fn = Mock(spec=[], return_value=torch.tensor([[0., 9., 2.]]))
    sampler = Mock(wraps=policy.diffusion_policy.sample_action_sequences)
    policy.diffusion_policy.sample_action_sequences = sampler
    builder = Mock(return_value=old.observation())
    if mode == 'multi_sample':
        result = gpc.evaluate_gpc_multi_sample(policy, path, 7, 1, 3, 42, tmp_path, {}, builder)
    else:
        result = gpc.evaluate_gpc_chunk_rollout(policy, path, 7, 3, 42, tmp_path, {}, builder, 2)
    out = Path(result['metadata']['output_dir'])
    assert result['metadata']['execute_first_candidate'] is first
    expected = 0 if first else 1
    filename = 'multi_sample_rollout.h5' if mode == 'multi_sample' else 'chunk_rollout.h5'
    with h5py.File(out / filename) as f:
        np.testing.assert_array_equal(f['best_idx'][()], result['best_idx'])
        assert np.all(f['best_idx'][()] == 1)
        if mode == 'multi_sample':
            assert f['selected_candidate_index'][()] == expected
            np.testing.assert_array_equal(f['selected_candidate_actions'], f['predicted_actions'][expected])
        else:
            assert np.all(f['selected_candidate_indices'][:] == expected)
            np.testing.assert_array_equal(f['selected_chunks'], f['predicted_chunks'][:, expected])
        assert f['candidate_rewards'].shape[-1] == 3
    metrics = json.loads((out / f'{mode}_metrics.json').read_text())
    assert metrics['best_idx'] == result['best_idx']
    assert policy.reward_fn.call_count == sampler.call_count == len(policy.world_model.calls)
    assert all(call[1].shape == (1, 3, 2, 8) for call in policy.world_model.calls)


@pytest.mark.parametrize('first', [False, True])
def test_latent_diagnostics_use_executed_candidate(tmp_path, first):
    path, policy, builder, _ = latent_setup(tmp_path)
    policy.execute_first_candidate = first
    with patch.object(latent, 'plot_latent_dist'):
        result = latent.evaluate_latent_dist(policy, path, 7, 20, 42, tmp_path, {}, builder, 8)
    with h5py.File(Path(result['metadata']['output_dir']) / 'latent_distance.h5') as f:
        best = f['candidate_rewards'][:].argmax(1)
        assert np.any(best != 0)
        np.testing.assert_array_equal(f['best_idx'], best)
        np.testing.assert_array_equal(f['selected_candidate_indices'], np.zeros_like(best) if first else best)
        for i, index in enumerate(f['selected_candidate_indices']):
            info, actions = policy.world_model.calls[2*i]
            expected = info['proprio'][0, index, 0] + actions[0, index].sum(0)
            np.testing.assert_allclose(f['gpc_terminal_latents'][i], expected, atol=1e-6)
            assert actions.shape == (1, 20, 8, 8)


def test_recorder_logs_and_renders_execution_separately_from_best(tmp_path):
    policy, candidates = make_policy()
    policy.execute_first_candidate = True
    with vis.GPCRecorder(policy) as recorder:
        recorder.begin_step(7)
        result = policy.get_action_chunk({})
    np.testing.assert_array_equal(result, candidates[:, 0, 2:].numpy())
    assert recorder.records[0]['best_idx'] == 1
    assert recorder.records[0]['selected_candidate_index'] == 0
    recorder.save(tmp_path)
    archive = tmp_path / 'gpc/data/candidates.npz'
    with np.load(archive) as data:
        assert data['record_0000_best_idx'] == 1
        assert data['record_0000_selected_candidate_index'] == 0
    vis.render_archive(archive, tmp_path / 'render')
    assert (tmp_path / 'render/step_0007.png').exists()
    fig, axes = vis.plot_candidates(candidates[0].numpy(), [1., 4., 4., -2.], 1, 7,
                                   selected_candidate_index=0)
    try:
        for i, ax in enumerate(axes):
            np.testing.assert_array_equal(ax.lines[-1].get_ydata(), candidates[0, 0, :, i])
        assert 'WM best index=1' in fig._suptitle.get_text()
        assert 'selected index=0 (Rank 3)' in fig._suptitle.get_text()
    finally:
        import matplotlib.pyplot as plt
        plt.close(fig)
