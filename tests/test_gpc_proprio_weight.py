"""Weighted GPC integration across offline modes, with identical DP proposals."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import h5py
import numpy as np
import pytest
import torch
from stable_worldmodel.reward import LatentGoalReward
import offline_gpc_eval as gpc
import offline_latent_dist_eval as latent
import test_offline_latent_dist_eval as fixtures


def configure(policy, weight, view_weights=(1., 1.)):
    # Synthetic WM emits eight dimensions: two image blocks of 2, proprio of 4.
    def mlp(size):
        return SimpleNamespace(net=torch.nn.Sequential(torch.nn.Linear(1, size)))
    policy.world_model.overhead_projector = mlp(2)
    policy.world_model.wrist_projector = mlp(2)
    policy.world_model.prop_encoder = mlp(4)
    policy.reward_fn = LatentGoalReward(policy.world_model, weight,
                                        lambda_overhead=view_weights[0], lambda_wrist=view_weights[1])
    return policy


@pytest.mark.parametrize('weight', [0., .5, 1.])
@pytest.mark.parametrize('view_weights', [(1., 1.), (0., 2.), (2., .25)])
def test_latent_metrics_preserved_and_costs_saved(tmp_path, weight, view_weights):
    path, policy, builder, _ = fixtures.setup(tmp_path)
    configure(policy, weight, view_weights)
    before = path.read_bytes()
    with patch.object(latent, 'plot_latent_dist'):
        result = latent.evaluate_latent_dist(policy, path, 7, 3, 42, tmp_path, {}, builder, 8)
    assert result['metadata']['lambda_prop'] == weight
    assert result['metadata']['lambda_overhead'] == view_weights[0]
    assert result['metadata']['lambda_wrist'] == view_weights[1]
    with h5py.File(Path(result['metadata']['output_dir'])/'latent_distance.h5') as file:
        expected = np.linalg.norm(file['observed_latents'][:] - file['goal_latent'][:], axis=-1)
        np.testing.assert_allclose(file['observed_latent_distance'][:], expected, rtol=1e-6)
        np.testing.assert_allclose(file['candidate_rewards'][:],
                                   -(view_weights[0] * file['overhead_cost'][:] + view_weights[1] * file['wrist_cost'][:] + weight * file['prop_cost'][:]), rtol=1e-5)
        np.testing.assert_array_equal(file['image_cost'][:], file['overhead_cost'][:] + file['wrist_cost'][:])
        np.testing.assert_array_equal(file['weighted_total_cost'][:], -file['candidate_rewards'][:])
        np.testing.assert_array_equal(file['selected_candidate_indices'][:], file['candidate_rewards'][:].argmax(-1))
    assert path.read_bytes() == before


@pytest.mark.parametrize('mode', ['multi_sample', 'chunk_rollout'])
@pytest.mark.parametrize('weight', [0., 1.])
@pytest.mark.parametrize('view_weights', [(1., 1.), (0., 2.), (2., .25)])
def test_other_offline_modes(tmp_path, mode, weight, view_weights):
    path, policy, builder, _ = fixtures.setup(tmp_path)
    configure(policy, weight, view_weights)
    if mode == 'multi_sample':
        original_builder = builder
        def builder(data, row, prop):
            data['offline_step_idx'] = np.arange(len(data['pixels'])) + 100
            return original_builder(data, row, prop)
        with patch.object(gpc, 'plot_gpc'):
            result = gpc.evaluate_gpc_multi_sample(policy, path, 7, 100, 3, 42, tmp_path, {}, builder)
        name = 'multi_sample_rollout.h5'
    else:
        with patch.object(gpc, 'plot_gpc_chunk_rollout'):
            result = gpc.evaluate_gpc_chunk_rollout(policy, path, 7, 3, 42, tmp_path, {}, builder, 8)
        name = 'chunk_rollout.h5'
    assert result['metadata']['lambda_prop'] == weight
    assert result['metadata']['lambda_overhead'] == view_weights[0]
    assert result['metadata']['lambda_wrist'] == view_weights[1]
    with h5py.File(Path(result['metadata']['output_dir'])/name) as file:
        np.testing.assert_allclose(file['candidate_rewards'][:].reshape(-1, 3),
                                   -(view_weights[0] * file['overhead_cost'][:] + view_weights[1] * file['wrist_cost'][:] + weight * file['prop_cost'][:]).reshape(-1, 3), rtol=1e-5)


def test_default_configured_reward_exact_gpc_legacy(tmp_path):
    path, policy, builder, _ = fixtures.setup(tmp_path)
    data, steps, _ = gpc.load_episode(path, 7)
    data['offline_step_idx'] = steps
    info = builder(data, 0, np.zeros(8, dtype=np.float32))
    candidates = np.random.default_rng(42).normal(size=(3, 16, 8)).astype(np.float32)
    before = gpc.select_candidates(policy, candidates, info)
    configure(policy, 1.0)
    after = gpc.select_candidates(policy, candidates, info)
    assert before[0] == after[0] and before[2] == after[2]
    np.testing.assert_array_equal(before[1], after[1])


@pytest.mark.parametrize('weight', [0., 1.])
@pytest.mark.parametrize('view_weights', [(1., 1.), (0., 2.), (2., .25)])
def test_candidate_recorder_costs_and_legacy_archive(tmp_path, weight, view_weights):
    import test_gpc_visualization as visualization
    policy, _ = visualization.make_policy()
    # Configure the same model boundary with two image blocks and a prop block.
    policy.world_model = SimpleNamespace()
    configure(policy, weight, view_weights)
    output = dict(predicted_emb=torch.arange(4*8, dtype=torch.float32).reshape(1, 4, 1, 8),
                  goal_emb=torch.zeros(1, 1, 8))
    policy.rollout.return_value = output
    expected = policy.reward_fn(output, {})
    with visualization.vis.GPCRecorder(policy) as recorder:
        recorder.begin_step(3)
        policy.get_action({})
    recorder.save(tmp_path)
    with np.load(tmp_path/'gpc/data/candidates.npz', allow_pickle=False) as archive:
        assert float(archive['lambda_prop']) == weight
        assert float(archive['lambda_overhead']) == view_weights[0]
        assert float(archive['lambda_wrist']) == view_weights[1]
        prefix = 'record_0000_'
        np.testing.assert_array_equal(archive[prefix+'weighted_total_cost'], -expected.numpy()[0])
        np.testing.assert_array_equal(archive[prefix+'image_cost'], archive[prefix+'overhead_cost'] + archive[prefix+'wrist_cost'])
        np.testing.assert_allclose(-expected.numpy()[0],
                                   view_weights[0] * archive[prefix+'overhead_cost'] +
                                   view_weights[1] * archive[prefix+'wrist_cost'] +
                                   weight * archive[prefix+'prop_cost'], rtol=1e-5)
        assert int(archive[prefix+'selected_candidate_index']) == expected.argmax().item()
        assert int(archive[prefix+'best_idx']) == expected.argmax().item()
        assert prefix+'candidate_actions' in archive


def test_camera_weights_change_gpc_selection_for_identical_proposals(tmp_path):
    path, policy, builder, _ = fixtures.setup(tmp_path)
    data, steps, _ = gpc.load_episode(path, 7)
    data['offline_step_idx'] = steps
    info = builder(data, 0, np.zeros(8, dtype=np.float32))
    candidates = np.random.default_rng(42).normal(size=(3, 16, 8)).astype(np.float32)
    original = candidates.copy()
    terminal = torch.zeros(1, 3, 1, 8)
    terminal[0, 0, 0, 0] = 1  # overhead cost=1
    terminal[0, 1, 0, 2] = 2  # wrist cost=4
    terminal[0, 2, 0, 4] = 3  # proprio cost=9
    output = dict(predicted_emb=terminal, goal_emb=torch.zeros(1, 1, 8))
    with patch.object(policy, 'rollout', return_value=output):
        for views, prop, expected in (((1., 1.), 1., 0), ((10., 1.), 1., 1),
                                      ((1., 0.), 1., 1), ((1., 1.), 0., 2)):
            configure(policy, prop, views)
            diagnostics = {}
            index, rewards, _ = gpc.select_candidates(policy, candidates, info, diagnostics)
            assert index == expected
            np.testing.assert_array_equal(diagnostics['weighted_total_cost'], -rewards[0])
    np.testing.assert_array_equal(candidates, original)
