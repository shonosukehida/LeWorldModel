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


def configure(policy, weight):
    # Synthetic WM emits eight dimensions: two image blocks of 2, proprio of 4.
    def mlp(size):
        return SimpleNamespace(net=torch.nn.Sequential(torch.nn.Linear(1, size)))
    policy.world_model.overhead_projector = mlp(2)
    policy.world_model.wrist_projector = mlp(2)
    policy.world_model.prop_encoder = mlp(4)
    policy.reward_fn = LatentGoalReward(policy.world_model, weight)
    return policy


@pytest.mark.parametrize('weight', [0., .5, 1.])
def test_latent_metrics_preserved_and_costs_saved(tmp_path, weight):
    path, policy, builder, _ = fixtures.setup(tmp_path)
    configure(policy, weight)
    before = path.read_bytes()
    with patch.object(latent, 'plot_latent_dist'):
        result = latent.evaluate_latent_dist(policy, path, 7, 3, 42, tmp_path, {}, builder, 8)
    assert result['metadata']['lambda_prop'] == weight
    with h5py.File(Path(result['metadata']['output_dir'])/'latent_distance.h5') as file:
        expected = np.linalg.norm(file['observed_latents'][:] - file['goal_latent'][:], axis=-1)
        np.testing.assert_allclose(file['observed_latent_distance'][:], expected, rtol=1e-6)
        np.testing.assert_allclose(file['candidate_rewards'][:],
                                   -(file['image_cost'][:] + weight * file['prop_cost'][:]), rtol=1e-5)
        np.testing.assert_array_equal(file['weighted_total_cost'][:], -file['candidate_rewards'][:])
        np.testing.assert_array_equal(file['selected_candidate_indices'][:], file['candidate_rewards'][:].argmax(-1))
    assert path.read_bytes() == before


@pytest.mark.parametrize('mode', ['multi_sample', 'chunk_rollout'])
@pytest.mark.parametrize('weight', [0., 1.])
def test_other_offline_modes(tmp_path, mode, weight):
    path, policy, builder, _ = fixtures.setup(tmp_path)
    configure(policy, weight)
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
    with h5py.File(Path(result['metadata']['output_dir'])/name) as file:
        np.testing.assert_allclose(file['candidate_rewards'][:].reshape(-1, 3),
                                   -(file['image_cost'][:] + weight * file['prop_cost'][:]).reshape(-1, 3), rtol=1e-5)


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
def test_candidate_recorder_costs_and_legacy_archive(tmp_path, weight):
    import test_gpc_visualization as visualization
    policy, _ = visualization.make_policy()
    # Configure the same model boundary with two image blocks and a prop block.
    policy.world_model = SimpleNamespace()
    configure(policy, weight)
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
        prefix = 'record_0000_'
        np.testing.assert_array_equal(archive[prefix+'weighted_total_cost'], -expected.numpy()[0])
        assert int(archive[prefix+'selected_candidate_index']) == expected.argmax().item()
        assert int(archive[prefix+'best_idx']) == expected.argmax().item()
        assert prefix+'candidate_actions' in archive
