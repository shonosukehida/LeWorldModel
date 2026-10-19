"""Phase 1: no robot or trained checkpoints required."""
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hierarchical_gpc_policy import HierarchicalGPCPolicy, COST_KEYS
from stable_worldmodel.policy import GPCPolicy


def fixture(obs_horizon=3, pred_horizon=12):
    proposals = torch.arange(2*3*pred_horizon*8).float().reshape(2, 3, pred_horizon, 8)
    dp = SimpleNamespace(obs_horizon=obs_horizon, pred_horizon=pred_horizon, action_horizon=8,
                         action_dim=8, device=torch.device('cpu'),
                         sample_action_sequences=Mock(return_value=proposals))
    cost = torch.tensor([[9., 1., 5.], [2., 8., 0.]])
    hwm = SimpleNamespace(upper_world_model=SimpleNamespace(horizon=4),
                          get_cost=Mock(return_value={key: cost.clone() for key in COST_KEYS}))
    info = {key: np.full((2, 1, 6, 7, 3), 123, np.uint8)
            for key in ('pixels', 'wrist_pixels', 'goal', 'goal_wrist_pixels')}
    info.update(proprio=np.full((2,1,8), 4., np.float32), goal_proprio=np.full((2,1,8), 7., np.float32))
    dp_info = {key: np.zeros((2, obs_horizon, 6, 7, 3), np.uint8) for key in ('pixels', 'wrist_pixels')}
    return dp, hwm, info, dp_info, proposals


class HierarchicalGPCPolicyTests(unittest.TestCase):
    def test_future_slice_physical_inputs_argmin_and_one_step(self):
        dp, hwm, info, dp_info, proposals = fixture()
        policy = HierarchicalGPCPolicy(dp, hwm, num_candidates=3, action_horizon=8)
        self.assertIsInstance(policy, GPCPolicy)
        policy._prepare_info = Mock(side_effect=AssertionError('must not preprocess HWM input'))
        policy.normalize_action_for_world_model = Mock(side_effect=AssertionError('must not normalize actions'))
        before = {key: value.copy() for key, value in info.items()}
        action = policy.get_action(info, dp_info_dict=dp_info)
        dp.sample_action_sequences.assert_called_once_with(dp_info, num_samples=3, denormalize=True)
        hwm.get_cost.assert_called_once()
        wm_info, candidates = hwm.get_cost.call_args.args
        torch.testing.assert_close(candidates, proposals[:, :, 2:10])
        self.assertEqual(candidates.shape, (2,3,8,8))
        torch.testing.assert_close(wm_info['pixels'], torch.as_tensor(info['pixels']).permute(0,1,4,2,3))
        torch.testing.assert_close(wm_info['proprio'], torch.as_tensor(info['proprio']))
        np.testing.assert_array_equal(action, proposals[[0,1], [1,2], 2].numpy())
        self.assertEqual(action.shape, (2,8))
        torch.testing.assert_close(policy.last_diagnostics['best_idx'], torch.tensor([1,2]))
        self.assertEqual(policy.last_diagnostics['evaluated_steps'], 8)
        for key in COST_KEYS:
            self.assertEqual(policy.last_diagnostics[key].shape, (2,3))
        for key in info:
            np.testing.assert_array_equal(info[key], before[key])
        policy.get_action(info, dp_info_dict=dp_info)
        self.assertEqual(policy.last_diagnostics['selection_id'], 2)

    def test_checkpoint_horizons_and_explicit_layout(self):
        dp, hwm, info, dp_info, proposals = fixture(obs_horizon=5, pred_horizon=16)
        for key in ('pixels','wrist_pixels','goal','goal_wrist_pixels'):
            info[key] = torch.as_tensor(info[key]).permute(0,1,4,2,3)
        policy = HierarchicalGPCPolicy(dp, hwm, num_candidates=3, raw_image_layout='NCHW')
        policy.get_action(info, dp_info_dict=dp_info)
        torch.testing.assert_close(hwm.get_cost.call_args.args[1], proposals[:,:,4:12])

    def test_invalid_candidates_and_costs(self):
        for bad in (torch.zeros(2,3,12,7), torch.zeros(2,2,12,8), torch.zeros(1,3,12,8),
                    torch.zeros(2,3,11,8), torch.zeros(2,12,8),
                    torch.full((2,3,12,8), float('nan')), torch.full((2,3,12,8), float('inf'))):
            dp, hwm, info, dp_info, _ = fixture()
            dp.sample_action_sequences.return_value = bad
            policy = HierarchicalGPCPolicy(dp, hwm, num_candidates=3)
            with self.assertRaises(ValueError):
                policy.get_action(info, dp_info_dict=dp_info)
            hwm.get_cost.assert_not_called()
            self.assertIsNone(policy.last_diagnostics)
        for key in COST_KEYS:
            for bad in (torch.zeros(2,2), torch.full((2,3), float('nan')), torch.full((2,3), float('inf')), None):
                dp, hwm, info, dp_info, _ = fixture()
                hwm.get_cost.return_value[key] = bad
                with self.assertRaisesRegex(ValueError, key):
                    HierarchicalGPCPolicy(dp,hwm,num_candidates=3).get_action(info,dp_info_dict=dp_info)

    def test_nonmultiple_future_length_and_missing_history(self):
        # Available future length is 7: never silently round down to 4.
        dp,hwm,info,dp_info,_ = fixture(obs_horizon=3,pred_horizon=9)
        policy = HierarchicalGPCPolicy(dp,hwm,num_candidates=3)
        with self.assertRaisesRegex(ValueError, 'divisible'):
            policy.get_action(info,dp_info_dict=dp_info)
        hwm.get_cost.assert_not_called()
        with self.assertRaisesRegex(ValueError, 'dp_info_dict'):
            policy.get_action(info)
        with self.assertRaisesRegex(ValueError, 'obs_horizon'):
            policy.get_action(info,dp_info_dict={'pixels':dp_info['pixels'][:,:1]})


if __name__ == '__main__':
    unittest.main()
