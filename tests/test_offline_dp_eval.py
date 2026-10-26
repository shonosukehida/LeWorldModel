"""Hardware-free offline evaluation and actual DP method contract tests."""
import ast
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock
import json

import h5py
import numpy as np
import torch

from offline_dp_eval import load_episode, evaluate_episode, compute_metrics, run_offline_dp

ROOT = Path(__file__).resolve().parents[1]


def methods(path, class_name, names):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name)
    ns = dict(torch=torch, np=np)
    exec(compile(ast.Module(body=[n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name in names], type_ignores=[]), str(path), 'exec'), ns)
    return ns


class OfflineTests(unittest.TestCase):
    def dataset(self, path, with_steps=True):
        with h5py.File(path, 'w') as f:
            f['episode_idx'] = [2, 2, 7, 7, 7]
            if with_steps:
                f['step_idx'] = [0, 1, 0, 1, 2]
            f['ep_offset'], f['ep_len'] = [0, 2], [2, 3]
            for k in ('pixels', 'wrist_pixels'):
                f[k] = np.broadcast_to(np.arange(5, dtype=np.uint8)[:, None, None, None], (5, 4, 4, 3))
            gt = np.zeros((5, 8)); gt[:, 6] = 1; gt[:, 7] = np.arange(5) / 4
            f['action_cartesian'] = gt

    def test_history_outputs_raw_actions_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'data.h5'; self.dataset(path)
            calls = []
            def action(info):
                calls.append(info)
                # Outside robot bounds and gripper range: must remain untouched.
                return np.array([[10, -10, 2, 0, 0, 0, 1, 1.7]])
            policy = SimpleNamespace(action_dim=8, process={'action_cartesian': object()},
                                     obs_horizon=4, action_horizon=2, pred_horizon=6,
                                     num_inference_steps=3, get_action=action)
            out = Path(tmp) / 'out'
            evaluate_episode(policy, path, 7, out, {'checkpoint_path': 'selected.ckpt'})
            for k in ('pixels', 'wrist_pixels'):
                np.testing.assert_array_equal([x[k][0, :, 0, 0, 0] for x in calls],
                                              [[2, 2, 2, 2], [2, 2, 2, 3], [2, 2, 3, 4]])
            with h5py.File(out / 'offline_rollout.h5') as f:
                self.assertEqual(f['predicted_action'].shape, (3, 8))
                np.testing.assert_array_equal(f['predicted_action'][:, 7], [1.7]*3)
                np.testing.assert_array_equal(f['ground_truth_action'][:, 7], [.5, .75, 1])
                np.testing.assert_array_equal(f['episode_idx'], [7]*3)
                self.assertEqual(json.loads(f.attrs['metadata'])['checkpoint_path'], 'selected.ckpt')
            self.assertEqual(json.loads((out / 'metrics.json').read_text())['metadata']['model']['obs_horizon'], 4)
            for name in ('position', 'orientation', 'gripper'):
                self.assertGreater((out / f'predicted_vs_gt_{name}.png').stat().st_size, 0)

    def test_episode_order_validation(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'data.h5'; self.dataset(path, False)
            data, steps, _ = load_episode(path, 7)
            self.assertEqual(len(data['pixels']), 3)
            np.testing.assert_array_equal(steps, [0, 1, 2])
            with self.assertRaisesRegex(ValueError, 'not found'): load_episode(path, 0)
            with h5py.File(path, 'a') as f: del f['ep_offset']
            with self.assertRaisesRegex(ValueError, 'No step_idx'): load_episode(path, 7)
            self.dataset(path)
            with h5py.File(path, 'a') as f: f['step_idx'][3] = 0
            with self.assertRaisesRegex(ValueError, 'consecutive'): load_episode(path, 7)

    def test_metrics_sign_rotation_and_invalid(self):
        gt = np.array([[0., 0, 0, 0, 0, 0, 1, 0]])
        pred = np.array([[3., 4, 0, 0, 0, 0, -2, 1]])
        m = compute_metrics(pred, gt)
        self.assertEqual(m['position']['rmse_3d_m'], 5)
        self.assertEqual(m['orientation']['mean_angle_error_deg'], 0)
        self.assertEqual(m['gripper']['state_accuracy'], 0)
        pred[:, 3:7] = [0, 0, 1, 0]
        self.assertEqual(compute_metrics(pred, gt)['orientation']['mean_angle_error_deg'], 180)
        for value in (np.nan, np.inf):
            pred[0, 0] = value
            with self.assertRaisesRegex(ValueError, 'NaN/Inf'): compute_metrics(pred, gt)
        pred[:] = 0
        with self.assertRaisesRegex(ValueError, 'zero-norm'): compute_metrics(pred, gt)

    def test_actual_dp_slice_and_denormalization(self):
        ns = methods(ROOT.parent / 'stable-worldmodel/stable_worldmodel/policy.py',
                     'DiffusionPolicy', ('predict_action', 'get_action'))
        from action_normalization import restore_action_processor
        for mode in ('zscore', 'min_max'):
            stats = {'action_normalization': mode, 'action_mean': torch.ones(1, 8)*3,
                     'action_std': torch.ones(1, 8)*2, 'action_min': torch.zeros(1, 8),
                     'action_max': torch.ones(1, 8)*10}
            processor = restore_action_processor(stats, 8)
            policy = SimpleNamespace(obs_horizon=3, pred_horizon=6, action_horizon=2,
                                     action_dim=8, device='cpu', dtype=torch.float32,
                                     process={'action_cartesian': processor})
            policy._prepare_info = lambda x: x
            policy._encode_observation = lambda **kw: (None, 1, kw['pixels'].shape[1])
            trajectory = torch.arange(6, dtype=torch.float32)[None, :, None].expand(1, 6, 8)
            policy.conditional_sample = lambda **kw: trajectory
            policy.predict_action = lambda info: ns['predict_action'](policy, info)
            actual = ns['get_action'](policy, {'pixels': torch.zeros(1, 3, 3, 4, 4)})
            np.testing.assert_allclose(actual, processor.inverse_transform(np.full((1, 8), 2.)))

    def test_loader_restores_saved_settings_and_reports_missing(self):
        from unittest.mock import patch
        from action_normalization import validate_action_stats, restore_action_processor
        from normalization_stats import SafeStandardScaler
        model_cfg = dict(diffusion_step_embed_dim=12, down_dims=[8, 16], kernel_size=3,
                         n_groups=2, cond_predict_scale=False)
        checkpoint = dict(action_dim=8, obs_feature_dim=5, wrist_obs_feature_dim=5,
                          pred_horizon=11, obs_horizon=3, action_horizon=4,
                          num_inference_steps=7, num_train_timesteps=17,
                          beta_schedule='linear', prediction_type='sample',
                          action_key='action_cartesian', action_mean=torch.zeros(1, 8),
                          action_std=torch.ones(1, 8), model_state_dict={},
                          obs_encoder_state_dict={}, wrist_obs_encoder_state_dict={},
                          config=dict(model=model_cfg, diffusion=dict(clip_sample=False)))
        net = Mock(output_shape=lambda: (5,))
        unet, scheduler = Mock(return_value=net), Mock(return_value=SimpleNamespace(config={}))
        def make_policy(**kwargs): return SimpleNamespace(**kwargs)
        ns = dict(Path=Path, torch=torch, validate_action_stats=validate_action_stats,
                  restore_action_processor=restore_action_processor, SafeStandardScaler=SafeStandardScaler,
                  ResNet18ObsEncoder=Mock(return_value=net), ConditionalUnet1D=unet,
                  DDPMScheduler=scheduler, swm=SimpleNamespace(policy=SimpleNamespace(DiffusionPolicy=make_policy)))
        node = next(n for n in ast.parse((ROOT / 'eval_real_robot.py').read_text()).body
                    if isinstance(n, ast.FunctionDef) and n.name == 'load_diffusion_policy')
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<loader>', 'exec'), ns)
        with patch.object(torch, 'load', return_value=checkpoint) as read:
            policy = ns['load_diffusion_policy']('chosen.ckpt', 'transform', device='cpu')
            self.assertEqual(read.call_args.args[0], Path('chosen.ckpt'))
            for key in ('pred_horizon', 'obs_horizon', 'action_horizon', 'action_dim', 'num_inference_steps'):
                self.assertEqual(getattr(policy, key), checkpoint[key])
            self.assertEqual(unet.call_args.kwargs['global_cond_dim'], 30)
            self.assertEqual(unet.call_args.kwargs['down_dims'], (8, 16))
            self.assertEqual(scheduler.call_args.kwargs['num_train_timesteps'], 17)
            self.assertFalse(scheduler.call_args.kwargs['clip_sample'])
            del checkpoint['obs_horizon']
            with self.assertRaisesRegex(ValueError, 'obs_horizon'):
                ns['load_diffusion_policy']('chosen.ckpt', None, device='cpu')

    def test_dispatch_early_return_and_checkpoint_path(self):
        # Run the actual run() prefix, with every unrelated operation forbidden.
        tree = ast.parse((ROOT / 'eval_real_robot.py').read_text())
        run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
        branch = next(n for n in run.body if isinstance(n, ast.If))
        self.assertIn('run_offline_dp', ast.unparse(branch))
        self.assertIsInstance(branch.body[-1], ast.Return)
        robot = SimpleNamespace(dry_run=True, dry_run_dataset_path='dataset.h5', dry_run_episode_idx=7, output_dir='out')
        class Config(SimpleNamespace):
            def get(self, k, default=None): return getattr(self, k, default)
        cfg = Config(policy_type='diffusion', eval=SimpleNamespace(real_robot=robot, img_size=224),
                     gpc=SimpleNamespace(diffusion_checkpoint='chosen.ckpt'))
        import offline_dp_eval
        from unittest.mock import patch
        loader = Mock(return_value=SimpleNamespace(offline_checkpoint_metadata={}))
        with patch.object(offline_dp_eval, 'evaluate_episode', return_value={}) as evaluate:
            run_offline_dp(cfg, loader, 'existing_transform')
            loader.assert_called_once_with(checkpoint_path='chosen.ckpt', image_transform='existing_transform', device='cuda')
            evaluate.assert_called_once()
        prefix = ast.FunctionDef(name='dispatch', args=run.args, body=[branch], decorator_list=[])
        # Remove annotation so loading the isolated dispatcher has no Hydra dependency.
        prefix.args.args[0].annotation = None
        ns = dict(load_diffusion_policy=loader, dp_img_transform=lambda cfg: 'transform')
        exec(compile(ast.fix_missing_locations(ast.Module(body=[prefix], type_ignores=[])), '<dispatch>', 'exec'), ns)
        with patch.object(offline_dp_eval, 'run_offline_dp') as runner:
            ns['dispatch'](cfg)
            runner.assert_called_once_with(cfg, loader, 'transform')
        test = compile(ast.Expression(branch.test), '<dispatch>', 'eval')
        for kind, dry, expected in [('diffusion', True, True), ('diffusion', False, False), ('gpc', True, False), ('world_model', True, False)]:
            cfg.policy_type, robot.dry_run = kind, dry
            self.assertEqual(eval(test, {'cfg': cfg}), expected)


if __name__ == '__main__': unittest.main()
