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
        robot.dry_run_mode = 'episode'
        with patch.object(offline_dp_eval, 'run_offline_dp') as episode_runner:
            ns['dispatch'](cfg)
            episode_runner.assert_called_once()
        robot.dry_run_mode = 'multi_sample'
        with patch.object(offline_dp_eval, 'run_offline_dp_multi_sample') as multi_runner:
            ns['dispatch'](cfg)
            multi_runner.assert_called_once_with(cfg, loader, 'transform')
        robot.dry_run_mode = 'invalid'
        with self.assertRaisesRegex(ValueError, 'Unknown dry_run_mode'):
            ns['dispatch'](cfg)
        test = compile(ast.Expression(branch.test), '<dispatch>', 'eval')
        for kind, dry, expected in [('diffusion', True, True), ('diffusion', False, False), ('gpc', True, False), ('world_model', True, False)]:
            cfg.policy_type, robot.dry_run = kind, dry
            self.assertEqual(eval(test, {'cfg': cfg}), expected)
            if not expected:
                with patch.object(offline_dp_eval, 'run_offline_dp_multi_sample') as runner:
                    ns['dispatch'](cfg)
                    runner.assert_not_called()


class MultiSampleTests(unittest.TestCase):
    def policy(self, mode='zscore', device='cpu'):
        from action_normalization import restore_action_processor
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        ns = methods(ROOT.parent / 'stable-worldmodel/stable_worldmodel/policy.py',
                     'DiffusionPolicy', ('sample_action_sequences', 'conditional_sample'))
        stats = dict(action_normalization=mode, action_mean=torch.ones(1, 8)*3,
                     action_std=torch.ones(1, 8)*2, action_min=torch.zeros(1, 8),
                     action_max=torch.ones(1, 8)*10)
        policy = SimpleNamespace(obs_horizon=4, pred_horizon=6, action_horizon=2, action_dim=8,
                                 device=device, dtype=torch.float32, num_inference_steps=3,
                                 process={'action_cartesian': restore_action_processor(stats, 8)},
                                 noise_scheduler=DDPMScheduler(num_train_timesteps=6, clip_sample=False))
        policy._prepare_info = lambda x: x
        policy._encode_observation = lambda **kw: (torch.ones(1, 4, device=device), 1, 4)
        policy.model = lambda trajectory, *args, **kw: torch.zeros_like(trajectory)
        policy.conditional_sample = lambda **kw: ns['conditional_sample'](policy, **kw)
        policy.sample_action_sequences = lambda *args, **kw: ns['sample_action_sequences'](policy, *args, **kw)
        return policy

    def test_real_sampler_seed_independence_inverse_and_rng_restoration(self):
        from offline_dp_eval import sample_candidates
        observation = {'pixels': np.zeros((1, 4, 2, 2, 3), dtype=np.uint8)}
        for mode in ('zscore', 'min_max'):
            policy = self.policy(mode)
            before = torch.random.get_rng_state().clone()
            flags = (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic)
            a = sample_candidates(policy, observation, 3, 42)
            self.assertTrue(torch.equal(before, torch.random.get_rng_state()))
            self.assertEqual(flags, (torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic))
            np.testing.assert_array_equal(a, sample_candidates(policy, observation, 3, 42))
            self.assertFalse(np.array_equal(a, sample_candidates(policy, observation, 3, 43)))
            self.assertFalse(np.array_equal(a[0], a[1]))
            with torch.random.fork_rng():
                torch.manual_seed(42)
                raw = policy.sample_action_sequences(observation, num_samples=3, denormalize=False)[0].numpy()
            expected = policy.process['action_cartesian'].inverse_transform(raw.reshape(-1, 8)).reshape(3, 6, 8)
            np.testing.assert_allclose(a, expected, rtol=1e-6)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_seed_and_rng_restoration(self):
        from offline_dp_eval import sample_candidates
        policy = self.policy(device='cuda')
        states = [s.clone() for s in torch.cuda.get_rng_state_all()]
        a = sample_candidates(policy, {'pixels': np.zeros((1, 4, 2, 2, 3))}, 2, 42)
        np.testing.assert_array_equal(a, sample_candidates(policy, {'pixels': np.zeros((1, 4, 2, 2, 3))}, 2, 42))
        self.assertTrue(all(torch.equal(a, b) for a, b in zip(states, torch.cuda.get_rng_state_all())))

    def test_alignment_boundary_history_files_and_unique_runs(self):
        from offline_dp_eval import evaluate_multi_sample
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'data.h5'; OfflineTests().dataset(path)
            # Nonzero step origin ensures timestep is matched by ID, not row number.
            with h5py.File(path, 'a') as f: f['step_idx'][2:] = [10, 11, 12]
            policy = self.policy()
            original = policy.sample_action_sequences
            observations = []
            def capture(info, **kwargs):
                observations.append(info)
                return original(info, **kwargs)
            policy.sample_action_sequences = capture
            results = []
            for step in (10, 11, 12, 12):
                result = evaluate_multi_sample(policy, path, 7, step, 3, 42,
                                               Path(tmp) / 'multi_sample', {'checkpoint_path': 'chosen.ckpt'})
                results.append(result)
                out = Path(result['metadata']['output_dir'])
                with h5py.File(out / 'multi_sample_rollout.h5') as f:
                    self.assertEqual(f['predicted_actions'].shape, (3, 6, 8))
                    expected_steps = step - 3 + np.arange(6)
                    np.testing.assert_array_equal(f['prediction_step_idx'], expected_steps)
                    valid = (expected_steps >= 10) & (expected_steps <= 12)
                    np.testing.assert_array_equal(f['ground_truth_valid_mask'], valid)
                    gt = f['ground_truth_actions'][:]
                    self.assertTrue(np.isnan(gt[~valid]).all())
                    np.testing.assert_allclose(gt[valid, 7], [.5, .75, 1])
                    self.assertEqual(f['selected_action_index'][()], 3)
                    self.assertTrue(np.isfinite(f['predicted_actions'][:]).all())
                doc = json.loads((out / 'multi_sample_metrics.json').read_text())
                self.assertEqual(doc['num_valid_steps'], 3)
                self.assertEqual(doc['metadata']['seed'], 42)
                self.assertEqual(doc['metadata']['checkpoint_path'], 'chosen.ckpt')
                for name in ('position', 'orientation', 'gripper'):
                    self.assertGreater((out / f'multi_sample_{name}.png').stat().st_size, 0)
            for info, expected in zip(observations, ([2]*4, [2, 2, 2, 3], [2, 2, 3, 4], [2, 2, 3, 4])):
                for key in ('pixels', 'wrist_pixels'):
                    np.testing.assert_array_equal(info[key][0, :, 0, 0, 0], expected)
            self.assertNotEqual(results[-1]['metadata']['run_id'], results[-2]['metadata']['run_id'])
            # Missing step_idx uses load_episode's synthetic indices.
            with h5py.File(path, 'a') as f: del f['step_idx']
            evaluate_multi_sample(policy, path, 7, 0, 1, 42, Path(tmp) / 'multi_sample', {})
            with self.assertRaisesRegex(ValueError, 'not found'):
                evaluate_multi_sample(policy, path, 7, 100, 1, 42, tmp, {})

    def test_metric_values_mask_and_quaternion_sign(self):
        from offline_dp_eval import compute_multi_sample_metrics
        gt = np.zeros((3, 8)); gt[:, 6] = 1; gt[0] = np.nan
        p = np.zeros((2, 3, 8)); p[:, :, 6] = -2
        p[1, :, :2] = [3, 4]; p[1, :, 7] = 2
        m = compute_multi_sample_metrics(p, gt, [False, True, True])
        self.assertEqual(m['position']['candidate_rmse_3d_m'], [0, 5])
        self.assertEqual(m['position']['rmse_3d_m'], dict(mean=2.5, min=0, max=5))
        self.assertEqual(m['gripper']['candidate_mae'], [0, 2])
        self.assertEqual(m['gripper']['std_per_step'], [1]*3)
        self.assertEqual(m['orientation']['candidate_angle_error_deg_per_step'], [[None, 0., 0.]]*2)
        self.assertEqual(m['orientation']['angle_error_deg']['max'], 0.)
        p[1, 1:, 3:7] = [0, 0, 1, 0]
        self.assertEqual(compute_multi_sample_metrics(p, gt, [False, True, True])['orientation']['angle_error_deg']['max'], 180)

    def test_sampler_validation_and_rng_on_exception(self):
        from offline_dp_eval import sample_candidates
        policy = self.policy()
        for k in (0, -1, 1.5, True):
            with self.assertRaisesRegex(ValueError, 'dry_run_num_candidates'):
                sample_candidates(policy, {}, k, 42)
        policy.sample_action_sequences = lambda *args, **kwargs: torch.zeros(1, 2, 6, 8)
        with self.assertRaisesRegex(ValueError, 'shape'): sample_candidates(policy, {}, 3, 42)
        policy.sample_action_sequences = lambda *args, **kwargs: torch.full((1, 3, 6, 8), float('nan'))
        with self.assertRaisesRegex(ValueError, 'NaN/Inf'): sample_candidates(policy, {}, 3, 42)
        def fail(*args, **kwargs):
            torch.rand(10)
            raise RuntimeError('sampling failed')
        policy.sample_action_sequences = fail
        before = torch.random.get_rng_state().clone()
        with self.assertRaisesRegex(RuntimeError, 'sampling failed'): sample_candidates(policy, {}, 3, 42)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_plot_style_axis_and_masked_gt(self):
        from offline_dp_eval import plot_multi_sample
        from unittest.mock import patch
        import matplotlib.pyplot as plt
        predicted = np.zeros((2, 6, 8))
        gt = np.ones((6, 8)); gt[[0, 5]] = np.nan
        original = plt.subplots
        figures = []
        def capture(*args, **kwargs):
            fig, axes = original(*args, **kwargs)
            figures.append((fig, axes))
            return fig, axes
        with tempfile.TemporaryDirectory() as tmp, patch.object(plt, 'subplots', side_effect=capture):
            plot_multi_sample(predicted, gt, 3, tmp, 7, 100)
        self.assertEqual([len(axes) for _, axes in figures], [3, 4, 1])
        for fig, axes in figures:
            self.assertIn('Episode 7, step 100, candidates 2', fig._suptitle.get_text())
            self.assertFalse(plt.fignum_exists(fig.number))
            for ax in axes[:, 0]:
                self.assertEqual([line.get_color() for line in ax.lines], ['gray', 'gray', 'red', 'black'])
                np.testing.assert_array_equal(ax.lines[0].get_xdata(), np.arange(6))
                self.assertTrue(np.isnan(ax.lines[2].get_ydata()[[0, 5]]).all())
                self.assertEqual(list(ax.lines[3].get_xdata()), [3, 3])
                self.assertEqual(ax.lines[3].get_linestyle(), '--')

    def test_multi_sample_runner_uses_configured_loader(self):
        import offline_dp_eval
        from unittest.mock import patch
        robot = SimpleNamespace(dry_run_dataset_path='data.h5', dry_run_episode_idx=7,
                                dry_run_timestep=100, dry_run_num_candidates=20,
                                dry_run_seed=42, output_dir='out')
        cfg = SimpleNamespace(eval=SimpleNamespace(real_robot=robot, img_size=224),
                              gpc=SimpleNamespace(diffusion_checkpoint='chosen.ckpt'))
        loader = Mock(return_value=SimpleNamespace(offline_checkpoint_metadata={'checkpoint_path': 'chosen.ckpt'}))
        with patch.object(offline_dp_eval, 'evaluate_multi_sample') as evaluator:
            offline_dp_eval.run_offline_dp_multi_sample(cfg, loader, 'existing_transform')
            loader.assert_called_once_with(checkpoint_path='chosen.ckpt', image_transform='existing_transform', device='cuda')
            self.assertEqual(evaluator.call_args.args[1:7], ('data.h5', 7, 100, 20, 42, Path('out/offline_dp/multi_sample')))


if __name__ == '__main__': unittest.main()
