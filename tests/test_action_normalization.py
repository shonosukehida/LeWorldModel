"""Offline tests: never import the robot entrypoint or connect hardware."""
import ast
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import numpy as np
import torch
from action_normalization import (fit_action_stats, validate_action_stats, restore_action_processor,
                                  ActionTransform, normalizer_artifact)
from normalization_stats import SafeStandardScaler

ROOT = Path(__file__).resolve().parents[1]


def extract(source, name, namespace):
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[node], type_ignores=[]), name, 'exec'), namespace)
    return namespace[name]


class ActionNormalizationTests(unittest.TestCase):
    def setUp(self):
        self.x = torch.tensor([[1., 3., 8.], [2., 5., 9.], [4., 7., 11.]])

    def fit(self, mode, x=None):
        return fit_action_stats(SimpleNamespace(get_col_data=lambda key: (self.x if x is None else x).numpy()), 'action', mode)

    def test_roundtrips_endpoints_and_shared_stats(self):
        for mode in ('zscore', 'min_max'):
            state = self.fit(mode)
            y = ActionTransform(state, 'action')({'action': self.x.clone()})['action']
            scaler = restore_action_processor(state, 3)
            np.testing.assert_allclose(scaler.inverse_transform(y.numpy()), self.x.numpy(), atol=1e-6)
            np.testing.assert_allclose(scaler.transform(self.x.numpy()), y.numpy(), atol=1e-6)
            if mode == 'min_max':
                torch.testing.assert_close(y.amin(0), -torch.ones(3))
                torch.testing.assert_close(y.amax(0), torch.ones(3))

    def test_constant_tiny_variance_and_range(self):
        x = torch.tensor([[2., 0., 0.], [2., 1e-12, 1e-38], [2., 2e-12, 2e-38]])
        for mode in ('zscore', 'min_max'):
            state = self.fit(mode, x)
            y = ActionTransform(state, 'action')({'action': x})['action']
            self.assertTrue(torch.isfinite(y).all())
            np.testing.assert_allclose(restore_action_processor(state, 3).inverse_transform(y.numpy()), x.numpy(), atol=1e-44, rtol=1e-6)
            if mode == 'zscore':
                torch.testing.assert_close(state['action_std'], torch.ones(1, 3))
            else:
                torch.testing.assert_close(y[:, 1:].amin(0), -torch.ones(2))
                torch.testing.assert_close(y[:, 1:].amax(0), torch.ones(2))

    def test_original_training_numeric_equality(self):
        for dtype in (torch.float32, torch.float64):
            x = self.x.to(dtype)
            state = self.fit('zscore', x)
            std = x.std(0, keepdim=True).clone()
            expected = ((x - x.mean(0, keepdim=True).clone()) / torch.where(std < 1e-4, torch.ones_like(std), std)).float()
            self.assertTrue(torch.equal(ActionTransform(state, 'action')({'action': x})['action'], expected))

    def test_invalid(self):
        bad = [dict(action_normalization='other'), {},
               dict(action_mean=torch.zeros(1, 2), action_std=torch.ones(1, 2)),
               dict(action_mean=torch.zeros(1, 3), action_std=torch.zeros(1, 3)),
               dict(action_mean=torch.full((1, 3), float('nan')), action_std=torch.ones(1, 3)),
               dict(action_normalization='min_max', action_min=torch.zeros(1, 3)),
               dict(action_normalization='min_max', action_min=torch.ones(1, 3), action_max=torch.zeros(1, 3)),
               dict(action_normalization='min_max', action_min=torch.zeros(1, 3), action_max=torch.full((1, 3), float('inf')))]
        for state in bad:
            with self.subTest(state=state), self.assertRaises(ValueError):
                restore_action_processor(state, 3)
        for mode in ('other', 'zscore', 'min_max'):
            with self.assertRaises(ValueError):
                self.fit(mode, torch.full((3, 3), float('nan')))

    def test_legacy_shapes_and_exact_inverse(self):
        for shape in ((3,), (1, 3)):
            state = dict(action_mean=torch.arange(3.).reshape(shape), action_std=torch.tensor([1., 1e-6, 2.]).reshape(shape))
            old = SafeStandardScaler(eps=1e-4)
            old.mean_, old.scale_ = state['action_mean'].numpy(), state['action_std'].numpy()
            new = restore_action_processor(state, 3)
            self.assertIs(type(new), SafeStandardScaler)
            np.testing.assert_array_equal(old.inverse_transform(self.x.numpy()), new.inverse_transform(self.x.numpy()))

    def test_actual_checkpoint_writer_and_artifact(self):
        from omegaconf import OmegaConf
        net = Mock(output_shape=lambda: [4], state_dict=lambda: {})
        policy = SimpleNamespace(model=net, obs_encoder=net, wrist_obs_encoder=net, action_dim=3, num_inference_steps=4)
        cfg = OmegaConf.create(dict(policy=dict(action_key='action', pred_horizon=4, obs_horizon=2, action_horizon=2),
                                    diffusion=dict(num_train_timesteps=4, beta_schedule='squaredcos_cap_v2', prediction_type='epsilon')))
        writer = extract((ROOT/'train_policy.py').read_text(), 'save_checkpoint', dict(Path=Path, DiffusionPolicy=object, torch=torch,
                         OmegaConf=OmegaConf, validate_action_stats=validate_action_stats))
        for mode in ('zscore', 'min_max'):
            state = self.fit(mode)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory)/'new.ckpt'
                writer(path, policy, net, 1, .1, state.get('action_mean'), state.get('action_std'), cfg, action_stats=state)
                loaded = torch.load(path, weights_only=False)
                artifact = normalizer_artifact(state, 'action')
                torch.save(artifact, Path(directory)/'action_normalizer.pt')
                artifact = torch.load(Path(directory)/'action_normalizer.pt', weights_only=False)
                for key, value in state.items():
                    if torch.is_tensor(value):
                        self.assertTrue(torch.equal(loaded[key], artifact[key]))
                self.assertEqual(validate_action_stats(loaded, 3), mode)
                np.testing.assert_array_equal(restore_action_processor(loaded, 3).inverse_transform(self.x.numpy()),
                                              restore_action_processor(artifact, 3).inverse_transform(self.x.numpy()))

    def test_dp_gpc_use_same_loader(self):
        tree = ast.parse((ROOT/'eval_real_robot.py').read_text())
        calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'load_diffusion_policy']
        self.assertEqual(len(calls), 2)
        for call in calls:
            self.assertIn('cfg.gpc.diffusion_checkpoint', ast.unparse(call))

    def test_actual_loader_branch_and_gpc_ownership(self):
        from stable_worldmodel.policy import GPCPolicy
        tree = ast.parse((ROOT/'eval_real_robot.py').read_text())
        loader = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'load_diffusion_policy')
        branch = next(n for n in loader.body if isinstance(n, ast.If) and 'normalization_mode' in ast.unparse(n.test))
        for mode in ('zscore', 'min_max'):
            state = self.fit(mode)
            namespace = dict(checkpoint=state, normalization_mode=mode, SafeStandardScaler=SafeStandardScaler,
                             restore_action_processor=restore_action_processor)
            state['action_dim'] = 3
            exec(compile(ast.Module(body=[branch], type_ignores=[]), 'loader_branch', 'exec'), namespace)
            processor = namespace['action_processor']
            dp = SimpleNamespace(process={'action': processor}, action_horizon=2)
            gpc = GPCPolicy(diffusion_policy=dp, world_model=Mock(), reward_fn=Mock())
            self.assertIs(gpc.diffusion_policy.process['action'], processor)
            expected = restore_action_processor(state, 3).inverse_transform(self.x.numpy())
            np.testing.assert_array_equal(gpc.diffusion_policy.process['action'].inverse_transform(self.x.numpy()), expected)

    def test_existing_checkpoint_model_regression(self):
        import stable_worldmodel as swm
        from stable_worldmodel.diffusion import ConditionalUnet1D, ResNet18ObsEncoder
        from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
        paths = sorted(Path('/home/hida/.stable_worldmodel/checkpoints').glob('*/policy/diffusion_policy/*/seed_*/*/epoch_001.ckpt'))
        if not paths:
            self.skipTest('No existing DP checkpoint available')
        namespace = dict(Path=Path, torch=torch, swm=swm, ConditionalUnet1D=ConditionalUnet1D,
                         ResNet18ObsEncoder=ResNet18ObsEncoder, DDPMScheduler=DDPMScheduler,
                         SafeStandardScaler=SafeStandardScaler, validate_action_stats=validate_action_stats,
                         restore_action_processor=restore_action_processor)
        baseline = subprocess.check_output(['git', 'show', 'HEAD:eval_real_robot.py'], cwd=ROOT, text=True)
        old_loader = extract(baseline, 'load_diffusion_policy', dict(namespace))
        new_loader = extract((ROOT/'eval_real_robot.py').read_text(), 'load_diffusion_policy', dict(namespace))
        torch.set_num_threads(2)
        outputs = []
        for loader in (old_loader, new_loader):
            policy = loader(paths[0], lambda x: x, device='cpu')
            self.assertIs(type(next(iter(policy.process.values()))), SafeStandardScaler)
            condition = torch.zeros(1, policy.pred_horizon, policy.action_dim)
            cond_dim = (policy.obs_encoder.output_shape()[0] + policy.wrist_obs_encoder.output_shape()[0])*policy.obs_horizon
            with torch.no_grad():
                trajectory = policy.conditional_sample(condition, torch.zeros(1, cond_dim), generator=torch.Generator().manual_seed(42))
            outputs.append(next(iter(policy.process.values())).inverse_transform(trajectory.numpy()))
            del policy
        np.testing.assert_array_equal(*outputs)
        print('Existing checkpoint regression (full DDPM + inverse):', paths[0])


if __name__ == '__main__':
    unittest.main()
