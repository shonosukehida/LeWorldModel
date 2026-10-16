"""CPU-only upper JEPA tests; no checkpoint, robot, or image downloads."""
import sys
from pathlib import Path
from types import SimpleNamespace
import unittest
import tempfile
from unittest.mock import Mock

import numpy as np

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jepa import JEPA
from module import ARPredictor, Embedder, MLP
from upper_jepa import UpperJEPA


def image_template():
    class Image(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(3, 192)

        def forward(self, pixels, **kwargs):
            return SimpleNamespace(last_hidden_state=self.proj(pixels.mean((-1, -2)))[:, None])
    return Image()


def make_model(h=3, **kwargs):
    # Explicit synthetic statistics for geometry tests; never production defaults.
    for name, value in dict(action_mean=torch.zeros(8), action_scale=torch.ones(8),
                            delta_mean=torch.zeros(8), delta_scale=torch.ones(8)).items():
        kwargs.setdefault(name, value)
    kwargs.setdefault('predictor', ARPredictor(num_frames=3, depth=1, heads=2,
                      mlp_dim=32, input_dim=416, hidden_dim=32, output_dim=192, dim_head=8))
    return UpperJEPA(h, overhead_encoder=image_template(), wrist_encoder=image_template(), **kwargs).eval()


def actions(h, batch=2):
    result = torch.zeros(batch, h, 8)
    result[..., 6] = 1
    return result


class UpperJEPATests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_independent_modules(self):
        lower = JEPA(image_template(), image_template(), nn.Linear(416, 416),
                     Embedder(8, emb_dim=416), MLP(8, 64, 32),
                     MLP(192, 32, 192), MLP(192, 32, 192), MLP(416, 32, 416))
        names = ('overhead_encoder', 'wrist_encoder', 'prop_encoder',
                 'overhead_projector', 'wrist_projector', 'predictor', 'pred_proj')
        upper = UpperJEPA(3, **{name: getattr(lower, name) for name in names})
        self.assertIsInstance(upper, JEPA)
        for name in names:
            for a, b in zip(getattr(lower, name).parameters(), getattr(upper, name).parameters()):
                self.assertNotEqual(a.data_ptr(), b.data_ptr())
                torch.testing.assert_close(a, b)
        self.assertNotEqual(lower.action_encoder.patch_embed.weight.data_ptr(),
                            upper.action_encoder.patch_embed.weight.data_ptr())

    def test_macro_shapes_and_values(self):
        for h in (2, 3, 5):
            for use_delta in (True, False):
                model = make_model(h, use_action_delta=use_delta)
                a = actions(h)
                a[..., :3] = torch.arange(h)[None, :, None] * torch.tensor([1., 2., 3.])
                a[..., 7] = torch.arange(h) * .25
                macro = model.build_macro_action(a)
                self.assertEqual(macro.shape, (2, 8 * (2*h-1 if use_delta else h)))
                torch.testing.assert_close(macro[:, :8*h], a.flatten(1))
                if use_delta:
                    expected = torch.tensor([1., 2., 3., 0., 0., 0., 1., .25]).expand(2, h-1, 8)
                    torch.testing.assert_close(macro[:, 8*h:].reshape(2, h-1, 8), expected)

    def test_rotation_and_sign_equivalence(self):
        model = make_model(2)
        a = actions(2)
        # Noncommuting rotations: inv(Rx(90)) * Ry(90).
        a[:, 0, 3:7] = torch.tensor([1., 0., 0., 1.]) * 2
        a[:, 1, 3:7] = torch.tensor([0., 1., 0., 1.]) * 3
        expected = torch.tensor([-.5, .5, -.5, .5]).expand(2, 4)
        torch.testing.assert_close(model.build_macro_action(a)[:, 19:23], expected)
        for index in (0, 1):
            a[:, index, 3:7] *= -1
            torch.testing.assert_close(model.build_macro_action(a)[:, 19:23], expected)
        a = actions(2)
        a[:, 1, 3:7] = torch.tensor([-1., 0., 0., 0.])
        torch.testing.assert_close(model.build_macro_action(a)[:, 19:23], torch.tensor([1., 0., 0., 0.]).expand(2, 4))

    def test_normalization_shared_path_and_gradients(self):
        mean = torch.arange(8.) * .1
        scale = torch.arange(1., 9.)
        raw = actions(3)
        raw[:, :, 0] = torch.arange(3.)
        standardized = ((raw - mean) / scale).requires_grad_()
        normalized_model = make_model(actions_normalized=True, action_mean=mean, action_scale=scale)
        macro = normalized_model.build_macro_action(standardized)
        torch.testing.assert_close(macro, make_model(action_mean=mean, action_scale=scale).build_macro_action(raw))
        macro.sum().backward()
        self.assertTrue(torch.isfinite(standardized.grad).all())
        self.assertIn('action_mean', normalized_model.state_dict())
        with self.assertRaises(ValueError):
            make_model(actions_normalized=True, action_mean=None).build_macro_action(raw)

    def test_encode_predict_and_existing_jepa(self):
        model = make_model()
        info = dict(pixels=torch.randn(2, 3, 3, 4, 4),
                    wrist_pixels=torch.randn(2, 3, 3, 4, 4), proprio=torch.randn(2, 3, 8))
        info['action'] = model.build_macro_action(actions(3).unsqueeze(1).expand(-1, 3, -1, -1))
        encoded = model.encode(info)
        self.assertEqual(encoded['emb'].shape, (2, 3, 416))
        self.assertEqual(encoded['act_emb'].shape, (2, 3, 416))
        self.assertEqual(model.predict(encoded['emb'], encoded['act_emb']).shape, (2, 3, 416))
        for method in ('predict', 'criterion', 'get_cost'):
            self.assertIs(getattr(UpperJEPA, method), getattr(JEPA, method))
        lower = JEPA(model.overhead_encoder, model.wrist_encoder, model.predictor,
                     Embedder(8, emb_dim=416), model.prop_encoder,
                     model.overhead_projector, model.wrist_projector, model.pred_proj).eval()
        info['action'] = actions(3)
        self.assertEqual(lower.encode(info)['act_emb'].shape, (2, 3, 416))
        self.assertEqual(lower.predict(info['emb'], info['act_emb']).shape, (2, 3, 416))

    def test_rollout_time_alignment(self):
        class EchoAction(nn.Module):
            def forward(self, x):
                return x[..., :1].expand(*x.shape[:-1], 416)
        class Add(nn.Module):
            def __init__(self):
                super().__init__()
                self.calls = []
            def forward(self, z, a):
                self.calls.append((z.clone(), a.clone()))
                return z + a
        for h in (2, 3, 5):
            model = make_model(h, predictor=Add(), pred_proj=nn.Identity())
            model.action_encoder = EchoAction()
            # Supply known states at 0,H,2H; macro starts have x=1,2,4,8,16.
            model.encode = lambda info: {'emb': torch.tensor([0., 1., 3.]).view(1, 3, 1).expand(2, 3, 416)}
            info = dict(pixels=torch.zeros(2, 2, 3, 3, 2, 2),
                        wrist_pixels=torch.zeros(2, 2, 3, 3, 2, 2), proprio=torch.zeros(2, 2, 3, 8))
            a = actions(5*h).view(2, 1, 5*h, 8).expand(-1, 2, -1, -1).clone()
            a[:, :, ::h, 0] = torch.tensor([1., 2., 4., 8., 16.])
            a[:, 1, 2*h, 0] = 40
            out = model.rollout(info, a)['predicted_emb']
            self.assertEqual(out.shape, (2, 2, 6, 416))
            torch.testing.assert_close(out[0, 0, :, 0], torch.tensor([0., 1., 3., 7., 15., 31.]))
            torch.testing.assert_close(out[0, 1, :, 0], torch.tensor([0., 1., 3., 43., 51., 67.]))
            self.assertEqual(len(model.predictor.calls), 3)
            self.assertTrue(all(z.shape[1] == a.shape[1] == 3 for z, a in model.predictor.calls))
            with self.assertRaises(ValueError):
                model.rollout(info, a[:, :, :-1])

    def test_inherited_cost_and_single_transition(self):
        model = make_model(2)
        info = dict(pixels=torch.randn(2, 2, 1, 3, 4, 4),
                    wrist_pixels=torch.randn(2, 2, 1, 3, 4, 4),
                    proprio=torch.randn(2, 2, 1, 8),
                    goal=torch.randn(2, 2, 1, 3, 4, 4),
                    goal_wrist_pixels=torch.randn(2, 2, 1, 3, 4, 4),
                    goal_proprio=torch.randn(2, 2, 1, 8))
        a = actions(2).unsqueeze(1).expand(-1, 2, -1, -1)
        with torch.no_grad():
            cost = model.get_cost(info, a)
        self.assertEqual(cost.shape, (2, 2))
        self.assertTrue(torch.isfinite(cost).all())
        self.assertEqual(info['predicted_emb'].shape, (2, 2, 2, 416))

    def test_separate_statistics_match_existing_scaler(self):
        from normalization_stats import SafeStandardScaler
        mean = torch.tensor([1., -2., 3., .1, .2, .3, .4, .5])
        scale = torch.tensor([2., 3., 4., 5., 6., 7., 8., 9.])
        delta_mean = torch.arange(8.) / 10
        delta_scale = torch.arange(1., 9.) / 2
        for h in (2, 3, 5):
            raw = actions(h)
            raw[..., :3] = torch.arange(h)[None, :, None] * torch.tensor([.2, -.3, .4])
            raw[..., 7] = torch.arange(h) * .25
            for use_delta in (False, True):
                model = make_model(h, use_action_delta=use_delta, action_mean=mean,
                                   action_scale=scale, delta_mean=delta_mean, delta_scale=delta_scale)
                macro = model.build_macro_action(raw)
                scaler = SafeStandardScaler()
                scaler.mean_, scaler.scale_ = mean.numpy(), scale.numpy()
                np.testing.assert_allclose(macro[:, :8*h].numpy(),
                                           scaler.transform(raw.numpy()).reshape(2, -1))
                if use_delta:
                    physical_delta = torch.tensor([.2, -.3, .4, 0., 0., 0., 1., .25])
                    expected = ((physical_delta - delta_mean) / delta_scale).expand(2, h-1, 8)
                    torch.testing.assert_close(macro[:, 8*h:].reshape(2, h-1, 8), expected)
                normalized = make_model(h, use_action_delta=use_delta, actions_normalized=True,
                                        action_mean=mean, action_scale=scale,
                                        delta_mean=delta_mean, delta_scale=delta_scale)
                torch.testing.assert_close(macro, normalized.build_macro_action((raw - mean) / scale))

    def test_raw_absolute_quaternions_and_canonical_relative_rotation(self):
        from normalization_stats import SafeStandardScaler
        mean = torch.arange(8.) / 10
        scale = torch.arange(1., 9.)
        for use_delta in (False, True):
            model = make_model(2, use_action_delta=use_delta, action_mean=mean, action_scale=scale)
            normalized = make_model(2, use_action_delta=use_delta, action_mean=mean,
                                    action_scale=scale, actions_normalized=True)
            raw = actions(2)
            # Preserve nonunit magnitudes and signs in absolute features.
            raw[:, 0, 3:7] = torch.tensor([0., 0., 0., -2.])
            raw[:, 1, 3:7] = torch.tensor([-3., 0., 0., -3.])
            scaler = SafeStandardScaler()
            scaler.mean_, scaler.scale_ = mean.numpy(), scale.numpy()
            for index in (None, 0, 1):
                if index is not None:
                    raw[:, index, 3:7] *= -1
                macro = model.build_macro_action(raw)
                expected_absolute = scaler.transform(raw.numpy()).reshape(2, -1)
                np.testing.assert_allclose(macro[:, :16].numpy(), expected_absolute)
                torch.testing.assert_close(normalized.build_macro_action((raw - mean) / scale), macro)
                if use_delta:
                    # inv(identity) * Rx(90), independent of input magnitude/sign.
                    expected_rotation = torch.tensor([2**-.5, 0., 0., 2**-.5]).expand(2, 4)
                    torch.testing.assert_close(macro[:, 19:23], expected_rotation)

    def test_statistics_missing_or_invalid_stop_before_encoder(self):
        names = ('action_mean', 'action_scale', 'delta_mean', 'delta_scale')
        for name in names:
            for normalized in (False, True):
                model = make_model(**{name: None}, actions_normalized=normalized)
                with self.assertRaisesRegex(ValueError, name):
                    model.build_macro_action(actions(3))
                model.action_encoder.forward = Mock(side_effect=AssertionError('must not reach action encoder'))
                with self.assertRaisesRegex(ValueError, name):
                    model.encode({'action': torch.zeros(2, 3, 40)})
                model.action_encoder.forward.assert_not_called()
            for value in (torch.zeros(7), torch.zeros(2, 4), torch.full((8,), float('nan')),
                          torch.full((8,), float('inf'))):
                with self.assertRaisesRegex(ValueError, name):
                    make_model(**{name: value})
            # Revalidate checkpoint-loaded or externally modified buffers at use time.
            model = make_model()
            setattr(model, name, torch.full((8,), float('nan')))
            with self.assertRaisesRegex(ValueError, name):
                model.build_macro_action(actions(3))
        for name in ('action_scale', 'delta_scale'):
            for value in (torch.zeros(8), -torch.ones(8)):
                with self.assertRaisesRegex(ValueError, name):
                    make_model(**{name: value})
        model = make_model(2, use_action_delta=False, delta_mean=None, delta_scale=None)
        self.assertEqual(model.build_macro_action(actions(2)).shape, (2, 16))
        # Even H=1 requires delta stats when use_action_delta=True.
        with self.assertRaisesRegex(ValueError, 'delta_mean'):
            make_model(1, delta_mean=None).build_macro_action(actions(1))

    def test_stats_file_errors_and_round_trip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'upper_stats.npz'
            empty = dict(action_mean=None, action_scale=None, delta_mean=None, delta_scale=None)
            with self.assertRaises(FileNotFoundError):
                make_model(normalization_stats_path=path, **empty)
            np.savez(path, action_mean=np.zeros(8))
            with self.assertRaisesRegex(ValueError, 'action_scale'):
                make_model(normalization_stats_path=path, **empty)
            values = dict(action_mean=np.arange(8.) / 10, action_scale=np.arange(1., 9.),
                          delta_mean=np.arange(8.) / 5, delta_scale=np.arange(2., 10.))
            np.savez(path, **values)
            model = make_model(normalization_stats_path=path, **empty)
            torch.testing.assert_close(model.build_macro_action(actions(3)),
                                       make_model(**values).build_macro_action(actions(3)))
            for name in values:
                self.assertIn(name, model.state_dict())
            values['delta_scale'][0] = 0
            np.savez(path, **values)
            with self.assertRaisesRegex(ValueError, 'delta_scale'):
                make_model(normalization_stats_path=path, **empty)

    def test_training_and_rollout_use_identical_normalized_features(self):
        model = make_model(3, action_mean=torch.arange(8.) / 10,
                           action_scale=torch.arange(1., 9.), delta_mean=torch.arange(8.) / 5,
                           delta_scale=torch.arange(2., 10.))
        raw = actions(9).reshape(2, 1, 9, 8)
        raw[..., 0] = torch.arange(9.)
        info = dict(pixels=torch.randn(2, 1, 3, 3, 4, 4),
                    wrist_pixels=torch.randn(2, 1, 3, 3, 4, 4), proprio=torch.randn(2, 1, 3, 8))
        features = model.build_macro_action(raw.reshape(2, 3, 3, 8))
        captured = []
        handle = model.action_encoder.register_forward_pre_hook(lambda module, args: captured.append(args[0].clone()))
        try:
            model.encode({**{key: value[:, 0] for key, value in info.items()}, 'action': features})
            model.rollout(info, raw)
        finally:
            handle.remove()
        self.assertEqual(len(captured), 2)
        torch.testing.assert_close(captured[0], features)
        torch.testing.assert_close(captured[1], features)
        model.delta_scale = None
        with self.assertRaisesRegex(ValueError, 'delta_scale'):
            model.rollout(info, raw)
        self.assertEqual(len(captured), 2)

    def test_invalid_inputs_and_horizon_one(self):
        for h in (0, -1, True, 2.5):
            with self.assertRaises(ValueError):
                make_model(h)
        model = make_model(2)
        for a in (torch.zeros(2, 2, 8), torch.zeros(2, 3, 8), torch.full((2, 2, 8), float('nan'))):
            with self.assertRaises(ValueError):
                model.build_macro_action(a)
        torch.testing.assert_close(make_model(1).build_macro_action(actions(1)), actions(1).flatten(1))


if __name__ == '__main__':
    unittest.main()
