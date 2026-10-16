"""CPU composition tests: real JEPA rollouts with small mock encoders/predictors."""
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import lightning as pl
import numpy as np
from omegaconf import OmegaConf
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from hierarchical_world_model import HierarchicalWorldModel
from jepa import JEPA
from normalization_stats import SafeStandardScaler, save_normalization_process
from upper_jepa import UpperJEPA
from upper_training import UpperTrainingModule, serialize_process
from test_upper_training import config as upper_config


class TinyImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
    def forward(self, pixels, **kwargs):
        feature = pixels.mean((1, 2, 3))[:, None].expand(-1, 192) * self.scale
        return SimpleNamespace(last_hidden_state=feature[:, None])


class TinyProprioEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
    def forward(self, value):
        return value[..., :1].expand(*value.shape[:-1], 32) * self.scale


class TinyActionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))
    def forward(self, value):
        if value.shape[-1] != 8:
            raise ValueError('Lower requires eight Cartesian components')
        return value[..., :1].expand(*value.shape[:-1], 416) * self.scale


class AddPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.zeros(1, 3, 416))
    def forward(self, z, a):
        return z + a + self.pos_embedding[:, :z.shape[1]]


def process(action_mean=0., action_scale=1., prop_mean=0., prop_scale=1.):
    result = {}
    for key, mean, scale in (('action_cartesian', action_mean, action_scale),
                             ('proprio', prop_mean, prop_scale), ('action_delta', .25, 2.)):
        scaler = SafeStandardScaler()
        # Explicit synthetic fitted fixtures, not production fallbacks.
        scaler.mean_ = np.full((1, 8), mean, dtype=np.float32)
        scaler.scale_ = np.full((1, 8), scale, dtype=np.float32)
        scaler.raw_min_, scaler.raw_max_ = scaler.mean_ - scale, scaler.mean_ + scale
        scaler.normed_min_, scaler.normed_max_ = -np.ones((1, 8)), np.ones((1, 8))
        result[key] = scaler
    return result


def lower_model():
    return JEPA(TinyImageEncoder(), TinyImageEncoder(), AddPredictor(),
                TinyActionEncoder(), TinyProprioEncoder())


def upper_model(stats):
    return UpperJEPA(4, overhead_encoder=TinyImageEncoder(), wrist_encoder=TinyImageEncoder(),
                     prop_encoder=TinyProprioEncoder(), predictor=AddPredictor(),
                     overhead_projector=nn.Identity(), wrist_projector=nn.Identity(), pred_proj=nn.Identity(),
                     action_mean=stats['action_cartesian'].mean_, action_scale=stats['action_cartesian'].scale_,
                     delta_mean=stats['action_delta'].mean_, delta_scale=stats['action_delta'].scale_)


def inputs():
    info = dict(pixels=torch.full((2, 1, 3, 16, 16), 100, dtype=torch.uint8),
                wrist_pixels=torch.full((2, 1, 3, 16, 16), 110, dtype=torch.uint8),
                proprio=torch.full((2, 1, 8), 10.),
                goal=torch.full((2, 1, 3, 16, 16), 200, dtype=torch.uint8),
                goal_wrist_pixels=torch.full((2, 1, 3, 16, 16), 210, dtype=torch.uint8),
                goal_proprio=torch.full((2, 1, 8), 18.), nested={'preserve': [1, 2]})
    actions = torch.zeros(2, 3, 8, 8)
    actions[..., 0] = torch.arange(8.) + 12
    actions[..., 6] = 1
    actions[:, 1, :, 0] += 2
    actions[:, 2, :, 0] += 4
    return info, actions


class HierarchicalWorldModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def make(self, **kwargs):
        self.lower_process = process(1., 2., 2., 2.)
        self.upper_process = process(10., 4., 8., 4.)
        self.lower, self.upper = lower_model(), upper_model(self.upper_process)
        return HierarchicalWorldModel(self.lower, self.upper,
                                      lower_process=self.lower_process, upper_process=self.upper_process,
                                      lower_img_size=8, upper_img_size=12, **kwargs)

    def test_rollout_terminal_times_normalization_and_no_input_mutation(self):
        model = self.make()
        info, actions = inputs()
        before, actions_before = deepcopy(info), actions.clone()
        weights = {name: value.clone() for name, value in model.state_dict().items()}
        for layer in (self.lower, self.upper):
            layer.rollout = Mock(wraps=layer.rollout)
            layer.predict = Mock(wraps=layer.predict)
            layer.encode = Mock(wraps=layer.encode)
            layer.get_cost = Mock(side_effect=AssertionError('do not repeat rollout'))
        macros = []
        handle = self.upper.action_encoder.register_forward_pre_hook(lambda module, args: macros.append(args[0].clone()))
        model.train()  # Public inference restores both layers to eval mode.
        try:
            with torch.no_grad():
                result = model.get_cost(info, actions)
        finally:
            handle.remove()
        self.assertEqual(result['lower_rollout']['predicted_emb'].shape, (2, 3, 9, 416))
        self.assertEqual(result['upper_rollout']['predicted_emb'].shape, (2, 3, 3, 416))
        self.assertEqual(self.lower.predict.call_count, 8)
        self.assertEqual(self.upper.predict.call_count, 2)
        self.assertEqual([call.args[0].shape[1] for call in self.lower.predict.call_args_list], [1, 2, 3, 3, 3, 3, 3, 3])
        self.assertEqual([call.args[0].shape[1] for call in self.upper.predict.call_args_list], [1, 2])
        torch.testing.assert_close(self.lower.rollout.call_args.args[1], (actions-1)/2)
        torch.testing.assert_close(self.upper.rollout.call_args.args[1], actions)
        torch.testing.assert_close(macros[0], self.upper.build_macro_action(actions.reshape(6, 2, 4, 8)))
        for layer, img_size, current_prop, goal_prop in ((self.lower, 8, 4., 8.), (self.upper, 12, .5, 2.5)):
            self.assertFalse(layer.training)
            self.assertEqual(layer.rollout.call_count, 1)
            self.assertEqual(layer.encode.call_count, 2)  # current plus this layer's goal
            current = layer.encode.call_args_list[0].args[0]
            goal = layer.encode.call_args_list[1].args[0]
            self.assertEqual(current['pixels'].shape[-2:], (img_size, img_size))
            torch.testing.assert_close(current['proprio'], torch.full((2, 1, 8), current_prop))
            torch.testing.assert_close(goal['proprio'], torch.full((2, 1, 8), goal_prop))
            layer.get_cost.assert_not_called()
        self.assertFalse(torch.equal(result['lower_rollout']['goal_emb'], result['upper_rollout']['goal_emb']))
        for layer in ('lower', 'upper'):
            rollout = result[f'{layer}_rollout']
            expected = (rollout['predicted_emb'][:, :, -1:] - rollout['goal_emb']).square().sum((-1, -2))
            torch.testing.assert_close(result[f'{layer}_cost'], expected)
            self.assertEqual(result[f'{layer}_cost'].shape, (2, 3))
        torch.testing.assert_close(result['total_cost'], result['lower_cost']+result['upper_cost'])
        self.assertEqual(set(info), set(before))
        self.assertEqual(info['nested'], before['nested'])
        for key, value in before.items():
            if torch.is_tensor(value):
                torch.testing.assert_close(info[key], value, rtol=0, atol=0)
        torch.testing.assert_close(actions, actions_before, rtol=0, atol=0)
        for key, value in weights.items():
            torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)

    def test_fixed_cost_scales_and_weights(self):
        model = self.make(lower_weight=2., upper_weight=3., lower_cost_scale=4., upper_cost_scale=5.)
        low = torch.tensor([[4., 8., 12.], [16., 20., 24.]])
        up = torch.tensor([[5., 10., 15.], [20., 25., 30.]])
        self.lower.criterion = Mock(return_value=low)
        self.upper.criterion = Mock(return_value=up)
        with torch.no_grad():
            result = model.get_cost(*inputs())
        torch.testing.assert_close(result['normalized_lower_cost'], low/4)
        torch.testing.assert_close(result['normalized_upper_cost'], up/5)
        torch.testing.assert_close(result['total_cost'], 2*(low/4)+3*(up/5))
        self.lower.criterion.assert_called_once()
        self.upper.criterion.assert_called_once()

    def test_bad_scales_weights_costs_and_action_shapes(self):
        for name in ('lower_cost_scale', 'upper_cost_scale'):
            for value in (0., -1., float('nan'), float('inf')):
                with self.subTest(name=name, value=value), self.assertRaisesRegex(ValueError, name):
                    self.make(**{name: value})
        for name in ('lower_weight', 'upper_weight'):
            with self.assertRaisesRegex(ValueError, name):
                self.make(**{name: float('nan')})
        for layer in ('lower', 'upper'):
            for value in (float('nan'), float('inf')):
                model = self.make()
                getattr(model, f'{layer}_world_model').criterion = Mock(return_value=torch.full((2, 3), value))
                with self.assertRaisesRegex(ValueError, f'{layer}_cost'):
                    model.get_cost(*inputs())
        model = self.make()
        info, actions = inputs()
        model.lower_world_model.rollout = Mock(side_effect=AssertionError('must validate before rollout'))
        with self.assertRaisesRegex(ValueError, 'divisible'):
            model.rollout(info, actions[:, :, :-1])
        with self.assertRaisesRegex(ValueError, 'NaN/Inf'):
            model.rollout(info, torch.full_like(actions, float('nan')))
        with self.assertRaisesRegex(ValueError, 'positive'):
            model.rollout(info, actions[:, :, :0])
        info['pixels'] = info['pixels'][:, None].expand(2, 3, 1, 3, 16, 16)
        with self.assertRaisesRegex(ValueError, 'without a candidate axis'):
            model.rollout(info, actions)

    def test_missing_or_mismatched_statistics_and_shared_weights_rejected(self):
        model = self.make()
        for layer, key in (('lower', 'action_cartesian'), ('lower', 'proprio'), ('upper', 'action_delta')):
            lower_process, upper_process = deepcopy(self.lower_process), deepcopy(self.upper_process)
            del (lower_process if layer == 'lower' else upper_process)[key]
            with self.assertRaisesRegex(ValueError, key):
                HierarchicalWorldModel(self.lower, self.upper, lower_process=lower_process,
                                       upper_process=upper_process, lower_img_size=8, upper_img_size=12)
        inconsistent = deepcopy(self.upper_process)
        inconsistent['action_cartesian'].mean_ += 1
        with self.assertRaisesRegex(ValueError, 'does not match'):
            HierarchicalWorldModel(self.lower, self.upper, lower_process=self.lower_process,
                                   upper_process=inconsistent, lower_img_size=8, upper_img_size=12)
        self.upper.overhead_encoder = self.lower.overhead_encoder
        with self.assertRaisesRegex(ValueError, 'share'):
            HierarchicalWorldModel(self.lower, self.upper, lower_process=self.lower_process,
                                   upper_process=self.upper_process, lower_img_size=8, upper_img_size=12)
        model = self.make()
        self.upper.actions_normalized = True
        with self.assertRaisesRegex(ValueError, 'physical actions'):
            HierarchicalWorldModel(self.lower, self.upper, lower_process=self.lower_process,
                                   upper_process=self.upper_process, lower_img_size=8, upper_img_size=12)

    def test_original_lower_object_and_upper_lightning_checkpoint_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lower_path, upper_path, stats_path = root/'lower_epoch_1_object.ckpt', root/'upper.ckpt', root/'process_stats.npz'
            lower_stats, upper_stats = process(1., 2., 2., 2.), process(10., 4., 8., 4.)
            lower = lower_model()
            torch.save(lower, lower_path)  # Same object format as ModelObjectCallBack.
            save_normalization_process(stats_path, lower_stats, 'action_cartesian')
            OmegaConf.save(OmegaConf.create(dict(img_size=28, wm=dict(history_size=3, num_preds=1),
                                                 data=dict(dataset=dict(frameskip=1, keys_to_load=['action_cartesian'])))), root/'config.yaml')
            cfg = upper_config(h=4)
            upper = UpperTrainingModule(cfg, serialize_process(upper_stats))
            # Save through Lightning, including actual optimizer/scheduler state.
            sample = dict(pixels=torch.zeros(4, 3, 28, 28), wrist_pixels=torch.zeros(4, 3, 28, 28),
                          proprio=torch.zeros(4, 8), action_cartesian=torch.zeros(3, 4, 8))
            sample['action_cartesian'][..., 6] = 1
            loader = DataLoader([sample]*4, batch_size=2)
            trainer = pl.Trainer(accelerator='cpu', devices=1, max_epochs=1, logger=False,
                                 enable_checkpointing=False, enable_progress_bar=False,
                                 enable_model_summary=False, num_sanity_val_steps=0)
            trainer.fit(upper, loader)
            trainer.save_checkpoint(upper_path)
            loaded = HierarchicalWorldModel.from_checkpoints(lower_path, upper_path, stats_path)
            self.assertIsInstance(loaded.lower_world_model, JEPA)
            self.assertIsInstance(loaded.upper_world_model, UpperJEPA)
            for a, b in zip(lower.parameters(), loaded.lower_world_model.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            for a, b in zip(upper.model.parameters(), loaded.upper_world_model.parameters()):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            torch.testing.assert_close(loaded.lower_action_mean, torch.full((8,), 1.))
            torch.testing.assert_close(loaded.upper_world_model.action_mean, torch.full((8,), 10.))
            # Upper statistics come from its checkpoint, not the lower NPZ.
            with torch.no_grad():
                costs = loaded.get_cost(*inputs())
            self.assertEqual(costs['total_cost'].shape, (2, 3))
            self.assertTrue(torch.isfinite(costs['total_cost']).all())
            # Existing directory-style AutoCostModel lookup is also supported.
            from hydra.utils import instantiate
            factory = OmegaConf.load(Path(__file__).resolve().parents[1] / 'config/hierarchical_world_model.yaml')
            factory.lower_checkpoint, factory.upper_checkpoint = str(root), str(upper_path)
            factory.lower_stats_path, factory.lower_weight = str(stats_path), 2.
            second = instantiate(factory)
            self.assertEqual(second.lower_weight, 2.)
            torch.testing.assert_close(second.lower_action_scale, loaded.lower_action_scale)
            with self.assertRaises(FileNotFoundError):
                HierarchicalWorldModel.from_checkpoints(lower_path, upper_path, root/'missing.npz')
            wrong = root/'weights.ckpt'
            wrong.touch()
            with self.assertRaisesRegex(ValueError, 'object.ckpt'):
                HierarchicalWorldModel.from_checkpoints(wrong, upper_path, stats_path)
            cfg_path = root/'config.yaml'
            bad_cfg = OmegaConf.load(cfg_path)
            bad_cfg.data.dataset.frameskip = 2
            OmegaConf.save(bad_cfg, cfg_path)
            with self.assertRaisesRegex(ValueError, 'frameskip=1'):
                HierarchicalWorldModel.from_checkpoints(lower_path, upper_path, stats_path)


if __name__ == '__main__':
    unittest.main()
