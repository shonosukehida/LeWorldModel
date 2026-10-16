"""CPU-only upper training, episode leakage, compatible logs and checkpoint tests."""
import ast
from copy import deepcopy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import h5py
import lightning as pl
from lightning.pytorch.loggers import CSVLogger
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from module import SIGReg
from normalization_stats import load_normalization_process
from upper_data import (fit_training_process, prepare_upper_data, read_metadata,
                        split_episodes, UpperSequenceDataset)
from upper_training import (UpperTrainingModule, compute_upper_losses, load_upper_checkpoint,
                            serialize_process)
from test_upper_jepa import make_model

ROOT = Path(__file__).resolve().parents[1]


def config(h=4, k=3, idm=True):
    return dict(encoder_scale='tiny', img_size=28, patch_size=14, projector_hidden_dim=32,
                wm=dict(horizon=h, history_size=k, use_action_delta=True,
                        embed_dim=192, prop_embed_dim=32, prop_dim=8),
                predictor=dict(depth=1, heads=2, mlp_dim=32, dim_head=8, dropout=0., emb_dropout=0.),
                loss=dict(sigreg=dict(weight=.5, kwargs=dict(knots=5, num_proj=8)),
                          idm=dict(use=idm, weight=.1, hidden_dim=16)),
                optimizer=dict(type='AdamW', lr=1e-3, weight_decay=1e-3),
                scheduler=dict(type='LinearWarmupCosineAnnealingLR'))


class UpperTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / 'push.h5'
        self.length = 21
        n = self.length * 4
        self.actions = np.zeros((n, 8), dtype=np.float32)
        self.actions[:, 0] = np.arange(n)
        self.actions[:, 1] = np.arange(n) * .01
        self.actions[:, 6] = 1
        self.actions[:, 7] = (np.arange(n) % 3) * .5
        self.prop = np.tile(np.arange(n, dtype=np.float32)[:, None], (1, 8))
        self.episodes = np.repeat(np.arange(4, dtype=np.int32), self.length)
        with h5py.File(self.path, 'w') as file:
            images = np.broadcast_to((np.arange(n) % 255).astype(np.uint8)[:, None, None, None],
                                     (n, 28, 28, 3))
            file['pixels'] = images
            file['wrist_pixels'] = images
            file['action_cartesian'] = self.actions
            file['proprio'] = self.prop
            file['episode_idx'] = self.episodes
            file['step_idx'] = np.tile(np.arange(self.length, dtype=np.int32), 4)

    def prepare(self, name='run', h=4, k=3):
        return prepare_upper_data(self.path, self.root / name, horizon=h, history_size=k,
                                  train_split=.75, seed=3072, img_size=28)

    def test_time_indices_teacher_and_macro_for_multiple_horizons(self):
        class LastAction(nn.Module):
            def __init__(self, h):
                super().__init__()
                self.index = 8 * (h - 1)
            def forward(self, macro):
                return (macro[..., self.index:self.index+1] + 1).expand(-1, -1, 416)
        class Echo(nn.Module):
            def forward(self, emb, act):
                return act
        class Zero(nn.Module):
            def forward(self, emb):
                return emb.sum() * 0
        for h in (2, 3, 5):
            path, columns = read_metadata(self.path)
            data = UpperSequenceDataset(path, columns, [0], h, 3, lambda value: value)
            observations, action_rows = data.indices(0)
            np.testing.assert_array_equal(observations, [0, h, 2*h, 3*h])
            np.testing.assert_array_equal(action_rows, np.arange(3*h).reshape(3, h))
            batch = {key: value[None] for key, value in data[0].items()}
            model = make_model(h, predictor=Echo(), pred_proj=nn.Identity())
            model.encode = lambda value: {'emb': value['proprio'][..., :1].expand(-1, -1, 416)}
            model.action_encoder = LastAction(h)
            output = compute_upper_losses(model, Zero(), None, batch, sigreg_weight=0., idm_weight=0.)
            self.assertEqual(output['pred_loss'].item(), 0.)
            torch.testing.assert_close(output['predicted_emb'][0, :, 0], torch.tensor([h, 2*h, 3*h]).float())
        _, columns = read_metadata(self.path)
        data = UpperSequenceDataset(self.path, columns, [0], 4, 3, lambda value: value)
        np.testing.assert_array_equal(data.indices(0)[0], [0, 4, 8, 12])

    def test_only_final_prediction_is_supervised_for_any_history_and_horizon(self):
        class FixedPrediction(nn.Module):
            def __init__(self, k, target):
                super().__init__()
                values = torch.full((1, k, 416), 1000.)
                values[:, -1] = target
                values[:, -1, :192] += 1
                values[:, -1, 192:384] += 2
                values[:, -1, 384:] += 3
                self.values = nn.Parameter(values)
            def forward(self, z, a):
                return self.values
        class Zero(nn.Module):
            def forward(self, z):
                return z.sum() * 0
        for h, k in ((2, 1), (3, 2), (4, 3), (5, 3)):
            path, columns = read_metadata(self.path)
            data = UpperSequenceDataset(path, columns, [0], h, k, lambda value: value)
            batch = {key: value[None] for key, value in data[0].items()}
            model = make_model(h, history_size=k, predictor=FixedPrediction(k, k*h), pred_proj=nn.Identity())
            model.encode = lambda value: {'emb': value['proprio'][..., :1].expand(-1, -1, 416)}
            output = compute_upper_losses(model, Zero(), None, batch, sigreg_weight=0., idm_weight=0.)
            self.assertEqual(output['emb'][0, -1, 0].item(), k*h)
            for name, value in (('overhead_pred_loss', 1.), ('wrist_pred_loss', 4.), ('prop_pred_loss', 9.)):
                self.assertAlmostEqual(output[name].item(), value)
            self.assertAlmostEqual(output['pred_loss'].item(), (192+192*4+32*9)/416, places=6)
            output['pred_loss'].backward()
            self.assertTrue((model.predictor.values.grad[:, :-1] == 0).all())
            self.assertTrue((model.predictor.values.grad[:, -1] != 0).all())
            # Arbitrarily changing earlier predictions must not change any prediction loss.
            with torch.no_grad():
                model.predictor.values[:, :-1] *= -10
            changed = compute_upper_losses(model, Zero(), None, batch, sigreg_weight=0., idm_weight=0.)
            for name in ('pred_loss', 'overhead_pred_loss', 'wrist_pred_loss', 'prop_pred_loss'):
                torch.testing.assert_close(output[name], changed[name], rtol=0, atol=0)

    def test_idm_is_fresh_random_and_frozen(self):
        _, _, process, _ = self.prepare()
        statistics = serialize_process(process)
        first = UpperTrainingModule(config(), statistics)
        second = UpperTrainingModule(config(), statistics)
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in first.idm.parameters()))
        self.assertFalse(torch.equal(first.idm.net[0].weight, second.idm.net[0].weight))
        self.assertNotEqual(first.idm.net[0].weight.data_ptr(), second.idm.net[0].weight.data_ptr())

    def test_windows_never_cross_episodes_or_gaps_and_short_episodes_excluded(self):
        with h5py.File(self.path, 'a') as file:
            # Split episode 0 into two segments, each too short for H=4,K=3.
            steps = file['step_idx'][:]
            steps[10:21] += 2
            file['step_idx'][:] = steps
        path, columns = read_metadata(self.path)
        data = UpperSequenceDataset(path, columns, [0, 1, 2, 3], 4, 3, lambda value: value)
        self.assertFalse(np.isin(data.starts, np.arange(21)).any())
        for index in range(len(data)):
            observations, action_rows = data.indices(index)
            rows = np.arange(observations[0], observations[-1]+1)
            self.assertEqual(len(np.unique(columns['episode_idx'][rows])), 1)
            np.testing.assert_array_equal(np.diff(columns['step_idx'][rows]), np.ones(len(rows)-1))
            self.assertEqual(action_rows[-1, -1] + 1, observations[-1])
        with self.assertRaisesRegex(ValueError, 'No valid'):
            UpperSequenceDataset(path, columns, [0], 4, 3, lambda value: value)
        # No fabricated samples when the requested span exceeds an episode.
        with self.assertRaisesRegex(ValueError, 'No valid'):
            UpperSequenceDataset(path, columns, [1], 10, 3, lambda value: value)

    def test_episode_split_and_train_only_statistics(self):
        train, val, process, split = self.prepare()
        train_ids, val_ids = set(split['train_episodes']), set(split['validation_episodes'])
        self.assertFalse(train_ids & val_ids)
        self.assertEqual(train_ids | val_ids, {0, 1, 2, 3})
        mask = np.isin(self.episodes, list(train_ids))
        for key, values in (('action_cartesian', self.actions), ('proprio', self.prop)):
            np.testing.assert_allclose(process[key].mean_, values[mask].mean(0, keepdims=True))
        # Change validation absolute values AND within-validation deltas drastically.
        with h5py.File(self.path, 'a') as file:
            changed = self.actions.copy()
            changed[~mask, :3] += (np.arange((~mask).sum())[:, None]+1) * 10000
            file['action_cartesian'][:] = changed
            changed_prop = self.prop.copy()
            changed_prop[~mask] += 1e6
            file['proprio'][:] = changed_prop
        _, altered_val, again, again_split = self.prepare('second')
        self.assertEqual(split, again_split)
        for key in ('action_cartesian', 'action_delta', 'proprio'):
            np.testing.assert_array_equal(process[key].mean_, again[key].mean_)
            np.testing.assert_array_equal(process[key].scale_, again[key].scale_)
        sample = altered_val[0]
        obs, act = altered_val.indices(0)
        expected = (self.prop[obs] + 1e6 - process['proprio'].mean_) / process['proprio'].scale_
        np.testing.assert_allclose(sample['proprio'].numpy(), expected, rtol=1e-5)
        # Dataset leaves all Cartesian features physical.
        np.testing.assert_allclose(sample['action_cartesian'].numpy(), changed[act])
        restored, _ = load_normalization_process(self.root / 'run/process_stats.npz')
        self.assertNotIn('goal_action_delta', restored)
        self.assertEqual(set(train.episodes), train_ids)
        self.assertEqual(set(val.episodes), val_ids)
        self.assertTrue((self.root / 'run/episode_split.json').is_file())

    def test_boundary_validation_and_optional_step_idx(self):
        with h5py.File(self.path, 'a') as file:
            del file['step_idx']
        self.prepare('without_steps')
        with h5py.File(self.path, 'a') as file:
            del file['episode_idx']
        with self.assertRaisesRegex(ValueError, 'episode_idx'):
            read_metadata(self.path)
        with self.assertRaises(ValueError):
            split_episodes(np.zeros(5, dtype=int), .9, 1)

    def test_losses_backward_idm_gradients_and_rollout_preprocessing(self):
        train, _, process, split = self.prepare()
        batch = next(iter(DataLoader(train, batch_size=2)))
        module = UpperTrainingModule(config(), serialize_process(process), split)
        output = module(batch)
        self.assertEqual(output['idm_prediction'].shape, (2, 3, 56))
        torch.testing.assert_close(output['idm_prediction'],
                                   module.idm(output['emb'][:, :-1], output['emb'][:, 1:]))
        torch.testing.assert_close(output['idm_loss'],
                                   (output['idm_prediction'] - output['macro_action']).square().mean())
        # Frozen IDM must pass its input gradients through to every encoder.
        output['idm_loss'].backward(retain_graph=True)
        for network in (module.model.overhead_encoder, module.model.wrist_encoder,
                        module.model.prop_encoder):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in network.parameters()))
        self.assertTrue(all(not p.requires_grad and p.grad is None for p in module.idm.parameters()))
        fixed_idm = [p.detach().clone() for p in module.idm.parameters()]
        module.zero_grad(set_to_none=True)
        optimizer = torch.optim.AdamW((p for p in module.parameters() if p.requires_grad), lr=1e-3)
        for iteration in range(2):
            output = module(batch)
            self.assertTrue(all(torch.isfinite(value).all() for name, value in output.items() if 'loss' in name))
            output['loss'].backward()
            for name, network in (('overhead', module.model.overhead_encoder), ('wrist', module.model.wrist_encoder),
                                  ('proprio', module.model.prop_encoder), ('action', module.model.action_encoder),
                                  ('predictor', module.model.predictor)):
                grads = [p.grad for p in network.parameters() if p.grad is not None]
                self.assertTrue(grads, name)
                self.assertTrue(all(torch.isfinite(grad).all() for grad in grads), name)
                # AdaLN-zero initially blocks conditioning gradients; after one
                # update its gates open and Action Encoder receives nonzero grads.
                if iteration:
                    self.assertTrue(any(grad.abs().sum() > 0 for grad in grads), name)
            self.assertTrue(all(p.grad is None for p in module.idm.parameters()))
            optimizer.step()
            for before, after in zip(fixed_idm, module.idm.parameters()):
                torch.testing.assert_close(after, before, rtol=0, atol=0)
            optimizer.zero_grad(set_to_none=True)
        weighted = output['pred_loss'] + .5*output['sigreg_loss'] + .1*output['idm_loss']
        torch.testing.assert_close(output['loss'], weighted)
        torch.testing.assert_close(output['pred_loss'],
                                   (192*output['overhead_pred_loss'] + 192*output['wrist_pred_loss'] +
                                    32*output['prop_pred_loss'])/416)
        module.eval()
        captured = []
        handle = module.model.action_encoder.register_forward_pre_hook(lambda m, a: captured.append(a[0].detach().clone()))
        try:
            with torch.no_grad():
                train_output = module(batch)
                info = {key: batch[key][:, None, :-1] for key in ('pixels', 'wrist_pixels', 'proprio')}
                module.model.rollout(info, batch['action_cartesian'].reshape(2, 1, 12, 8))
        finally:
            handle.remove()
        torch.testing.assert_close(captured[0], captured[1])
        torch.testing.assert_close(train_output['predicted_emb'][:, -1], info['predicted_emb'][:, 0, -1])
        # Configurable IDM disable, including omission of the log key.
        disabled = UpperTrainingModule(config(idm=False), serialize_process(process))
        self.assertIsNone(disabled.idm)
        self.assertNotIn('idm_loss', disabled(batch))

    def test_log_names_and_frequencies_match_existing_forward(self):
        train, _, process, _ = self.prepare()
        batch = next(iter(DataLoader(train, batch_size=2)))
        module = UpperTrainingModule(config(), serialize_process(process))
        module.log_dict = Mock()
        source = ast.parse((ROOT / 'train.py').read_text())
        forward = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == 'lejepa_forward')
        statements = [node for node in forward.body if
                      (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in
                       ('losses_dict', 'step_losses_dict') for t in node.targets)) or
                      (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) and
                       isinstance(node.value.func, ast.Attribute) and node.value.func.attr == 'log_dict')]
        required = {'loss', 'pred_loss', 'overhead_pred_loss', 'wrist_pred_loss', 'prop_pred_loss', 'sigreg_loss', 'idm_loss'}
        for stage in ('fit', 'validate'):
            module.log_dict.reset_mock()
            output = module(batch, stage=stage)
            reference = SimpleNamespace(log_dict=Mock())
            exec(compile(ast.Module(body=statements, type_ignores=[]), 'reference_logging', 'exec'),
                 dict(self=reference, output=output, stage=stage))
            self.assertEqual(len(module.log_dict.call_args_list), 2)
            for actual, expected in zip(module.log_dict.call_args_list, reference.log_dict.call_args_list):
                self.assertEqual(actual.kwargs, expected.kwargs)
                self.assertEqual(set(actual.args[0]), set(expected.args[0]))
            self.assertEqual(set(module.log_dict.call_args_list[0].args[0]), {f'{stage}/{key}' for key in required})
            self.assertEqual(set(module.log_dict.call_args_list[1].args[0]), {f'{stage}/{key}_step' for key in required})

    def test_real_cpu_training_and_portable_checkpoint_round_trip(self):
        train, val, process, split = self.prepare()
        module = UpperTrainingModule(config(), serialize_process(process), split)
        train_loader = DataLoader(train, batch_size=2)
        val_loader = DataLoader(val, batch_size=2)
        trainer = pl.Trainer(accelerator='cpu', devices=1, precision='32-true', max_epochs=1,
                             limit_train_batches=2, limit_val_batches=1, num_sanity_val_steps=0,
                             gradient_clip_val=1., enable_progress_bar=False, enable_model_summary=False,
                             enable_checkpointing=False, log_every_n_steps=1,
                             logger=CSVLogger(str(self.root), name='logs'))
        before = [p.detach().clone() for p in module.idm.parameters()]
        model_before = module.model.prop_encoder.net[0].weight.detach().clone()
        trainer.fit(module, train_loader, val_loader)
        for a, b in zip(before, module.idm.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertFalse(b.requires_grad)
            self.assertIsNone(b.grad)
        self.assertFalse(torch.equal(model_before, module.model.prop_encoder.net[0].weight))
        optimizer_ids = {id(p) for group in trainer.optimizers[0].param_groups for p in group['params']}
        self.assertTrue(all(id(p) not in optimizer_ids for p in module.idm.parameters()))
        for stage in ('fit', 'validate'):
            for key in ('loss', 'pred_loss', 'overhead_pred_loss', 'wrist_pred_loss',
                        'prop_pred_loss', 'sigreg_loss', 'idm_loss'):
                self.assertIn(f'{stage}/{key}', trainer.callback_metrics)
        path = self.root / 'portable.ckpt'
        trainer.save_checkpoint(path)
        batch = next(iter(val_loader))
        module.eval()
        with torch.no_grad():
            expected = module(batch)['predicted_emb']
        # Remove dataset/stats to establish that inference is self-contained.
        (self.root / 'run/process_stats.npz').unlink()
        self.path.unlink()
        restored = load_upper_checkpoint(path)
        with torch.no_grad():
            actual = restored(batch)['predicted_emb']
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for a, b in zip(module.idm.parameters(), restored.idm.parameters()):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
            self.assertFalse(b.requires_grad)
            self.assertIsNone(b.grad)
        saved = torch.load(path, weights_only=True)
        self.assertEqual(saved['upper_jepa_metadata']['horizon'], 4)
        self.assertEqual(saved['upper_jepa_metadata']['normalization_stats_path'], 'process_stats.npz')
        self.assertEqual(saved['hyper_parameters']['split'], split)
        self.assertTrue(saved['optimizer_states'])
        # Restored observation preprocessing also uses the embedded train stats.
        raw = dict(pixels=torch.zeros(4, 28, 28, 3, dtype=torch.uint8),
                   wrist_pixels=torch.zeros(4, 28, 28, 3, dtype=torch.uint8), proprio=torch.zeros(4, 8))
        torch.testing.assert_close(module.observation_transform(raw)['proprio'],
                                   restored.observation_transform(raw)['proprio'])

    def test_missing_statistics_and_invalid_action_input_raise(self):
        train, _, process, _ = self.prepare()
        serialized = serialize_process(process)
        for key in ('action_cartesian', 'action_delta', 'proprio'):
            missing = deepcopy(serialized)
            del missing[key]
            with self.assertRaisesRegex(ValueError, key):
                UpperTrainingModule(config(), missing)
        bad = deepcopy(serialized)
        bad['action_delta']['scale_'] = [[0.]*8]
        with self.assertRaisesRegex(ValueError, 'action_delta'):
            UpperTrainingModule(config(), bad)
        module = UpperTrainingModule(config(), serialized)
        module.model.actions_normalized = True
        batch = next(iter(DataLoader(train, batch_size=2)))
        with self.assertRaisesRegex(ValueError, 'physical'):
            module(batch)
        with self.assertRaises(FileExistsError):
            fit_training_process(read_metadata(self.path)[1], [0], self.root / 'run/process_stats.npz')


if __name__ == '__main__':
    unittest.main()
