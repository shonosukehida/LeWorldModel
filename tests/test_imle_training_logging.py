from pathlib import Path
"""CPU-only logging regression against the original epoch implementation."""
import copy
import subprocess
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import torch
from torch_ema import ExponentialMovingAverage
import train_imle_policy as trainer


class Policy(torch.nn.Linear):
    @property
    def model(self):
        return self

    @property
    def obs_encoder(self):
        return torch.nn.Identity()

    @property
    def wrist_obs_encoder(self):
        return torch.nn.Identity()

    def compute_loss(self, batch):
        loss = ((self(batch['action']) - torch.rand_like(batch['action'])) ** 2).mean()
        self.loss_metrics = {'imle': loss.detach(), 'gripper_l1': loss.detach() / 2,
                             'extra': 0.25}
        return loss


class LoggingTests(unittest.TestCase):
    def test_updates_match_original_and_axes_are_separate(self):
        source = subprocess.check_output(
            ['git', '-C', str(Path(trainer.__file__).resolve().parent), 'show',
             'a6bb419:train_imle_policy.py'], text=True)
        namespace = dict(trainer.__dict__)
        exec(source[source.index('def run_epoch('):source.index('@hydra.main')], namespace)
        original = namespace['run_epoch']
        torch.manual_seed(19)
        initial = Policy(2, 2)
        loader = [{'action': torch.ones(n, 2)} for n in (2, 1)]
        results = []
        for enabled in (None, False, True):
            policy = copy.deepcopy(initial)
            optimizer = torch.optim.AdamW(policy.parameters(), lr=0.01)
            ema = ExponentialMovingAverage(policy.parameters(), decay=0.9)
            run = Mock() if enabled else None
            logger = trainer.StepLogger(run)
            torch.manual_seed(7)
            summaries = []
            for epoch in range(2):
                args = (policy, loader, 'cpu', optimizer, ema)
                summaries.append(original(*args) if enabled is None else
                                 trainer.run_epoch(*args, step_logger=logger))
            results.append((summaries, copy.deepcopy(policy.state_dict()), ema.state_dict()))
            if enabled is not None:
                self.assertEqual(logger.global_step, 4)
                trainer.run_epoch(policy, loader, 'cpu', step_logger=logger)
                self.assertEqual(logger.global_step, 4)
            if enabled:
                payloads = [c.args[0] for c in run.log.call_args_list]
                self.assertEqual([p['global_step'] for p in payloads], [1, 2, 3, 4])
                self.assertEqual(set(payloads[0]), {'global_step', 'train/step_loss',
                    'train/step_imle', 'train/step_gripper_l1', 'train/step_extra'})
                definitions = {c.args[0]: c.kwargs for c in run.define_metric.call_args_list}
                self.assertEqual(definitions['train_loss'], {'step_metric': 'epoch'})
                self.assertEqual(definitions['val_loss'], {'step_metric': 'epoch'})
                self.assertEqual(definitions['train_eval_loss'], {'step_metric': 'epoch'})
                self.assertEqual(definitions['train_eval/*'], {'step_metric': 'epoch'})
                self.assertEqual(definitions['train/step_imle'], {'step_metric': 'global_step'})
        for summaries, state, ema in results[1:]:
            self.assertEqual(summaries, results[0][0])
            for key in state:
                self.assertTrue(torch.equal(state[key], results[0][1][key]))
            for actual, expected in zip(ema['shadow_params'], results[0][2]['shadow_params']):
                self.assertTrue(torch.equal(actual, expected))

    def test_checkpoint_path(self):
        for name, path in [('flip_mug/demo/push', '/unused.h5'),
                           (None, '/data/flip_mug/demo/push.h5')]:
            cfg = NS(data=NS(dataset=NS(name=name, path=path)))
            with patch.object(trainer.swm.data.utils, 'get_cache_dir', return_value='/tmp/cache'):
                output = trainer.checkpoint_directory(cfg)
            self.assertEqual(str(output.parent), '/tmp/cache/checkpoints/flip_mug/policy/imle')
            self.assertRegex(output.name, r'^\d{8}_\d{6}_\d{6}$')


class EvaluationTests(unittest.TestCase):
    def test_ema_snapshot_full_dataset_and_rng(self):
        for seed in (None, 37):
            policy = Policy(2, 2)
            ema = ExponentialMovingAverage(policy.parameters(), decay=0.9)
            with torch.no_grad():
                for p in policy.parameters():
                    p.add_(2)
            ordinary = [p.clone() for p in policy.parameters()]
            shadows = [p.clone() for p in ema.shadow_params]
            updates = ema.num_updates
            loader = torch.utils.data.DataLoader(
                [{'action': torch.ones(2) * i} for i in range(5)],
                batch_size=2, shuffle=False, drop_last=False,
            )
            seen = []
            compute = policy.compute_loss

            def check(batch):
                self.assertFalse(policy.training)
                self.assertFalse(torch.is_grad_enabled())
                for p, shadow in zip(policy.parameters(), shadows):
                    self.assertTrue(torch.equal(p, shadow))
                seen.extend(batch['action'][:, 0].tolist())
                return compute(batch)

            cpu_rng = torch.get_rng_state()
            cuda_rng = torch.cuda.get_rng_state_all()
            # Legacy validation consumes RNG only when no fixed seed is set.
            with ema.average_parameters(trainer.parameters(policy)):
                if seed is None:
                    trainer.run_epoch(policy, loader, 'cpu')
            expected_rng = torch.get_rng_state()
            torch.set_rng_state(cpu_rng)
            with patch.object(policy, 'compute_loss', side_effect=check):
                result = trainer.evaluate_ema(policy, loader, loader, 'cpu', ema, seed)
            self.assertEqual(seen, list(range(5)) * 2)
            self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))
            for actual, expected in zip(torch.cuda.get_rng_state_all(), cuda_rng):
                self.assertTrue(torch.equal(actual, expected))
            self.assertTrue(all(torch.isfinite(torch.tensor(r[0])) for r in result))
            for p, original in zip(policy.parameters(), ordinary):
                self.assertTrue(torch.equal(p, original))
                self.assertIsNone(p.grad)
            for actual, expected in zip(ema.shadow_params, shadows):
                self.assertTrue(torch.equal(actual, expected))
            self.assertEqual(ema.num_updates, updates)
            if seed is not None:
                self.assertEqual(result[0], result[1])
                torch.rand(13)
                self.assertEqual(result, trainer.evaluate_ema(
                    policy, loader, loader, 'cpu', ema, seed))
