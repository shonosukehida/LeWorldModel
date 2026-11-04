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
            ['git', 'show', 'HEAD:train_imle_policy.py'], text=True)
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
