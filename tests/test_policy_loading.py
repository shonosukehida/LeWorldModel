"""Execute production policy-loading branches without models, GPU or hardware."""
import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[1]


class Config(SimpleNamespace):
    def get(self, key, default=None):
        return getattr(self, key, default)


class PolicyLoadingTests(unittest.TestCase):
    def load(self, policy_type='world_model', *, random=False, random_encoder=False):
        tree = ast.parse((ROOT / 'eval_real_robot.py').read_text())
        run = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'run')
        start = next(i for i, n in enumerate(run.body) if isinstance(n, ast.Assign)
                     and any(isinstance(t, ast.Name) and t.id == 'policy' for t in n.targets))
        cfg = Config(policy='random' if random else 'world/checkpoint', plan_config={}, solver='solver',
                     eval=Config(probing=Config(use_random_encoder=random_encoder)),
                     gpc=Config(diffusion_checkpoint='diffusion.ckpt', num_candidates=8))
        if policy_type is not None:
            cfg.policy_type = policy_type
        model = Mock()
        model.to.return_value = model
        model.eval.return_value = model
        model.encoder.parameters.side_effect = lambda: iter([
            SimpleNamespace(device='cuda', dtype='float32')])
        replacement = Mock()
        replacement.to.return_value = replacement
        factories = Config(AutoCostModel=Mock(return_value=model), WorldModelPolicy=Mock(),
                           GPCPolicy=Mock(), RandomPolicy=Mock())
        ns = dict(cfg=cfg, swm=Config(policy=factories, PlanConfig=Mock()),
                  hydra=Config(utils=Config(instantiate=Mock())), torch=Mock(),
                  ViTModel=Mock(return_value=replacement), load_diffusion_policy=Mock(),
                  img_transform=Mock(return_value='image-transform'), process={}, transform={},
                  latent_goal_reward=object())
        exec(compile(ast.Module(body=run.body[start:start + 2], type_ignores=[]),
                     'eval_real_robot.py', 'exec'), ns)
        return ns, factories, model

    def test_diffusion_skips_world_checkpoint_and_random_encoder(self):
        for enabled in (False, True):
            ns, factories, model = self.load('diffusion', random_encoder=enabled)
            factories.AutoCostModel.assert_not_called()
            factories.WorldModelPolicy.assert_not_called()
            factories.GPCPolicy.assert_not_called()
            ns['ViTModel'].assert_not_called()
            ns['torch'].manual_seed.assert_not_called()
            model.to.assert_not_called()
            self.assertIsNone(ns['model'])
            ns['load_diffusion_policy'].assert_called_once_with(
                checkpoint_path='diffusion.ckpt', image_transform='image-transform', device='cuda')

    def test_world_model_and_default_load_world_checkpoint(self):
        for policy_type in ('world_model', None):
            ns, factories, model = self.load(policy_type)
            factories.AutoCostModel.assert_called_once_with('world/checkpoint')
            ns['hydra'].utils.instantiate.assert_called_once_with('solver', model=model)
            factories.WorldModelPolicy.assert_called_once_with(
                solver=ns['hydra'].utils.instantiate.return_value,
                config=ns['swm'].PlanConfig.return_value, process=ns['process'], transform=ns['transform'])
            ns['load_diffusion_policy'].assert_not_called()
            model.to.assert_called_once_with('cuda')
            model.eval.assert_called_once_with()
            model.requires_grad_.assert_called_once_with(False)
            self.assertTrue(model.interpolate_pos_encoding)

    def test_gpc_loads_both_and_passes_world_model(self):
        ns, factories, model = self.load('gpc')
        factories.AutoCostModel.assert_called_once_with('world/checkpoint')
        ns['load_diffusion_policy'].assert_called_once()
        factories.GPCPolicy.assert_called_once_with(
            diffusion_policy=ns['load_diffusion_policy'].return_value, world_model=model,
            reward_fn=ns['latent_goal_reward'], num_candidates=8,
            process=ns['process'], transform=ns['transform'])

    def test_random_encoder_is_preserved_for_world_model_and_gpc(self):
        for policy_type in ('world_model', 'gpc'):
            ns, factories, model = self.load(policy_type, random_encoder=True)
            factories.AutoCostModel.assert_called_once()
            ns['ViTModel'].assert_called_once()
            ns['torch'].manual_seed.assert_called_once_with(0)
            self.assertIs(model.encoder, ns['ViTModel'].return_value)
            model.encoder.to.assert_called_once_with(device='cuda', dtype='float32')
            model.encoder.eval.assert_called_once_with()

    def test_random_only_builds_random_policy(self):
        ns, factories, model = self.load('diffusion', random=True)
        factories.RandomPolicy.assert_called_once_with()
        factories.AutoCostModel.assert_not_called()
        ns['load_diffusion_policy'].assert_not_called()

    def test_unknown_policy_type_raises(self):
        with self.assertRaisesRegex(ValueError, 'Unknown policy_type: unknown'):
            self.load('unknown')


if __name__ == '__main__':
    unittest.main()
