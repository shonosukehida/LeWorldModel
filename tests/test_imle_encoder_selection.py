"""Encoder selection, legacy ResNet numerical baseline and both full CPU paths."""

from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from imle_checkpoint import (
    build_policy,
    modules,
    parameters,
    save_checkpoint,
    load_imle_policy,
)
import test_imle_integration as integration

_BASE_CONFIG = integration.config


def convnet_config():
    cfg = _BASE_CONFIG()
    cfg.encoder.type = "convnet"
    return cfg


class ConvNetIntegrationTests(integration.IntegrationTests):
    """Run every existing end-to-end IMLE test with the new encoder selected."""

    def setUp(self):
        from unittest.mock import patch

        patcher = patch.object(integration, "config", side_effect=convnet_config)
        patcher.start()
        self.addCleanup(patcher.stop)


class EncoderSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_resnet_missing_and_explicit_type_match_before_change(self):
        fixture = np.load(
            Path(__file__).parent
            / "fixtures/imle_resnet18_before_encoder_selection.npz"
        )
        for kind in (None, "resnet18"):
            with self.subTest(kind=kind):
                cfg = _BASE_CONFIG()
                if kind is None:
                    cfg.encoder.pop("type")
                else:
                    cfg.encoder.type = kind
                torch.manual_seed(137)
                p = build_policy(cfg)
                for module in modules(p):
                    module.eval()
                batch = dict(
                    pixels=torch.linspace(-1, 1, 2 * 2 * 3 * 32 * 32).reshape(
                        2, 2, 3, 32, 32
                    ),
                    wrist_pixels=torch.linspace(1, -1, 2 * 2 * 3 * 32 * 32).reshape(
                        2, 2, 3, 32, 32
                    ),
                    action=torch.linspace(-0.9, 0.9, 2 * 16 * 8).reshape(2, 16, 8),
                )
                info = {k: v for k, v in batch.items() if k != "action"}
                torch.manual_seed(41)
                with torch.no_grad():
                    pred = p.predict_action(info)["action_pred"]
                np.testing.assert_array_equal(pred.numpy(), fixture["prediction"])
                np.testing.assert_array_equal(
                    torch.get_rng_state().numpy(), fixture["inference_rng"]
                )
                opt = torch.optim.AdamW(parameters(p), lr=1e-4)
                for module in modules(p):
                    module.train()
                torch.manual_seed(42)
                loss = p.compute_loss(batch)
                loss.backward()
                opt.step()
                np.testing.assert_array_equal(loss.detach().numpy(), fixture["loss"])
                np.testing.assert_array_equal(
                    torch.get_rng_state().numpy(), fixture["training_rng"]
                )
                for module in modules(p):
                    module.eval()
                torch.manual_seed(41)
                with torch.no_grad():
                    pred = p.predict_action(info)["action_pred"]
                np.testing.assert_array_equal(
                    pred.numpy(), fixture["after_step_prediction"]
                )

    def test_legacy_checkpoint_and_architecture_validation(self):
        cfg = _BASE_CONFIG()
        torch.manual_seed(137)
        p = build_policy(cfg)
        for module in modules(p):
            module.eval()
        info = {k: torch.zeros(1, 2, 3, 32, 32) for k in ("pixels", "wrist_pixels")}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.ckpt"
            save_checkpoint(
                path,
                p,
                torch.optim.AdamW(parameters(p)),
                0,
                1.0,
                cfg,
                integration.stats(),
            )
            state = torch.load(path, weights_only=False)
            state.pop("encoder_architecture")
            state["config"]["encoder"].pop("type")
            state["config"]["encoder"].pop("convnet", None)
            torch.save(state, path)
            loaded = load_imle_policy(path, lambda x: x, "cpu")
            # Test inputs here are already CHW/preprocessed tensors.
            loaded.transform = {}
            self.assertEqual(loaded.encoder_architecture["type"], "resnet18")
            torch.manual_seed(5)
            expected = p.sample_action_sequences(info, 2)
            torch.manual_seed(5)
            actual = loaded.sample_action_sequences(info, 2)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            state["encoder_architecture"] = dict(type="convnet")
            torch.save(state, path)
            with self.assertRaisesRegex(ValueError, "encoder architecture"):
                load_imle_policy(path, lambda x: x, "cpu")

    def test_custom_convnet_checkpoint_restores_saved_architecture(self):
        cfg = convnet_config()
        cfg.encoder.convnet.channels = [8, 16]
        cfg.encoder.convnet.kernel_sizes = [5, 3]
        cfg.encoder.convnet.strides = [2, 2]
        cfg.encoder.convnet.paddings = [2, 1]
        cfg.encoder.convnet.pool_size = 2
        cfg.imle.feature_dim = 16
        p = build_policy(cfg)
        for module in modules(p):
            module.eval()
        info = {k: torch.zeros(1, 2, 3, 48, 64) for k in ("pixels", "wrist_pixels")}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "convnet.ckpt"
            save_checkpoint(
                path,
                p,
                torch.optim.AdamW(parameters(p)),
                0,
                1.0,
                cfg,
                integration.stats(),
            )
            loaded = load_imle_policy(path, lambda x: x, "cpu")
            # Test inputs here are already CHW/preprocessed tensors.
            loaded.transform = {}
            self.assertEqual(loaded.encoder_architecture["feature_dim"], 16)
            self.assertEqual(
                loaded.encoder_architecture["convnet"]["channels"], [8, 16]
            )
            torch.manual_seed(5)
            expected = p.sample_action_sequences(info, 2)
            torch.manual_seed(5)
            actual = loaded.sample_action_sequences(info, 2)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_invalid_encoder_and_pretrained_convnet(self):
        cfg = _BASE_CONFIG()
        cfg.encoder.type = "invalid"
        with self.assertRaisesRegex(ValueError, "encoder.type"):
            build_policy(cfg)
        cfg.encoder.type = "convnet"
        cfg.encoder.pretrained = True
        with self.assertRaisesRegex(ValueError, "pretrained"):
            build_policy(cfg)
