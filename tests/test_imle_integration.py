"""Actual IMLE CPU training, checkpoint, offline/GPC tests. No robot IO."""

from pathlib import Path
import ast
from collections import deque
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch
import h5py
import numpy as np
import torch
from omegaconf import OmegaConf
from stable_worldmodel.policy import GPCPolicy
from stable_worldmodel.reward import latent_goal_reward
from imle_checkpoint import (
    build_policy,
    save_checkpoint,
    load_imle_policy,
    modules,
    parameters,
)
from imle_runtime import validate_checkpoint_selection, validate_imle_action
from action_normalization import restore_action_processor
from offline_dp_eval import (
    evaluate_episode,
    evaluate_multi_sample,
    evaluate_chunk_rollout,
)
from offline_imle_eval import evaluate_imle_gpc

ROOT = Path(__file__).resolve().parents[1]


def config():
    cfg = OmegaConf.load(ROOT / "config/train/policy/imle_policy.yaml")
    cfg.imle.channels = [8, 16, 32]
    cfg.imle.gripper_channels = [8, 16, 32]
    cfg.imle.feature_dim = 8
    cfg.imle.n_samples_per_condition = 2
    cfg.policy.obs_horizon = 2
    cfg.policy.action_horizon = 2
    cfg.img_size = 32
    cfg.device = "cpu"
    cfg.loader.batch_size = 2
    cfg.loader.num_workers = 0
    cfg.wandb.enabled = False
    cfg.trainer.max_epochs = 1
    cfg.data = OmegaConf.load(ROOT / "config/train/policy/data/flip_mug.yaml")
    return cfg


def stats():
    return dict(
        action_normalization="min_max",
        action_min=torch.tensor([[0.4, -0.2, 0.1, -1.0, -1.0, -1.0, -1.0, 0.0]]),
        action_max=torch.tensor([[0.6, 0.2, 0.3, 1.0, 1.0, 1.0, 1.0, 1.0]]),
    )


def dataset(path, n=24):
    rng = np.random.default_rng(3)
    action = rng.uniform(-0.05, 0.05, (n, 8)).astype("f4")
    action[:, :3] += [0.5, 0, 0.2]
    action[:, 3:7] = [0, 0, 0, 1]
    action[:, 7] = np.linspace(0, 1, n)
    with h5py.File(path, "w") as f:
        for key in ("pixels", "wrist_pixels"):
            f[key] = rng.integers(0, 256, (n, 3, 32, 32), dtype="u1")
        f["action_cartesian"] = action
        f["proprio"] = action
        f["episode_idx"] = np.zeros(n, dtype="i8")
        f["step_idx"] = np.arange(n)
        f["ep_offset"] = [0]
        f["ep_len"] = [n]


class TinyWorldModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.seen = None

    def encode(self, info):
        return {"emb": info["proprio"]}

    def rollout(self, info, actions):
        self.seen = actions.clone()
        return {"predicted_emb": actions.cumsum(2)}


class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def policy(self):
        cfg = config()
        p = build_policy(cfg)
        for m in modules(p):
            m.eval()
        return p, cfg

    def test_checkpoint_ema_restore_optimizer_normalization(self):
        from torch_ema import ExponentialMovingAverage

        p, cfg = self.policy()
        optimizer = torch.optim.AdamW(parameters(p), lr=1e-4)
        batch = dict(
            pixels=torch.rand(2, 2, 3, 32, 32),
            wrist_pixels=torch.rand(2, 2, 3, 32, 32),
            action=torch.rand(2, 16, 8),
        )
        p.compute_loss(batch).backward()
        optimizer.step()
        ema = ExponentialMovingAverage(parameters(p), decay=0.999, use_num_updates=True)
        ema.update(parameters(p))
        transform = lambda x: x.float() / 255
        info = {
            k: np.full((1, 2, 32, 32, 3), 10, dtype="u1")
            for k in ("pixels", "wrist_pixels")
        }
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "imle.ckpt"
            save_checkpoint(path, p, optimizer, 3, 1.5, cfg, stats(), ema)
            saved = torch.load(path, weights_only=False)
            self.assertEqual(saved["noise_dim"], 7)
            self.assertEqual(saved["epoch"], 3)
            self.assertEqual(saved["policy_type"], "imle")
            other = torch.optim.AdamW(parameters(p))
            other.load_state_dict(saved["optimizer_state_dict"])
            self.assertEqual(len(other.state), len(optimizer.state))
            restored = load_imle_policy(path, transform, "cpu")
            p.process = {"action_cartesian": restore_action_processor(stats(), 8)}
            p.transform = {"pixels": transform, "wrist_pixels": transform}
            with ema.average_parameters(parameters(p)):
                torch.manual_seed(4)
                expected = p.sample_action_sequences(info, 3, True)
            torch.manual_seed(4)
            actual = restored.sample_action_sequences(info, 3, True)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.manual_seed(4)
            raw = restored.sample_action_sequences(info, 3, False)
            scaled = restored.process["action_cartesian"].transform(
                actual.numpy().reshape(-1, 8)
            )
            np.testing.assert_allclose(scaled, raw.numpy().reshape(-1, 8), atol=1e-6)
            torch.manual_seed(4)
            chunk = restored.get_action_chunk(info)
            torch.manual_seed(4)
            full = restored.sample_action_sequences(info, 1, True)
            np.testing.assert_allclose(chunk, full[:, 0, 1:3], atol=1e-7)
            saved["noise_dim"] = 99
            torch.save(saved, path)
            with self.assertRaisesRegex(ValueError, "noise dimension"):
                load_imle_policy(path, transform, "cpu")

    def test_actual_training_entrypoint(self):
        from train_imle_policy import run

        with tempfile.TemporaryDirectory() as tmp:
            cfg = config()
            path = Path(tmp) / "data.h5"
            dataset(path)
            cfg.data.dataset.path = str(path)
            cfg.output_dir = str(Path(tmp) / "output")
            run.__wrapped__(cfg)
            checkpoints = list(Path(cfg.output_dir).glob("*/latest.ckpt"))
            self.assertEqual(len(checkpoints), 1)
            state = torch.load(checkpoints[0], map_location="cpu", weights_only=False)
            self.assertTrue(np.isfinite(state["val_loss"]))
            self.assertEqual(state["epoch"], 0)
            self.assertIn("ema", state)
            self.assertGreater(len(state["optimizer_state_dict"]["state"]), 0)
            self.assertTrue(checkpoints[0].with_name("best.ckpt").exists())

    def test_all_offline_modes_actual_imle(self):
        p, cfg = self.policy()
        p.process = {"action_cartesian": restore_action_processor(stats(), 8)}
        p.transform = {k: lambda x: x.float() / 255 for k in ("pixels", "wrist_pixels")}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.h5"
            dataset(path, 3)
            out = Path(tmp) / "out"
            evaluate_episode(
                p, path, 0, out, {"policy_type": "imle"}, dp_execute_steps=1
            )
            self.assertTrue((out / "predicted_vs_gt_position.png").exists())
            evaluate_episode(p, path, 0, Path(tmp) / "chunk", {}, dp_execute_steps=2)
            result = evaluate_multi_sample(
                p, path, 0, 1, 3, 42, Path(tmp) / "multi", {}
            )
            self.assertTrue(Path(result["metadata"]["output_dir"]).exists())
            result = evaluate_chunk_rollout(p, path, 0, 3, 42, Path(tmp) / "roll", {})
            self.assertTrue(Path(result["metadata"]["output_dir"]).exists())

    def test_gpc_rollout_reward_selection_buffer_and_plots(self):
        p, cfg = self.policy()
        scaler = restore_action_processor(stats(), 8)
        p.process = {"action_cartesian": scaler}
        p.transform = {k: lambda x: x.float() / 255 for k in ("pixels", "wrist_pixels")}
        world = TinyWorldModel()
        identity = NS(transform=lambda x: x, inverse_transform=lambda x: x)
        process = {
            "action_cartesian": scaler,
            "proprio": identity,
            "goal_proprio": identity,
        }
        gpc = GPCPolicy(
            action_policy=p,
            world_model=world,
            reward_fn=latent_goal_reward,
            num_candidates=3,
            process=process,
        )
        self.assertIs(gpc.diffusion_policy, p)
        self.assertIs(gpc.action_policy, p)
        info = dict(
            pixels=np.zeros((1, 2, 32, 32, 3), dtype="u1"),
            wrist_pixels=np.zeros((1, 2, 32, 32, 3), dtype="u1"),
            proprio=np.zeros((1, 1, 8), dtype="f4"),
            goal=np.zeros((1, 1, 32, 32, 3), dtype="u1"),
            goal_proprio=np.zeros((1, 1, 8), dtype="f4"),
        )
        torch.manual_seed(5)
        candidates = p.sample_action_sequences(
            {k: info[k] for k in ("pixels", "wrist_pixels")}, 3, True
        )
        expected = gpc.normalize_action_for_world_model(candidates[:, :, 1:3])
        rewards = -(expected.cumsum(2)[:, :, -1] ** 2).sum(-1)
        torch.manual_seed(5)
        chunk = gpc.get_action_chunk(info)
        np.testing.assert_array_equal(
            chunk, candidates[:, rewards.argmax(1).item(), 1:3].numpy()
        )
        torch.testing.assert_close(world.seen, expected, rtol=0, atol=0)
        node = next(
            n
            for n in ast.parse((ROOT / "eval_real_robot.py").read_text()).body
            if getattr(n, "name", None) == "_DPActionBuffer"
        )
        ns = dict(deque=deque, np=np)
        exec(compile(ast.Module(body=[node], type_ignores=[]), "buffer", "exec"), ns)
        buffer = ns["_DPActionBuffer"](gpc, 2)
        with patch.object(
            gpc, "get_action_chunk", wraps=gpc.get_action_chunk
        ) as sampled:
            buffer.get_action(info)
            buffer.get_action(info)
            self.assertEqual(sampled.call_count, 1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "data.h5"
            dataset(path, 3)
            output = evaluate_imle_gpc(gpc, path, 0, Path(tmp) / "out", 2)
            self.assertEqual(output.shape, (3, 8))
            self.assertTrue((Path(tmp) / "out/gpc/data/candidates.npz").exists())
            self.assertTrue((Path(tmp) / "out/gpc/gpc_actions/step_0000.png").exists())

    def test_real_jepa_rollout_with_imle_proposals(self):
        from jepa import JEPA
        from transformers import ViTConfig, ViTModel

        cfg_vit = ViTConfig(
            image_size=32,
            patch_size=16,
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=4,
            intermediate_size=32,
        )

        class Predictor(torch.nn.Module):
            def forward(self, emb, act):
                return emb + act

        torch.manual_seed(72)
        world = JEPA(
            ViTModel(cfg_vit),
            ViTModel(cfg_vit),
            Predictor(),
            torch.nn.Linear(8, 48),
            torch.nn.Linear(8, 16),
        ).eval()
        p, cfg = self.policy()
        scaler = restore_action_processor(stats(), 8)
        p.process = {"action_cartesian": scaler}
        image_transform = lambda x: x.float() / 255
        p.transform = {k: image_transform for k in ("pixels", "wrist_pixels")}
        identity = NS(transform=lambda x: x, inverse_transform=lambda x: x)
        gpc = GPCPolicy(
            action_policy=p,
            world_model=world,
            reward_fn=latent_goal_reward,
            num_candidates=3,
            process={
                "action_cartesian": scaler,
                "proprio": identity,
                "goal_proprio": identity,
            },
            transform={
                k: image_transform
                for k in ("pixels", "wrist_pixels", "goal", "goal_wrist_pixels")
            },
        )
        image = np.zeros((1, 1, 32, 32, 3), dtype="u1")
        info = dict(
            pixels=image,
            wrist_pixels=image,
            goal=image,
            goal_wrist_pixels=image,
            proprio=np.zeros((1, 1, 8), dtype="f4"),
            goal_proprio=np.ones((1, 1, 8), dtype="f4"),
        )
        history = {k: np.repeat(image, 2, axis=1) for k in ("pixels", "wrist_pixels")}
        with torch.no_grad():
            torch.manual_seed(7)
            proposals = p.sample_action_sequences(history, 3, True)
            wm_actions = gpc.normalize_action_for_world_model(proposals[:, :, 1:3])
            output = gpc.rollout(info, wm_actions)
            self.assertEqual(output["predicted_emb"].shape, (1, 3, 3, 48))
            rewards = latent_goal_reward(output, info)
            torch.manual_seed(7)
            selected = gpc.get_action_chunk(info, dp_info_dict=history)
        np.testing.assert_array_equal(
            selected, proposals[:, rewards.argmax(1).item(), 1:3].numpy()
        )

    def test_checkpoint_combinations_and_safety(self):
        for kind, dp, imle, valid in [
            ("diffusion", "old", None, True),
            ("imle", None, "new", True),
            ("gpc", "old", None, True),
            ("gpc", None, "new", True),
            ("gpc", None, None, False),
            ("gpc", "old", "new", False),
            ("imle", None, None, False),
            ("imle", None, "none", False),
            ("gpc", "old", "null", False),
            ("imle", "null", "new", False),
        ]:
            cfg = OmegaConf.create(
                dict(
                    policy_type=kind,
                    gpc=dict(diffusion_checkpoint=dp, imle_checkpoint=imle),
                )
            )
            if valid:
                validate_checkpoint_selection(cfg)
            else:
                with self.assertRaises(ValueError):
                    validate_checkpoint_selection(cfg)
        validate_checkpoint_selection(
            OmegaConf.create(
                {"policy_type": "gpc", "gpc": {"diffusion_checkpoint": "old"}}
            )
        )
        ee = np.array([0.5, 0, 0.2, 0, 0, 0, 1.0])
        action = np.r_[ee, 0.5]
        env = NS(
            cfg=NS(
                workspace_bounds_m=[[0.4, 0.6], [-0.2, 0.2], [0.1, 0.3]],
                max_cartesian_delta_m=0.02,
            ),
            clip_cartesian_action=lambda raw, current_ee: np.array(
                [0.51, 0, 0.2, 0, 0, 0, 1, 1.0]
            ),
        )
        validate_imle_action(action, env, ee)
        for raw in (np.full(8, np.nan), np.zeros(8), np.zeros(9)):
            with self.assertRaises(ValueError):
                validate_imle_action(raw, env, ee)
        env.clip_cartesian_action = lambda raw, current_ee: np.r_[
            ee[:3] + 0.1, ee[3:], 0.5
        ]
        with self.assertRaises(ValueError):
            validate_imle_action(action, env, ee)


if __name__ == "__main__":
    unittest.main()
