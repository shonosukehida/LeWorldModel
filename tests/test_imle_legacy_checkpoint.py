"""Opt-in numerical regression against the pre-IMLE trained DP checkpoint.

Set IMLE_LEGACY_DP_CHECKPOINT to an existing trusted checkpoint. Loading and
inference are CPU only; no robot module is imported. Baselines are immutable
repository revisions from the start of this implementation.
"""

import ast
import gc
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any
import unittest

import numpy as np
import torch
from torchvision import tv_tensors
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import stable_worldmodel as swm
from stable_worldmodel.diffusion import ConditionalUnet1D, ResNet18ObsEncoder, create_obs_encoder
from action_normalization import validate_action_stats, restore_action_processor
from normalization_stats import SafeStandardScaler

ROOT = Path(__file__).resolve().parents[1]
SWM = ROOT.parent / "stable-worldmodel"
LEWM_BASE = "cc06f7de7e1bc5dc6ee2440fca8131d39215a772"
SWM_BASE = "ab3e681039977bcd978989082f0b7654a90ef07b"


def source(repo, revision, file):
    return subprocess.check_output(
        ["git", "-C", str(repo), "show", f"{revision}:{file}"], text=True
    )


class LegacyCheckpointTests(unittest.TestCase):
    def test_legacy_algorithms_and_dispatch_source(self):
        for file in (
            "train_policy.py",
            "jepa.py",
            "offline_dp_eval.py",
            "action_normalization.py",
        ):
            self.assertEqual(source(ROOT, LEWM_BASE, file), (ROOT / file).read_text())
        before = ast.parse(source(SWM, SWM_BASE, "stable_worldmodel/policy.py"))
        after = ast.parse((SWM / "stable_worldmodel/policy.py").read_text())
        for name in ("DiffusionPolicy", "GPCPolicy", "WorldModelPolicy"):
            a = next(n for n in before.body if getattr(n, "name", None) == name)
            b = next(n for n in after.body if getattr(n, "name", None) == name)
            if name == "GPCPolicy":
                a.body = [n for n in a.body if getattr(n, "name", None) != "__init__"]
                b.body = [n for n in b.body if getattr(n, "name", None) != "__init__"]
            self.assertEqual(ast.dump(a), ast.dump(b), name)
        a = ast.parse(source(ROOT, LEWM_BASE, "eval_real_robot.py"))
        b = ast.parse((ROOT / "eval_real_robot.py").read_text())
        for name in ("load_diffusion_policy", "dp_img_transform", "_DPActionBuffer"):
            aa = next(n for n in a.body if getattr(n, "name", None) == name)
            bb = next(n for n in b.body if getattr(n, "name", None) == name)
            self.assertEqual(ast.dump(aa), ast.dump(bb), name)

    @unittest.skipUnless(
        os.environ.get("IMLE_LEGACY_DP_CHECKPOINT"),
        "Set IMLE_LEGACY_DP_CHECKPOINT for trained checkpoint test",
    )
    def test_trained_dp_physical_chunk_and_rng_exact(self):
        torch.set_num_threads(1)
        baseline = source(SWM, SWM_BASE, "stable_worldmodel/policy.py")
        ns = dict(torch=torch, np=np, Any=Any, tv_tensors=tv_tensors)
        nodes = [
            n
            for n in ast.parse(baseline).body
            if isinstance(n, ast.ClassDef)
            and n.name in ("BasePolicy", "DiffusionPolicy")
        ]
        exec(
            compile(ast.Module(body=nodes, type_ignores=[]), "baseline_policy", "exec"),
            ns,
        )
        outputs = []
        for old in (True, False):
            text = (
                source(ROOT, LEWM_BASE, "eval_real_robot.py")
                if old
                else (ROOT / "eval_real_robot.py").read_text()
            )
            node = next(
                n
                for n in ast.parse(text).body
                if getattr(n, "name", None) == "load_diffusion_policy"
            )
            namespace = dict(
                Path=Path,
                torch=torch,
                swm=SimpleNamespace(
                    policy=SimpleNamespace(DiffusionPolicy=ns["DiffusionPolicy"])
                )
                if old
                else swm,
                ConditionalUnet1D=ConditionalUnet1D,
                ResNet18ObsEncoder=ResNet18ObsEncoder,
                create_obs_encoder=create_obs_encoder,
                DDPMScheduler=DDPMScheduler,
                SafeStandardScaler=SafeStandardScaler,
                validate_action_stats=validate_action_stats,
                restore_action_processor=restore_action_processor,
            )
            exec(
                compile(
                    ast.Module(body=[node], type_ignores=[]),
                    "checkpoint_loader",
                    "exec",
                ),
                namespace,
            )
            torch.manual_seed(122)
            policy = namespace["load_diffusion_policy"](
                os.environ["IMLE_LEGACY_DP_CHECKPOINT"],
                lambda x: x.float() / 255,
                device="cpu",
            )
            info = {
                k: np.full((1, policy.obs_horizon, 32, 32, 3), 17, dtype="u1")
                for k in ("pixels", "wrist_pixels")
            }
            torch.manual_seed(42)
            outputs.append(
                (policy.get_action_chunk(info), torch.get_rng_state().clone())
            )
            del policy
            gc.collect()
        np.testing.assert_array_equal(outputs[0][0], outputs[1][0])
        self.assertTrue(torch.equal(outputs[0][1], outputs[1][1]))


if __name__ == "__main__":
    unittest.main()
