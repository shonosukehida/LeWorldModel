from unittest.mock import patch
import subprocess

import h5py
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from stable_worldmodel.subgoal import SubgoalMLP
import train_subgoal as training
from test_imle_integration import dataset, ROOT


def test_training_config_matches_previous_defaults(tmp_path):
    baseline = "dc3fe8ba1edeea50e7d317b38656a4c4433bad86"
    for name in ("train_subgoal.yaml", "flip_mug.yaml", "launcher/local.yaml", "solver/cem.yaml"):
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(subprocess.check_output(
            ["git", "show", f"{baseline}:config/eval/{name}"], cwd=ROOT))
    with initialize_config_dir(version_base=None, config_dir=str(tmp_path)):
        previous = compose(config_name="train_subgoal")
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "config/train")):
        current = compose(config_name="train_subgoal")
        overridden = compose(config_name="train_subgoal", overrides=[
            "+eval.real_robot.dry_run_dataset_path=/different/dataset.h5"])

    for key in ("seed", "policy", "eval.img_size", "eval.normalization_stats_path",
                "eval.real_robot.goal_image_path", "eval.real_robot.goal_wrist_image_path",
                "eval.real_robot.goal_proprio"):
        assert OmegaConf.to_container(OmegaConf.create({"value": OmegaConf.select(previous, key)}),
                                      resolve=True) == OmegaConf.to_container(
            OmegaConf.create({"value": OmegaConf.select(current, key)}), resolve=True)
    assert set(current.subgoal) == set(previous.subgoal)
    for key in current.subgoal:
        if key == "output_path":
            assert OmegaConf.to_container(previous.subgoal, resolve=False)[key] == (
                OmegaConf.to_container(current.subgoal, resolve=False)[key])
        else:
            assert current.subgoal[key] == previous.subgoal[key]
    assert overridden.subgoal.dataset_path == current.subgoal.dataset_path


def test_pairs_use_episode_and_time_even_for_interleaved_rows(tmp_path):
    path = tmp_path / "demo.h5"
    dataset(path, n=6)
    with h5py.File(path, "r+") as f:
        f["episode_idx"][:] = [3, 2, 3, 2, 3, 2]
        f["step_idx"][:] = [4, 1, 3, 3, 5, 2]
        pairs = training.training_pairs(f, [1, 2])
        assert set(map(tuple, pairs.tolist())) == {
            (0, 4, 1), (2, 0, 1), (2, 4, 2), (1, 5, 1), (5, 3, 1), (1, 3, 2)
        }


def test_training_uses_measured_multiview_states_and_one_goal(tmp_path):
    from test_gpc_opt_integration import components
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "config/train")):
        cfg = compose(config_name="train_subgoal")
    path = tmp_path / "demo.h5"
    dataset(path, n=5)
    cfg.subgoal.update(dict(dataset_path=str(path), output_path=str(tmp_path / "mlp.pt"),
                            horizons=[1, 3], epochs=1, encode_batch_size=2, batch_size=2,
                            hidden_dim=5, device="cpu"))
    cfg.eval.img_size = 32
    for name in ("goal_image_path", "goal_wrist_image_path"):
        cfg.eval.real_robot[name] = str(ROOT / cfg.eval.real_robot[name])
    _, wm = components()
    wm.train().requires_grad_(True)
    before = {k: v.clone() for k, v in wm.state_dict().items()}
    encodings = []
    def capture(info):
        assert not wm.training and not torch.is_grad_enabled()
        assert set(info) == {"pixels", "wrist_pixels", "proprio"}
        latent = original(info)
        encodings.append(latent["emb"][:, 0].clone())
        return latent
    original = wm.encode
    # Keep the real JEPA.encode and the trainer. Reject all predictor calls.
    with (patch.object(training, "AutoCostModel", return_value=wm),
          patch.object(training, "load_normalization_process", return_value=({"proprio": Identity()}, None)),
          patch.object(wm, "encode", side_effect=capture),
          patch.object(wm.predictor, "forward", side_effect=AssertionError("predictor used")),
          patch.object(SubgoalMLP, "forward", autospec=True, side_effect=SubgoalMLP.forward) as forward,
          patch.object(torch.nn.functional, "mse_loss", wraps=torch.nn.functional.mse_loss) as mse):
        network = training.train(cfg)
    assert all(torch.equal(before[k], v) for k, v in wm.state_dict().items())
    assert all(p.grad is None and not p.requires_grad for p in wm.parameters())
    assert [len(z) for z in encodings] == [1, 2, 2, 1]
    measured = torch.cat(encodings[1:])
    visited = set()
    for call, loss in zip(forward.call_args_list, mse.call_args_list):
        _, current, goal, horizons = call.args
        assert torch.equal(goal, encodings[0].expand_as(goal))
        for z, h, target in zip(current, horizons.tolist(), loss.args[1]):
            row = int(torch.where((measured == z).all(-1))[0].item())
            assert torch.equal(target, measured[row + h])
            visited.add((row, h))
    assert visited == {(t, h) for h in (1, 3) for t in range(5 - h)}
    restored = SubgoalMLP.load(cfg.subgoal.output_path, measured.shape[-1], "cpu")
    assert all(torch.equal(a, b) for a, b in zip(network.parameters(), restored.parameters()))


class Identity:
    def transform(self, value):
        return value
