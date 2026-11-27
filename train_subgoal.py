"""Train a horizon-conditioned subgoal MLP from measured demonstration states."""
from pathlib import Path
import tempfile

import h5py
import hydra
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from normalization_stats import load_normalization_process
from offline_dp_eval import dataset_images_to_hwc
from saved_robot_goal import load_saved_goal, validate_goal_config
from stable_worldmodel.policy import AutoCostModel, BasePolicy
from stable_worldmodel.subgoal import SubgoalMLP
from utils import get_eval_img_preprocessor


def training_pairs(dataset, horizons):
    """Return storage row indices for (episode, t) and (episode, t+H)."""
    if not horizons or any(type(h) is not int or h < 1 for h in horizons):
        raise ValueError("horizons must contain positive integers")
    if len(set(horizons)) != len(horizons):
        raise ValueError("horizons must be unique")
    episodes = np.asarray(dataset["episode_idx"]).reshape(-1)
    steps = np.asarray(dataset["step_idx"]).reshape(-1)
    if (not len(steps) or len(steps) != len(episodes)
            or steps.dtype.kind not in 'iu' or episodes.dtype.kind not in 'iu'):
        raise ValueError("Expected aligned integer episode_idx and step_idx columns")
    for key in ("pixels", "wrist_pixels", "proprio"):
        if len(dataset[key]) != len(steps):
            raise ValueError(f"Unaligned demonstration column: {key}")
    pairs = []
    for episode in np.unique(episodes):
        rows = np.flatnonzero(episodes == episode)
        rows = rows[np.argsort(steps[rows])]
        times = steps[rows].astype(np.int64)
        if np.any(np.diff(times) == 0):
            raise ValueError(f"Duplicate step_idx in episode {episode}")
        for h in horizons:
            targets = times + h
            positions = np.searchsorted(times, targets)
            valid = positions < len(rows)
            sources = rows[valid]
            ends = rows[positions[valid]]
            matched = steps[ends] == targets[valid]
            if matched.any():
                pairs.append(np.stack((sources[matched], ends[matched],
                                       np.full(matched.sum(), h)), axis=1))
    if not pairs:
        raise ValueError("No within-episode demonstration pairs for the configured horizons")
    return torch.from_numpy(np.concatenate(pairs))


@torch.no_grad()
def encode_demonstrations(wm, prep, dataset, goal, batch_size):
    device = next(wm.parameters()).device

    def encode(overhead, wrist, proprio):
        info = prep._prepare_info(dict(pixels=overhead[:, None], wrist_pixels=wrist[:, None],
                                       proprio=proprio[:, None]))
        return wm.encode({k: v.to(device) for k, v in info.items()})["emb"][:, 0].cpu()

    final_goal = encode(*(x[None] for x in goal))
    latents = []
    for start in range(0, len(dataset["episode_idx"]), batch_size):
        batch = slice(start, start + batch_size)
        images = [dataset_images_to_hwc(dataset[key][batch], key) for key in ("pixels", "wrist_pixels")]
        latents.append(encode(*images, np.asarray(dataset["proprio"][batch], dtype=np.float32)))
    return torch.cat(latents), final_goal


def train(cfg):
    options = cfg.subgoal
    for name in ("epochs", "batch_size", "encode_batch_size"):
        if type(options[name]) is not int or options[name] < 1:
            raise ValueError(f"{name} must be a positive integer")
    torch.manual_seed(cfg.seed)
    validate_goal_config(cfg.eval.real_robot)
    goal = load_saved_goal(cfg.eval.real_robot)
    wm = AutoCostModel(cfg.policy).to(options.device).eval().requires_grad_(False)
    wm.interpolate_pos_encoding = True
    process, _ = load_normalization_process(Path(cfg.eval.normalization_stats_path).expanduser())
    if "proprio" not in process:
        raise ValueError("WM statistics must include proprio")
    image_transform = get_eval_img_preprocessor(cfg.eval.img_size)
    prep = BasePolicy(process=process, transform=dict(pixels=image_transform, wrist_pixels=image_transform))
    with h5py.File(Path(options.dataset_path).expanduser(), "r") as dataset:
        pairs = training_pairs(dataset, list(options.horizons))
        latents, final_goal = encode_demonstrations(wm, prep, dataset, goal, options.encode_batch_size)
    del wm
    final_goal = final_goal.to(options.device)
    network = SubgoalMLP(latents.shape[-1], options.hidden_dim).to(options.device)
    optimizer = torch.optim.Adam(network.parameters(), lr=options.learning_rate)
    batches = DataLoader(TensorDataset(*pairs.unbind(dim=1)), batch_size=options.batch_size, shuffle=True)
    for epoch in range(options.epochs):
        total = 0.0
        for rows, ends, horizons in batches:
            current = latents[rows].to(options.device)
            predicted = network(current, final_goal.expand_as(current), horizons)
            loss = torch.nn.functional.mse_loss(predicted, latents[ends].to(options.device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.detach().item() * len(rows)
        print(f"epoch {epoch + 1}: MSE={total / len(pairs):.8g} ({len(pairs)} pairs)", flush=True)

    output = Path(options.output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, delete=False) as tmp:
        staging = Path(tmp.name)
    try:
        torch.save(network.checkpoint(), staging)
        staging.replace(output)
    finally:
        staging.unlink(missing_ok=True)
    print(f"Saved subgoal MLP to {output}")
    return network


@hydra.main(version_base=None, config_path="config/eval", config_name="train_subgoal")
def main(cfg):
    train(cfg)


if __name__ == "__main__":
    main()
