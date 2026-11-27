"""Train a horizon-conditioned subgoal MLP from measured demonstration states."""
from pathlib import Path
import tempfile
import random
from contextlib import contextmanager

from omegaconf import OmegaConf

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



@contextmanager
def preserve_rng():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def split_pairs(pairs, episode_ids, ratio, seed):
    if not 0 <= ratio < 1:
        raise ValueError("validation_ratio must be in [0, 1)")
    if ratio == 0:
        return pairs, pairs[:0]
    episodes = np.unique(episode_ids)
    if len(episodes) < 2:
        raise ValueError("Validation requires at least two episodes")
    count = min(len(episodes) - 1, max(1, int(round(len(episodes) * ratio))))
    selected = np.random.default_rng(seed).permutation(episodes)[:count]
    mask = torch.from_numpy(np.isin(episode_ids[pairs[:, 0].numpy()], selected))
    train_pairs, val_pairs = pairs[~mask], pairs[mask]
    if not len(train_pairs) or not len(val_pairs):
        raise ValueError("Train and validation must both contain valid pairs")
    return train_pairs, val_pairs


@torch.no_grad()
def evaluate(network, pairs, latents, goal, batch_size, device, prefix):
    total = 0.0
    sums, counts = {}, {}
    for batch in pairs.split(batch_size):
        rows, ends, horizons = batch.unbind(dim=1)
        current = latents[rows].to(device)
        errors = (network(current, goal.expand_as(current), horizons) -
                  latents[ends].to(device)).square().mean(dim=-1).cpu()
        total += errors.sum().item()
        for h in horizons.unique().tolist():
            mask = horizons == h
            sums[h] = sums.get(h, 0.0) + errors[mask].sum().item()
            counts[h] = counts.get(h, 0) + int(mask.sum())
    result = {f"{prefix}/loss": total / len(pairs)}
    result.update({f"{prefix}/loss_h{h}": value / counts[h] for h, value in sums.items()})
    return result

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
        pairs, val_pairs = split_pairs(pairs, np.asarray(dataset["episode_idx"]).reshape(-1),
                                       options.get("validation_ratio", 0.1), cfg.seed)
        latents, final_goal = encode_demonstrations(wm, prep, dataset, goal, options.encode_batch_size)
    del wm
    final_goal = final_goal.to(options.device)
    network = SubgoalMLP(latents.shape[-1], options.hidden_dim).to(options.device)
    optimizer = torch.optim.Adam(network.parameters(), lr=options.learning_rate)
    batches = DataLoader(TensorDataset(*pairs.unbind(dim=1)), batch_size=options.batch_size, shuffle=True)
    run = None
    if cfg.get("wandb", {}).get("enabled", False):
        with preserve_rng():
            import wandb
            run = wandb.init(project=cfg.wandb.project, entity=cfg.wandb.entity,
                             config=OmegaConf.to_container(cfg, resolve=True))
            run.define_metric("epoch")
            run.define_metric("train/*", step_metric="epoch")
            run.define_metric("val/*", step_metric="epoch")
    for epoch in range(options.epochs):
        network.train()
        total = 0.0
        for rows, ends, horizons in batches:
            current = latents[rows].to(options.device)
            predicted = network(current, final_goal.expand_as(current), horizons)
            loss = torch.nn.functional.mse_loss(predicted, latents[ends].to(options.device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total += loss.detach().item() * len(rows)
        with preserve_rng():
            network.eval()
            metrics = evaluate(network, pairs, latents, final_goal, options.batch_size, options.device, "train")
            if len(val_pairs):
                metrics.update(evaluate(network, val_pairs, latents, final_goal,
                                        options.batch_size, options.device, "val"))
            metrics.update({"epoch": epoch + 1, "train/optimization_loss": total / len(pairs)})
            if run is not None:
                run.log(metrics)
        print(f"epoch {epoch + 1}: {metrics}", flush=True)
    if run is not None:
        with preserve_rng():
            run.finish()

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


@hydra.main(version_base=None, config_path="config/train", config_name="train_subgoal")
def main(cfg):
    train(cfg)


if __name__ == "__main__":
    main()
