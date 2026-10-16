"""Episode-disjoint H-step sequences and train-only normalization for UpperJEPA."""
import json
import os
from pathlib import Path
import tempfile

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from create_process_stats import build_action_delta_column
from normalization_stats import (build_normalization_process, load_normalization_process,
                                 save_normalization_process)


STAT_KEYS = ("action_cartesian", "proprio", "action_delta")


def validate_process(process):
    """Required training/inference statistics; never substitute missing values."""
    for key in STAT_KEYS:
        if key not in process:
            raise ValueError(f"Missing required normalization processor: {key}")
        for name in ("mean_", "scale_"):
            value = np.asarray(getattr(process[key], name))
            if value.shape not in ((8,), (1, 8)) or not np.isfinite(value).all():
                raise ValueError(f"{key}.{name} must contain eight finite values")
            if name == "scale_" and (value <= 0).any():
                raise ValueError(f"{key}.scale_ must be strictly positive")


def read_metadata(path):
    """Read small numeric columns; images remain on disk. No index inference."""
    path = Path(path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as file:
        for key in ("pixels", "wrist_pixels", "proprio", "action_cartesian", "episode_idx"):
            if key not in file:
                raise ValueError(f"Missing required dataset column: {key}")
        n = len(file["action_cartesian"])
        if n < 2:
            raise ValueError("Dataset needs at least two rows")
        columns = {key: file[key][:] for key in ("action_cartesian", "proprio")}
        for key, value in columns.items():
            if value.shape != (n, 8) or not np.issubdtype(value.dtype, np.number) or np.iscomplexobj(value):
                raise ValueError(f"{key} must have shape (N, 8) and real numeric values")
        for key in ("episode_idx", "step_idx"):
            if key not in file:
                continue
            value = file[key][:]
            if value.shape not in ((n,), (n, 1)) or not np.issubdtype(value.dtype, np.integer) or (value < 0).any():
                raise ValueError(f"{key} must contain N nonnegative integer indices")
            columns[key] = value.reshape(n)
        for key in ("pixels", "wrist_pixels"):
            shape = file[key].shape
            if len(shape) != 4 or shape[0] != n or (shape[1] != 3 and shape[-1] != 3):
                raise ValueError(f"{key} must contain N RGB images (NHWC or NCHW)")
    episode = columns["episode_idx"]
    starts = np.flatnonzero(np.r_[True, episode[1:] != episode[:-1]])
    if len(np.unique(episode[starts])) != len(starts):
        raise ValueError("episode_idx contains repeated noncontiguous episode blocks")
    same = episode[1:] == episode[:-1]
    if "step_idx" in columns:
        steps = columns["step_idx"]
        if (same & (steps[1:] <= steps[:-1])).any():
            raise ValueError("step_idx must increase strictly within each episode")
    return path, columns


def split_episodes(episodes, train_split, seed):
    ids = np.unique(episodes)
    if not 0 < train_split < 1:
        raise ValueError("train_split must be strictly between zero and one")
    count = int(len(ids) * train_split)
    if not 0 < count < len(ids):
        raise ValueError("Episode split must leave at least one episode on each side")
    order = torch.randperm(len(ids), generator=torch.Generator().manual_seed(seed)).numpy()
    return np.sort(ids[order[:count]]), np.sort(ids[order[count:]])


class _Columns:
    def __init__(self, values):
        self.file = values  # Boundary-key presence interface used by the shared fitter.

    def get_col_data(self, name):
        return self.file[name]


def fit_training_process(columns, train_episodes, output):
    """Fit raw absolute/proprio and shared physical deltas on train episodes only.

    Uses the same ddof=1 / safe-std convention and NPZ layout as
    create_process_stats.py. Splitting must precede this call.
    """
    output = Path(output)
    if output.exists():
        raise FileExistsError(f"Refusing to replace training statistics: {output}")
    mask = np.isin(columns["episode_idx"], train_episodes)
    selected = {key: value[mask] for key, value in columns.items()}
    adapter = _Columns(selected)
    selected["action_delta"] = build_action_delta_column(adapter)
    for key in STAT_KEYS:
        values = selected[key]
        valid = values[~np.isnan(values).any(axis=1)]
        if len(valid) < 2 or not np.isfinite(valid).all():
            raise ValueError(f"{key} needs at least two finite training rows (NaN rows excluded)")
    process, action_key = build_normalization_process(adapter, STAT_KEYS)
    validate_process(process)
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".train_stats.") as directory:
        staged = Path(directory) / "process_stats.npz"
        save_normalization_process(staged, process, action_key)
        restored, _ = load_normalization_process(staged)
        validate_process(restored)
        for key in STAT_KEYS:
            for name in ("mean_", "scale_", "raw_min_", "raw_max_", "normed_min_", "normed_max_", "eps"):
                if not np.array_equal(getattr(restored[key], name), getattr(process[key], name)):
                    raise ValueError(f"Normalization round-trip mismatch: {key}.{name}")
        os.replace(staged, output)
    return process


class UpperObservationTransform:
    """Same image preprocessing as lower JEPA; standardize only proprio here.

    Cartesian actions stay physical. UpperJEPA alone standardizes macro features.
    Apply this transform to raw observations for checkpoint inference as well.
    """
    def __init__(self, process, img_size):
        validate_process(process)
        from utils import get_img_preprocessor
        self.images = [get_img_preprocessor(key, key, img_size) for key in ("pixels", "wrist_pixels")]
        self.prop_mean = torch.as_tensor(process["proprio"].mean_, dtype=torch.float32).reshape(8)
        self.prop_scale = torch.as_tensor(process["proprio"].scale_, dtype=torch.float32).reshape(8)

    def __call__(self, observations):
        result = dict(observations)
        for key, transform in zip(("pixels", "wrist_pixels"), self.images):
            value = torch.as_tensor(result[key])
            if value.shape[-3] != 3:
                if value.shape[-1] != 3:
                    raise ValueError(f"{key} must contain RGB images")
                value = value.movedim(-1, -3)
            result[key] = value
            result = transform(result)
        proprio = torch.as_tensor(result["proprio"]).float()
        result["proprio"] = (proprio - self.prop_mean.to(proprio)) / self.prop_scale.to(proprio)
        return result


class UpperSequenceDataset(Dataset):
    """K+1 observations spaced H apart and K physical H-action chunks.

    A window uses rows [start, start+K*H], inclusive. Every intermediate row
    must be in the same episode with contiguous steps, even when observations
    are subsampled. Without step_idx, continuity means stored row adjacency.
    """
    def __init__(self, path, columns, episodes, horizon, history_size, transform):
        if any(isinstance(x, bool) or not isinstance(x, int) or x < 1 for x in (horizon, history_size)):
            raise ValueError("horizon and history_size must be positive integers")
        self.path = str(path)
        self.columns = columns
        self.episodes = np.asarray(episodes)
        self.horizon, self.history_size = horizon, history_size
        self.transform = transform
        self.actions_normalized = False
        episode = columns["episode_idx"]
        n = len(episode)
        span = horizon * history_size
        link_ok = episode[1:] == episode[:-1]
        if "step_idx" in columns:
            steps = columns["step_idx"]
            link_ok &= (steps[1:] > steps[:-1]) & (steps[1:] - steps[:-1] == 1)
        # Prefix sums avoid enumerating all observations/actions per candidate.
        broken = np.r_[0, np.cumsum(~link_ok)]
        actions = columns["action_cartesian"]
        valid_action = np.isfinite(actions).all(axis=1)
        # Both delta fitting and macro generation require nonzero physical quaternions.
        valid_action &= np.linalg.norm(actions[:, 3:7], axis=-1) >= 1e-8
        invalid_action = np.r_[0, np.cumsum(~valid_action)]
        candidates = np.arange(max(0, n - span))
        keep = np.isin(episode[candidates], self.episodes)
        keep &= broken[candidates + span] == broken[candidates]
        keep &= invalid_action[candidates + span] == invalid_action[candidates]
        for offset in range(history_size + 1):
            keep &= np.isfinite(columns["proprio"][candidates + offset * horizon]).all(axis=1)
        self.starts = candidates[keep]
        if not len(self.starts):
            raise ValueError("No valid upper-level windows: episodes are too short or contain gaps/invalid data")

    def __len__(self):
        return len(self.starts)

    def indices(self, index):
        start = int(self.starts[index])
        observations = start + np.arange(self.history_size + 1) * self.horizon
        action_rows = np.arange(start, start + self.history_size * self.horizon).reshape(self.history_size, self.horizon)
        return observations, action_rows

    def __getitem__(self, index):
        observations, action_rows = self.indices(index)
        with h5py.File(self.path, "r") as file:
            sample = {key: torch.from_numpy(file[key][observations]) for key in ("pixels", "wrist_pixels")}
        sample["proprio"] = torch.from_numpy(self.columns["proprio"][observations].copy())
        sample = self.transform(sample)
        sample["action_cartesian"] = torch.from_numpy(self.columns["action_cartesian"][action_rows].copy()).float()
        return sample


def prepare_upper_data(path, run_dir, *, horizon, history_size, train_split, seed, img_size):
    path, columns = read_metadata(path)
    train_ids, val_ids = split_episodes(columns["episode_idx"], train_split, seed)
    run_dir = Path(run_dir)
    process = fit_training_process(columns, train_ids, run_dir / "process_stats.npz")
    transform = UpperObservationTransform(process, img_size)
    train = UpperSequenceDataset(path, columns, train_ids, horizon, history_size, transform)
    val = UpperSequenceDataset(path, columns, val_ids, horizon, history_size, transform)
    split = {"dataset": str(path), "seed": seed, "train_split": train_split,
             "train_episodes": train_ids.tolist(), "validation_episodes": val_ids.tolist()}
    (run_dir / "episode_split.json").write_text(json.dumps(split, indent=2) + "\n")
    return train, val, process, split
