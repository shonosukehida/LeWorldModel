"""Generate evaluation normalization statistics directly from an HDF5 file."""
import argparse
import os
import tempfile
from pathlib import Path

import h5py
import numpy as np

from normalization_stats import (build_normalization_process, save_normalization_process,
                                 load_normalization_process)


class HDF5Columns:
    def __init__(self, file, path, derived=None):
        self.file = file
        self.path = path
        self.derived = derived or {}

    def get_col_data(self, column):
        if column in self.derived:
            return self.derived[column]
        if column not in self.file:
            raise KeyError(f"Missing dataset column {column!r} in {self.path}")
        return self.file[column][:]


def build_action_delta_column(columns):
    """Use only adjacent stored rows of one contiguous episode block.

    episode_idx is mandatory. With step_idx, require strictly increasing steps
    within each block and select only +1 pairs (gaps are excluded). Without
    step_idx, stored row adjacency defines continuity; no sorting, resampling,
    alias inference, or joining separated episode blocks is performed.
    NaN rows invalidate both neighboring pairs and are never bridged.
    """
    import torch
    from action_features import action_delta

    actions = np.asarray(columns.get_col_data("action_cartesian"))
    if actions.ndim != 2 or actions.shape[1] != 8 or not np.issubdtype(actions.dtype, np.number) or np.iscomplexobj(actions):
        raise ValueError("action_cartesian must be a real numeric array with shape (N, 8) for action_delta")
    n = len(actions)
    if n < 2:
        raise ValueError("action_delta requires at least two action rows")

    def index_column(name):
        if name not in columns.file:
            raise ValueError(f"Missing required boundary column: {name}")
        values = np.asarray(columns.get_col_data(name))
        if values.shape not in ((n,), (n, 1)) or not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f"{name} must contain N integer indices with shape (N,) or (N, 1)")
        values = values.reshape(n)
        if (values < 0).any():
            raise ValueError(f"{name} must contain nonnegative indices")
        return values

    episodes = index_column("episode_idx")
    starts = np.r_[True, episodes[1:] != episodes[:-1]]
    blocks = episodes[starts]
    if len(np.unique(blocks)) != len(blocks):
        raise ValueError("episode_idx contains repeated noncontiguous episode blocks; row order is ambiguous")
    same_episode = episodes[1:] == episodes[:-1]
    continuous = same_episode.copy()
    if "step_idx" in columns.file:
        steps = index_column("step_idx")
        if (same_episode & (steps[1:] <= steps[:-1])).any():
            raise ValueError("step_idx must be strictly increasing within each episode")
        # The explicit increasing check also rejects unsigned reset wraparound.
        continuous &= (steps[1:] > steps[:-1]) & (steps[1:] - steps[:-1] == 1)
    if np.isinf(actions).any():
        raise ValueError("action_cartesian contains infinite values")
    valid_rows = np.isfinite(actions).all(axis=1)
    continuous &= valid_rows[:-1] & valid_rows[1:]
    indices = np.flatnonzero(continuous)
    if len(indices) < 2:
        raise ValueError("action_delta needs at least two valid continuous within-episode pairs for ddof=1")
    pairs = np.stack((actions[indices], actions[indices + 1]), axis=1)
    # Float64 preserves source precision for fitting; the same function is used
    # by UpperJEPA on tensors, including its unit/sign quaternion convention.
    return action_delta(torch.as_tensor(pairs, dtype=torch.float64)).squeeze(1).numpy()


def _save_verified_process(output, process):
    """Verify the saved bytes before atomically replacing any existing output."""
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".process_stats.") as directory:
        staged = Path(directory) / "process_stats.npz"
        save_normalization_process(staged, process, "action_cartesian")
        restored, action_key = load_normalization_process(staged)
        if action_key != "action_cartesian" or "goal_action_delta" in restored:
            raise ValueError("Invalid normalization round-trip metadata or action_delta alias")
        for key in ("action_cartesian", "proprio", "action_delta"):
            if key not in restored:
                raise ValueError(f"Missing normalization processor after round trip: {key}")
            for name in ("mean_", "scale_", "raw_min_", "raw_max_", "normed_min_", "normed_max_", "eps"):
                if not np.array_equal(getattr(process[key], name), getattr(restored[key], name)):
                    raise ValueError(f"Normalization round-trip mismatch: {key}.{name}")
        os.replace(staged, output)
    print(f"Verified normalization statistics saved to: {output}")


def create_process_stats(dataset, output, keys=None, *, overwrite=True, include_action_delta=False):
    dataset = Path(dataset).expanduser().resolve()
    output_dir = Path(output).expanduser().resolve()
    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(f"Output path is not a directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "process_stats.npz"
    if dataset == output or (
        dataset.exists() and output.exists() and dataset.samefile(output)
    ):
        raise ValueError(f"Output must differ from input dataset: {dataset}")
    if not dataset.is_file():
        raise FileNotFoundError(f"Dataset file was not found: {dataset}")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Overwriting normalization statistics is forbidden: {output}")
    keys = list(keys if keys is not None else ("action_cartesian", "proprio"))
    if "action_cartesian" not in keys:
        raise ValueError(f"Required column 'action_cartesian' must be included in --keys: {dataset}")
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate columns in --keys")
    if any(key == "pixels" or key.startswith("goal_") for key in keys):
        raise ValueError(f"--keys must contain base numeric columns, not pixels or goal aliases: {keys}")
    if "action_delta" in keys:
        raise ValueError("action_delta is derived; enable --include-action-delta instead of adding it to --keys")
    if include_action_delta and "proprio" not in keys:
        raise ValueError("--include-action-delta requires proprio in --keys")
    with h5py.File(dataset, "r") as file:
        columns = HDF5Columns(file, dataset)
        if include_action_delta:
            columns.derived["action_delta"] = build_action_delta_column(columns)
            keys.append("action_delta")
        # Validate every requested column before fitting or creating an output.
        for key in keys:
            values = np.asarray(columns.get_col_data(key))
            if values.ndim not in (1, 2) or (values.ndim == 2 and values.shape[1] == 0):
                raise ValueError(f"Column {key!r} in {dataset} must have shape (N,) or (N, D)")
            if not np.issubdtype(values.dtype, np.number):
                raise ValueError(f"Column {key!r} in {dataset} must be numeric")
            valid = values[~np.isnan(values).reshape(len(values), -1).any(axis=1)] if len(values) else values
            if len(valid) == 0:
                raise ValueError(f"No valid samples for column {key!r} in {dataset}")
            if len(valid) < 2:
                raise ValueError(f"Column {key!r} in {dataset} needs at least two valid rows for ddof=1")
            if not np.isfinite(valid).all():
                raise ValueError(f"Column {key!r} in {dataset} contains infinite values")
        process, _ = build_normalization_process(columns, keys)
    for key in keys:
        processor = process[key]
        for name in ("mean_", "scale_", "raw_min_", "raw_max_", "normed_min_", "normed_max_"):
            if not np.isfinite(getattr(processor, name)).all():
                raise ValueError(f"Invalid {name} for column {key!r} in {dataset}")
    if include_action_delta:
        _save_verified_process(output, process)
    else:
        save_normalization_process(output, process, "action_cartesian")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Input HDF5 dataset")
    parser.add_argument(
        "--output",
        required=True,
        help="Output directory; process_stats.npz is written inside it",
    )
    parser.add_argument("--keys", nargs="+", default=["action_cartesian", "proprio"])
    parser.add_argument("--no-overwrite", action="store_true", help="Reject an existing output")
    parser.add_argument("--include-action-delta", action="store_true",
                        help="Fit physical action deltas within episode_idx boundaries; validate step_idx if present")
    args = parser.parse_args()
    create_process_stats(args.dataset, args.output, args.keys, overwrite=not args.no_overwrite,
                         include_action_delta=args.include_action_delta)


if __name__ == "__main__":
    main()
