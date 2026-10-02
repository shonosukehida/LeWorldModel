"""Shared normalization statistics; no robot, GUI or model dependencies."""
import json
import os
from pathlib import Path
import tempfile

import numpy as np


class SafeStandardScaler:
    def __init__(self, eps=1e-4):
        self.eps = eps
        self.mean_ = None
        self.scale_ = None
        self.raw_min_ = 1000000000000.
        self.raw_max_ = -1000000000000.
        self.normed_min_ = 1000000000000. 
        self.normed_max_ = -1000000000000.
        

    def fit(self, x):
        self.mean_ = np.mean(x, axis=0, keepdims=True)
        std = np.std(x, axis=0, keepdims=True, ddof=1,)
        self.scale_ = np.where(std < self.eps, 1.0, std)
        return self

    def transform(self, x):
        return (x - self.mean_) / self.scale_

    def inverse_transform(self, x):
        return x * self.scale_ + self.mean_


def build_normalization_process(stats_dataset, keys_to_cache):
    """
    データセットから正規化processorを作成する。
    """
    process = {}
    action_key = ""

    action_keys = {
        "action",
        "action_cartesian",
        "action_joint",
    }

    for col in keys_to_cache:
        if col == "pixels":
            continue

        col_data = stats_dataset.get_col_data(col)
        col_data = np.asarray(col_data)

        # shape (N,) のデータにも対応
        if col_data.ndim == 1:
            col_data = col_data[:, None]

        valid_mask = ~np.isnan(col_data).any(axis=1)
        col_data = col_data[valid_mask]

        if len(col_data) == 0:
            raise ValueError(
                f"No valid samples are available for normalization: {col}"
            )

        processor = SafeStandardScaler(eps=1e-4)
        processor.fit(col_data)

        processor.raw_min_ = col_data.min(
            axis=0,
            keepdims=True,
        )
        processor.raw_max_ = col_data.max(
            axis=0,
            keepdims=True,
        )
        processor.normed_min_ = processor.transform(
            processor.raw_min_
        )
        processor.normed_max_ = processor.transform(
            processor.raw_max_
        )

        process[col] = processor

        if col in action_keys:
            action_key = col
        else:
            # 元の実装と同じprocessorを共有する
            process[f"goal_{col}"] = processor

    return process, action_key


def save_normalization_process(
    stats_path,
    process,
    action_key,
):
    """
    process内のSafeStandardScalerをnpzファイルに保存する。

    goal_qposなどの別名は保存せず、元の列だけを保存する。
    ロード時にgoal_*を再構成する。
    """
    stats_path = Path(stats_path).expanduser()
    stats_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    base_keys = [
        key
        for key in process.keys()
        if not key.startswith("goal_")
    ]

    metadata = {
        "version": 1,
        "keys": base_keys,
        "action_key": action_key,
    }

    arrays = {
        "metadata": np.asarray(
            json.dumps(metadata),
        ),
    }

    for index, key in enumerate(base_keys):
        processor = process[key]
        prefix = f"processor_{index}"

        arrays[f"{prefix}_mean"] = np.asarray(
            processor.mean_,
        )
        arrays[f"{prefix}_scale"] = np.asarray(
            processor.scale_,
        )
        arrays[f"{prefix}_raw_min"] = np.asarray(
            processor.raw_min_,
        )
        arrays[f"{prefix}_raw_max"] = np.asarray(
            processor.raw_max_,
        )
        arrays[f"{prefix}_normed_min"] = np.asarray(
            processor.normed_min_,
        )
        arrays[f"{prefix}_normed_max"] = np.asarray(
            processor.normed_max_,
        )
        arrays[f"{prefix}_eps"] = np.asarray(
            processor.eps,
            dtype=np.float64,
        )

    # Write once to a sibling temporary file, then atomically publish it.
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=stats_path.parent, prefix=f".{stats_path.name}.", suffix=".npz",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            np.savez_compressed(temporary, **arrays)
        os.replace(temporary_path, stats_path)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    print(
        f"Saved normalization statistics to: "
        f"{stats_path}"
    )


def load_normalization_process(stats_path):
    """
    npzファイルからprocessを復元する。
    """
    stats_path = Path(stats_path).expanduser()

    if not stats_path.is_file():
        raise FileNotFoundError(
            "Normalization statistics file was not found: "
            f"{stats_path}"
        )

    process = {}

    with np.load(
        stats_path,
        allow_pickle=False,
    ) as stats:
        metadata = json.loads(
            str(stats["metadata"].item())
        )

        if metadata.get("version") != 1:
            raise ValueError(
                "Unsupported normalization statistics version: "
                f"{metadata.get('version')}"
            )

        keys = metadata["keys"]
        action_key = metadata.get("action_key", "")

        for index, key in enumerate(keys):
            prefix = f"processor_{index}"

            processor = SafeStandardScaler(
                eps=float(stats[f"{prefix}_eps"].item())
            )
            processor.mean_ = stats[
                f"{prefix}_mean"
            ].copy()
            processor.scale_ = stats[
                f"{prefix}_scale"
            ].copy()
            processor.raw_min_ = stats[
                f"{prefix}_raw_min"
            ].copy()
            processor.raw_max_ = stats[
                f"{prefix}_raw_max"
            ].copy()
            processor.normed_min_ = stats[
                f"{prefix}_normed_min"
            ].copy()
            processor.normed_max_ = stats[
                f"{prefix}_normed_max"
            ].copy()

            process[key] = processor

            if key != action_key:
                process[f"goal_{key}"] = processor

    print(
        f"Loaded normalization statistics from: "
        f"{stats_path}"
    )
    print(f"Normalization keys: {list(process.keys())}")
    print(f"action_key: {action_key}")

    return process, action_key
