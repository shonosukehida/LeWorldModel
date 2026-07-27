"""Visualize normalized leader joint angles for all episodes."""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


# ============================================================
# 設定
# ============================================================

DATASET_PATH = Path(
    "/home/hida/.stable_worldmodel/datasets/flip_mug/ep200_tm300/push.h5"
)

# joints.png を保存するディレクトリ
SAVE_DIR = Path(
    "/home/hida/LeWorldModel/robot/utils/plots/figures"
)

NUM_JOINTS = 7


def normalize_per_joint(
    joint_angles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """各関節を、全エピソード中の最小値・最大値で-1〜1に正規化する。

    Args:
        joint_angles:
            Shape (total_steps, 7)

    Returns:
        normalized:
            Shape (total_steps, 7)
        joint_min:
            Shape (7,)
        joint_max:
            Shape (7,)
    """
    joint_min = joint_angles.min(axis=0)
    joint_max = joint_angles.max(axis=0)

    joint_range = joint_max - joint_min

    # 最大値と最小値が同じ関節でゼロ除算を防ぐ
    safe_range = np.where(
        joint_range > 0.0,
        joint_range,
        1.0,
    )

    # [min, max] -> [-1, 1]
    normalized = (
        2.0 * (joint_angles - joint_min) / safe_range - 1.0
    )

    # 最大値と最小値が同じ場合は、正規化値を0にする
    constant_joint_mask = joint_range <= 0.0
    normalized[:, constant_joint_mask] = 0.0

    return normalized, joint_min, joint_max


def load_dataset(
    dataset_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """HDF5からleader、ep_len、ep_offsetを読み込む。"""
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset was not found: {dataset_path}"
        )

    with h5py.File(dataset_path, "r") as file:
        required_keys = [
            "leader",
            "ep_len",
            "ep_offset",
        ]

        for key in required_keys:
            if key not in file:
                raise KeyError(
                    f"Dataset key '{key}' was not found in "
                    f"{dataset_path}"
                )

        leader = np.asarray(
            file["leader"][:],
            dtype=np.float32,
        )
        ep_len = np.asarray(
            file["ep_len"][:],
            dtype=np.int64,
        )
        ep_offset = np.asarray(
            file["ep_offset"][:],
            dtype=np.int64,
        )

    if leader.ndim != 2 or leader.shape[1] < NUM_JOINTS:
        raise ValueError(
            "leader must have shape (total_steps, >=7), "
            f"but got {leader.shape}"
        )

    if ep_len.ndim != 1 or ep_offset.ndim != 1:
        raise ValueError(
            "ep_len and ep_offset must be one-dimensional arrays"
        )

    if len(ep_len) != len(ep_offset):
        raise ValueError(
            "ep_len and ep_offset have different lengths: "
            f"{len(ep_len)} != {len(ep_offset)}"
        )

    # leaderの先頭7次元がxArm7の関節角
    joint_angles = leader[:, :NUM_JOINTS]

    return joint_angles, ep_len, ep_offset


def plot_leader_joint_angles(
    dataset_path: Path,
    save_dir: Path,
) -> None:
    """全エピソードのleader関節角を7行で可視化する。"""
    joint_angles, ep_len, ep_offset = load_dataset(
        dataset_path
    )

    normalized, joint_min, joint_max = normalize_per_joint(
        joint_angles
    )

    num_episodes = len(ep_len)

    print(f"Dataset       : {dataset_path}")
    print(f"Leader shape  : {joint_angles.shape}")
    print(f"Num episodes  : {num_episodes}")
    print()

    for joint_index in range(NUM_JOINTS):
        print(
            f"Joint {joint_index + 1}: "
            f"min={joint_min[joint_index]: .6f} rad, "
            f"max={joint_max[joint_index]: .6f} rad"
        )

    fig, axes = plt.subplots(
        nrows=NUM_JOINTS,
        ncols=1,
        figsize=(12, 16),
        sharex=True,
    )
    
    joint_color = "tab:blue"

    for episode_index, (offset, length) in enumerate(
        zip(ep_offset, ep_len)
    ):
        start = int(offset)
        end = start + int(length)

        if start < 0 or end > len(normalized):
            raise IndexError(
                "Invalid episode range:\n"
                f"  episode={episode_index}\n"
                f"  start={start}\n"
                f"  end={end}\n"
                f"  total_steps={len(normalized)}"
            )

        episode_joints = normalized[start:end]
        timesteps = np.arange(
            int(length),
            dtype=np.int64,
        )

        for joint_index, axis in enumerate(axes):
            axis.plot(
                timesteps,
                episode_joints[:, joint_index],
                color=joint_color,
                alpha=0.35,
                linewidth=1.0,
            )

    for joint_index, axis in enumerate(axes):
        axis.set_ylabel(
            f"Joint {joint_index + 1}\nnormalized"
        )
        axis.set_ylim(-1.05, 1.05)
        axis.set_yticks(
            [-1.0, -0.5, 0.0, 0.5, 1.0]
        )
        axis.grid(
            True,
            alpha=0.3,
        )

        axis.set_title(
            f"Joint {joint_index + 1}: "
            f"min={joint_min[joint_index]:.3f} rad, "
            f"max={joint_max[joint_index]:.3f} rad"
        )

    axes[-1].set_xlabel("Timestep")

    fig.suptitle(
        "Normalized Leader Joint Angles\n"
        f"All {num_episodes} Episodes",
        fontsize=16,
    )

    fig.tight_layout(
        rect=(0.0, 0.0, 1.0, 0.97)
    )

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_path = save_dir / "joints.png"

    fig.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight",
    )

    print(f"\nSaved figure to: {save_path}")

    plt.show()
    plt.close(fig)


def main() -> None:
    plot_leader_joint_angles(
        dataset_path=DATASET_PATH,
        save_dir=SAVE_DIR,
    )


if __name__ == "__main__":
    main()