"""Visualize raw and normalized xArm follower trajectories from an HDF5 file."""

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray


JOINT_LIMITS = [
    (-2 * np.pi, 2 * np.pi),  # Joint 1
    (-2.0595, 2.0944),        # Joint 2
    (-2 * np.pi, 2 * np.pi),  # Joint 3
    (-0.1920, 3.9270),        # Joint 4
    (-2 * np.pi, 2 * np.pi),  # Joint 5
    (-1.6930, np.pi),         # Joint 6
    (-2 * np.pi, 2 * np.pi),  # Joint 7
]

# robopyではグリッパが [0, 1] として記録される想定
GRIPPER_LIMIT = (0.0, 1.0)

OUTPUT_DIR = Path("./robot/teleop/plots/figures")


def load_follower(h5_path: str | Path) -> NDArray[np.float32]:
    """Load arms/follower from an HDF5 episode."""
    h5_path = Path(h5_path)

    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    with h5py.File(h5_path, "r") as h5_file:
        key = "arms/follower"

        if key not in h5_file:
            raise KeyError(f"Dataset '{key}' was not found in {h5_path}")

        follower = np.asarray(h5_file[key], dtype=np.float32)

    if follower.ndim != 2 or follower.shape[1] != 8:
        raise ValueError(
            "Expected follower shape (frames, 8), "
            f"but got {follower.shape}"
        )

    return follower


def min_max_normalize(
    values: NDArray[np.float32],
    lower: float,
    upper: float,
    clip: bool = True,
) -> NDArray[np.float32]:
    """Normalize values into [0, 1] using fixed lower and upper limits."""
    if upper <= lower:
        raise ValueError(
            f"upper must be greater than lower: lower={lower}, upper={upper}"
        )

    normalized = (values - lower) / (upper - lower)

    if clip:
        normalized = np.clip(normalized, 0.0, 1.0)

    return normalized.astype(np.float32)


def normalize_follower(
    follower: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Normalize seven joints and the gripper into [0, 1]."""
    normalized = np.empty_like(follower, dtype=np.float32)

    for joint_idx, (lower, upper) in enumerate(JOINT_LIMITS):
        normalized[:, joint_idx] = min_max_normalize(
            follower[:, joint_idx],
            lower=lower,
            upper=upper,
        )

    normalized[:, 7] = min_max_normalize(
        follower[:, 7],
        lower=GRIPPER_LIMIT[0],
        upper=GRIPPER_LIMIT[1],
    )

    return normalized


def plot_raw_follower(
    follower: NDArray[np.float32],
    time_sec: NDArray[np.float64],
    output_path: Path,
) -> None:
    """Plot seven raw joint trajectories and one gripper trajectory."""
    figure, axes = plt.subplots(
        nrows=8,
        ncols=1,
        figsize=(14, 18),
        sharex=True,
    )

    for joint_idx in range(7):
        # lower, upper = JOINT_LIMITS[joint_idx]
        axis = axes[joint_idx]

        axis.plot(time_sec, follower[:, joint_idx])
        # axis.axhline(lower, linestyle="--", linewidth=1)
        # axis.axhline(upper, linestyle="--", linewidth=1)

        axis.set_ylabel(f"Joint {joint_idx + 1}\n[rad]")
        axis.grid(True)

    axes[7].plot(time_sec, follower[:, 7])
    axes[7].axhline(
        GRIPPER_LIMIT[0],
        linestyle="--",
        linewidth=1,
    )
    axes[7].axhline(
        GRIPPER_LIMIT[1],
        linestyle="--",
        linewidth=1,
    )
    axes[7].set_ylabel("Gripper")
    axes[7].set_xlabel("Time [s]")
    axes[7].grid(True)

    figure.suptitle("xArm Follower Trajectories", fontsize=16)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def plot_normalized_follower(
    normalized: NDArray[np.float32],
    time_sec: NDArray[np.float64],
    output_path: Path,
) -> None:
    """Plot normalized follower trajectories in eight rows."""
    figure, axes = plt.subplots(
        nrows=8,
        ncols=1,
        figsize=(14, 18),
        sharex=True,
    )

    for joint_idx in range(7):
        axis = axes[joint_idx]
        axis.plot(time_sec, normalized[:, joint_idx])

        axis.set_ylabel(f"Joint {joint_idx + 1}")
        axis.set_ylim(-0.05, 1.05)
        axis.grid(True)

    axes[7].plot(time_sec, normalized[:, 7])
    axes[7].set_ylabel("Gripper")
    axes[7].set_xlabel("Time [s]")
    axes[7].set_ylim(-0.05, 1.05)
    axes[7].grid(True)

    figure.suptitle(
        "Normalized xArm Follower Trajectories",
        fontsize=16,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def visualize_follower(
    h5_path: str | Path,
    fps: float = 10.0,
) -> None:
    """Create raw and normalized follower trajectory figures."""
    if fps <= 0:
        raise ValueError("fps must be greater than 0")

    h5_path = Path(h5_path)
    follower = load_follower(h5_path)
    normalized = normalize_follower(follower)

    num_frames = follower.shape[0]
    time_sec = np.arange(num_frames, dtype=np.float64) / fps

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    episode_name = h5_path.stem

    raw_output_path = (
        OUTPUT_DIR / f"{episode_name}_follower_raw.png"
    )
    normalized_output_path = (
        OUTPUT_DIR / f"{episode_name}_follower_normalized.png"
    )

    plot_raw_follower(
        follower=follower,
        time_sec=time_sec,
        output_path=raw_output_path,
    )
    plot_normalized_follower(
        normalized=normalized,
        time_sec=time_sec,
        output_path=normalized_output_path,
    )

    print(f"Follower shape: {follower.shape}")
    print(f"Duration: {num_frames / fps:.2f} seconds")
    print(f"Raw figure:        {raw_output_path.resolve()}")
    print(f"Normalized figure: {normalized_output_path.resolve()}")

    print("\nObserved ranges:")

    for joint_idx in range(7):
        values = follower[:, joint_idx]
        lower, upper = JOINT_LIMITS[joint_idx]

        print(
            f"Joint {joint_idx + 1}: "
            f"observed=[{values.min():.4f}, {values.max():.4f}], "
            f"limit=[{lower:.4f}, {upper:.4f}]"
        )

    print(
        "Gripper: "
        f"observed=[{follower[:, 7].min():.4f}, "
        f"{follower[:, 7].max():.4f}]"
    )


if __name__ == "__main__":
    visualize_follower(
        h5_path=(
            "/home/hida/workspace/.stable_worldmodel/datasets/flip_mug/ep200_tm300/per_episode/episode_20260715_162509.h5"
        ),
        fps=10.0,
    )