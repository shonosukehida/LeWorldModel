"""Visualize raw and normalized xArm follower/leader trajectories."""

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


def load_arm_trajectories(
    h5_path: str | Path,
) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
    """Load arms/follower and arms/leader from an HDF5 episode."""
    h5_path = Path(h5_path)

    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    follower_key = "arms/follower"
    leader_key = "arms/leader"

    with h5py.File(h5_path, "r") as h5_file:
        if follower_key not in h5_file:
            raise KeyError(
                f"Dataset '{follower_key}' was not found in {h5_path}"
            )

        if leader_key not in h5_file:
            raise KeyError(
                f"Dataset '{leader_key}' was not found in {h5_path}"
            )

        follower = np.asarray(
            h5_file[follower_key],
            dtype=np.float32,
        )
        leader = np.asarray(
            h5_file[leader_key],
            dtype=np.float32,
        )

    validate_arm_trajectory(
        trajectory=follower,
        name="follower",
    )
    validate_arm_trajectory(
        trajectory=leader,
        name="leader",
    )

    if follower.shape != leader.shape:
        raise ValueError(
            "Follower and leader must have the same shape: "
            f"follower={follower.shape}, leader={leader.shape}"
        )

    return follower, leader


def validate_arm_trajectory(
    trajectory: NDArray[np.float32],
    name: str,
) -> None:
    """Validate an xArm trajectory with seven joints and one gripper value."""
    if trajectory.ndim != 2 or trajectory.shape[1] != 8:
        raise ValueError(
            f"Expected {name} shape (frames, 8), "
            f"but got {trajectory.shape}"
        )

    if not np.all(np.isfinite(trajectory)):
        raise ValueError(
            f"{name} contains NaN or Inf values"
        )


def min_max_normalize(
    values: NDArray[np.float32],
    lower: float,
    upper: float,
    clip: bool = True,
) -> NDArray[np.float32]:
    """Normalize values into [0, 1] using fixed lower and upper limits."""
    if upper <= lower:
        raise ValueError(
            f"upper must be greater than lower: "
            f"lower={lower}, upper={upper}"
        )

    normalized = (values - lower) / (upper - lower)

    if clip:
        normalized = np.clip(
            normalized,
            0.0,
            1.0,
        )

    return normalized.astype(np.float32)


def normalize_arm_trajectory(
    trajectory: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Normalize seven joints and one gripper dimension into [0, 1]."""
    normalized = np.empty_like(
        trajectory,
        dtype=np.float32,
    )

    for joint_idx, (lower, upper) in enumerate(JOINT_LIMITS):
        normalized[:, joint_idx] = min_max_normalize(
            trajectory[:, joint_idx],
            lower=lower,
            upper=upper,
        )

    normalized[:, 7] = min_max_normalize(
        trajectory[:, 7],
        lower=GRIPPER_LIMIT[0],
        upper=GRIPPER_LIMIT[1],
    )

    return normalized


def plot_raw_trajectories(
    follower: NDArray[np.float32],
    leader: NDArray[np.float32],
    time_sec: NDArray[np.float64],
    output_path: Path,
) -> None:
    """Plot raw follower and leader trajectories in eight rows."""
    figure, axes = plt.subplots(
        nrows=8,
        ncols=1,
        figsize=(14, 18),
        sharex=True,
    )

    for joint_idx in range(7):
        axis = axes[joint_idx]

        axis.plot(
            time_sec,
            follower[:, joint_idx],
            label="Follower",
        )
        axis.plot(
            time_sec,
            leader[:, joint_idx],
            label="Leader",
            linestyle="--",
        )

        axis.set_ylabel(
            f"Joint {joint_idx + 1}\n[rad]"
        )
        axis.grid(True)

        if joint_idx == 0:
            axis.legend()

    axes[7].plot(
        time_sec,
        follower[:, 7],
        label="Follower",
    )
    axes[7].plot(
        time_sec,
        leader[:, 7],
        label="Leader",
        linestyle="--",
    )

    axes[7].axhline(
        GRIPPER_LIMIT[0],
        linestyle=":",
        linewidth=1,
    )
    axes[7].axhline(
        GRIPPER_LIMIT[1],
        linestyle=":",
        linewidth=1,
    )

    axes[7].set_ylabel("Gripper")
    axes[7].set_xlabel("Time [s]")
    axes[7].grid(True)
    axes[7].legend()

    figure.suptitle(
        "xArm Follower and Leader Trajectories",
        fontsize=16,
    )
    figure.tight_layout(
        rect=(0, 0, 1, 0.98),
    )
    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def plot_normalized_trajectories(
    normalized_follower: NDArray[np.float32],
    normalized_leader: NDArray[np.float32],
    time_sec: NDArray[np.float64],
    output_path: Path,
) -> None:
    """Plot normalized follower and leader trajectories in eight rows."""
    figure, axes = plt.subplots(
        nrows=8,
        ncols=1,
        figsize=(14, 18),
        sharex=True,
    )

    for joint_idx in range(7):
        axis = axes[joint_idx]

        axis.plot(
            time_sec,
            normalized_follower[:, joint_idx],
            label="Follower",
        )
        axis.plot(
            time_sec,
            normalized_leader[:, joint_idx],
            label="Leader",
            linestyle="--",
        )

        axis.set_ylabel(
            f"Joint {joint_idx + 1}"
        )
        axis.set_ylim(-0.05, 1.05)
        axis.grid(True)

        if joint_idx == 0:
            axis.legend()

    axes[7].plot(
        time_sec,
        normalized_follower[:, 7],
        label="Follower",
    )
    axes[7].plot(
        time_sec,
        normalized_leader[:, 7],
        label="Leader",
        linestyle="--",
    )

    axes[7].set_ylabel("Gripper")
    axes[7].set_xlabel("Time [s]")
    axes[7].set_ylim(-0.05, 1.05)
    axes[7].grid(True)
    axes[7].legend()

    figure.suptitle(
        "Normalized xArm Follower and Leader Trajectories",
        fontsize=16,
    )
    figure.tight_layout(
        rect=(0, 0, 1, 0.98),
    )
    figure.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)


def print_observed_ranges(
    follower: NDArray[np.float32],
    leader: NDArray[np.float32],
) -> None:
    """Print follower and leader ranges for each trajectory dimension."""
    print("\nObserved ranges:")

    for joint_idx in range(7):
        follower_values = follower[:, joint_idx]
        leader_values = leader[:, joint_idx]
        lower, upper = JOINT_LIMITS[joint_idx]

        print(
            f"Joint {joint_idx + 1}:\n"
            f"  follower=[{follower_values.min():.4f}, "
            f"{follower_values.max():.4f}]\n"
            f"  leader  =[{leader_values.min():.4f}, "
            f"{leader_values.max():.4f}]\n"
            f"  limit   =[{lower:.4f}, {upper:.4f}]"
        )

    print(
        "Gripper:\n"
        f"  follower=[{follower[:, 7].min():.4f}, "
        f"{follower[:, 7].max():.4f}]\n"
        f"  leader  =[{leader[:, 7].min():.4f}, "
        f"{leader[:, 7].max():.4f}]"
    )


def print_tracking_errors(
    follower: NDArray[np.float32],
    leader: NDArray[np.float32],
) -> None:
    """Print basic follower-versus-leader tracking errors."""
    error = follower - leader

    print("\nFollower - Leader tracking error:")

    for joint_idx in range(7):
        joint_error = error[:, joint_idx]

        mean_absolute_error = np.mean(
            np.abs(joint_error)
        )
        maximum_absolute_error = np.max(
            np.abs(joint_error)
        )

        print(
            f"Joint {joint_idx + 1}: "
            f"MAE={mean_absolute_error:.6f} rad, "
            f"MaxAE={maximum_absolute_error:.6f} rad"
        )

    gripper_error = error[:, 7]

    print(
        "Gripper: "
        f"MAE={np.mean(np.abs(gripper_error)):.6f}, "
        f"MaxAE={np.max(np.abs(gripper_error)):.6f}"
    )


def visualize_arm_trajectories(
    h5_path: str | Path,
    fps: float = 10.0,
) -> None:
    """Create raw and normalized follower/leader trajectory figures."""
    if fps <= 0:
        raise ValueError("fps must be greater than 0")

    h5_path = Path(h5_path)

    follower, leader = load_arm_trajectories(
        h5_path,
    )

    normalized_follower = normalize_arm_trajectory(
        follower,
    )
    normalized_leader = normalize_arm_trajectory(
        leader,
    )

    num_frames = follower.shape[0]

    time_sec = (
        np.arange(
            num_frames,
            dtype=np.float64,
        )
        / fps
    )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    episode_name = h5_path.stem

    raw_output_path = (
        OUTPUT_DIR
        / f"{episode_name}_follower_leader_raw.png"
    )
    normalized_output_path = (
        OUTPUT_DIR
        / f"{episode_name}_follower_leader_normalized.png"
    )

    plot_raw_trajectories(
        follower=follower,
        leader=leader,
        time_sec=time_sec,
        output_path=raw_output_path,
    )

    plot_normalized_trajectories(
        normalized_follower=normalized_follower,
        normalized_leader=normalized_leader,
        time_sec=time_sec,
        output_path=normalized_output_path,
    )

    print(f"Follower shape: {follower.shape}")
    print(f"Leader shape:   {leader.shape}")
    print(f"Duration: {num_frames / fps:.2f} seconds")
    print(f"Raw figure:        {raw_output_path.resolve()}")
    print(
        f"Normalized figure: "
        f"{normalized_output_path.resolve()}"
    )

    print_observed_ranges(
        follower=follower,
        leader=leader,
    )

    print_tracking_errors(
        follower=follower,
        leader=leader,
    )


if __name__ == "__main__":
    visualize_arm_trajectories(
        h5_path=(
            "/home/shonosukehida/.stable_worldmodel/datasets/flip_mug/ep200_tm300/per_episode/episode_0.h5"
        ),
        fps=10.0,
    )