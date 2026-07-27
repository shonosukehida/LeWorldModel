"""Visualize end-effector trajectories for all episodes in XYZ space."""

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

SAVE_DIR = Path(
    "/home/hida/LeWorldModel/robot/utils/plots/figures"
)

SAVE_FILENAME = "end_effector_trajectories_xyz.png"


# ワークスペースの範囲[m]
WORKSPACE_MIN_X = 499.5 / 1000.0

WORKSPACE_MIN_Y = -216.8 / 1000.0
WORKSPACE_MAX_Y = 207.3 / 1000.0

WORKSPACE_MIN_Z = 0.0 / 1000.0
WORKSPACE_MAX_Z = 475.2 / 1000.0


def load_dataset(
    dataset_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """HDF5からaction_cartesian、ep_len、ep_offsetを読み込む。

    Returns:
        xyz:
            action_cartesianの先頭3次元。
            Shape: (total_steps, 3)

        ep_len:
            各エピソードの長さ。
            Shape: (num_episodes,)

        ep_offset:
            各エピソードの開始位置。
            Shape: (num_episodes,)
    """
    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Dataset was not found: {dataset_path}"
        )

    with h5py.File(dataset_path, "r") as file:
        required_keys = [
            "action_cartesian",
            "ep_len",
            "ep_offset",
        ]

        for key in required_keys:
            if key not in file:
                raise KeyError(
                    f"Dataset key '{key}' was not found in "
                    f"{dataset_path}"
                )

        action_cartesian = np.asarray(
            file["action_cartesian"][:],
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

    if (
        action_cartesian.ndim != 2
        or action_cartesian.shape[1] < 3
    ):
        raise ValueError(
            "action_cartesian must have shape "
            "(total_steps, >=3), "
            f"but got {action_cartesian.shape}"
        )

    if ep_len.ndim != 1 or ep_offset.ndim != 1:
        raise ValueError(
            "ep_len and ep_offset must be "
            "one-dimensional arrays"
        )

    if len(ep_len) != len(ep_offset):
        raise ValueError(
            "ep_len and ep_offset have different lengths: "
            f"{len(ep_len)} != {len(ep_offset)}"
        )

    # action_cartesianの先頭3次元がx, y, z
    xyz = action_cartesian[:, :3]
    print(f"X: {xyz[:,0].min():.4f} ~ {xyz[:,0].max():.4f}")
    print(f"Y: {xyz[:,1].min():.4f} ~ {xyz[:,1].max():.4f}")
    print(f"Z: {xyz[:,2].min():.4f} ~ {xyz[:,2].max():.4f}")

    if not np.all(np.isfinite(xyz)):
        raise ValueError(
            "action_cartesian contains NaN or infinity "
            "in its XYZ dimensions"
        )

    return xyz, ep_len, ep_offset


def validate_episode_ranges(
    total_steps: int,
    ep_len: np.ndarray,
    ep_offset: np.ndarray,
) -> None:
    """各エピソードの範囲がデータ内に収まっているか確認する。"""
    for episode_index, (offset, length) in enumerate(
        zip(ep_offset, ep_len)
    ):
        start = int(offset)
        episode_length = int(length)
        end = start + episode_length

        if episode_length <= 0:
            raise ValueError(
                "Episode length must be positive:\n"
                f"  episode={episode_index}\n"
                f"  length={episode_length}"
            )

        if start < 0 or end > total_steps:
            raise IndexError(
                "Invalid episode range:\n"
                f"  episode={episode_index}\n"
                f"  start={start}\n"
                f"  end={end}\n"
                f"  total_steps={total_steps}"
            )


def set_axes_equal_scale(
    axis: plt.Axes,
    x_limits: tuple[float, float],
    y_limits: tuple[float, float],
    z_limits: tuple[float, float],
) -> None:
    """XYZの数値スケールに応じて3Dボックスの比率を設定する。"""
    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]

    axis.set_box_aspect(
        (
            max(x_range, 1.0),
            max(y_range, 1.0),
            max(z_range, 1.0),
        )
    )


def plot_end_effector_trajectories(
    dataset_path: Path,
    save_dir: Path,
) -> None:
    """全エピソードのエンドエフェクタ軌跡を3D表示する。"""
    xyz, ep_len, ep_offset = load_dataset(
        dataset_path
    )

    validate_episode_ranges(
        total_steps=len(xyz),
        ep_len=ep_len,
        ep_offset=ep_offset,
    )

    num_episodes = len(ep_len)

    # x軸上限は、データ中のx座標の最大値を使用する
    data_max_x = float(np.max(xyz[:, 0]))

    if data_max_x <= WORKSPACE_MIN_X:
        raise ValueError(
            "The maximum x coordinate is not greater than "
            "the workspace minimum:\n"
            f"  workspace min x={WORKSPACE_MIN_X}\n"
            f"  data max x={data_max_x}"
        )

    x_limits = (
        WORKSPACE_MIN_X,
        data_max_x,
    )

    y_limits = (
        WORKSPACE_MIN_Y,
        WORKSPACE_MAX_Y,
    )

    z_limits = (
        WORKSPACE_MIN_Z,
        WORKSPACE_MAX_Z,
    )

    print(f"Dataset      : {dataset_path}")
    print(f"XYZ shape    : {xyz.shape}")
    print(f"Num episodes : {num_episodes}")
    print()
    print(
        f"Data X range : "
        f"[{xyz[:, 0].min():.3f}, "
        f"{xyz[:, 0].max():.3f}]"
    )
    print(
        f"Data Y range : "
        f"[{xyz[:, 1].min():.3f}, "
        f"{xyz[:, 1].max():.3f}]"
    )
    print(
        f"Data Z range : "
        f"[{xyz[:, 2].min():.3f}, "
        f"{xyz[:, 2].max():.3f}]"
    )
    print()
    print(f"Plot X range : {x_limits}")
    print(f"Plot Y range : {y_limits}")
    print(f"Plot Z range : {z_limits}")

    fig = plt.figure(
        figsize=(12, 10)
    )

    axis = fig.add_subplot(
        111,
        projection="3d",
    )

    for episode_index, (offset, length) in enumerate(
        zip(ep_offset, ep_len)
    ):
        start = int(offset)
        end = start + int(length)

        episode_xyz = xyz[start:end]

        axis.plot(
            episode_xyz[:, 0],
            episode_xyz[:, 1],
            episode_xyz[:, 2],
            color="tab:blue",
            alpha=0.35,
            linewidth=1.2,
        )

    axis.set_xlim(x_limits)
    axis.set_ylim(y_limits)
    axis.set_zlim(z_limits)

    set_axes_equal_scale(
        axis=axis,
        x_limits=x_limits,
        y_limits=y_limits,
        z_limits=z_limits,
    )

    axis.set_xlabel("X [m]")
    axis.set_ylabel("Y [m]")
    axis.set_zlabel("Z [m]")

    axis.set_title(
        "End-Effector XYZ Trajectories\n"
        f"All {num_episodes} Episodes"
    )

    axis.grid(
        True,
        alpha=0.3,
    )

    # 見やすい角度に設定
    axis.view_init(
        elev=25,
        azim=-60,
    )

    fig.tight_layout()

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    save_path = save_dir / SAVE_FILENAME

    fig.savefig(
        save_path,
        dpi=200,
        bbox_inches="tight",
    )

    print(f"\nSaved figure to: {save_path}")

    plt.show()
    plt.close(fig)


def main() -> None:
    plot_end_effector_trajectories(
        dataset_path=DATASET_PATH,
        save_dir=SAVE_DIR,
    )


if __name__ == "__main__":
    main()