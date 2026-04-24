import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
import imageio
from tqdm import tqdm

from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection


def check_h5_dataset(h5_path: str, save_dir: str = "dataset_gen/franka/push/check/h5dataset"):
    os.makedirs(save_dir, exist_ok=True)

    print(f"=== Checking HDF5 dataset: {h5_path} ===")

    with h5py.File(h5_path, "r") as f:
        keys = list(f.keys())

        print("\n[1] Keys / shapes / dtypes")
        for k in keys:
            print(f"  - {k}: shape={f[k].shape}, dtype={f[k].dtype}")

        required = ["pixels", "action", "ep_len", "ep_offset"]
        print("\n[2] Required keys")
        for k in required:
            if k in f:
                print(f"  ✅ {k}")
            else:
                print(f"  ❌ Missing: {k}")

        pixels = f["pixels"][:]
        action = f["action"][:]
        ep_len = f["ep_len"][:]
        ep_offset = f["ep_offset"][:]

        print("\n[3] Basic consistency checks")

        total_steps = int(ep_len.sum())
        print(f"  total_steps from ep_len sum = {total_steps}")
        print(f"  pixels.shape[0]             = {pixels.shape[0]}")
        print(f"  action.shape[0]             = {action.shape[0]}")

        if total_steps == pixels.shape[0] == action.shape[0]:
            print("  ✅ total steps are consistent")
        else:
            print("  ❌ total steps mismatch")

        if len(ep_len) == len(ep_offset):
            print("  ✅ len(ep_len) == len(ep_offset)")
        else:
            print("  ❌ len(ep_len) != len(ep_offset)")

        ok_offset = True
        cur = 0
        for i, (l, off) in enumerate(zip(ep_len, ep_offset)):
            if off != cur:
                print(f"  ❌ ep_offset mismatch at episode {i}: got {off}, expected {cur}")
                ok_offset = False
                break
            cur += int(l)
        if ok_offset:
            print("  ✅ ep_offset is consistent with ep_len")

        print("\n[4] Value range checks")
        print(f"  pixels min/max = {pixels.min()} / {pixels.max()}")
        print(f"  action min/max = {action.min():.6f} / {action.max():.6f}")

        if "qpos" in f:
            qpos = f["qpos"][:]
            print(f"  qpos   min/max = {qpos.min():.6f} / {qpos.max():.6f}")

        if "qvel" in f:
            qvel = f["qvel"][:]
            print(f"  qvel   min/max = {qvel.min():.6f} / {qvel.max():.6f}")

        if "ee_pos" in f:
            ee_pos = f["ee_pos"][:]
            print(f"  ee_pos min/max = {ee_pos.min():.6f} / {ee_pos.max():.6f}")

        if "bluebox_pos" in f:
            bluebox_pos = f["bluebox_pos"][:]
            print(f"  bluebox_pos min/max = {bluebox_pos.min():.6f} / {bluebox_pos.max():.6f}")

        print("\n[5] First episode info")
        first_len = int(ep_len[0])
        first_off = int(ep_offset[0])
        print(f"  first episode length = {first_len}")
        print(f"  first episode offset = {first_off}")

        # 先頭数枚の画像を保存
        print("\n[6] Saving sample images")
        n_save = min(5, pixels.shape[0])
        for i in range(n_save):
            img = pixels[i]
            out_path = os.path.join(save_dir, f"sample_{i}.png")
            plt.imsave(out_path, img)
            print(f"  saved: {out_path}")

        # 1 episode 内の action / qpos / ee_pos をプロット
        print("\n[7] Saving diagnostic plots for first episode")
        ep_slice = slice(first_off, first_off + first_len)

        # action
        plt.figure(figsize=(10, 4))
        for d in range(action[ep_slice].shape[1]):
            plt.plot(action[ep_slice][:, d], label=f"action_{d}")
        plt.title("First Episode: Action")
        plt.xlabel("step")
        plt.ylabel("value")
        plt.legend(ncol=2, fontsize=8)
        plt.tight_layout()
        act_plot = os.path.join(save_dir, "first_episode_action.png")
        plt.savefig(act_plot, dpi=150)
        plt.close()
        print(f"  saved: {act_plot}")

        if "qpos" in f:
            qpos = f["qpos"][:]
            plt.figure(figsize=(10, 4))
            for d in range(qpos[ep_slice].shape[1]):
                plt.plot(qpos[ep_slice][:, d], label=f"qpos_{d}")
            plt.title("First Episode: qpos")
            plt.xlabel("step")
            plt.ylabel("value")
            plt.legend(ncol=2, fontsize=8)
            plt.tight_layout()
            qpos_plot = os.path.join(save_dir, "first_episode_qpos.png")
            plt.savefig(qpos_plot, dpi=150)
            plt.close()
            print(f"  saved: {qpos_plot}")

        if "ee_pos" in f:
            ee_pos = f["ee_pos"][:]
            plt.figure(figsize=(10, 4))
            labels = ["x", "y", "z"]
            for d in range(ee_pos[ep_slice].shape[1]):
                plt.plot(ee_pos[ep_slice][:, d], label=f"ee_{labels[d]}")
            plt.title("First Episode: ee_pos")
            plt.xlabel("step")
            plt.ylabel("value")
            plt.legend()
            plt.tight_layout()
            ee_plot = os.path.join(save_dir, "first_episode_ee_pos.png")
            plt.savefig(ee_plot, dpi=150)
            plt.close()
            print(f"  saved: {ee_plot}")

        if "bluebox_pos" in f:
            bluebox_pos = f["bluebox_pos"][:]
            plt.figure(figsize=(10, 4))
            labels = ["x", "y", "z"]
            for d in range(bluebox_pos[ep_slice].shape[1]):
                plt.plot(bluebox_pos[ep_slice][:, d], label=f"bluebox_{labels[d]}")
            plt.title("First Episode: bluebox_pos")
            plt.xlabel("step")
            plt.ylabel("value")
            plt.legend()
            plt.tight_layout()
            box_plot = os.path.join(save_dir, "first_episode_bluebox_pos.png")
            plt.savefig(box_plot, dpi=150)
            plt.close()
            print(f"  saved: {box_plot}")

    print("\n✅ Dataset check finished.")


def make_video_from_h5(
    h5_path=None,
    save_dir="dataset_gen/franka/push/check/video",
    fps = 10
):
    """
    LeWM 用 HDF5 dataset から episode ごとの mp4 を保存する。

    前提 key:
      - pixels    : (N, H, W, C)
      - ep_len    : (E,)
      - ep_offset : (E,)
    """
    if h5_path is None:
        h5_path = os.path.expanduser("~/.stable_worldmodel/datasets/franka/push.h5")

    os.makedirs(save_dir, exist_ok=True)

    fps = fps
    print("📦 Loading HDF5 dataset...")
    print(f"   path: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        pixels = f["pixels"]
        ep_len = f["ep_len"][:]
        ep_offset = f["ep_offset"][:]

        print(f"📐 pixels shape: {pixels.shape}")
        print(f"📚 total episodes: {len(ep_len)}")
        print(f"🎞️ FPS: {fps:.3f}")

        for ep_idx, (length, offset) in enumerate(
            tqdm(zip(ep_len, ep_offset), total=len(ep_len), desc="🎬 Saving episodes as videos")
        ):
            length = int(length)
            offset = int(offset)

            episode_frames = pixels[offset: offset + length]  # shape: (T, H, W, C)

            save_path = os.path.join(save_dir, f"episode_{ep_idx:03d}.mp4")
            imageio.mimsave(save_path, episode_frames, fps=fps)

    print("✅ All HDF5 episode videos are saved.")






def confirm_endeffector_trajectory_from_h5(
    h5_path,
    axes: str = "xy",
    save_dir="dataset_gen/franka/push/check/endeffector_trajectory",
    x_range=[0.315, 0.715],
    y_range=[-0.2, 0.2],
    z_range=[0.1, 0.1],
    mgn_x_range=None,
    mgn_y_range=None,
    mgn_z_range=None,
):
    """
    HDF5 dataset から episode ごとの
    - End Effector trajectory
    - Blue box trajectory
    を可視化して保存する。

    必須 key:
      - ee_pos      : (N, 3)
      - bluebox_pos : (N, 3)
      - ep_len      : (E,)
      - ep_offset   : (E,)
    """

    axes_to_num = {"x": 0, "y": 1, "z": 2}
    assert len(axes) == 2, "axes must be like 'xy', 'xz', 'yz'"
    axis_num = [axes_to_num[axes[0]], axes_to_num[axes[1]]]

    # 範囲指定
    ranges = {
        "x": x_range,
        "y": y_range,
        "z": z_range,
    }

    if ranges[axes[0]] is None or ranges[axes[1]] is None:
        raise ValueError(f"x_range/y_range/z_range must be provided for axes='{axes}'")

    xlim = list(ranges[axes[0]])
    ylim = list(ranges[axes[1]])

    if abs(xlim[0] - xlim[1]) < 1e-3:
        xlim[0] -= 0.2
        xlim[1] += 0.2
    if abs(ylim[0] - ylim[1]) < 1e-3:
        ylim[0] -= 0.2
        ylim[1] += 0.2

    os.makedirs(os.path.join(save_dir, axes), exist_ok=True)

    print(f"📦 Loading HDF5 dataset from: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        ee_all = f["ee_pos"][:]
        box_all = f["bluebox_pos"][:]
        ep_len = f["ep_len"][:]
        ep_offset = f["ep_offset"][:]

        print(f"📚 total episodes: {len(ep_len)}")

        for ep_idx, (length, offset) in enumerate(zip(ep_len, ep_offset)):
            length = int(length)
            offset = int(offset)

            ee_xyz = ee_all[offset: offset + length]         # (T, 3)
            bluebox_xyz = box_all[offset: offset + length]   # (T, 3)

            fig, ax = plt.subplots(figsize=(6, 6))

            x_ee = ee_xyz[:, axis_num[0]]
            y_ee = ee_xyz[:, axis_num[1]]

            x_box = bluebox_xyz[:, axis_num[0]]
            y_box = bluebox_xyz[:, axis_num[1]]

            # ---- End-effector trajectory (red) ----
            ax.scatter(
                x_ee, y_ee,
                color="red",
                s=10,
                alpha=0.8,
                zorder=2,
                label="End Effector",
            )
            ax.plot(
                x_ee, y_ee,
                color="red",
                linewidth=1,
                alpha=0.5,
                zorder=1,
            )

            # Start / Goal markers for EE
            ax.text(
                x_ee[0], y_ee[0], "S",
                color="red", fontsize=10, fontweight="bold",
                zorder=5,
            )
            ax.text(
                x_ee[-1], y_ee[-1], "G",
                color="red", fontsize=10, fontweight="bold",
                zorder=5,
            )

            # ---- Blue box trajectory (blue) ----
            ax.scatter(
                x_box, y_box,
                color="blue",
                s=10,
                alpha=0.8,
                zorder=4,
                label="Blue Box",
            )
            ax.plot(
                x_box, y_box,
                color="blue",
                linewidth=1,
                alpha=0.5,
                zorder=3,
            )

            # Start / Goal markers for blue box
            ax.text(
                x_box[0], y_box[0], "S",
                color="blue", fontsize=10, fontweight="bold",
                zorder=5,
            )
            ax.text(
                x_box[-1], y_box[-1], "G",
                color="blue", fontsize=10, fontweight="bold",
                zorder=5,
            )

            # ---- workspace rectangle (optional) ----
            mgn_ranges = {
                "x": mgn_x_range,
                "y": mgn_y_range,
                "z": mgn_z_range,
            }
            if mgn_ranges[axes[0]] is not None and mgn_ranges[axes[1]] is not None:
                low0, high0 = mgn_ranges[axes[0]]
                low1, high1 = mgn_ranges[axes[1]]

                rect = Rectangle(
                    (low0, low1),
                    high0 - low0,
                    high1 - low1,
                    linewidth=1,
                    edgecolor="red",
                    facecolor="none",
                    linestyle="--",
                    alpha=0.3,
                    zorder=3,
                )
                ax.add_patch(rect)

            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.set_xticks(np.linspace(xlim[0], xlim[1], 5))
            ax.set_yticks(np.linspace(ylim[0], ylim[1], 5))

            ax.set_xlabel(axes[0].upper())
            ax.set_ylabel(axes[1].upper())
            ax.set_title(f"EE Trajectory - Episode {ep_idx}")
            ax.set_aspect("equal", adjustable="box")
            ax.legend(loc="best")

            save_path = os.path.join(save_dir, axes, f"ee_trajectory_ep{ep_idx}.png")
            fig.savefig(save_path, dpi=300)
            plt.close(fig)

    print("✅ All trajectory figures are saved.")






def plot_all_endeffector_trajectories_from_h5(
    h5_path,
    axes: str = "xy",
    stride: int = 1,
    max_episodes: int | None = None,
    alpha: float = 1.0,
    lw: float = 1.8,
    show_workspace: bool = True,
    save_dir: str = "dataset_gen/franka/push/check/endeffector_trajectory_all",
    filename: str | None = None,
    mgn_x_range=[0.315, 0.715],
    mgn_y_range=[-0.2, 0.2],
    mgn_z_range=[0.1, 0.1],
):

    """
    HDF5 の全エピソードについて EE 軌跡を1枚に重ね描きする。

    必須 key:
      - ee_pos      : (N, 3)
      - ep_len      : (E,)
      - ep_offset   : (E,)
    """
    print(f"[EE-ALL] Loading: {h5_path}")

    axes_to_num = {"x": 0, "y": 1, "z": 2}
    a0, a1 = axes[0], axes[1]
    i0, i1 = axes_to_num[a0], axes_to_num[a1]

    mgn_ranges = {"x": mgn_x_range, "y": mgn_y_range, "z": mgn_z_range}
    if mgn_ranges[a0] is None or mgn_ranges[a1] is None:
        raise ValueError(f"mgn range for axes '{axes}' must be provided.")

    xlim = list(mgn_ranges[a0])
    ylim = list(mgn_ranges[a1])

    pad_x = 0.02 * (xlim[1] - xlim[0] + 1e-9)
    pad_y = 0.02 * (ylim[1] - ylim[0] + 1e-9)
    xlim = (xlim[0] - pad_x, xlim[1] + pad_x)
    ylim = (ylim[0] - pad_y, ylim[1] + pad_y)

    with h5py.File(h5_path, "r") as f:
        ee_all = f["ee_pos"][:]
        ep_len = f["ep_len"][:]
        ep_offset = f["ep_offset"][:]

        n_eps = len(ep_len) if max_episodes is None else min(len(ep_len), max_episodes)

        segments = []
        for ep_idx in range(n_eps):
            length = int(ep_len[ep_idx])
            offset = int(ep_offset[ep_idx])

            ee_xyz = ee_all[offset: offset + length]          # (T, 3)
            pts = ee_xyz[::stride, :][:, [i0, i1]]            # (K, 2)

            if len(pts) < 2:
                continue

            seg = np.stack([pts[:-1], pts[1:]], axis=1)       # (K-1, 2, 2)
            segments.append(seg)

    if len(segments) == 0:
        print("[EE-ALL] No segments to plot.")
        return

    segments = np.concatenate(segments, axis=0)
    print(f"[EE-ALL] total segments: {len(segments)}")

    fig, ax = plt.subplots(figsize=(7, 7))
    lc = LineCollection(segments, linewidths=lw, alpha=alpha, colors="black")
    ax.add_collection(lc)

    if show_workspace:
        low = {"x": mgn_x_range[0], "y": mgn_y_range[0], "z": mgn_z_range[0]}
        high = {"x": mgn_x_range[1], "y": mgn_y_range[1], "z": mgn_z_range[1]}
        rect = Rectangle(
            (low[a0], low[a1]),
            high[a0] - low[a0],
            high[a1] - low[a1],
            linewidth=1.2,
            edgecolor="black",
            facecolor="none",
            linestyle="--",
            alpha=0.6,
        )
        ax.add_patch(rect)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(a0.upper())
    ax.set_ylabel(a1.upper())
    ax.set_title(f"All End-Effector Trajectories ({axes})")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)

    os.makedirs(save_dir, exist_ok=True)
    if filename is None:
        filename = f"ee_traj_all_{axes}_stride{stride}_ep{n_eps}.png"
    save_path = os.path.join(save_dir, filename)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"[EE-ALL] Saved: {save_path}")




def plot_all_bluebox_trajectories_from_h5(
    h5_path,
    axes: str = "xy",
    stride: int = 1,
    max_episodes: int | None = None,
    alpha: float = 0.25,
    lw: float = 1.6,
    show_workspace: bool = True,
    save_dir: str = "dataset_gen/franka/push/check/bluebox_trajectory_all",
    filename: str | None = None,
    mgn_x_range=[0.315, 0.715],
    mgn_y_range=[-0.2, 0.2],
    mgn_z_range=[0.1, 0.1],
):
    """
    HDF5 の全エピソードについて bluebox 軌跡を1枚に重ね描きする。

    必須 key:
      - bluebox_pos : (N, 3)
      - ep_len      : (E,)
      - ep_offset   : (E,)
    """
    print(f"[BOX-ALL] Loading: {h5_path}")

    axes_to_num = {"x": 0, "y": 1, "z": 2}
    a0, a1 = axes[0], axes[1]
    i0, i1 = axes_to_num[a0], axes_to_num[a1]

    mgn_ranges = {"x": mgn_x_range, "y": mgn_y_range, "z": mgn_z_range}
    if mgn_ranges[a0] is None or mgn_ranges[a1] is None:
        raise ValueError(f"mgn range for axes '{axes}' must be provided.")

    xlim = list(mgn_ranges[a0])
    ylim = list(mgn_ranges[a1])

    pad_x = 0.02 * (xlim[1] - xlim[0] + 1e-9)
    pad_y = 0.02 * (ylim[1] - ylim[0] + 1e-9)
    xlim = (xlim[0] - pad_x, xlim[1] + pad_x)
    ylim = (ylim[0] - pad_y, ylim[1] + pad_y)

    with h5py.File(h5_path, "r") as f:
        box_all = f["bluebox_pos"][:]
        ep_len = f["ep_len"][:]
        ep_offset = f["ep_offset"][:]

        n_eps = len(ep_len) if max_episodes is None else min(len(ep_len), max_episodes)

        segments = []
        for ep_idx in range(n_eps):
            length = int(ep_len[ep_idx])
            offset = int(ep_offset[ep_idx])

            box_xyz = box_all[offset: offset + length]        # (T, 3)
            pts = box_xyz[::stride, :][:, [i0, i1]]           # (K, 2)

            if len(pts) < 2:
                continue

            seg = np.stack([pts[:-1], pts[1:]], axis=1)       # (K-1, 2, 2)
            segments.append(seg)

    if len(segments) == 0:
        print("[BOX-ALL] No segments to plot.")
        return

    segments = np.concatenate(segments, axis=0)
    print(f"[BOX-ALL] total segments: {len(segments)}")

    fig, ax = plt.subplots(figsize=(7, 7))
    lc = LineCollection(segments, linewidths=lw, alpha=alpha, colors="black")
    ax.add_collection(lc)

    if show_workspace:
        low = {"x": mgn_x_range[0], "y": mgn_y_range[0], "z": mgn_z_range[0]}
        high = {"x": mgn_x_range[1], "y": mgn_y_range[1], "z": mgn_z_range[1]}
        rect = Rectangle(
            (low[a0], low[a1]),
            high[a0] - low[a0],
            high[a1] - low[a1],
            linewidth=1.2,
            edgecolor="black",
            facecolor="none",
            linestyle="--",
            alpha=0.6,
        )
        ax.add_patch(rect)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(a0.upper())
    ax.set_ylabel(a1.upper())
    ax.set_title(f"All Bluebox Trajectories ({axes})")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.2)

    os.makedirs(save_dir, exist_ok=True)
    if filename is None:
        filename = f"bluebox_traj_all_{axes}_stride{stride}_ep{n_eps}.png"
    save_path = os.path.join(save_dir, filename)
    fig.savefig(save_path, dpi=300)
    plt.close(fig)
    print(f"[BOX-ALL] Saved: {save_path}")



if __name__ == "__main__":
    DATA_PATH = "/home/shonosukehida/.stable_worldmodel/datasets/franka/pairs_1_ep_1_timestep_500_sample_mix_direction_towards_bluebox_1p00_1p00_view_top_reverse/push.h5"
    
    h5_path = os.path.expanduser(DATA_PATH)
    print("h5_path:", h5_path)
    check_h5_dataset(h5_path)
    make_video_from_h5(h5_path)
    confirm_endeffector_trajectory_from_h5(h5_path)
    plot_all_endeffector_trajectories_from_h5(h5_path)
    plot_all_bluebox_trajectories_from_h5(h5_path)