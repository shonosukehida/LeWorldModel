# scripts/check_push_y_straight.py

import os
import sys
os.environ["MUJOCO_GL"] = "egl"

PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import argparse
import numpy as np
import pandas as pd
import imageio
import yaml
from tqdm import tqdm
from dm_control import mujoco

from env.franka.env import FrankaSimEnv

from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import LineCollection
import matplotlib.cm as cm
import numpy as np
import pandas as pd

def plot_franka_box_traj_xy(
    csv_path,
    save_path=None,
    workspace_x=(0.315, 0.715),
    workspace_y=(-0.2, 0.2),
    box_half_size=(0.05, 0.05),
    title="Franka EE and Bluebox Trajectory",
):
    df = pd.read_csv(csv_path)

    bluebox_traj = df[["bluebox_x", "bluebox_y", "bluebox_z"]].values
    ee_traj = df[["ee_x", "ee_y", "ee_z"]].values
    target_traj = df[["target_x", "target_y", "target_z"]].values

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)

    # workspace: 横軸Y, 縦軸X
    ws = Rectangle(
        (workspace_y[0], workspace_x[0]),
        workspace_y[1] - workspace_y[0],
        workspace_x[1] - workspace_x[0],
        fill=False,
        linestyle="--",
        linewidth=1.5,
        edgecolor="gray",
        label="workspace",
    )
    ax.add_patch(ws)

    # bluebox trajectory
    ax.plot(
        bluebox_traj[:, 1],
        bluebox_traj[:, 0],
        linewidth=2.5,
        label="bluebox traj",
    )

    ax.scatter(
        bluebox_traj[0, 1],
        bluebox_traj[0, 0],
        marker="x",
        s=70,
        linewidths=2,
        label="bluebox start",
    )

    ax.scatter(
        bluebox_traj[-1, 1],
        bluebox_traj[-1, 0],
        marker="o",
        s=50,
        label="bluebox end",
    )

    # EE trajectory with time gradient
    points = np.stack([ee_traj[:, 1], ee_traj[:, 0]], axis=1)
    segments = np.concatenate(
        [points[:-1, None, :], points[1:, None, :]],
        axis=1,
    )

    t = np.linspace(0, 1, max(len(segments), 1))
    lc = LineCollection(
        segments,
        cmap=cm.get_cmap("Reds"),
        norm=plt.Normalize(0, 1),
    )
    lc.set_array(t)
    lc.set_linewidth(2.5)
    ax.add_collection(lc)

    ax.scatter(
        ee_traj[0, 1],
        ee_traj[0, 0],
        s=40,
        label="EE start",
    )

    ax.scatter(
        ee_traj[-1, 1],
        ee_traj[-1, 0],
        s=40,
        label="EE end",
    )

    # target trajectory
    ax.plot(
        target_traj[:, 1],
        target_traj[:, 0],
        linestyle=":",
        linewidth=1.5,
        label="target traj",
    )

    # bluebox start/end region
    hx, hy = box_half_size

    start_rect = Rectangle(
        (bluebox_traj[0, 1] - hy, bluebox_traj[0, 0] - hx),
        2 * hy,
        2 * hx,
        fill=False,
        linewidth=1.5,
        alpha=0.8,
        label="box start region",
    )
    ax.add_patch(start_rect)

    end_rect = Rectangle(
        (bluebox_traj[-1, 1] - hy, bluebox_traj[-1, 0] - hx),
        2 * hy,
        2 * hx,
        fill=False,
        linewidth=1.8,
        linestyle="--",
        alpha=0.9,
        label="box end region",
    )
    ax.add_patch(end_rect)

    ax.text(
        bluebox_traj[0, 1],
        bluebox_traj[0, 0] + 0.01,
        "S",
        fontsize=18,
        weight="bold",
        ha="center",
    )

    ax.text(
        bluebox_traj[-1, 1],
        bluebox_traj[-1, 0] + 0.01,
        "E",
        fontsize=18,
        weight="bold",
        ha="center",
    )

    ax.set_xlabel("Y [m]")
    ax.set_ylabel("X [m]")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_aspect("equal", adjustable="box")
    ax.relim()
    ax.autoscale_view()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight")

    plt.show()


def build_target_rotmat(config):
    if not config.get("freeze_quat", False):
        return None

    x = np.array(config["ee_x"], dtype=np.float32)
    y = np.array(config["ee_y"], dtype=np.float32)
    z = np.array(config["ee_z"], dtype=np.float32)
    return np.stack([x, y, z], axis=1)


def get_bluebox_pos(env):
    gid = env.physics.model.name2id("blue_box", mujoco.mjtObj.mjOBJ_GEOM)
    env.physics.forward()
    return env.physics.data.geom_xpos[gid].copy()


def run_straight_x_push(config, direction="+x"):
    save_dir = config.get(
        "check_save_dir",
        "dataset_gen/franka/push/analyze_onedir_move/result"
    )
    os.makedirs(save_dir, exist_ok=True)

    env = FrankaSimEnv(config)

    image_size = tuple(config["image_size"])
    target_rotmat = build_target_rotmat(config)

    # ---- 検証用パラメータ ----
    box_start = np.array(config.get("box_start_pos", [0.65, 0.00, 0.05]), dtype=np.float32)
    ee_start = np.array(config.get("ee_start_pos", [0.55, 0.00, 0.05]), dtype=np.float32)

    push_sign = 1.0 if direction == "+x" else -1.0

    # box の手前側に EE を置く
    behind_dist = config.get("check_behind_dist", 0.08)
    push_dist = config.get("check_push_dist", 0.18)
    z = config.get("check_ee_z", 0.05)


    # ee_start[0] -= push_sign * behind_dist
    ee_start[2] = z

    ee_goal = ee_start.copy()
    ee_goal[0] += push_sign * push_dist

    start_marker = box_start.copy()
    goal_marker = box_start.copy()
    goal_marker[0] += push_sign * push_dist

    # ---- 初期姿勢を IK で作って reset ----
    ik = env.calc_inverse_kinematic(
        ee_start,
        target_rotmat=target_rotmat,
        rot_weight=config.get("rot_weight", 1.0),
    )
    if not ik.success:
        raise RuntimeError(f"Initial IK failed: ee_start={ee_start}")

    env.reset_and_place_all(
        box_pos=box_start,
        start_marker_pos=start_marker,
        goal_marker_pos=goal_marker,
        init_position=ik.qpos[:7],
    )
    env.physics.forward()

    # 少し安定化
    for _ in range(config.get("check_settle_steps", 50)):
        env.physics.step()
    env.physics.forward()

    frames = []
    logs = []

    n_targets = config.get("check_n_targets", 80)
    steps_per_target = config.get("check_steps_per_target", 5)
    max_dq = config.get("max_dq", 0.01)
    tol = float(config.get("tol", 0.02))
    rot_weight = config.get("rot_weight", 1.0)

    targets = np.linspace(ee_start, ee_goal, n_targets)

    for i, target_xyz in enumerate(tqdm(targets, desc=f"Straight push {direction}")):
        try:
            _, ee_pos, dist_steps, reached = env.step_xyz(
                target_xyz,
                target_rotmat=target_rotmat,
                steps=steps_per_target,
                tol=tol,
                rot_weight=rot_weight,
                max_dq=max_dq,
            )
        except Exception as e:
            print(f"[WARN] step failed at {i}: target={target_xyz}, error={e}")
            ee_pos = env.get_ee_position()
            reached = False

        bluebox_pos = get_bluebox_pos(env)

        frame = env.physics.render(
            height=image_size[0],
            width=image_size[1],
            camera_id=env.camera_id,
        )
        frames.append(frame)

        logs.append({
            "step": i,
            "target_x": target_xyz[0],
            "target_y": target_xyz[1],
            "target_z": target_xyz[2],
            "ee_x": ee_pos[0],
            "ee_y": ee_pos[1],
            "ee_z": ee_pos[2],
            "bluebox_x": bluebox_pos[0],
            "bluebox_y": bluebox_pos[1],
            "bluebox_z": bluebox_pos[2],
            "reached": reached,
        })

    direction_tag = "plus_y" if direction == "+y" else "minus_y"

    video_path = os.path.join(save_dir, f"straight_push_{direction_tag}.mp4")
    csv_path = os.path.join(save_dir, f"straight_push_{direction_tag}.csv")

    imageio.mimsave(video_path, frames, fps=config.get("check_video_fps", 20))
    pd.DataFrame(logs).to_csv(csv_path, index=False)

    blue_start = np.array([logs[0]["bluebox_x"], logs[0]["bluebox_y"], logs[0]["bluebox_z"]])
    blue_end = np.array([logs[-1]["bluebox_x"], logs[-1]["bluebox_y"], logs[-1]["bluebox_z"]])

    print("✅ saved video:", video_path)
    print("✅ saved csv:", csv_path)
    print("bluebox displacement:", blue_end - blue_start)
    
    plot_path = os.path.join(save_dir, f"straight_push_{direction_tag}_traj.png")

    plot_franka_box_traj_xy(
        csv_path=csv_path,
        save_path=plot_path,
        workspace_x=tuple(config.get("x_range", [0.315, 0.715])),
        workspace_y=tuple(config.get("y_range", [-0.2, 0.2])),
        box_half_size=tuple(config.get("box_half_size", [0.05, 0.05])),
        title=f"Straight Push {direction}",
    )

    print("✅ saved plot:", plot_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--direction", choices=["+x", "-x"], default="+x")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    run_straight_x_push(config, direction=args.direction)


if __name__ == "__main__":
    main()