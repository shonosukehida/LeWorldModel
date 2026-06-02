import os
import sys
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ["MUJOCO_GL"] = "egl"

import yaml
import numpy as np
import imageio
import pandas as pd
from tqdm import tqdm
from dm_control import mujoco

from env.franka.env import FrankaSimEnv
from scipy.spatial.transform import Rotation as R


def normalize(v, eps=1e-8):
    v = np.asarray(v, dtype=np.float64)
    return v / (np.linalg.norm(v) + eps)


def make_rotmat_plate_perpendicular(push_dir_xy, plate_normal_axis="y"):
    """
    push_dir_xy: box -> goal の xy 方向
    plate_normal_axis:
        "y": EE local-y を push_dir に合わせる
        "x": EE local-x を push_dir に合わせる

    目的:
        板の「法線」を push_dir に合わせる
        => 板の面は push_dir に垂直になる
    """
    push = normalize([push_dir_xy[0], push_dir_xy[1], 0.0])
    z = np.array([0.0, 0.0, -1.0])  # hand を下向きに固定

    if plate_normal_axis == "y":
        y = push
        x = normalize(np.cross(y, z))
        z = normalize(np.cross(x, y))
    elif plate_normal_axis == "x":
        x = push
        y = normalize(np.cross(z, x))
        z = normalize(np.cross(x, y))
    else:
        raise ValueError("plate_normal_axis must be 'x' or 'y'")

    return np.stack([x, y, z], axis=1)


def make_yaw_aligned_rotmat_from_current(env, push_dir_xy, axis="y"):
    """
    現在のEE姿勢を基準にして, world z軸まわりのyawだけ回す。
    これにより Franka が取りやすいpitch/rollを保ったまま,
    local-x or local-y のxy投影を push_dir に合わせる。
    """
    sid = env.physics.model.name2id("ee_target", mujoco.mjtObj.mjOBJ_SITE)
    R_cur = env.physics.data.site_xmat[sid].reshape(3, 3).copy()

    push = normalize([push_dir_xy[0], push_dir_xy[1], 0.0])

    idx = {"x": 0, "y": 1}[axis]
    cur_axis = R_cur[:, idx].copy()
    cur_axis_xy = normalize([cur_axis[0], cur_axis[1], 0.0])

    cur_yaw = np.arctan2(cur_axis_xy[1], cur_axis_xy[0])
    tgt_yaw = np.arctan2(push[1], push[0])
    dyaw = tgt_yaw - cur_yaw

    R_yaw = R.from_euler("z", dyaw).as_matrix()
    return R_yaw @ R_cur

def get_actual_axis(env, axis="y"):
    sid = env.physics.model.name2id("ee_target", mujoco.mjtObj.mjOBJ_SITE)
    R_world = env.physics.data.site_xmat[sid].reshape(3, 3)

    idx = {"x": 0, "y": 1, "z": 2}[axis]
    return R_world[:, idx].copy()


def main():
    config_path = "dataset_gen/franka/push/analyze_gripper_dir/config.yaml"  # ここだけ変更
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    env = FrankaSimEnv(config)

    save_dir = "dataset_gen/franka/push/analyze_gripper_dir/check_plate_perpendicular"
    os.makedirs(save_dir, exist_ok=True)

    box_start = np.array([0.65, 0.00, 0.05])
    box_goal  = np.array([0.75, 0.12, 0.05])

    push_dir = normalize(box_goal[:2] - box_start[:2])

    # ここが重要:
    # まずは "y" で試す。90度ズレるなら "x" に変える。
    plate_normal_axis = "y"
    # target_rotmat = make_rotmat_plate_perpendicular(
    #     push_dir,
    #     plate_normal_axis=plate_normal_axis,
    # )
    # target_rotmat = None

    behind_dist = 0.002
    forward_dist = 0.002
    z_ee = 0.15

    start_xy = box_start[:2] - behind_dist * push_dir
    end_xy   = box_start[:2] + forward_dist * push_dir

    n_targets = 80
    targets = []
    for a in np.linspace(0.0, 1.0, n_targets):
        xy = (1 - a) * start_xy + a * end_xy
        targets.append(np.array([xy[0], xy[1], z_ee]))
    targets = np.asarray(targets)


    # まず位置だけIK
    result0 = env.calc_inverse_kinematic(
        targets[0],
        target_rotmat=None,
    )

    if not result0.success:
        raise RuntimeError("Initial position-only IK failed")

    env.reset_and_place_all(
        box_pos=box_start,
        start_marker_pos=box_start,
        goal_marker_pos=box_goal,
        init_position=result0.qpos[:7],
    )
    env.physics.forward()

    # 位置だけIKで取れた姿勢を基準に, yawだけ合わせる
    plate_normal_axis = "y"
    target_rotmat = make_yaw_aligned_rotmat_from_current(
        env,
        push_dir,
        axis=plate_normal_axis,
    )





    # 初期姿勢を IK で作る
    result = env.calc_inverse_kinematic(
        targets[0],
        target_rotmat=target_rotmat,
        rot_weight=0.1,
    )
    if not result.success:
        raise RuntimeError("Initial IK failed")

    env.reset_and_place_all(
        box_pos=box_start,
        start_marker_pos=box_start,
        goal_marker_pos=box_goal,
        init_position=result.qpos[:7],
    )
    env.physics.forward()

    frames = []
    logs = []

    for i, target_xyz in enumerate(tqdm(targets)):
        try:
            joint_angles, ee_pos, dist_steps, reached = env.step_xyz(
                target_xyz,
                target_rotmat=target_rotmat,
                steps=20,
                tol=0.015,
                rot_weight=0.1,
                max_dq=0.01,
            )
            ik_success = True
        except Exception as e:
            print("[FAILED]", i, target_xyz, e)
            ik_success = False
            ee_pos = np.array([np.nan, np.nan, np.nan])
            reached = False

        env.physics.forward()

        actual_axis = get_actual_axis(env, axis=plate_normal_axis)
        actual_axis_xy = normalize(actual_axis[:2])
        dot = float(np.clip(np.dot(actual_axis_xy, push_dir), -1.0, 1.0))
        angle_deg = np.degrees(np.arccos(dot))

        blue_id = env.physics.model.name2id("blue_box", mujoco.mjtObj.mjOBJ_GEOM)
        blue_pos = env.physics.data.geom_xpos[blue_id].copy()

        logs.append({
            "step": i,
            "target_x": target_xyz[0],
            "target_y": target_xyz[1],
            "target_z": target_xyz[2],
            "ee_x": ee_pos[0],
            "ee_y": ee_pos[1],
            "ee_z": ee_pos[2],
            "box_x": blue_pos[0],
            "box_y": blue_pos[1],
            "box_z": blue_pos[2],
            "push_dir_x": push_dir[0],
            "push_dir_y": push_dir[1],
            "actual_axis_x": actual_axis_xy[0],
            "actual_axis_y": actual_axis_xy[1],
            "angle_deg_axis_to_push": angle_deg,
            "ik_success": ik_success,
            "reached": reached,
        })

        img = env.physics.render(
            height=256,
            width=256,
            camera_id=env.camera_id,
        )
        frames.append(img)

    pd.DataFrame(logs).to_csv(
        os.path.join(save_dir, "plate_perpendicular_log.csv"),
        index=False,
    )
    imageio.mimsave(
        os.path.join(save_dir, "plate_perpendicular_check.mp4"),
        frames,
        fps=10,
    )

    print("saved:", save_dir)
    print("mean angle error [deg]:",
          np.nanmean([r["angle_deg_axis_to_push"] for r in logs]))


if __name__ == "__main__":
    main()