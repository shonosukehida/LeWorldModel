import os
import sys
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../../../")
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

os.environ["MUJOCO_GL"] = "egl"

import yaml
import h5py
import numpy as np
from tqdm import tqdm
from dm_control import mujoco

from env.franka.env import FrankaSimEnv
from stable_worldmodel.data.utils import get_cache_dir

import imageio


def get_center_xyz(config):
    x = np.mean(config["x_range"])
    y = np.mean(config["y_range"])
    z = np.mean(config["z_range"])
    return np.array([x, y, z], dtype=np.float32)


def render_fixed_box_dataset(
    config,
    save_name="shadow_occlusion_probe_box0p03",
    n_shadow=100,
    n_clear=100,
    n_nobox=100,
):
    image_size = tuple(config["image_size"])
    camera_name = config.get("camera_name", "")
    

    env = FrankaSimEnv(config)

    bluebox_geom_id = env.physics.model.name2id(
        "blue_box", mujoco.mjtObj.mjOBJ_GEOM
    )

    if camera_name == "default":
        camera_id = -1
    else:
        camera_id = env.physics.model.name2id(
            camera_name, mujoco.mjtObj.mjOBJ_CAMERA
        )

    # ===== Franka の位置を固定 =====
    init_xyz = get_center_xyz(config)

    target_rotmat = None
    if config.get("freeze_quat", False):
        x_axis = np.array(config["ee_x"])
        y_axis = np.array(config["ee_y"])
        z_axis = np.array(config["ee_z"])
        target_rotmat = np.stack([x_axis, y_axis, z_axis], axis=1)

    result = env.calc_inverse_kinematic(
        init_xyz,
        target_rotmat=target_rotmat,
    )
    init_joint = result.qpos[:7]

    # ===== bluebox 配置リスト =====
    x_min, x_max = config["start_goal_x_range"]
    z_box = float(np.mean(config["start_goal_z_range"]))

    samples = []

    # 1. shadow: y=0
    xs = np.linspace(x_min, x_max, n_shadow)
    for x in xs:
        samples.append({
            "label": 0,  # shadow
            "box_pos": np.array([x, 0.0, z_box], dtype=np.float32),
        })

    # 2. clear: y=±0.2
    # total n_clear 枚になるように半分ずつ
    n_half = n_clear // 2
    xs_pos = np.linspace(x_min, x_max, n_half)
    xs_neg = np.linspace(x_min, x_max, n_clear - n_half)

    y_range = config.get("y_range", "")
    for x in xs_pos:
        samples.append({
            "label": 1,  # clear
            "box_pos": np.array([x, y_range[1], z_box], dtype=np.float32),
        })

    for x in xs_neg:
        samples.append({
            "label": 1,  # clear
            "box_pos": np.array([x, y_range[0], z_box], dtype=np.float32),
        })

    # 3. no_box: blueboxを画面外に置く
    for _ in range(n_nobox):
        samples.append({
            "label": 2,  # no_box
            "box_pos": np.array([1000.0, 1000.0, z_box], dtype=np.float32),
        })

    pixels = []
    actions = []
    action_joint = []
    action_cartesian = []
    qpos_list = []
    qvel_list = []
    ee_pos_list = []
    bluebox_pos_list = []
    labels = []
    ep_len = []
    ep_offset = []
    ep_idx = []
    step_idx = []

    print(f"total samples: {len(samples)}")

    for i, s in enumerate(tqdm(samples)):
        box_pos = s["box_pos"]

        env.reset_and_place_all(
            box_pos=box_pos,
            start_marker_pos=box_pos,
            goal_marker_pos=box_pos,
            init_position=init_joint,
        )

        # settle
        for _ in range(config.get("settle_steps", 10)):
            env.physics.forward()

        img = env.physics.render(
            height=image_size[0],
            width=image_size[1],
            camera_id=camera_id,
        )
        
        save_debug_image(
            img=img,
            save_dir="dataset_gen/franka/push/analyze_shade_image/images",
            sample_idx=i,
            label=s["label"],
            box_pos=box_pos,
        )

        bluebox_pos = env.physics.data.geom_xpos[bluebox_geom_id].copy()
        ee_pos = env.get_ee_position().copy()

        pixels.append(img.astype(np.uint8))

        # PCA/encoder解析用なので action は dummy
        actions.append(np.zeros(7, dtype=np.float32))
        action_joint.append(np.zeros(7, dtype=np.float32))
        action_cartesian.append(ee_pos.astype(np.float32))

        qpos_list.append(env.physics.data.qpos[:7].copy().astype(np.float32))
        qvel_list.append(env.physics.data.qvel[:7].copy().astype(np.float32))
        ee_pos_list.append(ee_pos.astype(np.float32))
        bluebox_pos_list.append(bluebox_pos.astype(np.float32))
        labels.append(s["label"])

        # 1画像 = 1 timestep episode として保存
        ep_len.append(1)
        ep_offset.append(i)
        ep_idx.append(i)
        step_idx.append(0)

    pixels = np.asarray(pixels, dtype=np.uint8)
    actions = np.asarray(actions, dtype=np.float32)
    action_joint = np.asarray(action_joint, dtype=np.float32)
    action_cartesian = np.asarray(action_cartesian, dtype=np.float32)
    qpos_list = np.asarray(qpos_list, dtype=np.float32)
    qvel_list = np.asarray(qvel_list, dtype=np.float32)
    ee_pos_list = np.asarray(ee_pos_list, dtype=np.float32)
    bluebox_pos_list = np.asarray(bluebox_pos_list, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int32)

    ep_len = np.asarray(ep_len, dtype=np.int32)
    ep_offset = np.asarray(ep_offset, dtype=np.int64)
    ep_idx = np.asarray(ep_idx, dtype=np.int32)
    step_idx = np.asarray(step_idx, dtype=np.int32)

    datasets_dir = get_cache_dir(sub_folder="datasets")
    save_dir = os.path.join(datasets_dir, "franka", save_name)
    os.makedirs(save_dir, exist_ok=True)

    h5_path = os.path.join(save_dir, "push.h5")

    with h5py.File(h5_path, "w") as f:
        f.create_dataset("pixels", data=pixels, compression="gzip")
        f.create_dataset("action", data=actions, compression="gzip")
        f.create_dataset("action_joint", data=action_joint, compression="gzip")
        f.create_dataset("action_cartesian", data=action_cartesian, compression="gzip")

        f.create_dataset("qpos", data=qpos_list)
        f.create_dataset("qvel", data=qvel_list)
        f.create_dataset("ee_pos", data=ee_pos_list)
        f.create_dataset("bluebox_pos", data=bluebox_pos_list)

        f.create_dataset("label", data=labels)
        f.create_dataset("ep_len", data=ep_len)
        f.create_dataset("ep_offset", data=ep_offset)
        f.create_dataset("ep_idx", data=ep_idx)
        f.create_dataset("step_idx", data=step_idx)

        f.attrs["label_0"] = "shadow_y0"
        f.attrs["label_1"] = "clear_y_pm_0p2"
        f.attrs["label_2"] = "no_box"

    print("saved:", h5_path)
    print("pixels:", pixels.shape)
    print("labels:", np.bincount(labels))

    return h5_path



def save_debug_image(
    img,
    save_dir,
    sample_idx,
    label,
    box_pos,
):
    label_names = {
        0: "shadow",
        1: "clear",
        2: "no_box",
    }

    os.makedirs(save_dir, exist_ok=True)

    label_name = label_names[label]
    
    label_dir = os.path.join(
        save_dir,
        label_name,
    )

    os.makedirs(label_dir, exist_ok=True)

    x, y, z = box_pos

    filename = (
        f"{sample_idx:04d}"
        f"_label{label}"
        f"_{label_name}"
        f"_x{x:.3f}"
        f"_y{y:.3f}.png"
    )

    imageio.imwrite(
        os.path.join(label_dir, filename),
        img,
    )



if __name__ == "__main__":
    with open("/home/shonosukehida/work/LeWorldModel/dataset_gen/franka/push/analyze_shade_image/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    print(config)

    render_fixed_box_dataset(
        config,
        save_name="shadow_occlusion_probe_box0p05",
        n_shadow=config["num_data"],
        n_clear=config["num_data"],
        n_nobox=config["num_data"],
    )