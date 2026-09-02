import os

os.environ["MUJOCO_GL"] = "egl"

import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
from stable_worldmodel.data.utils import get_cache_dir
import stable_worldmodel as swm
import env.franka

from stable_worldmodel.probing.flip_mug.probe_evaluator import ProbingEvaluator
from stable_worldmodel.probing.flip_mug.probe_evaluator_no_propio import ProbingEvaluator_NoProprio
from env.franka.env import FrankaSimEnv
import h5py
from transformers import ViTModel

import signal

import cv2
import gymnasium as gym
from scipy.spatial.transform import Rotation
import ctypes

import matplotlib.pyplot as plt
import json
import subprocess

from action_projector import XArmActionProjector


def img_transform(cfg):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"

    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    
    keys_to_load = list(cfg.dataset.keys_to_cache)
    # if "pixels" not in keys_to_load:
    #     keys_to_load.append("pixels")
    # if "step_idx" not in keys_to_load:
    #     keys_to_load.append("step_idx")
    # if "ep_idx" not in keys_to_load:
    #     keys_to_load.append("ep_idx")
    # if "bluebox_pos" not in keys_to_load:
    #     keys_to_load.append("bluebox_pos")
    # if "ee_pos" not in keys_to_load:
    #     keys_to_load.append("ee_pos")
    # if "qpos" not in keys_to_load:
    #     keys_to_load.append("qpos") 
    # if "qvel" not in keys_to_load:
    #     keys_to_load.append("qvel")
    
    # print("key_to_cache:", cfg.dataset.keys_to_cache)
        
    dataset = swm.data.HDF5Dataset(
        dataset_name,
        # keys_to_load=keys_to_load,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )
    return dataset

#影置き, 影なし置き, 置かずの画像を集めたデータセットを取得
def get_shaded_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())

    keys_to_load = [
        "pixels",
        "label",
        "bluebox_pos",
        "ee_pos",
        "qpos",
        "qvel",
        "step_idx",
        "ep_idx",
        "action_cartesian",
    ]

    keys_to_cache = [
        "label",
        "bluebox_pos",
        "ee_pos",
        "qpos",
        "qvel",
        "action_cartesian",
    ]

    dataset = swm.data.HDF5Dataset(
        dataset_name,
        keys_to_load=keys_to_load,
        keys_to_cache=keys_to_cache,
        cache_dir=dataset_path,
    )
    return dataset


def get_workspace_center_from_h5(dataset_name):
    h5_path = os.path.join(
        get_cache_dir(sub_folder="datasets"),
        f"{dataset_name}.h5"
    )

    with h5py.File(h5_path, "r") as f:
        x_range = np.asarray(f.attrs["x_range"], dtype=np.float32)
        y_range = np.asarray(f.attrs["y_range"], dtype=np.float32)
        z_range = np.asarray(f.attrs["z_range"], dtype=np.float32)

    center = np.array([
        (x_range[0] + x_range[1]) / 2,
        (y_range[0] + y_range[1]) / 2,
        (z_range[0] + z_range[1]) / 2,
    ], dtype=np.float32)

    return center

def polar_to_xyz(polar, center):
    """
    polar: [r, theta, z]
    theta は radian 想定
    center: workspace center [cx, cy, cz]
    """
    r, theta_deg, z = polar
    theta_deg = - (theta_deg - 90.)
    
    theta = np.deg2rad(theta_deg)
    
    return np.array([
        center[0] + r * np.cos(theta),
        center[1] + r * np.sin(theta),
        z,
    ], dtype=np.float32)



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
        std = np.std(x, axis=0, keepdims=True)
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

    np.savez_compressed(stats_path, **arrays)
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



class XArmInferenceEnv:
    """Minimal xArm7/RealSense adapter used only by the real-robot rollout.

    Robot positions are exposed in metres/radians.

    Real-robot motion is delegated to Robopy's XArmFollower so that inference
    uses the same low-level controller family as data collection:
        command_joint_state()
            -> Robopy background control loop
            -> max_delta smoothing
            -> FK
            -> xArm SDK set_position()
    """

    def __init__(self, robot_cfg, plan_cfg, use_camera=True):
        self.cfg = robot_cfg
        self.num_envs = 1
        self.dry_run = bool(robot_cfg.dry_run)
        self.use_camera = bool(use_camera)

        bounds = np.asarray(robot_cfg.workspace_bounds_m, dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=np.array([[
                bounds[0, 0], bounds[1, 0], bounds[2, 0],
                -1.0, -1.0, -1.0, -1.0, 0.0,
            ]], dtype=np.float32),
            high=np.array([[
                bounds[0, 1], bounds[1, 1], bounds[2, 1],
                1.0, 1.0, 1.0, 1.0, 1.0,
            ]], dtype=np.float32),
            dtype=np.float32,
        )

        self._last_qpos = np.zeros(7, dtype=np.float32)
        self._last_qvel = np.zeros(7, dtype=np.float32)
        self._last_ee = np.array(
            [0.5, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0],
            dtype=np.float32,
        )
        self._last_gripper = np.float32(0.0)

        # Robopy owns the xArm connection/control thread.
        self._follower = None

        # Read-only access to the XArmAPI instance owned by XArmFollower.
        # This is retained for SDK FK, TCP/world offsets, and qvel queries.
        self._robot = None
        self._overhead_pipeline = None
        self._wrist_pipeline = None
        
        self._ik_solver = None
        self._fk_solver = None

        # Desired joint target given to the Robopy controller.
        self._last_target_qpos = np.full(7, np.nan, dtype=np.float32)
        self._last_command_qpos = np.full(7, np.nan, dtype=np.float32)

        # Compatibility alias for existing logging code.
        # With the Robopy backend this is NOT Robopy's internal per-cycle
        # max_delta-limited joint state; it is the target submitted to
        # command_joint_state().
        self._last_safe_qpos = np.full(7, np.nan, dtype=np.float32)

        if not self.dry_run:
            try:
                from robopy.config.robot_config import (
                    XArmConfig,
                    XArmWorkspaceBounds,
                )
                from robopy.robots.xarm.xarm_follower import XArmFollower
            except ImportError as exc:
                raise ImportError(
                    "Real execution requires robopy with xArm support"
                ) from exc

            if self.use_camera:
                try:
                    import pyrealsense2 as rs
                except ImportError as exc:
                    raise ImportError(
                        "RealSense execution requires pyrealsense2"
                    ) from exc

            def _optional_cfg(name, default):
                try:
                    value = getattr(robot_cfg, name)
                except (AttributeError, KeyError):
                    return default
                return default if value is None else value

            # XArmWorkspaceBounds in Robopy uses millimetres.
            workspace = XArmWorkspaceBounds(
                min_x=float(bounds[0, 0] * 1000.0),
                max_x=float(bounds[0, 1] * 1000.0),
                min_y=float(bounds[1, 0] * 1000.0),
                max_y=float(bounds[1, 1] * 1000.0),
                min_z=float(bounds[2, 0] * 1000.0),
                max_z=float(bounds[2, 1] * 1000.0),
            )

            # Defaults intentionally match Robopy XArmConfig defaults used by
            # the collection script when these fields were not specified.
            follower_cfg = XArmConfig(
                follower_ip=str(robot_cfg.follower_ip),
                workspace_bounds=workspace,
                control_frequency=float(
                    _optional_cfg("control_frequency", 50.0)
                ),
                max_delta=float(
                    _optional_cfg("max_delta", 0.05)
                ),
                cartesian_speed=int(
                    _optional_cfg("cartesian_speed_mm_s", 300)
                ),
                cartesian_mvacc=int(
                    _optional_cfg("cartesian_mvacc", 1000)
                ),
                collision_sensitivity=int(
                    _optional_cfg("collision_sensitivity", 3)
                ),
                gripper_open=int(robot_cfg.gripper.open_position),
                gripper_close=int(robot_cfg.gripper.closed_position),
                gripper_speed=int(robot_cfg.gripper.speed),
            )

            self._follower = XArmFollower(follower_cfg)
            self._follower.connect()

            # XArmFollower owns this XArmAPI object. Do not connect/disconnect
            # it separately.
            self._robot = self._follower._robot
            if self._robot is None:
                raise RuntimeError(
                    "Robopy XArmFollower connected without an XArmAPI handle"
                )

            print(
                "Robopy follower control_frequency:",
                follower_cfg.control_frequency,
            )
            print(
                "Robopy follower max_delta:",
                follower_cfg.max_delta,
            )

            tcp_offset = getattr(self._robot, "tcp_offset", None)
            world_offset = getattr(self._robot, "world_offset", None)

            print("SDK tcp_offset:", tcp_offset)
            print("SDK world_offset:", world_offset)

            self._ik_solver = XArm7IK(
                "xarm_kinematics_user_lib_20251009_x86_64_fPIC_gcc9/"
                "libxarm7_capi.so",
                tcp_offset=tcp_offset,
                world_offset=world_offset,
            )

            self._fk_solver = XArm7FK(
                "xarm_kinematics_user_lib_20251009_x86_64_fPIC_gcc9/"
                "libxarm7_capi.so",
                tcp_offset=tcp_offset,
                world_offset=world_offset,
            )


            def _start_camera(camera_cfg):
                pipeline = rs.pipeline()
                rs_cfg = rs.config()

                rs_cfg.enable_device(str(camera_cfg.serial))

                rs_cfg.enable_stream(
                    rs.stream.color,
                    int(camera_cfg.width),
                    int(camera_cfg.height),
                    rs.format.bgr8,
                    int(camera_cfg.fps),
                )

                pipeline.start(rs_cfg)

                for _ in range(15):
                    pipeline.wait_for_frames()

                return pipeline


            self._overhead_pipeline = _start_camera(
                robot_cfg.cameras.overhead
            )

            self._wrist_pipeline = _start_camera(
                robot_cfg.cameras.wrist
            )

                # Discard auto-exposure warm-up frames.
                for _ in range(15):
                    pipeline.wait_for_frames()

        self._dry_run_overhead_image = None
        self._dry_run_wrist_image = None


        if self.dry_run:
            overhead_path = str(
                self.cfg.dry_run_image_path or ""
            )
            wrist_path = str(
                self.cfg.dry_run_wrist_image_path or ""
            )

            # 俯瞰画像
            if overhead_path:
                overhead_bgr = cv2.imread(
                    overhead_path,
                    cv2.IMREAD_COLOR,
                )

                if overhead_bgr is None:
                    raise FileNotFoundError(
                        f"Could not read overhead dry-run image: "
                        f"{overhead_path}"
                    )

                overhead_rgb = cv2.cvtColor(
                    overhead_bgr,
                    cv2.COLOR_BGR2RGB,
                )

                overhead_width = int(
                    self.cfg.cameras.overhead.width
                )
                overhead_height = int(
                    self.cfg.cameras.overhead.height
                )

                self._dry_run_overhead_image = cv2.resize(
                    overhead_rgb,
                    (overhead_width, overhead_height),
                )

            # 手先画像
            if wrist_path:
                wrist_bgr = cv2.imread(
                    wrist_path,
                    cv2.IMREAD_COLOR,
                )

                if wrist_bgr is None:
                    raise FileNotFoundError(
                        f"Could not read wrist dry-run image: "
                        f"{wrist_path}"
                    )

                wrist_rgb = cv2.cvtColor(
                    wrist_bgr,
                    cv2.COLOR_BGR2RGB,
                )

                wrist_width = int(
                    self.cfg.cameras.wrist.width
                )
                wrist_height = int(
                    self.cfg.cameras.wrist.height
                )

                self._dry_run_wrist_image = cv2.resize(
                    wrist_rgb,
                    (wrist_width, wrist_height),
                )



        self._last_target_qpos = np.full(
            7,
            np.nan,
            dtype=np.float32,
        )

        self._last_safe_qpos = np.full(
            7,
            np.nan,
            dtype=np.float32,
        )




        if plan_cfg.action_space == "joint":
            self.action_space = gym.spaces.Box(
                low=np.full(7, -np.pi, dtype=np.float32),
                high=np.full(7, np.pi, dtype=np.float32),
                dtype=np.float32,
            )





    def close(self):
        if self._overhead_pipeline is not None:
            self._overhead_pipeline.stop()

        if self._wrist_pipeline is not None:
            self._wrist_pipeline.stop()

        if self._robot is not None:
            self._robot.set_state(state=4)
            self._robot.disconnect()

        if self._follower is not None:
            # XArmFollower stops its background control thread and disconnects
            # the XArmAPI object it owns.
            self._follower.disconnect()
            self._follower = None
            self._robot = None


    def _get_image_from_pipeline(self, pipeline):
        frames = pipeline.wait_for_frames(
            timeout_ms=3000
        )

        frame = frames.get_color_frame()

        if not frame:
            raise RuntimeError(
                "RealSense did not return a color frame"
            )

        return cv2.cvtColor(
            np.asanyarray(frame.get_data()),
            cv2.COLOR_BGR2RGB,
        )


    def get_images(self):

        if self.dry_run:
            return (
                self._dry_run_overhead_image.copy(),
                self._dry_run_wrist_image.copy(),
            )
            
        overhead = self._get_image_from_pipeline(
            self._overhead_pipeline
        )

        wrist = self._get_image_from_pipeline(
            self._wrist_pipeline
        )

        return overhead, wrist



    @staticmethod
    def _sdk_value(result, name):
        code, value = result
        if code != 0:
            raise RuntimeError(
                f"xArm {name} failed with SDK code {code}"
            )
        return np.asarray(value, dtype=np.float32)

    def get_robot_state(self):
        if self.dry_run:
            return self._last_qpos, self._last_qvel, self._last_ee

        if self._follower is None or self._robot is None:
            raise RuntimeError("Robopy XArmFollower is not connected")

        # Use Robopy's cached follower state for qpos / EE / gripper so the
        # observation path is consistent with the low-level controller.
        follower_state = np.asarray(
            self._follower.get_joint_state(),
            dtype=np.float32,
        )
        qpos = follower_state[:7].copy()
        self._last_gripper = np.float32(follower_state[7])

        ee = np.asarray(
            self._follower.get_ee_pos_quat(),
            dtype=np.float32,
        ).copy()

        # XArmFollower does not expose qvel, so retain the SDK query for it.
        try:
            code, joint_states = self._robot.get_joint_states(
                is_radian=True
            )
            if code != 0:
                raise RuntimeError(
                    f"xArm get_joint_states failed with SDK code {code}"
                )
            qvel = np.asarray(
                joint_states[1],
                dtype=np.float32,
            )[:7]
        except (
            AttributeError,
            IndexError,
            TypeError,
            RuntimeError,
        ):
            qvel = np.zeros(7, dtype=np.float32)
        ee_rpy = self._sdk_value(
            self._robot.get_position(is_radian=True), "get_position"
        )[:6]

        quat = Rotation.from_euler(
            "xyz",
            ee_rpy[3:6],
        ).as_quat().astype(np.float32)

        quat /= np.clip(
            np.linalg.norm(quat),
            1e-8,
            None,
        )

        previous_quat = self._last_ee[3:7]

        if np.dot(
            previous_quat,
            quat,
        ) < 0:
            quat *= -1.0

        ee = np.concatenate(
            [
                ee_rpy[:3] / 1000.0,
                quat,
            ]
        ).astype(np.float32)

        try:
            code, gripper_position = self._robot.get_gripper_position()
            if code == 0:
                open_pos = float(self.cfg.gripper.open_position)
                closed_pos = float(self.cfg.gripper.closed_position)
                denominator = closed_pos - open_pos
                if abs(denominator) > 1e-6:
                    self._last_gripper = np.float32(np.clip(
                        (float(gripper_position) - open_pos) / denominator,
                        0.0,
                        1.0,
                    ))
        except (AttributeError, TypeError, ValueError):
            pass
        self._last_qpos, self._last_qvel, self._last_ee = qpos, qvel, ee
        return qpos, qvel, ee

    def execute(self, action, action_space):
        """Execute one policy target through Robopy's XArmFollower.

        The outer per-joint ``max_joint_delta_rad`` clip used by the previous
        XArmAPI/set_servo_angle implementation is intentionally not applied
        here. Robopy's XArmFollower performs its own norm-based ``max_delta``
        smoothing at ``control_frequency`` in the background thread.
        """
        action = np.asarray(
            action,
            dtype=np.float32,
        ).reshape(-1)

        if action_space == "joint":
            if action.size < 7:
                raise ValueError(
                    f"joint action needs 7 values, got {action.size}"
                )

            target_qpos = action[:7].copy()
            command_qpos = target_qpos.copy()

            self._last_target_qpos = target_qpos.copy()
            self._last_command_qpos = command_qpos.copy()
            self._last_safe_qpos = command_qpos.copy()

            if not self.dry_run:
                if self._follower is None:
                    raise RuntimeError(
                        "Robopy XArmFollower is not connected"
                    )
                self._follower.command_joint_state(
                    command_qpos.astype(np.float32)
                )

            return command_qpos.astype(np.float32)

        if action.size < 8:
            raise ValueError(
                "flip-mug Cartesian action must be "
                "[x,y,z,qx,qy,qz,qw,gripper] (8 values), "
                f"got {action.size}"
            )

        _, _, current_ee = self.get_robot_state()
        clipped_action = self.clip_cartesian_action(
            action,
            current_ee=current_ee,
        )

        target_xyz = clipped_action[:3]
        target_quat = clipped_action[3:7]
        target_rotation = Rotation.from_quat(target_quat)
        target_rpy = target_rotation.as_euler("xyz")
        pose = np.concatenate([
            target_xyz * 1000.0,
            target_rpy,
        ])
        target_gripper = float(clipped_action[7])

        if not self.dry_run:
            current_qpos, _, _ = self.get_robot_state()

            target_qpos = self._ik_solver.solve(
                pose_rpy=pose,
                q_pre=current_qpos,
            )

            # Keep the existing IK/FK consistency checks.
            fk_target_sdk = self.forward_kinematics(
                target_qpos
            )
            fk_target_local = self._fk_solver.solve(
                target_qpos
            )

            position_error_m = np.linalg.norm(
                target_xyz - fk_target_sdk[:3]
            )

            rotation_target = Rotation.from_quat(
                target_quat
            )
            rotation_fk = Rotation.from_quat(
                fk_target_sdk[3:7]
            )
            relative_rotation = (
                rotation_fk * rotation_target.inv()
            )
            orientation_error_deg = np.rad2deg(
                np.linalg.norm(
                    relative_rotation.as_rotvec()
                )
            )

            # Robopy performs the low-level joint smoothing. Do not apply the
            # old outer max_joint_delta_rad clip here, otherwise the command
            # would be limited twice.
            command_qpos = target_qpos.copy()

            fk_sdk = self.forward_kinematics(
                command_qpos
            )
            fk_local = self._fk_solver.solve(
                command_qpos
            )

            position_error_m = np.linalg.norm(
                fk_sdk[:3] - fk_local[:3]
            )

            rotation_sdk = Rotation.from_quat(
                fk_sdk[3:7]
            )
            rotation_local = Rotation.from_quat(
                fk_local[3:7]
            )
            relative_rotation = (
                rotation_local * rotation_sdk.inv()
            )
            orientation_error_rad = np.linalg.norm(
                relative_rotation.as_rotvec()
            )
            orientation_error_deg = np.rad2deg(
                orientation_error_rad
            )

            self._last_target_qpos = target_qpos.copy()
            self._last_command_qpos = command_qpos.copy()
            self._last_safe_qpos = command_qpos.copy()

            if self._follower is None:
                raise RuntimeError(
                    "Robopy XArmFollower is not connected"
                )

            # Passing 8 values lets the same Robopy background controller
            # update both the joint target and gripper target.
            follower_action = np.concatenate([
                command_qpos,
                np.asarray(
                    [target_gripper],
                    dtype=np.float32,
                ),
            ]).astype(np.float32)

            self._follower.command_joint_state(
                follower_action
            )

        self._last_gripper = np.float32(
            target_gripper
        )

        return np.concatenate([
            target_xyz,
            target_quat,
            [target_gripper],
        ]).astype(np.float32)

    def clip_cartesian_action(self, action, current_ee=None):
        """
        Cartesian actionを安全制限後の値へ変換する。

        Args:
            action:
                shape (8,)
                [x, y, z, qx, qy, qz, qw, gripper]

            current_ee:
                shape (7,)
                [x, y, z, qx, qy, qz, qw]
                Noneなら実機またはdry-run状態から取得する。

        Returns:
            clipped_action:
                shape (8,)
                [x, y, z, qx, qy, qz, qw, gripper]
        """

        
        action = np.asarray(action, dtype=np.float32).reshape(-1)

        if action.size < 8:
            raise ValueError(
                "Cartesian action must have 8 values, "
                f"got {action.size}"
            )

        if current_ee is None:
            _, _, current_ee = self.get_robot_state()

        current_ee = np.asarray(current_ee, dtype=np.float32)

        # Position clip
        target_xyz = action[:3].copy()
        current_xyz = current_ee[:3]

        delta_xyz = np.clip(
            target_xyz - current_xyz,
            -float(self.cfg.max_cartesian_delta_m),
            float(self.cfg.max_cartesian_delta_m),
        )

        target_xyz = current_xyz + delta_xyz

        bounds = np.asarray(
            self.cfg.workspace_bounds_m,
            dtype=np.float32,
        )
        target_xyz = np.clip(
            target_xyz,
            bounds[:, 0],
            bounds[:, 1],
        )

        # Orientation clip
        current_rotation = Rotation.from_quat(current_ee[3:7])

        target_quat = action[3:7]
        quat_norm = float(np.linalg.norm(target_quat))

        if quat_norm < 1e-6:
            raise ValueError(
                "Predicted quaternion has near-zero norm"
            )

        target_rotation = Rotation.from_quat(
            target_quat / quat_norm
        )

        relative = target_rotation * current_rotation.inv()
        rotation_vector = relative.as_rotvec()
        angle = float(np.linalg.norm(rotation_vector))

        max_angle = float(self.cfg.max_orientation_delta_rad)

        if angle > max_angle:
            relative = Rotation.from_rotvec(
                rotation_vector * (max_angle / angle)
            )
            target_rotation = relative * current_rotation

        target_quat = target_rotation.as_quat().astype(np.float32)

        # Gripper clip
        target_gripper = float(np.clip(action[7], 0.0, 1.0))
        target_gripper = float(np.clip(
            target_gripper,
            float(self._last_gripper)
            - float(self.cfg.gripper.max_delta),
            float(self._last_gripper)
            + float(self.cfg.gripper.max_delta),
        ))

        return np.concatenate([
            target_xyz,
            target_quat,
            [target_gripper],
        ]).astype(np.float32)
        
    def forward_kinematics(self, qpos):
        """
        xArm7 joint angles -> EE pose

        Args:
            qpos:
                shape (7,)
                joint angles [rad]

        Returns:
            ee:
                shape (7,)
                [x, y, z, qx, qy, qz, qw]
                position unit: metre
        """
        qpos = np.asarray(
            qpos,
            dtype=np.float32,
        ).reshape(-1)

        if qpos.size < 7:
            raise ValueError(
                f"qpos needs 7 values, got {qpos.size}"
            )

        if self.dry_run:
            raise RuntimeError(
                "forward_kinematics requires a real xArm connection"
            )

        code, fk_pose = self._robot.get_forward_kinematics(
            qpos[:7].tolist(),
            input_is_radian=True,
            return_is_radian=True,
        )

        if code != 0 or fk_pose is None:
            raise RuntimeError(
                "xArm get_forward_kinematics failed\n"
                f"code: {code}\n"
                f"qpos: {qpos[:7]}"
            )

        fk_pose = np.asarray(
            fk_pose,
            dtype=np.float64,
        )

        if fk_pose.shape != (6,):
            raise ValueError(
                f"FK pose must have shape (6,), got {fk_pose.shape}"
            )

        # xArm SDK:
        # [x_mm, y_mm, z_mm, roll, pitch, yaw]
        position_m = fk_pose[:3] / 1000.0

        quaternion = Rotation.from_euler(
            "xyz",
            fk_pose[3:6],
            degrees=False,
        ).as_quat()

        ee = np.concatenate(
            [
                position_m,
                quaternion,
            ]
        ).astype(np.float32)

        return ee


class XArm7IK:
    def __init__(
        self,
        lib_path: str,
        tcp_offset=None,
        world_offset=None,
    ):
        self.lib = ctypes.CDLL(
            str(Path(lib_path).resolve())
        )

        double_ptr = ctypes.POINTER(
            ctypes.c_double
        )

        self.lib.xarm7_init.argtypes = [
            double_ptr,
            double_ptr,
            double_ptr,
            double_ptr,
        ]
        self.lib.xarm7_init.restype = ctypes.c_int

        self.lib.xarm7_ik.argtypes = [
            double_ptr,
            double_ptr,
            double_ptr,
        ]
        self.lib.xarm7_ik.restype = ctypes.c_int

        # init呼び出し中に配列が生存するよう、
        # インスタンス属性として保持
        self._tcp_offset = self._prepare_offset(
            tcp_offset
        )
        self._world_offset = self._prepare_offset(
            world_offset
        )

        tcp_ptr = self._as_pointer(
            self._tcp_offset
        )
        world_ptr = self._as_pointer(
            self._world_offset
        )

        code = self.lib.xarm7_init(
            None,
            None,
            tcp_ptr,
            world_ptr,
        )

        if code != 0:
            raise RuntimeError(
                f"xarm7_init failed: {code}"
            )

        print(
            "IK tcp_offset:",
            self._tcp_offset,
        )
        print(
            "IK world_offset:",
            self._world_offset,
        )

    @staticmethod
    def _prepare_offset(offset):
        if offset is None:
            return None

        offset = np.asarray(
            offset,
            dtype=np.float64,
        ).reshape(-1)

        if offset.size < 6:
            raise ValueError(
                "Offset must contain 6 values: "
                "[x_mm, y_mm, z_mm, "
                "roll, pitch, yaw]"
            )

        return np.ascontiguousarray(
            offset[:6],
            dtype=np.float64,
        )

    @staticmethod
    def _as_pointer(offset):
        if offset is None:
            return None

        return offset.ctypes.data_as(
            ctypes.POINTER(ctypes.c_double)
        )

    def solve(
        self,
        pose_rpy: np.ndarray,
        q_pre: np.ndarray,
    ) -> np.ndarray:
        pose = np.ascontiguousarray(pose_rpy, dtype=np.float64)
        seed = np.ascontiguousarray(q_pre, dtype=np.float64)
        theta = np.empty(7, dtype=np.float64)

        code = self.lib.xarm7_ik(
            pose.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            seed.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            theta.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )

        if code != 0:
            raise RuntimeError(f"xarm7_ik failed: {code}")

        return theta.astype(np.float32)



class XArm7FK:
    def __init__(
        self,
        lib_path: str,
        tcp_offset=None,
        world_offset=None,
    ):
        self.lib = ctypes.CDLL(
            str(Path(lib_path).resolve())
        )

        double_ptr = ctypes.POINTER(
            ctypes.c_double
        )

        # xarm7_init(
        #     q_max,
        #     q_min,
        #     tcp_offset,
        #     world_offset,
        # )
        self.lib.xarm7_init.argtypes = [
            double_ptr,
            double_ptr,
            double_ptr,
            double_ptr,
        ]
        self.lib.xarm7_init.restype = ctypes.c_int

        # xarm7_fk(
        #     theta,
        #     pose,
        # )
        self.lib.xarm7_fk.argtypes = [
            double_ptr,
            double_ptr,
        ]
        self.lib.xarm7_fk.restype = ctypes.c_int

        self._tcp_offset = self._prepare_offset(
            tcp_offset
        )
        self._world_offset = self._prepare_offset(
            world_offset
        )

        tcp_ptr = self._as_pointer(
            self._tcp_offset
        )
        world_ptr = self._as_pointer(
            self._world_offset
        )

        code = self.lib.xarm7_init(
            None,
            None,
            tcp_ptr,
            world_ptr,
        )

        if code != 0:
            raise RuntimeError(
                f"xarm7_init failed: {code}"
            )

        print(
            "FK tcp_offset:",
            self._tcp_offset,
        )
        print(
            "FK world_offset:",
            self._world_offset,
        )

    @staticmethod
    def _prepare_offset(offset):
        if offset is None:
            return None

        offset = np.asarray(
            offset,
            dtype=np.float64,
        ).reshape(-1)

        if offset.size < 6:
            raise ValueError(
                "Offset must contain 6 values: "
                "[x_mm, y_mm, z_mm, "
                "roll, pitch, yaw]"
            )

        return np.ascontiguousarray(
            offset[:6],
            dtype=np.float64,
        )

    @staticmethod
    def _as_pointer(offset):
        if offset is None:
            return None

        return offset.ctypes.data_as(
            ctypes.POINTER(ctypes.c_double)
        )

    def solve(
        self,
        qpos: np.ndarray,
    ) -> np.ndarray:
        """
        xArm7 joint angles -> EE pose

        Args:
            qpos:
                shape (7,)
                joint angles [rad]

        Returns:
            ee:
                shape (7,)
                [x, y, z, qx, qy, qz, qw]
                position [m]
        """
        qpos = np.asarray(
            qpos,
            dtype=np.float64,
        ).reshape(-1)

        if qpos.size < 7:
            raise ValueError(
                f"qpos needs 7 values, got {qpos.size}"
            )

        theta = np.ascontiguousarray(
            qpos[:7],
            dtype=np.float64,
        )

        # C library output:
        # [x_mm, y_mm, z_mm, roll, pitch, yaw]
        pose = np.empty(
            6,
            dtype=np.float64,
        )

        code = self.lib.xarm7_fk(
            theta.ctypes.data_as(
                ctypes.POINTER(ctypes.c_double)
            ),
            pose.ctypes.data_as(
                ctypes.POINTER(ctypes.c_double)
            ),
        )

        if code != 0:
            raise RuntimeError(
                "xarm7_fk failed\n"
                f"code: {code}\n"
                f"qpos: {theta}"
            )

        position_m = pose[:3] / 1000.0

        quaternion = Rotation.from_euler(
            "xyz",
            pose[3:6],
            degrees=False,
        ).as_quat()

        return np.concatenate([
            position_m,
            quaternion,
        ]).astype(np.float32)




def _load_or_capture_goal(
    env,
    real_cfg,
):

    goal_path = str(real_cfg.goal_image_path or "")
    goal_wrist_path = str(real_cfg.goal_wrist_image_path or "")

    if goal_path and goal_wrist_path:
        goal_bgr = cv2.imread(
            goal_path,
            cv2.IMREAD_COLOR,
        )
        goal_wrist_bgr = cv2.imread(
            goal_wrist_path,
            cv2.IMREAD_COLOR,
        )

        if goal_bgr is None:
            raise FileNotFoundError(
                f"Could not read goal image: {goal_path}"
            )
            
        if goal_wrist_bgr is None:
            raise FileNotFoundError(
                "Could not read wrist goal image: "
                f"{goal_wrist_path}"
            )

        goal_image = cv2.cvtColor(
            goal_bgr,
            cv2.COLOR_BGR2RGB,
        )
        goal_wrist_image = cv2.cvtColor(
            goal_wrist_bgr,
            cv2.COLOR_BGR2RGB,
        )

        if real_cfg.get(
            "goal_proprio",
            None,
        ) is None:
            raise ValueError(
                "goal_proprio must be specified when "
                "using a saved goal image"
            )

        goal_proprio = np.asarray(
            real_cfg.goal_proprio,
            dtype=np.float32,
        )

        if goal_proprio.shape != (8,):
            raise ValueError(
                "goal_proprio must have shape (8,), "
                f"got {goal_proprio.shape}"
            )
            
        goal_quat = goal_proprio[3:7]

        quat_norm = float(
            np.linalg.norm(goal_quat)
        )

        if quat_norm < 1e-8:
            raise ValueError(
                "goal_proprio contains a zero quaternion"
            )

        goal_proprio[3:7] = (
            goal_quat / quat_norm
        )

        if goal_proprio[6] < 0:
            goal_proprio[3:7] *= -1.0

        return goal_image, goal_wrist_image, goal_proprio

    if not real_cfg.non_interactive:
        input(
            "Place the scene and robot in the GOAL "
            "state, then press Enter to capture it: "
        )

    goal_image, goal_wrist_image = env.get_images()
    _, _, goal_ee = env.get_robot_state()

    goal_proprio = np.concatenate(
        [goal_ee, np.asarray([env._last_gripper], dtype=np.float32,),]
    ).astype(np.float32)



    return (goal_image, goal_wrist_image, goal_proprio)





def _policy_observation(
    image,
    wrist_image,
    goal,
    goal_wrist,
    ee,
    gripper,
    goal_proprio,
    step_idx,
    process,
):
    """Build observation for the proprio-aware world model."""
    current_proprio = np.concatenate(
        [
            np.asarray(ee, dtype=np.float32),
            np.asarray(
                [gripper],
                dtype=np.float32,
            ),
        ]
    ).astype(np.float32)

    goal_proprio = np.asarray(
        goal_proprio,
        dtype=np.float32,
    ).reshape(8)

    obs = {
        "pixels": image[None, None],
        "wrist_pixels": wrist_image[None, None],
        "goal": goal[None, None],
        "goal_wrist_pixels": goal_wrist[None, None],

        # (environment=1, history=1, dim=8)
        "proprio": current_proprio[None, None],
        "goal_proprio": goal_proprio[None, None],

        "step_idx": np.asarray(
            [[step_idx]],
            dtype=np.int64,
        ),
    }

    keep = {"pixels", "wrist_pixels", "goal", "goal_wrist_pixels", "step_idx",} | set(process.keys())

    return {
        key: value for key, value in obs.items() if key in keep
    }


def run_xarm_task(cfg, policy, process, results_path):

    """Run MPC against xArm and persist synchronized observations/actions."""
    real_cfg = cfg.eval.real_robot
    env = XArmInferenceEnv(real_cfg, cfg.plan_config) # <__main__.XArmInferenceEnv object at 0x7f94ec6751b0>
    policy.set_env(env)


    if str(cfg.plan_config.action_space) == "cartesian":
        if real_cfg.use_action_projector: 
            action_projector = XArmActionProjector(
                ik_solver=env._ik_solver,
                fk_solver=env._fk_solver,
                workspace_bounds_m=real_cfg.workspace_bounds_m,
                max_cartesian_delta_m=real_cfg.max_cartesian_delta_m,
                max_orientation_delta_rad=real_cfg.max_orientation_delta_rad,
                max_joint_delta_rad=real_cfg.max_joint_delta_rad,
                max_gripper_delta=real_cfg.gripper.max_delta,
            )
        else:
            action_projector = None

        policy.set_action_projector(action_projector)



    if str(cfg.plan_config.action_space) == "cartesian":
        # WorldModelPolicy currently contains a Push-specific 3-D Cartesian
        # Box. Flip-mug was trained with the 8-D pose+gripper action above,
        # so reconfigure only the solver boundary while retaining the same
        # loaded policy/model and action normalizer.
        
        
        policy.action_space = env.action_space #ここで workspace_bounds_m が入る
        policy.solver.configure(
            n_envs=env.num_envs,
            config=policy.cfg,
            action_processor=policy.action_processor,
            action_space=env.action_space,
        )
    policy.results_path = results_path
    if hasattr(policy, "_action_buffer") and policy._action_buffer is not None:
        policy._action_buffer.clear()
    if hasattr(policy, "_next_init"):
        policy._next_init = None


    # run_dir = Path(real_cfg.output_dir).expanduser() / time.strftime("%Y%m%d_%H%M%S")
    run_dir = results_path
    run_dir.mkdir(parents=True, exist_ok=True)
    stop_requested = False

    def request_stop(_signum, _frame):
        nonlocal stop_requested
        stop_requested = True

    previous_sigint = signal.signal(signal.SIGINT, request_stop)
    records = {key: [] for key in (
        "pixels",
        "wrist_pixels",
        "proprio",
        "commanded_action",
        "target_qpos",
        "safe_qpos",
        "qpos",
        "qvel",
        "ee_pos_quat",
        "gripper",
        "timestamp",
    )}
    try:
        
        goal, goal_wrist, goal_proprio = (
            _load_or_capture_goal(
                env,
                real_cfg,
            )
        )        


        cv2.imwrite(
            str(run_dir / "goal.png"), cv2.cvtColor(goal, cv2.COLOR_RGB2BGR)
        )
        if not real_cfg.non_interactive:
            input("Place the scene in the START state, then press Enter to run: ")

        started = time.monotonic()
        period = 1.0 / float(real_cfg.control_hz)
        for step_idx in range(int(real_cfg.max_steps)):
            if stop_requested:
                break
            tick = time.monotonic()
            image, wrist_image = env.get_images()
            print("overhead:", image.shape, image.dtype)
            print("wrist:", wrist_image.shape, wrist_image.dtype)
            qpos, qvel, ee = env.get_robot_state()

            current_proprio = np.concatenate(
                [
                    ee,
                    np.asarray(
                        [env._last_gripper],
                        dtype=np.float32,
                    ),
                ]
            ).astype(np.float32)

            info = _policy_observation(
                image=image,
                wrist_image=wrist_image,
                goal=goal,
                goal_wrist=goal_wrist,
                ee=ee,
                gripper=float(env._last_gripper),
                goal_proprio=goal_proprio,
                step_idx=step_idx,
                process=process,
            )
            
            # print("pixels:", info["pixels"].shape)
            # print("wrist_pixels:", info["wrist_pixels"].shape)
            projection_state = {
                "qpos": qpos,
                "ee": ee,
                "gripper": env._last_gripper,
            }

            action_result = policy.get_action(
                info,
                projection_state=projection_state,
            )
            if isinstance(action_result, tuple):
                action, outputs = action_result
            else:
                action = action_result
                outputs = None
                
            # print("run_dir:", run_dir)
            if outputs is not None:
                visualize_cem_actions(
                    outputs=outputs,
                    action_processor=policy.action_processor,
                    env=env,
                    current_ee=ee,
                    save_dir=run_dir / "cem" / "cem_actions",
                    step_idx=step_idx,
                    receding_horizon=int(policy.cfg.receding_horizon),
                )
                
            # action = action_result[0] if isinstance(action_result, tuple) else action_result 
            commanded = env.execute(action, str(cfg.plan_config.action_space))

            records["pixels"].append(image)
            records["wrist_pixels"].append(wrist_image)
            
            records["proprio"].append(current_proprio)
            records["commanded_action"].append(commanded)
            records["target_qpos"].append(env._last_target_qpos.copy())
            records["safe_qpos"].append(env._last_safe_qpos.copy())
            records["qpos"].append(qpos)
            records["qvel"].append(qvel)
            records["ee_pos_quat"].append(ee)
            records["gripper"].append(env._last_gripper)
            records["timestamp"].append(time.monotonic() - started)
            remaining = period - (time.monotonic() - tick)
            if remaining > 0:
                time.sleep(remaining)
    finally:
        signal.signal(signal.SIGINT, previous_sigint)
        env.close()

        # rollout.h5 を保存
        with h5py.File(run_dir / "rollout.h5", "w") as h5:
            h5.attrs["config"] = OmegaConf.to_yaml(cfg)

            for key, values in records.items():
                array = np.asarray(values)
                kwargs = {
                    "compression": "gzip",
                    "compression_opts": 4
                } if array.size else {}

                h5.create_dataset(
                    key,
                    data=array,
                    **kwargs
                )

            h5.create_dataset(
                "goal",
                data=goal if "goal" in locals() else np.empty(0)
            )

            h5.create_dataset(
                "goal_wrist_pixels",
                data=goal_wrist,
            )

            h5.create_dataset(
                "goal_proprio",
                data=(
                    goal_proprio
                    if "goal_proprio" in locals()
                    else np.empty(0, dtype=np.float32)
                ),
            )


        print(
            f"Real-robot rollout saved to: "
            f"{run_dir / 'rollout.h5'}"
        )


        # commanded Cartesian position と
        # 次stepで観測された実際のEE positionを比較
        if (
            len(records["commanded_action"]) > 1
            and len(records["ee_pos_quat"]) > 1
        ):
            commanded_actions = np.asarray(
                records["commanded_action"],
                dtype=np.float32,
            )

            ee_states = np.asarray(
                records["ee_pos_quat"],
                dtype=np.float32,
            )



            # commanded_action[t] に対して、
            # その命令後の ee_pos_quat[t+1] を比較する
            commanded_xyz = commanded_actions[:-1, :3]
            actual_xyz = ee_states[1:, :3]

            # 横軸を step にする
            plot_steps = np.arange(1, len(commanded_xyz) + 1)

            fig, axes = plt.subplots(
                3,
                1,
                figsize=(12, 10),
                sharex=True,
            )

            axis_names = ["x", "y", "z"]

            for i, axis_name in enumerate(axis_names):
                axes[i].plot(
                    plot_steps,
                    commanded_xyz[:, i],
                    label=f"Commanded {axis_name}",
                )

                axes[i].plot(
                    plot_steps,
                    actual_xyz[:, i],
                    label=f"Actual {axis_name}",
                )

                axes[i].set_ylabel(
                    f"{axis_name.upper()} Position [m]"
                )

                axes[i].legend()
                axes[i].grid(True)

            axes[2].set_xlabel("Step")

            fig.suptitle(
                "Commanded vs Actual EE Position"
            )

            fig.tight_layout()

            position_plot_path = (
                run_dir / "commanded_vs_actual_position.png"
            )

            fig.savefig(
                position_plot_path,
                dpi=150,
                bbox_inches="tight",
            )

            plt.close(fig)

            print(
                f"Commanded vs actual EE position plot saved to: "
                f"{position_plot_path}"
            )


        # set_servo_angle() に投入した関節角と、
        # 次stepで観測された実測関節角を比較
        if (
            len(records["safe_qpos"]) > 1
            and len(records["qpos"]) > 1
        ):
            safe_qpos_states = np.asarray(
                records["safe_qpos"],
                dtype=np.float32,
            )

            actual_qpos_states = np.asarray(
                records["qpos"],
                dtype=np.float32,
            )

            # safe_qpos[t] の命令後に取得されるのが qpos[t+1]
            commanded_qpos = safe_qpos_states[:-1]
            actual_qpos = actual_qpos_states[1:]

            plot_steps = np.arange(
                1,
                len(commanded_qpos) + 1,
            )

            fig, axes = plt.subplots(
                7,
                1,
                figsize=(12, 18),
                sharex=True,
            )

            for joint_idx in range(7):
                axes[joint_idx].plot(
                    plot_steps,
                    commanded_qpos[:, joint_idx],
                    label=f"Commanded Joint {joint_idx + 1}",
                )

                axes[joint_idx].plot(
                    plot_steps,
                    actual_qpos[:, joint_idx],
                    label=f"Actual Joint {joint_idx + 1}",
                )

                axes[joint_idx].set_ylabel(
                    f"J{joint_idx + 1} [rad]"
                )

                axes[joint_idx].legend()
                axes[joint_idx].grid(True)

            axes[-1].set_xlabel("Step")

            fig.suptitle(
                "Commanded vs Actual Joint Position"
            )

            fig.tight_layout(
                rect=[0, 0, 1, 0.98]
            )

            joint_plot_path = (
                run_dir
                / "commanded_vs_actual_joint_position.png"
            )

            fig.savefig(
                joint_plot_path,
                dpi=150,
                bbox_inches="tight",
            )

            plt.close(fig)

            print(
                "Commanded vs actual joint position plot "
                f"saved to: {joint_plot_path}"
            )


        # pixels を rollout.mp4 として保存
        if len(records["pixels"]) > 0:
            raw_video_path = run_dir / "rollout_raw.mp4"
            video_path = run_dir / "rollout.mp4"

            frames = np.asarray(records["pixels"])
            height, width = frames[0].shape[:2]

            # 一旦 mp4v で保存
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")

            writer = cv2.VideoWriter(
                str(raw_video_path),
                fourcc,
                float(real_cfg.control_hz),
                (width, height),
            )

            if not writer.isOpened():
                raise RuntimeError(
                    f"Could not open video writer: {raw_video_path}"
                )

            for frame in frames:
                frame_bgr = cv2.cvtColor(
                    frame.astype(np.uint8),
                    cv2.COLOR_RGB2BGR,
                )

                writer.write(frame_bgr)

            writer.release()

            # H.264 に変換
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-i", str(raw_video_path),
                    "-c:v", "libx264",
                    "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart",
                    str(video_path),
                ],
                check=True,
            )

            # 中間ファイルを削除
            raw_video_path.unlink()

            print(
                f"Real-robot rollout video saved to: "
                f"{video_path}"
            )
    return run_dir




@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    """Run evaluation of dinowm vs random policy."""

    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"
    


    results_path = (
        Path(swm.data.utils.get_cache_dir(), "eval", cfg.policy).parent
        if cfg.policy != "random"
        else Path(__file__).parent
    ) 


    # create world environment
    # cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    # world = swm.World(**cfg.world, image_shape=(cfg.world.height, cfg.world.width))
    
    

    # create the transform
    transform = {
        "pixels": img_transform(cfg),
        "wrist_pixels": img_transform(cfg),
        "goal": img_transform(cfg),
        "goal_wrist_pixels": img_transform(cfg),
    }


    dataset_name = cfg.eval.dataset_name

    cache_dir = Path(
        cfg.cache_dir
        or swm.data.utils.get_cache_dir()
    ).expanduser()

    dataset_path = (
        cache_dir
        / "datasets"
        / f"{dataset_name}.h5"
    )


    stats_path = Path(
        cfg.eval.normalization_stats_path
    ).expanduser()
    print("stats_path:", stats_path) #/home/shonosukehida/.stable_worldmodel/datasets/flip_mug/ep200_tm300_gripper


    dataset = None

    if dataset_path.is_file():
        print(
            "Training dataset was found. "
            "Computing normalization statistics."
        )
        print(f"dataset_path: {dataset_path}")

        dataset = get_dataset(
            cfg,
            dataset_name,
        )

        process, action_key = (
            build_normalization_process(
                stats_dataset=dataset,
                keys_to_cache=cfg.dataset.keys_to_cache,
            )
        )

        save_normalization_process(
            stats_path=stats_path,
            process=process,
            action_key=action_key,
        )

    elif stats_path.is_file():
        print(
            "Training dataset was not found. "
            "Loading saved normalization statistics."
        )
        print(f"dataset_path: {dataset_path}")

        process, action_key = (
            load_normalization_process(
                stats_path
            )
        )

    else:
        raise FileNotFoundError(
            "Neither the training dataset nor the "
            "normalization statistics file was found.\n"
            f"dataset: {dataset_path}\n"
            f"statistics: {stats_path}"
        )

    #正規化統計が正しいフィールドになっているか検証
    required_process_keys = {
        "action_cartesian",
        "proprio",
        "goal_proprio",
    }

    missing_process_keys = (
        required_process_keys
        - set(process.keys())
    )

    if missing_process_keys:
        raise KeyError(
            "Normalization process is missing keys: "
            f"{sorted(missing_process_keys)}"
        )  
    ##        

    # -- run evaluation
    policy = cfg.get("policy", "random") #flip_mug/ep200_tm300_gripper/lewm

    
    if policy != "random":
        model = swm.policy.AutoCostModel(cfg.policy) #cfg.policy: flip_mug/ep200_tm300_gripper/lewm
        
        if cfg.eval.probing.get("use_random_encoder", False):
            print("Using a randomly reinitialized encoder")
            old_encoder = model.encoder
            device = next(old_encoder.parameters()).device
            dtype = next(old_encoder.parameters()).dtype

            torch.manual_seed(0)

            model.encoder = ViTModel(old_encoder.config)
            model.encoder = model.encoder.to(device=device, dtype=dtype)
            model.encoder.eval()
            print("set random encoder")
            
                
        model = model.to("cuda")
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver, model=model)

        # print("process[action_cartesian]")
        # print("process.mean:", process["action_cartesian"].mean_)
        # print("process.scale:", process["action_cartesian"].scale_)
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform
        )
        

    else:
        policy = swm.policy.RandomPolicy()


    ##実機タスクコード
    if cfg.eval.real_robot.execute:
        if policy == "random" or isinstance(policy, swm.policy.RandomPolicy):
            raise ValueError("Real-robot inference requires a trained policy")
        run_xarm_task(cfg, policy, process, results_path)





    dataset = get_dataset(cfg, cfg.eval.probing.dataset_name)
    val_dataset = get_dataset(cfg, cfg.eval.probing.val_dataset_name)


    if cfg.eval.probing.exe_probe:
        results_path = (
            Path(swm.data.utils.get_cache_dir(), "eval", cfg.policy).parent
        ) #results_path: /home/shonosukehida/.stable_worldmodel/eval/flip_mug/ep200_tm300_gripper
        
        if hasattr(model, "prop_encoder") and model.prop_encoder is not None:  
            prober = ProbingEvaluator(
                dataset,
                model,
                config = cfg.eval.probing, 
                transform = transform,
                process = process,
                results_path = results_path,
                val_dataset = val_dataset,
            )
        else:
            prober = ProbingEvaluator_NoProprio(
                dataset,
                model,
                config = cfg.eval.probing, 
                transform = transform,
                process = process,
                results_path = results_path,
                val_dataset = val_dataset,
            )
            
        
        prober.run()
        
        
def clip_cem_action_sequence(
    env,
    actions_physical,
    current_ee,
):
    """
    CEMの物理空間の行動列に、Cartesian安全制限を逐次適用する。

    Args:
        env:
            XArmInferenceEnv

        actions_physical:
            shape (horizon, 8)
            [x, y, z, qx, qy, qz, qw, gripper]

        current_ee:
            shape (7,)
            現在の[x, y, z, qx, qy, qz, qw]

    Returns:
        clipped_actions:
            shape (horizon, 8)
    """
    actions_physical = np.asarray(
        actions_physical,
        dtype=np.float32,
    )

    if actions_physical.ndim != 2 or actions_physical.shape[1] != 8:
        raise ValueError(
            "actions_physical must have shape "
            f"(horizon, 8), got {actions_physical.shape}"
        )

    simulated_ee = np.asarray(
        current_ee,
        dtype=np.float32,
    ).reshape(7).copy()

    clipped_actions = []

    for action in actions_physical:
        clipped_action = env.clip_cartesian_action(
            action,
            current_ee=simulated_ee,
        )

        clipped_actions.append(clipped_action)

        # 前stepのclip後EE姿勢を、
        # 次stepの仮想的な現在EE姿勢として使う
        simulated_ee = clipped_action[:7].copy()

    return np.stack(clipped_actions, axis=0)




def visualize_cem_actions(
    outputs,
    action_processor,
    env,
    current_ee,
    save_dir,
    step_idx,
    receding_horizon,
):
    """
    CEMが最終的に選んだ全horizonの行動列を可視化する。

    outputs["actions"]:
        shape (num_envs, horizon, action_dim)
        CEMの正規化空間での出力
    """
    if outputs is None:
        return

    if "actions" not in outputs:
        raise KeyError("CEM outputs does not contain 'actions'")

    actions = outputs["actions"]

    if torch.is_tensor(actions):
        actions = actions.detach().cpu().numpy()
    else:
        actions = np.asarray(actions)

    if actions.ndim != 3:
        raise ValueError(
            "CEM actions must have shape "
            f"(num_envs, horizon, action_dim), got {actions.shape}"
        )

    # 今回は実機1台なのでenv_idx=0
    actions_normalized = actions[0]

    # 逆正規化して物理空間へ戻す
    actions_physical = action_processor.inverse_transform(
        actions_normalized
    )
    actions_clipped = clip_cem_action_sequence(
        env=env,
        actions_physical=actions_physical,
        current_ee=current_ee,
    )

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    np.savez(
        save_dir / f"step_{step_idx:04d}_cem_actions.npz",
        normalized=actions_normalized,
        physical=actions_physical,
        clipped=actions_clipped,
        current_ee=np.asarray(current_ee, dtype=np.float32),
        receding_horizon=np.int64(receding_horizon),
    )

    _plot_cem_action_sequence(
        actions=actions_normalized,
        save_path=save_dir
        / f"step_{step_idx:04d}_normalized.png",
        title=f"CEM normalized actions — step {step_idx}",
        receding_horizon=receding_horizon,
    )

    _plot_cem_action_sequence(
        actions=actions_physical,
        save_path=save_dir
        / f"step_{step_idx:04d}_physical.png",
        title=f"CEM physical actions — step {step_idx}",
        receding_horizon=receding_horizon,
    )
    
    _plot_cem_action_sequence(
        actions=actions_clipped,
        save_path=save_dir
        / f"step_{step_idx:04d}_clipped.png",
        title=f"CEM clipped actions — step {step_idx}",
        receding_horizon=receding_horizon,
    )

def _plot_cem_action_sequence(
    actions,
    save_path,
    title,
    receding_horizon,
):
    """
    actions:
        shape (horizon, 8)
        [x, y, z, qx, qy, qz, qw, gripper]
    """
    actions = np.asarray(actions)

    if actions.ndim != 2 or actions.shape[1] != 8:
        raise ValueError(
            f"Expected actions shape (horizon, 8), got {actions.shape}"
        )

    horizon = actions.shape[0]
    steps = np.arange(horizon)

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(10, 10),
        sharex=True,
    )

    # 位置
    axes[0].plot(steps, actions[:, 0], marker="o", label="x")
    axes[0].plot(steps, actions[:, 1], marker="o", label="y")
    axes[0].plot(steps, actions[:, 2], marker="o", label="z")
    axes[0].set_ylabel("Position [m]")
    axes[0].legend()
    axes[0].grid(True)

    # Quaternion
    axes[1].plot(steps, actions[:, 3], marker="o", label="qx")
    axes[1].plot(steps, actions[:, 4], marker="o", label="qy")
    axes[1].plot(steps, actions[:, 5], marker="o", label="qz")
    axes[1].plot(steps, actions[:, 6], marker="o", label="qw")
    axes[1].set_ylabel("Quaternion")
    axes[1].legend()
    axes[1].grid(True)

    # Gripper
    axes[2].plot(
        steps,
        actions[:, 7],
        marker="o",
        label="gripper",
    )
    axes[2].set_ylabel("Gripper")
    axes[2].set_xlabel("Planning step")
    axes[2].legend()
    axes[2].grid(True)

    # 実際に採用されるreceding horizonの境界
    if 0 < receding_horizon < horizon:
        boundary = receding_horizon - 0.5

        for ax in axes:
            ax.axvline(
                boundary,
                linestyle="--",
                label="receding horizon",
            )

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def print_normalization_process(process, title="normalization process"):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    for key, processor in process.items():
        print(f"\n[key] {key}")
        print(f"  object id    : {id(processor)}")
        print(f"  eps          : {processor.eps}")
        print(f"  mean_        : {processor.mean_}")
        print(f"  scale_       : {processor.scale_}")
        print(f"  raw_min_     : {processor.raw_min_}")
        print(f"  raw_max_     : {processor.raw_max_}")
        print(f"  normed_min_  : {processor.normed_min_}")
        print(f"  normed_max_  : {processor.normed_max_}")

    print("\nprocess keys:")
    print(list(process.keys()))
    print("=" * 80 + "\n")

if __name__ == "__main__":
    run()
