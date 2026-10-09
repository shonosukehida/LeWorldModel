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

from normalization_stats import (
    SafeStandardScaler,
    build_normalization_process,
    save_normalization_process,
    load_normalization_process,
)
from action_projector import XArmActionProjector

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from stable_worldmodel.diffusion import (
    ConditionalUnet1D,
    ResNet18ObsEncoder,
)
from stable_worldmodel.reward import latent_goal_reward

from collections import deque
from utils import get_eval_img_preprocessor


def _reinforced_diffusion_action(policy, observation, ee, state, config, gripper_config):
    """Select a Cartesian command using a monotonic deadline, not step count."""
    now = time.monotonic()
    if state.get("deadline") is not None and now >= state["deadline"]:
        state.clear()
    if not state:
        result = policy.get_action(observation)
        action = result[0] if isinstance(result, tuple) else result
        action = np.asarray(action).reshape(-1)
        if action.size != 8:
            raise ValueError("Gripper reinforcement requires an 8D Cartesian action")
        policy_command = float(action[7])  # Already inverse-transformed by DP.
        if policy_command > float(config.threshold):
            state["pose"] = np.asarray(ee, dtype=np.float32).copy()
            state["deadline"] = time.monotonic() + float(config.duration_sec)
        else:
            return result, False, policy_command
    else:
        policy_command = float("nan")
    command = (float(config.target_opening_mm) - float(gripper_config.open_position)) / (
        float(gripper_config.closed_position) - float(gripper_config.open_position)
    )
    # execute() retains Cartesian/IK safety and gripper.max_delta clipping.
    # Its 8D follower command replaces the persistent background target;
    # Robopy only reissues G2 commands when that target changes. gripper.wait
    # controls SDK waiting in that thread, not this monotonic deadline.
    return np.concatenate([state["pose"], [command]]), True, policy_command


def _reinforced_gpc_action(policy, observation, dp_info, projection_state, ee,
                           state, config, gripper_config):
    """Hold the observed EE pose after GPC selects a physical closing action."""
    values = [float(config.threshold), float(config.target_opening_mm),
              float(config.duration_sec)]
    open_mm = float(gripper_config.open_position)
    closed_mm = float(gripper_config.closed_position)
    if (not np.isfinite(values).all() or not 0 <= values[0] <= 1
            or values[2] <= 0 or not np.isfinite([open_mm, closed_mm]).all()
            or open_mm <= closed_mm or not closed_mm <= values[1] <= open_mm):
        raise ValueError("Invalid GPC gripper reinforcement configuration")
    now = time.monotonic()
    if state.get("deadline") is not None and now >= state["deadline"]:
        state.clear()
    if not state:
        result = policy.get_action(
            observation, dp_info_dict=dp_info, projection_state=projection_state,
        )
        action = np.asarray(result[0] if isinstance(result, tuple) else result)
        if action.shape not in ((8,), (1, 8)) or not np.isfinite(action).all():
            raise ValueError("GPC reinforcement requires a finite 8D Cartesian action")
        policy_command = float(action.reshape(-1)[7])
        if policy_command <= values[0]:
            return result, False, policy_command
        pose = np.asarray(ee, dtype=np.float32)
        if pose.shape != (7,) or not np.isfinite(pose).all():
            raise ValueError("GPC reinforcement requires a finite 7D EE pose")
        state["pose"] = pose.copy()
        state["deadline"] = time.monotonic() + values[2]
    else:
        policy_command = float("nan")
    command = (values[1] - open_mm) / (closed_mm - open_mm)
    return np.concatenate([state["pose"], [command]]), True, policy_command


def plot_commanded_vs_actual_gripper(commanded_action, actual_gripper, save_path):
    """Compare command[t, 7] with pre-command measurement[t+1].

    Missing measurements remain NaN gaps. The final command has no later
    observation. Joint-only actions have no gripper command and are skipped.
    """
    commands = np.asarray(commanded_action, dtype=np.float32)
    actual = np.asarray(actual_gripper, dtype=np.float32)
    if commands.size == 0 or commands.ndim != 2 or commands.shape[1] < 8:
        return
    if actual.ndim != 1 or len(actual) != len(commands):
        raise ValueError("Gripper measurements must match command steps")
    steps = np.arange(1, len(commands))
    fig, ax = plt.subplots(figsize=(12, 4))
    try:
        ax.plot(steps, commands[:-1, 7], marker="o", label="Command (previous step)")
        ax.plot(steps, actual[1:], marker="o", label="Actual (G2 measurement)")
        ax.set_xlabel("Observation step")
        ax.set_ylabel("Gripper (0=open, 1=closed)")
        ax.set_ylim(-0.05, 1.05)
        ax.set_title("Commanded vs Actual Gripper")
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    finally:
        plt.close(fig)


def img_transform(cfg):
    return get_eval_img_preprocessor(cfg.eval.img_size)


def dp_img_transform(cfg):
    """Keep the diffusion policy's existing image preprocessing."""
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


def get_dataset(cfg, dataset_name, num_steps=1):
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
        num_steps=num_steps,
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
        self._last_actual_gripper = np.float32(np.nan)

        # Robopy owns the xArm connection/control thread.
        self._robot_owner = None
        self._follower = None
        self._admittance_enabled = False

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
                    XArmAdmittanceConfig,
                    XArmConfig,
                    XArmWorkspaceBounds,
                    XArmSensorParams,
                )
                from robopy.config.sensor_config.params_config import CameraParams
                from robopy.robots.xarm import XArmRobot
            except ImportError as exc:
                raise ImportError(
                    "Real execution requires robopy with xArm support"
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

            admittance_config = XArmAdmittanceConfig(
                translational_mass=float(robot_cfg.admittance.translational_mass),
                rotational_inertia_mass_ratio=float(
                    robot_cfg.admittance.rotational_inertia_mass_ratio
                ),
                position_stiffness=float(robot_cfg.admittance.position_stiffness),
                orientation_stiffness=float(robot_cfg.admittance.orientation_stiffness),
                damping=tuple(float(value) for value in robot_cfg.admittance.damping),
                reference_frame=int(robot_cfg.admittance.reference_frame),
                compliant_axis=tuple(int(value) for value in robot_cfg.admittance.compliant_axis),
            )

            # Defaults intentionally match Robopy XArmConfig defaults used by
            # the collection script when these fields were not specified.
            follower_cfg = XArmConfig(
                admittance=admittance_config,
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
                gripper_open=float(robot_cfg.gripper.open_position),
                gripper_close=float(robot_cfg.gripper.closed_position),
                gripper_speed=int(robot_cfg.gripper.speed),
                gripper_force=int(robot_cfg.gripper.force),
                gripper_wait=bool(getattr(robot_cfg.gripper, "wait", False)),
            )

            camera_configs = {
                "overhead": robot_cfg.cameras.overhead,
                "wrist": robot_cfg.cameras.wrist,
            }
            follower_cfg.sensors = XArmSensorParams(
                cameras=[
                    CameraParams(
                        name=str(camera.serial),
                        serial_no=str(camera.serial),
                        width=int(camera.width),
                        height=int(camera.height),
                        fps=int(camera.fps),
                        auto_exposure=bool(camera.auto_exposure),
                        exposure=camera.exposure,
                        auto_white_balance=bool(camera.auto_white_balance),
                        white_balance=camera.white_balance,
                    )
                    for camera in camera_configs.values()
                ] if self.use_camera else []
            )
            start_joints = _optional_cfg("start_joints", None)
            if start_joints is not None:
                follower_cfg.start_joints = np.deg2rad(start_joints).astype(np.float32)
            follower_cfg.leader_port = _optional_cfg("leader_port", None)
            self._robot_owner = XArmRobot(follower_cfg)
            try:
                self._robot_owner.connect(
                    connect_leader=bool(_optional_cfg("connect_leader", False))
                )
                self._follower = self._robot_owner.robot_system.follower
                # SDK handle is borrowed from the robot-owned follower.
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


                if self.use_camera:
                    managed_cameras = {
                        camera.name: camera for camera in self._robot_owner.sensors.cameras
                    }
                    self._overhead_pipeline = managed_cameras[str(camera_configs["overhead"].serial)]
                    self._wrist_pipeline = managed_cameras[str(camera_configs["wrist"].serial)]
            except BaseException:
                # Cleanup must not replace the initialization error (including Ctrl-C).
                try:
                    self.close()
                except Exception:
                    pass
                raise

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


    def enable_admittance(self):
        if self.dry_run:
            return

        if self._follower is None:
            raise RuntimeError("XArmFollower is not connected")

        self._follower.enable_admittance_control()
        self._admittance_enabled = True

        # enable_admittance_control() は通常の位置指令を停止するため、
        # 推論による位置指令を再開する。
        self._follower.resume_motion_commands()

        print("Admittance control enabled.")


    def disable_admittance(self):
        if self.dry_run or self._follower is None:
            return

        # 通常の位置指令を停止してから無効化する。
        with self._follower._control_lock:
            self._follower._motion_paused = True

        try:
            self._follower.disable_admittance_control()
        finally:
            self._admittance_enabled = False

        print("Admittance control disabled.")


    def close(self):
        owner = self._robot_owner
        try:
            if self._follower is not None:
                with self._follower._control_lock:
                    self._follower._motion_paused = True
                if self._admittance_enabled:
                    self.disable_admittance()
        finally:
            try:
                if owner is not None:
                    owner.disconnect()
            finally:
                self._robot_owner = None
                self._follower = None
                self._robot = None
                self._admittance_enabled = False
                self._overhead_pipeline = None
                self._wrist_pipeline = None


    def _get_image_from_pipeline(self, pipeline):
        if pipeline is None:
            raise RuntimeError("RealSense camera is not enabled or connected")
        frame_chw = np.asarray(pipeline.read(specific_color="rgb"))
        if frame_chw.ndim != 3 or frame_chw.shape[0] != 3:
            raise RuntimeError(
                f"Unexpected RealSense frame shape: {frame_chw.shape}"
            )
        return np.transpose(frame_chw, (1, 2, 0)).clip(0, 255).astype(np.uint8)


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
        # Measurement only: never reuse an old sample or substitute a command.
        self._last_actual_gripper = np.float32(np.nan)
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
            code, gripper_position = self._robot.get_gripper_g2_position()
            if code == 0:
                open_pos = float(self.cfg.gripper.open_position)
                closed_pos = float(self.cfg.gripper.closed_position)
                denominator = closed_pos - open_pos
                if abs(denominator) > 1e-6:
                    self._last_actual_gripper = np.float32(np.clip(
                        (float(gripper_position) - open_pos) / denominator,
                        0.0,
                        1.0,
                    ))
                    self._last_gripper = self._last_actual_gripper
        except (AttributeError, TypeError, ValueError, RuntimeError, OSError):
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
    env = XArmInferenceEnv(real_cfg, cfg.plan_config)

    try:
        return _run_xarm_task_with_env(cfg, policy, process, results_path, env)
    finally:
        env.close()


def _run_xarm_task_with_env(cfg, policy, process, results_path, env):
    real_cfg = cfg.eval.real_robot
    policy.set_env(env)

    # --------------------------------------------------
    # WorldModelPolicy-specific setup
    # --------------------------------------------------

    if isinstance(policy, swm.policy.WorldModelPolicy):

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

            policy.set_action_projector(
                action_projector
            )

            # WorldModelPolicy / CEM only
            policy.action_space = env.action_space

            policy.solver.configure(
                n_envs=env.num_envs,
                config=policy.cfg,
                action_processor=policy.action_processor,
                action_space=env.action_space,
            )

    # --------------------------------------------------
    # Common setup
    # --------------------------------------------------

    policy.results_path = results_path

    if (
        hasattr(policy, "_action_buffer")
        and policy._action_buffer is not None
    ):
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

    reinforcement_config = getattr(real_cfg, "gripper_reinforcement", None)
    reinforcement_enabled = (
        isinstance(policy, swm.policy.DiffusionPolicy)
        and reinforcement_config is not None
        and bool(reinforcement_config.enabled)
    )
    reinforcement_state = {}

    if reinforcement_enabled:
        if str(cfg.plan_config.action_space) != "cartesian":
            raise ValueError("Gripper reinforcement requires Cartesian control")
        values = [float(reinforcement_config.threshold),
                  float(reinforcement_config.target_opening_mm),
                  float(reinforcement_config.duration_sec)]
        open_mm = float(real_cfg.gripper.open_position)
        closed_mm = float(real_cfg.gripper.closed_position)
        if (not np.isfinite(values).all() or not 0 <= values[0] <= 1
                or values[2] <= 0 or not np.isfinite([open_mm, closed_mm]).all()
                or open_mm <= closed_mm or not closed_mm <= values[1] <= open_mm):
            raise ValueError("Invalid gripper reinforcement configuration")

    gpc_reinforcement_enabled = (
        isinstance(policy, swm.policy.GPCPolicy)
        and reinforcement_config is not None
        and bool(reinforcement_config.enabled)
    )
    gpc_reinforcement_state = {}
    if gpc_reinforcement_enabled and str(cfg.plan_config.action_space) != "cartesian":
        raise ValueError("GPC gripper reinforcement requires Cartesian control")

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
        "actual_gripper",
        "gripper_reinforcement_active",
        "gripper_policy_command",
        "gripper_command",
        "gripper_actual",
        "timestamp",
    )}


    dp_policy = None
    dp_image_history = None
    dp_wrist_image_history = None

    if isinstance(policy, swm.policy.DiffusionPolicy,):
        dp_policy = policy

    elif isinstance(policy, swm.policy.GPCPolicy,):
        dp_policy = policy.diffusion_policy


    if dp_policy is not None:
        dp_image_history = deque(maxlen=dp_policy.obs_horizon)

        dp_wrist_image_history = deque(maxlen=dp_policy.obs_horizon)

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

        # 推論開始前にアドミッタンス制御を有効化
        use_admittance = bool(
            OmegaConf.select(
                cfg,
                "eval.real_robot.admittance.enabled",
                default=False,
            )
        )

        if use_admittance:
            env.enable_admittance()

        started = time.monotonic()
        period = 1.0 / float(real_cfg.control_hz)



        for step_idx in range(int(real_cfg.max_steps)):
            if stop_requested:
                break
            tick = time.monotonic()
            image, wrist_image = env.get_images()

            qpos, qvel, ee = env.get_robot_state()
            # Snapshot before execute, which may read state again.
            actual_gripper = env._last_actual_gripper

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
            
            #Build observation history for DP

            dp_info = None
            
            if dp_policy is not None:

                if len(dp_image_history) == 0:

                    for _ in range(dp_policy.obs_horizon):
                        dp_image_history.append(image.copy())

                        dp_wrist_image_history.append(wrist_image.copy())

                else:
                    dp_image_history.append(image.copy())

                    dp_wrist_image_history.append(wrist_image.copy())

                dp_pixels = np.stack(list(dp_image_history), axis=0,)

                dp_wrist_pixels = np.stack(list(dp_wrist_image_history), axis=0,)

                dp_info = {
                    "pixels": dp_pixels[None],
                    "wrist_pixels": dp_wrist_pixels[None],
                }
                
                
            projection_state = {"qpos": qpos, "ee": ee, "gripper": env._last_gripper,}


            reinforcement_active = False
            policy_gripper_command = float("nan")
            if reinforcement_enabled:
                action_result, reinforcement_active, policy_gripper_command = _reinforced_diffusion_action(
                    policy, dp_info, ee, reinforcement_state, reinforcement_config, real_cfg.gripper
                )
            elif isinstance(policy, swm.policy.DiffusionPolicy,):
                action_result = policy.get_action(dp_info)
                dp_action = action_result[0] if isinstance(action_result, tuple) else action_result
                dp_action = np.asarray(dp_action).reshape(-1)
                if dp_action.size == 8:
                    policy_gripper_command = float(dp_action[7])


            elif gpc_reinforcement_enabled:
                action_result, reinforcement_active, policy_gripper_command = _reinforced_gpc_action(
                    policy, info, dp_info, projection_state, ee,
                    gpc_reinforcement_state, reinforcement_config, real_cfg.gripper,
                )
            elif isinstance(policy, swm.policy.GPCPolicy,):

                action_result = policy.get_action(
                    info,
                    dp_info_dict=dp_info,
                    projection_state=projection_state,
                )

            else:

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
            if isinstance(policy, swm.policy.WorldModelPolicy) and outputs is not None:
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
            if stop_requested:
                break
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
            records["actual_gripper"].append(actual_gripper)
            records["gripper_reinforcement_active"].append(reinforcement_active)
            records["gripper_policy_command"].append(policy_gripper_command)
            records["gripper_command"].append(float(commanded[7]) if len(commanded) == 8 else np.nan)
            records["gripper_actual"].append(actual_gripper)
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

            h5["actual_gripper"].attrs["open_position"] = float(real_cfg.gripper.open_position)
            h5["actual_gripper"].attrs["closed_position"] = float(real_cfg.gripper.closed_position)
            h5["actual_gripper"].attrs["sampling"] = (
                "Normalized: 0=open, 1=closed. Before inference/command at t; compare command[t] with measurement[t+1]."
            )

            for key in ("gripper_actual", "gripper_command", "gripper_policy_command"):
                h5[key].attrs["units"] = "normalized: 0=open, 1=closed"
            h5["gripper_actual"].attrs["sampling"] = h5["actual_gripper"].attrs["sampling"]

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


        plot_commanded_vs_actual_gripper(
            records["commanded_action"], records["actual_gripper"],
            run_dir / "commanded_vs_actual_gripper.png",
        )

        def save_rollout_video(frames, video_name):
            raw_video_path = run_dir / f"{video_name}_raw.mp4"
            video_path = run_dir / f"{video_name}.mp4"

            frames = np.asarray(frames)
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

        # Keep the legacy output and save both camera views explicitly.
        if len(records["pixels"]) > 0:
            save_rollout_video(records["pixels"], "rollout")
            save_rollout_video(records["pixels"], "rollout_overhead")
            save_rollout_video(records["wrist_pixels"], "rollout_wrist")
    return run_dir



def load_diffusion_policy(
    checkpoint_path,
    image_transform,
    device="cuda",
):
    checkpoint_path = Path(checkpoint_path).expanduser()

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    train_cfg = checkpoint["config"]

    # --------------------------------------------------
    # Observation encoder
    # --------------------------------------------------

    obs_encoder = ResNet18ObsEncoder(
        pretrained=False,
    )

    wrist_obs_encoder = ResNet18ObsEncoder(
        pretrained=False,
    )

    # --------------------------------------------------
    # Conditional U-Net
    # --------------------------------------------------

    model = ConditionalUnet1D(
        input_dim=checkpoint["action_dim"],

        global_cond_dim=(
            (
                checkpoint["obs_feature_dim"]
                + checkpoint["wrist_obs_feature_dim"]
            )
            * checkpoint["obs_horizon"]
        ),

        diffusion_step_embed_dim=(
            train_cfg["model"]["diffusion_step_embed_dim"]
        ),

        down_dims=tuple(
            train_cfg["model"]["down_dims"]
        ),

        kernel_size=(
            train_cfg["model"]["kernel_size"]
        ),

        n_groups=(
            train_cfg["model"]["n_groups"]
        ),

        cond_predict_scale=(
            train_cfg["model"]["cond_predict_scale"]
        ),
    )

    # --------------------------------------------------
    # DDPM scheduler
    # --------------------------------------------------

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=(
            checkpoint["num_train_timesteps"]
        ),

        beta_schedule=(
            checkpoint["beta_schedule"]
        ),

        clip_sample=(
            train_cfg["diffusion"]["clip_sample"]
        ),

        prediction_type=(
            checkpoint["prediction_type"]
        ),
    )

    # --------------------------------------------------
    # DP normalization
    # --------------------------------------------------

    action_processor = SafeStandardScaler(
        eps=1e-4
    )

    action_processor.mean_ = (checkpoint["action_mean"].cpu().numpy())
    action_processor.scale_ = (checkpoint["action_std"].cpu().numpy())
    action_key = checkpoint["action_key"]

    dp_process = {action_key: action_processor,}

    # --------------------------------------------------
    # DiffusionPolicy
    # --------------------------------------------------

    diffusion_policy = swm.policy.DiffusionPolicy(
        model=model,

        obs_encoder=obs_encoder,
        wrist_obs_encoder=wrist_obs_encoder,

        noise_scheduler=noise_scheduler,

        pred_horizon=checkpoint["pred_horizon"],
        obs_horizon=checkpoint["obs_horizon"],
        action_horizon=checkpoint["action_horizon"],
        action_dim=checkpoint["action_dim"],

        num_inference_steps=(
            checkpoint["num_inference_steps"]
        ),

        process=dp_process,

        transform={
            "pixels": image_transform,
            "wrist_pixels": image_transform,
        },
    )

    # --------------------------------------------------
    # Load weights
    # --------------------------------------------------

    diffusion_policy.model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    diffusion_policy.obs_encoder.load_state_dict(
        checkpoint["obs_encoder_state_dict"]
    )

    diffusion_policy.wrist_obs_encoder.load_state_dict(
        checkpoint[
            "wrist_obs_encoder_state_dict"
        ]
    )

    diffusion_policy.model.to(device)
    diffusion_policy.obs_encoder.to(device)
    diffusion_policy.wrist_obs_encoder.to(device)

    diffusion_policy.model.eval()
    diffusion_policy.obs_encoder.eval()
    diffusion_policy.wrist_obs_encoder.eval()

    diffusion_policy.model.requires_grad_(False)
    diffusion_policy.obs_encoder.requires_grad_(False)
    diffusion_policy.wrist_obs_encoder.requires_grad_(False)

    return diffusion_policy



@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    """Run evaluation of dinowm vs random policy."""
    


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

    if stats_path.is_file():
        print(
            "Loading saved normalization statistics."
        )
        print(f"dataset_path: {dataset_path}")

        process, action_key = (
            load_normalization_process(
                stats_path
            )
        )

    elif dataset_path.is_file():
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
        policy_type = cfg.get("policy_type", "world_model",)

        model = None
        if policy_type in ("world_model", "gpc"):
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

        if policy_type == "world_model":

            config = swm.PlanConfig(**cfg.plan_config)

            solver = hydra.utils.instantiate(cfg.solver, model=model,)

            policy = swm.policy.WorldModelPolicy(
                solver=solver,
                config=config,
                process=process,
                transform=transform,
            )


        elif policy_type == "diffusion":

            policy = load_diffusion_policy(
                checkpoint_path=cfg.gpc.diffusion_checkpoint,
                image_transform=dp_img_transform(cfg),
                device="cuda",
            )


        elif policy_type == "gpc":

            diffusion_policy = load_diffusion_policy(
                checkpoint_path=cfg.gpc.diffusion_checkpoint,
                image_transform=dp_img_transform(cfg),
                device="cuda",
            )

            policy = swm.policy.GPCPolicy(
                diffusion_policy=diffusion_policy,
                world_model=model,
                reward_fn=latent_goal_reward,
                num_candidates=cfg.gpc.num_candidates,
                process=process,
                transform=transform,
            )

        else:
            raise ValueError(
                f"Unknown policy_type: {policy_type}"
            )




    else:
        policy = swm.policy.RandomPolicy()


    ##実機タスクコード
    if cfg.eval.real_robot.execute:
        if policy == "random" or isinstance(policy, swm.policy.RandomPolicy):
            raise ValueError("Real-robot inference requires a trained policy")
        run_xarm_task(cfg, policy, process, results_path)





    if cfg.eval.probing.exe_probe:
        # One-step JEPA probing needs the complete context plus its target.
        num_steps = 1
        if cfg.eval.probing.plot_open_data and getattr(model, "prop_encoder", None) is not None:
            num_steps = ProbingEvaluator.resolve_history_size(cfg.eval.probing, model) + 1
        dataset = get_dataset(cfg, cfg.eval.probing.dataset_name, num_steps=num_steps)
        val_dataset = get_dataset(cfg, cfg.eval.probing.val_dataset_name, num_steps=num_steps)
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
