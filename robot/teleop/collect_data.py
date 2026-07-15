"""xArm7 + GELLO teleoperation and data collection."""

from dataclasses import asdict
from logging import INFO, getLogger
from pathlib import Path
from datetime import datetime

import numpy as np
import yaml
from box import Box

from robopy.config.robot_config import (
    XArmConfig,
    XArmWorkspaceBounds,
    XArmSensorParams,
)
from robopy.config.sensor_config.params_config import CameraParams

logger = getLogger(__name__)
logger.setLevel(INFO)


def load_robot_config() -> Box:
    config_path = (
        Path(__file__).resolve().parents[2]
        / "config"
        / "robot"
        / "collect_data.yaml"
    )

    with open(config_path, "r", encoding="utf-8") as f:
        return Box(yaml.safe_load(f))


def xarm_collect() -> None:
    from robopy.config.robot_config import (
        XArmConfig,
        XArmWorkspaceBounds,
    )
    from robopy.robots.xarm import XArmRobot
    from robopy.utils.h5_handler import H5Handler

    cfg = load_robot_config()

    robot_config = XArmConfig(
        follower_ip=cfg.robot.follower_ip,
        leader_port=cfg.robot.leader_port,
        workspace_bounds=XArmWorkspaceBounds(),
        start_joints=np.deg2rad(
            cfg.robot.start_joints
        ).astype(np.float32),
        
        sensors=XArmSensorParams(
            cameras=[
                CameraParams(
                    name=cfg.robot.camera.name,
                    width=cfg.robot.camera.width,
                    height=cfg.robot.camera.height,
                    fps=cfg.robot.camera.fps,
                    index=cfg.robot.camera.index,
                )
            ]
        ),
    )

    robot = XArmRobot(robot_config)

    fps = int(cfg.dataset.fps)
    max_frames = int(cfg.dataset.max_frames)
    teleop_hz = int(cfg.dataset.teleop_hz)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_dir = f"ep{str(cfg.dataset.episode)}_tm{cfg.dataset.max_frames}"
    
    
    #置きたい場所: /home/shonosukehida/.stable_worldmodel/datasets/flip_mug
    save_dir = (
        Path(__file__).resolve().parents[3]
        / ".stable_worldmodel"
        / "datasets"
        / "flip_mug"
        / dataset_dir 
        / "per_episode"
    )
    save_dir.mkdir(parents=True, exist_ok=True)
    print("save_dir:", save_dir)

    save_path = save_dir / f"episode_{timestamp}.h5"

    try:
        logger.info("Connecting robot...")
        robot.connect() #XArm, GELLO, realsense を接続
        logger.info("Robot and sensors connected")

        
        
        input("GELLO と xArm の姿勢を確認し、Enter で収集開始...")

        logger.info(
            "Recording started: fps=%d, max_frames=%d",
            fps,
            max_frames,
        )

        observation = robot.record_parallel(
            max_frame=max_frames,
            fps=fps,
            teleop_hz=teleop_hz,
        )
        
        logger.info("leader shape: %s", observation.arms.leader.shape)
        logger.info("follower shape: %s", observation.arms.follower.shape)
        logger.info("ee shape: %s", observation.arms.ee_pos_quat.shape)

        data = asdict(observation)

        H5Handler.save_hierarchical(
            data_dict=data,
            file_path=str(save_path),
            compress=True,
        )

        logger.info("Dataset saved: %s", save_path)

    except KeyboardInterrupt:
        logger.info("Recording interrupted by user.")

    finally:
        robot.disconnect()


if __name__ == "__main__":
    xarm_collect()