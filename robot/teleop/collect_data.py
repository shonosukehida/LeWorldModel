"""xArm7 + GELLO teleoperation and data collection."""

from dataclasses import asdict
from logging import INFO, getLogger
import logging
from pathlib import Path

import numpy as np
import yaml
from box import Box

from robopy.config.robot_config import (
    XArmConfig,
    XArmWorkspaceBounds,
    XArmSensorParams,
)
from robopy.config.sensor_config.params_config import CameraParams
import re

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(message)s"
)

logger = getLogger(__name__)
logger.setLevel(INFO)

print(logger.handlers)
print(logger.propagate)
print(logger.parent)


def load_robot_config() -> Box:
    config_path = (
        Path(__file__).resolve().parents[2]
        / "config"
        / "robot"
        / "collect_data.yaml"
    )

    with open(config_path, "r", encoding="utf-8") as f:
        return Box(yaml.safe_load(f))


def get_next_episode_index(save_dir: Path) -> int:
    """保存先にある episode_N.h5 の最大番号を調べ、次の番号を返す。"""
    episode_pattern = re.compile(r"episode_(\d+)\.h5")
    indices: list[int] = []

    for file_path in save_dir.glob("episode_*.h5"):
        match = episode_pattern.fullmatch(file_path.name)

        if match is not None:
            indices.append(int(match.group(1)))

    if not indices:
        return 0

    return max(indices) + 1


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

    # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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

    # save_path = save_dir / f"episode_{timestamp}.h5"
    episode_index = get_next_episode_index(save_dir)
    save_path = save_dir / f"episode_{episode_index}.h5"
    
    logger.info("Next episode index: %d", episode_index)
    logger.info("Save path: %s", save_path)

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
        
        save = input("do you save the episode? [Y/N]")
        if (save == "Y" or save == "yes" or save == "y"):
            H5Handler.save_hierarchical(
                data_dict=data,
                file_path=str(save_path),
                compress=True,
            )

            logger.info("Dataset saved: %s", save_path)
        else:
            logger.info("Episode discarded.")

    except KeyboardInterrupt:
        logger.info("Recording interrupted by user.")

    finally:
        robot.disconnect()


if __name__ == "__main__":
    xarm_collect()