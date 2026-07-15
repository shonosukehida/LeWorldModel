"""Minimal xArm7 + GELLO teleoperation example.

Usage:
    uv run python examples/robot/xarm_teleoperate.py

Prerequisites:
    * UFactory xArm7 reachable at ``192.168.1.240`` (override via
      ``XArmConfig.follower_ip``).
    * GELLO Dynamixel controller connected via USB — the port is auto-detected
      from ``/dev/serial/by-id/*`` when ``leader_port`` is left as ``None``.

Simulator (recommended before running on the real robot):
    Launch UFactory Studio in simulation mode, select xArm7 as the virtual
    robot, then change ``follower_ip`` below to the simulator address
    (typically ``127.0.0.1``). No other change is required — the GELLO leader
    still connects to real USB hardware. See ``docs/robots/xarm.md`` for a
    full setup walk-through and the pre-flight checklist for switching back
    to the real robot.
"""

from logging import INFO, getLogger
from pathlib import Path

import numpy as np
import yaml
from box import Box

logger = getLogger(__name__)
logger.setLevel(INFO)

def load_robot_config() -> Box:
    config_path = Path(__file__).resolve().parents[2] / "config" / "robot" / "teleop.yaml"

    with open(config_path, "r") as f:
        return Box(yaml.safe_load(f))



def xarm_teleoperate() -> None:
    from robopy.config.robot_config import XArmConfig, XArmWorkspaceBounds
    from robopy.robots.xarm import XArmRobot
    
    cfg = load_robot_config()
    
    print("cfg.robot.follower_ip:", cfg.robot.follower_ip)
    print("cfg .robot.leader_port:", cfg .robot.leader_port)
    print("cfg.robot.start_joints:", cfg.robot.start_joints)

    config = XArmConfig(
        follower_ip=cfg.robot.follower_ip, #cfg.robot.follower_ip
        leader_port=cfg .robot.leader_port,  # auto-detect #cfg.robot.leader_port
        workspace_bounds=XArmWorkspaceBounds(),
        start_joints=np.deg2rad(cfg.robot.start_joints).astype(np.float32), #[0, -90, 90, -90, -90, 0, 0] #cfg.robot.start_joints
    )
    robot = XArmRobot(config)
    try:
        robot.connect()
        robot.teleoperation(max_seconds=cfg.max_seconds)
    except KeyboardInterrupt:
        logger.info("Teleoperation interrupted by user.")
    finally:
        robot.disconnect()


if __name__ == "__main__":
    xarm_teleoperate()