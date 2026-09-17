import time

import numpy as np

from robopy.config.robot_config.xarm_config import XArmConfig
from robopy.robots.xarm.xarm_follower import XArmFollower


ROBOT_IP = "192.168.1.240"


def mm_to_normalized(pos_mm: float, open_mm: float, close_mm: float) -> float:
    """
    Convert Gripper G2 position [mm] to robopy normalized gripper command.

    0.0 -> open
    1.0 -> close
    """
    return (pos_mm - open_mm) / (close_mm - open_mm)


def main() -> None:
    config = XArmConfig(
        follower_ip=ROBOT_IP,
        gripper_open=84.0,
        gripper_close=0.0,
        gripper_speed=50,
        gripper_force=30,
    )

    follower = XArmFollower(config)

    try:
        print("Connecting to xArm via robopy...")
        follower.connect()
        time.sleep(1.0)

        # 現在の状態を取得
        state = follower.get_joint_state()

        print("Initial state:")
        print(f"  joints: {state[:7]}")
        print(f"  gripper(normalized): {state[7]:.3f}")

        # ------------------------------------------------------------
        # Open: 84 mm -> normalized 0.0
        # ------------------------------------------------------------
        print("\nOpening gripper...")

        action = follower.get_joint_state().copy()
        action[7] = 0.0

        follower.command_joint_state(action)
        time.sleep(5.0)

        state = follower.get_joint_state()

        print(f"  gripper(normalized): {state[7]:.3f}")
        print(
            "  estimated position:"
            f" {config.gripper_open + state[7] * (config.gripper_close - config.gripper_open):.2f} mm"
        )

        # ------------------------------------------------------------
        # Close to 20 mm
        # ------------------------------------------------------------
        target_mm = 20.0
        target_normalized = mm_to_normalized(
            target_mm,
            config.gripper_open,
            config.gripper_close,
        )

        print(
            f"\nClosing gripper to {target_mm:.1f} mm "
            f"(normalized={target_normalized:.3f})..."
        )

        action = follower.get_joint_state().copy()
        action[7] = target_normalized

        follower.command_joint_state(action)
        time.sleep(5.0)

        state = follower.get_joint_state()

        print(f"  gripper(normalized): {state[7]:.3f}")
        print(
            "  estimated position:"
            f" {config.gripper_open + state[7] * (config.gripper_close - config.gripper_open):.2f} mm"
        )

        # ------------------------------------------------------------
        # Open again
        # ------------------------------------------------------------
        print("\nOpening gripper again...")

        action = follower.get_joint_state().copy()
        action[7] = 0.0

        follower.command_joint_state(action)
        time.sleep(5.0)

        state = follower.get_joint_state()

        print(f"  gripper(normalized): {state[7]:.3f}")
        print(
            "  estimated position:"
            f" {config.gripper_open + state[7] * (config.gripper_close - config.gripper_open):.2f} mm"
        )

        print("\nrobopy Gripper G2 test completed successfully.")

    finally:
        print("Disconnecting...")
        follower.disconnect()


if __name__ == "__main__":
    main()