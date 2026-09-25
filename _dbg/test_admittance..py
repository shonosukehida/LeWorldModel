"""XArmFollower admittance control: stationary test.

This script does not send Cartesian/joint motion commands and does not
call resume_motion_commands().

CAUTION:
XArmFollower.connect() initializes the robot and opens Gripper G2.
Run only after checking the physical workspace and emergency stop.
"""

import logging
import time

import numpy as np

from robopy.config.robot_config.xarm_config import XArmConfig
from robopy.robots.xarm.xarm_follower import XArmFollower


ROBOT_IP = "192.168.1.240"
TEST_DURATION = 15.0  # seconds
PRINT_INTERVAL = 0.5  # seconds


def print_state(robot: XArmFollower) -> None:
    """Display the latest state cached by the control thread."""

    ee = robot.get_ee_pos_quat()
    joints = robot.get_joint_state()

    xyz_mm = ee[:3] * 1000.0

    print(
        "EE xyz [mm]: "
        f"{np.round(xyz_mm, 2)} | "
        f"gripper: {joints[7]:.3f} | "
        f"motion_paused: {robot._motion_paused} | "
        f"admittance_enabled: {robot._admittance_enabled}"
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    cfg = XArmConfig(follower_ip=ROBOT_IP)
    robot = XArmFollower(cfg)

    connected = False
    admittance_attempted = False

    try:
        print("\n=== Admittance control test ===")
        print("No joint/Cartesian motion commands will be sent by this script.")
        print("WARNING: connect() initializes the arm and opens Gripper G2.")

        answer = input(
            "\nWorkspace and emergency stop checked? "
            "Type YES to connect: "
        )

        if answer != "YES":
            print("Cancelled before connecting.")
            return

        # 1. Connect
        print("\n[1] Connecting...")
        robot.connect()
        connected = True

        time.sleep(1.0)

        if robot._thread is None or not robot._thread.is_alive():
            raise RuntimeError("Control thread is not running.")

        print_state(robot)

        # 2. Enable admittance control
        answer = input(
            "\nType ENABLE to activate admittance control: "
        )

        if answer != "ENABLE":
            print("Admittance activation cancelled.")
            return

        print("\n[2] Enabling admittance control...")
        admittance_attempted = True
        robot.enable_admittance_control()

        if not robot._admittance_enabled:
            raise RuntimeError("Admittance control was not enabled.")

        if not robot._motion_paused:
            raise RuntimeError(
                "Motion commands are unexpectedly enabled."
            )

        print("Admittance control enabled.")
        print("Normal motion commands remain paused.")

        # 3. Observe state
        print(f"\n[3] Observing for {TEST_DURATION:.1f} seconds...")
        print("Press Ctrl+C to finish early.")

        start = time.monotonic()

        while time.monotonic() - start < TEST_DURATION:
            if robot._thread is None or not robot._thread.is_alive():
                raise RuntimeError("Control thread has stopped.")

            if not robot._motion_paused:
                raise RuntimeError(
                    "Motion commands are unexpectedly enabled."
                )

            print_state(robot)
            time.sleep(PRINT_INTERVAL)

        print("\nObservation completed.")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    except Exception:
        logging.exception("Admittance control test failed.")

    finally:
        # 4. Cleanup
        print("\n[4] Cleanup...")

        if connected:
            if admittance_attempted:
                try:
                    robot.disable_admittance_control()
                    print("Admittance control disabled.")
                except Exception:
                    logging.exception(
                        "Failed to disable admittance control. "
                        "Check the physical robot/controller state."
                    )

            try:
                robot.disconnect()
                print("Disconnected.")
            except Exception:
                logging.exception(
                    "Disconnect failed. "
                    "Check the physical robot/controller state."
                )

        print("Test script finished.")


if __name__ == "__main__":
    main()