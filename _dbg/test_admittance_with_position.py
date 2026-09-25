"""Test Cartesian position commands together with xArm admittance control.

Phase 1: Hold the current Cartesian pose while admittance is enabled.
Phase 2: Apply a small force along Z while that pose command is active.
Phase 3: Optionally move the X target by +2 mm without contact.

This is a staged experimental test, not an emergency-stop mechanism.
"""

import logging
import time

import numpy as np

from robopy.config.robot_config.xarm_config import XArmConfig
from robopy.robots.xarm.xarm_follower import XArmFollower


ROBOT_IP = "192.168.1.240"

PRINT_INTERVAL = 0.25

HOLD_DURATION = 3.0
CONTACT_DURATION = 10.0
STEP_DURATION = 3.0

X_STEP_MM = 2

# Software observation limits, not hardware safety limits.
MAX_XY_DEVIATION_MM = 5.0
MAX_Z_DEVIATION_NO_FORCE_MM = 10.0
MAX_Z_DEVIATION_CONTACT_MM = 35.0


def get_pose_aa(robot: XArmFollower) -> np.ndarray:
    """Read the current Cartesian pose [mm, rad] directly from the SDK."""

    with robot._control_lock:
        if robot._robot is None:
            raise RuntimeError("xArm is not connected.")

        code, pose = robot._robot.get_position_aa(
            is_radian=True
        )

        if code != 0 or pose is None:
            raise RuntimeError(
                f"get_position_aa failed: code={code}"
            )

        return np.asarray(pose, dtype=np.float32).copy()


def check_control_thread(robot: XArmFollower) -> None:
    if robot._thread is None or not robot._thread.is_alive():
        raise RuntimeError("Control thread has stopped.")

    with robot._control_lock:
        if robot._motion_paused:
            raise RuntimeError(
                "Motion commands have been paused, possibly due to an error."
            )

        if not robot._admittance_enabled:
            raise RuntimeError(
                "Admittance control is not enabled."
            )


def observe(
    robot: XArmFollower,
    reference_xyz_mm: np.ndarray,
    duration: float,
    max_z_deviation_mm: float,
) -> None:
    """Observe movement and abort this test if limits are exceeded."""

    start = time.monotonic()

    while time.monotonic() - start < duration:
        check_control_thread(robot)

        ee = robot.get_ee_pos_quat()
        xyz_mm = ee[:3] * 1000.0
        delta_mm = xyz_mm - reference_xyz_mm

        print(
            f"xyz [mm]: {np.round(xyz_mm, 2)} | "
            f"delta [mm]: {np.round(delta_mm, 2)} | "
            f"paused: {robot._motion_paused} | "
            f"admittance: {robot._admittance_enabled}",
            flush=True,
        )

        if np.any(np.abs(delta_mm[:2]) > MAX_XY_DEVIATION_MM):
            raise RuntimeError(
                "Unexpected X/Y displacement detected."
            )

        if abs(float(delta_mm[2])) > max_z_deviation_mm:
            raise RuntimeError(
                "Z displacement exceeded the test observation limit."
            )

        time.sleep(PRINT_INTERVAL)


def pause_motion_commands(robot: XArmFollower) -> None:
    """Stop sending new position commands from the control thread."""

    with robot._control_lock:
        robot._motion_paused = True


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    robot = XArmFollower(
        XArmConfig(follower_ip=ROBOT_IP)
    )

    connected = False
    admittance_attempted = False

    try:
        print("\n=== Admittance + position command test ===")
        print("connect() initializes the robot and opens Gripper G2.")
        print("The test will send repeated Cartesian position commands.")

        answer = input(
            "\nWorkspace and emergency stop checked? "
            "Type YES to connect: "
        )

        if answer != "YES":
            print("Cancelled.")
            return

        # --------------------------------------------------
        # 1. Connect and read the initial pose
        # --------------------------------------------------

        print("\n[1] Connecting...")
        robot.connect()
        connected = True

        time.sleep(1.0)

        initial_pose = get_pose_aa(robot)

        print(
            "Initial xyz [mm]:",
            np.round(initial_pose[:3], 2),
        )

        # --------------------------------------------------
        # 2. Enable admittance control
        # --------------------------------------------------

        answer = input(
            "\nType ENABLE to enable admittance control: "
        )

        if answer != "ENABLE":
            print("Cancelled before enabling admittance.")
            return

        print("\n[2] Enabling admittance...")
        admittance_attempted = True
        robot.enable_admittance_control()

        if not robot._admittance_enabled:
            raise RuntimeError(
                "Admittance control was not enabled."
            )

        if not robot._motion_paused:
            raise RuntimeError(
                "Motion commands are unexpectedly active."
            )

        # --------------------------------------------------
        # 3. Resume position commands at the CURRENT pose
        # --------------------------------------------------

        answer = input(
            "\nType HOLD to start sending the current pose "
            "as a position target: "
        )

        if answer != "HOLD":
            print("Position-command test cancelled.")
            return

        print("\n[3] Starting Cartesian position commands...")

        # Keep mode switching, pose capture and command setup
        # together so the control thread cannot intervene.
        with robot._control_lock:
            hold_pose = get_pose_aa(robot)

            robot.resume_motion_commands()

            robot.command_cartesian_absolute(
                hold_pose.copy(),
                gripper=None,
            )

        reference_xyz_mm = hold_pose[:3].copy()

        print(
            "Commanded xyz [mm]:",
            np.round(reference_xyz_mm, 2),
        )

        # --------------------------------------------------
        # 4. Observe without external force
        # --------------------------------------------------

        print(
            f"\n[4] No-force observation: {HOLD_DURATION:.1f} s"
        )
        print("Do not touch the robot during this phase.")

        observe(
            robot,
            reference_xyz_mm,
            HOLD_DURATION,
            MAX_Z_DEVIATION_NO_FORCE_MM,
        )

        print("\nNo-force observation completed.")

        # --------------------------------------------------
        # 5. Observe Z response while position commands continue
        # --------------------------------------------------

        answer = input(
            "\nType CONTACT to begin the Z-response observation: "
        )

        if answer != "CONTACT":
            print("Contact test skipped.")
            return

        print(
            f"\n[5] Z-response observation: {CONTACT_DURATION:.1f} s"
        )
        print(
            "Use only the approved gentle contact procedure. "
            "Stop immediately if the motion is unexpected."
        )

        observe(
            robot,
            reference_xyz_mm,
            CONTACT_DURATION,
            MAX_Z_DEVIATION_CONTACT_MM,
        )

        print("\nZ-response observation completed.")

        # --------------------------------------------------
        # 6. Optional tiny X-direction target change
        # --------------------------------------------------

        print("\nRelease the external force before the next phase.")

        answer = input(
            f"Type STEP to command X +{X_STEP_MM:.1f} mm, "
            "or press Enter to skip: "
        )

        if answer != "STEP":
            print("X-step test skipped.")
            return

        print("\n[6] Sending a small X-direction target change...")

        # Re-read the actual pose rather than reusing the old
        # target, avoiding an unintended return to the old Z.
        with robot._control_lock:
            current_pose = get_pose_aa(robot)

            step_pose = current_pose.copy()
            step_pose[0] += X_STEP_MM

            robot.command_cartesian_absolute(
                step_pose,
                gripper=None,
            )

        print(
            "New target xyz [mm]:",
            np.round(step_pose[:3], 2),
        )

        observe(
            robot,
            current_pose[:3].copy(),
            STEP_DURATION,
            MAX_Z_DEVIATION_NO_FORCE_MM,
        )

        print("\nX-step observation completed.")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    except Exception:
        logging.exception("Test failed.")

    finally:
        print("\n[Cleanup]")

        if connected:
            # Pause new position commands BEFORE disabling admittance.
            try:
                pause_motion_commands(robot)
                print("New position commands paused.")
            except Exception:
                logging.exception(
                    "Failed to pause position commands."
                )

            if admittance_attempted:
                try:
                    robot.disable_admittance_control()
                    print("Admittance disabled.")
                except Exception:
                    logging.exception(
                        "Failed to disable admittance. "
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