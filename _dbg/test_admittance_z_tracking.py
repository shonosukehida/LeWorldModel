"""Compare Z-axis Cartesian tracking with admittance OFF / ON.

- No GELLO.
- No external contact.
- Command a small +Z Cartesian step.
- Record target Z and actual Z.
- Do not automatically return to the initial pose.

This script is for diagnostic observation, not emergency stopping.
"""

import logging
import time

import numpy as np

from robopy.config.robot_config.xarm_config import XArmConfig
from robopy.robots.xarm.xarm_follower import XArmFollower


ROBOT_IP = "192.168.1.240"

# 最初は小さな移動量で確認する
Z_STEP_MM = 2.0

# 指令後の観測時間
OBSERVE_DURATION = 5.0

# 表示間隔
PRINT_INTERVAL = 0.1

# 診断用の変位上限（ハードウェアの安全制限ではない）
MAX_XY_DEVIATION_MM = 5.0
MAX_Z_DEVIATION_MM = 10.0


def get_pose_aa(robot: XArmFollower) -> np.ndarray:
    """Get Cartesian pose [x, y, z, rx, ry, rz] in mm/rad."""

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

        return np.asarray(
            pose,
            dtype=np.float32,
        ).copy()


def pause_commands(robot: XArmFollower) -> None:
    """Stop sending new commands from robopy's control thread."""

    with robot._control_lock:
        robot._motion_paused = True


def observe(
    robot: XArmFollower,
    initial_xyz: np.ndarray,
    target_xyz: np.ndarray,
    duration: float,
) -> None:

    start = time.monotonic()

    print("\nTime [s] | Actual Z [mm] | Target Z [mm] | Error [mm]")
    print("-" * 62)

    while True:
        elapsed = time.monotonic() - start

        if elapsed >= duration:
            break

        with robot._control_lock:
            paused = robot._motion_paused

        if paused:
            raise RuntimeError(
                "Motion commands were paused unexpectedly."
            )

        if robot._thread is None or not robot._thread.is_alive():
            raise RuntimeError(
                "Robopy control thread has stopped."
            )

        # 実機の現在位置をSDKから取得
        current_pose = get_pose_aa(robot)
        actual_xyz = current_pose[:3]

        error_z = target_xyz[2] - actual_xyz[2]
        displacement = actual_xyz - initial_xyz

        print(
            f"{elapsed:8.3f} | "
            f"{actual_xyz[2]:13.3f} | "
            f"{target_xyz[2]:13.3f} | "
            f"{error_z:10.3f}",
            flush=True,
        )

        if np.any(
            np.abs(displacement[:2]) > MAX_XY_DEVIATION_MM
        ):
            raise RuntimeError(
                "Unexpected X/Y displacement detected."
            )

        if abs(float(displacement[2])) > MAX_Z_DEVIATION_MM:
            raise RuntimeError(
                "Z displacement exceeded observation limit."
            )

        time.sleep(PRINT_INTERVAL)


def main():

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    robot = XArmFollower(
        XArmConfig(follower_ip=ROBOT_IP)
    )

    connected = False
    admittance_attempted = False

    print("\n=== Cartesian Z Tracking Test ===")
    print("GELLO is not used.")
    print("No external force should be applied.")
    print("The robot will receive a +2 mm Z command.")
    print("connect() initializes xArm and opens Gripper G2.")

    # --------------------------------------------------
    # Select test condition
    # --------------------------------------------------

    mode = input(
        "\nAdmittance mode [ON/OFF]: "
    ).strip().upper()

    if mode not in ("ON", "OFF"):
        print("Invalid mode.")
        return

    use_admittance = mode == "ON"

    print(f"Selected mode: {mode}")

    answer = input(
        "\nWorkspace and emergency stop checked? "
        "Type YES to connect: "
    )

    if answer != "YES":
        print("Cancelled.")
        return

    try:

        # --------------------------------------------------
        # 1. Connect
        # --------------------------------------------------

        print("\n[1] Connecting...")

        robot.connect()
        connected = True

        # 接続直後は位置指令を停止し、現在位置を確認する
        pause_commands(robot)

        initial_pose = get_pose_aa(robot)

        print(
            "Initial xyz [mm]:",
            np.round(initial_pose[:3], 3),
        )

        # --------------------------------------------------
        # 2. Enable admittance if requested
        # --------------------------------------------------

        if use_admittance:

            answer = input(
                "\nType ENABLE to enable admittance: "
            )

            if answer != "ENABLE":
                print("Cancelled.")
                return

            print("\n[2] Enabling admittance...")

            admittance_attempted = True
            robot.enable_admittance_control()

            if not robot._admittance_enabled:
                raise RuntimeError(
                    "Admittance control was not enabled."
                )

        else:
            print("\n[2] Admittance remains OFF.")

        # --------------------------------------------------
        # 3. Set the Cartesian target
        # --------------------------------------------------

        print("\n[3] Preparing Cartesian target...")

        # 実際の現在位置を改めて取得する
        with robot._control_lock:

            current_pose = get_pose_aa(robot)

            target_pose = current_pose.copy()

            # Z方向だけ+2 mm
            target_pose[2] += Z_STEP_MM

        print(
            "Current xyz [mm]:",
            np.round(current_pose[:3], 3),
        )

        print(
            "Target xyz [mm]: ",
            np.round(target_pose[:3], 3),
        )

        answer = input(
            "\nType MOVE to execute the +Z command: "
        )

        if answer != "MOVE":
            print("Movement cancelled.")
            return

        # --------------------------------------------------
        # 4. Resume control and send Cartesian target
        # --------------------------------------------------

        print("\n[4] Sending Cartesian Z command...")

        with robot._control_lock:

            # resume_motion_commands() により、
            # 一時停止していた位置指令を再開する。
            robot.resume_motion_commands()

            # Cartesian絶対位置指令へ切り替える。
            robot.command_cartesian_absolute(
                target_pose.copy(),
                gripper=None,
            )

        # --------------------------------------------------
        # 5. Observe actual response
        # --------------------------------------------------

        print(
            f"\n[5] Observing for {OBSERVE_DURATION:.1f} seconds..."
        )

        observe(
            robot=robot,
            initial_xyz=current_pose[:3].copy(),
            target_xyz=target_pose[:3].copy(),
            duration=OBSERVE_DURATION,
        )

        print("\nObservation completed.")

        final_pose = get_pose_aa(robot)

        final_error = (
            target_pose[2] - final_pose[2]
        )

        print("\n===== Summary =====")

        print("Admittance:", mode)

        print(
            "Initial Z [mm]:",
            round(float(current_pose[2]), 3),
        )

        print(
            "Target Z [mm]: ",
            round(float(target_pose[2]), 3),
        )

        print(
            "Final Z [mm]:  ",
            round(float(final_pose[2]), 3),
        )

        print(
            "Final error [mm]:",
            round(float(final_error), 3),
        )

    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    except Exception:
        logging.exception("Test failed.")

    finally:

        print("\n[Cleanup]")

        if connected:

            try:
                pause_commands(robot)
                print("Position commands paused.")

            except Exception:
                logging.exception(
                    "Failed to pause motion commands."
                )

            if admittance_attempted:

                try:
                    robot.disable_admittance_control()
                    print("Admittance disabled.")

                except Exception:
                    logging.exception(
                        "Failed to disable admittance. "
                        "Check the controller state."
                    )

            try:
                robot.disconnect()
                print("Disconnected.")

            except Exception:
                logging.exception(
                    "Failed to disconnect."
                )

        print("Test finished.")


if __name__ == "__main__":
    main()