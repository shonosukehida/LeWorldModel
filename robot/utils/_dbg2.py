"""Check the orientation format returned by xArm get_forward_kinematics()."""

from __future__ import annotations

import time

import numpy as np
from scipy.spatial.transform import Rotation
from xarm.wrapper import XArmAPI


ROBOT_IP = "192.168.1.240"


def quaternion_distance_rad(
    q1: np.ndarray,
    q2: np.ndarray,
) -> float:
    """Return the shortest angular distance between two quaternions."""
    q1 = np.asarray(q1, dtype=np.float64)
    q2 = np.asarray(q2, dtype=np.float64)

    q1 /= np.linalg.norm(q1)
    q2 /= np.linalg.norm(q2)

    # q と -q は同じ回転を表す
    dot = float(np.clip(np.abs(np.dot(q1, q2)), 0.0, 1.0))
    return 2.0 * np.arccos(dot)


def check_once(arm: XArmAPI) -> None:
    # 現在の実機関節角を取得
    joint_code, servo_angles = arm.get_servo_angle(is_radian=True)
    if joint_code != 0 or servo_angles is None:
        raise RuntimeError(
            f"get_servo_angle failed: code={joint_code}"
        )

    joints = np.asarray(servo_angles[:7], dtype=np.float64)

    # 現在のEE姿勢をaxis-angle形式で取得
    measured_code, measured_pose_aa = arm.get_position_aa(
        is_radian=True
    )
    if measured_code != 0 or measured_pose_aa is None:
        raise RuntimeError(
            f"get_position_aa failed: code={measured_code}"
        )

    measured_pose_aa = np.asarray(
        measured_pose_aa,
        dtype=np.float64,
    )

    # 同じ現在関節角に対してFKを計算
    fk_code, fk_pose = arm.get_forward_kinematics(
        joints.tolist(),
        input_is_radian=True,
    )
    if fk_code != 0 or fk_pose is None:
        raise RuntimeError(
            f"get_forward_kinematics failed: code={fk_code}"
        )

    fk_pose = np.asarray(fk_pose, dtype=np.float64)

    # 実測姿勢:
    # get_position_aa() の後半3要素は回転ベクトル
    measured_quat = Rotation.from_rotvec(
        measured_pose_aa[3:6]
    ).as_quat()

    # 仮説1:
    # FKの後半3要素がaxis-angle / rotation vector
    fk_quat_as_rotvec = Rotation.from_rotvec(
        fk_pose[3:6]
    ).as_quat()

    # 仮説2:
    # FKの後半3要素がroll, pitch, yaw
    fk_quat_as_rpy_xyz = Rotation.from_euler(
        "xyz",
        fk_pose[3:6],
        degrees=False,
    ).as_quat()

    rotvec_error = quaternion_distance_rad(
        measured_quat,
        fk_quat_as_rotvec,
    )
    rpy_error = quaternion_distance_rad(
        measured_quat,
        fk_quat_as_rpy_xyz,
    )

    position_error_mm = np.linalg.norm(
        measured_pose_aa[:3] - fk_pose[:3]
    )

    print("=" * 70)
    print("Current joints [rad]")
    print(joints)

    print("\nMeasured pose from get_position_aa()")
    print(measured_pose_aa)

    print("\nFK pose from get_forward_kinematics()")
    print(fk_pose)

    print("\nPosition error")
    print(f"{position_error_mm:.6f} mm")

    print("\nMeasured quaternion [qx, qy, qz, qw]")
    print(measured_quat)

    print("\nFK interpreted as rotation vector")
    print(fk_quat_as_rotvec)
    print(
        f"Orientation error: "
        f"{np.degrees(rotvec_error):.6f} deg"
    )

    print("\nFK interpreted as RPY xyz")
    print(fk_quat_as_rpy_xyz)
    print(
        f"Orientation error: "
        f"{np.degrees(rpy_error):.6f} deg"
    )

    print("\nResult")
    if rotvec_error < rpy_error:
        print(
            "get_forward_kinematics() の pose[3:6] は、"
            "axis-angle / rotation vector と解釈する方が一致します。"
        )
    else:
        print(
            "get_forward_kinematics() の pose[3:6] は、"
            "roll-pitch-yaw と解釈する方が一致します。"
        )


def main() -> None:
    arm = XArmAPI(ROBOT_IP, is_radian=True)

    try:
        # 接続直後の状態取得を少し待つ
        time.sleep(0.5)

        # ロボットは動かさず、読み取りとFK計算だけ行う
        check_once(arm)

    finally:
        arm.disconnect()


if __name__ == "__main__":
    main()