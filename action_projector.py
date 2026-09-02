from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


@dataclass
class XArmProjectionResult:
    action: np.ndarray
    qpos: np.ndarray
    ee: np.ndarray
    target_qpos: np.ndarray
    clipped_action: np.ndarray
    feasible: bool
    error: str | None = None


class XArmActionProjector:
    """
    Cartesian actionを、実際にxArmへ送る関節角制約まで考慮した
    実行可能Cartesian actionへ射影する。

    このクラスは実機を操作しない。
    """

    def __init__(
        self,
        ik_solver,
        fk_solver,
        workspace_bounds_m,
        max_cartesian_delta_m,
        max_orientation_delta_rad,
        max_delta,
        max_gripper_delta,
    ):
        self.ik_solver = ik_solver
        self.fk_solver = fk_solver

        self.workspace_bounds_m = np.asarray(
            workspace_bounds_m,
            dtype=np.float32,
        )

        if self.workspace_bounds_m.shape != (3, 2):
            raise ValueError(
                "workspace_bounds_m must have shape (3, 2), "
                f"got {self.workspace_bounds_m.shape}"
            )

        self.max_cartesian_delta_m = float(
            max_cartesian_delta_m
        )

        self.max_orientation_delta_rad = float(
            max_orientation_delta_rad
        )
        
        self.max_delta = float(max_delta)

        self.max_gripper_delta = float(
            max_gripper_delta
        )

    def project(
        self,
        action,
        current_qpos,
        current_ee,
        current_gripper,
    ):
        """
        Args:
            action:
                shape (8,)
                [x, y, z, qx, qy, qz, qw, gripper]

            current_qpos:
                shape (7,)
                現在の関節角 [rad]

            current_ee:
                shape (7,)
                [x, y, z, qx, qy, qz, qw]

            current_gripper:
                scalar
                0=open, 1=closed

        Returns:
            XArmProjectionResult

            action:
                joint clip後のqposをFKした
                実際に実行可能とみなすCartesian action

            qpos:
                joint clip後のsafe_qpos

            ee:
                safe_qposをFKしたEE pose
        """

        action = np.asarray(
            action,
            dtype=np.float32,
        ).reshape(-1)

        current_qpos = np.asarray(
            current_qpos,
            dtype=np.float32,
        ).reshape(-1)

        current_ee = np.asarray(
            current_ee,
            dtype=np.float32,
        ).reshape(-1)

        if action.size != 8:
            raise ValueError(
                f"action must have 8 values, got {action.size}"
            )

        if current_qpos.size != 7:
            raise ValueError(
                "current_qpos must have 7 values, "
                f"got {current_qpos.size}"
            )

        if current_ee.size != 7:
            raise ValueError(
                "current_ee must have 7 values, "
                f"got {current_ee.size}"
            )

        # 1. Cartesian safety clip
        clipped_action = self.clip_cartesian_action(
            action=action,
            current_ee=current_ee,
            current_gripper=current_gripper,
        )

        # 2. Cartesian pose -> IK input
        target_xyz = clipped_action[:3]
        target_quat = clipped_action[3:7]

        target_rpy = Rotation.from_quat(
            target_quat
        ).as_euler("xyz")

        pose_rpy = np.concatenate([
            target_xyz * 1000.0,
            target_rpy,
        ])

        # 3. IK
        try:
            target_qpos = self.ik_solver.solve(
                pose_rpy=pose_rpy,
                q_pre=current_qpos,
            )
        except Exception as exc:
            return XArmProjectionResult(
                action=clipped_action.copy(),
                qpos=current_qpos.copy(),
                ee=current_ee.copy(),
                target_qpos=np.full(
                    7,
                    np.nan,
                    dtype=np.float32,
                ),
                clipped_action=clipped_action.copy(),
                feasible=False,
                error=str(exc),
            )

        # 4. joint delta clip

        safe_qpos = current_qpos.copy()
        num_inner_steps = 5  # 50 Hz / 10 Hz

        for _ in range(num_inner_steps):
            joint_delta = target_qpos - safe_qpos
            delta_norm = np.linalg.norm(joint_delta)

            if delta_norm > self.max_delta:
                joint_delta = (
                    joint_delta / delta_norm
                    * self.max_delta
                )

            safe_qpos = safe_qpos + joint_delta

        # 5. FK
        try:
            effective_ee = self.fk_solver.solve(
                safe_qpos
            )
        except Exception as exc:
            return XArmProjectionResult(
                action=clipped_action.copy(),
                qpos=safe_qpos.copy(),
                ee=current_ee.copy(),
                target_qpos=target_qpos.copy(),
                clipped_action=clipped_action.copy(),
                feasible=False,
                error=str(exc),
            )

        # gripperはCartesian clip後の値を使用
        effective_gripper = float(
            clipped_action[7]
        )

        effective_action = np.concatenate([
            effective_ee,
            [effective_gripper],
        ]).astype(np.float32)

        return XArmProjectionResult(
            action=effective_action,
            qpos=safe_qpos.astype(np.float32),
            ee=effective_ee.astype(np.float32),
            target_qpos=target_qpos.astype(np.float32),
            clipped_action=clipped_action.astype(np.float32),
            feasible=True,
            error=None,
        )

    def clip_cartesian_action(
        self,
        action,
        current_ee,
        current_gripper,
    ):
        """
        Cartesian actionに対して、
        position / orientation / workspace / gripper制約を適用する。
        """

        action = np.asarray(
            action,
            dtype=np.float32,
        ).reshape(-1)

        current_ee = np.asarray(
            current_ee,
            dtype=np.float32,
        ).reshape(-1)

        # ----------------------------
        # position
        # ----------------------------
        target_xyz = action[:3].copy()
        current_xyz = current_ee[:3]

        delta_xyz = np.clip(
            target_xyz - current_xyz,
            -self.max_cartesian_delta_m,
            self.max_cartesian_delta_m,
        )

        target_xyz = current_xyz + delta_xyz

        target_xyz = np.clip(
            target_xyz,
            self.workspace_bounds_m[:, 0],
            self.workspace_bounds_m[:, 1],
        )

        # ----------------------------
        # orientation
        # ----------------------------
        current_rotation = Rotation.from_quat(
            current_ee[3:7]
        )

        target_quat = action[3:7].copy()

        quat_norm = float(
            np.linalg.norm(target_quat)
        )

        if quat_norm < 1e-6:
            raise ValueError(
                "Predicted quaternion has near-zero norm"
            )

        target_rotation = Rotation.from_quat(
            target_quat / quat_norm
        )

        relative_rotation = (
            target_rotation
            * current_rotation.inv()
        )

        rotation_vector = (
            relative_rotation.as_rotvec()
        )

        angle = float(
            np.linalg.norm(rotation_vector)
        )

        if (
            angle
            > self.max_orientation_delta_rad
        ):
            relative_rotation = Rotation.from_rotvec(
                rotation_vector
                * (
                    self.max_orientation_delta_rad
                    / angle
                )
            )

            target_rotation = (
                relative_rotation
                * current_rotation
            )

        target_quat = (
            target_rotation
            .as_quat()
            .astype(np.float32)
        )

        # ----------------------------
        # gripper
        # ----------------------------
        target_gripper = float(
            np.clip(
                action[7],
                0.0,
                1.0,
            )
        )

        target_gripper = float(
            np.clip(
                target_gripper,
                float(current_gripper)
                - self.max_gripper_delta,
                float(current_gripper)
                + self.max_gripper_delta,
            )
        )

        return np.concatenate([
            target_xyz,
            target_quat,
            [target_gripper],
        ]).astype(np.float32)