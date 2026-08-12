import time
from pathlib import Path

import sys
# LeWorldModel のルートディレクトリを import path に追加
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import hydra
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig

from eval_real_robot import XArmInferenceEnv


@hydra.main(
    version_base=None,
    config_path="../config/eval",
    config_name="flip_mug",
)
def run(cfg: DictConfig):
    real_cfg = cfg.eval.real_robot

    if str(cfg.plan_config.action_space) != "cartesian":
        raise ValueError(
            "This test requires plan_config.action_space=cartesian"
        )

    env = XArmInferenceEnv(
        real_cfg,
        cfg.plan_config,
    )

    output_dir = Path(
        "./robot/figures"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:
        input(
            "Place the robot at a safe initial pose, "
            "then press Enter..."
        )

        # -----------------------------
        # 初期状態取得
        # -----------------------------
        initial_qpos, _, initial_ee = (
            env.get_robot_state()
        )

        initial_xyz = initial_ee[:3].copy()
        initial_quat = initial_ee[3:7].copy()

        print("initial_xyz:", initial_xyz)
        print("initial_quat:", initial_quat)
        print("initial_qpos:", initial_qpos)

        # -----------------------------
        # 固定Cartesian target
        # -----------------------------
        target_xyz = initial_xyz.copy()

        # まずは安全に x 方向へ +2 cm
        # target_xyz[0] += 0.02

        # 姿勢は初期姿勢のまま固定
        target_quat = initial_quat.copy()

        # グリッパも現在状態のまま
        target_gripper = float(
            env._last_gripper
        )

        target_action = np.concatenate(
            [
                target_xyz,
                target_quat,
                [target_gripper],
            ]
        ).astype(np.float32)

        print("target_xyz:", target_xyz)
        print("target_quat:", target_quat)
        print(
            "target_gripper:",
            target_gripper,
        )

        # -----------------------------
        # 実行設定
        # -----------------------------
        num_steps = 30
        control_hz = float(
            real_cfg.control_hz
        )
        period = 1.0 / control_hz

        commanded_xyz_log = []
        actual_xyz_log = []
        qpos_log = []
        target_qpos_log = []
        safe_qpos_log = []

        print()
        print(
            f"Start fixed Cartesian test: "
            f"{num_steps} steps"
        )

        for step_idx in range(num_steps):
            tick = time.monotonic()

            # command送信前の状態
            qpos_before, _, ee_before = (
                env.get_robot_state()
            )

            # eval_real_robot.py と
            # 同じ経路でCartesian actionを実行
            commanded = env.execute(
                target_action,
                "cartesian",
            )
            
            #デバッグ用
            target_qpos = env._last_target_qpos.copy()
            safe_qpos = env._last_safe_qpos.copy()
            target_qpos_fk = env.forward_kinematics(
                target_qpos
            )

            safe_qpos_fk = env.forward_kinematics(
                safe_qpos
            )


            # 少し待ってから実状態を取得
            elapsed = (
                time.monotonic() - tick
            )
            remaining = period - elapsed

            if remaining > 0:
                time.sleep(remaining)

            qpos_after, _, ee_after = (
                env.get_robot_state()
            )
            
            #デバッグ用
            target_qpos_log.append(
                target_qpos
            )

            safe_qpos_log.append(
                safe_qpos
            )


            print(
                f"\nstep={step_idx:02d}"
            )

            print(
                "  commanded xyz    :",
                commanded[:3],
            )

            print(
                "  target_qpos FK xyz:",
                target_qpos_fk[:3],
            )

            print(
                "  safe_qpos FK xyz  :",
                safe_qpos_fk[:3],
            )

            print(
                "  actual xyz        :",
                ee_after[:3],
            )
            ##

            commanded_xyz = (
                commanded[:3].copy()
            )
            actual_xyz = (
                ee_after[:3].copy()
            )

            commanded_xyz_log.append(
                commanded_xyz
            )
            actual_xyz_log.append(
                actual_xyz
            )
            qpos_log.append(
                qpos_after.copy()
            )

            error_xyz = (
                commanded_xyz
                - actual_xyz
            )

            print(
                f"step={step_idx:02d} | "
                f"commanded="
                f"{commanded_xyz} | "
                f"actual="
                f"{actual_xyz} | "
                f"error="
                f"{error_xyz}"
            )

        # -----------------------------
        # numpy化
        # -----------------------------
        commanded_xyz_log = np.asarray(
            commanded_xyz_log,
            dtype=np.float32,
        )

        actual_xyz_log = np.asarray(
            actual_xyz_log,
            dtype=np.float32,
        )

        steps = np.arange(
            num_steps
        )

        # -----------------------------
        # プロット
        # -----------------------------
        fig, axes = plt.subplots(
            3,
            1,
            figsize=(12, 10),
            sharex=True,
        )

        axis_names = [
            "x",
            "y",
            "z",
        ]

        for i, axis_name in enumerate(
            axis_names
        ):
            axes[i].plot(
                steps,
                commanded_xyz_log[:, i],
                label=(
                    f"Commanded "
                    f"{axis_name}"
                ),
            )

            axes[i].plot(
                steps,
                actual_xyz_log[:, i],
                label=(
                    f"Actual "
                    f"{axis_name}"
                ),
            )

            axes[i].set_ylabel(
                f"{axis_name.upper()} "
                f"Position [m]"
            )

            axes[i].legend()
            axes[i].grid(True)

        axes[2].set_xlabel(
            "Step"
        )

        fig.suptitle(
            "Fixed Cartesian Target Test"
        )

        fig.tight_layout()

        save_path = (
            output_dir
            / "fixed_cartesian_test.png"
        )

        fig.savefig(
            save_path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close(fig)

        print(
            f"Saved plot to: "
            f"{save_path}"
        )


        # -----------------------------
        # Joint position の可視化
        # target_qpos / safe_qpos / actual_qpos
        # -----------------------------
        print("target_qpos_log shape:", len(target_qpos_log))
        print("safe_qpos_log shape:", len(safe_qpos_log))
        print("qpos_log shape:", len(qpos_log))
        print("steps shape:", steps.shape)
        target_qpos_log = np.asarray(
            target_qpos_log,
            dtype=np.float32,
        )

        safe_qpos_log = np.asarray(
            safe_qpos_log,
            dtype=np.float32,
        )

        qpos_log = np.asarray(
            qpos_log,
            dtype=np.float32,
        )

        steps = np.arange(
            len(target_qpos_log)
        )

        fig, axes = plt.subplots(
            7,
            1,
            figsize=(12, 18),
            sharex=True,
        )

        for joint_idx in range(7):
            axes[joint_idx].plot(
                steps,
                target_qpos_log[:, joint_idx],
                label="Target qpos",
            )

            axes[joint_idx].plot(
                steps,
                safe_qpos_log[:, joint_idx],
                label="Safe qpos",
            )

            axes[joint_idx].plot(
                steps,
                qpos_log[:, joint_idx],
                label="Actual qpos",
            )

            axes[joint_idx].set_ylabel(
                f"Joint {joint_idx + 1}\n[rad]"
            )

            axes[joint_idx].grid(True)
            axes[joint_idx].legend()

        axes[-1].set_xlabel(
            "Step"
        )

        fig.suptitle(
            "Target vs Safe vs Actual Joint Position"
        )

        fig.tight_layout()

        joint_plot_path = (
            output_dir
            / "target_safe_actual_qpos.png"
        )

        fig.savefig(
            joint_plot_path,
            dpi=150,
            bbox_inches="tight",
        )

        plt.close(fig)

        print(
            f"Saved joint position plot to: "
            f"{joint_plot_path}"
        )

    finally:
        env.close()


if __name__ == "__main__":
    run()