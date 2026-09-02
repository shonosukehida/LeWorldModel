import sys
import time
from pathlib import Path

import h5py
import hydra
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig


# LeWorldModel のルートディレクトリを import path に追加
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eval_real_robot import XArmInferenceEnv


EPISODE_PATH = Path(
    "/home/hida/.stable_worldmodel/datasets/"
    "flip_mug/ep200_tm300_gripper/"
    "per_episode/episode_0.h5"
)


@hydra.main(
    version_base=None,
    config_path="../config/robot",
    config_name="test_episode",
)
def run(cfg: DictConfig):
    real_cfg = cfg.eval.real_robot

    # ============================================================
    # Dataset 読み込み
    # ============================================================
    with h5py.File(EPISODE_PATH, "r") as h5file:
        leader = np.asarray(
            h5file["arms/leader"][:],
            dtype=np.float32,
        )

        follower = np.asarray(
            h5file["arms/follower"][:],
            dtype=np.float32,
        )

    if leader.ndim != 2 or leader.shape[1] < 7:
        raise ValueError(
            f"leader must have shape (T, >=7), got {leader.shape}"
        )

    if follower.ndim != 2 or follower.shape[1] < 7:
        raise ValueError(
            f"follower must have shape (T, >=7), got {follower.shape}"
        )

    if leader.shape[0] != follower.shape[0]:
        raise ValueError(
            "leader and follower must have the same number of steps: "
            f"{leader.shape[0]} != {follower.shape[0]}"
        )

    leader_qpos = leader[:, :7]
    original_follower_qpos = follower[:, :7]

    num_steps = leader_qpos.shape[0]

    print("========================================")
    print("Dataset")
    print("========================================")
    print("episode path :", EPISODE_PATH)
    print("leader shape :", leader.shape)
    print("follower shape:", follower.shape)
    print("num_steps    :", num_steps)

    # ============================================================
    # 推論時と同じ xArm 環境
    # ============================================================
    env = XArmInferenceEnv(
        real_cfg,
        cfg.plan_config,
        use_camera=False,
    )

    control_hz = float(real_cfg.control_hz)
    period = 1.0 / control_hz

    # ------------------------------------------------------------
    # Logs
    # ------------------------------------------------------------
    target_qpos_log = []
    command_qpos_log = []

    # command を送る「前」の qpos
    qpos_before_log = []

    # command を送った「後」の qpos
    qpos_after_log = []

    timestamp_log = []

    try:
        # ========================================================
        # 初期姿勢確認
        # ========================================================
        current_qpos, _, _ = env.get_robot_state()

        initial_target = leader_qpos[0]

        initial_error = (
            initial_target - current_qpos
        )

        print()
        print("========================================")
        print("Initial state")
        print("========================================")

        print("Current xArm qpos:")
        print(current_qpos)

        print()
        print("Dataset leader[0]:")
        print(initial_target)

        print()
        print("Difference:")
        print(initial_error)

        print()
        print(
            "max abs difference [rad]:",
            np.max(np.abs(initial_error)),
        )

        print()
        print(
            "max abs difference [deg]:",
            np.rad2deg(
                np.max(np.abs(initial_error))
            ),
        )

        input(
            "\nロボットを episode の初期姿勢付近に配置し、"
            "周囲の安全を確認して Enter..."
        )

        # Enter後にもう一度取得
        current_qpos, _, _ = env.get_robot_state()

        print()
        print("qpos immediately before replay:")
        print(current_qpos)

        print(
            "difference from leader[0]:",
            leader_qpos[0] - current_qpos,
        )

        print(
            "max difference [deg]:",
            np.rad2deg(
                np.max(
                    np.abs(
                        leader_qpos[0]
                        - current_qpos
                    )
                )
            ),
        )

        input(
            "\nEnter で episode replay を開始..."
        )

        # ========================================================
        # Episode replay
        # ========================================================
        print()
        print("========================================")
        print(
            f"Replay start: "
            f"{num_steps} steps, "
            f"{control_hz:.1f} Hz"
        )
        print("========================================")

        started = time.monotonic()

        for step_idx in range(num_steps):
            tick = time.monotonic()

            # ----------------------------------------------------
            # command 前の実測状態
            # ----------------------------------------------------
            qpos_before, _, _ = (
                env.get_robot_state()
            )

            qpos_before_log.append(
                qpos_before.copy()
            )

            # Dataset の leader joint
            action = leader_qpos[
                step_idx
            ].copy()

            # ----------------------------------------------------
            # 推論時と同じ execute() を使用
            # ----------------------------------------------------
            commanded = env.execute(
                action,
                "joint",
            )

            target_qpos = (
                env._last_target_qpos.copy()
            )

            command_qpos = (
                env._last_command_qpos.copy()
            )

            target_qpos_log.append(
                target_qpos
            )

            command_qpos_log.append(
                command_qpos
            )

            # ----------------------------------------------------
            # control_hz を維持
            # ----------------------------------------------------
            elapsed = (
                time.monotonic() - tick
            )

            remaining = (
                period - elapsed
            )

            if remaining > 0:
                time.sleep(remaining)

            # ----------------------------------------------------
            # command 後の実測状態
            # ----------------------------------------------------
            qpos_after, _, _ = (
                env.get_robot_state()
            )

            qpos_after_log.append(
                qpos_after.copy()
            )

            timestamp_log.append(
                time.monotonic()
                - started
            )

            # ----------------------------------------------------
            # tracking error
            # ----------------------------------------------------
            tracking_error = (
                command_qpos
                - qpos_after
            )

            leader_to_command_error = (
                target_qpos
                - command_qpos
            )

            print(
                f"step={step_idx:03d} | "
                f"leader->command max="
                f"{np.max(np.abs(leader_to_command_error)):.5f} rad | "
                f"command->actual max="
                f"{np.max(np.abs(tracking_error)):.5f} rad"
            )

    except KeyboardInterrupt:
        print("\nReplay interrupted.")

    finally:
        env.close()

    # ============================================================
    # numpy 化
    # ============================================================
    target_qpos_log = np.asarray(
        target_qpos_log,
        dtype=np.float32,
    )

    command_qpos_log = np.asarray(
        command_qpos_log,
        dtype=np.float32,
    )

    qpos_before_log = np.asarray(
        qpos_before_log,
        dtype=np.float32,
    )

    qpos_after_log = np.asarray(
        qpos_after_log,
        dtype=np.float32,
    )

    timestamp_log = np.asarray(
        timestamp_log,
        dtype=np.float32,
    )

    executed_steps = len(
        command_qpos_log
    )

    if executed_steps == 0:
        print("No steps were executed.")
        return

    # 中断された場合にも対応
    leader_qpos = leader_qpos[
        :executed_steps
    ]

    original_follower_qpos = (
        original_follower_qpos[
            :executed_steps
        ]
    )

    # ============================================================
    # Error statistics
    # ============================================================
    tracking_error = (
        command_qpos_log
        - qpos_after_log
    )

    leader_command_error = (
        target_qpos_log
        - command_qpos_log
    )

    # データ収集時の leader - follower
    original_tracking_error = (
        leader_qpos
        - original_follower_qpos
    )

    print()
    print("========================================")
    print("Results")
    print("========================================")

    print(
        "Replay command -> actual follower "
        "MAE [rad]:",
        np.mean(
            np.abs(tracking_error)
        ),
    )

    print(
        "Replay command -> actual follower "
        "max error [rad]:",
        np.max(
            np.abs(tracking_error)
        ),
    )

    print(
        "Replay command -> actual follower "
        "MAE [deg]:",
        np.rad2deg(
            np.mean(
                np.abs(tracking_error)
            )
        ),
    )

    print(
        "Leader action -> Robopy command "
        "MAE [deg]:",
        np.rad2deg(
            np.mean(
                np.abs(leader_command_error)
            ),
        )
    )

    print(
        "Dataset leader -> follower "
        "MAE [deg]:",
        np.rad2deg(
            np.mean(
                np.abs(
                    original_tracking_error
                )
            ),
        )
    )

    # ============================================================
    # 保存
    # ============================================================
    output_dir = (
        PROJECT_ROOT
        / "robot"
        / "figures"
        / "episode_replay"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.savez_compressed(
        output_dir
        / "episode_0_joint_replay.npz",
        leader_qpos=leader_qpos,
        original_follower_qpos=(
            original_follower_qpos
        ),
        target_qpos=target_qpos_log,
        command_qpos=command_qpos_log,
        qpos_before=qpos_before_log,
        qpos_after=qpos_after_log,
        timestamp=timestamp_log,
    )

    # ============================================================
    # Plot 1:
    # leader / Robopy command / replay actual
    # ============================================================
    steps = np.arange(
        executed_steps
    )

    fig, axes = plt.subplots(
        7,
        1,
        figsize=(14, 20),
        sharex=True,
    )

    for joint_idx in range(7):
        axes[joint_idx].plot(
            steps,
            leader_qpos[:, joint_idx],
            label="Dataset Leader",
        )

        axes[joint_idx].plot(
            steps,
            command_qpos_log[:, joint_idx],
            label="Robopy Command qpos",
        )

        axes[joint_idx].plot(
            steps,
            qpos_after_log[:, joint_idx],
            label="Replay Actual",
        )

        axes[joint_idx].set_ylabel(
            f"J{joint_idx + 1} [rad]"
        )

        axes[joint_idx].grid(True)
        axes[joint_idx].legend()

    axes[-1].set_xlabel(
        "Step"
    )

    fig.suptitle(
        "Dataset Leader vs "
        "Robopy Command vs Replay Actual"
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.98]
    )

    save_path = (
        output_dir
        / "leader_robopy_command_replay_qpos.png"
    )

    fig.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        "Saved:",
        save_path,
    )

    # ============================================================
    # Plot 2:
    # original follower vs replay follower
    # ============================================================
    fig, axes = plt.subplots(
        7,
        1,
        figsize=(14, 20),
        sharex=True,
    )

    for joint_idx in range(7):
        axes[joint_idx].plot(
            steps,
            leader_qpos[:, joint_idx],
            label="Leader",
        )

        axes[joint_idx].plot(
            steps,
            original_follower_qpos[
                :, joint_idx
            ],
            label="Original Follower",
        )

        axes[joint_idx].plot(
            steps,
            qpos_after_log[
                :, joint_idx
            ],
            label="Replay Follower",
        )

        axes[joint_idx].set_ylabel(
            f"J{joint_idx + 1} [rad]"
        )

        axes[joint_idx].grid(True)
        axes[joint_idx].legend()

    axes[-1].set_xlabel(
        "Step"
    )

    fig.suptitle(
        "Original vs Replay "
        "Joint Trajectory"
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.98]
    )

    save_path = (
        output_dir
        / "original_vs_replay_qpos.png"
    )

    fig.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        "Saved:",
        save_path,
    )


if __name__ == "__main__":
    run()