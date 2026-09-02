<<<<<<< HEAD
import h5py
import matplotlib.pyplot as plt

h5_path = "/home/shonosukehida/.stable_worldmodel/datasets/flip_mug/ep200_tm300_multiview/push.h5"

# push.h5 内では
# 0 -> episode_2.h5
# 1 -> episode_3.h5
episode_id = 1

with h5py.File(h5_path, "r") as f:
    start = int(f["ep_offset"][episode_id])
    length = int(f["ep_len"][episode_id])
    end = start + length

    ee_pose = f["ee_pos_quat"][start:end]

    if "episode_names" in f:
        print("episode_name:", f["episode_names"][episode_id])

print("shape:", ee_pose.shape)

for t, pose in enumerate(ee_pose):
    print(
        f"t={t:03d} | "
        f"x={pose[0]: .6f}, "
        f"y={pose[1]: .6f}, "
        f"z={pose[2]: .6f}, "
        f"qx={pose[3]: .6f}, "
        f"qy={pose[4]: .6f}, "
        f"qz={pose[5]: .6f}, "
        f"qw={pose[6]: .6f}"
    )

plt.figure(figsize=(14, 5))
plt.plot(ee_pose[:, 0], label="x")
plt.plot(ee_pose[:, 1], label="y")
plt.plot(ee_pose[:, 2], label="z")
plt.xlabel("Timestep")
plt.ylabel("Position [m]")
plt.title("episode_3.h5 - Follower EE Position")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(14, 5))
plt.plot(ee_pose[:, 3], label="qx")
plt.plot(ee_pose[:, 4], label="qy")
plt.plot(ee_pose[:, 5], label="qz")
plt.plot(ee_pose[:, 6], label="qw")
plt.xlabel("Timestep")
plt.ylabel("Quaternion")
plt.title("episode_3.h5 - Follower EE Orientation")
plt.legend()
plt.grid()
plt.show()
=======
from pathlib import Path

import cv2
import h5py
import numpy as np


H5_PATH = Path(
    "/home/hida/.stable_worldmodel/datasets/"
    "flip_mug/ep200_tm300_gripper/"
    "per_episode/episode_0.h5"
)

OUTPUT_PATH = Path(
    "/home/hida/workspace/LeWorldModel/"
    "robot/figures/episode_0.mp4"
)

FPS = 10


def find_camera_datasets(group, prefix=""):
    """sensors/cameras 以下にある画像 Dataset を列挙する。"""
    datasets = []

    for key, obj in group.items():
        path = f"{prefix}/{key}" if prefix else key

        if isinstance(obj, h5py.Dataset):
            datasets.append(
                (path, obj.shape, obj.dtype)
            )

        elif isinstance(obj, h5py.Group):
            datasets.extend(
                find_camera_datasets(
                    obj,
                    prefix=path,
                )
            )

    return datasets


def main():
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with h5py.File(H5_PATH, "r") as h5file:

        cameras = h5file[
            "sensors/cameras"
        ]

        datasets = find_camera_datasets(
            cameras
        )

        print("Camera datasets:")

        for path, shape, dtype in datasets:
            print(
                f"  {path}: "
                f"shape={shape}, "
                f"dtype={dtype}"
            )

        # --------------------------------------------------------
        # 画像 Dataset を自動選択
        #
        # 想定:
        #   (T, H, W, 3)
        # --------------------------------------------------------
        image_candidates = []

        for path, shape, dtype in datasets:
            if (len(shape) == 4 and (shape[-1] == 3 or shape[1] == 3)):
                image_candidates.append(path)

        if len(image_candidates) == 0:
            raise RuntimeError(
                "No image dataset with shape "
                "(T, H, W, 3) was found."
            )

        print()
        print(
            "Image candidates:",
            image_candidates,
        )

        # 最初のカメラを使用
        camera_path = (
            "sensors/cameras/"
            + image_candidates[0]
        )

        print(
            "Using camera:",
            camera_path,
        )

        frames = np.asarray(h5file[camera_path][:])
        if (frames.ndim == 4 and frames.shape[1] == 3):
            frames = np.transpose(frames, (0, 2, 3, 1),)

    print()
    print("frames shape:", frames.shape)
    print("frames dtype:", frames.dtype)

    num_frames, height, width, channels = (
        frames.shape
    )

    if channels != 3:
        raise ValueError(
            f"Expected RGB image, got "
            f"{channels} channels"
        )

    # ------------------------------------------------------------
    # VideoWriter
    # ------------------------------------------------------------
    fourcc = cv2.VideoWriter_fourcc(
        *"mp4v"
    )

    writer = cv2.VideoWriter(
        str(OUTPUT_PATH),
        fourcc,
        FPS,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            f"Could not open VideoWriter: "
            f"{OUTPUT_PATH}"
        )

    try:
        for frame_idx, frame in enumerate(
            frames
        ):
            frame = np.asarray(frame)

            # float画像の場合に備える
            if frame.dtype != np.uint8:
                if (
                    frame.min() >= 0.0
                    and frame.max() <= 1.0
                ):
                    frame = (
                        frame * 255.0
                    )

                frame = np.clip(
                    frame,
                    0,
                    255,
                ).astype(np.uint8)

            # Dataset が RGB なら
            # OpenCV 用に BGR へ変換
            frame_bgr = cv2.cvtColor(
                frame,
                cv2.COLOR_RGB2BGR,
            )

            writer.write(
                frame_bgr
            )

            print(
                f"\rWriting frame "
                f"{frame_idx + 1}/"
                f"{num_frames}",
                end="",
            )

    finally:
        writer.release()

    print()
    print()
    print(
        f"Saved video to: "
        f"{OUTPUT_PATH}"
    )


if __name__ == "__main__":
    main()
>>>>>>> real_robot_add_prop
