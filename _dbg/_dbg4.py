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