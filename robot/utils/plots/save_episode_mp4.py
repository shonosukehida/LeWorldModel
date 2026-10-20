
from pathlib import Path

import cv2
import h5py
import numpy as np


H5_PATH = Path(
    "/data/hida/flip_mug/ep200_tm300_multiview_demo/push.h5"
)

OUTPUT_DIR = H5_PATH.parent / "videos"

EPISODE_INDEX = 0
FPS = 10

CAMERAS = {
    "pixels": "overhead",
    "wrist_pixels": "wrist",
}


def to_uint8_hwc(frame):
    """Convert CHW/HWC image into uint8 HWC RGB."""

    frame = np.asarray(frame)

    if frame.ndim != 3:
        raise ValueError(f"Invalid frame shape: {frame.shape}")

    # CHW -> HWC
    if frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
        frame = np.transpose(frame, (1, 2, 0))

    if frame.dtype != np.uint8:
        frame = np.nan_to_num(frame)

        if frame.min() >= 0 and frame.max() <= 1:
            frame = frame * 255.0

        frame = np.clip(frame, 0, 255).astype(np.uint8)

    if frame.shape[-1] == 4:
        frame = frame[:, :, :3]

    if frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)

    if frame.shape[-1] != 3:
        raise ValueError(f"Invalid channel count: {frame.shape}")

    return frame


def save_video(dataset, start, end, output_path, fps):
    """Save selected frames from HDF5 to MP4."""

    first_frame = to_uint8_hwc(dataset[start])
    height, width = first_frame.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_path),
        fourcc,
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Failed to open writer: {output_path}")

    try:
        for step in range(start, end):
            frame_rgb = to_uint8_hwc(dataset[step])
            frame_bgr = cv2.cvtColor(
                frame_rgb,
                cv2.COLOR_RGB2BGR,
            )
            writer.write(frame_bgr)
    finally:
        writer.release()

    print(f"Saved: {output_path}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with h5py.File(H5_PATH, "r") as f:
        start = int(f["ep_offset"][EPISODE_INDEX])
        length = int(f["ep_len"][EPISODE_INDEX])
        end = start + length

        print(f"Episode: {EPISODE_INDEX}")
        print(f"Start: {start}")
        print(f"Length: {length}")
        print(f"FPS: {FPS}")

        for key, camera_name in CAMERAS.items():
            if key not in f:
                raise KeyError(f"Dataset not found: {key}")

            dataset = f[key]
            print(f"{key}: {dataset.shape}, {dataset.dtype}")

            output_path = (
                OUTPUT_DIR / f"episode_{EPISODE_INDEX}_{camera_name}.mp4"
            )

            save_video(
                dataset=dataset,
                start=start,
                end=end,
                output_path=output_path,
                fps=FPS,
            )


if __name__ == "__main__":
    main()
