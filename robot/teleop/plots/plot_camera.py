"""Convert camera frames in an xArm HDF5 episode to MP4."""

from pathlib import Path

import cv2
import h5py
import numpy as np
from numpy.typing import NDArray


def to_uint8_rgb(frame: NDArray[np.generic]) -> NDArray[np.uint8]:
    """Convert one CHW or HWC frame to uint8 RGB."""
    frame = np.asarray(frame)

    # CHW -> HWC
    if frame.ndim == 3 and frame.shape[0] in (1, 3, 4):
        frame = np.transpose(frame, (1, 2, 0))

    if frame.ndim != 3:
        raise ValueError(f"Expected 3D frame, but got shape={frame.shape}")

    # RGBA -> RGB
    if frame.shape[2] == 4:
        frame = frame[:, :, :3]

    # Grayscale -> RGB
    if frame.shape[2] == 1:
        frame = np.repeat(frame, 3, axis=2)

    if frame.shape[2] != 3:
        raise ValueError(
            f"Expected 3 channels after conversion, but got shape={frame.shape}"
        )

    # float32画像が [0, 1] の場合
    if np.issubdtype(frame.dtype, np.floating):
        frame_min = float(np.nanmin(frame))
        frame_max = float(np.nanmax(frame))

        if 0.0 <= frame_min and frame_max <= 1.0:
            frame = frame * 255.0

        frame = np.nan_to_num(
            frame,
            nan=0.0,
            posinf=255.0,
            neginf=0.0,
        )

    frame = np.clip(frame, 0, 255).astype(np.uint8)

    return frame


def hdf5_camera_to_mp4(
    h5_path: str | Path,
    output_path: str | Path,
    camera_name: str = "main",
    fps: float = 10.0,
) -> None:
    """Save HDF5 camera frames as an MP4 video.

    Args:
        h5_path:
            Input HDF5 episode path.
        output_path:
            Output MP4 path.
        camera_name:
            Camera key under sensors/cameras.
        fps:
            Output video frame rate.
    """
    h5_path = Path(h5_path)
    output_path = Path(output_path)

    if not h5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {h5_path}")

    if fps <= 0:
        raise ValueError("fps must be greater than 0")

    dataset_key = f"sensors/cameras/{camera_name}"

    with h5py.File(h5_path, "r") as h5_file:
        if dataset_key not in h5_file:
            available = []

            if "sensors/cameras" in h5_file:
                available = list(h5_file["sensors/cameras"].keys())

            raise KeyError(
                f"Camera dataset '{dataset_key}' was not found. "
                f"Available cameras: {available}"
            )

        frames = h5_file[dataset_key]

        if frames.ndim != 4:
            raise ValueError(
                f"Expected camera data with 4 dimensions, "
                f"but got shape={frames.shape}"
            )

        if frames.shape[0] == 0:
            raise ValueError("Camera dataset contains no frames")

        first_rgb = to_uint8_rgb(frames[0])
        height, width = first_rgb.shape[:2]

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # mp4vは比較的広く利用可能
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_path),
            fourcc,
            fps,
            (width, height),
        )

        if not writer.isOpened():
            raise RuntimeError(
                "Failed to open VideoWriter. "
                "Check whether OpenCV has MP4 codec support."
            )

        try:
            for frame_idx in range(frames.shape[0]):
                rgb_frame = to_uint8_rgb(frames[frame_idx])

                if rgb_frame.shape[:2] != (height, width):
                    raise ValueError(
                        f"Frame {frame_idx} has inconsistent shape: "
                        f"{rgb_frame.shape}"
                    )

                # OpenCV VideoWriterはBGRを想定
                bgr_frame = cv2.cvtColor(
                    rgb_frame,
                    cv2.COLOR_RGB2BGR,
                )

                writer.write(bgr_frame)

        finally:
            writer.release()

    print(f"Input:  {h5_path}")
    print(f"Camera: {dataset_key}")
    print(f"Frames: {frames.shape[0]}")
    print(f"Size:   {width}x{height}")
    print(f"FPS:    {fps}")
    print(f"Saved:  {output_path.resolve()}")


if __name__ == "__main__":
    hdf5_camera_to_mp4(
        h5_path=(
            "/home/shonosukehida/.stable_worldmodel/datasets/flip_mug/ep200_tm300/per_episode/episode_20260715_162509.h5"
        ),
        output_path=(
            "./robot/teleop/plots/figures/episode_main.mp4"
        ),
        camera_name="main",
        fps=10.0,
    )