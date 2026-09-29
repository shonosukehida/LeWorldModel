"""Export camera datasets from an episode H5 file to MP4 files."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import h5py
import numpy as np


DEFAULT_H5_PATH = Path(
    "/home/hida/.stable_worldmodel/datasets/flip_mug/"
    "ep200_tm300_multiview_demo/per_episode/episode_24.h5"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export all camera streams in an episode H5 file to MP4."
    )
    parser.add_argument(
        "h5_path",
        nargs="?",
        type=Path,
        default=DEFAULT_H5_PATH,
        help="Path to an episode H5 file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to <h5 parent>/mp4.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=10.0,
        help="Output video frame rate.",
    )
    return parser.parse_args()


def frame_to_bgr(frame: np.ndarray) -> np.ndarray:
    """Convert a stored camera frame to an 8-bit BGR image for OpenCV."""
    image = np.asarray(frame)
    if image.ndim != 3:
        raise ValueError(f"Expected a 3D frame, got shape {image.shape}.")

    if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.moveaxis(image, 0, -1)

    if image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=-1)
    elif image.shape[-1] == 4:
        image = image[..., :3]
    elif image.shape[-1] != 3:
        raise ValueError(f"Expected 1, 3, or 4 channels, got shape {image.shape}.")

    image = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0)
    if image.dtype.kind == "f" and float(image.max(initial=0.0)) <= 1.0:
        image = image * 255.0
    image = np.clip(image, 0, 255).astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


def export_camera(dataset: h5py.Dataset, output_path: Path, fps: float) -> None:
    if dataset.ndim != 4:
        raise ValueError(f"Expected camera dataset with 4 dimensions, got {dataset.shape}.")

    first_frame = frame_to_bgr(dataset[0])
    height, width = first_frame.shape[:2]
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not open MP4 writer: {output_path}")

    try:
        writer.write(first_frame)
        for index in range(1, dataset.shape[0]):
            frame = frame_to_bgr(dataset[index])
            if frame.shape[:2] != (height, width):
                raise ValueError(
                    f"Frame {index} has shape {frame.shape}; "
                    f"expected {(height, width)}."
                )
            writer.write(frame)
    finally:
        writer.release()


def main() -> None:
    args = parse_args()
    h5_path = args.h5_path.expanduser().resolve()
    if not h5_path.is_file():
        raise FileNotFoundError(h5_path)
    if args.fps <= 0:
        raise ValueError("--fps must be greater than zero.")

    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else h5_path.parent / "mp4"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    with h5py.File(h5_path, "r") as handle:
        camera_group = handle.get("sensors/cameras")
        if camera_group is None:
            raise KeyError("H5 file does not contain sensors/cameras.")

        camera_names = sorted(camera_group.keys())
        if not camera_names:
            raise ValueError("No camera datasets found in sensors/cameras.")

        for camera_name in camera_names:
            output_path = output_dir / f"{camera_name}.mp4"
            export_camera(camera_group[camera_name], output_path, args.fps)
            print(
                f"wrote {output_path} "
                f"(frames={camera_group[camera_name].shape[0]}, fps={args.fps:g})"
            )


if __name__ == "__main__":
    main()