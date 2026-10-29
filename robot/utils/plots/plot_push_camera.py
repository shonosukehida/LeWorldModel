
from pathlib import Path
import argparse

import cv2
import h5py
import numpy as np


def to_uint8_rgb(frame):
    """Convert CHW/HWC image to uint8 RGB."""
    frame = np.asarray(frame)

    if frame.ndim == 3 and frame.shape[0] in (1, 3, 4):
        frame = np.transpose(frame, (1, 2, 0))

    if frame.ndim != 3:
        raise ValueError(f"Invalid frame shape: {frame.shape}")

    if frame.shape[-1] == 4:
        frame = frame[..., :3]
    elif frame.shape[-1] == 1:
        frame = np.repeat(frame, 3, axis=-1)

    if frame.shape[-1] != 3:
        raise ValueError(f"Invalid channels: {frame.shape}")

    if np.issubdtype(frame.dtype, np.floating):
        if np.nanmin(frame) >= 0 and np.nanmax(frame) <= 1:
            frame = frame * 255.0

    frame = np.nan_to_num(frame, nan=0, posinf=255, neginf=0)
    return np.clip(frame, 0, 255).astype(np.uint8)


def save_first_episode(h5_path, output_path, camera="pixels", fps=10):
    h5_path = Path(h5_path)
    output_path = Path(output_path)

    with h5py.File(h5_path, "r") as f:
        offset = int(f["ep_offset"][0])
        length = int(f["ep_len"][0])

        frames = f[camera]

        first_frame = to_uint8_rgb(frames[offset])
        height, width = first_frame.shape[:2]

        output_path.parent.mkdir(parents=True, exist_ok=True)

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        if not writer.isOpened():
            raise RuntimeError("Failed to initialize VideoWriter")

        try:
            for i in range(offset, offset + length):
                rgb = to_uint8_rgb(frames[i])
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                writer.write(bgr)
        finally:
            writer.release()

    print(f"Episode: 0")
    print(f"Camera: {camera}")
    print(f"Frames: {length}")
    print(f"FPS: {fps}")
    print(f"Saved: {output_path.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--camera",
        choices=["pixels", "wrist"],
        default="pixels",
    )
    parser.add_argument("--fps", type=float, default=10.0)

    args = parser.parse_args()

    save_first_episode(
        h5_path=args.input,
        output_path=args.output,
        camera=args.camera,
        fps=args.fps,
    )
