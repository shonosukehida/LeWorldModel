
from pathlib import Path
from tqdm import tqdm
import cv2
import h5py
import numpy as np


# ============================================================
# Config
# ============================================================

H5_PATH = Path(
    "/home/shonosukehida/.stable_worldmodel/datasets/flip_mug/val_ep10_tm300_multiview_play/push.h5"
)

EPISODE_INDEX = 1  # 0-based index

CAMERA = "pixels"  # pixels: overhead, wrist: wrist camera

FPS = 10.0

OUTPUT_DIR = Path(
    "/home/shonosukehida/work/LeWorldModel/robot/utils/plots/figure"
)

# ============================================================


def to_uint8_rgb(frame):
    """Convert CHW/HWC image to uint8 RGB."""
    frame = np.asarray(frame)

    if frame.ndim != 3:
        raise ValueError(f"Invalid frame shape: {frame.shape}")

    if frame.shape[0] in (1, 3, 4) and frame.shape[-1] not in (1, 3, 4):
        frame = np.transpose(frame, (1, 2, 0))

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


def save_episode_video():
    if not H5_PATH.is_file():
        raise FileNotFoundError(H5_PATH)

    if FPS <= 0:
        raise ValueError("FPS must be positive")

    with h5py.File(H5_PATH, "r") as f:
        ep_offsets = f["ep_offset"]
        ep_lengths = f["ep_len"]

        num_episodes = len(ep_lengths)

        if not 0 <= EPISODE_INDEX < num_episodes:
            raise IndexError(
                f"Invalid episode index: {EPISODE_INDEX}. "
                f"Available: 0 to {num_episodes - 1}"
            )

        if CAMERA not in f:
            raise KeyError(f"Camera '{CAMERA}' not found")

        offset = int(ep_offsets[EPISODE_INDEX])
        length = int(ep_lengths[EPISODE_INDEX])

        frames = f[CAMERA]

        if length <= 0 or offset < 0 or offset + length > len(frames):
            raise ValueError(
                f"Invalid episode range: offset={offset}, "
                f"length={length}, total_frames={len(frames)}"
            )

        first_frame = to_uint8_rgb(frames[offset])
        height, width = first_frame.shape[:2]

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        output_path = (
            OUTPUT_DIR / f"episode_{EPISODE_INDEX}_{CAMERA}.mp4"
        )

        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            FPS,
            (width, height),
        )

        if not writer.isOpened():
            raise RuntimeError(
                f"Failed to initialize VideoWriter: {output_path}"
            )

        try:

            for i in tqdm(
                range(offset, offset + length),
                total=length,
                desc=f"Saving episode {EPISODE_INDEX}",
                unit="frame",
            ):

                rgb = to_uint8_rgb(frames[i])
                bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                writer.write(bgr)
        finally:
            writer.release()

    print(f"Episode: {EPISODE_INDEX} / {num_episodes - 1}")
    print(f"Camera: {CAMERA}")
    print(f"Offset: {offset}")
    print(f"Frames: {length}")
    print(f"FPS: {FPS}")
    print(f"Resolution: {width}x{height}")
    print(f"Saved: {output_path.resolve()}")


if __name__ == "__main__":
    save_episode_video()
