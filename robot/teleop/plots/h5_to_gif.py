
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

# ==========================================
# Configuration
# ==========================================

H5_PATH = Path("/home/hida/.stable_worldmodel/datasets/flip_mug/ep200_tm300_multiview_demo/per_episode/episode_0.h5")

OUTPUT_DIR = H5_PATH.parent / "gifs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FPS = 10

CAMERAS = {
    "044322070202": "wrist",
    "937622072677": "overhead",
}


def to_uint8(frame):
    """Convert CHW float32 image to HWC uint8."""

    # CHW -> HWC
    frame = np.transpose(frame, (1, 2, 0))

    # Handle [0, 1] or [0, 255]
    if frame.max() <= 1.0 and frame.min() >= 0.0:
        frame = frame * 255.0

    frame = np.clip(frame, 0, 255).astype(np.uint8)

    return frame


def save_camera_gif(h5_file, camera_id, camera_name):
    dataset_path = f"sensors/cameras/{camera_id}"
    dataset = h5_file[dataset_path]

    print(f"Processing {camera_name}: {dataset.shape}")

    frames = []

    for i in range(dataset.shape[0]):
        frame = to_uint8(dataset[i])
        frames.append(Image.fromarray(frame))

    output_path = OUTPUT_DIR / f"{H5_PATH.stem}_{camera_name}.gif"

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=1000 // FPS,
        loop=0,
        optimize=False,
    )

    print(f"Saved: {output_path}")


def main():
    with h5py.File(H5_PATH, "r") as h5_file:
        for camera_id, camera_name in CAMERAS.items():
            save_camera_gif(h5_file, camera_id, camera_name)


if __name__ == "__main__":
    main()
