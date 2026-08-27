import h5py
import numpy as np
import imageio.v2 as imageio
from pathlib import Path

episode_path = Path(
    "/home/hida/.stable_worldmodel/datasets/flip_mug/ep200_tm300/per_episode/episode_3.h5"
)

output_dir = episode_path.parent / "videos"
output_dir.mkdir(parents=True, exist_ok=True)

with h5py.File(episode_path, "r") as f:
    print("=== HDF5 structure ===")

    def print_structure(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(
                f"{name}: "
                f"shape={obj.shape}, "
                f"dtype={obj.dtype}"
            )

    f.visititems(print_structure)

    camera_group = f["sensors"]["cameras"]

    print("\n=== Cameras ===")
    print("camera keys:", list(camera_group.keys()))

    for camera_name in camera_group.keys():
        frames = camera_group[camera_name][:]

        print(
            f"{camera_name}: "
            f"shape={frames.shape}, "
            f"dtype={frames.dtype}, "
            f"min={frames.min()}, "
            f"max={frames.max()}"
        )

        # 想定:
        # (T, C, H, W) -> (T, H, W, C)
        if frames.ndim == 4 and frames.shape[1] == 3:
            frames = np.transpose(frames, (0, 2, 3, 1))

        # float画像への対応
        if np.issubdtype(frames.dtype, np.floating):
            if frames.max() <= 1.0:
                frames = frames * 255.0

            frames = np.clip(
                frames,
                0,
                255,
            ).astype(np.uint8)

        elif frames.dtype != np.uint8:
            frames = np.clip(
                frames,
                0,
                255,
            ).astype(np.uint8)

        output_path = (
            output_dir
            / f"episode_0_{camera_name}.mp4"
        )

        imageio.mimwrite(
            output_path,
            frames,
            fps=10,
            codec="libx264",
            quality=8,
        )

        print("saved:", output_path)