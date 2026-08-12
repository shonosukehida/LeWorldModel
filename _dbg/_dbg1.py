import h5py
import imageio.v2 as imageio
import numpy as np

h5_path = "/home/hida/.stable_worldmodel/eval/flip_mug/ep200_tm300_gripper/rollout.h5"
output_path = "/home/hida/.stable_worldmodel/eval/flip_mug/ep200_tm300_gripper/rollout.mp4"

with h5py.File(h5_path, "r") as f:
    print("keys:", list(f.keys()))

    frames = f["pixels"][:]   # (T, H, W, 3)

    print("frames.shape:", frames.shape)
    print("dtype:", frames.dtype)

    with imageio.get_writer(output_path, fps=5) as writer:
        for frame in frames:
            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            writer.append_data(frame)

print("Saved:", output_path)