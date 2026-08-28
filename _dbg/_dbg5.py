import h5py
import numpy as np
import cv2

h5_path = "/home/shonosukehida/.stable_worldmodel/datasets/flip_mug/ep200_tm300_multiview/push.h5"
output_path = "/home/shonosukehida/.stable_worldmodel/datasets/flip_mug/ep200_tm300_multiview/episode_1_main.mp4"

episode_id = 1
fps = 10

with h5py.File(h5_path, "r") as f:
    start = int(f["ep_offset"][episode_id])
    length = int(f["ep_len"][episode_id])
    end = start + length

    frames = f["pixels"][start:end]

    if "episode_names" in f:
        print("episode_name:", f["episode_names"][episode_id])

print("frames shape:", frames.shape)


# (T, C, H, W) -> (T, H, W, C)
frames = np.transpose(frames, (0, 2, 3, 1))

print("converted shape:", frames.shape)
print("dtype:", frames.dtype)
print("min:", frames.min())
print("max:", frames.max())

# float32 で 0〜1 に正規化されている場合
if frames.dtype != np.uint8:
    if frames.max() <= 1.0:
        frames = frames * 255.0

    frames = np.clip(frames, 0, 255).astype(np.uint8)

height, width = frames.shape[1:3]

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    output_path,
    fourcc,
    fps,
    (width, height),
)

for frame in frames:
    # HDF5内がRGBならOpenCV用にBGRへ変換
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    writer.write(frame_bgr)

writer.release()

print(f"Saved {len(frames)} frames")
print(f"Saved video to: {output_path}")