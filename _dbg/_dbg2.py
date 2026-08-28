import cv2
import os

video_path = "/home/shonosukehida/.stable_worldmodel/datasets/flip_mug/ep200_tm300_multiview/per_episode/videos/episode_0_main.mp4"

output_dir = os.path.join(
    os.path.dirname(video_path),
    "episode_0_main_frames"
)
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)

frame_idx = 0

while True:
    ret, frame = cap.read()

    if not ret:
        break

    output_path = os.path.join(
        output_dir,
        f"frame_{frame_idx:06d}.png"
    )

    cv2.imwrite(output_path, frame)
    frame_idx += 1

cap.release()

print(f"Saved {frame_idx} frames to:")
print(output_dir)