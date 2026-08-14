import h5py
import numpy as np

with h5py.File("/path/to/push.h5", "r") as file:
    action = file["action_cartesian"][:]
    ee_pos_quat = file["ee_pos_quat"][:]
    episode_idx = file["episode_idx"][:]

for key, values in {
    "action_cartesian": action[:, 3:7],
    "ee_pos_quat": ee_pos_quat[:, 3:7],
}.items():
    total_flips = 0
    min_dot = 1.0

    for episode in np.unique(episode_idx):
        quat = values[episode_idx == episode]

        dots = np.sum(
            quat[:-1] * quat[1:],
            axis=1,
        )

        total_flips += int(np.sum(dots < 0))
        min_dot = min(min_dot, float(dots.min()))

    norms = np.linalg.norm(values, axis=1)

    print(key)
    print("  total flips:", total_flips)
    print("  minimum adjacent dot:", min_dot)
    print("  norm min/max:", norms.min(), norms.max())