import h5py
import matplotlib.pyplot as plt

h5_path = "/home/shonosukehida/.stable_worldmodel/datasets/flip_mug/ep200_tm300_multiview/push.h5"

# push.h5 内では
# 0 -> episode_2.h5
# 1 -> episode_3.h5
episode_id = 1

with h5py.File(h5_path, "r") as f:
    start = int(f["ep_offset"][episode_id])
    length = int(f["ep_len"][episode_id])
    end = start + length

    ee_pose = f["ee_pos_quat"][start:end]

    if "episode_names" in f:
        print("episode_name:", f["episode_names"][episode_id])

print("shape:", ee_pose.shape)

for t, pose in enumerate(ee_pose):
    print(
        f"t={t:03d} | "
        f"x={pose[0]: .6f}, "
        f"y={pose[1]: .6f}, "
        f"z={pose[2]: .6f}, "
        f"qx={pose[3]: .6f}, "
        f"qy={pose[4]: .6f}, "
        f"qz={pose[5]: .6f}, "
        f"qw={pose[6]: .6f}"
    )

plt.figure(figsize=(14, 5))
plt.plot(ee_pose[:, 0], label="x")
plt.plot(ee_pose[:, 1], label="y")
plt.plot(ee_pose[:, 2], label="z")
plt.xlabel("Timestep")
plt.ylabel("Position [m]")
plt.title("episode_3.h5 - Follower EE Position")
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(14, 5))
plt.plot(ee_pose[:, 3], label="qx")
plt.plot(ee_pose[:, 4], label="qy")
plt.plot(ee_pose[:, 5], label="qz")
plt.plot(ee_pose[:, 6], label="qw")
plt.xlabel("Timestep")
plt.ylabel("Quaternion")
plt.title("episode_3.h5 - Follower EE Orientation")
plt.legend()
plt.grid()
plt.show()