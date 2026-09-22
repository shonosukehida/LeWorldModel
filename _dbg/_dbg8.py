from pathlib import Path

import h5py


data_dir = Path(
    "/data/hida/flip_mug/ep200_tm300_multiview_demo/per_episode"
)


def episode_number(path: Path) -> int:
    return int(path.stem.split("_")[1])


episode_files = sorted(
    data_dir.glob("episode_*.h5"),
    key=episode_number,
)

bad_episodes = []

all_camera_names = set()

for file_path in episode_files:
    with h5py.File(file_path, "r") as f:

        if "sensors/cameras" not in f:
            bad_episodes.append(
                (file_path.name, [])
            )
            continue

        camera_names = list(
            f["sensors/cameras"].keys()
        )

        all_camera_names.update(camera_names)

        if len(camera_names) != 2:
            bad_episodes.append(
                (file_path.name, camera_names)
            )


print("===== Camera serial numbers found =====")
for name in sorted(all_camera_names):
    print(name)


print("\n===== Episodes with missing cameras =====")

if not bad_episodes:
    print("✅ 全episodeに2台分のカメラがあります")
else:
    for episode_name, camera_names in bad_episodes:
        print(
            f"❌ {episode_name}: "
            f"{len(camera_names)} camera(s) "
            f"{camera_names}"
        )

print(
    f"\nTotal episodes: {len(episode_files)}"
)
print(
    f"Bad episodes  : {len(bad_episodes)}"
)