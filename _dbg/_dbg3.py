import h5py

h5_path = "/home/shonosukehida/.stable_worldmodel/datasets/flip_mug/ep200_tm300_multiview/per_episode/episode_3.h5"

with h5py.File(h5_path, "r") as f:

    def print_structure(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"[Dataset] {name}: shape={obj.shape}, dtype={obj.dtype}")
        elif isinstance(obj, h5py.Group):
            print(f"[Group]   {name}")

    f.visititems(print_structure)