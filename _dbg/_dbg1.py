import h5py

path = "/home/shonosukehida/.stable_worldmodel/datasets/ogbench/cube_single_expert.h5"

with h5py.File(path, "r") as f:
    def print_structure(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f"{name}: shape={obj.shape}, dtype={obj.dtype}")
    f.visititems(print_structure)