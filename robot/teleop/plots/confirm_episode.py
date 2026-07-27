import h5py

path = "/data/hida/flip_mug/ep200_tm300/per_episode/episode_166.h5"


def print_h5_tree(g, indent=0):
    for key in g.keys():
        obj = g[key]

        if isinstance(obj, h5py.Group):
            print("    " * indent + f"📂 {key}/")
            print_h5_tree(obj, indent + 1)

        elif isinstance(obj, h5py.Dataset):
            print(
                "    " * indent
                + f"📄 {key} "
                + f"shape={obj.shape}, dtype={obj.dtype}"
            )


with h5py.File(path, "r") as f:
    print("HDF5 Structure")
    print("=" * 60)
    print_h5_tree(f)