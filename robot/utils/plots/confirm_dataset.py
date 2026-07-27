import h5py

file_path = "/home/hida/.stable_worldmodel/datasets/flip_mug/ep25_tm300/push.h5"
# file_path = "/home/shonosukehida/.stable_worldmodel/datasets/franka/pairs_500_ep_1_timestep_100_sample_mix_direction_towards_bluebox_1p00_1p00_view_top_reverse1_ws_x0p45_0p85_y-0p20_0p20_z0p05_0p05/push.h5"

def print_h5_structure(name, obj):
    indent = "  " * name.count("/")
    if isinstance(obj, h5py.Group):
        print(f"{indent}[Group] {name}")
    elif isinstance(obj, h5py.Dataset):
        print(
            f"{indent}[Dataset] {name} "
            f"shape={obj.shape}, dtype={obj.dtype}"
        )

with h5py.File(file_path, "r") as f:
    print("=== HDF5 Structure ===")
    f.visititems(print_h5_structure)