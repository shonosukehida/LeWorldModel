import h5py
import numpy as np

path = "/home/shonosukehida/.stable_worldmodel/datasets/franka/pairs_500_ep_1_timestep_500_sample_mix_direction_towards_bluebox_1p00_1p00_view_top_reverse/push.h5"

with h5py.File(path, "r") as f:

    print("=== Keys ===")
    for key in f.keys():
        print(key)

    print("\n=== Dataset Info ===")
    for key in f.keys():
        obj = f[key]

        # Dataset
        if isinstance(obj, h5py.Dataset):
            print(f"\n[{key}]")
            print("shape :", obj.shape)
            print("dtype :", obj.dtype)

            # 少しだけ中身を見る
            if obj.ndim == 1:
                print("head  :", obj[:5])

            else:
                print("head shape :", obj[:5].shape)
                print(obj[:5])

        # Group
        elif isinstance(obj, h5py.Group):
            print(f"\n[{key}] (Group)")
            print("subkeys :", list(obj.keys()))