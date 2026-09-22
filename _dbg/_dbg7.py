import h5py
import numpy as np

file_path = "/data/hida/flip_mug/ep200_tm300_multiview_demo/per_episode/episode_35.h5"


def inspect_h5(name, obj):
    if isinstance(obj, h5py.Dataset):
        print(f"[Dataset] {name}")
        print(f"  shape : {obj.shape}")
        print(f"  dtype : {obj.dtype}")

        # 小さいデータなら先頭だけ表示
        if obj.size > 0:
            data = obj[()]
            arr = np.asarray(data)

            if arr.ndim == 0:
                print(f"  value : {arr}")
            else:
                # print(f"  first : {arr[0]}")
                pass

    elif isinstance(obj, h5py.Group):
        print(f"[Group]   {name}")


with h5py.File(file_path, "r") as f:
    print("===== HDF5 structure =====")
    f.visititems(inspect_h5)