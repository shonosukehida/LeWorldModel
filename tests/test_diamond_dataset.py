import h5py
import numpy as np
import pytest
import torch
from normalization_stats import SafeStandardScaler, save_normalization_process

from diamond_dataset import DiamondDataset, inspect_file, load_wm_stats, split_episodes
from stable_worldmodel.wm.diamond.preprocessing import pack_views


def test_episode_windows_and_alignment(tmp_path):
    p = tmp_path/"rows.h5"
    lengths = np.array([7, 8, 5])
    n = lengths.sum()
    rows = np.arange(n, dtype=np.float32)
    actions = np.broadcast_to(rows[:, None], (n, 8)).copy()
    with h5py.File(p, "w") as f:
        f["ep_len"] = lengths
        f["ep_offset"] = lengths.cumsum()-lengths
        f["episode_idx"] = np.repeat([2, 4, 9], lengths)
        f["step_idx"] = np.concatenate([np.arange(k) for k in lengths])
        f["proprio"] = actions
        f["action_cartesian"] = actions
        for i, view in enumerate(("pixels", "wrist_pixels")):
            f[view] = np.broadcast_to(rows[:, None, None, None]+i*30, (n, 3, 6, 8))
    scaler = SafeStandardScaler().fit(actions)
    s = tmp_path/"process_stats.npz"
    save_normalization_process(s, {"action_cartesian":scaler}, "action_cartesian")
    norm = load_wm_stats(s)
    meta, episodes = inspect_file(p)
    split = split_episodes(episodes, 42, 0.3)
    assert not set(split["train"]) & set(split["val"])
    ds = DiamondDataset(p, episodes, [2, 4, 9], norm)
    assert ds.windows == [(2, 0), (2, 1), (4, 7), (4, 8), (4, 9)]
    window = ds[2]
    assert torch.equal(window.obs[:, 0, 16, 16], torch.arange(7, 13)/255*2-1)
    assert torch.equal(window.act, torch.from_numpy(scaler.transform(actions[7:12])))
    assert window.act.shape == (5, 8) and window.mask_padding.all()
    raw, _ = ds.read(2)
    assert torch.equal(pack_views({k:v[:4] for k,v in raw.items()}, ds.views), window.obs[:4])
    ds.close()
    with h5py.File(p, "r+") as f:
        f["pixels"][11:13] = 255
        f["action_cartesian"][12] = -123
    changed = ds[2]
    assert torch.equal(window.obs[:4], changed.obs[:4])
    assert torch.equal(window.act, changed.act)
    assert not torch.equal(window.obs[4:], changed.obs[4:])
    ds.close()
    with h5py.File(p, "r+") as f:
        f["step_idx"][7] = 1
    with pytest.raises(ValueError):
        inspect_file(p)
