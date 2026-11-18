"""Episode-local Flip Mug windows. Record t supplies the action for target frame t+1."""
import hashlib
import json
import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from normalization_stats import load_normalization_process
from stable_worldmodel.wm.diamond.action_embedding import ACTION_FIELDS, normalize_actions, validate_normalizer
from stable_worldmodel.wm.diamond.model import DEFAULT_VIEWS
from stable_worldmodel.wm.diamond.preprocessing import image_spec, pack_views
from stable_worldmodel.wm.diamond.upstream.types import Batch


def load_wm_stats(path):
    path = Path(path).expanduser()
    processes, key = load_normalization_process(path)
    if key != "action_cartesian":
        raise ValueError(f"Expected WM action_cartesian statistics, got {key}")
    scaler = processes[key]
    if scaler.mean_.shape != (1, 8) or scaler.scale_.shape != (1, 8):
        raise ValueError("Expected WM statistics with shape (1,8)")
    spec = dict(method="WM.SafeStandardScaler.v1", key=key, eps=scaler.eps, fields=ACTION_FIELDS,
                mean=scaler.mean_[0].tolist(), scale=scaler.scale_[0].tolist(),
                stats_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    validate_normalizer(spec)
    return spec


def inspect_file(path, views=DEFAULT_VIEWS):
    """Fingerprint indexes, all actions, image schema and first/last frames of each view."""
    path = Path(path).expanduser()
    image_spec(views)
    digest = hashlib.sha256()
    with h5py.File(path, "r") as f:
        keys = [*views, "action_cartesian", "proprio", "ep_offset", "ep_len", "episode_idx", "step_idx"]
        if missing := set(keys).difference(f):
            raise ValueError(f"Missing HDF5 fields {sorted(missing)}")
        schema = {k: [list(f[k].shape), str(f[k].dtype)] for k in keys}
        digest.update(json.dumps(schema, sort_keys=True).encode())
        for k in ("ep_offset", "ep_len", "episode_idx", "step_idx"):
            if f[k].ndim != 1 or f[k].dtype.kind not in "iu":
                raise ValueError(f"Invalid HDF5 index {k}")
        offsets, lengths = f["ep_offset"][:], f["ep_len"][:]
        n = int(lengths.sum())
        if (not len(lengths) or np.any(lengths <= 0)
                or not np.array_equal(offsets, np.cumsum(lengths)-lengths)):
            raise ValueError("Non-contiguous ep_offset / ep_len")
        if (f["action_cartesian"].shape != (n, 8) or f["proprio"].shape != (n, 8)
                or any(f[k].shape != (n,) for k in ("episode_idx", "step_idx"))):
            raise ValueError("Inconsistent row counts")
        for k in views:
            ds = f[k]
            if ds.ndim != 4 or ds.shape[:2] != (n, 3) or ds.shape != f[views[0]].shape:
                raise ValueError(f"Expected matching NCHW RGB in {k}")
            for i in (0, n-1):
                digest.update(ds[i].tobytes())
        labels, steps, actions = f["episode_idx"][:], f["step_idx"][:], f["action_cartesian"][:]
        if not np.isfinite(actions).all():
            raise ValueError("Non-finite recorded action")
        for arr in (offsets, lengths, labels, steps, actions):
            digest.update(arr.tobytes())
        episodes = {}
        duplicates = {}
        for start, length in zip(offsets.tolist(), lengths.tolist()):
            rows = slice(start, start+length)
            label = int(labels[start])
            if (label in episodes or np.any(labels[rows] != label)
                    or not np.array_equal(steps[rows], np.arange(length))):
                raise ValueError("Episode or step index inconsistent with offsets")
            episodes[label] = (start, length)
            signature = hashlib.sha256(actions[rows].tobytes()).hexdigest()
            duplicates.setdefault(signature, []).append(label)
    meta = dict(id=path.parent.name + "/" + path.name, fingerprint=digest.hexdigest(), schema=schema,
                fingerprint_method="sha256: schema, first/last RGB by view, indexes, all actions",
                duplicate_action_episodes=[v for v in duplicates.values() if len(v)>1])
    return meta, episodes


def split_episodes(episodes, seed, val_fraction):
    ids = sorted(episodes)
    if len(ids) < 2 or not 0 < val_fraction < 1:
        raise ValueError("Need at least two episodes and 0 < val_fraction < 1")
    shuffled = np.random.default_rng(seed).permutation(ids).tolist()
    count = max(1, min(len(ids)-1, round(len(ids)*val_fraction)))
    return dict(seed=seed, train=sorted(shuffled[count:]), val=sorted(shuffled[:count]))


class DiamondDataset(Dataset):
    def __init__(self, path, episodes, ids, normalizer, views=DEFAULT_VIEWS, frames=6, limit=None):
        self.path = Path(path).expanduser()
        self.views = tuple(views)
        image_spec(self.views)
        validate_normalizer(normalizer)
        self.normalizer = normalizer
        self.frames = frames
        if frames < 5 or (limit is not None and limit < 1):
            raise ValueError("Invalid window length or limit")
        self.windows = [(ep, start+t) for ep in ids for start, length in [episodes[ep]]
                        for t in range(length-frames+1)]
        if limit is not None:
            self.windows = self.windows[:limit]
        if not self.windows:
            raise ValueError("No complete episode-local windows")
        self._file = None
        self._pid = None

    def __len__(self):
        return len(self.windows)

    def _open(self):
        if self._file is None or self._pid != os.getpid():
            self.close()
            self._file = h5py.File(self.path, "r")
            self._pid = os.getpid()
        return self._file

    def read_actions(self, index):
        _, start = self.windows[index]
        # Last observation is a target only. Its outgoing transition is never read.
        actions = self._open()["action_cartesian"][start:start+self.frames-1]
        return normalize_actions(torch.from_numpy(actions), self.normalizer)

    def read(self, index):
        _, start = self.windows[index]
        rows = slice(start, start+self.frames)
        file = self._open()
        raw = {k: torch.from_numpy(file[k][rows]) for k in self.views}
        return raw, self.read_actions(index)

    def __getitem__(self, index):
        raw, actions = self.read(index)
        return Batch(pack_views(raw, self.views), actions, torch.ones(self.frames, dtype=torch.bool))

    def close(self):
        if self._file is not None:
            self._file.close()
            self._file = None

    def __getstate__(self):
        return dict(self.__dict__, _file=None, _pid=None)
