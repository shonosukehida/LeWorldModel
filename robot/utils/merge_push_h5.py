#!/usr/bin/env python3
"""Safely concatenate LeWorldModel flattened-time push.h5 files.

Usage:
    1. Edit INPUT_FILES and OUTPUT_FILE below.
    2. Run: python merge_push_h5.py

Inputs are never modified. Existing outputs are never overwritten. Requires
h5py and numpy. Dataset arrays are copied in bounded blocks (not loaded all
at once); a partially written output is deleted if an error occurs.

Supported input layout is LeWorldModel's flattened_time format, where
``ep_len`` and ``ep_offset`` delimit episodes. All other top-level datasets
must be per-step arrays (first axis = total steps), except ``episode_names``
(per-episode strings). The generated index datasets are rebuilt from ep_len.
Unknown structures are rejected rather than silently corrupted.
"""
from __future__ import annotations

import os
from pathlib import Path
from contextlib import ExitStack

import h5py
import numpy as np

# ===== Edit these paths before running =====
# Input order is preserved in the merged dataset.
INPUT_FILES = [
    Path("/data/hida/flip_mug/ep200_tm300_multiview_demo/push.h5"),
    Path("/data/hida/flip_mug/ep100_tm300_multiview_play_normal_brightness/push.h5"),
]
OUTPUT_FILE = Path("/data/hida/flip_mug/ep300_tm300_multiview_demo_play_normal/push.h5")
BLOCK_ROWS = 32  # Number of rows to process per chunk (keeps memory use low).
# ==========================================

BOUNDARY = {"ep_len", "ep_offset"}
REBUILT = {"episode_idx", "step_idx"}
EPISODE_FIELDS = {"episode_names"}
PROVENANCE_ATTRS = {"source_directory", "robot_ip_used_for_fk"}


def _attr_equal(a, b):
    return np.array_equal(np.asarray(a), np.asarray(b))


def _validate(f: h5py.File, path: Path) -> tuple[np.ndarray, int, int, dict]:
    if "ep_len" not in f or "ep_offset" not in f:
        raise ValueError(f"{path}: missing ep_len/ep_offset")
    for name in f:
        if not isinstance(f[name], h5py.Dataset):
            raise ValueError(f"{path}: groups unsupported: {name!r}")
    lengths = np.asarray(f["ep_len"][:]); offsets = np.asarray(f["ep_offset"][:])
    if lengths.ndim != 1 or offsets.shape != lengths.shape or not np.issubdtype(lengths.dtype, np.integer) or not np.issubdtype(offsets.dtype, np.integer):
        raise ValueError(f"{path}: invalid ep_len/ep_offset types or shapes")
    if len(lengths) == 0 or np.any(lengths <= 0):
        raise ValueError(f"{path}: zero episodes or nonpositive episode length")
    expected_offsets = np.concatenate(([0], np.cumsum(lengths[:-1], dtype=np.int64)))
    if not np.array_equal(offsets, expected_offsets):
        raise ValueError(f"{path}: inconsistent ep_offset")
    nstep = int(np.sum(lengths, dtype=np.int64)); neps = len(lengths)
    if f.attrs.get("storage_layout", "flattened_time") != "flattened_time":
        raise ValueError(f"{path}: unsupported storage_layout")
    if "num_episodes" in f.attrs and int(f.attrs["num_episodes"]) != neps:
        raise ValueError(f"{path}: num_episodes attribute mismatch")
    fields = {}
    for name in f:
        if name in BOUNDARY or name in REBUILT:
            continue
        ds = f[name]
        if ds.ndim == 0:
            raise ValueError(f"{path}: unsupported scalar dataset: {name}")
        if name in EPISODE_FIELDS:
            if ds.shape != (neps,):
                raise ValueError(f"{path}: {name} must have shape ({neps},)")
            fields[name] = "episode"
        else:
            if ds.shape[0] != nstep:
                raise ValueError(f"{path}: {name} has {ds.shape[0]} rows, expected {nstep} time steps")
            fields[name] = "step"
    for name in REBUILT & set(f.keys()):
        ds = f[name]
        if ds.shape != (nstep,) or not np.issubdtype(ds.dtype, np.integer):
            raise ValueError(f"{path}: invalid {name}")
    return lengths.astype(np.int64), neps, nstep, fields


def _same_dtype(a, b) -> bool:
    # Use strict HDF5 dtype metadata checks; variable-length UTF-8 requires
    # h5py.check_string_dtype to handle its object dtype.
    sa, sb = h5py.check_string_dtype(a), h5py.check_string_dtype(b)
    if sa or sb:
        return sa == sb
    return a == b


def _copy_blocks(src, dst, start, count, rows):
    for offset in range(0, count, rows):
        end = min(offset + rows, count)
        dst[start + offset:start + end] = src[offset:end]


def merge(inputs: list[Path], output: Path, block_rows: int = 32):
    if len(inputs) < 2:
        raise ValueError("At least two input files are required")
    if block_rows <= 0:
        raise ValueError("block_rows must be positive")
    inputs = [p.expanduser().resolve(strict=True) for p in inputs]
    output = output.expanduser().resolve()
    if len(set(inputs)) != len(inputs):
        raise ValueError("Duplicate input paths")
    if output in inputs:
        raise ValueError("Output path must not be an input file")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    if not output.parent.is_dir():
        raise FileNotFoundError(f"Output directory does not exist: {output.parent}")

    with ExitStack() as stack:
        files = [stack.enter_context(h5py.File(p, "r")) for p in inputs]
        info = [_validate(f, p) for f, p in zip(files, inputs)]
        first = files[0]
        base_fields = info[0][3]
        for f, path, (_, _, _, fields) in zip(files[1:], inputs[1:], info[1:]):
            if fields != base_fields:
                raise ValueError(f"{path}: dataset key/layout mismatch: {fields.keys()} vs {base_fields.keys()}")
            for name in base_fields:
                a, b = first[name], f[name]
                if a.shape[1:] != b.shape[1:] or not _same_dtype(a.dtype, b.dtype):
                    raise ValueError(f"{path}: incompatible shape/dtype for {name}")
                if set(a.attrs) != set(b.attrs) or any(not _attr_equal(a.attrs[k], b.attrs[k]) for k in a.attrs):
                    raise ValueError(f"{path}: inconsistent dataset attributes on {name}")
            a_attrs = {k for k in first.attrs if k not in PROVENANCE_ATTRS | {"num_episodes"}}
            b_attrs = {k for k in f.attrs if k not in PROVENANCE_ATTRS | {"num_episodes"}}
            if a_attrs != b_attrs or any(not _attr_equal(first.attrs[k], f.attrs[k]) for k in a_attrs):
                raise ValueError(f"{path}: incompatible file attributes; refusing to mix different formats/cameras")

        total_eps = sum(x[1] for x in info)
        total_steps = sum(x[2] for x in info)
        if total_steps > np.iinfo(np.int32).max or total_eps > np.iinfo(np.int32).max:
            raise ValueError("Episode/step indices exceed int32 range")
        print(f"Merging {len(inputs)} inputs: {total_eps} episodes, {total_steps} steps")
        # Exclusive create prevents overwrites even if a concurrent process
        # creates the output after our check.
        try:
            with h5py.File(output, "x") as dest:
                for k, v in first.attrs.items():
                    if k not in PROVENANCE_ATTRS and k != "num_episodes":
                        dest.attrs[k] = v
                dest.attrs["num_episodes"] = total_eps
                dest.attrs["source_files"] = np.asarray([str(p) for p in inputs], dtype=h5py.string_dtype("utf-8"))
                dest.attrs["merge_tool"] = "merge_push_h5.py"
                dest.create_dataset("ep_len", shape=(total_eps,), dtype="i8")
                dest.create_dataset("ep_offset", shape=(total_eps,), dtype="i8")
                dest.create_dataset("episode_idx", shape=(total_steps,), dtype="i4", chunks=(min(block_rows * 4, total_steps),))
                dest.create_dataset("step_idx", shape=(total_steps,), dtype="i4", chunks=(min(block_rows * 4, total_steps),))
                for name, kind in base_fields.items():
                    ds = first[name]
                    size = total_eps if kind == "episode" else total_steps
                    # Preserve dtype and original compression/chunks when practical.
                    kwargs = {"dtype": ds.dtype, "shape": (size, *ds.shape[1:])}
                    if ds.chunks is not None:
                        kwargs.update(chunks=(min(ds.chunks[0], size), *ds.chunks[1:]), compression=ds.compression,
                                      compression_opts=ds.compression_opts, shuffle=ds.shuffle,
                                      fletcher32=ds.fletcher32, scaleoffset=ds.scaleoffset)
                    out_ds = dest.create_dataset(name, **kwargs)
                    for k, v in ds.attrs.items():
                        out_ds.attrs[k] = v

                ep_base = step_base = 0
                for path, f, (lengths, neps, nstep, _) in zip(inputs, files, info):
                    dest["ep_len"][ep_base: ep_base+neps] = lengths
                    dest["ep_offset"][ep_base:ep_base+neps] = np.cumsum(
                        np.r_[step_base, lengths[:-1]], dtype=np.int64
                    )
                    # Small blocks avoid loading huge video datasets into memory.
                    for name, kind in base_fields.items():
                        count = neps if kind == "episode" else nstep
                        offset = ep_base if kind == "episode" else step_base
                        _copy_blocks(f[name], dest[name], offset, count, block_rows)
                    local_start = step_base
                    for j, length in enumerate(lengths):
                        length = int(length)
                        start = local_start
                        for p in range(0, length, block_rows):
                            stop = min(p + block_rows, length)
                            dest["episode_idx"][start+p:start+stop] = ep_base+j
                            dest["step_idx"][start+p:start+stop] = np.arange(p, stop, dtype=np.int32)
                        local_start += length
                    ep_base += neps
                    step_base += nstep
                    print(f"  Added {path.name}: {neps} episodes / {nstep} steps")
                assert ep_base == total_eps and step_base == total_steps
                dest.flush()
        except Exception:
            # Do not leave an incomplete output for downstream training.
            if output.exists():
                output.unlink()
            raise
    print(f"Created: {output}")
    with h5py.File(output, "r") as f:
        lengths = f["ep_len"][:]
        offsets = f["ep_offset"][:]
        if len(lengths) != total_eps or int(lengths.sum()) != total_steps or not np.array_equal(
            offsets, np.r_[0, np.cumsum(lengths[:-1])]
        ):
            raise RuntimeError("Post-write boundary verification failed")
    print("Boundary verification passed")


def main():
    """Merge the file paths configured at the top of this script."""
    merge(INPUT_FILES, OUTPUT_FILE, BLOCK_ROWS)


if __name__ == "__main__":
    main()
