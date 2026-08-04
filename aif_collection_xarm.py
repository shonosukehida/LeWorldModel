#!/usr/bin/env python3
"""AIF data collection on real xArm7 + RealSense.

Hardware analogue of ``scripts/aif_collection_uv.py``: each episode
resets the arm, finds an affordance target in the static camera, moves
to it via xArm Cartesian motion, then runs the trained policy in a
receding-horizon loop while saving per-step observations, actions, and
affordance masks. Outputs land in the same per-episode blosc2 layout
that `aif_collection_uv.py` writes, so downstream training picks them
up unchanged.

Key differences vs. the CALVIN-sim version:
  * The env is `XArmRealEnv` (this file), wrapping robopy's `XArmFollower`
    plus two `RealsenseCamera` instances. No teleop layer.
  * Static + gripper cam extrinsics are loaded from a `camera_info.npz`
    produced by `scripts/calibrate_handeye_xarm{,_dual}.py`.
  * `--use_data_initial_state` is not ported (real hardware can't
    teleport); episodes always start from `start_joints`.
  * Both `actions.blosc2` (absolute EE pose) and `rel_actions.blosc2`
    (per-step EE delta) are always written regardless of the policy's
    `action_type`, so the dataset is symmetric for downstream training.

Affordance viz matches `aif_collection_uv.py` — same renderer, same file
names, same layout, all inside the episode directory and gated on
`--save_video`:
  * `<episode>/affordance_init/static_{orig,masks,aff,dirs,selected}.png`
    — episode-start static-cam detection.
  * `<episode>/gripper/rgb_gripper_aff_dirs.mp4` — per-step gripper-cam
    affordance direction field, alongside `rgb_gripper.mp4`.
The `--debug_affordance` modes (no uv counterpart) write the same
`_{orig,masks,aff,dirs}.png` set per frame via the same renderer.

Example::

    uv run python scripts/aif_collection_xarm.py \\
        --policy_cfg act-zprior --world_cfg rssm_s-dec_no-sphery \\
        --num_episodes 5 --episode_length 100 \\
        --calc_efe --calc_efe_every 8 --num_candidate_policies 8 \\
        --extrinsics datasets/real_world/xArm/camera_info.npz \\
        --follower_ip 192.168.1.240 \\
        --action_type rel_actions

Safety: keep the E-stop within reach, verify `XArmWorkspaceBounds`
defaults match the cell, and start with the arm in a posture close to
`--start_joints_deg`.
"""
from __future__ import annotations

import argparse
import glob
import logging
import os
import signal
import sys
import threading
import time
import traceback as _tb
import typing as _typing
from collections import deque
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# robopy (dev/add-xarm-support) imports ``typing.override`` which only exists
# on Python 3.12+. Shim it in from ``typing_extensions`` before any robopy
# import. (Same shim used by gello_teleop_recorder_xarm.py.)
if not hasattr(_typing, "override"):
    from typing_extensions import override as _override
    _typing.override = _override  # type: ignore[attr-defined]

# robopy renamed ``robopy.motor.control_table`` → ``dynamixel_control_table``
# but several xarm modules still import the old name. Alias before any
# robopy import so the package init can complete.
if "robopy.motor.control_table" not in sys.modules:
    try:
        from robopy.motor import dynamixel_control_table as _ct  # type: ignore
        sys.modules["robopy.motor.control_table"] = _ct
    except Exception:
        pass

# Disable JAX/XLA preallocation BEFORE jax is imported (via flax / hydra /
# direct jax import below). Without this, every JAX-backed model
# (affordance + world + policy) tries to grab ~90% of the GPU on first
# use, which on this multi-model script trips the BFC allocator into
# requesting absurd contiguous blocks (e.g. 8+ GiB) and OOM-ing. The
# settings below mirror what scripts/aif_collection_uv.py already uses
# and let JAX grow its arena on demand.
os.environ.setdefault("XLA_FLAGS",
                      "--xla_gpu_strict_conv_algorithm_picker=false")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")
os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.85")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# Compact tracebacks (rich's verbose locals-display is unhelpful for hardware
# loops; we want a short stack so we can read the actual fault line).
def _compact_excepthook(exc_type, exc_value, exc_tb):
    _tb.print_exception(exc_type, exc_value, exc_tb, limit=None)


sys.excepthook = _compact_excepthook
import rich.traceback
rich.traceback.install = lambda *a, **kw: None

import cv2
from flax import nnx
import hydra
from hydra.utils import instantiate
import imageio
import jax
jax.numpy.set_printoptions(threshold=10, edgeitems=2)
import jax.numpy as jnp
import numpy as np
np.set_printoptions(threshold=10, edgeitems=2)
from omegaconf import OmegaConf
from rich.progress import track
import torch

import ml_networks as ml
from ml_networks import load_blosc2, save_blosc2
from ml_networks.jax import jax_fix_seed
from scipy.spatial.transform import Rotation as _R

from vapo.affordance.utils.img_utils import (
    get_transforms,
    resize_center,
    transform_and_predict,
    viz_aff_centers_preds,
)
from vapo.affordance.utils.static_crop import (
    box_for_shape,
    box_to_fractions,
    crop_frame_hwc,
    load_crop_info,
)
from vapo.policy.config import EBMPolicyConfig, VQBeTConfig, IMLEConfig
from vapo.policy.dataset import (
    delta_chunk_to_rel_norm,
    joint_transform,
    joint_detransform,
    joint_detransform_mixed,
    setup_delta_action_eval,
)
from vapo.utils.rel_actions import split_rel_action
from vapo.policy.policy import (
    ACTConfig, ACTPolicy, DiTConfig, EBMPolicy, IMLEPolicy, Policy, RNNConfig,
    SimplePolicy, TransformerConfig, TransformerPolicy, UNetConfig, VQBeTPolicy,
)
from vapo.utils.hydra_config import setup_console_logging
from vapo.utils.utils import affordance_cli_overrides, init_aff_net
from vapo.world.modules import WorldModel
from vapo.wrappers.utils import depth_preprocessing, get_transforms_and_shape

logger = logging.getLogger("aif_collection_xarm")


# ────────────────────────────────────────────────── step profiler ──
# `--profile` turns this on. It answers "why is the loop slower than
# --fps?" by splitting each recorded frame into camera / IK / affordance /
# policy / EFE / world / viz / sleep, so the dominant stage is visible on
# the real robot instead of guessed at.


class _Stage:
    """One `with _PROF.stage(...)` scope. See :class:`_StageProfiler`."""

    __slots__ = ("prof", "name", "t0", "child")

    def __init__(self, prof, name):
        self.prof = prof
        self.name = name
        self.child = 0.0

    def __enter__(self):
        self.t0 = time.perf_counter()
        self.prof._stack.append(self)
        return self

    def __exit__(self, *_exc):
        dt = time.perf_counter() - self.t0
        stack = self.prof._stack
        stack.pop()
        if stack:
            # Nested scope: bill our whole span to the parent's child time so
            # the parent records EXCLUSIVE time and nothing is double-counted.
            stack[-1].child += dt
        self.prof._record(self.name, dt - self.child)
        return False


class _NullStage:
    """Shared no-op scope used when profiling is off (no allocation)."""

    __slots__ = ()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        return False


_NULL_STAGE = _NullStage()


class _StageProfiler:
    """Per-stage wall-clock accounting for the collection loop.

    Usage::

        _PROF.start(target_hz=30.0)         # once, before the step loop
        ...
        with _PROF.stage("cam_static"):
            rgb, depth = cam.get_image()
        ...
        _PROF.mark_step(tag=f"ep {n} t={t}")   # once per recorded frame
        ...
        _PROF.report_episode(tag=f"ep {n}")    # once, after the step loop

    Times are **exclusive**: when a scope nests inside another (e.g. the
    static-cam read that ``_select_nearest_efe_target`` triggers inside the
    EFE scope) the inner span is subtracted from the outer one, so the
    columns sum to the measured step time and ``(unaccounted)`` is the
    genuinely uninstrumented remainder.

    Disabled by default, in which case ``stage()`` returns a shared no-op
    object — cheap enough to leave the instrumentation in the hot loop.

    JAX caveat: JAX dispatches asynchronously, so a scope only accounts for
    work that has actually been forced by the time it exits. The policy /
    EFE / world scopes wrap their results in :func:`_jax_sync` (a no-op when
    profiling is off) so they report real compute rather than dispatch time.
    """

    def __init__(self):
        self.enabled = False
        self.every = 30
        self._stack: List[_Stage] = []
        self._win: Dict[str, List[float]] = {}     # name -> per-call seconds
        self._tot: Dict[str, List[float]] = {}
        self._win_steps = 0
        self._tot_steps = 0
        self._win_time = 0.0
        self._tot_time = 0.0
        self._last_mark: Optional[float] = None
        self.target_hz = 0.0

    def enable(self, every: int = 30) -> None:
        self.enabled = True
        self.every = max(1, int(every))

    def start(self, target_hz: float = 0.0) -> None:
        """Anchor the frame clock (call once, just before the step loop).

        ``target_hz`` is the rate of *recorded frames* the loop is pacing to
        — i.e. ``--fps`` times the sub-step factor, not ``--fps`` itself.

        Also drops anything recorded before the anchor. Episode setup
        (``affordance_random_init`` → target search, the pre-loop
        ``_compute_step_obs``) runs scopes too, but its wall clock is not in
        any frame — counting those samples is what made ``(unaccounted)``
        come out negative.
        """
        if self.enabled:
            self.target_hz = float(target_hz)
            self._win, self._tot = {}, {}
            self._win_steps = self._tot_steps = 0
            self._win_time = self._tot_time = 0.0
            self._stack.clear()
            self._last_mark = time.perf_counter()

    def stage(self, name: str):
        if not self.enabled:
            return _NULL_STAGE
        return _Stage(self, name)

    def _record(self, name: str, dt: float) -> None:
        if self._last_mark is None:
            return          # clock not running (episode setup) — see start()
        self._win.setdefault(name, []).append(dt)
        self._tot.setdefault(name, []).append(dt)

    def mark_step(self, tag: str = "") -> None:
        """Close one recorded frame; emit a window report every ``every`` steps.

        The frame duration is the wall-clock delta since the previous mark,
        NOT the sub-step's own span — that way the per-policy-step work that
        runs between sub-frames (policy inference, EFE, world dynamics) is
        inside the denominator instead of vanishing from the accounting.
        """
        if not self.enabled:
            return
        now = time.perf_counter()
        if self._last_mark is None:      # no anchor yet: just start the clock
            self._last_mark = now
            return
        step_seconds = now - self._last_mark
        self._last_mark = now
        self._win_steps += 1
        self._tot_steps += 1
        self._win_time += step_seconds
        self._tot_time += step_seconds
        if self._win_steps >= self.every:
            self._emit(tag, self._win, self._win_steps, self._win_time, "window")
            self._win = {}
            self._win_steps = 0
            self._win_time = 0.0

    def report_episode(self, tag: str = "") -> None:
        """Emit the whole-episode summary and reset all accumulators."""
        if not self.enabled or self._tot_steps == 0:
            return
        self._emit(tag, self._tot, self._tot_steps, self._tot_time, "episode")
        self._win, self._tot = {}, {}
        self._win_steps = self._tot_steps = 0
        self._win_time = self._tot_time = 0.0
        self._last_mark = None
        self._stack.clear()

    def _emit(self, tag, acc, steps, total_s, scope) -> None:
        if steps <= 0:
            return
        ms_per_step = total_s / steps * 1000.0
        achieved = steps / total_s if total_s > 0 else float("inf")
        target = (f" / target {self.target_hz:.1f} Hz"
                  if self.target_hz > 0 else "")
        logger.info(
            f"[profile:{scope}] {tag} | {achieved:.1f} Hz achieved{target} | "
            f"{ms_per_step:.1f} ms/step over {steps} steps"
        )
        logger.info(
            f"[profile:{scope}]   {'stage':<16}{'calls/step':>11}"
            f"{'p50 ms':>9}{'mean ms':>9}{'max ms':>9}{'ms/step':>9}{'%':>7}"
        )
        rows = sorted(acc.items(), key=lambda kv: sum(kv[1]), reverse=True)
        accounted = 0.0
        for name, samples in rows:
            tot = sum(samples)
            calls = len(samples)
            accounted += tot
            per_step_ms = tot / steps * 1000.0
            ordered = sorted(samples)
            p50 = ordered[len(ordered) // 2]
            logger.info(
                f"[profile:{scope}]   {name:<16}{calls / steps:>11.2f}"
                f"{p50 * 1000.0:>9.1f}{tot / calls * 1000.0:>9.1f}"
                f"{ordered[-1] * 1000.0:>9.1f}"
                f"{per_step_ms:>9.1f}{per_step_ms / ms_per_step * 100.0:>7.1f}"
            )
        rest_ms = (total_s - accounted) / steps * 1000.0
        logger.info(
            f"[profile:{scope}]   {'(unaccounted)':<16}{'-':>11}{'-':>9}{'-':>9}"
            f"{'-':>9}{rest_ms:>9.1f}{rest_ms / ms_per_step * 100.0:>7.1f}"
        )
        # A stage whose mean sits far above its p50 was dominated by a few
        # outliers — on the first episode of a run that is JAX tracing the
        # policy / EFE / affordance graphs, not steady-state cost.
        for name, samples in rows:
            if len(samples) >= 4:
                p50 = sorted(samples)[len(samples) // 2]
                mean = sum(samples) / len(samples)
                if p50 > 0 and mean > 5.0 * p50:
                    logger.info(
                        f"[profile:{scope}]   note: '{name}' mean is "
                        f"{mean / p50:.0f}x its p50 — one-off cost (JIT "
                        f"compile?) dominates; read p50 for steady state."
                    )


_PROF = _StageProfiler()


def _jax_sync(x):
    """Force pending JAX work so the enclosing profiler scope measures real
    compute instead of async dispatch. No-op unless ``--profile`` is on."""
    if _PROF.enabled:
        try:
            jax.block_until_ready(x)
        except Exception:
            pass
    return x


# ─────────────────────────────────────────────────────────── constants ──

# Fallback xArm7 home joints used when no real hardware is available
# (``--dry_run``). For real-hardware runs the *default* home is the pose
# the robot is in at script-launch time -- captured inside
# ``XArmRealEnv.__init__`` -- so the user just positions the arm where
# they want home before launching and that pose is used for the entire
# session (no GELLO joint-limit incompatibilities).
#
# xArm7 joint limits (rad, approximate):
#   j1: ±2π   j2: -2.059..2.094   j3: ±π   j4: -0.07..3.93
#   j5: ±π    j6: -1.69..π        j7: ±π
DRY_RUN_FALLBACK_JOINTS_DEG = np.array([0, -45, 0, 45, 0, 45, 0], dtype=np.float32)
DEFAULT_ENV_CAMERA_SERIAL = "937622072677"
DEFAULT_WRIST_CAMERA_SERIAL = "044322070202"
GRIPPER_OPEN_WIDTH_M = 0.085

# Affordance-net input resize, in (H, W). Must match training time. The
# values below mirror ``config/viz_affordances.yaml``'s
# ``dataset.img_resize.{static,gripper}`` (which itself matches the labeler
# output_size in ``config/cfg_datacollection.yaml`` halved to preserve
# aspect). Square defaults from ``cfg_aif_datacollection.yaml`` would
# squash the input image and produce geometrically wrong target pixel
# coords (≈ the cause of "あらぬ方向" + IK branch-flip on move_to_target).
DEFAULT_STATIC_AFF_SIZE = [96, 128]   # (H, W)
DEFAULT_GRIPPER_AFF_SIZE = [48, 64]   # (H, W)


# ───────────────────────────────────────────────── two-stage SIGINT ──

_ABORT_EVENT = threading.Event()
_ABORT_COUNT = {"n": 0}


def _install_sigint_handler() -> None:
    """First ^C requests a graceful stop; second ^C raises immediately."""
    def _sigint(_sig, _frm):
        _ABORT_COUNT["n"] += 1
        if _ABORT_COUNT["n"] == 1:
            _ABORT_EVENT.set()
            logger.warning("Ctrl-C received — finishing current step then stopping.")
        else:
            raise KeyboardInterrupt()
    signal.signal(signal.SIGINT, _sigint)


# ───────────────────────────────────────────── RealSense helpers ──

def resolve_realsense_index(serial: str) -> int:
    """Return the pyrealsense2 device index matching ``serial``."""
    import pyrealsense2 as rs

    devices = rs.context().query_devices()
    for idx, d in enumerate(devices):
        if d.get_info(rs.camera_info.serial_number) == serial:
            return idx
    detected = [d.get_info(rs.camera_info.serial_number) for d in devices]
    raise RuntimeError(
        f"RealSense serial {serial!r} not found. Detected: {detected}"
    )


# ───────────────────────────────────────────── camera_info.npz loader ──

def _load_camera_info(path: Path) -> Dict[str, np.ndarray | dict]:
    """Load static+gripper intrinsics + extrinsics from ``camera_info.npz``.

    The file is produced by ``gello_teleop_recorder_xarm.write_camera_info_files``
    and contains:
      * ``static_intrinsics`` / ``gripper_intrinsics`` — dict (fx, fy, cx, cy, ...)
      * ``static_extrinsic_calibration`` — 4x4 ``T_world_cam`` (base → static cam)
      * ``gripper_extrinsic_calibration`` — 4x4 ``T_tcp_cam`` (TCP → wrist cam)
    """
    if not path.is_file():
        raise FileNotFoundError(f"camera_info.npz not found: {path}")
    with np.load(path, allow_pickle=True) as d:
        out = {
            "static_intrinsics": dict(d["static_intrinsics"].item())
                if "static_intrinsics" in d.files else None,
            "gripper_intrinsics": dict(d["gripper_intrinsics"].item())
                if "gripper_intrinsics" in d.files else None,
            "T_world_static": np.asarray(d["static_extrinsic_calibration"], dtype=np.float32)
                if "static_extrinsic_calibration" in d.files else np.eye(4, dtype=np.float32),
            "T_tcp_gripper": np.asarray(d["gripper_extrinsic_calibration"], dtype=np.float32)
                if "gripper_extrinsic_calibration" in d.files else np.eye(4, dtype=np.float32),
        }
    return out


# ──────────────────────────────────────────────────── CameraView ──

class CameraView:
    """Pinhole + RealSense camera adapter exposing the interface that
    :class:`vapo.agent.core.target_search.TargetSearch` (real_world mode) and
    :class:`vapo.wrappers.affordance.AffordanceWrapperRealWorld` expect.

    Owns a :class:`robopy.sensors.visual.RealsenseCamera` plus the matching
    intrinsics/extrinsics loaded from ``camera_info.npz``. The robopy camera
    streams CHW float32 frames; this wrapper exposes HWC uint8 for vapo's
    pipeline. The depth stream is requested only when ``is_depth_camera=True``
    in the underlying config.
    """

    def __init__(
        self,
        rs_camera,
        intrinsics: dict,
        T_world_or_tcp_cam: np.ndarray,
        depth_scale_to_m: float = 1e-3,
        depth_max_m: float = 2.0,
        dry_run: bool = False,
        name: str = "cam",
    ):
        # Profiler tag. Timing lives in ``get_image`` rather than in the
        # callers so that EVERY frame grab is counted once, wherever it comes
        # from — including the one TargetSearch pulls through CroppedStaticCam
        # during the EFE target scan, which used to hide inside ``efe_target``.
        self._prof_tag = f"cam_{name}"
        self._cam = rs_camera
        self._intrinsics = intrinsics
        self._T = np.asarray(T_world_or_tcp_cam, dtype=np.float32)
        self._depth_scale = float(depth_scale_to_m)
        self._depth_max = float(depth_max_m)
        self._dry_run = bool(dry_run)
        # Cache of the last (rgb, depth) for `get_image()` — needed by
        # TargetSearch which calls `get_image()` then later wants the same
        # frame referenced by `self.orig_img`.
        self._last_rgb_hwc: np.ndarray | None = None
        self._last_depth: np.ndarray | None = None

    # ── geometry ─────────────────────────────────────────────
    @property
    def width(self) -> int:
        return int(self._intrinsics.get("width", 640))

    @property
    def height(self) -> int:
        return int(self._intrinsics.get("height", 480))

    @property
    def crop_coords(self):
        # Identity crop — matches what write_camera_info_files writes.
        return self._intrinsics.get("crop_coords", [0, self.height, 0, self.width])

    @property
    def resize_resolution(self):
        return self._intrinsics.get("resize_resolution", [self.width, self.height])

    def get_extrinsic_calibration(self, _robot_name: str = "panda") -> np.ndarray:
        """Return the 4x4 extrinsic matrix.

        For the static camera this is ``T_world_cam`` (base → cam).
        For the gripper camera this is ``T_tcp_cam`` (TCP → cam).
        TargetSearch / AffordanceWrapper consume both via the same API.
        """
        return self._T.copy()

    # ── projection ───────────────────────────────────────────
    def deproject(self, pixel_uv, depth_image, homogeneous: bool = False):
        """Pinhole deproject a pixel + sampled depth to camera frame.

        ``pixel_uv = (u, v)`` in pixel coords; ``depth_image`` is the full
        depth frame from `get_image()` (mm). Returns 3-vec ``[x, y, z]`` in
        meters or a 4-vec ``[x, y, z, 1]`` when ``homogeneous=True``.
        Returns None when the sampled depth is zero (invalid).
        """
        u, v = int(pixel_uv[0]), int(pixel_uv[1])
        depth_image = np.asarray(depth_image)
        if depth_image.ndim == 3:
            # depth often shipped as (1, H, W) — squeeze the channel
            depth_image = depth_image.squeeze(0) if depth_image.shape[0] == 1 else depth_image[..., 0]
        H, W = depth_image.shape[:2]
        if not (0 <= v < H and 0 <= u < W):
            return None
        z_raw = float(depth_image[v, u])
        if z_raw <= 0:
            return None
        z = z_raw * self._depth_scale  # mm → m
        if z <= 0 or z > self._depth_max:
            return None
        fx = float(self._intrinsics["fx"])
        fy = float(self._intrinsics["fy"])
        cx = float(self._intrinsics["cx"])
        cy = float(self._intrinsics["cy"])
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        pt = np.array([x, y, z], dtype=np.float32)
        if homogeneous:
            return np.array([x, y, z, 1.0], dtype=np.float32)
        return pt

    def project(self, world_pt: np.ndarray) -> Tuple[int, int]:
        """Project a 3-vec world point back to (u, v) pixel coords.

        Used by AffordanceWrapperRealWorld.viz_curr_target. We invert
        ``T_world_cam`` then apply the pinhole matrix.
        """
        T_inv = np.linalg.inv(self._T)
        p_world = np.array([float(world_pt[0]), float(world_pt[1]), float(world_pt[2]), 1.0],
                            dtype=np.float32)
        p_cam = T_inv @ p_world
        if abs(p_cam[2]) < 1e-6:
            return 0, 0
        fx = float(self._intrinsics["fx"])
        fy = float(self._intrinsics["fy"])
        cx = float(self._intrinsics["cx"])
        cy = float(self._intrinsics["cy"])
        u = int(round(fx * p_cam[0] / p_cam[2] + cx))
        v = int(round(fy * p_cam[1] / p_cam[2] + cy))
        return u, v

    # ── frame capture ────────────────────────────────────────
    def get_image(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(rgb_hwc_uint8, depth_hw_mm_uint16)`` from the live camera."""
        if self._dry_run:
            # No hardware — return synthetic black frames matching the
            # configured intrinsics so downstream tensor shapes are valid.
            H, W = self.height, self.width
            rgb_hwc = np.zeros((H, W, 3), dtype=np.uint8)
            depth_hw = np.zeros((H, W), dtype=np.uint16)
            self._last_rgb_hwc = rgb_hwc
            self._last_depth = depth_hw
            return rgb_hwc, depth_hw
        with _PROF.stage(self._prof_tag):
            # robopy's async_read blocks until a FRESH frame despite its
            # "non-blocking" docstring (it clears new_frame_event on every
            # read), so this can cost up to a full camera period.
            rgb_chw = self._cam.async_read(timeout_ms=200)
            try:
                depth = self._cam.async_read_depth(timeout_ms=200)
            except Exception:
                # Some cameras (env_cam without depth) won't have depth — fall
                # back to a zero array so TargetSearch's deproject sees no valid
                # samples (it then returns no candidates).
                depth = None
        rgb_hwc = _to_hwc_uint8(rgb_chw)
        if depth is None:
            depth = np.zeros((rgb_hwc.shape[0], rgb_hwc.shape[1]), dtype=np.uint16)
        depth_hw = np.asarray(depth)
        if depth_hw.ndim == 3:
            depth_hw = depth_hw.squeeze(0) if depth_hw.shape[0] == 1 else depth_hw[..., 0]
        self._last_rgb_hwc = rgb_hwc
        self._last_depth = depth_hw
        return rgb_hwc, depth_hw

    def latest(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return the most recent ``(rgb, depth)`` without re-reading the bus."""
        if self._last_rgb_hwc is None:
            return self.get_image()
        return self._last_rgb_hwc, self._last_depth

    def disconnect(self) -> None:
        if self._dry_run:
            return
        try:
            self._cam.disconnect()
        except Exception as exc:
            logger.warning(f"{self._cam} disconnect failed: {exc}")


class CroppedStaticCam:
    """Crop-adapter around a static :class:`CameraView` used *only* by
    TargetSearch so the affordance net sees the same field of view it was
    trained on (see ``vapo/affordance/utils/static_crop.py``).

    Cropping the image shifts the principal point but leaves the focal length
    and extrinsic untouched, so deprojection stays geometrically exact: for a
    pixel ``(u, v)`` in the cropped frame, ``(u - (cx - x1))`` equals
    ``(u_full - cx)``. We therefore just offset ``cx, cy`` by the crop origin
    and deproject on the cropped depth — the recovered camera-frame point is
    identical to the un-cropped pipeline.

    Only TargetSearch is given this wrapper; every other consumer of the static
    camera (observation recording, full-frame viz) keeps the real frame.
    """

    def __init__(self, base: "CameraView", crop_info: dict):
        self._base = base
        # The live-frame crop box is derived from the stored fractions at the
        # base camera's native resolution (stable across frames).
        hw = (base.height, base.width)
        fracs = crop_info.get("fractions_xyxy")
        if fracs is None:
            from vapo.affordance.utils.static_crop import box_to_fractions
            fracs = box_to_fractions(crop_info["box_xyxy"], crop_info["ref_hw"])
        self._box = box_for_shape(fracs, hw)  # (x1, y1, x2, y2) in live px
        self._info = crop_info
        x1, y1, x2, y2 = self._box
        base_intr = dict(base._intrinsics)
        base_intr["cx"] = float(base_intr["cx"]) - x1
        base_intr["cy"] = float(base_intr["cy"]) - y1
        base_intr["width"] = int(x2 - x1)
        base_intr["height"] = int(y2 - y1)
        self._intrinsics = base_intr
        self._T = base._T
        self._depth_scale = base._depth_scale
        self._depth_max = base._depth_max
        self._last_rgb_hwc = None
        self._last_depth = None
        logger.info(
            "[affordance] static crop active for target search: box(x1,y1,x2,y2)=%s "
            "of live frame %dx%d (WxH); cx,cy offset by (%d,%d)",
            list(self._box), base.width, base.height, x1, y1,
        )

    # ── geometry ─────────────────────────────────────────────
    @property
    def width(self) -> int:
        return int(self._intrinsics["width"])

    @property
    def height(self) -> int:
        return int(self._intrinsics["height"])

    @property
    def crop_coords(self):
        return [0, self.height, 0, self.width]

    @property
    def resize_resolution(self):
        return [self.width, self.height]

    def get_extrinsic_calibration(self, robot_name: str = "panda") -> np.ndarray:
        return self._base.get_extrinsic_calibration(robot_name)

    # ── projection (offset principal point) ──────────────────
    def deproject(self, pixel_uv, depth_image, homogeneous: bool = False):
        u, v = int(pixel_uv[0]), int(pixel_uv[1])
        depth_image = np.asarray(depth_image)
        if depth_image.ndim == 3:
            depth_image = depth_image.squeeze(0) if depth_image.shape[0] == 1 else depth_image[..., 0]
        H, W = depth_image.shape[:2]
        if not (0 <= v < H and 0 <= u < W):
            return None
        z_raw = float(depth_image[v, u])
        if z_raw <= 0:
            return None
        z = z_raw * self._depth_scale
        if z <= 0 or z > self._depth_max:
            return None
        fx, fy = float(self._intrinsics["fx"]), float(self._intrinsics["fy"])
        cx, cy = float(self._intrinsics["cx"]), float(self._intrinsics["cy"])
        x = (u - cx) * z / fx
        y = (v - cy) * z / fy
        if homogeneous:
            return np.array([x, y, z, 1.0], dtype=np.float32)
        return np.array([x, y, z], dtype=np.float32)

    def project(self, world_pt: np.ndarray) -> Tuple[int, int]:
        # Full-frame projection then shift into cropped pixel coords.
        u, v = self._base.project(world_pt)
        return u - self._box[0], v - self._box[1]

    # ── frame capture (crop rgb + depth identically) ─────────
    def get_image(self) -> Tuple[np.ndarray, np.ndarray]:
        rgb, depth = self._base.get_image()
        rgb_c = np.ascontiguousarray(crop_frame_hwc(rgb, self._box))
        depth_c = np.ascontiguousarray(crop_frame_hwc(depth, self._box))
        self._last_rgb_hwc, self._last_depth = rgb_c, depth_c
        return rgb_c, depth_c

    def latest(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._last_rgb_hwc is None:
            return self.get_image()
        return self._last_rgb_hwc, self._last_depth

    def disconnect(self) -> None:
        self._base.disconnect()


def crop_static_rgb_for_net(static_net, rgb_hwc):
    """Apply the training-time static crop attached to ``static_net`` to a live
    HWC RGB frame (fractions scaled to this frame's resolution). Returns the
    frame unchanged when no crop was saved."""
    info = getattr(static_net, "_static_crop", None)
    if info is None:
        return rgb_hwc
    fracs = info.get("fractions_xyxy") or box_to_fractions(info["box_xyxy"], info["ref_hw"])
    box = box_for_shape(fracs, rgb_hwc.shape[:2])
    return np.ascontiguousarray(crop_frame_hwc(rgb_hwc, box))


def resolve_static_model_dir(affordance_cfg, seed=None, dataset_name=None):
    """Return the directory of the static affordance checkpoint that
    :func:`vapo.utils.utils.init_aff_net` would load, mirroring its candidate
    resolution. Used to locate ``static_crop.json``. ``None`` if unresolvable.

    Uses the same variant tag (color + orientation head) as init_aff_net so the
    crop is read from the correct variant directory. The color is derived from
    the affordance validation transforms (Grayscale present → grayscale) and
    ``predict_orientation`` from the per-cam config (default True).
    """
    try:
        orig_path = affordance_cfg["static_cam"].model_path
    except Exception:
        return None
    from vapo.affordance.utils.utils import get_abs_path
    from vapo.utils.utils import (
        affordance_ckpt_candidates, affordance_color_from_transforms,
        affordance_variant_tag,
    )
    try:
        color = affordance_color_from_transforms(
            affordance_cfg.transforms["validation"])
    except Exception:
        color = "grayscale"
    try:
        po = bool(affordance_cfg["static_cam"].get("predict_orientation", True))
    except Exception:
        po = True
    variant = affordance_variant_tag(po, color=color)
    candidates = affordance_ckpt_candidates(
        orig_path, seed=seed, dataset_name=dataset_name, variant=variant)
    for c in candidates:
        if os.path.exists(get_abs_path(c)):
            return os.path.dirname(get_abs_path(c))
    return None


def load_static_crop_for_run(affordance_cfg, seed=None, dataset_name=None):
    """Load the static crop info persisted next to the trained weights, or
    ``None`` (no crop → full frame, backward compatible)."""
    model_dir = resolve_static_model_dir(affordance_cfg, seed, dataset_name)
    info = load_crop_info(model_dir)
    if info is not None:
        logger.info("[affordance] loaded static crop from %s: box=%s (ref HxW=%s)",
                    model_dir, info.get("box_xyxy"), info.get("ref_hw"))
    return info


def _to_hwc_uint8(frame: np.ndarray) -> np.ndarray:
    """Normalize a robopy RealsenseCamera frame to HWC uint8 RGB."""
    if frame.ndim == 3 and frame.shape[0] == 3:
        frame = frame.transpose(1, 2, 0)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    return frame


# ──────────────────────────────────────────── camera_manager façade ──

class _CameraManager:
    """Thin façade exposing ``static_cam`` / ``gripper_cam`` attributes —
    matches the interface that vapo's real-world affordance wrapper and
    TargetSearch dereference (``env.camera_manager.static_cam``)."""
    def __init__(self, static_cam: CameraView, gripper_cam: CameraView):
        self.static_cam = static_cam
        self.gripper_cam = gripper_cam


# ────────────────────────────── axis-angle send patch (XArmFollower) ──

def _patch_follower_aa_send(follower) -> None:
    """Wrap ``XArmFollower._send_cartesian`` so it can dispatch to xArm
    SDK's ``set_position_aa`` (axis-angle target) when the current
    ``_target_command`` carries a ``format == "aa"`` marker.

    robopy's stock control thread always calls ``set_position(roll,
    pitch, yaw)`` (Euler). At our working pose roll lives ≈±π and the
    Euler representation flips sign whenever the policy nudges past
    the wrap point — the SDK then interprets it as a near-2π rotation
    command and the wrist visibly snaps. The axis-angle API doesn't
    wrap, so the same SO(3) target stays continuous to the controller.

    Patches the bound method in-place; no robopy edit required. Pose
    layout for both modes is ``[x_mm, y_mm, z_mm, ?_rad, ?_rad, ?_rad]``
    — the last three components are interpreted per ``format``.
    """
    if getattr(follower, "_aa_send_patched", False):
        return
    original_send = follower._send_cartesian

    def _send_cartesian_dispatched(pose: np.ndarray) -> None:
        # Snapshot the format marker from the cmd dict. ``_send_cartesian``
        # is invoked both from the control-thread loop (where the cmd was
        # written by env.step) and from other internal callers (joint
        # homing FK → set_position). Only the cmd dict knows whether
        # ``pose`` is axis-angle or Euler.
        with follower._target_command_lock:
            fmt = follower._target_command.get("format")
        if fmt != "aa":
            original_send(pose)
            return
        sdk = follower._robot
        if sdk is None:
            return
        # Apply workspace bounds in mm — matches ``_send_cartesian``'s
        # logic so the safety clip remains active on the AA path.
        ws = getattr(follower, "_workspace", None)
        pose = np.asarray(pose, dtype=np.float32).copy()
        if ws is not None:
            pose[0] = float(np.clip(pose[0], ws.min_x, ws.max_x))
            pose[1] = float(np.clip(pose[1], ws.min_y, ws.max_y))
            min_z = ws.effective_min_z(pose[0], pose[1])
            pose[2] = float(np.clip(pose[2], min_z, ws.max_z))
        ret = sdk.set_position_aa(
            pose.tolist(),
            wait=False, is_radian=True,
            speed=follower._cartesian_speed,
            mvacc=follower._cartesian_mvacc,
            radius=0,
        )
        if ret in (1, 9):
            follower._clear_error_states()

    follower._send_cartesian = _send_cartesian_dispatched
    follower._aa_send_patched = True


# ───────────────────────────────────────────────── XArmRealEnv ──

class XArmRealEnv:
    """Minimal env adapter for xArm7 + RealSense.

    Exposes the surface that:
      * TargetSearch (mode="real_world") expects: ``camera_manager``,
        ``save_images``, ``viz``, ``task``, ``get_obs()``.
      * Our episode loop expects: ``reset()``, ``move_to_target(pos)``,
        ``step(action_dict)``, ``get_obs()``.

    Action convention for `step`:
      * ``action = {"type": "cartesian_abs", "action": [x, y, z, roll, pitch, yaw, grip_pm1]}``
        — absolute pose in (m + rad), Euler-RPY (xArm convention).
      * ``action = {"type": "cartesian_rel", "action": [dx, dy, dz, drx, dry, drz, grip_pm1]}``
        — delta in (m + rad), Euler-RPY.
      * A bare 7-vec is interpreted as ``cartesian_rel`` (matches the
        sim-side convention where 7-vec = `rel_actions`).
    """

    def __init__(
        self,
        follower_ip: str,
        env_cam_serial: str,
        wrist_cam_serial: str,
        camera_info: dict,
        start_joints_deg: np.ndarray | None = None,
        max_delta: float = 0.05,
        control_hz: float = 100.0,
        img_width: int = 640,
        img_height: int = 480,
        task: str = "slide",
        save_images: bool = False,
        viz: bool = False,
        termination_radius: float = 0.15,
        gripper_aff_min_robustness: float = 0.0,
        dry_run: bool = False,
        use_workspace_bounds: bool = True,
        pause_on_sdk_error: bool = True,
        sdk_error_poll_hz: float = 4.0,
        sdk_error_auto_recover: bool = False,
        sdk_error_auto_recover_after_s: float = 3.0,
        sdk_error_max_wait_s: float = 0.0,
    ):
        from robopy.config.robot_config import XArmConfig, XArmWorkspaceBounds
        from robopy.robots.xarm.xarm_follower import XArmFollower
        from robopy.sensors.visual.realsense_camera import (
            RealsenseCamera, RealsenseCameraConfig,
        )

        self.task = task
        self.save_images = save_images
        self.viz = viz
        self.termination_radius = float(termination_radius)
        # gripper-cam affordance のクラスタ信頼度(mean foreground prob)がこの値
        # 未満なら _refine_target_via_gripper で棄却し、curr_detected_obj を更新
        # しない（0.0 で棄却なし）。sim 版 aff_wrapper_base の同名しきい値と対応。
        self.gripper_aff_min_robustness = float(gripper_aff_min_robustness)
        self.reward_fail = 0.0
        self.reward_success = 1.0
        self.img_size = 64
        self.dry_run = bool(dry_run)
        # Last gripper value actually transmitted to the controller.
        # robopy's control thread blindly re-issues
        # ``set_modbus_gripper_position`` every 50 Hz tick whenever the
        # cached ``target_command["gripper"]`` is non-None; with TE the
        # ensembled gripper value oscillates by < 1% per step and that
        # modbus call rate produces a flood of ``code=2`` (busy) errors.
        # We instead drive the gripper from ``XArmRealEnv.step`` only
        # when the value moves more than ``_GRIP_CHANGE_THRESHOLD``,
        # and always pass ``gripper=None`` through the control thread.
        self._last_grip_norm: Optional[float] = None
        self._GRIP_CHANGE_THRESHOLD = 0.03
        # ── SDK error-gate configuration ──
        # When ``pause_on_sdk_error`` is set, ``step()`` (and any other
        # motion-emitting path that opts in) polls the controller error
        # / warn flags via ``XArmAPI.has_err_warn`` before forwarding
        # the next command. While a fault is latched:
        #   1. The follower's 50 Hz control thread is stopped so it
        #      stops flooding the SDK with commands that would only
        #      be rejected (which is what produces the ``code=2`` /
        #      ``code=-1`` spam in the log).
        #   2. The main thread blocks on the gate, printing the current
        #      err/warn codes every ~2s so the operator can see what
        #      the arm is complaining about.
        #   3. If ``sdk_error_auto_recover`` is set, after
        #      ``sdk_error_auto_recover_after_s`` seconds the gate calls
        #      ``clean_error() / clean_warn() / motion_enable(True) /
        #      set_state(0)`` and resumes; otherwise it waits for a
        #      human to clear the fault on the pendant.
        #   4. If ``sdk_error_max_wait_s > 0``, the gate hard-aborts
        #      after that many seconds without recovery.
        self._pause_on_sdk_error = bool(pause_on_sdk_error)
        self._sdk_error_poll_hz = float(sdk_error_poll_hz)
        self._sdk_error_auto_recover = bool(sdk_error_auto_recover)
        self._sdk_error_auto_recover_after_s = float(
            sdk_error_auto_recover_after_s
        )
        self._sdk_error_max_wait_s = float(sdk_error_max_wait_s)
        # ``start_joints_deg`` may be:
        #   * ``None`` (default)  → capture the robot's pose at script-launch
        #                           and use that as the session home.
        #   * 7-vector (deg)      → explicit user override (with full xArm7
        #                           joint-limit validation in ``reset()``).
        # We assign ``_start_joints_rad`` provisionally here so XArmConfig
        # has a value; if the user opted in to dynamic capture, we overwrite
        # it right after ``connect()`` succeeds.
        self._dynamic_home = start_joints_deg is None
        provisional_deg = (
            DRY_RUN_FALLBACK_JOINTS_DEG if start_joints_deg is None
            else np.asarray(start_joints_deg, dtype=np.float32)
        )
        self._start_joints_rad = np.deg2rad(provisional_deg).astype(np.float32)

        # ── xArm follower (solo, no leader) ──
        # ``use_workspace_bounds=False`` passes ``None`` so robopy's
        # ``_send_cartesian`` skips its silent xyz/z clipping. Useful for
        # debugging an extrinsic-calibration miss when the affordance
        # world point legitimately sits outside the cell-specific bounds.
        xarm_cfg = XArmConfig(
            follower_ip=follower_ip,
            workspace_bounds=XArmWorkspaceBounds() if use_workspace_bounds else None,
            restriction=["default", "drawer"] if use_workspace_bounds else None,
            start_joints=self._start_joints_rad,
            control_frequency=control_hz,
            max_delta=max_delta,
        )
        self._follower = XArmFollower(xarm_cfg)
        # Monkey-patch ``_send_cartesian`` so the control thread can ship
        # the cached target via either ``set_position`` (Euler RPY — robopy
        # default) **or** ``set_position_aa`` (axis-angle) depending on a
        # ``format`` key we attach to ``_target_command``. Needed because
        # roll ≈ ±π is the steady-state working pose for this cell, and
        # the Euler API discontinuity at ±π was producing apparent
        # "RX flip" behaviour during policy execution. The AA path keeps
        # the underlying SO(3) target continuous all the way to the SDK.
        _patch_follower_aa_send(self._follower)
        if not self.dry_run:
            self._follower.connect()
            # robopy's connect() leaves the control thread in "joint" mode
            # holding ``cmd["joints"] = current_joints``. Each cycle then
            # calls ``set_position(FK(current))`` at 50 Hz -- harmless if
            # the arm is at its current pose, but it primes the SDK's
            # motion queue with Cartesian commands. Switch the thread to
            # ``cartesian_abs`` at the current EE pose so it just sends
            # the same set_position repeatedly (no IK branch flips), and
            # so ``reset()`` doesn't have to fight a joint-mode override
            # when it stops/restarts the thread.
            self._prime_cartesian_hold(self._follower)
            logger.info(f"xArm follower connected ({follower_ip}).")

            # If dynamic home was requested, snapshot the current joints now
            # and use them as the session home. Whatever pose the user has
            # the arm in at launch becomes the home -- guaranteed reachable
            # (since the arm is physically there) and matches the user's
            # specific table/mount geometry.
            if self._dynamic_home:
                self._start_joints_rad = (
                    self._follower.get_joint_state()[:7].astype(np.float32)
                )
                logger.info(
                    "[home] using script-launch pose as session home: "
                    "joints_deg=%s",
                    np.rad2deg(self._start_joints_rad).round(1).tolist(),
                )

        # Remember the start EE orientation so move_to_target preserves it.
        # NOTE: this is captured at *connect-time* pose (arbitrary, possibly
        # leftover from previous run). ``reset()`` refreshes this once the
        # arm is at home; ``move_to_target`` reads orientation fresh.
        if not self.dry_run:
            ee = self._follower.get_ee_pos_quat()  # [x, y, z, qx, qy, qz, qw]
            self._start_orn_euler = _R.from_quat(ee[3:7]).as_euler("xyz").astype(np.float32)
        else:
            self._start_orn_euler = np.array([np.pi, 0.0, 0.0], dtype=np.float32)

        # ── RealSense cameras (env + wrist, both with depth) ──
        env_cfg = RealsenseCameraConfig(
            name="env", fps=30, serial_no=env_cam_serial, is_depth_camera=True,
        )
        env_cfg.width, env_cfg.height = img_width, img_height
        wrist_cfg = RealsenseCameraConfig(
            name="wrist", fps=30, serial_no=wrist_cam_serial, is_depth_camera=True,
        )
        wrist_cfg.width, wrist_cfg.height = img_width, img_height
        env_rs = RealsenseCamera(env_cfg)
        wrist_rs = RealsenseCamera(wrist_cfg)
        if not self.dry_run:
            env_rs.index = resolve_realsense_index(env_cam_serial)
            wrist_rs.index = resolve_realsense_index(wrist_cam_serial)
            env_rs.connect()
            wrist_rs.connect()
            logger.info("RealSense cameras connected (env + wrist).")

        self._static_view = CameraView(
            env_rs,
            intrinsics=(camera_info["static_intrinsics"] or {
                "fx": 600.0, "fy": 600.0, "cx": img_width / 2.0, "cy": img_height / 2.0,
                "width": img_width, "height": img_height,
            }),
            T_world_or_tcp_cam=camera_info["T_world_static"],
            dry_run=self.dry_run,
            name="static",
        )
        self._gripper_view = CameraView(
            wrist_rs,
            intrinsics=(camera_info["gripper_intrinsics"] or {
                "fx": 600.0, "fy": 600.0, "cx": img_width / 2.0, "cy": img_height / 2.0,
                "width": img_width, "height": img_height,
            }),
            T_world_or_tcp_cam=camera_info["T_tcp_gripper"],
            dry_run=self.dry_run,
            name="gripper",
        )
        self.camera_manager = _CameraManager(self._static_view, self._gripper_view)

        # ── episode-level state set by target_search / aff wrapper ──
        self.curr_detected_obj: np.ndarray | None = None
        self.curr_detected_orn: np.ndarray | None = None  # 予測 orientation(world euler)
        self.static_initial_target: np.ndarray | None = None
        self.target_pos: np.ndarray | None = None
        self.target_orn = self._start_orn_euler.copy()
        self.target = None
        # Safety guard: reject affordance-predicted wrist orientations that
        # deviate more than this from the (known-good, down-pointing) home
        # orientation -- a bad prediction otherwise commands an impossible
        # pose. Overridden from CLI (--max_pred_orn_dev_deg) in main().
        self.max_pred_orn_dev_rad = float(np.deg2rad(60.0))

    # ── lifecycle ───────────────────────────────────────────
    def disconnect(self) -> None:
        if not self.dry_run:
            try:
                self._follower.disconnect()
            except Exception as exc:
                logger.warning(f"follower disconnect failed: {exc}")
        self._static_view.disconnect()
        self._gripper_view.disconnect()

    @staticmethod
    def _prime_cartesian_hold(follower) -> None:
        """Switch follower's control-thread target to ``cartesian_abs`` at
        the current EE pose so it just holds position. Avoids the
        ``set_position(FK(joints))`` joint-mode spam that can prime
        spurious IK solutions on a 7-DoF arm."""
        sdk = follower._robot
        if sdk is None:
            return
        cur_aa_full = np.asarray(
            sdk.get_position_aa(is_radian=True)[1], dtype=np.float64,
        )
        cur_xyz_mm = cur_aa_full[:3]
        cur_aa = cur_aa_full[3:6]
        aa_norm = float(np.linalg.norm(cur_aa))
        cur_rpy = (
            _R.from_rotvec(cur_aa).as_euler("xyz")
            if aa_norm > 1e-9 else np.zeros(3, dtype=np.float64)
        )
        cur_pose_euler = np.concatenate(
            [cur_xyz_mm, cur_rpy]
        ).astype(np.float32)
        with follower._target_command_lock:
            follower._target_command = {
                "mode": "cartesian_abs",
                "pose": cur_pose_euler,
                "gripper": 0.0,
            }

    @property
    def robot(self):
        """Compatibility shim — vapo's AffordanceWrapperRealWorld dereferences
        ``env.robot.get_tcp_pos_orn()``. We expose the xArm follower's TCP
        through a small adapter object that mimics the Panda API."""
        env = self
        class _RobotShim:
            def get_tcp_pos_orn(_self):
                ee = env._follower.get_ee_pos_quat()
                return ee[:3].astype(np.float32), ee[3:7].astype(np.float32)
            def open_gripper(_self, blocking: bool = False):
                env._follower.command_joint_state(
                    np.concatenate([env._follower.get_joint_state()[:7], [0.0]])
                )
                if blocking:
                    time.sleep(0.5)
        return _RobotShim()

    # ── reset / move ────────────────────────────────────────
    def _drive_to_home_joints(self, log_tag: str = "home") -> None:
        """Drive the arm to ``self._start_joints_rad`` via direct
        ``set_servo_angle`` (joint motion). Stops/restarts the control
        thread so the joint command isn't fighting the 50 Hz Cartesian
        ``set_position`` loop, and so the controller's 7-DoF IK does
        not pick an unrelated joint solution.

        Used by both ``reset()`` (start of episode) and
        ``move_to_target()`` (every motion goes via home first).
        """
        if self.dry_run:
            return
        follower = self._follower
        sdk = follower._robot
        if sdk is None:
            return

        # 1. Stop control thread (set_position spam vs set_servo_angle).
        follower._running = False
        if getattr(follower, "_thread", None) is not None:
            follower._thread.join(timeout=2.0)
            follower._thread = None

        # 2. Clear any leftover error / re-enable motion.
        try:
            sdk.clean_error()
            sdk.clean_warn()
            sdk.motion_enable(True)
            sdk.set_mode(0)
            sdk.set_state(state=0)
        except Exception as exc:
            logger.warning(f"[{log_tag}] error-clear sequence: {exc}")

        # 2b. Open the gripper *before* moving so the arm always returns
        #     to home empty-handed (drops whatever it was holding).
        try:
            follower._set_gripper_position(follower._gripper_open)
        except Exception as exc:
            logger.warning(f"[{log_tag}] open gripper before home failed: {exc}")

        # 3. Direct joint motion to home -- guaranteed configuration.
        ret = sdk.set_servo_angle(
            angle=self._start_joints_rad.tolist(),
            is_radian=True,
            speed=0.5,
            mvacc=2.0,
            wait=True,
        )
        if ret == -8:
            joints_deg = np.rad2deg(self._start_joints_rad).round(1).tolist()
            raise SystemExit(
                f"\n[FATAL] xArm rejected home joints (code=-8, "
                f"out_of_joint_range).\n"
                f"        Requested: {joints_deg} deg\n"
                f"        xArm7 joint limits (deg, approximate):\n"
                f"          j1: ±360   j2: -118..+120   j3: ±180   "
                f"j4: -4..+225\n"
                f"          j5: ±180   j6: -97..+180    j7: ±180\n"
                f"        Pass a reachable home via "
                f"`--start_joints_deg J1 J2 J3 J4 J5 J6 J7`."
            )
        if ret != 0:
            logger.warning(
                f"[{log_tag}] set_servo_angle returned code={ret} "
                "(robot may not be at home pose)."
            )

        # 4. Refresh state and switch control thread to cartesian_abs
        #    at the *current* EE so it just holds position on restart.
        follower._last_state = follower._update_last_state()
        cur_aa_full = np.asarray(
            sdk.get_position_aa(is_radian=True)[1], dtype=np.float64,
        )
        cur_xyz_mm = cur_aa_full[:3]
        cur_aa = cur_aa_full[3:6]
        aa_norm = float(np.linalg.norm(cur_aa))
        if aa_norm > 1e-9:
            cur_rpy = _R.from_rotvec(cur_aa).as_euler("xyz")
        else:
            cur_rpy = np.zeros(3, dtype=np.float64)
        cur_pose_euler = np.concatenate([cur_xyz_mm, cur_rpy]).astype(np.float32)
        with follower._target_command_lock:
            follower._target_command = {
                "mode": "cartesian_abs",
                "pose": cur_pose_euler,
                "gripper": 0.0,
            }

        # 5. Restart the control thread.
        follower._running = True
        follower._thread = threading.Thread(
            target=follower._robot_thread, daemon=True,
        )
        follower._thread.start()
        time.sleep(0.2)

        # 6. Refresh stored start orientation now that we're at home.
        ee_home = follower.get_ee_pos_quat()
        self._start_orn_euler = (
            _R.from_quat(ee_home[3:7].astype(np.float64))
              .as_euler("xyz").astype(np.float32)
        )
        cur_joints = follower.get_joint_state()[:7]
        logger.info(
            "[%s] home reached: ee_xyz=%s rpy_deg=%s joint_gap=%.3frad",
            log_tag,
            ee_home[:3].round(3).tolist(),
            np.rad2deg(self._start_orn_euler).round(1).tolist(),
            float(np.abs(cur_joints - self._start_joints_rad).max()),
        )

    def reset(self) -> dict:
        """Drive arm to ``start_joints`` with gripper open + clear episode state."""
        if self.dry_run:
            return self.get_obs()
        self._drive_to_home_joints(log_tag="reset")
        # Open gripper (the cartesian_abs target command sets gripper=0.0
        # which already opens it, but call the explicit helper too).
        try:
            self._follower._set_gripper_position(self._follower._gripper_open)
            # Record the position we just commanded so the first env.step
            # after reset uses it as the change-detection baseline.
            self._last_grip_norm = 0.0
        except Exception as exc:
            logger.warning(f"[reset] open gripper failed: {exc}")
            self._last_grip_norm = None
        # Belt-and-braces: clear any leftover gripper field in the
        # robopy ``target_command`` so the control thread doesn't spam
        # modbus writes between reset and the first env.step.
        try:
            with self._follower._target_command_lock:
                if "gripper" in self._follower._target_command:
                    self._follower._target_command["gripper"] = None
        except Exception:
            pass
        # Reset episode-level state.
        self.curr_detected_obj = None
        self.static_initial_target = None
        self.target_pos = None
        return self.get_obs()

    def move_to_target(self, target_pos: np.ndarray,
                       target_orn: np.ndarray = None,
                       per_phase_timeout_s: float = 8.0,
                       **_legacy_kwargs) -> List[dict]:
        """Two-phase motion: **current → home → target**.

        No intermediate lift is inserted. Instead the arm always passes
        through the home joint configuration first, then runs a single
        Cartesian-absolute command to the affordance target. This keeps
        every move's starting configuration deterministic (= home), which
        on a 7-DoF arm avoids the IK branch-flip that a direct
        current→target Cartesian solve can hit, and the user-visible
        "current → up → over → down" zig-zag is replaced by a clean
        current → home → target sequence.

        ``target_pos`` is in **meters** (robot-base frame).
        """
        target_pos = np.asarray(target_pos, dtype=np.float32).flatten()
        if target_pos.shape[0] < 3:
            return []
        if self.dry_run:
            return [self.get_obs()]

        # ── Phase 1: drive to home via direct joint motion. ──
        logger.info(
            "[move_to_target] phase 1/2 -> home (start_joints) "
            "target_xyz_m=%s",
            target_pos.round(3).tolist(),
        )
        self._drive_to_home_joints(log_tag="move_to_target.home")

        # ── Phase 2: home → target (single Cartesian-absolute command).
        # Orientation is read fresh from the at-home EE pose so the
        # wrist orientation stays consistent with the home configuration.
        ee_home = self._follower.get_ee_pos_quat()
        home_xyz = ee_home[:3].astype(np.float32)
        home_rpy = _R.from_quat(
            ee_home[3:7].astype(np.float64)
        ).as_euler("xyz").astype(np.float32)
        rpy = home_rpy
        # affordance が予測した orientation(world euler 'xyz')があればそれを使う。
        # 注意: affordance world フレーム == robot-base フレームを仮定。
        if target_orn is not None:
            to = np.asarray(target_orn, dtype=np.float32).flatten()
            if to.shape[0] >= 3:
                pred_rpy = to[:3].astype(np.float32)
                # Safety guard: reject predictions that deviate wildly from
                # the home wrist orientation (e.g. a mis-trained orientation
                # head) and fall back to the safe home pose.
                dev_rad = float(
                    (_R.from_euler("xyz", pred_rpy.astype(np.float64))
                     * _R.from_euler("xyz", home_rpy.astype(np.float64)).inv()
                     ).magnitude()
                )
                if dev_rad > self.max_pred_orn_dev_rad:
                    logger.warning(
                        "[move_to_target] predicted orientation rejected: "
                        "deviates %.1f deg from home (limit %.1f deg). Using "
                        "home wrist orientation instead. pred_rpy_deg=%s",
                        np.rad2deg(dev_rad),
                        np.rad2deg(self.max_pred_orn_dev_rad),
                        np.rad2deg(pred_rpy).round(1).tolist(),
                    )
                else:
                    rpy = pred_rpy

        # Workspace warning only (no bail-out). When robopy's bounds are
        # enabled, ``_send_cartesian`` will silently clip the target into
        # the box -- the user wants to see the resulting motion either
        # way so they can debug extrinsic / affordance.
        ws = getattr(self._follower, "_workspace", None)
        if ws is not None:
            tx = float(target_pos[0]) * 1000.0
            ty = float(target_pos[1]) * 1000.0
            tz = float(target_pos[2]) * 1000.0
            min_z = ws.effective_min_z(tx, ty)
            oob_axes = []
            if not (ws.min_x <= tx <= ws.max_x):
                oob_axes.append(f"x={tx:.1f}∉[{ws.min_x},{ws.max_x}]")
            if not (ws.min_y <= ty <= ws.max_y):
                oob_axes.append(f"y={ty:.1f}∉[{ws.min_y},{ws.max_y}]")
            if not (min_z <= tz <= ws.max_z):
                oob_axes.append(f"z={tz:.1f}∉[{min_z},{ws.max_z}]")
            if oob_axes:
                logger.warning(
                    "[move_to_target] target outside XArmWorkspaceBounds "
                    "-- robopy will silently clip. Offending axes: %s. "
                    "Pass ``--no_workspace_bounds`` to disable clipping.",
                    ", ".join(oob_axes),
                )

        logger.info(
            "[move_to_target] phase 2/2 home_xyz=%s -> target_xyz=%s rpy_deg=%s",
            home_xyz.round(3).tolist(),
            target_pos.round(3).tolist(),
            np.rad2deg(rpy).round(1).tolist(),
        )

        # Send via the axis-angle path (``set_position_aa``) instead of the
        # Euler ``set_position``. When the affordance-predicted orientation
        # nudges roll across the ±π wrap (home roll ≈ +π, predicted ≈ -π),
        # the Euler path returns SDK ``code=1`` (the "roll ≈ ±π" discontinuity
        # already avoided in ``_move_env_to_init_pose``). Axis-angle has no
        # such wrap, so the same physical pose is commanded cleanly.
        aa = (
            _R.from_euler("xyz", rpy.astype(np.float64))
              .as_rotvec().astype(np.float32)
        )
        self.step({
            "type": "cartesian_abs_aa",
            "action": np.array([
                target_pos[0], target_pos[1], target_pos[2],
                aa[0], aa[1], aa[2],
                1.0,  # gripper open during the reach
            ], dtype=np.float32),
        }, return_obs=False)

        # Poll until convergence or timeout.
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < per_phase_timeout_s:
            if _ABORT_EVENT.is_set():
                break
            ee_cur = self._follower.get_ee_pos_quat()
            gap = float(np.linalg.norm(ee_cur[:3] - target_pos))
            if gap < 0.01:
                break
            time.sleep(0.05)
        ee_final = self._follower.get_ee_pos_quat()
        final_gap_mm = float(
            np.linalg.norm(ee_final[:3] - target_pos)
        ) * 1000.0
        logger.info(
            "[move_to_target] reached: ee_xyz=%s gap=%.1fmm",
            ee_final[:3].round(3).tolist(), final_gap_mm,
        )
        return [self.get_obs()]

    # ── SDK error gate ─────────────────────────────────────
    def _wait_until_sdk_clear(self, tag: str = "step") -> None:
        """Block until the controller has no latched error / warn code.

        The xArm controller latches faults (joint-limit violation, cmd
        rejection, self-collision, modbus busy, etc.) and refuses new
        motion commands while ``has_err_warn`` is true. If we keep
        forwarding commands anyway, robopy's 50 Hz control thread spams
        the SDK and the log fills with ``code=1`` / ``code=2`` while
        the arm sits idle -- worse, the *next* successful command
        picks up a stale target that has drifted from the actual arm
        pose (because the arm didn't execute the queued moves).

        This gate:
          * stops the control thread so the SDK sees no more traffic
            while the fault is latched;
          * polls the controller at ``sdk_error_poll_hz`` and prints
            the current err/warn code every ~2s;
          * either waits for a human to clear the fault (default), or
            auto-recovers after ``sdk_error_auto_recover_after_s``
            when ``--sdk_error_auto_recover`` is set;
          * on clear, refreshes the cached ``target_command`` to the
            arm's *actual* current pose and restarts the control
            thread so motion picks up from where the arm truly is.
        """
        if self.dry_run or not self._pause_on_sdk_error:
            return
        follower = self._follower
        sdk = follower._robot
        if sdk is None:
            return
        try:
            if not sdk.has_err_warn:
                return
        except Exception as exc:
            logger.warning(f"[{tag}] has_err_warn probe failed: {exc}")
            return

        # 1. Stop the 50 Hz thread so it stops flooding the SDK.
        follower._running = False
        if getattr(follower, "_thread", None) is not None:
            follower._thread.join(timeout=2.0)
            follower._thread = None

        start = time.perf_counter()
        last_log = 0.0
        last_ec: Optional[int] = None
        last_wc: Optional[int] = None
        auto = self._sdk_error_auto_recover
        auto_after = self._sdk_error_auto_recover_after_s
        max_wait = self._sdk_error_max_wait_s  # 0 = infinite
        poll_dt = 1.0 / max(self._sdk_error_poll_hz, 0.1)
        recovered = False

        while True:
            if _ABORT_EVENT.is_set():
                break
            try:
                if not sdk.has_err_warn:
                    break
            except Exception as exc:
                logger.warning(f"[{tag}] has_err_warn probe failed: {exc}")
                break
            now = time.perf_counter()
            elapsed = now - start
            try:
                _, (ec, wc) = sdk.get_err_warn_code()
            except Exception:
                ec, wc = -1, -1
            if (ec, wc) != (last_ec, last_wc) or (now - last_log) >= 2.0:
                logger.warning(
                    f"[{tag}] SDK fault latched: err={ec} warn={wc} — "
                    f"pausing action stream (elapsed {elapsed:.1f}s). "
                    f"Clear the fault on the pendant or pass "
                    f"--sdk_error_auto_recover to auto-clean."
                )
                last_log = now
                last_ec, last_wc = ec, wc
            if auto and elapsed >= auto_after and not recovered:
                logger.warning(
                    f"[{tag}] auto-recover: clean_error + clean_warn + "
                    f"motion_enable(True) + set_state(0)"
                )
                try:
                    sdk.clean_error()
                    sdk.clean_warn()
                    sdk.motion_enable(True)
                    sdk.set_mode(0)
                    sdk.set_state(state=0)
                except Exception as exc:
                    logger.warning(f"[{tag}] auto-recover failed: {exc}")
                recovered = True
                time.sleep(0.5)
                continue
            if max_wait > 0.0 and elapsed >= max_wait:
                raise SystemExit(
                    f"[{tag}] SDK fault latched > {max_wait:.1f}s without "
                    f"recovery (err={last_ec}, warn={last_wc}). Aborting."
                )
            time.sleep(poll_dt)

        # 2. Refresh the target_command to the actual current pose so the
        #    restarted control thread doesn't snap the arm back to a
        #    pre-fault stale target.
        try:
            cur_aa_full = np.asarray(
                sdk.get_position_aa(is_radian=True)[1], dtype=np.float64,
            )
            cur_xyz_mm = cur_aa_full[:3]
            cur_aa = cur_aa_full[3:6]
            aa_norm = float(np.linalg.norm(cur_aa))
            cur_rpy = (
                _R.from_rotvec(cur_aa).as_euler("xyz")
                if aa_norm > 1e-9
                else np.zeros(3, dtype=np.float64)
            )
            cur_pose_euler = np.concatenate(
                [cur_xyz_mm, cur_rpy]
            ).astype(np.float32)
            with follower._target_command_lock:
                follower._target_command = {
                    "mode": "cartesian_abs",
                    "pose": cur_pose_euler,
                    "gripper": None,
                }
        except Exception as exc:
            logger.warning(f"[{tag}] pose refresh after clear failed: {exc}")

        # 3. Restart the control thread.
        follower._running = True
        follower._thread = threading.Thread(
            target=follower._robot_thread, daemon=True,
        )
        follower._thread.start()
        time.sleep(0.2)
        logger.warning(
            f"[{tag}] SDK cleared "
            f"(waited {time.perf_counter() - start:.1f}s"
            f"{', auto-recovered' if recovered else ''}). Resuming."
        )

    # ── step ────────────────────────────────────────────────
    def step(self, action, move_to_box: bool = False, return_obs: bool = True):
        """Send one Cartesian command. Returns ``(obs, reward, done, info)``.

        Supported transports (``action["type"]``):

          * ``"cartesian_abs"``     — ``[xyz_m, roll, pitch, yaw, grip]``
                                      (Euler XYZ). Forwarded to robopy's
                                      ``command_cartesian_absolute`` →
                                      SDK ``set_position``.
          * ``"cartesian_abs_aa"``  — ``[xyz_m, rx, ry, rz, grip]``
                                      (axis-angle, rad). Bypasses the
                                      Euler path and routes through SDK
                                      ``set_position_aa`` via the
                                      ``_patch_follower_aa_send`` hook —
                                      avoids the ``roll ≈ ±π`` Euler
                                      discontinuity that produces the
                                      "RX flip" symptom during policy
                                      execution.
          * ``"cartesian_rel"``     — ``[dxyz_m, drot_rad, grip]``
                                      (Euler delta). Note: robopy adds
                                      this to ``get_position_aa`` and
                                      sends via Euler ``set_position``,
                                      which is itself approximate for
                                      non-trivial rotations.
        """
        # Gate: if the controller has a latched fault, block here until
        # it is cleared (either by a human on the pendant or by the
        # auto-recover path). Prevents the caller's action from being
        # silently dropped by the SDK while the log fills with
        # ``code=1`` / ``code=2`` rejections.
        with _PROF.stage("sdk_gate"):
            self._wait_until_sdk_clear(tag="step")

        if isinstance(action, dict):
            atype = action.get("type", "cartesian_rel")
            avec = np.asarray(action["action"], dtype=np.float32).flatten()
        else:
            atype = "cartesian_rel"
            avec = np.asarray(action, dtype=np.float32).flatten()
        if avec.shape[0] < 7:
            raise ValueError(f"action must have ≥7 elements, got {avec.shape[0]}")
        pose_mm_rad = np.array([
            avec[0] * 1000.0, avec[1] * 1000.0, avec[2] * 1000.0,
            avec[3], avec[4], avec[5],
        ], dtype=np.float32)
        grip_pm1 = float(avec[6])
        grip_norm = float(np.clip((1.0 - grip_pm1) / 2.0, 0.0, 1.0))  # +1→0, -1→1
        # Send a fresh gripper command ONLY when the target moves more
        # than ``_GRIP_CHANGE_THRESHOLD`` — otherwise pass None so the
        # control thread skips its 50 Hz modbus write. The threshold
        # gates near-noise oscillation introduced by TE / averaged
        # chunks, which otherwise floods the SDK with
        # ``set_modbus_gripper_position code=2`` (gripper busy).
        if self._last_grip_norm is None or \
                abs(grip_norm - self._last_grip_norm) > self._GRIP_CHANGE_THRESHOLD:
            grip_for_thread = grip_norm
            self._last_grip_norm = grip_norm
        else:
            grip_for_thread = None
        with _PROF.stage("send_cmd"):
            self._dispatch_command(atype, pose_mm_rad, grip_for_thread, avec)
        # ``return_obs=False`` skips this read entirely. Every caller in this
        # script discards it and then calls ``_compute_step_obs`` →
        # ``env.get_obs()`` for the frame it actually records, so reading here
        # was a second full camera + robot-state round-trip per control step
        # (measured on hardware: ~17 ms/step, ~15% of the loop). The recorded
        # frame is unchanged — this only drops the throw-away one.
        obs = self.get_obs() if return_obs else None
        info = {"success": False}
        return obs, 0.0, False, info

    def _dispatch_command(self, atype, pose_mm_rad, grip_for_thread, avec) -> None:
        """Hand one command to robopy's control thread (no observation read)."""
        if not self.dry_run:
            if atype == "cartesian_abs":
                # Clear any leftover AA marker from a prior step so
                # the control thread reverts to the Euler ``set_position``
                # path for this command.
                with self._follower._target_command_lock:
                    self._follower._target_command = {
                        "mode": "cartesian_abs",
                        "pose": pose_mm_rad.copy(),
                        "gripper": grip_for_thread,
                    }
            elif atype == "cartesian_abs_aa":
                # Tag the cmd dict with ``format="aa"`` so the patched
                # ``_send_cartesian`` routes pose through
                # ``set_position_aa`` instead of ``set_position``.
                with self._follower._target_command_lock:
                    self._follower._target_command = {
                        "mode": "cartesian_abs",
                        "pose": pose_mm_rad.copy(),
                        "gripper": grip_for_thread,
                        "format": "aa",
                    }
            elif atype == "cartesian_rel":
                self._follower.command_cartesian_relative(pose_mm_rad, gripper=grip_for_thread)
            elif atype == "joint_abs":
                # Local-IK pipeline: target is already in joint space,
                # so we hand it straight to the follower as a joint
                # command — no controller IK is invoked. ``avec`` is
                # ``[j1..j7, grip_pm1]`` (rad + ±1); robopy's
                # ``command_joint_state`` takes the 8-vec and routes
                # the gripper for us.
                joint_target = np.concatenate(
                    [avec[:7], [grip_for_thread if grip_for_thread is not None
                                else (self._last_grip_norm or 0.0)]],
                ).astype(np.float32)
                self._follower.command_joint_state(joint_target)
                # ``command_joint_state`` always sets gripper in the
                # cached target_command; immediately overwrite it to
                # None when we're rate-limiting so the next 50 Hz tick
                # doesn't re-send the same modbus position.
                if grip_for_thread is None:
                    with self._follower._target_command_lock:
                        self._follower._target_command["gripper"] = None
            else:
                raise ValueError(f"unknown action type: {atype!r}")

    # ── observation ─────────────────────────────────────────
    def get_tcp_xyz(self) -> np.ndarray:
        """TCP xyz only, straight from the cached robot state.

        ``get_obs()`` would also pull a *fresh* frame from both cameras, which
        blocks on the RealSense stream. Callers that only need the arm pose —
        ``_select_nearest_efe_target`` picking the affordance candidate closest
        to the TCP — must not pay for two camera reads to get three floats.
        """
        if self.dry_run:
            return np.array([0.5, 0.0, 0.3], dtype=np.float32)
        with _PROF.stage("robot_state"):
            return self._follower.get_ee_pos_quat()[:3].astype(np.float32)

    def get_obs(self) -> dict:
        """CALVIN-style flat observation dict (real-world flavour).

        Keys (used downstream by AffordanceWrapperRealWorld + our policy
        pipeline):
          * ``rgb_static``, ``depth_static``  (flat, real-world layout)
          * ``rgb_gripper``, ``depth_gripper``
          * ``rgb_obs``, ``depth_obs`` (nested duplicates for sim-style consumers)
          * ``robot_obs`` (15-DoF: ee_pos, ee_quat, joints, gripper)
          * ``robot_state`` (dict with tcp_pos/tcp_orn/gripper_opening_width)
          * ``ee_pos_quat``
        """
        # Timing for these lives inside CameraView.get_image so that frame
        # grabs from other call paths (TargetSearch's EFE scan) are counted
        # under the same cam_* rows instead of hiding in their caller.
        rgb_s, depth_s = self._static_view.get_image()
        rgb_g, depth_g = self._gripper_view.get_image()
        if self.dry_run:
            ee = np.array([0.5, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            joints = self._start_joints_rad.astype(np.float32)
            gripper_norm = np.float32(0.0)
        else:
            with _PROF.stage("robot_state"):
                ee = self._follower.get_ee_pos_quat().astype(np.float32)
                js = self._follower.get_joint_state().astype(np.float32)
            joints = js[:7]
            gripper_norm = float(js[7]) if js.shape[0] > 7 else 0.0
        robot_obs = np.concatenate(
            [ee[:3], ee[3:7], joints, [gripper_norm]]
        ).astype(np.float32)
        robot_state = {
            "tcp_pos": ee[:3].astype(np.float32),
            "tcp_orn": ee[3:7].astype(np.float32),
            "gripper_opening_width": float(
                np.clip(1.0 - float(gripper_norm), 0.0, 1.0)
            ) * GRIPPER_OPEN_WIDTH_M,
        }
        # Both flat and nested keys so either pipeline branch finds what it expects.
        return {
            "rgb_static": rgb_s,
            "depth_static": depth_s,
            "rgb_gripper": rgb_g,
            "depth_gripper": depth_g,
            "rgb_obs": {"rgb_static": rgb_s, "rgb_gripper": rgb_g},
            "depth_obs": {"depth_static": depth_s, "depth_gripper": depth_g},
            "robot_obs": robot_obs,
            "robot_state": robot_state,
            "ee_pos_quat": ee,
        }


# ───────────────────────────────────────────── affordance forward ──

def build_affordance_models(
    affordance_cfg, img_size: int,
    aff_seed=None, aff_dataset_name=None,
    static_resize=None, gripper_resize=None,
):
    """Load (gripper_aff_net, static_aff_net, aff_transforms) once.

    ``{static,gripper}_resize`` must match the *training-time* input
    shapes that the model was learned on (typically rectangular
    ``[H, W]`` — see ``DEFAULT_{STATIC,GRIPPER}_AFF_SIZE``). Passing a
    bare scalar would make torchvision Resize squash the input to a
    square, and the downstream pixel→world projection would land on the
    wrong object — see :func:`viz_affordances.viz` for the reference
    pipeline.
    """
    transforms_cfg = affordance_cfg.transforms["validation"]
    static_resize = list(static_resize) if static_resize else DEFAULT_STATIC_AFF_SIZE
    gripper_resize = list(gripper_resize) if gripper_resize else DEFAULT_GRIPPER_AFF_SIZE
    # ``aff_shape`` (= input_channels) probe — square test tensor at
    # ``img_size`` is fine here since we only need shape[0] afterwards.
    _, aff_shape = get_transforms_and_shape(transforms_cfg, img_size)
    aff_transforms = {
        "gripper": get_transforms_and_shape(
            transforms_cfg, in_size=max(gripper_resize), out_size=gripper_resize,
        )[0],
        "static":  get_transforms_and_shape(
            transforms_cfg, in_size=max(static_resize), out_size=static_resize,
        )[0],
    }
    logger.info(
        f"[affordance] input resize (HxW): static={static_resize} "
        f"gripper={gripper_resize}"
    )
    in_channels = aff_shape[0]
    gripper_net = init_aff_net(
        affordance_cfg, "gripper", in_channels,
        seed=aff_seed, dataset_name=aff_dataset_name,
    )
    static_net = init_aff_net(
        affordance_cfg, "static", in_channels, use=True,
        seed=aff_seed, dataset_name=aff_dataset_name,
    )
    # Attach the training-time static crop (if any) so per-step static forwards
    # (``_compute_step_obs``) feed the same cropped frame the net was trained on.
    if static_net is not None:
        static_net._static_crop = load_static_crop_for_run(
            affordance_cfg, aff_seed, aff_dataset_name)
    return gripper_net, static_net, aff_transforms, aff_shape


def predict_gripper_affordance(net, rgb_hwc_uint8: np.ndarray,
                                aff_transforms_gripper) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the gripper affordance net on a single HWC uint8 RGB frame.

    Returns ``(aff_mask_hw, aff_probs_hwc, centers_hw_directions)``. Matches
    the keys ``find_target_center`` in aff_wrapper_base consumes
    (``gripper_aff``, ``gripper_aff_probs``, ``gripper_center_dir``).
    """
    if net is None:
        # Return zeros so downstream consumers degrade gracefully.
        h, w = rgb_hwc_uint8.shape[:2]
        return (np.zeros((1, h, w), dtype=np.uint8),
                np.zeros((1, h, w, 2), dtype=np.float32),
                np.zeros((1, h, w, 2), dtype=np.float32),
                None)
    # Split into two profiler scopes: ``aff_g_prep`` is the CPU torchvision
    # transform plus the host→device copy, ``aff_g_fwd`` the net itself. They
    # scale with completely different things (frame resolution vs model size),
    # so lumping them together hides which one to attack.
    with _PROF.stage("aff_g_prep"):
        img_chw = np.transpose(rgb_hwc_uint8, (2, 0, 1))
        processed = aff_transforms_gripper(torch.from_numpy(img_chw).float())
        obs_jax = _jax_sync(jnp.asarray(processed.unsqueeze(0).numpy()))
    # orientation 予測対応モデルなら orient_6d も取得。
    orient_6d = None
    with _PROF.stage("aff_g_fwd"):
        if getattr(net, "predict_orientation", False) and hasattr(net, "forward_orient"):
            _, aff_probs, aff_mask, directions, orient_6d = _jax_sync(
                net.forward_orient(obs_jax))
        else:
            _, aff_probs, aff_mask, directions = _jax_sync(net.forward(obs_jax))
    return aff_mask, aff_probs, directions, orient_6d


# ──────────────────── affordance viz (aif_collection_uv parity) ──
# The viz / save layout below mirrors ``scripts/aif_collection_uv.py`` so a
# real-robot run can be reviewed with exactly the same tooling as a sim run:
#   * static-cam episode-init detection → ``<episode_path>/affordance_init/``
#   * per-step gripper-cam direction field →
#     ``<episode_path>/gripper/rgb_gripper_aff_dirs.mp4``
# Both live INSIDE the episode directory (uv's convention) rather than in the
# run-level ``init_viz/`` + ``gripper_aff_viz/`` PNG dumps this script used to
# write. ``_aff_viz_images`` is the single renderer for every affordance
# overlay this script produces — episode-init, per-step, and the
# ``--debug_affordance`` modes — so the naming can't drift apart again.


def _aff_viz_images(orig_img, aff_mask, directions, centers, cam_type) -> dict:
    """Render uv's ``viz_aff_centers_preds`` image set for one frame.

    Returns ``{name: BGR image}`` keyed ``<cam_type>_{orig,masks,aff,dirs}``
    at ``orig_img``'s native resolution, ready to hand to ``cv2.imwrite``
    (``viz_aff_centers_preds`` already flips RGB→BGR).

    ``get_aff_imgs`` (called inside ``viz_aff_centers_preds``) hands its
    ``out_shape`` to ``cv2.resize`` as ``(w, h)`` but to ``resize_center`` as
    ``(h, w)`` — the two only agree when the shape is square, which is why
    ``viz_affordances.py`` always renders square. aif_collection_uv never hits
    this (sim frames are square); the real cameras are 640x480, so we render
    square (``S = max(H, W)``) and undo the stretch afterwards.
    """
    orig_img = np.ascontiguousarray(np.asarray(orig_img).astype(np.uint8))
    H, W = orig_img.shape[:2]
    S = int(max(H, W))
    viz_dict = viz_aff_centers_preds(
        orig_img, aff_mask, directions, centers,
        cam_type=cam_type, obs_it=0, viz=False, resize=(S, S),
    )
    out = {}
    for key, img in viz_dict.items():
        # key looks like './images/static_aff/img_0000.png'; keep only the
        # category name (e.g. 'static_aff'), as aif_collection_uv does.
        name = os.path.basename(os.path.dirname(key))
        img = np.asarray(img)
        if img.shape[:2] != (H, W):
            img = cv2.resize(img, (W, H))
        out[name] = np.ascontiguousarray(img)
    return out


def _filter_centers_by_robustness(centers, aff_probs, object_masks, min_robustness):
    """Drop affordance centers whose confidence (robustness = mean foreground
    probability over the center's instance-mask cluster) is below
    ``min_robustness``, mirroring the rejection in ``_refine_target_via_gripper``
    so the overlay shows the candidates actually kept. Returns the surviving
    centers (same order). Ported from ``aif_collection_uv``.
    """
    object_masks = np.asarray(object_masks)
    aff_probs = np.asarray(aff_probs)
    obj_class = np.unique(object_masks)
    obj_class = obj_class[obj_class != 0]                 # drop background
    n = min(len(centers), len(obj_class))
    fg_channel = min(1, aff_probs.shape[-1] - 1)
    kept = []
    for i in range(n):
        cluster = aff_probs[object_masks == obj_class[i], fg_channel]
        if cluster.size and float(np.mean(cluster)) >= min_robustness:
            kept.append(centers[i])
    return kept


def gripper_affordance_dir_frame(gripper_net, gripper_rgb_hwc, aff_raw,
                                 min_robustness: float = 0.0):
    """Return an RGB ``(H, W, 3)`` uint8 frame of the gripper-cam affordance
    direction (center-vector) field overlaid on the gripper observation, to be
    written to an MP4 alongside ``rgb_gripper.mp4``. Mirrors
    ``aif_collection_uv.gripper_affordance_dir_frame``.

    ``aff_raw`` is the ``{"mask", "probs", "dirs"}`` bundle
    ``_compute_step_obs`` already produced for this step, so — like uv, which
    reuses the wrapper's cached ``env.gripper_dirs_overlay`` instead of
    re-forwarding — no second net forward runs on the 30 Hz control loop; only
    the (cheap) center clustering plus drawing.

    When ``min_robustness`` > 0 low-confidence centers are dropped here too
    (same threshold as ``_refine_target_via_gripper``'s rejection), so the MP4
    shows only the post-rejection candidates. Returns ``None`` when the
    gripper aff net is unavailable or the overlay fails (caller skips the
    frame).
    """
    if gripper_net is None or aff_raw is None:
        return None
    try:
        mask = np.asarray(aff_raw["mask"])
        centers, dirs_out, object_masks = gripper_net.get_centers(
            mask, aff_raw["dirs"])
        centers = [np.asarray(c) for c in centers]
        mask_2d = (mask[0] if mask.ndim == 3 else mask).astype(np.uint8)
        dirs_2d = np.asarray(dirs_out)[0]
        probs_2d = np.asarray(aff_raw["probs"])[0]
        obj_masks_2d = np.asarray(object_masks)[0]
        if min_robustness > 0.0 and len(centers) > 0:
            centers = _filter_centers_by_robustness(
                centers, probs_2d, obj_masks_2d, min_robustness)
        imgs = _aff_viz_images(
            gripper_rgb_hwc, mask_2d, dirs_2d, centers, "gripper")
    except Exception as exc:
        logger.debug(f"[gripper-aff-dirs] skipping frame: {exc}")
        return None
    frame = imgs.get("gripper_dirs")
    if frame is None:
        return None
    # _aff_viz_images returns BGR (cv2.imwrite convention); flip back to RGB
    # for imageio's MP4 writer.
    return np.ascontiguousarray(frame[:, :, ::-1])


# ─────────────────────────────────────── target search wrapper ──

def build_target_search(env: XArmRealEnv, run_cfg,
                        aff_seed=None, aff_dataset_name=None,
                        static_resize=None):
    """Construct vapo's TargetSearch in real_world mode bound to our env.

    ``TargetSearch.__init__`` internally calls ``init_aff_net(aff_cfg)``
    without forwarding ``seed`` / ``dataset_name``, so its
    ``aff_net_static_cam`` always falls back to the legacy generic
    checkpoint path (``trained_models/affordance/static/best_val_miou.ckpt``)
    -- *not* the dataset-specific weights at
    ``trained_models/affordance/<dataset_label>/static/seed:<seed>/...``.
    To keep the target-search affordance consistent with what
    ``build_affordance_models`` loads (which *does* honour
    ``aff_dataset_name``), we re-resolve the net here and swap it in
    after construction. No vapo-core changes required.

    ``static_resize`` (HxW) selects the resize fed to the affordance net
    via ``aff_transforms``. Must match the training-time shape (e.g.
    ``[96, 128]``) -- otherwise the detected target pixel is in a squashed
    coordinate system and the deprojected world point will be off.
    """
    from vapo.agent.core.target_search import TargetSearch
    static_resize = list(static_resize) if static_resize else DEFAULT_STATIC_AFF_SIZE
    aff_transforms = get_transforms(
        run_cfg.affordance.transforms.validation,
        static_resize,
    )
    args = {
        "initial_pos": np.zeros(3, dtype=np.float32),
        "aff_transforms": aff_transforms,
        # Forward seed/dataset so TargetSearch.__init__ resolves the
        # dataset-specific weights at
        # ``trained_models/affordance/<dataset_label>/static/seed:<seed>/...``
        # at construction time (instead of falling back to the legacy
        # generic ``.../static/best_val_miou.ckpt`` and raising).
        "aff_seed": aff_seed,
        "aff_dataset_name": aff_dataset_name,
        **run_cfg.target_search,
    }
    # Force the mode to real_world regardless of what config says.
    args["mode"] = "real_world"
    ts = TargetSearch(env, **args)
    # Apply the training-time static crop to *only* TargetSearch's view: wrap
    # its static_cam so the affordance net sees the trained field of view while
    # deprojection stays exact (offset principal point). Other consumers of the
    # static camera are unaffected. No-op when no crop was saved.
    crop_info = load_static_crop_for_run(run_cfg.affordance, aff_seed, aff_dataset_name)
    if crop_info is not None:
        ts.static_cam = CroppedStaticCam(ts.static_cam, crop_info)
    return ts


def _save_static_affordance_viz(env: XArmRealEnv, target_search,
                                all_targets, selected_world_pt,
                                episode_path) -> None:
    """Save the static-cam affordance detection viz for one episode start.

    Same output set, names and layout as
    ``aif_collection_uv._save_static_affordance_viz`` — everything lands in
    ``<episode_path>/affordance_init/`` (BGR on disk, cv2.imwrite convention):

      * ``static_orig.png``     — the static RGB the net was run on (already
                                   cropped when a training-time static crop is
                                   active)
      * ``static_masks.png``    — predicted foreground mask
      * ``static_aff.png``      — RGB with the affordance-mask overlay
      * ``static_dirs.png``     — RGB with the center-direction vectors overlay
      * ``static_selected.png`` — RGB with all detected candidates marked
                                   (red cross) and the picked one circled
                                   (green)

    Called BEFORE ``move_to_target`` so the snapshot survives a failed move.
    """
    save_dir = Path(episode_path) / "affordance_init"
    save_dir.mkdir(parents=True, exist_ok=True)

    # The frame TargetSearch actually ran the net on (cropped when a static
    # crop is active), so the markers below line up with the overlays. uv
    # reads ``env.get_obs()`` here instead, which is the same frame in sim —
    # on hardware only the cached cropped one is guaranteed to match.
    orig_img = np.asarray(target_search.orig_img)
    if orig_img.ndim == 3 and orig_img.shape[0] == 3:
        orig_img = orig_img.transpose(1, 2, 0)
    orig_img = np.ascontiguousarray(orig_img.astype(np.uint8))

    try:
        centers, aff_mask, directions, _probs, _init_masks = transform_and_predict(
            target_search.aff_net_static_cam,
            target_search.aff_transforms,
            orig_img,
            class_label=getattr(target_search, "class_label", None),
        )
        for name, img in _aff_viz_images(
                orig_img, aff_mask, directions, centers, "static").items():
            cv2.imwrite(str(save_dir / f"{name}.png"), img)
    except Exception as exc:
        # Keep going: static_selected.png below is the one that actually
        # explains where the arm is about to move.
        logger.warning(f"[affordance-viz] mask/dirs overlays skipped "
                       f"(NN forward failed): {exc}")
        cv2.imwrite(str(save_dir / "static_orig.png"), orig_img[:, :, ::-1])

    # Annotated overlay: red cross on every detected candidate + green circle
    # on the one that was picked. Project through the SAME camera TargetSearch
    # used (the cropped wrapper when a crop is active) so markers land on
    # ``orig_img``.
    static_cam = (getattr(target_search, "static_cam", None)
                  or env.camera_manager.static_cam)

    def _world_to_uv(world_pt):
        p = np.asarray(world_pt, dtype=np.float64).reshape(-1)[:3]
        try:
            u, v = static_cam.project(p)
        except Exception:
            return -1, -1
        return int(u), int(v)

    annotated = orig_img.copy()
    H, W = annotated.shape[:2]
    n_drawn = 0
    for tgt in all_targets:
        wp = tgt["target_pos"] if isinstance(tgt, dict) else tgt
        wp = np.asarray(wp, dtype=np.float32).reshape(-1)[:3]
        u, v = _world_to_uv(wp)
        if 0 <= u < W and 0 <= v < H:
            cv2.drawMarker(annotated, (u, v), (255, 0, 0),
                           markerType=cv2.MARKER_CROSS,
                           markerSize=14, thickness=2)
            n_drawn += 1
        else:
            logger.warning(
                f"[affordance-viz] candidate at world={wp.tolist()} projects "
                f"to off-image pixel (u={u}, v={v}); skipping marker")

    sel_arr = np.asarray(selected_world_pt, dtype=np.float32).reshape(-1)[:3]
    su, sv = _world_to_uv(sel_arr)
    if 0 <= su < W and 0 <= sv < H:
        cv2.circle(annotated, (su, sv), 14, (0, 255, 0), 2)
    else:
        logger.warning(
            f"[affordance-viz] selected at world={sel_arr.tolist()} projects "
            f"to off-image pixel (u={su}, v={sv}); circle skipped")
    cv2.imwrite(str(save_dir / "static_selected.png"), annotated[:, :, ::-1])
    logger.info(f"[affordance-viz] saved → {save_dir} "
                f"({n_drawn}/{len(all_targets)} candidate markers drawn)")


def affordance_random_init(env: XArmRealEnv, target_search,
                           rng: np.random.RandomState,
                           episode_path: str | Path | None = None,
                           z_offset_m: float = 0.0,
                           use_orientation: bool = True) -> List[dict]:
    """Detect a target via static-cam affordance, pick one, move to it.

    Mirrors aif_collection_uv.affordance_random_init. If ``episode_path``
    is given, the static-cam affordance viz is written to
    ``<episode_path>/affordance_init/`` *before* motion is commanded (so the
    record is preserved even if the move fails).

    ``z_offset_m`` is added to the selected target's z before the move
    command, so the arm hovers above (positive) or dips below (negative)
    the detected affordance point instead of plunging onto it. The
    *stored* target attributes (``curr_detected_obj`` /
    ``static_initial_target`` / ``target_pos``) keep the raw detection
    so the downstream gripper-cam refinement and EFE extrinsic still
    target the actual object, not the offset hover point.
    """
    target_pos, no_target, all_targets = target_search.compute(
        env, return_all_centers=True, rand_sample=True,
    )
    if no_target or not all_targets:
        tcp = np.asarray(env.get_obs()["robot_obs"][:3], dtype=np.float32).copy()
        env.curr_detected_obj = tcp
        env.static_initial_target = tcp
        env.target_pos = tcp
        logger.warning("[affordance] no target detected; using TCP as fallback")
        if episode_path is not None:
            try:
                _save_static_affordance_viz(
                    env, target_search, [], tcp, episode_path)
            except Exception as exc:
                logger.warning(f"[affordance-viz] save failed: {exc}")
        return []
    pick_idx = int(rng.randint(len(all_targets)))
    pick = all_targets[pick_idx]
    selected = np.asarray(pick["target_pos"] if isinstance(pick, dict) else pick,
                          dtype=np.float32).copy()
    logger.info(
        f"[affordance] selected target #{pick_idx}/{len(all_targets)} "
        f"at xyz_m={selected.tolist()}"
    )
    # Save viz BEFORE move so the snapshot is preserved even if the move
    # collides with workspace limits or fails.
    if episode_path is not None:
        try:
            _save_static_affordance_viz(
                env, target_search, all_targets, selected, episode_path)
        except Exception as exc:
            logger.warning(f"[affordance-viz] save failed: {exc}")
    env.curr_detected_obj = selected
    env.static_initial_target = selected
    env.target_pos = selected
    # affordance が予測した orientation(world euler)。未予測/無効なら None(=home姿勢)。
    _orns = getattr(target_search, "last_orientations", None)
    target_orn = (_orns[pick_idx] if use_orientation and _orns
                  and pick_idx < len(_orns) else None)
    if target_orn is not None:
        env.target_orn = np.asarray(target_orn, dtype=np.float32)[:3]
    move_target = selected.copy()
    if float(z_offset_m) != 0.0:
        move_target[2] = float(move_target[2]) + float(z_offset_m)
        logger.info(
            f"[affordance] applying z offset {z_offset_m:+.3f} m: "
            f"move_to xyz_m={move_target.tolist()} (stored target unchanged)"
        )
    return env.move_to_target(move_target, target_orn=target_orn)


# ──────────────────────────────────────── policy / world builders ──
#   (ported verbatim from scripts/aif_collection_uv.py:253-386)

def _maybe_merge_ebm_phase1(config) -> Optional[str]:
    target = config.policy.get("_target_", "")
    if "EBMPolicyConfig" not in target:
        return None
    phase1_name = config.policy.get("phase1_config", None)
    if phase1_name is None:
        return None
    phase1_cfg = OmegaConf.load(f"config/policy/{phase1_name}.yaml")
    if config.policy.get("policy", None) is None:
        config.policy.policy = phase1_cfg.policy.policy
    if config.policy.get("framework_cfg", None) is None:
        config.policy.framework_cfg = phase1_cfg.policy.framework_cfg
    for k in ("ot_method", "sinkhorn_epsilon", "use_uniform",
              "policy_length", "obs_length", "pred_obs_action"):
        if config.policy.get(k, None) is None:
            v = phase1_cfg.policy.get(k, None)
            if v is not None:
                config.policy[k] = v
    if not config.policy.get("ema", None):
        config.policy.ema = phase1_cfg.policy.get("ema", {})
    return phase1_name


def build_policy(policy_cfg_name: str, data_cfg, seed: int, dataset: str):
    config = OmegaConf.load(f"config/policy/{policy_cfg_name}.yaml")
    config.seed = seed
    config.datamodule.env = dataset
    phase1_name = _maybe_merge_ebm_phase1(config)
    config.action_dim = len(config.datamodule.use_joint_indices)
    static_resize = config.datamodule.static_resize
    gripper_resize = config.datamodule.gripper_resize
    config.static_obs_shape = [data_cfg.static_obs_shape[0], static_resize, static_resize]
    depth_channels = 1 if config.datamodule.use_depth else 0
    config.gripper_obs_shape = [
        data_cfg.gripper_obs_shape[0] + depth_channels, gripper_resize, gripper_resize,
    ]
    OmegaConf.resolve(config)
    policy_cfg = instantiate(config)
    if isinstance(policy_cfg.policy, EBMPolicyConfig):
        policy = EBMPolicy(policy_cfg.policy)
    elif isinstance(policy_cfg.policy, VQBeTConfig):
        policy = VQBeTPolicy(policy_cfg.policy)
    elif isinstance(policy_cfg.policy, IMLEConfig):
        policy = IMLEPolicy(policy_cfg.policy)
    elif isinstance(policy_cfg.policy, ACTConfig):
        policy = ACTPolicy(policy_cfg.policy)
    elif isinstance(policy_cfg.policy.policy, UNetConfig):
        policy = Policy(policy_cfg.policy)
    elif isinstance(policy_cfg.policy.policy, (TransformerConfig, DiTConfig)):
        policy = TransformerPolicy(policy_cfg.policy)
    elif isinstance(policy_cfg.policy.policy, (ml.MLPConfig, RNNConfig)):
        policy = SimplePolicy(policy_cfg.policy)
    else:
        raise ValueError(
            f"Unknown policy type: {type(policy_cfg.policy).__name__} / "
            f"{type(policy_cfg.policy.policy).__name__}"
        )
    return policy, policy_cfg, phase1_name


def configure_ebm_sampler(policy, sampler=None, dfo_iters=None,
                          dfo_noise_scale=0.33, dfo_noise_shrink=0.5):
    """Override the EBM action-generation sampler at collection time.

    No-op for non-EBM policies or when ``sampler`` is None (keeps the trained
    default path). ``sampler`` selects how the N(0,1) latent candidates are
    refined before the phase-1 denoise (see vapo.policy.policy.EBMPolicy):
    'default', 'langevin', 'dfo' (derivative-free optimiser), or 'langevin_dfo'
    (Langevin MCMC → DFO)."""
    if sampler is None or not isinstance(policy, EBMPolicy):
        return
    policy.ebm_sampler = sampler
    if dfo_iters is not None:
        policy.ebm_cfg.dfo_iters = dfo_iters
    policy.ebm_cfg.dfo_noise_scale = dfo_noise_scale
    policy.ebm_cfg.dfo_noise_shrink = dfo_noise_shrink
    logger.info(f"[ebm-sampler] using '{sampler}'"
                + (f" (dfo_iters={dfo_iters},"
                   f" noise_scale={dfo_noise_scale},"
                   f" noise_shrink={dfo_noise_shrink})"
                   if sampler in ("dfo", "langevin_dfo") else ""))


def add_ebm_sampler_args(parser):
    """Register the shared ``--ebm-sampler`` / DFO CLI flags on ``parser``."""
    parser.add_argument(
        "--ebm-sampler", dest="ebm_sampler", type=str, default=None,
        choices=["default", "langevin", "dfo", "langevin_dfo"],
        help="EBMPolicy action-generation sampler override (EBM policies only). "
             "'dfo'=derivative-free optimiser, 'langevin_dfo'=Langevin MCMC "
             "then DFO. Omit to keep the trained default path.")
    parser.add_argument("--dfo-iters", dest="dfo_iters", type=int, default=None,
                        help="DFO iteration count (defaults to langevin_steps).")
    parser.add_argument("--dfo-noise-scale", dest="dfo_noise_scale",
                        type=float, default=0.33,
                        help="Initial DFO Gaussian noise scale.")
    parser.add_argument("--dfo-noise-shrink", dest="dfo_noise_shrink",
                        type=float, default=0.5,
                        help="Per-iteration DFO noise shrink factor.")


def load_policy_checkpoint(policy, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "jax_state" in ckpt:
        jax_state = jax.tree.map(jnp.asarray, ckpt["jax_state"])
        policy._set_jax_state(jax_state)
        if hasattr(policy, "_params"):
            policy._params = jax_state
        if "jax_ema_state" in ckpt and hasattr(policy, "ema"):
            ema_state = jax.tree.map(jnp.asarray, ckpt["jax_ema_state"])
            policy.ema.load_state_dict(ema_state)
        elif hasattr(policy, "ema") and "policy" in jax_state:
            logger.warning("No EMA state in checkpoint. Using model params as EMA.")
            policy.ema.ema_state = jax.tree.map(jnp.asarray, jax_state["policy"])
        if "jax_ebm_ema_state" in ckpt and hasattr(policy, "ebm_ema"):
            ebm_ema_state = jax.tree.map(jnp.asarray, ckpt["jax_ebm_ema_state"])
            policy.ebm_ema.load_state_dict(ebm_ema_state)
    else:
        policy.load_state_dict(ckpt["state_dict"])
    policy.eval()
    policy.freeze()


def maybe_load_ebm_phase1(policy, policy_cfg, dataset, phase1_name,
                          n_train=None):
    if (phase1_name is None
            or not isinstance(policy_cfg.policy, EBMPolicyConfig)
            or getattr(policy_cfg.policy, "phase", None) != "ebm"):
        return
    # Mirror train_policy.py's "+n{n_train}" checkpoint-dir suffix.
    phase1_dir = phase1_name if n_train is None else f"{phase1_name}+n{n_train}"
    gen_ckpt = (f"trained_models/policy/{policy_cfg.datamodule.env}/{phase1_dir}/"
                f"seed:{policy_cfg.seed}/model_best_loss.ckpt")
    if not os.path.exists(gen_ckpt):
        gen_ckpt = (f"trained_models/policy/{policy_cfg.datamodule.env}/{phase1_dir}/"
                    f"seed:{policy_cfg.seed}/last.ckpt")
    if os.path.exists(gen_ckpt):
        logger.info(f"[ebm] Loading phase1 generative model from {gen_ckpt}")
        policy._load_generative_checkpoint(gen_ckpt)


def build_world(world_cfg_name: str, data_cfg, seed: int, dataset: str):
    config = OmegaConf.load(f"config/world/{world_cfg_name}.yaml")
    config.seed = seed
    config.datamodule.env = dataset
    config.action_dim = data_cfg.action_dim
    static_resize = config.datamodule.static_resize
    gripper_resize = config.datamodule.gripper_resize
    config.static_obs_shape = [data_cfg.static_obs_shape[0], static_resize, static_resize]
    # Match the policy-side convention: only add the depth channel when
    # the world model was actually trained with depth. ``aif_collection_uv.py``
    # hardcoded ``+1`` because sim configs always set ``use_depth=True``;
    # but real-world world models (e.g. ``mtrssm_am_nodep_ll``) are RGB-only,
    # so an unconditional ``+1`` makes the decoder reshape from a (..., 3)
    # JAX tensor into a (..., 4) target shape and crash inside
    # ``_calc_efe_jax``.
    world_use_depth = bool(getattr(config.datamodule, "use_depth", True))
    depth_channels = 1 if world_use_depth else 0
    config.gripper_obs_shape = [
        data_cfg.gripper_obs_shape[0] + depth_channels,
        gripper_resize, gripper_resize,
    ]
    world_cfg = instantiate(config)
    return WorldModel(world_cfg.world), world_cfg


def load_world_checkpoint(world, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "jax_state" in ckpt:
        jax_state = jax.tree.map(jnp.asarray, ckpt["jax_state"])
        world._set_jax_state(jax_state)
        if "jax_opt_state" in ckpt:
            opt_state = jax.tree.map(jnp.asarray, ckpt["jax_opt_state"])
            world._set_opt_state(opt_state)
    else:
        world.load_state_dict(ckpt["state_dict"])
    world.eval()
    world.freeze()


# ─────────────────────────────── jax obs helpers (lifted) ──

def _resize_nchw_raw(x, target_h, target_w):
    x_nhwc = jnp.transpose(x, (0, 2, 3, 1))
    x_nhwc = jax.image.resize(
        x_nhwc, (x.shape[0], target_h, target_w, x.shape[1]), method="bilinear")
    return jnp.transpose(x_nhwc, (0, 3, 1, 2))


_resize_nchw = jax.jit(_resize_nchw_raw, static_argnums=(1, 2))


def resize_for_policy(policy_cfg, static_rgb, gripper_obs):
    sh = (getattr(policy_cfg.policy, "static_obs_shape", None)
          or policy_cfg.static_obs_shape)
    gh = (getattr(policy_cfg.policy, "gripper_obs_shape", None)
          or policy_cfg.gripper_obs_shape)

    def _resize(x, target_h, target_w):
        if x.ndim == 5:
            N, T = x.shape[:2]
            flat = x.reshape(N * T, *x.shape[2:])
            flat = _resize_nchw(flat, target_h, target_w)
            return flat.reshape(N, T, *flat.shape[1:])
        return _resize_nchw(x, target_h, target_w)

    return _resize(static_rgb, sh[1], sh[2]), _resize(gripper_obs, gh[1], gh[2])


# Cache of jit-compiled (transpose + encoder forward) closures keyed on the
# world model identity. The host-side cv2 resize stays in Python — it's almost
# always a no-op once shapes match — so the jit'd region is pure XLA.
_ENCODE_OBS_JIT_CACHE: Dict[int, object] = {}


def _build_encode_obs_jit(world):
    has_static = world.static_encoder is not None

    @jax.jit
    def _fn(static_obs, gripper_obs):
        if has_static:
            s = jnp.transpose(static_obs, (0, 2, 3, 1))
            static_emb = world.static_encoder(s)
            static_emb = world.static_encoder_fc(static_emb)
        else:
            static_emb = None
        g = jnp.transpose(gripper_obs, (0, 2, 3, 1))
        gripper_emb = world.gripper_encoder(g)
        gripper_emb = world.gripper_encoder_fc(gripper_emb)
        return static_emb, gripper_emb

    return _fn


def _gripper_input(jax_obs: dict, use_depth: bool) -> jnp.ndarray:
    """Assemble the gripper input the way the (world / policy) model expects.

    For ``use_depth=True`` models, returns ``[rgb, depth]`` concatenated on
    the channel axis (4 channels for RGB+D). For ``use_depth=False`` models
    (suffix ``-nodep`` configs), returns RGB only (3 channels). The
    real_world dataset doesn't record depth at all, so feeding a 4-channel
    tensor to a 3-channel-trained network would error -- this helper is
    the single point that branches on the flag.
    """
    if use_depth:
        return jnp.concatenate(
            [jax_obs["gripper_rgb"], jax_obs["gripper_depth"]], axis=1,
        )
    return jax_obs["gripper_rgb"]


def encode_obs(world, static_obs, gripper_obs, obs_low=0.0, obs_high=1.0):
    cfg = world.cfg
    if world.static_encoder is not None:
        sh = cfg.static_obs_shape
        static_obs = _resize_nchw(static_obs, sh[1], sh[2])
    gh = cfg.gripper_obs_shape
    gripper_obs = _resize_nchw(gripper_obs, gh[1], gh[2])
    # get_obs_dataのRGB/depthは[0,1]。world側のみ学習時の正規化レンジ
    # [obs_low, obs_high]へ線形写像する（policy入力は元の[0,1]のまま）。
    if obs_low != 0.0 or obs_high != 1.0:
        scale = obs_high - obs_low
        gripper_obs = gripper_obs * scale + obs_low
        if world.static_encoder is not None:
            static_obs = static_obs * scale + obs_low
    key = id(world)
    fn = _ENCODE_OBS_JIT_CACHE.get(key)
    if fn is None:
        fn = _build_encode_obs_jit(world)
        _ENCODE_OBS_JIT_CACHE[key] = fn
    return fn(static_obs, gripper_obs)


# Cache one jit'd (dynamics.imagine + _calc_efe_jax) closure per world model
# identity. Mirrors WorldModel._create_jit_imagine_and_efe but exposes
# `disable_extrinsic` as a static arg so the aif data-collection path (where
# it toggles per-step based on affordance availability) caches one
# compilation per value (2 in total) instead of re-tracing on every flip.
# nnx.jit (not jax.jit) is required because dynamics.imagine reads/writes
# the dynamics' nnx Variables (prev_hiddens, prev_stoch); plain jax.jit
# would freeze them at trace time and step_dynamics updates between calls
# wouldn't be picked up.
_IMAGINE_EFE_JIT_CACHE: Dict[int, object] = {}


def _build_imagine_efe_jit(world):
    @partial(nnx.jit, static_argnames=("disable_extrinsic", "pos_time_reduce"))
    def _fn(core, policies, target_pos, pref_std, disable_extrinsic,
            pos_time_reduce):
        world._bind_core(core)
        states = world.dynamics.imagine(policies)
        efe, argmin_val, extrinsic, epistemic = world._calc_efe_jax(
            states, num_samples=2,
            target_pos=target_pos, pref_std=pref_std,
            disable_extrinsic=disable_extrinsic,
            pos_time_reduce=pos_time_reduce,
        )
        return efe, argmin_val, extrinsic, epistemic
    return _fn


def imagine_and_calc_efe(world, policies, target_pos, pref_std,
                         disable_extrinsic, pos_time_reduce="mean"):
    """JIT'd replacement for ``world.dynamics.imagine(policies) +
    world.calc_efe(states, ...)``.

    ``pos_time_reduce`` ("mean"|"min") selects how the target_pos extrinsic is
    reduced over the horizon (all-steps average vs. closest-approach step).

    Returns the same 4-tuple ``(efe, argmin, extrinsic, epistemic)``
    ``world.calc_efe`` returns (``argmin`` already converted to a Python int).
    """
    key = id(world)
    fn = _IMAGINE_EFE_JIT_CACHE.get(key)
    if fn is None:
        fn = _build_imagine_efe_jit(world)
        _IMAGINE_EFE_JIT_CACHE[key] = fn
    efe, argmin_val, extrinsic, epistemic = fn(
        world._core, policies, target_pos, pref_std, bool(disable_extrinsic),
        str(pos_time_reduce),
    )
    # _fn's `world._bind_core(core)` rebinds world.dynamics / world.pos_encoder_fc
    # / etc. to the *traced* core's submodules. Without rebinding back to the
    # concrete `_core`, the next non-jit call (e.g. world.step_dynamics) hits
    # an UnexpectedTracerError because it tries to use the leaked tracer refs.
    # WorldModel._sync_from_core() does exactly this re-bind; existing JIT'd
    # paths in modules.py (_jit_train / _jit_val) follow the same pattern.
    world._sync_from_core()
    return efe, int(argmin_val), extrinsic, epistemic


def _safe_resize_hwc(img_hwc, target_h, target_w):
    cur_h, cur_w = img_hwc.shape[0], img_hwc.shape[1]
    out_h = min(cur_h, target_h)
    out_w = min(cur_w, target_w)
    if cur_h == out_h and cur_w == out_w:
        return img_hwc
    return cv2.resize(img_hwc, (out_w, out_h))


def _processed_obs_from_raw(raw_obs: dict, gripper_aff: np.ndarray,
                            static_aff: np.ndarray, detected_target_pos: np.ndarray,
                            img_size: int) -> dict:
    """Bundle a raw env obs + freshly-computed affordance masks into the dict
    consumed by ``get_obs_data``. Mirrors aif_collection_uv's
    ``raw_obs_to_processed`` plus an inline affordance step."""
    return {
        "static_img_obs": np.transpose(raw_obs["rgb_static"], (2, 0, 1)),
        "gripper_img_obs": np.transpose(raw_obs["rgb_gripper"], (2, 0, 1)),
        "gripper_depth_obs": depth_preprocessing(raw_obs["depth_gripper"], img_size),
        "robot_obs": raw_obs["robot_obs"],
        "static_aff": static_aff,
        "gripper_aff": gripper_aff,
        "detected_target_pos": detected_target_pos,
    }


def get_obs_data(x: Dict[str, np.ndarray], data_cfg,
                 policy_action_type: str = "actions") -> Dict[str, jnp.ndarray]:
    """Convert processed obs → JAX dict consumed by policy/world inference.

    Ported verbatim from aif_collection_uv.get_obs_data (449-490) and
    extended with a joint-state ``robot_pos`` slice for
    ``policy_action_type == "joint_actions"``. The default behaviour
    (EE-pose ``pos``) is unchanged for ``actions`` / ``rel_actions`` /
    ``mixed`` policies.
    """
    out: Dict[str, jnp.ndarray] = {}
    static_rgb = x["static_img_obs"].astype(np.float32) / 255.0
    static_mask = x["static_aff"].astype(np.float32)
    static_mask_2d = (static_mask.transpose(1, 2, 0)[..., 0]
                      if static_mask.ndim == 3 else static_mask.squeeze())
    s_rgb_h, s_rgb_w = static_rgb.shape[1], static_rgb.shape[2]
    s_mask_h, s_mask_w = static_mask_2d.shape[0], static_mask_2d.shape[1]
    s_cfg_h, s_cfg_w = data_cfg["static_obs_shape"][1], data_cfg["static_obs_shape"][2]
    s_common_h = min(s_rgb_h, s_mask_h, s_cfg_h)
    s_common_w = min(s_rgb_w, s_mask_w, s_cfg_w)
    static_rgb = _safe_resize_hwc(
        static_rgb.transpose(1, 2, 0), s_common_h, s_common_w).transpose(2, 0, 1)
    out["static_rgb"] = jnp.asarray(static_rgb).reshape(1, *static_rgb.shape)
    static_mask_2d = _safe_resize_hwc(static_mask_2d, s_common_h, s_common_w)
    static_mask = static_mask_2d[np.newaxis, ...]
    out["static_mask"] = jnp.asarray(static_mask).reshape(1, *static_mask.shape)

    gripper_rgb = x["gripper_img_obs"].astype(np.float32) / 255.0
    # ``depth_gripper_max`` is only present in the simulation dataset config
    # (sim depth is in metres, normalised to ~1). The real-world dataset
    # config typically omits it because the recorder doesn't save depth.
    # Default to 1.0 so the divide is a no-op when the key is absent --
    # the policy / world model determine via their ``use_depth`` flag
    # whether the depth channel is actually consumed downstream.
    _depth_norm = float(data_cfg.get("depth_gripper_max", 1.0)) or 1.0
    gripper_depth = x["gripper_depth_obs"].astype(np.float32) / _depth_norm
    gripper_depth_2d = gripper_depth.squeeze()
    rgb_h, rgb_w = gripper_rgb.shape[1], gripper_rgb.shape[2]
    depth_h, depth_w = gripper_depth_2d.shape[0], gripper_depth_2d.shape[1]
    cfg_h, cfg_w = data_cfg["gripper_obs_shape"][1], data_cfg["gripper_obs_shape"][2]
    common_h = min(rgb_h, depth_h, cfg_h)
    common_w = min(rgb_w, depth_w, cfg_w)
    gripper_rgb = _safe_resize_hwc(
        gripper_rgb.transpose(1, 2, 0), common_h, common_w).transpose(2, 0, 1)
    out["gripper_rgb"] = jnp.asarray(gripper_rgb).reshape(1, *gripper_rgb.shape)
    gripper_depth_2d = _safe_resize_hwc(gripper_depth_2d, common_h, common_w)
    gripper_depth = gripper_depth_2d[np.newaxis, ...]
    out["gripper_depth"] = jnp.asarray(gripper_depth).reshape(1, *gripper_depth.shape)
    gripper_mask = x["gripper_aff"].astype(np.float32)
    out["gripper_mask"] = jnp.asarray(gripper_mask).reshape(1, *gripper_mask.shape)
    # ``robot_pos`` must mirror the action representation the policy
    # was trained on (see ``vapo/policy/dataset.py``). For EE-pose
    # action types the legacy first-N-columns slice is correct
    # (xyz + quat). For ``joint_actions`` the policy observes the
    # follower joint state — robot_obs columns [7:15] (= 7 joints +
    # gripper) — and we renormalise against the joint subrange of
    # pos_max/pos_min so the obs distribution matches training.
    if policy_action_type == "joint_actions":
        pos_raw = x["robot_obs"].astype(np.float32)[..., 7:15]
        pos_max = np.asarray(data_cfg["pos_max"])[7:15]
        pos_min = np.asarray(data_cfg["pos_min"])[7:15]
        pos = joint_transform(pos_raw, pos_max, pos_min)
    else:
        pos = x["robot_obs"].astype(np.float32)[..., :data_cfg["action_dim"]]
        pos = joint_transform(pos, data_cfg["pos_max"], data_cfg["pos_min"])
    out["robot_pos"] = jnp.asarray(pos).reshape(1, *pos.shape)
    target_pos = np.asarray(x["detected_target_pos"], dtype=np.float32)
    if target_pos.shape[-1] < data_cfg["action_dim"]:
        # Pad with zeros to action_dim so joint_transform doesn't IndexError.
        pad = np.zeros(data_cfg["action_dim"] - target_pos.shape[-1], dtype=np.float32)
        target_pos = np.concatenate([target_pos, pad])
    target_pos = target_pos[..., :data_cfg["action_dim"]]
    target_pos = joint_transform(target_pos, data_cfg["pos_max"], data_cfg["pos_min"])
    out["target_pos"] = jnp.asarray(target_pos).reshape(1, *target_pos.shape)
    return out


def _cat_dict_jax(x: List[Dict[str, jnp.ndarray]], dim=0) -> Dict[str, jnp.ndarray]:
    keys = x[0].keys()
    return {k: jnp.concatenate([d[k] for d in x], axis=dim) for k in keys}


class _ObsQueue:
    def __init__(self, maxlen=1):
        self.queue = deque(maxlen=maxlen)
    def append(self, x):
        self.queue.append(x)
    def get(self):
        return _cat_dict_jax(list(self.queue), dim=0)


# ─────────────────────────────────────────── core: collect_episode ──

def reset_policy_state(policy):
    if hasattr(policy, "reset_ensemble"):
        policy.reset_ensemble()


def _select_nearest_efe_target(env, target_search, data_cfg, t, n, use_orientation=True):
    """Pick the affordance point nearest to the current TCP, combining the
    env's most-recent gripper-cam selection (``env.curr_detected_obj``) with
    a freshly-recomputed static-cam scan via ``target_search.compute(...)``.

    Returns ``(target_pos_jax, source)`` where ``target_pos_jax`` is a
    ``jnp.ndarray`` of shape ``(1, 1, action_dim)`` already normalized via
    ``joint_transform`` (matching ``jax_obs["target_pos"][None]``'s format)
    so it can be passed straight as ``target_pos=`` into ``world.calc_efe``.
    When neither view yields a candidate, returns ``(None, "none")`` and the
    caller should set ``disable_extrinsic=True`` (epistemic-only step).

    ``source`` is one of ``{"gripper", "static", "none"}``.

    Note: xarm's env has no ``gripper_target_lost`` flag like the sim wrapper
    has; we always include ``env.curr_detected_obj`` when set. Any per-step
    gripper-cam refinement failure leaves ``curr_detected_obj`` at its
    previous (possibly stale) value, but the nearest-TCP selection naturally
    deprioritizes stale entries once the arm has moved away.
    """
    # Robot state only — ``env.get_obs()`` here would grab a fresh frame from
    # both cameras purely to read three floats, and TargetSearch below pulls
    # its own static frame anyway.
    tcp = np.asarray(env.get_tcp_xyz(), dtype=np.float64)
    action_dim = int(data_cfg["action_dim"])

    candidates = []  # (source, world_xyz, world_orn|None)
    if getattr(env, "curr_detected_obj", None) is not None:
        candidates.append(
            ("gripper", np.asarray(env.curr_detected_obj, dtype=np.float64),
             getattr(env, "curr_detected_orn", None))
        )

    try:
        # Static-cam scan: fresh frame (counted under cam_static) + the
        # affordance forward (jitted, see AffordanceModel._JIT_UNET) + Hough
        # voting for the centers, which is plain NumPy in
        # ``vapo/affordance/hough_voting`` and does NOT benefit from the jit.
        with _PROF.stage("efe_t_search"):
            _, no_static, all_static = target_search.compute(
                env, return_all_centers=True, rand_sample=False,
            )
        if not no_static and all_static:
            static_orns = getattr(target_search, "last_orientations", None)
            for i, c in enumerate(all_static):
                orn = (static_orns[i] if static_orns is not None
                       and i < len(static_orns) else None)
                candidates.append(
                    ("static", np.asarray(c["target_pos"], dtype=np.float64), orn)
                )
    except Exception as exc:
        logger.debug(f"[ep {n} t={t}] static-cam affordance compute failed: {exc}")

    if not candidates:
        return None, "none"

    dists = np.array([np.linalg.norm(tcp - pos) for _, pos, _ in candidates])
    idx = int(np.argmin(dists))
    src, best, best_orn = candidates[idx]

    # 予測 orientation があれば rpy(3:6) に入れた full ベクトルを joint_transform し、
    # EFE の extrinsic が orientation も引き込むようにする（recon_pos と同じ正規化空間）。
    # orientation 無しなら従来通り xyz のみ正規化（EFE側で0埋め）。
    if use_orientation and best_orn is not None and action_dim >= 6:
        full = np.zeros(action_dim, dtype=np.float32)
        full[:3] = best.astype(np.float32)[:3]
        full[3:6] = np.asarray(best_orn, dtype=np.float32).flatten()[:3]
        norm = joint_transform(full, data_cfg["pos_max"], data_cfg["pos_min"])
    else:
        norm = joint_transform(
            best.astype(np.float32)[..., :action_dim],
            data_cfg["pos_max"], data_cfg["pos_min"],
        )
    return jnp.asarray(norm).reshape(1, 1, -1), src


def _clip_to_max_step_delta(
    action_dict: dict,
    current_ee_pos_quat: np.ndarray,
    max_xyz_m: float,
    max_rot_rad: float,
) -> Tuple[dict, bool]:
    """Cap the per-step EE pose delta against the measured current EE.

    When the policy commands a target far from the current EE, the xArm
    SDK's IK can pick a different joint-space branch and the wrist
    snaps to a visibly different configuration even though the
    requested Cartesian target is only slightly different. Limiting
    the per-step delta closes that window.

    Operates on ``cartesian_abs_aa`` payloads (axis-angle orientation,
    matching the AA send transport). Compares ``out[:3]`` against
    ``current[:3]`` for position and the SO(3) rotation
    ``R_target * R_current.inv()`` for orientation — i.e. the actual
    shortest-arc rotation, not the AA-vector difference which would
    be wrong near the ±π axis-angle wrap.

    ``max_*_*`` ≤ 0 disables that axis's clipping. Returns the (possibly
    rewritten) action dict and a flag indicating whether any clip fired.
    """
    atype = action_dict.get("type", "")
    if atype != "cartesian_abs_aa":
        # Only the AA absolute path has a meaningful "current vs target"
        # comparison here; relative deltas already have a built-in cap.
        return action_dict, False
    avec = np.asarray(action_dict["action"], dtype=np.float64).copy()
    cur = np.asarray(current_ee_pos_quat, dtype=np.float64).reshape(-1)
    cur_xyz = cur[:3]
    cur_quat = cur[3:7]

    was_clipped = False

    if max_xyz_m > 0.0:
        dxyz = avec[:3] - cur_xyz
        dnorm = float(np.linalg.norm(dxyz))
        if dnorm > max_xyz_m:
            avec[:3] = cur_xyz + dxyz * (max_xyz_m / dnorm)
            was_clipped = True

    if max_rot_rad > 0.0:
        R_target = _R.from_rotvec(avec[3:6])
        R_current = _R.from_quat(cur_quat)
        R_delta = R_target * R_current.inv()
        delta_aa = R_delta.as_rotvec()
        angle = float(np.linalg.norm(delta_aa))
        if angle > max_rot_rad:
            delta_aa = delta_aa * (max_rot_rad / angle)
            R_delta_clipped = _R.from_rotvec(delta_aa)
            R_target_clipped = R_delta_clipped * R_current
            avec[3:6] = R_target_clipped.as_rotvec()
            was_clipped = True

    out = dict(action_dict)
    out["action"] = avec.astype(np.float32)
    return out, was_clipped


def _preflight_ik_check(
    action_dict: dict,
    follower,
    current_joints: np.ndarray,
    max_joint_delta_rad: float,
) -> Tuple[dict, str]:
    """Query the controller's IK for ``action_dict`` and verify the
    predicted joint solution doesn't branch-flip from the current
    joints.

    Why this is needed
    ------------------
    The xArm SDK's ``get_inverse_kinematics`` has no seed parameter
    (verified on SDK 1.17.3), so the controller returns one fixed
    branch from the 7-DoF null space regardless of the arm's current
    state. Near singularities or simply where the elbow needs to flip
    side, two adjacent EE poses can map to wildly different joint
    configurations — the arm visibly snaps even though the requested
    Cartesian target moved by less than ``--max_step_xyz_m``.

    ``_clip_to_max_step_delta`` mitigates this in EE space, but a
    small EE delta is *not* the same thing as a small joint delta. We
    close the loop by asking the controller what joints it would pick
    for the target *before* sending the command:

      1. Get the SDK's IK solution for the target pose.
      2. Compare against the current joints.
      3. If any joint would move by more than ``max_joint_delta_rad``,
         **refuse the command and hold the current pose** (sets the
         action back to ``current_ee``). The policy may issue a
         different (smaller-delta) target next step.

    The hold strategy is preferable to dropping the step entirely
    because robopy's control thread re-sends the cached target every
    cycle — silently dropping a command would let an old (potentially
    far-away) target keep being sent, making things worse.

    Returns
    -------
    ``(possibly_modified_action_dict, status)`` where ``status`` is
    one of ``"ok"`` / ``"ik_failed"`` / ``"branch_flip"``.
    """
    atype = action_dict.get("type", "")
    if atype != "cartesian_abs_aa":
        return action_dict, "ok"
    if max_joint_delta_rad <= 0.0:
        return action_dict, "ok"
    sdk = getattr(follower, "_robot", None)
    if sdk is None:
        return action_dict, "ok"

    avec = np.asarray(action_dict["action"], dtype=np.float64).copy()
    # SDK get_inverse_kinematics consumes Euler [x_mm, y_mm, z_mm,
    # roll, pitch, yaw]. Convert the AA rotation to Euler XYZ first.
    # (The IK answer doesn't depend on representation, only on the
    # SO(3) pose, so any consistent encoding works — Euler is what
    # ``set_position`` uses internally and matches the SDK API best.)
    R_target = _R.from_rotvec(avec[3:6])
    rpy = R_target.as_euler("xyz")
    pose_for_ik = [
        float(avec[0]) * 1000.0, float(avec[1]) * 1000.0, float(avec[2]) * 1000.0,
        float(rpy[0]), float(rpy[1]), float(rpy[2]),
    ]
    try:
        code, target_joints = sdk.get_inverse_kinematics(
            pose_for_ik, input_is_radian=True, return_is_radian=True,
        )
    except Exception as exc:
        logger.debug(f"[preflight-ik] SDK threw: {exc}")
        return action_dict, "ik_failed"
    if code != 0 or target_joints is None:
        # IK couldn't solve (unreachable / singular). Holding is the
        # safe default — never send an unreachable target.
        return _hold_action_dict(action_dict, follower), "ik_failed"

    target_joints = np.asarray(target_joints[:7], dtype=np.float64)
    current_joints_arr = np.asarray(current_joints[:7], dtype=np.float64)
    if current_joints_arr.shape[0] < 7 or target_joints.shape[0] < 7:
        return action_dict, "ok"
    joint_delta = np.abs(target_joints - current_joints_arr)
    max_delta = float(np.max(joint_delta))
    if max_delta > max_joint_delta_rad:
        # Branch flip predicted. Hold current EE pose instead of sending
        # the offending target.
        return _hold_action_dict(action_dict, follower), "branch_flip"
    return action_dict, "ok"


def _hold_action_dict(action_dict: dict, follower) -> dict:
    """Rewrite ``action_dict`` so its position+rotation channels
    target the *current* EE pose. The gripper channel is preserved
    so an in-flight grasp action isn't dropped along with the motion.
    """
    out = dict(action_dict)
    avec = np.asarray(action_dict["action"], dtype=np.float64).copy()
    try:
        cur_ee = np.asarray(follower.get_ee_pos_quat(), dtype=np.float64)
        cur_xyz = cur_ee[:3]
        cur_quat = cur_ee[3:7]
        cur_aa = _R.from_quat(cur_quat).as_rotvec()
        avec[:3] = cur_xyz
        avec[3:6] = cur_aa
    except Exception:
        # If we can't read the current pose, fall through with the
        # original avec; ``_clip_to_max_step_delta``'s prior pass has
        # already bounded the EE delta so this is still safe.
        pass
    out["action"] = avec.astype(np.float32)
    return out


# ──────────────────────────────── local-IK pipeline (seeded DLS) ──
#
# Why we need a local IK
# ----------------------
# The xArm SDK's ``get_inverse_kinematics`` does not accept a seed
# (verified on SDK 1.17.3 — see ``get_inverse_kinematics`` signature),
# so the controller's IK picks a fixed branch from the redundant
# 7-DoF null space regardless of the arm's current state. Near
# singularities the chosen branch can be far from where the arm
# physically is, producing visible elbow snaps even for small EE
# pose deltas. The pre-flight check (``_preflight_ik_check``) only
# mitigates this by holding when a flip is predicted.
#
# This block sidesteps the SDK IK entirely:
#
#   * ``_xarm7_dh_fk(q)``        — numpy MDH forward kinematics. Fast,
#     no TCP RPC.
#   * ``_validate_dh_against_sdk(follower)`` — runtime sanity check
#     that confirms the DH parameters match the controller's
#     calibration at several random configurations. Aborts the local
#     IK path (with a clear error) if the agreement is worse than
#     5 mm / 2°, so a mis-spec'd DH never produces undetected motion.
#   * ``_xarm7_dls_ik(target, q_seed)`` — damped least-squares IK
#     seeded with ``q_seed`` (= current measured joints). Iterates
#     until pose error is below tolerance.
#
# The DH parameters below are the standard xArm7 modified-DH values
# (a in mm, alpha in rad). They come from UFactory's xarm7 user
# manual and the xarm_ros2 description package, with link offsets
# verified against a sample of saved ``robot_obs`` poses + the
# matching ``actions`` (= leader FK) when both are available.
#
# (alpha_{i-1}, a_{i-1}, d_i)  per joint i = 1..7
_XARM7_DH_MDH: tuple = (
    # (alpha_{i-1} [rad], a_{i-1} [m], d_i [m])
    (0.0,          0.0,         0.267),     # joint 1
    (-np.pi / 2,   0.0,         0.0),       # joint 2
    ( np.pi / 2,   0.0,         0.293),     # joint 3
    ( np.pi / 2,   0.0525,      0.0),       # joint 4
    ( np.pi / 2,   0.0775,      0.3425),    # joint 5
    ( np.pi / 2,   0.0,         0.0),       # joint 6
    (-np.pi / 2,   0.076,       0.097),     # joint 7
)

# xArm7 joint limits (rad, approximate; from xArm SDK service limits).
_XARM7_JOINT_LIMITS = np.array([
    [-2 * np.pi,   2 * np.pi],   # j1
    [-2.059,       2.094],       # j2
    [-2 * np.pi,   2 * np.pi],   # j3
    [-0.0698,      3.927],       # j4
    [-2 * np.pi,   2 * np.pi],   # j5
    [-1.6929,      np.pi],       # j6
    [-2 * np.pi,   2 * np.pi],   # j7
], dtype=np.float64)


def _xarm7_dh_fk(q: np.ndarray) -> np.ndarray:
    """Modified-DH forward kinematics for xArm7.

    Args:
        q: ``(7,)`` joint angles in radians.

    Returns:
        ``(4, 4)`` homogeneous transform of the flange in the base
        frame. Translation is in meters.
    """
    q = np.asarray(q, dtype=np.float64).flatten()
    T = np.eye(4, dtype=np.float64)
    for i, (alpha, a, d) in enumerate(_XARM7_DH_MDH):
        ca, sa = np.cos(alpha), np.sin(alpha)
        ct, st = np.cos(q[i]), np.sin(q[i])
        # MDH: T_i = Rot_x(alpha_{i-1}) Trans_x(a_{i-1}) Rot_z(q_i) Trans_z(d_i)
        Ti = np.array([
            [ct,       -st,      0,    a],
            [st * ca,  ct * ca, -sa, -d * sa],
            [st * sa,  ct * sa,  ca,  d * ca],
            [0,        0,        0,    1],
        ], dtype=np.float64)
        T = T @ Ti
    return T


def _validate_dh_against_sdk(
    follower, n_samples: int = 5, rng_seed: int = 0,
) -> tuple[bool, float, float]:
    """Confirm ``_xarm7_dh_fk`` agrees with the xArm SDK FK.

    Probes ``n_samples`` random valid joint configurations + the
    current measured pose. Compares the DH-FK output against
    ``follower.forward_kinematics`` and returns the maximum
    (translation_err_mm, rotation_err_deg) seen.

    Returns ``(ok, max_pos_err_mm, max_rot_err_deg)``. ``ok`` is True
    when both errors are within a conservative 5 mm / 2° envelope,
    which is comfortably below the typical controller calibration
    tolerance.
    """
    rng = np.random.RandomState(rng_seed)
    samples = []
    # Always include the current joints (most operationally relevant point).
    try:
        cur_joints = np.asarray(
            follower.get_joint_state()[:7], dtype=np.float64,
        )
        samples.append(cur_joints)
    except Exception:
        pass
    # Random valid configs sampled inside the joint limits.
    for _ in range(n_samples):
        q = np.array([
            rng.uniform(lo, hi) for lo, hi in _XARM7_JOINT_LIMITS
        ], dtype=np.float64)
        samples.append(q)
    max_pos = 0.0
    max_rot = 0.0
    for q in samples:
        try:
            sdk_pose = np.asarray(
                follower.forward_kinematics(q), dtype=np.float64,
            )  # (6,) [xyz_mm, rpy_rad]
        except Exception as exc:
            logger.warning(f"[local-ik] SDK FK probe failed: {exc}")
            return False, float("inf"), float("inf")
        sdk_xyz_m = sdk_pose[:3] / 1000.0
        sdk_R = _R.from_euler("xyz", sdk_pose[3:6])
        T_dh = _xarm7_dh_fk(q)
        dh_xyz_m = T_dh[:3, 3]
        dh_R = _R.from_matrix(T_dh[:3, :3])
        pos_err_mm = float(np.linalg.norm(dh_xyz_m - sdk_xyz_m)) * 1000.0
        rot_err = (sdk_R * dh_R.inv()).magnitude()
        rot_err_deg = float(rot_err) * 180.0 / np.pi
        max_pos = max(max_pos, pos_err_mm)
        max_rot = max(max_rot, rot_err_deg)
    ok = (max_pos <= 5.0) and (max_rot <= 2.0)
    return ok, max_pos, max_rot


def _xarm7_dls_ik(
    target_xyz_m: np.ndarray,
    target_rotation: "_R",
    q_seed: np.ndarray,
    *,
    max_iters: int = 30,
    damping: float = 0.05,
    pos_tol_m: float = 1e-4,
    rot_tol_rad: float = 1e-3,
    step_size: float = 1.0,
    fd_eps: float = 1e-5,
) -> tuple[np.ndarray, bool, float, float]:
    """Damped least-squares IK seeded with ``q_seed``.

    The Jacobian is computed via finite differences on
    :func:`_xarm7_dh_fk`. Joint angles are clipped to the limit table
    on every update, so the solver naturally stays in the operating
    branch.

    Returns ``(q, converged, final_pos_err_m, final_rot_err_rad)``.
    Even when not converged, ``q`` is the best iterate seen — caller
    can decide to send it (small residual) or hold (large residual).
    """
    q = np.asarray(q_seed, dtype=np.float64).copy()
    target_xyz_m = np.asarray(target_xyz_m, dtype=np.float64)

    def pose_error(q_arr):
        T = _xarm7_dh_fk(q_arr)
        pos_err = target_xyz_m - T[:3, 3]
        rot_err = (target_rotation * _R.from_matrix(T[:3, :3]).inv()).as_rotvec()
        return pos_err, rot_err

    best_q = q.copy()
    best_err_norm = float("inf")
    final_pos = np.zeros(3)
    final_rot = np.zeros(3)
    for _ in range(max_iters):
        pos_err, rot_err = pose_error(q)
        err = np.concatenate([pos_err, rot_err])
        err_norm = float(np.linalg.norm(err))
        if err_norm < best_err_norm:
            best_err_norm = err_norm
            best_q = q.copy()
            final_pos = pos_err
            final_rot = rot_err
        if (np.linalg.norm(pos_err) < pos_tol_m
                and np.linalg.norm(rot_err) < rot_tol_rad):
            return q, True, float(np.linalg.norm(pos_err)), float(np.linalg.norm(rot_err))
        # 6×7 Jacobian via central differences. ~14 FK evaluations per
        # iter, all numpy — no TCP RPC. Total ~15µs per iter on a
        # typical laptop.
        J = np.zeros((6, 7), dtype=np.float64)
        for i in range(7):
            qp = q.copy(); qp[i] += fd_eps
            qm = q.copy(); qm[i] -= fd_eps
            Tp = _xarm7_dh_fk(qp)
            Tm = _xarm7_dh_fk(qm)
            J[:3, i] = (Tp[:3, 3] - Tm[:3, 3]) / (2 * fd_eps)
            # Rotation column = axis-angle of R_p · R_m^{-1} / (2*eps),
            # which is the angular velocity for unit joint speed.
            Rp = _R.from_matrix(Tp[:3, :3])
            Rm = _R.from_matrix(Tm[:3, :3])
            J[3:, i] = (Rp * Rm.inv()).as_rotvec() / (2 * fd_eps)
        # DLS: dq = J^T (J J^T + λ²I)^{-1} e
        JJt = J @ J.T
        damp_mat = (damping ** 2) * np.eye(6, dtype=np.float64)
        try:
            sol = np.linalg.solve(JJt + damp_mat, err)
        except np.linalg.LinAlgError:
            break
        dq = J.T @ sol
        q = q + step_size * dq
        q = np.clip(q, _XARM7_JOINT_LIMITS[:, 0], _XARM7_JOINT_LIMITS[:, 1])
    return best_q, False, float(np.linalg.norm(final_pos)), float(np.linalg.norm(final_rot))


# ────────────────── joint_actions ↔ EE-pose bridge (for world model) ──
#
# When ``policy_cfg.datamodule.action_type == "joint_actions"`` the policy
# emits 8-DoF joint targets ``[j1..j7, grip_pm1]``. The world model may
# have been trained on a different ``action_type`` (``actions``,
# ``rel_actions``, ``mixed``). To let the two coexist without retraining,
# we bridge per-step at runtime: joints → FK → EE pose → normalize using
# the world's action stats → feed to ``world.step_dynamics`` / the
# EFE-imagination JIT.
#
# Tradeoff: this uses the local DH FK (``_xarm7_dh_fk``) which can drift
# from the controller's calibration by a few mm — fine for world-model
# imagination (consistency only; the policy still issues joint commands
# directly to the arm). The hot path is small (1 row per policy step) so
# the numpy implementation is fast enough.


def _joint_action_to_ee_pose_chunk(
    chunk: np.ndarray,
) -> np.ndarray:
    """Apply DH FK to every row of a ``(T, 8)`` joint_actions chunk.

    Returns ``(T, 7)`` ``[xyz_m, roll, pitch, yaw, grip_pm1]`` (xArm
    SDK Euler XYZ convention to match ``actions.blosc2``).
    """
    chunk = np.asarray(chunk, dtype=np.float64)
    T = chunk.shape[0]
    out = np.zeros((T, 7), dtype=np.float32)
    for t in range(T):
        H = _xarm7_dh_fk(chunk[t, :7])
        rpy = _R.from_matrix(H[:3, :3]).as_euler("xyz")
        out[t, :3] = H[:3, 3].astype(np.float32)
        out[t, 3:6] = rpy.astype(np.float32)
        out[t, 6] = np.float32(chunk[t, 7])
    return out


def _joint_chunk_to_world_action(
    joint_chunk_phys: np.ndarray,
    ee_anchor_xyz_quat: np.ndarray,
    world_action_type: str,
    data_cfg,
    mixed_sources: list | None = None,
) -> np.ndarray:
    """Convert a physical ``(T, 8)`` joint chunk into the world model's
    normalized action representation (``actions``, ``rel_actions``, or
    ``mixed``).

    Steps:
      1. FK every row → ``(T, 7)`` EE pose in xArm SDK Euler XYZ.
      2. Build the world's expected ``action_type`` stream:
         * ``actions``     → use the FK output directly.
         * ``rel_actions`` → derive frame-to-frame deltas anchored at
           ``ee_anchor`` (xyz delta + axis-angle rotation delta);
           ``rel[0]`` measured from the anchor (matches the recorder's
           first-frame convention against ``prev_ee=None`` ⇒ all-zero
           twist would also work, but anchoring gives world models that
           saw nonzero rel[0] the better fit).
         * ``mixed``       → per-channel: xyz from rel branch, rot+grip
           from abs branch (or whatever ``mixed_sources`` says).
      3. Normalize using ``data_cfg[f"{action_type}_min/max"]`` so the
         output shape matches what the world model was trained on.
    """
    ee_pose_chunk = _joint_action_to_ee_pose_chunk(joint_chunk_phys)
    if world_action_type == "actions":
        action_phys = ee_pose_chunk
        norm = joint_transform(
            action_phys,
            data_cfg["actions_max"], data_cfg["actions_min"],
        )
        return np.asarray(norm, dtype=np.float32)
    if world_action_type == "rel_actions":
        T = ee_pose_chunk.shape[0]
        rel = np.zeros((T, 7), dtype=np.float32)
        anchor_xyz = np.asarray(ee_anchor_xyz_quat[:3], dtype=np.float64)
        anchor_R = _R.from_quat(np.asarray(ee_anchor_xyz_quat[3:7], dtype=np.float64))
        prev_xyz = anchor_xyz
        prev_R = anchor_R
        for t in range(T):
            cur_xyz = ee_pose_chunk[t, :3].astype(np.float64)
            cur_R = _R.from_euler("xyz", ee_pose_chunk[t, 3:6].astype(np.float64))
            rel[t, :3] = (cur_xyz - prev_xyz).astype(np.float32)
            rel[t, 3:6] = (cur_R * prev_R.inv()).as_rotvec().astype(np.float32)
            rel[t, 6] = ee_pose_chunk[t, 6]
            prev_xyz = cur_xyz
            prev_R = cur_R
        norm = joint_transform(
            rel,
            data_cfg["rel_actions_max"], data_cfg["rel_actions_min"],
        )
        return np.asarray(norm, dtype=np.float32)
    if world_action_type == "mixed":
        # Reuse the rel-anchor path for the xyz delta and the abs path
        # for rotation + grip; combine per ``mixed_sources``.
        rel_norm = _joint_chunk_to_world_action(
            joint_chunk_phys, ee_anchor_xyz_quat, "rel_actions",
            data_cfg, mixed_sources,
        )
        abs_norm = _joint_chunk_to_world_action(
            joint_chunk_phys, ee_anchor_xyz_quat, "actions",
            data_cfg, mixed_sources,
        )
        src = mixed_sources or (
            "rel", "rel", "rel", "abs", "abs", "abs", "abs"
        )
        out = np.zeros_like(rel_norm)
        for ch in range(7):
            out[:, ch] = rel_norm[:, ch] if src[ch] == "rel" else abs_norm[:, ch]
        return out
    if world_action_type == "joint_actions":
        # World model trained on joints too — no bridge needed.
        norm = joint_transform(
            joint_chunk_phys,
            data_cfg["joint_actions_max"], data_cfg["joint_actions_min"],
        )
        return np.asarray(norm, dtype=np.float32)
    raise ValueError(
        f"unsupported world_action_type {world_action_type!r} for the "
        "joint_actions bridge"
    )


def _local_ik_send(
    action_dict: dict,
    follower,
    current_joints: np.ndarray,
    *,
    max_joint_delta_rad: float,
    pos_tol_m: float = 1e-3,   # 1 mm
    rot_tol_rad: float = 0.01, # ~0.57°
) -> tuple[dict, str]:
    """Replace a ``cartesian_abs_aa`` action with a joint-target
    action computed via the seeded local DLS IK.

    Returns the (possibly hold-current) action dict and a status:
      * ``"joint"``        — joint command issued (success).
      * ``"hold"``         — IK did not converge to spec or the joint
                              delta exceeded ``max_joint_delta_rad``;
                              caller should hold current EE pose.
      * ``"not_aa"``       — payload isn't ``cartesian_abs_aa``; pass
                              through unchanged.
    """
    atype = action_dict.get("type", "")
    if atype != "cartesian_abs_aa":
        return action_dict, "not_aa"
    avec = np.asarray(action_dict["action"], dtype=np.float64).copy()
    target_xyz_m = avec[:3]
    target_rot = _R.from_rotvec(avec[3:6])
    q_seed = np.asarray(current_joints, dtype=np.float64)[:7]

    q_target, converged, pos_err, rot_err = _xarm7_dls_ik(
        target_xyz_m, target_rot, q_seed,
    )
    if not converged and (pos_err > pos_tol_m or rot_err > rot_tol_rad):
        return _hold_action_dict(action_dict, follower), "hold"
    joint_delta = float(np.max(np.abs(q_target - q_seed)))
    if max_joint_delta_rad > 0 and joint_delta > max_joint_delta_rad:
        return _hold_action_dict(action_dict, follower), "hold"
    # Pack as a 7-vec joint command for the env. Gripper preserved.
    grip_pm1 = float(avec[6])
    return {
        "type": "joint_abs",
        "action": np.concatenate(
            [q_target.astype(np.float32), [np.float32(grip_pm1)]]
        ),
    }, "joint"


def _official_ik_send(
    action_dict: dict,
    follower,
    current_joints: np.ndarray,
    official_kin,
    *,
    max_joint_delta_rad: float,
) -> tuple[dict, str]:
    """Replace a ``cartesian_abs_aa`` action with a joint target
    computed via UFactory's **official** xArm7 kinematics library,
    seeded with the current measured joints.

    The official library is what ``MinnaZhong`` (xArm-Python-SDK
    contributor) pointed at on GitHub: its
    ``xarm7_inverse_kinematics`` accepts a ``q_pre`` seed, so the
    iterative solver stays in the null-space branch closest to
    ``current_joints``. Empirically (verified on a 30-frame smooth
    trajectory) this caps per-step joint deltas at ~3° on the same
    motion where the SDK / zero-seeded IK produced ~350° flips.

    Status values mirror ``_local_ik_send`` for downstream parity:
      * ``"joint"``  — joint command issued.
      * ``"hold"``   — IK returned non-zero (unreachable) or the
                        joint delta exceeded ``max_joint_delta_rad``.
      * ``"not_aa"`` — payload isn't ``cartesian_abs_aa``.
    """
    atype = action_dict.get("type", "")
    if atype != "cartesian_abs_aa":
        return action_dict, "not_aa"
    avec = np.asarray(action_dict["action"], dtype=np.float64).copy()
    # Library expects mm + Euler XYZ; we have m + axis-angle.
    target_xyz_mm = avec[:3] * 1000.0
    target_rpy = _R.from_rotvec(avec[3:6]).as_euler("xyz")
    pose_rpy = np.concatenate([target_xyz_mm, target_rpy])
    q_seed = np.asarray(current_joints, dtype=np.float64)[:7]
    q_target, ret = official_kin.ik(pose_rpy, q_pre=q_seed)
    if ret != 0:
        return _hold_action_dict(action_dict, follower), "hold"
    joint_delta = float(np.max(np.abs(q_target - q_seed)))
    if max_joint_delta_rad > 0 and joint_delta > max_joint_delta_rad:
        return _hold_action_dict(action_dict, follower), "hold"
    grip_pm1 = float(avec[6])
    return {
        "type": "joint_abs",
        "action": np.concatenate(
            [q_target.astype(np.float32), [np.float32(grip_pm1)]]
        ),
    }, "joint"


def _clip_action_to_xyz_bounds(action_dict, current_tcp, xyz_min, xyz_max):
    """Clip the xyz target the next env.step would drive the TCP to, keeping
    it inside ``[xyz_min, xyz_max]`` (the first 3 dims of
    ``data_cfg["pos_min/max"]``).

    Handles the transports the xarm env accepts (see ``XArmRealEnv.step``):
      * ``"cartesian_abs"`` / ``"cartesian_abs_aa"`` — ``action[:3]`` is
        the absolute TCP target (m); clip xyz directly. The rotation
        channels (Euler RPY vs axis-angle) differ between the two but
        the xyz clip is identical.
      * ``"cartesian_rel"`` — ``action[:3]`` is a delta in meters (no
        ``max_rel_pos`` scaling here, unlike CALVIN sim); predict
        ``proposed = current_tcp + delta``, clip, recompute delta so the
        actual landed target is exactly the clipped bound.

    Returns a new dict (does not mutate ``action_dict``).
    """
    xyz_min = np.asarray(xyz_min, dtype=np.float64)
    xyz_max = np.asarray(xyz_max, dtype=np.float64)
    atype = action_dict.get("type", "cartesian_rel")
    avec = np.asarray(action_dict["action"], dtype=np.float64).copy()
    if atype in ("cartesian_abs", "cartesian_abs_aa"):
        avec[:3] = np.clip(avec[:3], xyz_min, xyz_max)
    else:  # cartesian_rel
        tcp = np.asarray(current_tcp, dtype=np.float64)
        proposed = tcp + avec[:3]
        clipped = np.clip(proposed, xyz_min, xyz_max)
        avec[:3] = clipped - tcp
    out = dict(action_dict)
    out["action"] = avec.astype(np.float32)
    return out


def _compute_step_obs(env: XArmRealEnv, gripper_net, static_net,
                     aff_transforms, data_cfg, run_cfg) -> Tuple[dict, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Read frames, run both affordance nets, derive ``detected_target_pos``,
    return (raw_obs, gripper_aff_mask_hw, static_aff_mask_hw, target_pos_xyz,
    gripper_aff_raw).

    ``gripper_aff_raw`` is the ``{"mask", "probs", "dirs"}`` bundle straight
    off the gripper net, handed to :func:`gripper_affordance_dir_frame` so the
    per-step direction-overlay MP4 costs no extra forward pass.
    """
    raw_obs = env.get_obs()
    # Gripper affordance forward
    with _PROF.stage("aff_gripper"):
        gripper_mask, gripper_probs, gripper_dirs, gripper_orient = _jax_sync(
            predict_gripper_affordance(
                gripper_net, raw_obs["rgb_gripper"], aff_transforms["gripper"],
            )
        )
    # Static affordance forward (only for the saved mask channel; the
    # actual target search uses TargetSearch's separate net).
    static_mask_hw = np.zeros(
        (1, raw_obs["rgb_static"].shape[0], raw_obs["rgb_static"].shape[1]),
        dtype=np.uint8,
    )
    if static_net is not None:
        with _PROF.stage("aff_static"):
            try:
                # Match the training-time crop before the net sees the frame.
                static_rgb = crop_static_rgb_for_net(static_net, raw_obs["rgb_static"])
                centers, sm, sd, sp, _ = transform_and_predict(
                    static_net, aff_transforms["static"], static_rgb,
                )
                static_mask_hw = sm[np.newaxis, ...].astype(np.uint8)
            except Exception as exc:
                logger.debug(f"static aff forward failed: {exc}")
    # Update curr_detected_obj via gripper view + cached static_initial_target.
    target_pos = env.curr_detected_obj
    if gripper_net is not None and env.static_initial_target is not None:
        with _PROF.stage("aff_refine"):
            try:
                target_pos = _refine_target_via_gripper(
                    env, gripper_mask, gripper_probs, gripper_dirs,
                    raw_obs["depth_gripper"], run_cfg.affordance.gripper_cam,
                    gripper_net=gripper_net, orient_6d=gripper_orient,
                )
            except Exception as exc:
                logger.debug(f"gripper target refine failed: {exc}")
    if target_pos is None:
        target_pos = np.asarray(raw_obs["robot_obs"][:3], dtype=np.float32)
    env.curr_detected_obj = np.asarray(target_pos, dtype=np.float32)
    gripper_aff_raw = {
        "mask": gripper_mask, "probs": gripper_probs, "dirs": gripper_dirs,
    }
    return (raw_obs, np.asarray(gripper_mask), static_mask_hw,
            env.curr_detected_obj.copy(), gripper_aff_raw)


def _refine_target_via_gripper(env: XArmRealEnv, aff_mask, aff_probs, directions,
                                depth_gripper, aff_cfg,
                                gripper_net=None, orient_6d=None) -> np.ndarray | None:
    """Port of `AffordanceWrapperBase.find_target_center` (gripper branch).

    Picks the affordance cluster closest to the existing
    ``static_initial_target`` (within ``termination_radius/2``) and projects
    its pixel center through depth + ``T_world_cam`` to world coordinates.
    """
    from vapo.affordance.utils.img_utils import resize_center
    # Cluster centers via the gripper net's internal helper.
    gripper_net = env  # unused; we already have aff_mask/dirs
    static_target = env.static_initial_target
    if static_target is None:
        return env.curr_detected_obj

    # The net's get_centers returns object_centers in mask-resolution pixel coords.
    # We don't have direct access here — assume the same model produced these
    # (they share `get_centers`). For safety, fall back to highest-prob pixel
    # of the mask.
    aff_mask_2d = np.asarray(aff_mask).squeeze()
    if aff_mask_2d.ndim == 3:
        aff_mask_2d = aff_mask_2d[0]
    H_mask, W_mask = aff_mask_2d.shape
    if aff_mask_2d.sum() <= 0:
        return env.curr_detected_obj
    # Pick the centroid of the largest connected component.
    ys, xs = np.where(aff_mask_2d > 0)
    if len(ys) == 0:
        return env.curr_detected_obj
    # Robustness rejection (port of aff_wrapper_base.find_target_center): if the
    # mean foreground probability over the detected cluster is below
    # ``env.gripper_aff_min_robustness``, treat the detection as unreliable and
    # do NOT refine — keep the current target (0.0 disables rejection).
    min_robust = float(getattr(env, "gripper_aff_min_robustness", 0.0))
    if min_robust > 0.0 and aff_probs is not None:
        probs_arr = np.asarray(aff_probs)
        if probs_arr.ndim == 4:          # (1, H, W, C)
            probs_arr = probs_arr[0]
        if probs_arr.shape[:2] == aff_mask_2d.shape and probs_arr.ndim == 3:
            fg = min(1, probs_arr.shape[-1] - 1)   # foreground channel
            robustness = float(np.mean(probs_arr[ys, xs, fg]))
            if robustness < min_robust:
                logger.debug(
                    f"[gripper-aff] cluster rejected "
                    f"(robustness {robustness:.3f} < {min_robust})")
                return env.curr_detected_obj
    cy, cx = int(round(float(np.mean(ys)))), int(round(float(np.mean(xs))))
    depth_2d = np.asarray(depth_gripper)
    if depth_2d.ndim == 3:
        depth_2d = depth_2d.squeeze(0) if depth_2d.shape[0] == 1 else depth_2d[..., 0]
    H_dep, W_dep = depth_2d.shape
    # Resize the (cy, cx) from mask resolution to depth resolution.
    v_dep, u_dep = resize_center((cy, cx), (H_mask, W_mask), (H_dep, W_dep))
    cam_frame = env.camera_manager.gripper_cam.deproject((u_dep, v_dep), depth_2d)
    if cam_frame is None:
        return env.curr_detected_obj
    # tcp_mat @ T_tcp_cam @ [cam_frame, 1] → world point.
    tcp_pos, tcp_orn = env.robot.get_tcp_pos_orn()
    tcp_mat = np.eye(4, dtype=np.float32)
    tcp_mat[:3, :3] = _R.from_quat(tcp_orn).as_matrix().astype(np.float32)
    tcp_mat[:3, 3] = tcp_pos.astype(np.float32)
    T_tcp_cam = env.camera_manager.gripper_cam.get_extrinsic_calibration("panda")
    pt_h = np.array([cam_frame[0], cam_frame[1], cam_frame[2], 1.0], dtype=np.float32)
    world = (tcp_mat @ T_tcp_cam @ pt_h)[:3]
    # Accept only if within radius/2 of the static target — matches
    # AffordanceWrapperBase.find_target_center's locking criterion.
    if float(np.linalg.norm(world - static_target)) < env.termination_radius / 2.0:
        # 重心ピクセル(cy,cx, mask解像度=orient_6d解像度)で orientation を decode。
        if (orient_6d is not None and gripper_net is not None
                and hasattr(gripper_net, "decode_orient_euler_at_centers")):
            try:
                eul = gripper_net.decode_orient_euler_at_centers(orient_6d, [(cy, cx)])[0]
                env.curr_detected_orn = None if eul is None else np.asarray(eul, dtype=np.float32)
            except Exception as exc:
                logger.debug(f"gripper orientation decode failed: {exc}")
        return world
    return static_target


def _load_replay_data(replay_path: Path, action_type: str) -> dict:
    """Load saved actions + initial ``robot_obs`` for replay.

    When ``action_type == "rel_actions"`` the saved deltas use
    **axis-angle** rotation (see ``compare_actions_vs_rel.py`` header /
    ``_ee_rel_action``), but xArm SDK's ``command_cartesian_relative``
    interprets the rotation triplet as **Euler RPY** deltas — sending
    the raw stream therefore garbles the orientation. We integrate
    rel→abs here (same recipe as
    ``compare_actions_vs_rel.rel_actions_to_abs_actions`` /
    :func:`_integrate_rel_to_abs`) anchored at the first row of
    ``obs/robot_obs.blosc2``, and the caller drives the arm with
    ``cartesian_abs`` regardless of which source file was requested.

    Returned dict keys:
      * ``actions`` (T, 7) — physical-unit absolute EE poses
        ``[x_m, y_m, z_m, roll, pitch, yaw, grip_pm1]``. Always
        absolute (the rel branch has been integrated already).
      * ``action_type`` — ``"actions"``. The transport the loop must
        use to send each row.
      * ``source_action_type`` — what was on disk
        (``"actions"`` or ``"rel_actions"``).
      * ``init_robot_obs`` — first row of ``robot_obs.blosc2`` (15,).
      * ``source_path`` — episode directory.
    """
    replay_path = Path(replay_path)
    actions_file = replay_path / "actions" / f"{action_type}.blosc2"
    robot_obs_file = replay_path / "obs" / "robot_obs.blosc2"
    if not actions_file.is_file():
        raise FileNotFoundError(f"replay actions not found: {actions_file}")
    if not robot_obs_file.is_file():
        raise FileNotFoundError(f"replay robot_obs not found: {robot_obs_file}")
    actions_arr = np.asarray(load_blosc2(str(actions_file)))
    robot_obs_arr = np.asarray(load_blosc2(str(robot_obs_file)))
    if actions_arr.ndim != 2 or actions_arr.shape[1] < 7:
        raise ValueError(
            f"unexpected actions shape: {actions_arr.shape} "
            f"(expected (T, 7))"
        )
    if robot_obs_arr.ndim != 2 or robot_obs_arr.shape[1] < 7:
        raise ValueError(
            f"unexpected robot_obs shape: {robot_obs_arr.shape} "
            f"(expected (T, ≥7))"
        )
    logger.info(
        f"[replay] loaded {action_type}={actions_arr.shape} "
        f"robot_obs={robot_obs_arr.shape} from {replay_path}"
    )

    init_xyz = robot_obs_arr[0, :3].astype(np.float64)
    init_quat = robot_obs_arr[0, 3:7].astype(np.float64)
    if action_type == "rel_actions":
        abs_arr = _integrate_rel_to_abs(
            init_xyz, init_quat, actions_arr.astype(np.float64),
        )
        logger.info(
            "[replay] integrated rel_actions → absolute poses "
            "(anchored at robot_obs[0]; send transport=cartesian_abs). "
            "Raw rel_actions are NOT sent because rel rotation is "
            "axis-angle while command_cartesian_relative expects Euler "
            "RPY deltas — see compare_actions_vs_rel.py."
        )
        actions_out = abs_arr.astype(np.float32)
    else:
        actions_out = actions_arr.astype(np.float32)
    return {
        "actions": actions_out,
        "action_type": "actions",
        "source_action_type": action_type,
        "init_robot_obs": robot_obs_arr[0].astype(np.float32),
        "source_path": replay_path,
    }


def _move_env_to_init_pose(env: "XArmRealEnv", init_robot_obs: np.ndarray,
                            timeout_s: float = 8.0) -> None:
    """Drive the arm to the saved-episode's first ``robot_obs`` pose.

    Uses the AA send path (``cartesian_abs_aa``) for the orientation
    target so a saved pose with roll ≈ ±π — which is the steady-state
    in this cell — doesn't trip the Euler RPY discontinuity at the
    wrap point. The XYZ + axis-angle payload goes through xArm SDK's
    ``set_position_aa`` via the ``_patch_follower_aa_send`` hook.
    """
    init_xyz = np.asarray(init_robot_obs[:3], dtype=np.float32)
    init_quat = np.asarray(init_robot_obs[3:7], dtype=np.float32)
    init_aa = (
        _R.from_quat(init_quat.astype(np.float64))
          .as_rotvec().astype(np.float32)
    )
    logger.info(
        f"[replay] moving to init pose xyz_m={init_xyz.round(3).tolist()} "
        f"aa_rad={init_aa.round(3).tolist()}"
    )
    if env.dry_run:
        return

    # Workspace warning only (matches move_to_target's logic).
    ws = getattr(env._follower, "_workspace", None)
    if ws is not None:
        tx, ty, tz = (float(init_xyz[i]) * 1000.0 for i in range(3))
        min_z = ws.effective_min_z(tx, ty)
        oob = []
        if not (ws.min_x <= tx <= ws.max_x):
            oob.append(f"x={tx:.1f}∉[{ws.min_x},{ws.max_x}]")
        if not (ws.min_y <= ty <= ws.max_y):
            oob.append(f"y={ty:.1f}∉[{ws.min_y},{ws.max_y}]")
        if not (min_z <= tz <= ws.max_z):
            oob.append(f"z={tz:.1f}∉[{min_z},{ws.max_z}]")
        if oob:
            logger.warning(
                "[replay] init pose outside XArmWorkspaceBounds — robopy "
                "will silently clip. Offending axes: %s. Pass "
                "``--no_workspace_bounds`` to disable.",
                ", ".join(oob),
            )

    # Route via env.step's cartesian_abs_aa path so the AA marker is
    # attached to the target_command dict and the patched
    # _send_cartesian dispatches to set_position_aa.
    env.step({
        "type": "cartesian_abs_aa",
        "action": np.array([
            init_xyz[0], init_xyz[1], init_xyz[2],
            init_aa[0], init_aa[1], init_aa[2],
            1.0,  # gripper open during the init move
        ], dtype=np.float32),
    }, return_obs=False)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < timeout_s:
        if _ABORT_EVENT.is_set():
            break
        ee_cur = env._follower.get_ee_pos_quat()
        gap = float(np.linalg.norm(ee_cur[:3] - init_xyz))
        if gap < 0.01:
            break
        time.sleep(0.05)
    ee_final = env._follower.get_ee_pos_quat()
    final_gap_mm = float(np.linalg.norm(ee_final[:3] - init_xyz)) * 1000.0
    logger.info(f"[replay] reached init pose: gap={final_gap_mm:.1f}mm")


def collect_episode_replay(
    n: int,
    env: "XArmRealEnv",
    gripper_net, static_net, aff_transforms,
    data_cfg, run_cfg, args,
    replay_data: dict,
    save_video: bool = True,
) -> dict:
    """Replay an episode: play back saved actions through the same pipeline.

    Mirrors :func:`collect_episode` minus the policy / world / EFE /
    target_search machinery. The arm is reset to the home pose, then
    driven to the saved episode's first ``robot_obs`` xyz+quat before
    iterating the saved action array (``actions`` for absolute,
    ``rel_actions`` for delta). Every step still goes through
    ``env.step`` → ``_compute_step_obs`` so the output episode has the
    identical blosc2 layout as a normal collection run.

    The loop stops after ``min(--episode_length, len(saved_actions))``
    steps. ``candidate_policies`` / ``extrinsics`` / ``epistemics``
    arrays are not produced (there is no policy here); callers should
    save the episode with ``calc_efe=False``.
    """
    action_type = str(replay_data["action_type"])
    actions_arr = np.asarray(replay_data["actions"], dtype=np.float32)
    init_robot_obs = np.asarray(replay_data["init_robot_obs"], dtype=np.float32)

    # ── reset (home pose) then drive to saved init pose ──
    env.reset()
    _move_env_to_init_pose(env, init_robot_obs)

    gripper_aff_min_robustness = float(
        getattr(env, "gripper_aff_min_robustness", 0.0))

    # ── first obs ──
    (raw_obs, gripper_mask_hw, static_mask_hw, target_world,
     gripper_aff_raw) = _compute_step_obs(
        env, gripper_net, static_net, aff_transforms, data_cfg, run_cfg,
    )

    # ── accumulators ──
    static_rgbs, static_masks = [], []
    gripper_rgbs, gripper_depths, gripper_masks = [], [], []
    # Per-frame gripper-cam affordance-direction overlays → rgb_gripper_aff_dirs.mp4
    # (aif_collection_uv parity; collected only when --save_video is on).
    gripper_dir_overlays = []
    positions, actions_phys, actions_rel = [], [], []
    ee_prev = np.asarray(raw_obs["ee_pos_quat"], dtype=np.float32).copy()
    ee_initial_for_integration = ee_prev.copy()
    period = 1.0 / float(args.fps)

    # Cap replay length at user-requested episode_length.
    T_replay = min(int(args.episode_length), int(actions_arr.shape[0]))
    if T_replay < int(actions_arr.shape[0]):
        logger.info(
            f"[replay] truncating to --episode_length={args.episode_length} "
            f"(source has {actions_arr.shape[0]} steps)"
        )

    xyz_min = np.asarray(data_cfg["pos_min"][:3], dtype=np.float64)
    xyz_max = np.asarray(data_cfg["pos_max"][:3], dtype=np.float64)
    # _load_replay_data already integrates rel_actions → absolute poses
    # (anchored at robot_obs[0]) so action_type is always "actions" here.
    # Convert the RPY column to axis-angle once so the per-step send
    # routes through the gimbal-lock-free ``cartesian_abs_aa`` transport
    # (matches the policy path; see ``collect_episode`` chunk pre-compute).
    actions_arr_send = _rpy_to_axis_angle(actions_arr)
    send_type = "cartesian_abs_aa"

    # Rolling step-duration window so the user can see whether --fps is
    # actually achievable. The pacing block below is "best-effort": if
    # the per-step work exceeds ``period`` it just skips sleeping, so
    # the loop silently falls behind the requested rate without this log.
    _step_durs: deque = deque(maxlen=30)
    _last_fps_log_t = time.perf_counter()
    # Count step-delta clips (boxed so the inner block can mutate).
    _step_clip_count = [0]
    # Count preflight-IK rejections (branch-flip / unreachable holds).
    _ik_block_count = [0]

    _PROF.start(target_hz=float(args.fps))
    for t in track(
        range(T_replay),
        description=f"Replay episode {n} ...",
        transient=True, show_speed=True,
    ):
        if _ABORT_EVENT.is_set():
            break
        t_step = time.perf_counter()
        a_raw = actions_arr[t].copy().astype(np.float32)
        a_to_send = actions_arr_send[t].astype(np.float32)

        current_tcp = np.asarray(raw_obs["robot_obs"][:3], dtype=np.float64)
        cmd = _clip_action_to_xyz_bounds(
            {"type": send_type, "action": a_to_send},
            current_tcp, xyz_min, xyz_max,
        )
        cmd, was_step_clipped = _clip_to_max_step_delta(
            cmd, np.asarray(raw_obs["ee_pos_quat"], dtype=np.float64),
            args.max_step_xyz_m, args.max_step_rot_rad,
        )
        if was_step_clipped:
            _step_clip_count[0] += 1
            if _step_clip_count[0] <= 3 or _step_clip_count[0] % 30 == 0:
                logger.warning(
                    f"[ep {n} t={t}] step-delta clipped "
                    f"(total this episode: {_step_clip_count[0]})"
                )
        # IK strategy:
        # * --use_local_ik: compute joints via seeded DLS IK (numpy DH
        #   FK, current joints as seed) and send as joint_abs — the
        #   SDK IK never runs. Branch flips are structurally prevented.
        # * otherwise: send as cartesian_abs_aa with the controller's
        #   IK, but do the pre-flight check first to hold if a flip
        #   is predicted (best we can do with SDK 1.17.3, which lacks
        #   q_pre on get_inverse_kinematics).
        cur_joints = np.asarray(raw_obs["robot_obs"][7:14], dtype=np.float64)
        with _PROF.stage("ik"):
            if args.use_official_ik and official_kin is not None:
                cmd, ik_status = _official_ik_send(
                    cmd, env._follower, cur_joints, official_kin,
                    max_joint_delta_rad=args.max_step_joint_rad,
                )
            elif args.use_local_ik:
                cmd, ik_status = _local_ik_send(
                    cmd, env._follower, cur_joints,
                    max_joint_delta_rad=args.max_step_joint_rad,
                )
            else:
                cmd, ik_status = _preflight_ik_check(
                    cmd, env._follower, cur_joints, args.max_step_joint_rad,
                )
        if ik_status not in ("ok", "joint", "not_aa"):
            _ik_block_count[0] += 1
            if _ik_block_count[0] <= 3 or _ik_block_count[0] % 30 == 0:
                logger.warning(
                    f"[ep {n} t={t}] IK {ik_status} — holding "
                    f"current EE pose (total this episode: "
                    f"{_ik_block_count[0]})"
                )
        env.step(cmd, return_obs=False)

        (raw_obs, gripper_mask_hw, static_mask_hw, target_world,
         gripper_aff_raw) = _compute_step_obs(
            env, gripper_net, static_net, aff_transforms, data_cfg, run_cfg,
        )
        processed = _processed_obs_from_raw(
            raw_obs, gripper_mask_hw, static_mask_hw, target_world, env.img_size,
        )

        static_rgbs.append(processed["static_img_obs"].transpose(1, 2, 0))
        static_masks.append(static_mask_hw.squeeze().astype(np.uint8))
        gripper_rgb_hwc = processed["gripper_img_obs"].transpose(1, 2, 0)
        gripper_rgbs.append(gripper_rgb_hwc)
        if save_video:
            with _PROF.stage("aff_viz"):
                dir_frame = gripper_affordance_dir_frame(
                    gripper_net, gripper_rgb_hwc, gripper_aff_raw,
                    min_robustness=gripper_aff_min_robustness,
                )
            if dir_frame is not None:
                gripper_dir_overlays.append(dir_frame)
        gripper_depths.append(np.asarray(processed["gripper_depth_obs"]).squeeze())
        gripper_masks.append(np.asarray(gripper_mask_hw).squeeze().astype(np.uint8))
        positions.append(processed["robot_obs"])
        actions_phys.append(a_raw.astype(np.float32))
        ee_cur = np.asarray(raw_obs["ee_pos_quat"], dtype=np.float32)
        rel = _ee_rel_action(ee_cur, ee_prev, gripper_pm1=float(a_raw[6]))
        actions_rel.append(rel.astype(np.float32))
        ee_prev = ee_cur

        work_dur = time.perf_counter() - t_step
        _step_durs.append(work_dur)
        dt = period - work_dur
        if dt > 0:
            with _PROF.stage("sleep"):
                end_t = time.perf_counter() + dt
                while time.perf_counter() < end_t and not _ABORT_EVENT.is_set():
                    time.sleep(min(0.01, end_t - time.perf_counter()))
        _PROF.mark_step(tag=f"ep {n} t={t}")

        # Every ~30 steps, surface achieved vs. requested fps so the user
        # can spot pacing slips without digging into a profiler.
        if t > 0 and t % 30 == 0:
            avg_work = float(np.mean(_step_durs))
            max_work = float(np.max(_step_durs))
            avg_total = float(time.perf_counter() - _last_fps_log_t) / max(
                1, len(_step_durs)
            )
            achieved = 1.0 / max(avg_total, 1e-6)
            logger.info(
                f"[ep {n} t={t}] fps achieved={achieved:.1f} "
                f"(req={args.fps:.1f}) work_avg={avg_work*1000:.1f}ms "
                f"work_max={max_work*1000:.1f}ms period={period*1000:.1f}ms"
            )
            _last_fps_log_t = time.perf_counter()

    _PROF.report_episode(tag=f"ep {n} (replay)")

    def _stack(lst, dtype=None):
        if not lst:
            return np.zeros(0)
        arr = np.stack(lst, axis=0)
        return arr.astype(dtype) if dtype is not None else np.ascontiguousarray(arr)

    out = {
        "static_rgbs": _stack(static_rgbs),
        "static_masks": _stack(static_masks, dtype=np.uint8),
        "gripper_rgbs": _stack(gripper_rgbs),
        "gripper_depths": _stack(gripper_depths),
        "gripper_masks": _stack(gripper_masks, dtype=np.uint8),
        "positions": _stack(positions),
        "actions_abs": _build_actions_abs(
            positions, actions_phys, action_type,
            init_ee_pos_quat=ee_initial_for_integration,
        ),
        "actions_rel": _stack(actions_rel),
        "candidate_policies": _stack([]),
    }
    if gripper_dir_overlays:
        out["gripper_dir_overlays"] = _stack(gripper_dir_overlays)
    return out


def collect_episode(n, env: XArmRealEnv, target_search,
                    gripper_net, static_net, aff_transforms,
                    policy, policy_cfg, world, world_cfg,
                    data_cfg, run_cfg, args,
                    episode_path: str | Path | None = None,
                    save_video: bool = True,
                    official_kin=None) -> dict:
    """Run one episode and return stacked numpy arrays.

    ``episode_path`` is the (already-created) output directory for this
    episode; when set, the static-cam affordance detection viz is written to
    ``<episode_path>/affordance_init/`` at episode start, mirroring
    ``aif_collection_uv.collect_episode``.
    """
    # A delta_actions policy had its action_type rewritten to 'rel_actions' by
    # setup_delta_action_eval (main), which also forced --action_type
    # rel_actions; ``policy_is_delta`` only gates the per-chunk conversion.
    policy_is_delta = bool(getattr(args, "policy_is_delta", False))
    policy_is_abs = policy_cfg.datamodule.action_type == "actions"
    policy_is_mixed = policy_cfg.datamodule.action_type == "mixed"
    # When policy_is_joint, the policy outputs the 8-DoF leader-joint
    # stream and we ship it directly via ``joint_abs`` — no IK is run
    # on the inference side, so the SDK branch-flip problem is
    # structurally avoided. The world model may still be trained on
    # EE-pose actions/rel_actions/mixed; we FK-bridge between the two
    # representations at world-model-step time.
    policy_is_joint = policy_cfg.datamodule.action_type == "joint_actions"
    mixed_sources = list(getattr(
        policy_cfg.datamodule, "mixed_action_sources",
        ("rel", "rel", "rel", "abs", "abs", "abs", "abs")))
    action_type_disk = args.action_type  # what we save the policy actions as
    # World-model action_type (may differ from policy when policy is on
    # joints). When equal, no bridge is needed; when different, FK +
    # re-normalize per step before world.step_dynamics.
    world_action_type = (
        getattr(world_cfg.datamodule, "action_type", policy_cfg.datamodule.action_type)
        if world_cfg is not None else policy_cfg.datamodule.action_type
    )

    # ── reset + affordance-driven init pose ──
    raw_obs = env.reset()
    rng = np.random.RandomState(n)
    gripper_aff_min_robustness = float(
        getattr(env, "gripper_aff_min_robustness", 0.0))
    move_obs_history = affordance_random_init(
        env, target_search, rng, episode_path=episode_path,
        z_offset_m=float(getattr(args, "aff_target_z_offset", 0.0)),
        use_orientation=getattr(args, "use_orientation", True),
    )
    reset_policy_state(policy)

    # Whether the *policy* / *world model* expect RGB+depth (4 channels) or
    # RGB-only (3 channels) on the gripper input. Reading these here once
    # avoids per-step OmegaConf lookups (and lets us inline the branch
    # everywhere ``_gripper_input`` is called).
    policy_use_depth = bool(
        getattr(policy_cfg.datamodule, "use_depth", False)
    )
    world_use_depth = bool(
        getattr(world_cfg.datamodule, "use_depth", False)
        if world_cfg is not None else False
    )

    # Build the first processed/jax obs.
    (raw_obs, gripper_mask_hw, static_mask_hw, target_world,
     gripper_aff_raw) = _compute_step_obs(
        env, gripper_net, static_net, aff_transforms, data_cfg, run_cfg,
    )
    processed = _processed_obs_from_raw(
        raw_obs, gripper_mask_hw, static_mask_hw, target_world, env.img_size,
    )
    jax_obs = get_obs_data(
        processed, data_cfg,
        policy_action_type=policy_cfg.datamodule.action_type,
    )

    if args.calc_efe and world is not None:
        static_obs = jax_obs["static_rgb"]
        gripper_obs = _gripper_input(jax_obs, world_use_depth)
        pos = jax_obs["robot_pos"]
        static_emb, gripper_emb = encode_obs(
            world, static_obs, gripper_obs,
            obs_low=world_cfg.datamodule.obs_low, obs_high=world_cfg.datamodule.obs_high,
        )
        world.init_dynamics(1, static_emb, gripper_emb, pos)

    obs_length = policy_cfg.datamodule.obs_length
    obs_queue = _ObsQueue(maxlen=obs_length)
    while len(obs_queue.queue) < obs_length:
        obs_queue.append(jax_obs)

    # Frequency-aware sub-stepping. When the policy was trained with
    # rel-action accumulation at frequency > 1, each predicted action row is
    # an N-frame motion (translation summed / rotation composed over the
    # interval). To play it back AND record it at the native 30 Hz control
    # rate, split every row into ``sub_steps`` per-frame deltas, integrate to
    # absolute waypoints, and issue one env.step per sub-frame. The recorded
    # ``rel_actions`` (re-measured per frame) then match the recorder's 30 Hz
    # convention so the dataset loader's accumulation handles both uniformly.
    # ``sub_steps == 1`` reproduces the original one-send-per-row behaviour.
    policy_freq = int(getattr(policy_cfg.datamodule, "frequency", 1))
    rel_rot_repr = str(data_cfg.get(
        "delta_actions_rot_repr",
        getattr(policy_cfg.datamodule, "rel_action_rot_repr", "axis_angle")))
    sub_steps = (
        policy_freq
        if (policy_freq > 1
            and getattr(policy_cfg.datamodule, "rel_action_accumulate", False)
            and action_type_disk == "rel_actions"
            and not policy_is_mixed)   # mixed sends one abs waypoint per row
        else 1
    )
    if sub_steps > 1:
        logger.info(
            f"[freq] policy frequency={policy_freq} with rel-action "
            f"accumulation → replaying each action row as {sub_steps} "
            f"interpolated 30 Hz sub-frames."
        )

    # ── accumulators ──
    static_rgbs, static_masks = [], []
    gripper_rgbs, gripper_depths, gripper_masks = [], [], []
    # Per-frame gripper-cam affordance-direction overlays → rgb_gripper_aff_dirs.mp4
    # (aif_collection_uv parity; collected only when --save_video is on).
    gripper_dir_overlays = []
    positions, actions_phys, actions_rel = [], [], []
    extrinsics, epistemics, candidate_policies = [], [], []
    ee_prev = np.asarray(raw_obs["ee_pos_quat"], dtype=np.float32).copy()
    # Anchor for the rel→abs integration that builds ``actions.blosc2`` when
    # the policy outputs relative actions. Captured here, before the loop
    # issues its first command, so the integrated trajectory represents the
    # *commanded* EE poses (matches the recorder's `actions` convention).
    ee_initial_for_integration = ee_prev.copy()
    selected_policy = None
    # ``args.fps`` is resolved in main() (either user-supplied or the
    # auto-derived 30 / policy_cfg.datamodule.frequency default).
    period = 1.0 / float(args.fps)

    # Rolling step-duration window so the user can see whether --fps is
    # actually achievable. The pacing block at the loop tail is
    # "best-effort": when work_dur > period it skips sleeping, so the
    # loop silently runs at whatever rate the work allows.
    _step_durs: deque = deque(maxlen=30)
    _last_fps_log_t = time.perf_counter()
    # Count step-delta clips (boxed so the inner block can mutate).
    _step_clip_count = [0]
    # Count preflight-IK rejections (branch-flip / unreachable holds).
    _ik_block_count = [0]

    # ── Temporal Ensemble state ──
    # Re-created at every EFE-calc boundary when --calc_efe is on, or
    # lazily once at t=0 when --no_calc_efe. ``selected_policy`` /
    # ``selected_chunk_phys`` etc. are bypassed when TE is on — each
    # step yields a single ensembled action via ``_process_te_action``.
    te: Optional[TemporalEnsemble] = None
    use_te = bool(getattr(args, "use_temporal_ensemble", False))
    if use_te and sub_steps > 1:
        logger.warning(
            "[TE] --use_temporal_ensemble is incompatible with the "
            "sub-stepping pipeline (rel_action_accumulate); forcing "
            "sub_steps=1 for the TE branch."
        )

    # Recorded frames run at ``fps * sub_steps`` (== 30 Hz natively); TE
    # forces one frame per policy step.
    _PROF.start(target_hz=float(args.fps) * (1 if use_te else sub_steps))
    for t in track(
        range(args.episode_length),
        description=f"Collecting episode {n} ...",
        transient=True, show_speed=True,
    ):
        if _ABORT_EVENT.is_set():
            break
        t_step = time.perf_counter()
        jax_obs = obs_queue.get()

        # ── Temporal Ensemble path (per user spec) ──
        # When TE is on we do per-step inference + ensembling, so the
        # entire chunk pre-compute / sub-stepping that follows must be
        # bypassed. The TE block produces (te_a_raw, te_a_to_send) and
        # ``action`` (normalized) for world.step_dynamics; the
        # sub_steps loop below switches to one iteration that consumes
        # those values directly.
        te_a_raw: Optional[np.ndarray] = None
        te_a_to_send: Optional[np.ndarray] = None
        if use_te:
            boundary = (t % args.calc_efe_every == 0)
            with _PROF.stage("policy"):
                p_static_te, p_gripper_te = resize_for_policy(
                    policy_cfg,
                    jax_obs["static_rgb"][None],
                    _gripper_input(jax_obs, policy_use_depth)[None],
                )
                if not policy_cfg.datamodule.use_static:
                    p_static_te = jnp.zeros((1, 1, 1, 1, 1), dtype=jnp.float32)
                pos_in = jax_obs["robot_pos"][..., policy_cfg.datamodule.use_joint_indices][None]

            if boundary and args.calc_efe:
                # Reset TE, sample candidates, run EFE, store winner.
                with _PROF.stage("policy"):
                    policies_te = _jax_sync(policy.inference(
                        args.num_candidate_policies,
                        p_static_te, p_gripper_te, pos_in,
                    ))
                if policies_te.shape[-1] == 5:
                    xyz_ = policies_te[..., :3]
                    rp_z = jnp.zeros_like(policies_te[..., 3:5])
                    yg_ = policies_te[..., 3:]
                    policies_te = jnp.concatenate([xyz_, rp_z, yg_], axis=-1)
                expected_dim_te = 8 if policy_is_joint else 7
                assert policies_te.shape[-1] == expected_dim_te
                if policy_is_delta:
                    policies_te = jnp.asarray(delta_chunk_to_rel_norm(
                        np.asarray(policies_te), data_cfg))
                candidate_policies.append(np.asarray(policies_te))
                if args.constraint:
                    policies_te = policies_te.at[..., -3].set(0.0)
                    policies_te = policies_te.at[..., -4].set(0.0)

                # Bridge joint_actions → world action_type if needed.
                if policy_is_joint and world_action_type != "joint_actions":
                    j_phys_te = joint_detransform(
                        np.asarray(policies_te, dtype=np.float32),
                        data_cfg["joint_actions_max"],
                        data_cfg["joint_actions_min"],
                    )
                    ee_anchor_te = np.asarray(
                        raw_obs["ee_pos_quat"], dtype=np.float64,
                    )
                    conv_te = np.zeros(
                        (j_phys_te.shape[0], j_phys_te.shape[1], 7),
                        dtype=np.float32,
                    )
                    for c in range(j_phys_te.shape[0]):
                        conv_te[c] = _joint_chunk_to_world_action(
                            j_phys_te[c], ee_anchor_te,
                            world_action_type, data_cfg, mixed_sources,
                        )
                    policies_for_world_te = jnp.asarray(conv_te)
                else:
                    policies_for_world_te = policies_te

                stride_te = max(1, world_cfg.datamodule.frequency
                                // policy_cfg.datamodule.frequency)
                with _PROF.stage("efe_target"):
                    nearest_te, nearest_src_te = _select_nearest_efe_target(
                        env, target_search, data_cfg, t=t, n=n,
                        use_orientation=getattr(args, "use_orientation", True),
                    )
                if nearest_te is None:
                    disable_ext_te = True
                    target_efe_te = jax_obs["target_pos"][None]
                else:
                    disable_ext_te = False
                    target_efe_te = nearest_te
                with _PROF.stage("efe"):
                    efe_te, argmin_te, extr_te, epis_te = _jax_sync(
                        imagine_and_calc_efe(
                            world,
                            policies_for_world_te[:, ::stride_te],
                            target_efe_te,
                            np.sqrt(args.pref_var),
                            disable_ext_te,
                            pos_time_reduce=args.extrinsic_pos_reduce,
                        )
                    )
                extrinsics.append(np.asarray(extr_te))
                epistemics.append(np.asarray(epis_te))
                selected_policy = policies_te[argmin_te]
                te = TemporalEnsemble(decay=args.temporal_ensemble_decay)
                te.add_chunk(np.asarray(selected_policy))
                logger.debug(
                    f"[TE] ep={n} t={t}: reset + EFE-chosen chunk added"
                )
            else:
                # Single-chunk inference (no EFE), or first frame in
                # --no_calc_efe mode where TE is initialised lazily.
                with _PROF.stage("policy"):
                    policies_te = _jax_sync(policy.inference(
                        1, p_static_te, p_gripper_te, pos_in,
                    ))
                if policies_te.shape[-1] == 5:
                    xyz_ = policies_te[..., :3]
                    rp_z = jnp.zeros_like(policies_te[..., 3:5])
                    yg_ = policies_te[..., 3:]
                    policies_te = jnp.concatenate([xyz_, rp_z, yg_], axis=-1)
                if policy_is_delta:
                    policies_te = jnp.asarray(delta_chunk_to_rel_norm(
                        np.asarray(policies_te), data_cfg))
                if args.constraint:
                    policies_te = policies_te.at[..., -3].set(0.0)
                    policies_te = policies_te.at[..., -4].set(0.0)
                chunk_one = np.asarray(policies_te[0])
                if te is None:
                    te = TemporalEnsemble(decay=args.temporal_ensemble_decay)
                te.add_chunk(chunk_one)

            # Ensemble → single normalized action for this step.
            ensembled = te.select()
            te_a_raw, te_a_to_send = _process_te_action(
                ensembled,
                policy_cfg=policy_cfg, data_cfg=data_cfg,
                mixed_sources=mixed_sources,
                current_ee_pos_quat=raw_obs["ee_pos_quat"],
                policy_is_joint=policy_is_joint,
                policy_is_mixed=policy_is_mixed,
                action_type_disk=action_type_disk,
            )
            # ``action`` (normalized jax row) for world.step_dynamics.
            action = jnp.asarray(ensembled)

        if (not use_te) and t % args.calc_efe_every == 0:
            with _PROF.stage("policy"):
                p_static, p_gripper = resize_for_policy(
                    policy_cfg,
                    jax_obs["static_rgb"][None],
                    _gripper_input(jax_obs, policy_use_depth)[None],
                )
                if not policy_cfg.datamodule.use_static:
                    p_static = jnp.zeros((1, 1, 1, 1, 1), dtype=jnp.float32)
                policies = _jax_sync(policy.inference(
                    args.num_candidate_policies,
                    p_static, p_gripper,
                    jax_obs["robot_pos"][..., policy_cfg.datamodule.use_joint_indices][None],
                ))
            if policies.shape[-1] == 5:
                xyz = policies[..., :3]
                rp_zero = jnp.zeros_like(policies[..., 3:5])
                yaw_grip = policies[..., 3:]
                policies = jnp.concatenate([xyz, rp_zero, yaw_grip], axis=-1)
            expected_dim = 8 if policy_is_joint else 7
            assert policies.shape[-1] == expected_dim, (
                f"policies shape: {policies.shape}, expected last dim "
                f"{expected_dim} for action_type "
                f"{policy_cfg.datamodule.action_type!r}"
            )
            if policy_is_delta:
                # Chunk-relative offsets → per-step rel actions. Converted
                # before EFE / candidate logging / execution so every consumer
                # downstream — including the world model, which never saw a
                # chunk-relative stream — works on anchor-free deltas.
                policies = jnp.asarray(delta_chunk_to_rel_norm(
                    np.asarray(policies), data_cfg))
            candidate_policies.append(np.asarray(policies))

            if args.constraint:
                policies = policies.at[..., -3].set(0.0)
                policies = policies.at[..., -4].set(0.0)

            if args.calc_efe and world is not None:
                stride = max(1, world_cfg.datamodule.frequency
                              // policy_cfg.datamodule.frequency)
                # Combine gripper-cam (env.curr_detected_obj) with a fresh
                # static-cam scan, pick the candidate nearest to current TCP
                # for the extrinsic term. Only when *both* views come up
                # empty do we fall back to epistemic-only.
                # Runs a fresh static-cam scan, so its camera read shows up
                # under cam_static (exclusive timing) rather than here.
                with _PROF.stage("efe_target"):
                    nearest_target, nearest_src = _select_nearest_efe_target(
                        env, target_search, data_cfg, t=t, n=n,
                        use_orientation=getattr(args, "use_orientation", True),
                    )
                if nearest_target is None:
                    logger.info(f"[ep {n} t={t}] no affordance candidate in either "
                                f"view — extrinsic disabled for this step")
                    disable_ext = True
                    target_for_efe = jax_obs["target_pos"][None]  # placeholder, unused
                else:
                    disable_ext = False
                    target_for_efe = nearest_target
                    if nearest_src == "static":
                        logger.debug(f"[ep {n} t={t}] using nearest static-cam "
                                     f"candidate for extrinsic")
                # When the policy was trained on joint_actions but the
                # world model expects EE-pose actions, bridge per
                # candidate: denorm joints → FK → re-normalize to the
                # world's action_type. ``ee_anchor`` is the *current*
                # measured EE — that's where each candidate's chunk
                # logically starts (matches what rel-action integration
                # uses elsewhere). This is the ~35ms hot spot but only
                # fires every ``calc_efe_every`` steps. When policy and
                # world share ``action_type`` (incl. both joint_actions)
                # we just forward as-is.
                if policy_is_joint and world_action_type != "joint_actions":
                    j_phys = joint_detransform(
                        np.asarray(policies, dtype=np.float32),
                        data_cfg["joint_actions_max"],
                        data_cfg["joint_actions_min"],
                    )
                    ee_anchor_now = np.asarray(
                        raw_obs["ee_pos_quat"], dtype=np.float64,
                    )
                    converted = np.zeros(
                        (j_phys.shape[0], j_phys.shape[1], 7),
                        dtype=np.float32,
                    )
                    for c in range(j_phys.shape[0]):
                        converted[c] = _joint_chunk_to_world_action(
                            j_phys[c], ee_anchor_now,
                            world_action_type, data_cfg, mixed_sources,
                        )
                    policies_for_world = jnp.asarray(converted)
                else:
                    policies_for_world = policies
                with _PROF.stage("efe"):
                    efe, argmin, extrinsic, epistemic = _jax_sync(
                        imagine_and_calc_efe(
                            world,
                            policies_for_world[:, ::stride],
                            target_for_efe,
                            np.sqrt(args.pref_var),
                            disable_ext,
                            pos_time_reduce=args.extrinsic_pos_reduce,
                        )
                    )
                extrinsics.append(np.asarray(extrinsic))
                epistemics.append(np.asarray(epistemic))
                selected_policy = policies[argmin]
            else:
                selected_policy = policies[0]

            # Pre-compute the entire selected chunk in physical units +
            # absolute-pose form for the next ``calc_efe_every`` steps.
            # Done once at selection time (not per-step) so that for
            # `--action_type rel_actions` the rel→abs integration uses
            # a **single deterministic anchor** — the EE at the moment
            # the chunk was picked — and walks the chain forward via
            # commanded deltas. Composing per-step would re-anchor at
            # the measured EE each step, baking tracking drift into the
            # chunk. xArm SDK's ``command_cartesian_relative`` also
            # mis-interprets axis-angle as Euler RPY (see
            # compare_actions_vs_rel.py), so we always send via
            # ``cartesian_abs``.
            #
            # The gripper channel is *not* binarised here — the recorder
            # writes the follower's continuous opening fraction (mapped
            # to ±1 by ``_ee_rel_action``), and ``XArmRealEnv.step``'s
            # ``grip_norm = clip((1 - grip_pm1) / 2, 0, 1)`` accepts the
            # full continuous range, so a partially-closed grasp from
            # the policy passes through faithfully.
            _chunk_norm = np.asarray(selected_policy, dtype=np.float32).copy()
            if policy_is_mixed:
                selected_chunk_phys = np.asarray(joint_detransform_mixed(
                    _chunk_norm, data_cfg, mixed_sources), dtype=np.float32)
            else:
                selected_chunk_phys = joint_detransform(
                    _chunk_norm,
                    data_cfg[f"{policy_cfg.datamodule.action_type}_max"],
                    data_cfg[f"{policy_cfg.datamodule.action_type}_min"],
                ).astype(np.float32)
            # Build the chunk that gets *sent* to the SDK. The rotation
            # channel switches to **axis-angle** so the wrist target is
            # continuous around roll ≈ ±π (Euler RPY flips sign at the
            # wrap point, which the SDK reads as a near-2π rotation
            # command and the wrist visibly snaps). ``send_type`` is
            # ``cartesian_abs_aa`` which goes through xArm SDK's
            # ``set_position_aa`` via the ``_patch_follower_aa_send``
            # hook installed at env init.
            if policy_is_joint:
                # Policy outputs absolute joint targets — clip to xArm7
                # joint limits and pass straight through to the
                # ``joint_abs`` transport. No IK, no Cartesian work,
                # no branch-flip surface. Gripper col untouched.
                jclip = selected_chunk_phys.copy()
                jclip[:, :7] = np.clip(
                    jclip[:, :7],
                    _XARM7_JOINT_LIMITS[:, 0].astype(np.float32),
                    _XARM7_JOINT_LIMITS[:, 1].astype(np.float32),
                )
                selected_chunk_phys = jclip
                selected_chunk_send = selected_chunk_phys.copy()
            elif policy_is_mixed:
                # Mixed: integrate rel translation from the current EE anchor,
                # take rotation as absolute RPY→AA. One abs waypoint per row
                # (sub_steps forced to 1 above).
                ee_anchor = np.asarray(raw_obs["ee_pos_quat"], dtype=np.float64)
                selected_chunk_send = _integrate_mixed_to_abs_aa(
                    ee_anchor[:3], selected_chunk_phys.astype(np.float64),
                    mixed_sources,
                ).astype(np.float32)
            elif action_type_disk == "rel_actions":
                ee_anchor = np.asarray(raw_obs["ee_pos_quat"], dtype=np.float64)
                selected_chunk_send = _integrate_rel_to_abs_aa(
                    ee_anchor[:3], ee_anchor[3:7],
                    selected_chunk_phys.astype(np.float64),
                ).astype(np.float32)
            else:
                # `actions` mode: policy outputs absolute RPY (xArm SDK
                # convention). Convert to axis-angle in bulk for the
                # AA send transport.
                selected_chunk_send = _rpy_to_axis_angle(selected_chunk_phys)

            # Build the 30 Hz-resolution chunk for sub-stepped playback. Each
            # ``sub_steps``-frame policy row is split into per-frame deltas
            # (translation / N, rotation slerp, gripper held) and integrated
            # to absolute axis-angle waypoints from the SAME anchor, so the
            # interpolated intermediate poses lie on the chunk's own
            # trajectory. ``sub_steps == 1`` makes the fine arrays identical
            # to the coarse ones.
            if sub_steps > 1 and action_type_disk == "rel_actions":
                selected_chunk_phys_fine = np.concatenate(
                    [split_rel_action(row, sub_steps, rel_rot_repr)
                     for row in selected_chunk_phys],
                    axis=0,
                ).astype(np.float32)
                selected_chunk_send_fine = _integrate_rel_to_abs_aa(
                    ee_anchor[:3], ee_anchor[3:7],
                    selected_chunk_phys_fine.astype(np.float64),
                ).astype(np.float32)
            else:
                selected_chunk_phys_fine = selected_chunk_phys
                selected_chunk_send_fine = selected_chunk_send

        # Pick this policy step's row. ``action`` (normalized jax row) is
        # consumed by ``world.step_dynamics`` once per policy step. Gripper
        # kept continuous (no binarisation) so the world model sees the same
        # distribution it was trained on (full ±1 continuous range).
        # In the TE branch ``action`` was already set above (= the
        # ensembled normalized vector); skip the chunk index pull.
        if not use_te:
            local_t = t % args.calc_efe_every
            action = selected_policy[local_t]
        send_type = "joint_abs" if policy_is_joint else "cartesian_abs_aa"
        # Native-rate sub-frame period (== 1/30 s when fps == 30/sub_steps).
        # TE forces a single (non-sub-stepped) per-frame execution.
        sub_steps_loop = 1 if use_te else sub_steps
        sub_period = period / sub_steps_loop

        # Replay the policy row as ``sub_steps_loop`` interpolated frames.
        # Each sub-frame issues one env.step against an axis-angle absolute
        # waypoint from the fine chunk and records a per-frame sample, so the
        # saved episode is at the native control rate (sub_steps == 1 → exactly
        # one send/record, identical to the legacy behaviour).
        for s in range(sub_steps_loop):
            sub_t = time.perf_counter()
            # ``a_raw`` is the per-frame physical-unit delta (rel) used for the
            # integrated ``actions.blosc2``; ``a_to_send`` is the axis-angle
            # absolute waypoint for ``cartesian_abs_aa``.
            if use_te:
                a_raw = te_a_raw
                a_to_send = te_a_to_send
            else:
                fi = local_t * sub_steps + s
                a_raw = selected_chunk_phys_fine[fi]
                a_to_send = selected_chunk_send_fine[fi]
            if policy_is_joint:
                # Joint-target mode: skip every EE-space safety check
                # (xyz bounds clip, EE-delta cap, IK preflight, local IK)
                # because none of them apply when ``a_to_send`` is
                # already a joint vector. Apply joint-space caps
                # instead: clip to limits + cap per-step joint delta vs
                # the measured current joints.
                cur_joints = np.asarray(raw_obs["robot_obs"][7:14], dtype=np.float64)
                a_clipped = np.asarray(a_to_send, dtype=np.float64).copy()
                a_clipped[:7] = np.clip(
                    a_clipped[:7],
                    _XARM7_JOINT_LIMITS[:, 0], _XARM7_JOINT_LIMITS[:, 1],
                )
                if args.max_step_joint_rad > 0:
                    delta = a_clipped[:7] - cur_joints
                    dmax = float(np.max(np.abs(delta)))
                    if dmax > args.max_step_joint_rad:
                        a_clipped[:7] = cur_joints + delta * (
                            args.max_step_joint_rad / dmax
                        )
                        _step_clip_count[0] += 1
                        if _step_clip_count[0] <= 3 or _step_clip_count[0] % 30 == 0:
                            logger.warning(
                                f"[ep {n} t={t}.{s}] joint-delta clipped "
                                f"(max={dmax:.3f}rad, total this episode: "
                                f"{_step_clip_count[0]})"
                            )
                cmd = {
                    "type": "joint_abs",
                    "action": a_clipped.astype(np.float32),
                }
            else:
                # Clip TCP xyz to the human-data bounds (`data_cfg["pos_min/max"][:3]`)
                # so the model can't drive the arm outside the trained workspace.
                xyz_min = np.asarray(data_cfg["pos_min"][:3], dtype=np.float64)
                xyz_max = np.asarray(data_cfg["pos_max"][:3], dtype=np.float64)
                current_tcp = np.asarray(raw_obs["robot_obs"][:3], dtype=np.float64)
                cmd = _clip_action_to_xyz_bounds(
                    {"type": send_type, "action": a_to_send},
                    current_tcp, xyz_min, xyz_max,
                )
                # Cap per-step EE delta vs the measured current pose to prevent
                # IK branch-flip "snap" when the policy commands a target far from
                # the current EE. See ``_clip_to_max_step_delta``.
                cmd, was_step_clipped = _clip_to_max_step_delta(
                    cmd, np.asarray(raw_obs["ee_pos_quat"], dtype=np.float64),
                    args.max_step_xyz_m, args.max_step_rot_rad,
                )
                if was_step_clipped:
                    _step_clip_count[0] += 1
                    if _step_clip_count[0] <= 3 or _step_clip_count[0] % 30 == 0:
                        logger.warning(
                            f"[ep {n} t={t}.{s}] step-delta clipped "
                            f"(total this episode: {_step_clip_count[0]})"
                        )
                # IK strategy:
                #  * --use_official_ik → UFactory's seeded IK
                #  * --use_local_ik    → seeded DLS IK on local DH
                #  * otherwise         → SDK pre-flight + AA send
                cur_joints = np.asarray(raw_obs["robot_obs"][7:14], dtype=np.float64)
                # The default (no --use_*_ik) branch is one blocking SDK
                # round-trip per sub-frame — the profiler's "ik" row.
                with _PROF.stage("ik"):
                    if args.use_official_ik and official_kin is not None:
                        cmd, ik_status = _official_ik_send(
                            cmd, env._follower, cur_joints, official_kin,
                            max_joint_delta_rad=args.max_step_joint_rad,
                        )
                    elif args.use_local_ik:
                        cmd, ik_status = _local_ik_send(
                            cmd, env._follower, cur_joints,
                            max_joint_delta_rad=args.max_step_joint_rad,
                        )
                    else:
                        cmd, ik_status = _preflight_ik_check(
                            cmd, env._follower, cur_joints, args.max_step_joint_rad,
                        )
                if ik_status not in ("ok", "joint", "not_aa"):
                    _ik_block_count[0] += 1
                    if _ik_block_count[0] <= 3 or _ik_block_count[0] % 30 == 0:
                        logger.warning(
                            f"[ep {n} t={t}.{s}] IK {ik_status} — "
                            f"holding current EE pose (total this episode: "
                            f"{_ik_block_count[0]})"
                        )
            env.step(cmd, return_obs=False)

            # Re-read obs + recompute affordance for the saved frame.
            (raw_obs, gripper_mask_hw, static_mask_hw, target_world,
             gripper_aff_raw) = _compute_step_obs(
                env, gripper_net, static_net, aff_transforms, data_cfg, run_cfg,
            )
            processed = _processed_obs_from_raw(
                raw_obs, gripper_mask_hw, static_mask_hw, target_world, env.img_size,
            )

            # Accumulators (per 30 Hz frame).
            static_rgbs.append(processed["static_img_obs"].transpose(1, 2, 0))
            static_masks.append(static_mask_hw.squeeze().astype(np.uint8))
            gripper_rgb_hwc = processed["gripper_img_obs"].transpose(1, 2, 0)
            gripper_rgbs.append(gripper_rgb_hwc)
            if save_video:
                with _PROF.stage("aff_viz"):
                    dir_frame = gripper_affordance_dir_frame(
                        gripper_net, gripper_rgb_hwc, gripper_aff_raw,
                        min_robustness=gripper_aff_min_robustness,
                    )
                if dir_frame is not None:
                    gripper_dir_overlays.append(dir_frame)
            gripper_depths.append(np.asarray(processed["gripper_depth_obs"]).squeeze())
            gripper_masks.append(np.asarray(gripper_mask_hw).squeeze().astype(np.uint8))
            positions.append(processed["robot_obs"])
            actions_phys.append(a_raw.astype(np.float32))
            # Compute symmetric rel/abs pair so the dataset always carries both.
            ee_cur = np.asarray(raw_obs["ee_pos_quat"], dtype=np.float32)
            rel = _ee_rel_action(ee_cur, ee_prev, gripper_pm1=float(a_raw[6]))
            actions_rel.append(rel.astype(np.float32))
            ee_prev = ee_cur

            new_jax_obs = get_obs_data(
                processed, data_cfg,
                policy_action_type=policy_cfg.datamodule.action_type,
            )
            obs_queue.append(new_jax_obs)

            # Pace each sub-frame to the native control rate (interruptible).
            # Note there is no catch-up: when sub_work already exceeds
            # sub_period we simply don't sleep, and the loop silently runs
            # below --fps. A near-zero "sleep" row in the profile means the
            # requested rate is unreachable, not that pacing is disabled.
            sub_work = time.perf_counter() - sub_t
            _step_durs.append(sub_work)
            sub_dt = sub_period - sub_work
            if sub_dt > 0:
                with _PROF.stage("sleep"):
                    end_t = time.perf_counter() + sub_dt
                    while time.perf_counter() < end_t and not _ABORT_EVENT.is_set():
                        time.sleep(min(0.01, end_t - time.perf_counter()))
            _PROF.mark_step(tag=f"ep {n} t={t}.{s}")
            if _ABORT_EVENT.is_set():
                break

        # World-model dynamics step (once per policy step; uses the latest
        # sub-frame obs + the policy row action).
        if args.calc_efe and world is not None:
            with _PROF.stage("world"):
                static_obs = new_jax_obs["static_rgb"]
                gripper_obs = _gripper_input(new_jax_obs, world_use_depth)
                pos = new_jax_obs["robot_pos"]
                static_emb, gripper_emb = _jax_sync(encode_obs(
                    world, static_obs, gripper_obs,
                    obs_low=world_cfg.datamodule.obs_low,
                    obs_high=world_cfg.datamodule.obs_high,
                ))
                world_stride = max(1, world_cfg.datamodule.frequency
                                    // policy_cfg.datamodule.frequency)
                if t % world_stride == 0:
                    if policy_is_joint and world_action_type != "joint_actions":
                        # Convert just this one row: denorm joints → FK → re-
                        # normalize to world's action_type, then send to
                        # ``world.step_dynamics``. Anchor at the previous
                        # measured EE (= ``ee_prev``) so a rel-actions world
                        # model sees consistent deltas.
                        j_phys_row = joint_detransform(
                            np.asarray(action, dtype=np.float32).reshape(1, -1),
                            data_cfg["joint_actions_max"],
                            data_cfg["joint_actions_min"],
                        )
                        w_action_row = _joint_chunk_to_world_action(
                            j_phys_row, np.asarray(ee_prev, dtype=np.float64),
                            world_action_type, data_cfg, mixed_sources,
                        )
                        world_action = jnp.asarray(w_action_row[0])
                    else:
                        world_action = jnp.copy(action)
                    _jax_sync(world.step_dynamics(
                        world_action.reshape(1, -1), pos, static_emb, gripper_emb,
                    ))

        # Every ~30 steps surface achieved vs. requested fps + the
        # slowest sub-step so the operator can spot pacing slips
        # (calc_efe-step hits and affordance forwards are the usual
        # suspects on this multi-model loop).
        if t > 0 and t % 30 == 0:
            avg_work = float(np.mean(_step_durs))
            max_work = float(np.max(_step_durs))
            avg_total = float(time.perf_counter() - _last_fps_log_t) / max(
                1, len(_step_durs)
            )
            achieved = 1.0 / max(avg_total, 1e-6)
            logger.info(
                f"[ep {n} t={t}] fps achieved={achieved:.1f} "
                f"(req={args.fps:.1f}) work_avg={avg_work*1000:.1f}ms "
                f"work_max={max_work*1000:.1f}ms period={period*1000:.1f}ms"
            )
            _last_fps_log_t = time.perf_counter()

    _PROF.report_episode(tag=f"ep {n}")

    # ── pack ──
    def _stack(lst, dtype=None):
        if not lst:
            return np.zeros(0)
        arr = np.stack(lst, axis=0)
        return arr.astype(dtype) if dtype is not None else np.ascontiguousarray(arr)

    out = {
        "static_rgbs": _stack(static_rgbs),
        "static_masks": _stack(static_masks, dtype=np.uint8),
        "gripper_rgbs": _stack(gripper_rgbs),
        "gripper_depths": _stack(gripper_depths),
        "gripper_masks": _stack(gripper_masks, dtype=np.uint8),
        "positions": _stack(positions),
        "actions_abs": _build_actions_abs(
            positions, actions_phys, args.action_type,
            init_ee_pos_quat=ee_initial_for_integration,
        ),
        "actions_rel": _stack(actions_rel),
        "candidate_policies": _stack(candidate_policies),
    }
    if args.calc_efe:
        out["extrinsics"] = _stack(extrinsics)
        out["epistemics"] = _stack(epistemics)
    if gripper_dir_overlays:
        out["gripper_dir_overlays"] = _stack(gripper_dir_overlays)
    return out


def _ee_rel_action(cur_ee: np.ndarray, prev_ee: np.ndarray,
                   gripper_pm1: float) -> np.ndarray:
    """Compute a CALVIN-style 7-DoF relative action ``[dx, dy, dz, drx, dry, drz, grip]``.

    Lifted from gello_teleop_recorder_xarm.py:_ee_rel_action. Position delta
    in meters, rotation delta as quaternion-difference axis-angle (rad).
    """
    if prev_ee is None:
        return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper_pm1], dtype=np.float32)
    dxyz = (cur_ee[:3] - prev_ee[:3]).astype(np.float32)

    def _quat_mul(a, b):
        x1, y1, z1, w1 = a
        x2, y2, z2, w2 = b
        return np.array([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
        ], dtype=np.float32)

    q_prev_conj = np.array([-prev_ee[3], -prev_ee[4], -prev_ee[5], prev_ee[6]],
                            dtype=np.float32)
    q_delta = _quat_mul(cur_ee[3:7].astype(np.float32), q_prev_conj)
    n = float(np.linalg.norm(q_delta))
    if n == 0.0:
        drot = np.zeros(3, dtype=np.float32)
    else:
        q_delta = q_delta / n
        angle = 2.0 * float(np.arccos(np.clip(q_delta[3], -1.0, 1.0)))
        axis_norm = float(np.linalg.norm(q_delta[:3]))
        if axis_norm < 1e-8 or angle < 1e-8:
            drot = np.zeros(3, dtype=np.float32)
        else:
            drot = (q_delta[:3] / axis_norm * angle).astype(np.float32)
    return np.concatenate([dxyz, drot, [gripper_pm1]]).astype(np.float32)


def _integrate_rel_to_abs(
    init_xyz_m: np.ndarray,
    init_quat_xyzw: np.ndarray,
    rel_actions: np.ndarray,
) -> np.ndarray:
    """Integrate a sequence of ``_ee_rel_action``-format rel actions into
    an absolute-pose trajectory.

    Exact inverse of :func:`_ee_rel_action` (= the recorder's per-frame
    formula). Mirrors :func:`compare_actions_vs_rel.rel_actions_to_abs_actions`
    -- kept inline here to avoid cross-script imports.

      * position : ``pos_t = pos_{t-1} + dxyz``
      * rotation : ``q_t   = q_delta ⊗ q_{t-1}`` where
                   ``q_delta = exp_axis_angle(drot)``

    Args:
        init_xyz_m: ``(3,)`` starting EE position in meters.
        init_quat_xyzw: ``(4,)`` starting EE orientation as xyzw unit quat.
        rel_actions: ``(T, 7)`` array ``[dxyz, drot_axisangle, grip_pm1]``
                     in (m + rad + ±1).

    Returns:
        ``(T, 7)`` array ``[x, y, z, roll, pitch, yaw, grip_pm1]`` -- the
        same channel layout as the recorder's ``actions.blosc2`` (Euler
        extrinsic XYZ to match xArm SDK's ``return_is_radian=True``).
    """
    rel_actions = np.asarray(rel_actions, dtype=np.float64)
    if rel_actions.ndim != 2 or rel_actions.shape[1] != 7:
        raise ValueError(f"rel_actions must be (T, 7); got {rel_actions.shape}")

    T = rel_actions.shape[0]
    out = np.zeros((T, 7), dtype=np.float32)
    pos = np.asarray(init_xyz_m, dtype=np.float64).reshape(3).copy()
    rot = _R.from_quat(np.asarray(init_quat_xyzw, dtype=np.float64).reshape(4))
    for t in range(T):
        pos = pos + rel_actions[t, :3]
        rot = _R.from_rotvec(rel_actions[t, 3:6]) * rot
        rpy = rot.as_euler("xyz")
        out[t, :3] = pos.astype(np.float32)
        out[t, 3:6] = rpy.astype(np.float32)
        out[t, 6] = np.float32(rel_actions[t, 6])
    return out


class TemporalEnsemble:
    """ACT-style temporal action ensembling.

    Each step the policy emits a chunk of ``K`` predicted actions in
    normalized space. Multiple overlapping chunks make predictions for
    the same absolute step; this buffer averages them with exponential
    weights based on chunk age:

        w_i = exp(-decay * i)        normalized,
        i   = how many steps ago the prediction was generated.

    Bias-towards-recent: larger ``decay`` → trust newer chunks more.
    Smaller ``decay`` → more smoothing. Default 0.01 matches the value
    used in the original ACT paper.

    Use:
      * Construct (or call ``reset()``) at the start of an EFE window.
      * For every policy step, ``add_chunk(chunk)`` and then
        ``select()`` to get the ensembled action for the current step.
      * The step counter advances automatically each ``select()``.
    """

    def __init__(self, decay: float = 0.01) -> None:
        self.decay = float(decay)
        self.current_step = 0
        # abs_step -> list of (age_when_used, action_vec_normalized)
        self.predictions: Dict[int, List[Tuple[int, np.ndarray]]] = {}

    def reset(self) -> None:
        self.current_step = 0
        self.predictions = {}

    def add_chunk(self, chunk: np.ndarray) -> None:
        """Append predictions from a chunk emitted at the current step.

        Args:
            chunk: ``(K, A)`` normalized actions. ``chunk[i]`` is the
                prediction for absolute step ``current_step + i``.
        """
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.ndim != 2:
            raise ValueError(f"chunk must be (K, A); got {chunk.shape}")
        K = chunk.shape[0]
        for offset in range(K):
            future_t = self.current_step + offset
            self.predictions.setdefault(future_t, []).append(
                (offset, chunk[offset].copy())
            )

    def select(self) -> np.ndarray:
        """Return the ensembled action for the current step + advance."""
        t = self.current_step
        preds = self.predictions.pop(t, [])
        if not preds:
            raise RuntimeError(
                f"TemporalEnsemble has no predictions for step {t}; "
                "call ``add_chunk`` before ``select``."
            )
        ages = np.array([p[0] for p in preds], dtype=np.float32)
        actions = np.stack([p[1] for p in preds], axis=0)
        w = np.exp(-self.decay * ages)
        w = w / w.sum()
        ensembled = (actions * w[:, None]).sum(axis=0).astype(np.float32)
        self.current_step += 1
        return ensembled


def _process_te_action(
    a_norm: np.ndarray,
    *,
    policy_cfg,
    data_cfg,
    mixed_sources: list,
    current_ee_pos_quat: np.ndarray,
    policy_is_joint: bool,
    policy_is_mixed: bool,
    action_type_disk: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a single ensembled (normalized) action row into
    ``(a_raw, a_to_send)`` ready for ``env.step``.

    Mirrors the chunk-level pre-compute (line ~3290 area) but operates
    on one row at a time so it can be invoked per step under TE. The
    anchor for rel/mixed integration is the **current measured EE** —
    matches "policy thinks I should apply this delta from where the
    arm is now", which is the natural semantic for an ensembled
    instantaneous delta.
    """
    a_chunk_norm = np.asarray(a_norm, dtype=np.float32).reshape(1, -1)
    if policy_is_mixed:
        a_phys = np.asarray(
            joint_detransform_mixed(a_chunk_norm, data_cfg, mixed_sources),
            dtype=np.float32,
        )
    else:
        a_phys = joint_detransform(
            a_chunk_norm,
            data_cfg[f"{policy_cfg.datamodule.action_type}_max"],
            data_cfg[f"{policy_cfg.datamodule.action_type}_min"],
        ).astype(np.float32)

    if policy_is_joint:
        jclip = a_phys.copy()
        jclip[:, :7] = np.clip(
            jclip[:, :7],
            _XARM7_JOINT_LIMITS[:, 0].astype(np.float32),
            _XARM7_JOINT_LIMITS[:, 1].astype(np.float32),
        )
        return jclip[0], jclip[0]

    ee_anchor = np.asarray(current_ee_pos_quat, dtype=np.float64)
    if policy_is_mixed:
        a_send = _integrate_mixed_to_abs_aa(
            ee_anchor[:3], a_phys.astype(np.float64), mixed_sources,
        ).astype(np.float32)
    elif action_type_disk == "rel_actions":
        a_send = _integrate_rel_to_abs_aa(
            ee_anchor[:3], ee_anchor[3:7], a_phys.astype(np.float64),
        ).astype(np.float32)
    else:
        a_send = _rpy_to_axis_angle(a_phys)
    return a_phys[0], a_send[0]


def _integrate_rel_to_abs_aa(
    init_xyz_m: np.ndarray,
    init_quat_xyzw: np.ndarray,
    rel_actions: np.ndarray,
) -> np.ndarray:
    """Gimbal-lock-free counterpart of :func:`_integrate_rel_to_abs`.

    Same integration arithmetic, but each row is emitted as
    ``[x, y, z, rx, ry, rz, grip_pm1]`` where ``(rx, ry, rz)`` is the
    **axis-angle** (rotvec) of the absolute orientation at step ``t``.
    Avoids the ``as_euler("xyz")`` round-trip whose output sign jumps
    by ~2π whenever roll crosses ±π — the source of the
    "RX+ ↔ RX-" reversal observed when the working pose lives near
    roll ≈ ±π (≈64% of the xArm dataset). Consumed by
    ``XArmRealEnv.step`` with ``type="cartesian_abs_aa"`` which routes
    the pose through xArm SDK's ``set_position_aa`` instead of the
    Euler ``set_position``.
    """
    rel_actions = np.asarray(rel_actions, dtype=np.float64)
    if rel_actions.ndim != 2 or rel_actions.shape[1] != 7:
        raise ValueError(f"rel_actions must be (T, 7); got {rel_actions.shape}")
    T = rel_actions.shape[0]
    out = np.zeros((T, 7), dtype=np.float32)
    pos = np.asarray(init_xyz_m, dtype=np.float64).reshape(3).copy()
    rot = _R.from_quat(np.asarray(init_quat_xyzw, dtype=np.float64).reshape(4))
    for t in range(T):
        pos = pos + rel_actions[t, :3]
        rot = _R.from_rotvec(rel_actions[t, 3:6]) * rot
        aa = rot.as_rotvec()
        out[t, :3] = pos.astype(np.float32)
        out[t, 3:6] = aa.astype(np.float32)
        out[t, 6] = np.float32(rel_actions[t, 6])
    return out


def _integrate_mixed_to_abs_aa(
    init_xyz_m: np.ndarray,
    chunk_phys: np.ndarray,
    sources,
) -> np.ndarray:
    """Build absolute axis-angle waypoints for a MIXED-action chunk.

    ``chunk_phys`` is the per-channel-denormalised ``(T, 7)`` chunk:
    ``"rel"`` channels are deltas, ``"abs"`` channels are absolute targets,
    per ``sources``. Translation ``rel`` channels are integrated cumulatively
    from ``init_xyz_m`` (and ``abs`` translation channels used directly);
    the rotation block is taken as an ABSOLUTE Euler RPY (the default mixed
    spec) and converted to axis-angle. A rel ROTATION block would need SO(3)
    composition and is rejected (use abs rotation, the intended mixed config).

    Output rows are ``[x, y, z, rx, ry, rz, grip]`` (cartesian_abs_aa).
    """
    sources = [str(s).lower() for s in sources]
    chunk = np.asarray(chunk_phys, dtype=np.float64)
    if chunk.ndim != 2 or chunk.shape[1] != 7:
        raise ValueError(f"chunk_phys must be (T, 7); got {chunk.shape}")
    if any(sources[d] == "rel" for d in (3, 4, 5)):
        raise NotImplementedError(
            "mixed xArm send: rel ROTATION channels need SO(3) composition; "
            "use abs rotation channels (the default mixed spec).")
    T = chunk.shape[0]
    out = np.zeros((T, 7), dtype=np.float32)
    pos = np.asarray(init_xyz_m, dtype=np.float64).reshape(3).copy()
    for t in range(T):
        for d in range(3):
            if sources[d] == "rel":
                pos[d] = pos[d] + chunk[t, d]
            else:
                pos[d] = chunk[t, d]
        aa = _R.from_euler("xyz", chunk[t, 3:6]).as_rotvec()
        out[t, :3] = pos.astype(np.float32)
        out[t, 3:6] = aa.astype(np.float32)
        out[t, 6] = np.float32(chunk[t, 6])
    return out


def _rpy_to_axis_angle(rpy_chunk: np.ndarray) -> np.ndarray:
    """Convert an ``(N, ≥6)`` array of ``[..., roll, pitch, yaw]`` rows
    to ``[..., rx, ry, rz]`` axis-angle in-place-friendly.

    Used by the policy path when ``--action_type actions`` so the
    training-data Euler stream can still be shipped via the AA SDK call
    (``set_position_aa``) at execution time, sidestepping the gimbal
    lock that bites when ``roll ≈ ±π``.
    """
    arr = np.asarray(rpy_chunk, dtype=np.float64).copy()
    rpy = arr[..., 3:6]
    rot = _R.from_euler("xyz", rpy)
    aa = rot.as_rotvec()
    arr[..., 3:6] = aa
    return arr.astype(np.float32)


def _build_actions_abs(
    positions: List[np.ndarray],
    actions_phys: List[np.ndarray],
    policy_action_type: str,
    init_ee_pos_quat: np.ndarray | None = None,
) -> np.ndarray:
    """Return ``actions.blosc2`` content as a (T, 7) EE absolute-pose trajectory.

    Two branches matching the policy's training ``action_type``:

      * ``"actions"`` (absolute): ``actions_phys[t]`` is already an absolute
        EE pose the policy commanded -- pass through unchanged.
      * ``"rel_actions"`` (delta): integrate the policy's commanded
        ``actions_phys`` series via :func:`_integrate_rel_to_abs`, starting
        from ``init_ee_pos_quat`` (the EE measured just before the first
        policy step). This is the same recipe as
        ``compare_actions_vs_rel.rel_actions_to_abs_actions``, so the saved
        ``actions.blosc2`` represents the *commanded* absolute trajectory
        -- matching the recorder's convention where ``actions`` is the
        commanded EE pose, not the measured one.

    Args:
        positions: per-step ``robot_obs`` (only used to derive a fallback
                   anchor for the rel→abs integration when no init pose
                   was supplied).
        actions_phys: per-step physical-unit actions the policy commanded.
                      Shape ``(T, 7)``.
        policy_action_type: ``"actions"`` or ``"rel_actions"``.
        init_ee_pos_quat: ``(7,)`` ``[xyz_m, qx, qy, qz, qw]`` of the EE
                          right before the loop started (typically
                          ``ee_prev`` at loop entry). Required for the
                          rel-actions branch.
    """
    if not actions_phys:
        return np.zeros((0, 7), dtype=np.float32)
    if policy_action_type == "actions":
        return np.stack(actions_phys, axis=0).astype(np.float32)

    # rel_actions branch -- integrate the policy commands.
    if init_ee_pos_quat is None:
        # Fallback: use the first measured robot_obs minus the first
        # commanded delta to recover an approximate "pre-step-0" pose.
        # This is only a degenerate fallback; collect_episode always
        # passes the real init pose.
        first_pos = np.asarray(positions[0], dtype=np.float64)
        init_xyz = first_pos[:3] - np.asarray(actions_phys[0][:3], dtype=np.float64)
        init_quat = first_pos[3:7]
    else:
        init_ee_pos_quat = np.asarray(init_ee_pos_quat, dtype=np.float64).reshape(-1)
        init_xyz = init_ee_pos_quat[:3]
        init_quat = init_ee_pos_quat[3:7]
    rel_arr = np.stack(actions_phys, axis=0).astype(np.float64)
    return _integrate_rel_to_abs(init_xyz, init_quat, rel_arr)


# ─────────────────────────────────────────── save_episode ──

def _save_rgb_mp4(frames, save_path, fps=30):
    """Write HWC uint8 RGB frames as H.264 MP4 via imageio + imageio-ffmpeg.

    Switched away from cv2.VideoWriter(mp4v=...) / matplotlib ArtistAnimation:
    cv2's bundled mp4v codec produces MPEG-4 Part 2 streams that QuickTime /
    Safari / Chrome / Firefox refuse to play (the file is created but
    unplayable), and matplotlib ArtistAnimation is much slower (per-frame
    Artist allocation). imageio-ffmpeg ships its own ffmpeg binary and writes
    H.264 with yuv420p, the universally-supported MP4 baseline.

    ``macro_block_size=1`` disables imageio-ffmpeg's silent auto-resize for
    frame dims not divisible by 16 — important for the 64/84/200-px camera
    outputs in this pipeline.
    """
    if len(frames) == 0:
        return
    frames = np.asarray(frames)
    if frames.dtype != np.uint8:
        frames = np.clip(frames, 0, 255).astype(np.uint8)
    writer = imageio.get_writer(
        save_path, fps=fps, codec="libx264", quality=8,
        macro_block_size=1, pixelformat="yuv420p",
    )
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


def save_episode(episode_path: str, ep: dict, calc_efe: bool,
                 camera_info_src: Path | None = None,
                 save_video: bool = True) -> None:
    os.makedirs(episode_path, exist_ok=True)
    static_dir = os.path.join(episode_path, "static")
    gripper_dir = os.path.join(episode_path, "gripper")
    obs_dir = os.path.join(episode_path, "obs")
    actions_dir = os.path.join(episode_path, "actions")
    for d in (static_dir, gripper_dir, obs_dir, actions_dir):
        os.makedirs(d, exist_ok=True)

    if save_video:
        _save_rgb_mp4(ep["static_rgbs"], os.path.join(static_dir, "rgb_static.mp4"))
    save_blosc2(os.path.join(static_dir, "rgb_static.blosc2"), ep["static_rgbs"])
    save_blosc2(os.path.join(static_dir, "mask.blosc2"), ep["static_masks"])

    if save_video:
        _save_rgb_mp4(ep["gripper_rgbs"], os.path.join(gripper_dir, "rgb_gripper.mp4"))
        # Gripper-cam affordance direction-field overlay track, written next
        # to rgb_gripper.mp4 exactly as aif_collection_uv does.
        if ep.get("gripper_dir_overlays") is not None:
            _save_rgb_mp4(ep["gripper_dir_overlays"],
                          os.path.join(gripper_dir, "rgb_gripper_aff_dirs.mp4"))
    save_blosc2(os.path.join(gripper_dir, "rgb_gripper.blosc2"), ep["gripper_rgbs"])
    save_blosc2(os.path.join(gripper_dir, "depth_gripper.blosc2"), ep["gripper_depths"])
    save_blosc2(os.path.join(gripper_dir, "mask.blosc2"), ep["gripper_masks"])

    save_blosc2(os.path.join(obs_dir, "robot_obs.blosc2"), ep["positions"])
    # Always write both so the dataset is symmetric for downstream training.
    save_blosc2(os.path.join(actions_dir, "actions.blosc2"), ep["actions_abs"])
    save_blosc2(os.path.join(actions_dir, "rel_actions.blosc2"), ep["actions_rel"])

    # Created unconditionally, as aif_collection_uv does, so an episode
    # directory has the same shape with and without --calc_efe.
    efe_dir = os.path.join(episode_path, "efe")
    os.makedirs(efe_dir, exist_ok=True)
    if calc_efe:
        save_blosc2(os.path.join(efe_dir, "candidate_policies.blosc2"),
                    ep["candidate_policies"])
        if "extrinsics" in ep:
            save_blosc2(os.path.join(efe_dir, "extrinsics.blosc2"), ep["extrinsics"])
        if "epistemics" in ep:
            save_blosc2(os.path.join(efe_dir, "epistemics.blosc2"), ep["epistemics"])

    # Copy camera_info.npz so downstream affordance label / training tools
    # find the matching extrinsics next to the data.
    if camera_info_src is not None and camera_info_src.is_file():
        import shutil
        shutil.copyfile(camera_info_src,
                        os.path.join(os.path.dirname(episode_path), "camera_info.npz"))


# ─────────────────────────────────────────────── CLI / main ──

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # Policy / world / aff
    p.add_argument("--policy_cfg", type=str, default="act-zprior")
    add_ebm_sampler_args(p)
    p.add_argument("--world_cfg", type=str, default="rssm_s-dec_no-sphery")
    p.add_argument("--seed", type=int, default=4)
    p.add_argument("--dataset", type=str, default="real_world/xArm",
                   help="Dataset env key — used to locate config/ + ckpts.")
    p.add_argument("--ckpt_path", type=str, default=None)
    p.add_argument("--world_ckpt_path", type=str, default=None)
    p.add_argument("--n_train", "--n-train", type=int, default=None,
                   help="If set, load policy/world checkpoints trained on this "
                        "many episodes, i.e. from the '{cfg}+n{n_train}' "
                        "directories saved by train_policy.py / train_world.py. "
                        "Ignored for a model whose explicit --ckpt_path / "
                        "--world_ckpt_path is given.")
    p.add_argument("--num_episodes", type=int, default=5)
    p.add_argument("--episode_length", type=int, default=100)
    p.add_argument("--num_candidate_policies", type=int, default=8)
    p.add_argument("--calc_efe_every", type=int, default=8)
    p.add_argument("--calc_efe", dest="calc_efe", action="store_true", default=False)
    p.add_argument("--no_calc_efe", dest="calc_efe", action="store_false")
    p.add_argument("--pref_var", type=float, default=0.01)
    # EFE extrinsic(target_pos)の時間方向の縮約。
    #   "mean": 予測全ステップの対象位置との差の平均(既定・従来挙動)。
    #   "min" : 予測軌道の最接近(位置ベクトル距離が時間方向で最小)の1ステップのみ。
    p.add_argument("--extrinsic_pos_reduce", type=str, default="mean",
                   choices=["mean", "min"],
                   help="target_pos extrinsic の時間縮約: mean=全ステップ平均(既定)"
                        ", min=最接近ステップのみ。")
    # gripper-cam affordance の信頼度(robustness = 前景確率平均)がこの値未満の
    # クラスタを候補から棄却する。0.0(既定)で棄却なし。棄却されると gripper-cam
    # による curr_detected_obj の更新は行われず、その step の EFE extrinsic 標的には
    # 直前の(あるいは static-cam の)候補が使われる。
    p.add_argument("--gripper_aff_min_robustness", type=float, default=0.0,
                   help="gripper affordance のクラスタ信頼度(robustness)しきい値。"
                        "これ未満は候補として採用しない（0.0で棄却なし）。")
    p.add_argument("--use_temporal_ensemble", action="store_true",
                   help="Enable ACT-style Temporal Ensemble across "
                        "overlapping action chunks. When set:\n"
                        "  • --no_calc_efe: every step calls "
                        "    policy.inference(N=1) to get a fresh chunk, "
                        "    and the current step's action is the "
                        "    exponentially-weighted mean over all "
                        "    overlapping chunks' predictions.\n"
                        "  • --calc_efe: at each calc_efe_every "
                        "    boundary the TE buffer is reset, EFE picks "
                        "    one candidate chunk, and that goes into the "
                        "    fresh buffer. Within the window every step "
                        "    calls policy.inference(N=1) and the action "
                        "    is the TE mean — EFE does NOT re-run until "
                        "    the next boundary.")
    p.add_argument("--temporal_ensemble_decay", type=float, default=0.01,
                   help="TE weight decay m: w_i = exp(-m * chunk_age). "
                        "Higher = trust recent chunks more; lower = more "
                        "smoothing. ACT paper default 0.01.")
    # affordance が予測した orientation を利用するか（移動・EFE extrinsic 共通）。
    # 既定は有効。--no-use_orientation で無効化（モデルが予測しても使わない）。
    p.add_argument("--use_orientation", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="affordance 予測 orientation を移動/EFE extrinsic に利用する"
                        "（--no-use_orientation で無効）。")
    p.add_argument("--constraint", action="store_true",
                   help="Zero roll/pitch in policy actions.")
    p.add_argument("--max_pred_orn_dev_deg", type=float, default=60.0,
                   help="Reject affordance-predicted wrist orientations that "
                        "deviate more than this many degrees from the home "
                        "(down-pointing) orientation; falls back to the home "
                        "pose. Guards against a mis-trained orientation head "
                        "commanding an impossible pose.")
    p.add_argument("--task", type=str, default="slide",
                   help="env.task value (avoid 'pickup' — that branch needs a sim).")
    p.add_argument("--termination_radius", type=float, default=0.15)
    p.add_argument("--aff_target_z_offset", type=float, default=0.0,
                   help="Vertical offset (m, base frame) added to the "
                        "affordance-detected target xyz when the arm moves "
                        "to it at episode start. Positive = hover above the "
                        "detection (safer for plate/handle approaches), "
                        "negative = dip below. The stored target used by "
                        "downstream gripper-cam refinement / EFE keeps the "
                        "raw detection — only the initial move is offset.")
    p.add_argument("--max_step_xyz_m", type=float, default=0.05,
                   help="Per-step EE position delta cap (m) measured "
                        "against the current EE. Prevents IK branch flips "
                        "when the policy commands a target far from the "
                        "current pose. ≤0 disables. Default 0.05 m.")
    p.add_argument("--max_step_rot_rad", type=float, default=0.3,
                   help="Per-step EE rotation delta cap (rad, SO(3) "
                        "shortest-arc) measured against the current EE "
                        "orientation. Prevents wrist branch flips. "
                        "≤0 disables. Default 0.3 rad (≈17°).")
    p.add_argument("--max_step_joint_rad", type=float, default=0.35,
                   help="Per-step **joint-space** delta cap (rad). For "
                        "each commanded EE target the controller's IK is "
                        "queried in pre-flight; if any of the 7 joints "
                        "would move by more than this from the current "
                        "measurement, the command is REPLACED with a "
                        "hold-current-EE command. Catches the IK "
                        "branch-flip cases that ``--max_step_*`` "
                        "(EE-space) can't see — the SDK has no IK seed "
                        "parameter so even a small EE delta can land in "
                        "a different joint configuration near singular "
                        "regions. ≤0 disables. Default 0.35 rad (≈20°).")
    p.add_argument("--use_local_ik", action="store_true",
                   help="Bypass the SDK's branch-free IK entirely. For "
                        "each ``cartesian_abs_aa`` target, run a "
                        "**seeded** damped-least-squares IK locally "
                        "(numpy on hard-coded xArm7 DH parameters) "
                        "using the current measured joints as seed. "
                        "Send the result as a joint command via "
                        "``set_servo_angle``. Structurally eliminates "
                        "branch flips at the cost of a small (~1 mm / "
                        "0.5°) calibration offset between local DH FK "
                        "and the controller's internal kinematics — "
                        "the script verifies the offset is within "
                        "5 mm / 2° on startup and refuses to enable "
                        "this mode if not.")
    p.add_argument("--use_official_ik", action="store_true",
                   help="Bypass the SDK IK using UFactory's **official** "
                        "xArm7 kinematics user library "
                        "(xarm_kinematics_user_lib_*) seeded with the "
                        "current measured joints. This is the "
                        "recommended IK mode: it has the controller's "
                        "calibration baked in (no DH mismatch like "
                        "--use_local_ik) AND exposes the ``q_pre`` "
                        "seed knob that ``XArmAPI.get_inverse_"
                        "kinematics`` lacks. Empirically caps per-step "
                        "joint deltas at ~3° on a smooth trajectory "
                        "where the zero-seeded path produced ~350° "
                        "flips. Requires the library directory at "
                        "<repo>/xarm_kinematics_user_lib_20251009_x86_"
                        "64_fPIC_gcc9/. The .so is auto-built from the "
                        "shipped .a on first use.")
    # Action transport
    p.add_argument("--action_type", choices=["actions", "rel_actions"],
                   default="rel_actions",
                   help="Which transport to use when sending policy actions "
                        "to the robot (also names the *primary* action file). "
                        "Both actions.blosc2 and rel_actions.blosc2 are written "
                        "regardless.")
    # Hardware
    p.add_argument("--follower_ip", type=str, default="192.168.1.240")
    p.add_argument("--env_camera_serial", type=str, default=DEFAULT_ENV_CAMERA_SERIAL)
    p.add_argument("--wrist_camera_serial", type=str, default=DEFAULT_WRIST_CAMERA_SERIAL)
    p.add_argument("--extrinsics", type=str, required=False, default=None,
                   help="Path to camera_info.npz (intrinsics + extrinsics). "
                        "Required for accurate target search; falls back to "
                        "identity if missing.")
    p.add_argument("--img_width", type=int, default=640)
    p.add_argument("--img_height", type=int, default=480)
    p.add_argument("--max_delta", type=float, default=0.05,
                   help="XArmFollower joint max_delta (rad/cycle).")
    p.add_argument("--control_hz", type=float, default=50.0)
    p.add_argument("--no_workspace_bounds", action="store_true",
                   help="Disable robopy's XArmWorkspaceBounds. Without this "
                        "flag, ``_send_cartesian`` silently clips any target "
                        "outside the cell-specific [min_x..max_x]/[min_y..]/"
                        "[min_z..max_z] box -- useful for debugging an "
                        "extrinsic miscalibration that pushes the affordance "
                        "world point off the box.")
    p.add_argument("--start_joints_deg", type=float, nargs=7, default=None,
                   help="Home pose used between episodes, as 7 joint angles "
                        "in degrees. If omitted (default), the robot's pose "
                        "at script-launch is captured and used as the home "
                        "for the whole session -- just position the arm "
                        "where you want home before launching.")
    p.add_argument("--fps", type=float, default=None,
                   help="Policy-step rate [Hz]. If omitted, defaults to "
                        "``30 / policy_cfg.datamodule.frequency`` so the "
                        "data-collection sample rate stays at 30 Hz regardless "
                        "of the policy's training frequency. Pass an explicit "
                        "value to override.")
    p.add_argument("--static_aff_size", type=int, nargs=2,
                   default=DEFAULT_STATIC_AFF_SIZE, metavar=("H", "W"),
                   help="Resize (H, W) fed to the static affordance net. "
                        "Must match training-time img_resize.static "
                        "(see config/viz_affordances.yaml). Default "
                        f"{DEFAULT_STATIC_AFF_SIZE} matches the labeler's "
                        "preserve_aspect_ratio output_size halved.")
    p.add_argument("--gripper_aff_size", type=int, nargs=2,
                   default=DEFAULT_GRIPPER_AFF_SIZE, metavar=("H", "W"),
                   help="Resize (H, W) fed to the gripper affordance net. "
                        f"Default {DEFAULT_GRIPPER_AFF_SIZE} matches "
                        "training-time img_resize.gripper.")
    # Output
    p.add_argument("--out_root", type=str, default="datasets/real_world/xArm")
    p.add_argument("--additional_data_str", type=str, default="xarm",
                   help="Suffix appended to the uv-style intermediate "
                        "directory name (mirrors aif_collection_uv.py).")
    p.add_argument("--run_name", type=str, default=None,
                   help="Optional explicit subdirectory name under out_root. "
                        "If omitted, the uv-style "
                        "`{use_human_data_str}{additional_data_str}` name is "
                        "used so the layout matches aif_collection_uv.py.")
    p.add_argument("--overwrite", action="store_true")
    # Misc
    p.add_argument("--save_video", dest="save_video", action="store_true",
                   default=True,
                   help="Save per-episode MP4 videos (rgb_static.mp4 / "
                        "rgb_gripper.mp4 / rgb_gripper_aff_dirs.mp4). "
                        "Default: on.")
    p.add_argument("--no_save_video", dest="save_video", action="store_false",
                   help="Skip MP4 video saving. blosc2 RGB arrays are still saved.")
    p.add_argument("--save_images", action="store_true")
    p.add_argument("--viz", action="store_true")
    p.add_argument("--profile", action="store_true",
                   help="Log a per-stage wall-clock breakdown of each "
                        "recorded frame (camera / IK / affordance / policy / "
                        "EFE / world / viz / sleep) so a run that falls below "
                        "--fps can be diagnosed on the real robot. Times are "
                        "EXCLUSIVE (a nested scope is subtracted from its "
                        "parent), so the rows plus '(unaccounted)' sum to the "
                        "measured step time. The 'calls/step' column matters "
                        "as much as the times: cam_* reading ~2.00 means the "
                        "frame is fetched twice per step (env.step's obs is "
                        "discarded, then _compute_step_obs re-reads it), and "
                        "since robopy's async_read blocks for a FRESH frame "
                        "that doubles the camera wait. Adds a JAX host sync "
                        "around the policy/EFE/world scopes so their numbers "
                        "are real compute, not async dispatch — which makes "
                        "the run slightly slower than an unprofiled one.")
    p.add_argument("--profile_every", type=int, default=30,
                   help="--profile: emit a rolling window report every N "
                        "recorded frames. A per-episode summary is always "
                        "printed at episode end.")
    p.add_argument("--skip_static_aff", action="store_true",
                   help="Skip the per-step static-cam affordance forward "
                        "in `_compute_step_obs`. Saves a JAX inference per "
                        "step (~5-30ms) at the cost of `static/mask.blosc2` "
                        "being all zeros for every episode. target_search "
                        "(episode-start static-cam affordance scan) is "
                        "unaffected because it uses its own net instance.")
    # ── Replay mode ─────────────────────────────────────────────────────
    # Replay a saved episode: drive the arm with the recorded
    # actions/rel_actions instead of running the policy. Initial pose is
    # set from the first row of the source obs/robot_obs.blosc2.
    p.add_argument("--replay_path", type=Path, default=None,
                   help="Path to a saved episode directory (e.g. "
                        "datasets/real_world/xArm/episode_01). When set, "
                        "policy/world/target_search builds are skipped and "
                        "the loop replays the saved actions. The arm is "
                        "driven to the first robot_obs[:7] pose before the "
                        "loop. Each output episode replays the same source "
                        "(use --num_episodes 1 unless you want repeats).")
    p.add_argument("--replay_action_type", choices=["actions", "rel_actions"],
                   default=None,
                   help="Which saved file to read for replay. Defaults to "
                        "--action_type. Determines both the source "
                        "actions/<...>.blosc2 file and the env.step "
                        "transport (cartesian_abs / cartesian_rel).")
    # ── SDK error gate ─────────────────────────────────────────────────
    # While the xArm controller has a latched error/warn (e.g. joint-limit
    # rejection, self-collision, modbus busy), block the action loop so
    # the follower's 50 Hz control thread doesn't spam the SDK with
    # commands that will only be rejected. Only ``step()`` gates on this;
    # ``reset()`` / ``move_to_target()`` already recover via
    # ``_drive_to_home_joints`` which clears the fault as part of homing.
    p.add_argument("--pause_on_sdk_error",
                   dest="pause_on_sdk_error",
                   action="store_true", default=True,
                   help="Block env.step() while the xArm controller has a "
                        "latched error/warn code, and stop robopy's 50 Hz "
                        "control thread while paused so it stops flooding "
                        "the SDK with rejected commands. Default: on.")
    p.add_argument("--no_pause_on_sdk_error",
                   dest="pause_on_sdk_error", action="store_false",
                   help="Disable the SDK-error gate on env.step(). Not "
                        "recommended — the policy loop will keep issuing "
                        "commands into a latched-fault controller.")
    p.add_argument("--sdk_error_poll_hz", type=float, default=4.0,
                   help="How often the SDK-error gate polls "
                        "``has_err_warn`` while paused. Default 4 Hz.")
    p.add_argument("--sdk_error_auto_recover", action="store_true",
                   default=False,
                   help="If set, the SDK-error gate automatically calls "
                        "clean_error / clean_warn / motion_enable(True) / "
                        "set_state(0) after "
                        "--sdk_error_auto_recover_after_s seconds. "
                        "Default: off (wait for a human on the pendant).")
    p.add_argument("--sdk_error_auto_recover_after_s", type=float,
                   default=3.0,
                   help="Seconds to wait before the auto-recover fires "
                        "when --sdk_error_auto_recover is on. Default 3s.")
    p.add_argument("--sdk_error_max_wait_s", type=float, default=0.0,
                   help="Hard timeout for the SDK-error gate. 0 = wait "
                        "forever (default). If >0 and the fault is still "
                        "latched after this many seconds, the script "
                        "aborts with SystemExit.")
    p.add_argument("--dry_run", action="store_true",
                   help="Build everything but don't connect to hardware.")
    p.add_argument("--yes", action="store_true",
                   help="Skip the pre-flight confirmation prompt.")
    p.add_argument("-v", "--verbose", action="count", default=0)
    # ── Affordance debug modes ─────────────────────────────────────────
    # These bypass the robot entirely so you can verify the affordance
    # pipeline (resize / transforms / model / deprojection) is fed the
    # right pixels without risking the arm.
    p.add_argument("--debug_affordance", choices=["saved", "live"], default=None,
                   help="Affordance-only debugging mode. Does NOT move the "
                        "robot. 'saved' = run on images on disk. "
                        "'live' = run on the connected RealSense cameras "
                        "(opens cameras only; arm stays idle).")
    p.add_argument("--debug_data_path", type=Path, default=None,
                   help="--debug_affordance saved: path to either a single "
                        "image (.png/.jpg/.npz) or a directory containing "
                        "many. Also accepts a directory holding "
                        "rgb_static.blosc2 / rgb_gripper.blosc2 (e.g. an "
                        "episode_NN directory).")
    p.add_argument("--debug_n_frames", type=int, default=10,
                   help="--debug_affordance live: how many frames to "
                        "capture + process (one per second).")
    p.add_argument("--debug_output_dir", type=Path, default=None,
                   help="Where to drop the debug visualisations. Default: "
                        "./aff_debug_<mode>_<timestamp>/")
    p.add_argument("--debug_cam", choices=["static", "gripper", "both"],
                   default="both",
                   help="Which affordance net to run.")
    # ── Affordance variant selection (which trained checkpoint to load) ──
    # These pick the variant-tagged checkpoint dir
    # (.../<cam>/<color>-<orient>/seed:<seed>/) that train_affordance.py wrote.
    # NOTE: --predict_orientation selects WHICH model to load; --use_orientation
    # (above) controls whether a loaded orientation is USED at runtime.
    p.add_argument("--predict_orientation", choices=["true", "false"],
                   default="true",
                   help="Load the affordance variant trained WITH (true) or "
                        "WITHOUT (false) the orientation head, for both cams. "
                        "Per-cam override: --gripper_predict_orientation / "
                        "--static_predict_orientation. Default: true.")
    p.add_argument("--gripper_predict_orientation", choices=["true", "false"],
                   default=None,
                   help="Override --predict_orientation for the gripper cam "
                        "only (e.g. false to load a no-orientation gripper "
                        "model while static keeps orientation).")
    p.add_argument("--static_predict_orientation", choices=["true", "false"],
                   default=None,
                   help="Override --predict_orientation for the static cam only.")
    p.add_argument("--aff_color", choices=["grayscale", "rgb"],
                   default="grayscale",
                   help="Input color the affordance models were trained on: "
                        "'grayscale' (aff_transforms) or 'rgb' "
                        "(aff_transforms_rgb). Selects the color half of the "
                        "checkpoint variant. Default: grayscale.")
    return p.parse_args()


def _confirm_or_exit(prompt: str) -> None:
    try:
        ans = input(prompt).strip().lower()
    except (EOFError, KeyboardInterrupt):
        raise SystemExit("Aborted.")
    if ans not in ("y", "yes"):
        raise SystemExit("Aborted.")


def _find_start_episode(base_dir: Path, episode_prefix: str, overwrite: bool) -> int:
    if not base_dir.is_dir() or overwrite:
        return 0
    nums = []
    for d in base_dir.glob(f"{episode_prefix}*"):
        try:
            nums.append(int(d.name[len(episode_prefix):]))
        except ValueError:
            continue
    return (max(nums) + 1) if nums else 0


def _build_use_human_data_str(args, world_cfg) -> str:
    """Mirror ``aif_collection_uv.py``'s intermediate-directory name so the
    xArm dataset layout matches the simulation one exactly."""
    s = ""
    if not args.calc_efe and args.num_candidate_policies != 1:
        s += f"mean{args.num_candidate_policies}_"
    if world_cfg is not None:
        for idx in world_cfg.datamodule.train_episodes:
            s += f"{str(int(idx))}"
    if args.calc_efe:
        s += "_sigma{:.0e}".format(args.pref_var)
    else:
        s += "_dif"
    return s


# ─────────────────────────────────────────── affordance debug modes ──

def _save_affordance_debug_viz(
    rgb_hwc: np.ndarray,
    aff_mask: np.ndarray,
    directions: np.ndarray,
    centers,
    probs: np.ndarray,
    out_prefix: Path,
    cam_label: str,
) -> None:
    """Dump one frame's affordance viz, using the SAME renderer and the same
    file-name suffixes as the episode-start / per-step viz — i.e. as
    ``aif_collection_uv``.

    Produces under ``out_prefix.parent``, at the original frame resolution:

      * ``{prefix}_orig.png``   — raw RGB
      * ``{prefix}_masks.png``  — predicted foreground mask
      * ``{prefix}_aff.png``    — RGB with the affordance-mask overlay
      * ``{prefix}_dirs.png``   — RGB with the center-direction vectors
                                   overlay (detected centers marked)

    This used to be a separate inline implementation writing a different set
    (``_mask`` / ``_aff_overlay`` / ``_flow_overlay`` / ``_centers``) because
    ``get_aff_imgs`` transposes non-square frames. :func:`_aff_viz_images`
    now handles that (render square, undo the stretch), so the debug modes
    share one renderer with the rest of the script instead of drifting from
    it.
    """
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    try:
        imgs = _aff_viz_images(rgb_hwc, aff_mask, directions, centers, cam_label)
    except Exception as exc:
        logger.warning(f"[viz] overlay render failed ({cam_label}): {exc}")
        return
    for name, img in imgs.items():
        # ``name`` is e.g. 'static_dirs' → keep only the '_dirs' suffix so the
        # per-frame prefix (which already carries the cam label) stays readable.
        suffix = name[len(cam_label) + 1:] if name.startswith(cam_label + "_") else name
        cv2.imwrite(
            str(out_prefix.with_name(f"{out_prefix.name}_{suffix}.png")), img,
        )


def _collect_debug_images(
    path: Path, prefer_cam: str = "static", max_frames: int = 20,
    stride: int = 1,
) -> list[tuple[str, np.ndarray]]:
    """Resolve ``--debug_data_path`` into a list of ``(name, HWC uint8 RGB)``.

    Accepts:
      * a single image file (``.png`` / ``.jpg`` / ``.npz``)
      * a directory of image files
      * a directory containing ``rgb_{static,gripper}.blosc2`` (loads each
        frame as a sample; useful for replaying an existing episode)

    For blosc2 episodes, frames are sub-sampled to at most ``max_frames``
    by taking an evenly-spaced stride. Processing all 4000+ frames in an
    episode would take many minutes and is rarely what you want for a
    pipeline sanity check.
    """
    out: list[tuple[str, np.ndarray]] = []
    if not path.exists():
        raise FileNotFoundError(f"--debug_data_path not found: {path}")

    if path.is_file():
        out.append((path.stem, _load_one_image(path)))
        return out

    # Directory: try blosc2 first (episode-style), fall back to image files.
    blosc_candidates = [
        path / "static" / f"rgb_{prefer_cam}.blosc2",
        path / "gripper" / f"rgb_{prefer_cam}.blosc2",
        path / f"rgb_{prefer_cam}.blosc2",
    ]
    for bp in blosc_candidates:
        if bp.is_file():
            arr = np.asarray(load_blosc2(str(bp)))
            if arr.ndim == 4:
                n = arr.shape[0]
                step = max(1, n // max_frames, stride)
                idxs = list(range(0, n, step))[:max_frames]
                logger.info(
                    f"[debug] sub-sampled {len(idxs)}/{n} frames from {bp.name} "
                    f"(stride={step})"
                )
                for i in idxs:
                    out.append((f"{bp.stem}_{i:04d}", _to_hwc_uint8(arr[i])))
                return out

    # Plain directory of images.
    for ext in ("*.png", "*.jpg", "*.jpeg", "*.npz"):
        for f in sorted(path.glob(ext))[:max_frames]:
            try:
                out.append((f.stem, _load_one_image(f)))
            except Exception as exc:
                logger.warning(f"[debug] skipping {f}: {exc}")
    if not out:
        raise FileNotFoundError(
            f"No images found under {path} (looked for blosc2, png, jpg, npz)."
        )
    return out


def _load_one_image(path: Path) -> np.ndarray:
    """Load a single RGB image (HWC uint8) from .png/.jpg/.npz."""
    if path.suffix in (".png", ".jpg", ".jpeg"):
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise ValueError(f"cv2.imread failed for {path}")
        return bgr[:, :, ::-1].copy()  # BGR → RGB
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=True) as d:
            # viz_affordances convention uses "frame"; create_dataset / recorder
            # uses "rgb_static" / "rgb_gripper".
            for key in ("frame", "rgb_static", "rgb_gripper"):
                if key in d.files:
                    return _to_hwc_uint8(np.asarray(d[key]))
            raise KeyError(
                f"{path}: no 'frame' / 'rgb_static' / 'rgb_gripper' key."
            )
    raise ValueError(f"unsupported extension: {path.suffix}")


def debug_affordance_saved(args, run_cfg) -> None:
    """Mode 1: run affordance on saved images. Robot is never touched."""
    if args.debug_data_path is None:
        raise SystemExit(
            "--debug_affordance saved requires --debug_data_path <file_or_dir>"
        )
    out_dir = args.debug_output_dir or Path(
        f"./aff_debug_saved_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[debug-saved] output → {out_dir}")

    logger.info("[debug-saved] loading affordance models …")
    gripper_net, static_net, aff_transforms, _ = build_affordance_models(
        run_cfg.affordance, run_cfg.img_size,
        aff_seed=args.seed, aff_dataset_name=args.dataset,
        static_resize=args.static_aff_size,
        gripper_resize=args.gripper_aff_size,
    )

    nets_to_run = []
    if args.debug_cam in ("static", "both") and static_net is not None:
        nets_to_run.append(("static", static_net))
    if args.debug_cam in ("gripper", "both") and gripper_net is not None:
        nets_to_run.append(("gripper", gripper_net))
    if not nets_to_run:
        raise SystemExit("[debug-saved] no affordance net available "
                         f"for --debug_cam {args.debug_cam}.")

    for cam_label, net in nets_to_run:
        prefer = cam_label  # try to load matching blosc2 first
        images = _collect_debug_images(
            args.debug_data_path, prefer_cam=prefer,
            max_frames=args.debug_n_frames,
        )
        logger.info(f"[debug-saved] {cam_label}: {len(images)} images")
        for name, rgb_hwc in images:
            if cam_label == "static":
                rgb_hwc = crop_static_rgb_for_net(net, rgb_hwc)
            try:
                centers, mask, dirs, probs, _ = transform_and_predict(
                    net, aff_transforms[cam_label], rgb_hwc,
                )
            except Exception as exc:
                logger.warning(f"[debug-saved] forward failed on {name}: {exc}")
                continue
            _save_affordance_debug_viz(
                rgb_hwc, mask, dirs, centers, probs,
                out_dir / f"{cam_label}_{name}", cam_label,
            )
    logger.info(f"[debug-saved] done → {out_dir}")


def debug_affordance_live(args, run_cfg) -> None:
    """Mode 2: run affordance on live RealSense streams. Arm is never moved."""
    from robopy.sensors.visual.realsense_camera import (
        RealsenseCamera, RealsenseCameraConfig,
    )

    out_dir = args.debug_output_dir or Path(
        f"./aff_debug_live_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[debug-live] output → {out_dir}")

    logger.info("[debug-live] loading affordance models …")
    gripper_net, static_net, aff_transforms, _ = build_affordance_models(
        run_cfg.affordance, run_cfg.img_size,
        aff_seed=args.seed, aff_dataset_name=args.dataset,
        static_resize=args.static_aff_size,
        gripper_resize=args.gripper_aff_size,
    )

    # Open cameras only -- never connect the follower.
    env_cfg = RealsenseCameraConfig(
        name="env", fps=30, serial_no=args.env_camera_serial,
        is_depth_camera=False,
    )
    env_cfg.width, env_cfg.height = args.img_width, args.img_height
    wrist_cfg = RealsenseCameraConfig(
        name="wrist", fps=30, serial_no=args.wrist_camera_serial,
        is_depth_camera=False,
    )
    wrist_cfg.width, wrist_cfg.height = args.img_width, args.img_height
    env_cam = RealsenseCamera(env_cfg)
    wrist_cam = RealsenseCamera(wrist_cfg)
    env_cam.index = resolve_realsense_index(args.env_camera_serial)
    wrist_cam.index = resolve_realsense_index(args.wrist_camera_serial)
    env_cam.connect()
    wrist_cam.connect()
    logger.info("[debug-live] cameras connected (env + wrist). "
                "Robot follower NOT connected.")

    _install_sigint_handler()
    nets = {}
    if args.debug_cam in ("static", "both") and static_net is not None:
        nets["static"] = (static_net, env_cam)
    if args.debug_cam in ("gripper", "both") and gripper_net is not None:
        nets["gripper"] = (gripper_net, wrist_cam)
    if not nets:
        raise SystemExit("[debug-live] no affordance net available "
                         f"for --debug_cam {args.debug_cam}.")

    try:
        for i in range(args.debug_n_frames):
            if _ABORT_EVENT.is_set():
                break
            for cam_label, (net, cam) in nets.items():
                try:
                    rgb_chw = cam.async_read(timeout_ms=500)
                except Exception as exc:
                    logger.warning(f"[debug-live] {cam_label} read: {exc}")
                    continue
                rgb_hwc = _to_hwc_uint8(rgb_chw)
                if cam_label == "static":
                    rgb_hwc = crop_static_rgb_for_net(net, rgb_hwc)
                try:
                    centers, mask, dirs, probs, _ = transform_and_predict(
                        net, aff_transforms[cam_label], rgb_hwc,
                    )
                except Exception as exc:
                    logger.warning(
                        f"[debug-live] forward failed ({cam_label}, "
                        f"frame {i}): {exc}"
                    )
                    continue
                _save_affordance_debug_viz(
                    rgb_hwc, mask, dirs, centers, probs,
                    out_dir / f"{cam_label}_frame_{i:04d}", cam_label,
                )
                logger.info(
                    f"[debug-live] frame {i:04d} {cam_label}: "
                    f"{len(centers)} centers detected"
                )
            time.sleep(1.0)
    finally:
        try:
            env_cam.disconnect()
        except Exception:
            pass
        try:
            wrist_cam.disconnect()
        except Exception:
            pass
    logger.info(f"[debug-live] done → {out_dir}")


# ──────────────────────────────────────────────────────────── main ──

def main(args: argparse.Namespace, run_cfg) -> None:
    if args.verbose >= 1:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Affordance-only debug modes short-circuit the rest of the pipeline.
    if args.debug_affordance == "saved":
        return debug_affordance_saved(args, run_cfg)
    if args.debug_affordance == "live":
        return debug_affordance_live(args, run_cfg)

    _install_sigint_handler()

    data_cfg = OmegaConf.load("datasets/real_world/config.yaml")

    # ── Replay mode short-circuits policy/world/target_search builds ──
    replay_mode = args.replay_path is not None
    replay_data: dict | None = None
    if replay_mode:
        replay_action_type = args.replay_action_type or args.action_type
        replay_data = _load_replay_data(Path(args.replay_path), replay_action_type)
        # EFE requires the world model + policy; force off in replay mode.
        if args.calc_efe:
            logger.info("[replay] disabling --calc_efe (no policy/world in "
                        "replay mode)")
            args.calc_efe = False

    # ── Load policy + (optional) world ──
    policy = policy_cfg = phase1_name = None
    if not replay_mode:
        logger.info(f"[policy] building '{args.policy_cfg}' (seed={args.seed})")
        policy, policy_cfg, phase1_name = build_policy(
            args.policy_cfg, data_cfg, seed=args.seed, dataset=args.dataset,
        )
        configure_ebm_sampler(
            policy, args.ebm_sampler, args.dfo_iters,
            args.dfo_noise_scale, args.dfo_noise_shrink,
        )
        maybe_load_ebm_phase1(policy, policy_cfg, args.dataset, phase1_name,
                              n_train=args.n_train)
        # delta_actions policies execute through the rel_actions pipeline: the
        # per-chunk conversion yields per-step rel rows (accumulated over one
        # frequency interval), so both the transport and the on-disk stream
        # must be the rel one regardless of what --action_type asked for.
        args.policy_is_delta = setup_delta_action_eval(policy_cfg, data_cfg)
        if args.policy_is_delta and args.action_type != "rel_actions":
            logger.info(f"[delta_actions] forcing --action_type rel_actions "
                        f"(was {args.action_type!r})")
            args.action_type = "rel_actions"
        # When --n_train is given, train_policy.py saved the checkpoint under a
        # "{policy_cfg}+n{n_train}" directory; mirror that here so the matching
        # parameters are loaded.
        policy_dir = (args.policy_cfg if args.n_train is None
                      else f"{args.policy_cfg}+n{args.n_train}")
        ckpt_path = args.ckpt_path or (
            f"trained_models/policy/{policy_cfg.datamodule.env}/{policy_dir}/"
            f"seed:{policy_cfg.seed}/model_best_loss.ckpt"
        )
        if args.dry_run and not os.path.exists(ckpt_path):
            logger.warning(f"[policy] dry_run: ckpt not found at {ckpt_path} -- "
                           "skipping load. Policy.inference will likely fail; "
                           "use --num_episodes 0 to exit before the collection loop.")
        else:
            logger.info(f"[policy] loading checkpoint {ckpt_path}")
            load_policy_checkpoint(policy, ckpt_path)

    # ── Resolve --fps. In normal mode the recording side is locked at
    # 30 fps so the robot command rate is ``30 / policy_cfg.datamodule
    # .frequency`` by default. In replay mode there is no policy_cfg,
    # so we default to 30 Hz (the recorder's native sample rate).
    # ``--fps`` explicitly overrides in either mode.
    if replay_mode:
        if args.fps is None:
            args.fps = 30.0
            logger.info("[fps] replay default: 30.0 Hz (override with --fps)")
        else:
            logger.info(f"[fps] user-supplied: {args.fps:.3f} Hz (replay)")
    else:
        policy_freq = max(1, int(getattr(policy_cfg.datamodule, "frequency", 1)))
        if args.fps is None:
            args.fps = 30.0 / float(policy_freq)
            logger.info(f"[fps] auto: 30 / policy_cfg.datamodule.frequency "
                        f"= 30 / {policy_freq} = {args.fps:.3f} Hz")
        else:
            logger.info(f"[fps] user-supplied: {args.fps:.3f} Hz "
                        f"(policy frequency = {policy_freq})")

    if args.profile:
        _PROF.enable(every=args.profile_every)
        logger.info(
            f"[profile] per-stage timing ON (window = {args.profile_every} "
            f"frames). Adds a JAX host sync per policy/EFE/world scope, so "
            f"the run is marginally slower than without --profile."
        )

    world, world_cfg = None, None
    if args.calc_efe:
        logger.info(f"[world] building '{args.world_cfg}' (seed={args.seed})")
        world, world_cfg = build_world(
            args.world_cfg, data_cfg, seed=args.seed, dataset=args.dataset,
        )
        world_dir = (args.world_cfg if args.n_train is None
                     else f"{args.world_cfg}+n{args.n_train}")
        w_ckpt = args.world_ckpt_path or (
            f"trained_models/world/{world_cfg.datamodule.env}/{world_dir}/"
            f"seed:{world_cfg.seed}/model_best_loss.ckpt"
        )
        if args.dry_run and not os.path.exists(w_ckpt):
            logger.warning(f"[world] dry_run: ckpt not found at {w_ckpt} -- skipping load.")
        else:
            logger.info(f"[world] loading checkpoint {w_ckpt}")
            load_world_checkpoint(world, w_ckpt)

    # ── Load camera_info (intrinsics + extrinsics) ──
    # SILENTLY falling back to identity extrinsics is dangerous: with
    # ``T_world_cam = I``, ``TargetSearch._compute_real_world`` returns the
    # raw *camera-frame* xyz as if it were base-frame world coords. The
    # depth axis (z_cam, typically 0.5–1.5 m) then becomes a target ``z``
    # ~1 m above the base, and ``move_to_target`` tries to send the arm
    # there — which is the "あらぬ方向" failure mode. Require an explicit
    # extrinsics file (or auto-detect the conventional one), and refuse
    # to proceed without it on real hardware.
    candidate_paths: List[Path] = []
    if args.extrinsics:
        candidate_paths.append(Path(args.extrinsics))
    candidate_paths.extend([
        Path("datasets") / args.dataset / "camera_info.npz",
        Path("datasets/playdata") / args.dataset / "camera_info.npz",
        Path(args.out_root) / "camera_info.npz",
    ])
    camera_info_path: Path | None = None
    for p in candidate_paths:
        if p.is_file():
            camera_info_path = p
            break
    if camera_info_path is None:
        if args.dry_run:
            logger.warning(
                "[extrinsics] no camera_info.npz found — using identity "
                "fallback (DRY RUN only). Searched: %s",
                ", ".join(str(p) for p in candidate_paths),
            )
            camera_info = {
                "static_intrinsics": None,
                "gripper_intrinsics": None,
                "T_world_static": np.eye(4, dtype=np.float32),
                "T_tcp_gripper": np.eye(4, dtype=np.float32),
            }
        else:
            raise SystemExit(
                "[extrinsics] No camera_info.npz found. Searched:\n  "
                + "\n  ".join(str(p) for p in candidate_paths)
                + "\n\nWithout calibrated extrinsics the affordance target "
                "is computed in the camera frame, not the robot base frame, "
                "and the arm will move to nonsense coordinates. Pass "
                "--extrinsics <path/to/camera_info.npz> (produced by "
                "scripts/calibrate_handeye_xarm{,_dual}.py)."
            )
    else:
        logger.info(f"[extrinsics] using {camera_info_path}")
        camera_info = _load_camera_info(camera_info_path)
        args.extrinsics = str(camera_info_path)

    # ── Pre-flight confirmation ──
    if not args.yes and not args.dry_run:
        if replay_mode:
            src_type = replay_data["source_action_type"]
            send_type = replay_data["action_type"]
            conv_note = (
                "  [rel→abs integrated]"
                if src_type == "rel_actions" else ""
            )
            mode_line = (
                f"          Mode: REPLAY of {replay_data['source_path']} "
                f"(source={src_type}, "
                f"{replay_data['actions'].shape[0]} steps available).\n"
                f"          Send transport: {send_type}{conv_note}. "
            )
        else:
            mode_line = f"          Action transport: {args.action_type}. "
        _confirm_or_exit(
            f"\n[CONFIRM] About to drive real xArm @ {args.follower_ip} "
            f"and run {args.num_episodes} episodes × {args.episode_length} steps.\n"
            f"{mode_line}"
            f"E-stop ready? Proceed? (y/N): "
        )

    # ── Build env (connects to follower + cameras) ──
    logger.info("[env] connecting xArm + RealSense …")
    # ``start_joints_deg=None`` → XArmRealEnv captures the launch-time pose.
    start_joints = (
        np.array(args.start_joints_deg, dtype=np.float32)
        if args.start_joints_deg is not None else None
    )
    env = XArmRealEnv(
        follower_ip=args.follower_ip,
        env_cam_serial=args.env_camera_serial,
        wrist_cam_serial=args.wrist_camera_serial,
        camera_info=camera_info,
        start_joints_deg=start_joints,
        max_delta=args.max_delta,
        control_hz=args.control_hz,
        img_width=args.img_width,
        img_height=args.img_height,
        task=args.task,
        save_images=args.save_images,
        viz=args.viz,
        termination_radius=args.termination_radius,
        gripper_aff_min_robustness=args.gripper_aff_min_robustness,
        dry_run=args.dry_run,
        use_workspace_bounds=not args.no_workspace_bounds,
        pause_on_sdk_error=args.pause_on_sdk_error,
        sdk_error_poll_hz=args.sdk_error_poll_hz,
        sdk_error_auto_recover=args.sdk_error_auto_recover,
        sdk_error_auto_recover_after_s=args.sdk_error_auto_recover_after_s,
        sdk_error_max_wait_s=args.sdk_error_max_wait_s,
    )
    env.max_pred_orn_dev_rad = float(np.deg2rad(args.max_pred_orn_dev_deg))

    # ── Local-IK gate: validate DH params against the controller FK ──
    # Refuse to enable ``--use_local_ik`` if the DH model disagrees with
    # the controller by more than the operational tolerance. Without
    # this gate a mis-spec'd DH would silently mis-target the robot
    # by centimeters — strictly worse than the branch-flip we're
    # trying to fix.
    if args.use_local_ik:
        if args.dry_run:
            logger.info("[local-ik] dry_run set; skipping DH validation")
        else:
            logger.info("[local-ik] validating DH FK against controller …")
            ok, pos_err_mm, rot_err_deg = _validate_dh_against_sdk(env._follower)
            if not ok:
                raise SystemExit(
                    f"[local-ik] DH calibration mismatch with controller: "
                    f"max pos err {pos_err_mm:.2f} mm, max rot err "
                    f"{rot_err_deg:.3f}° (envelope: 5 mm / 2°). Refuse "
                    f"to enable --use_local_ik. Verify the DH parameters "
                    f"in _XARM7_DH_MDH match your robot's calibration."
                )
            logger.info(
                f"[local-ik] DH FK validated (max pos err {pos_err_mm:.2f}mm "
                f"max rot err {rot_err_deg:.3f}°). Local seeded DLS IK "
                "enabled — SDK get_inverse_kinematics will be bypassed."
            )

    # ── Official xArm7 kinematics library init (--use_official_ik) ──
    # Loaded once here so the per-step path doesn't pay the dlopen /
    # build cost. Also lets us mirror any TCP offset the controller
    # has so the lib's FK/IK frame matches ``XArmAPI.get_*_kinematics``.
    official_kin = None
    if args.use_official_ik:
        from vapo.utils.xarm_official_kinematics import (
            XArm7OfficialKin, is_available,
        )
        if not is_available():
            raise SystemExit(
                "[official-ik] xarm_kinematics_user_lib_*/ not found "
                "in the repo root. --use_official_ik requires the "
                "UFactory library directory. Skip this flag or check "
                "the repo layout."
            )
        # Mirror any non-zero TCP / world offset the controller has so
        # the lib's frame matches SDK FK. SDK 1.17.3 has no
        # ``get_tcp_offset()`` function — the cached values live on
        # the ``XArmAPI.tcp_offset`` / ``world_offset`` attributes.
        tcp_off = None
        world_off = None
        try:
            sdk = env._follower._robot
            if sdk is not None:
                if hasattr(sdk, "tcp_offset"):
                    off_arr = np.asarray(
                        sdk.tcp_offset, dtype=np.float64,
                    ).flatten()[:6]
                    if np.any(np.abs(off_arr) > 1e-6):
                        tcp_off = off_arr
                if hasattr(sdk, "world_offset"):
                    woff_arr = np.asarray(
                        sdk.world_offset, dtype=np.float64,
                    ).flatten()[:6]
                    if np.any(np.abs(woff_arr) > 1e-6):
                        world_off = woff_arr
        except Exception as exc:
            logger.warning(f"[official-ik] tcp/world offset probe failed: {exc}")
        official_kin = XArm7OfficialKin.get()
        cfg_kwargs = {}
        if tcp_off is not None:
            cfg_kwargs["tcp_offset"] = tcp_off
        if world_off is not None:
            cfg_kwargs["world_offset"] = world_off
        if cfg_kwargs:
            official_kin.configure(**cfg_kwargs)
            logger.info(
                f"[official-ik] mirroring controller offsets: "
                f"tcp={tcp_off.tolist() if tcp_off is not None else None} "
                f"world={world_off.tolist() if world_off is not None else None}"
            )
        # Auto-calibrate the lib's effective TCP frame to match the
        # controller's SDK FK output, by sampling FK at several mid-
        # workspace configs. Without this the lib's IK sees a target
        # in a frame that's offset by the gripper length (~172 mm)
        # and lands in completely different branches.
        try:
            from vapo.utils.xarm_official_kinematics import (
                calibrate_tcp_offset_from_sdk,
            )
            def _sdk_fk(q):
                code, pose = env._follower._robot.get_forward_kinematics(
                    np.asarray(q).tolist(),
                    input_is_radian=True, return_is_radian=True,
                )
                if code != 0 or pose is None:
                    raise RuntimeError(f"SDK FK returned code {code}")
                return np.asarray(pose, dtype=np.float64)
            auto_tcp = calibrate_tcp_offset_from_sdk(official_kin, _sdk_fk)
            if tcp_off is not None:
                auto_tcp = auto_tcp + tcp_off
            official_kin.configure(tcp_offset=auto_tcp)
        except Exception as exc:
            logger.warning(
                f"[official-ik] auto-calibration failed: {exc} — "
                "production IK may land on wrong joint branches"
            )
        logger.info(
            "[official-ik] UFactory seeded-IK ready. SDK IK + DH local "
            "IK will be bypassed."
        )

    # ── Affordance + target search ──
    logger.info("[affordance] loading networks")
    gripper_net, static_net, aff_transforms, _ = build_affordance_models(
        run_cfg.affordance, run_cfg.img_size,
        aff_seed=args.seed, aff_dataset_name=args.dataset,
        static_resize=args.static_aff_size,
        gripper_resize=args.gripper_aff_size,
    )
    if args.skip_static_aff:
        # Drop the per-step static net so `_compute_step_obs` short-circuits
        # to its zero-mask fallback. target_search keeps its own net so the
        # episode-start static-cam scan still runs.
        logger.info(
            "[affordance] --skip_static_aff set: per-step static forward "
            "disabled (static/mask.blosc2 will be all-zeros)."
        )
        static_net = None
    target_search = None
    if not replay_mode:
        logger.info("[affordance] building target_search (real_world mode)")
        target_search = build_target_search(
            env, run_cfg,
            aff_seed=args.seed, aff_dataset_name=args.dataset,
            static_resize=args.static_aff_size,
        )

    # ── Output layout ──
    # Mirror aif_collection_uv.py:
    #   {out_root}/{use_human_data_str}{additional_data_str}/{episode_prefix}{n}/
    # ``--run_name`` still overrides the intermediate dir name when given.
    use_human_data_str = _build_use_human_data_str(args, world_cfg)
    intermediate_dir = args.run_name or f"{use_human_data_str}{args.additional_data_str}"
    base_dir = Path(args.out_root) / intermediate_dir
    base_dir.mkdir(parents=True, exist_ok=True)
    if replay_mode:
        episode_prefix = "episode_replay"
    else:
        episode_prefix = "episode_aif" if args.calc_efe else "episode_dif"
    start_n = _find_start_episode(base_dir, episode_prefix, args.overwrite)
    # aif_collection_uv parity: the affordance viz lives INSIDE the episode
    # directory (<episode>/affordance_init/ for the static-cam detection,
    # <episode>/gripper/rgb_gripper_aff_dirs.mp4 for the per-step gripper
    # direction field), so a review tool only needs the episode path.
    logger.info(f"[paths] out_dir = {base_dir}  start_episode = {start_n}")

    try:
        for n in range(start_n, start_n + args.num_episodes):
            if _ABORT_EVENT.is_set():
                logger.warning("Abort requested before episode %d — stopping.", n)
                break
            jax_fix_seed(n)
            # Created up front so collect_episode can write the static-cam
            # affordance viz into <episode_path>/affordance_init/ before the
            # arm moves (mirrors aif_collection_uv.py's main loop).
            episode_path = base_dir / f"{episode_prefix}{n}"
            episode_path.mkdir(parents=True, exist_ok=True)
            if replay_mode:
                ep = collect_episode_replay(
                    n, env,
                    gripper_net, static_net, aff_transforms,
                    data_cfg, run_cfg, args,
                    replay_data=replay_data,
                    save_video=args.save_video,
                )
            else:
                ep = collect_episode(
                    n, env, target_search,
                    gripper_net, static_net, aff_transforms,
                    policy, policy_cfg, world, world_cfg,
                    data_cfg, run_cfg, args,
                    episode_path=episode_path,
                    save_video=args.save_video,
                    official_kin=official_kin,
                )
            save_episode(
                str(episode_path), ep, calc_efe=args.calc_efe,
                camera_info_src=Path(args.extrinsics) if args.extrinsics else None,
                save_video=args.save_video,
            )
            logger.info(f"[ep {n}] saved → {episode_path}")
    finally:
        env.disconnect()


if __name__ == "__main__":
    setup_console_logging()
    args = _parse_args()
    # Translate the affordance-variant CLI flags into hydra overrides so the
    # composed config points at the matching checkpoint dir + transforms preset.
    _aff_overrides = affordance_cli_overrides(
        predict_orientation=args.predict_orientation,
        static=args.static_predict_orientation,
        gripper=args.gripper_predict_orientation,
        color=args.aff_color,
    )
    with hydra.initialize(config_path="../config", version_base=None):
        run_cfg = hydra.compose(
            config_name="cfg_aif_datacollection", overrides=_aff_overrides)
    main(args, run_cfg)
