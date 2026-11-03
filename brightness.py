"""Opt-in WM brightness augmentation; no model or inference dependencies."""
from dataclasses import dataclass
from pathlib import Path
import math
from numbers import Real

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset


def validate_factors(min_factor, max_factor):
    for name, value in (("min_factor", min_factor), ("max_factor", max_factor)):
        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            or value <= 0
        ):
            raise ValueError(f"random_brightness.{name} must be a positive finite number")
    if min_factor > max_factor:
        raise ValueError("random_brightness.min_factor must be <= max_factor")


@dataclass(frozen=True)
class BrightnessConfig:
    enabled: bool = False
    min_factor: float = 0.5
    max_factor: float = 1.5
    preview_steps: int = 9

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError("random_brightness.enabled must be a boolean")
        validate_factors(self.min_factor, self.max_factor)
        if (
            isinstance(self.preview_steps, bool)
            or not isinstance(self.preview_steps, int)
            or self.preview_steps < 2
        ):
            raise ValueError("random_brightness.preview_steps must be an integer >= 2")

    @classmethod
    def from_config(cls, cfg):
        values = (cfg.get("augmentation") or {}).get("random_brightness") or {}
        return cls(**values)


def apply_brightness(image, factor):
    """Shared arithmetic for augmentation and deterministic RGB previews."""
    return torch.clamp(image * factor, min=0.0, max=1.0)


class RandomBrightness:
    def __init__(self, min_factor=0.5, max_factor=1.5):
        validate_factors(min_factor, max_factor)
        self.min_factor = min_factor
        self.max_factor = max_factor

    def __call__(self, image):
        if image.ndim not in (3, 4) or image.shape[-3] != 3:
            raise ValueError("RandomBrightness expects (T,3,H,W) or (3,H,W)")
        shape = (image.shape[0], 1, 1, 1) if image.ndim == 4 else (1, 1, 1)
        factors = torch.empty(shape, dtype=image.dtype, device=image.device).uniform_(
            self.min_factor, self.max_factor
        )
        return apply_brightness(image, factors)


class TransformDataset(Dataset):
    """Apply a split-specific transform to raw HDF5 samples.

    HDF5Dataset normalizes raw action rows before packing frameskip actions.
    Undo/reapply that packing here to preserve the exact normalization order.
    Deliberately do not proxy __getitems__: that would bypass this transform.
    """
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = dict(self.dataset[idx])
        if "action" in sample:
            raw = self.dataset.dataset
            sample["action"] = sample["action"].reshape(raw.num_steps * raw.frameskip, -1)
        sample = self.transform(sample)
        if "action" in sample:
            sample["action"] = sample["action"].reshape(raw.num_steps, -1)
        return sample


def save_brightness_preview(dataset, run_dir, config):
    """Read the first valid raw HDF5 clip directly, without sampling any RNG."""
    if not config.enabled:
        return
    destination = Path(run_dir) / "brightness_preview"
    try:
        import h5py
        ep, start = dataset.clip_indices[0]
        row = int(dataset.offsets[ep] + start)
        with h5py.File(dataset.h5_path, "r") as handle:
            images = {
                key: torch.from_numpy(np.array(handle[key][row]))
                for key in ("pixels", "wrist_pixels")
            }
        destination.mkdir(parents=True, exist_ok=True)
        for key, filename in (
            ("pixels", "overhead_gradient.png"),
            ("wrist_pixels", "wrist_gradient.png"),
        ):
            image = images[key]
            if image.ndim == 3 and image.shape[-1] == 3:
                image = image.permute(2, 0, 1)
            if image.ndim != 3 or image.shape[0] != 3:
                raise ValueError(f"{key}: expected RGB image, got {tuple(image.shape)}")
            # Import here to share the exact legacy range conversion without a cycle.
            from utils import scale_to_unit_range
            image = scale_to_unit_range(image)
            height, width = image.shape[-2:]
            label_height = 24
            canvas = Image.new(
                "RGB", (width * config.preview_steps, height + label_height), "white"
            )
            draw = ImageDraw.Draw(canvas)
            factors = np.linspace(
                config.min_factor, config.max_factor, config.preview_steps
            )
            for i, factor in enumerate(factors):
                pixels = apply_brightness(image, float(factor)).permute(1, 2, 0)
                tile = Image.fromarray((pixels * 255).round().to(torch.uint8).numpy())
                canvas.paste(tile, (i * width, label_height))
                draw.text((i * width + 2, 4), f"{factor:.4g}x", fill="black")
            canvas.save(destination / filename)
    except Exception as exc:
        raise RuntimeError(f"Failed to save brightness preview to {destination}: {exc}") from exc
