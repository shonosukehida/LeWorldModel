from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from stable_pretraining import data as dt


IMG_SIZE = 224


def print_info(name, x):
    print(f"\n{name}")
    print("type :", type(x))
    print("shape:", x.shape)
    print("dtype:", x.dtype)


def main():

    rng = np.random.default_rng(42)

    image = rng.integers(
        0,
        256,
        size=(480, 640, 3),
        dtype=np.uint8,
    )

    data = {
        "pixels": image.copy(),
    }

    print_info(
        "0. raw numpy",
        data["pixels"],
    )

    # ============================================
    # ToImageだけ
    # ============================================

    imagenet_stats = dt.dataset_stats.ImageNet

    to_image = dt.transforms.ToImage(
        **imagenet_stats,
        source="pixels",
        target="pixels",
    )

    data = to_image(data)

    print_info(
        "1. after ToImage",
        data["pixels"],
    )

    # ============================================
    # Resizeだけ
    # ============================================

    resize = dt.transforms.Resize(
        IMG_SIZE,
        source="pixels",
        target="pixels",
    )

    data = resize(data)

    print_info(
        "2. after Resize",
        data["pixels"],
    )


if __name__ == "__main__":
    main()