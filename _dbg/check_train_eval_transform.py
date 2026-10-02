from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

import stable_pretraining as spt
from torchvision.transforms import v2 as transforms
from torchvision import tv_tensors

from utils import get_img_preprocessor


IMG_SIZE = 224


def build_train_transform():
    """
    train_policy.py と同じ transform
    """
    return get_img_preprocessor(
        source="pixels",
        target="pixels",
        img_size=IMG_SIZE,
    )


def build_inference_transform():
    """
    eval_real_robot.py の img_transform() と同じ
    """
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(
                torch.float32,
                scale=True,
            ),
            transforms.Normalize(
                **spt.data.dataset_stats.ImageNet
            ),
            transforms.Resize(
                size=IMG_SIZE,
            ),
        ]
    )


def print_stats(name, x):
    print(f"\n{name}:")
    print("  shape:", tuple(x.shape))
    print("  dtype:", x.dtype)
    print("  min  :", x.min().item())
    print("  max  :", x.max().item())
    print("  mean :", x.mean().item())
    print("  std  :", x.std().item())


def main():

    rng = np.random.default_rng(42)

    # ============================================================
    # 元となる同一画像
    # HWC / uint8 / RGB
    # ============================================================

    image_hwc = rng.integers(
        low=0,
        high=256,
        size=(480, 640, 3),
        dtype=np.uint8,
    )

    print("raw HWC:")
    print("  shape:", image_hwc.shape)
    print("  dtype:", image_hwc.dtype)

    # ============================================================
    # 1. 学習時
    #
    # stable_pretraining.ToImage は numpy HWC を受け取り、
    # 内部で HWC -> CHW にする
    # ============================================================

    train_transform = build_train_transform()

    train_data = {
        "pixels": image_hwc.copy(),
    }

    train_out = train_transform(train_data)

    train_img = train_out["pixels"]

    if not torch.is_tensor(train_img):
        train_img = torch.as_tensor(train_img)

    train_img = train_img.float()

    # ============================================================
    # 2. 推論時
    #
    # eval_real_robot.py:
    #
    # camera:
    #   HWC
    #
    # _prepare_info():
    #   HWC -> CHW
    #
    # img_transform():
    #   CHW tensor/image を受け取る
    #
    # そのためここでは _prepare_info() の transpose を再現する
    # ============================================================

    inference_input_chw = np.transpose(
        image_hwc,
        (2, 0, 1),
    )

    inference_transform = build_inference_transform()

    inference_img = inference_transform(
        tv_tensors.Image(
            inference_input_chw.copy()
        )
    )

    inference_img = inference_img.float()

    # ============================================================
    # 3. 結果
    # ============================================================

    print_stats(
        "train transform",
        train_img,
    )

    print_stats(
        "inference transform",
        inference_img,
    )

    print("\n=== shape ===")
    print("train     :", tuple(train_img.shape))
    print("inference :", tuple(inference_img.shape))

    # ============================================================
    # 4. まず同shapeか
    # ============================================================

    if train_img.shape == inference_img.shape:

        diff = train_img - inference_img

        print("\n=== direct comparison ===")
        print(
            "max abs diff :",
            diff.abs().max().item(),
        )
        print(
            "mean abs diff:",
            diff.abs().mean().item(),
        )
        print(
            "RMSE         :",
            torch.sqrt(
                torch.mean(diff ** 2)
            ).item(),
        )
        print(
            "allclose     :",
            torch.allclose(
                train_img,
                inference_img,
                atol=1e-6,
                rtol=1e-5,
            ),
        )

    else:
        print("\nDIRECT SHAPE MISMATCH")

        # ========================================================
        # H/Wだけ逆なら、それも確認
        # ========================================================

        if (
            train_img.ndim == 3
            and inference_img.ndim == 3
            and train_img.shape[0] == inference_img.shape[0]
            and train_img.shape[1] == inference_img.shape[2]
            and train_img.shape[2] == inference_img.shape[1]
        ):

            print(
                "The shapes differ only by H/W transpose."
            )

            inference_transposed = (
                inference_img.transpose(1, 2)
            )

            diff = (
                train_img
                - inference_transposed
            )

            print("\n=== comparison after H/W transpose ===")
            print(
                "max abs diff :",
                diff.abs().max().item(),
            )
            print(
                "mean abs diff:",
                diff.abs().mean().item(),
            )
            print(
                "RMSE         :",
                torch.sqrt(
                    torch.mean(diff ** 2)
                ).item(),
            )
            print(
                "allclose     :",
                torch.allclose(
                    train_img,
                    inference_transposed,
                    atol=1e-6,
                    rtol=1e-5,
                ),
            )


if __name__ == "__main__":
    main()