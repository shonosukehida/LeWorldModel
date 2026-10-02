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
OBS_HORIZON = 2


def build_train_transform():
    """
    real_robot_add_prop / train_policy.py と同じ学習時 transform
    """
    return get_img_preprocessor(
        source="pixels",
        target="pixels",
        img_size=IMG_SIZE,
    )


def build_inference_transform():
    """
    real_robot_add_prop / eval_real_robot.py の img_transform() と同じ
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
    print(f"\n{name}")
    print("  type :", type(x))
    print("  shape:", tuple(x.shape))
    print("  dtype:", x.dtype)
    print("  min  :", x.min().item())
    print("  max  :", x.max().item())
    print("  mean :", x.mean().item())
    print("  std  :", x.std().item())


def main():

    torch.manual_seed(42)

    # ============================================================
    # 1. 学習時のdataset出力を再現
    #
    # 実際のdataset:
    #   torch.Tensor
    #   [T, C, H, W]
    #   float32
    #   値域 [0,255] を想定
    # ============================================================

    raw_dataset_pixels = torch.randint(
        low=0,
        high=256,
        size=(16, 3, 480, 640),
        dtype=torch.uint8,
    ).float()

    # train_policy.py:
    # pixels = sample["pixels"][:obs_horizon]
    train_input = raw_dataset_pixels[
        :OBS_HORIZON
    ].clone()

    print_stats(
        "RAW TRAIN INPUT",
        train_input,
    )

    # ============================================================
    # 2. 学習側 transform
    # ============================================================

    train_transform = build_train_transform()

    train_data = {
        "pixels": train_input.clone(),
    }

    train_out = train_transform(
        train_data
    )

    train_img = train_out[
        "pixels"
    ].float()

    print_stats(
        "TRAIN TRANSFORM OUTPUT",
        train_img,
    )

    # ============================================================
    # 3. 推論時の入力を再現
    #
    # eval_real_robot.py:
    #
    # _pipeline.read():
    #     CHW
    #
    # XArmInferenceEnv:
    #     CHW -> HWC
    #     clip(0,255)
    #     uint8
    #
    # dp_info:
    #     [B, T, H, W, C]
    #
    # policy._prepare_info():
    #     HWC -> CHW
    #
    # img_transform()
    # ============================================================

    # 同じ画素値から実機camera出力相当を作る
    inference_raw_hwc = (
        train_input
        .permute(0, 2, 3, 1)
        .cpu()
        .numpy()
        .clip(0, 255)
        .astype(np.uint8)
    )

    print(
        "\nRAW INFERENCE INPUT"
    )
    print(
        "  shape:",
        inference_raw_hwc.shape,
    )
    print(
        "  dtype:",
        inference_raw_hwc.dtype,
    )
    print(
        "  min  :",
        inference_raw_hwc.min(),
    )
    print(
        "  max  :",
        inference_raw_hwc.max(),
    )

    inference_transform = (
        build_inference_transform()
    )

    inference_outputs = []

    for t in range(
        OBS_HORIZON
    ):
        image_hwc = (
            inference_raw_hwc[t]
        )

        # policy._prepare_info() の
        # HWC -> CHW を再現
        image_chw = np.transpose(
            image_hwc,
            (2, 0, 1),
        )

        image_tensor = (
            tv_tensors.Image(
                image_chw.copy()
            )
        )

        transformed = (
            inference_transform(
                image_tensor
            )
        )

        inference_outputs.append(
            transformed.float()
        )

    inference_img = torch.stack(
        inference_outputs,
        dim=0,
    )

    print_stats(
        "INFERENCE TRANSFORM OUTPUT",
        inference_img,
    )

    # ============================================================
    # 4. shape確認
    # ============================================================

    print(
        "\n"
        "=============================="
    )
    print("SHAPE CHECK")
    print(
        "=============================="
    )

    print(
        "train     :",
        tuple(train_img.shape),
    )

    print(
        "inference :",
        tuple(
            inference_img.shape
        ),
    )

    shape_match = (
        train_img.shape
        == inference_img.shape
    )

    print(
        "shape match:",
        shape_match,
    )

    if not shape_match:
        raise RuntimeError(
            "Shape mismatch:\n"
            f"train     = "
            f"{train_img.shape}\n"
            f"inference = "
            f"{inference_img.shape}"
        )

    # ============================================================
    # 5. 数値比較
    # ============================================================

    diff = (
        train_img
        - inference_img
    )

    abs_diff = diff.abs()

    max_abs_diff = (
        abs_diff.max().item()
    )

    mean_abs_diff = (
        abs_diff.mean().item()
    )

    rmse = torch.sqrt(
        torch.mean(
            diff ** 2
        )
    ).item()

    allclose = torch.allclose(
        train_img,
        inference_img,
        atol=1e-6,
        rtol=1e-5,
    )

    print(
        "\n"
        "=============================="
    )
    print("NUMERICAL CHECK")
    print(
        "=============================="
    )

    print(
        "max abs diff :",
        max_abs_diff,
    )
    print(
        "mean abs diff:",
        mean_abs_diff,
    )
    print(
        "RMSE         :",
        rmse,
    )
    print(
        "allclose     :",
        allclose,
    )

    # ============================================================
    # 6. frameごとの比較
    # ============================================================

    print(
        "\n"
        "=============================="
    )
    print("PER-FRAME CHECK")
    print(
        "=============================="
    )

    for t in range(
        OBS_HORIZON
    ):
        frame_diff = (
            train_img[t]
            - inference_img[t]
        ).abs()

        frame_allclose = (
            torch.allclose(
                train_img[t],
                inference_img[t],
                atol=1e-6,
                rtol=1e-5,
            )
        )

        print(
            f"t={t}: "
            f"max="
            f"{frame_diff.max().item():.10f}, "
            f"mean="
            f"{frame_diff.mean().item():.10f}, "
            f"allclose="
            f"{frame_allclose}"
        )

    # ============================================================
    # 7. 最終判定
    # ============================================================

    print(
        "\n"
        "=============================="
    )
    print("RESULT")
    print(
        "=============================="
    )

    if allclose:
        print(
            "PASS: train and inference "
            "image transforms are "
            "numerically equivalent."
        )
    else:
        print(
            "FAIL: train and inference "
            "image transforms differ."
        )


if __name__ == "__main__":
    main()