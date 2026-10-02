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
    train_policy.py と同じ画像 transform
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
    # 1. 実際の学習データと同じ形式を作る
    #
    # 実測:
    # raw pixels = torch.Tensor [T, C, H, W], float32
    #
    # 本番 dataset は [0,255] 相当の float32 を保持している想定
    # ============================================================

    raw_train = torch.randint(
        low=0,
        high=256,
        size=(16, 3, 480, 640),
        dtype=torch.uint8,
    ).float()

    # train_policy.py:
    # pixels = sample["pixels"][:obs_horizon]
    train_input = raw_train[:OBS_HORIZON].clone()

    print_stats(
        "raw train input",
        train_input,
    )

    # ============================================================
    # 2. 学習側 transform
    #
    # 入力:
    # [To, C, H, W]
    # ============================================================

    train_transform = build_train_transform()

    train_data = {
        "pixels": train_input.clone(),
    }

    train_out = train_transform(train_data)

    train_img = train_out["pixels"].float()

    print_stats(
        "train transform output",
        train_img,
    )

    # ============================================================
    # 3. 推論側を再現
    #
    # eval_real_robot.py 側では、各時刻の camera image は
    # HWC uint8 RGB
    #
    # dp_info:
    # [B, To, H, W, C]
    #
    # _prepare_info():
    # HWC -> CHW
    #
    # その後 inference transform
    # ============================================================

    # 学習入力と完全に同じ画素値を使う
    #
    # train_input:
    # [To, C, H, W]
    #
    # camera形式に戻す:
    # [To, H, W, C]
    inference_raw_hwc = (
        train_input
        .permute(0, 2, 3, 1)
        .cpu()
        .numpy()
        .clip(0, 255)
        .astype(np.uint8)
    )

    print("\ninference raw HWC:")
    print("  shape:", inference_raw_hwc.shape)
    print("  dtype:", inference_raw_hwc.dtype)

    inference_transform = build_inference_transform()

    inference_outputs = []

    for t in range(OBS_HORIZON):

        image_hwc = inference_raw_hwc[t]

        # eval_real_robot.py / _prepare_info() を再現:
        # HWC -> CHW
        image_chw = np.transpose(
            image_hwc,
            (2, 0, 1),
        )

        image_tensor = tv_tensors.Image(
            image_chw.copy()
        )

        transformed = inference_transform(
            image_tensor
        )

        inference_outputs.append(
            transformed.float()
        )

    inference_img = torch.stack(
        inference_outputs,
        dim=0,
    )

    print_stats(
        "inference transform output",
        inference_img,
    )

    # ============================================================
    # 4. shape確認
    # ============================================================

    print("\n==============================")
    print("SHAPE CHECK")
    print("==============================")

    print(
        "train     :",
        tuple(train_img.shape),
    )

    print(
        "inference :",
        tuple(inference_img.shape),
    )

    assert (
        train_img.shape
        == inference_img.shape
    ), (
        "Shape mismatch:\n"
        f"train     = {train_img.shape}\n"
        f"inference = {inference_img.shape}"
    )

    # ============================================================
    # 5. 数値比較
    # ============================================================

    diff = (
        train_img
        - inference_img
    )

    abs_diff = diff.abs()

    max_abs_diff = abs_diff.max().item()
    mean_abs_diff = abs_diff.mean().item()

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

    print("\n==============================")
    print("NUMERICAL CHECK")
    print("==============================")

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
    # 6. 各時刻でも確認
    # ============================================================

    print("\n==============================")
    print("PER-FRAME CHECK")
    print("==============================")

    for t in range(OBS_HORIZON):

        frame_diff = (
            train_img[t]
            - inference_img[t]
        ).abs()

        print(
            f"t={t}: "
            f"max={frame_diff.max().item():.10f}, "
            f"mean={frame_diff.mean().item():.10f}, "
            f"allclose={torch.allclose(train_img[t], inference_img[t], atol=1e-6, rtol=1e-5)}"
        )

    # ============================================================
    # 7. 最終判定
    # ============================================================

    print("\n==============================")
    print("RESULT")
    print("==============================")

    if allclose:

        print(
            "PASS: "
            "train and inference image transforms are numerically equivalent."
        )

    else:

        print(
            "FAIL: "
            "train and inference image transforms differ."
        )


if __name__ == "__main__":
    main()