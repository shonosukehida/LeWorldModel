import numpy as np
import torch
from pathlib import Path
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback


def get_img_preprocessor(
    source: str,
    target: str,
    img_size: int = 224,
):
    imagenet_stats = dt.dataset_stats.ImageNet

    def scale_to_unit_range(x):
        x = x.float()

        min_val = x.min()
        max_val = x.max()

        if min_val < 0:
            raise ValueError(
                f"Unexpected image range: "
                f"min={min_val.item()}, max={max_val.item()}"
            )

        # すでに [0, 1]
        if max_val <= 1.0 + 1e-6:
            return x

        # [0, 255]
        if max_val <= 255.0 + 1e-6:
            return x / 255.0

        raise ValueError(
            f"Unexpected image range: "
            f"min={min_val.item()}, max={max_val.item()}"
        )

    scale = dt.transforms.WrapTorchTransform(
        scale_to_unit_range,
        source=source,
        target=target,
    )

    to_image = dt.transforms.ToImage(
        dtype=torch.float32,
        scale=False,
        mean=imagenet_stats["mean"],
        std=imagenet_stats["std"],
        source=source,
        target=target,
    )

    resize = dt.transforms.Resize(
        img_size,
        source=source,
        target=target,
    )

    return dt.transforms.Compose(
        scale,
        to_image,
        resize,
    )

def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()
    
    # print("mean: ", mean)
    # print("std: ", std)
    eps = 1e-4
    std_safe = torch.where(std < eps, torch.ones_like(std), std)
    # print("std_safe:", std_safe)

    def norm_fn(x):
        # print("before norm:", x)
        mean_ = mean.to(x.device)
        std_ = std_safe.to(x.device)
        
        normed = ((x - mean_) / std_).float()
        # print("after norm:", normed) #z:0 になるべき
        return normed

    normalizer = dt.transforms.WrapTorchTransform(norm_fn, source=source, target=target)
    return normalizer

class ModelObjectCallBack(Callback):
    """Callback to pickle model object after each epoch."""

    def __init__(self, dirpath, filename="model_object", epoch_interval: int = 1):
        super().__init__()
        self.dirpath = Path(dirpath)
        self.filename = filename
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        output_path = (
            self.dirpath
            / f"{self.filename}_epoch_{trainer.current_epoch + 1}_object.ckpt"
        )

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._dump_model(pl_module.model, output_path)

            # save final epoch
            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._dump_model(pl_module.model, output_path)

    def _dump_model(self, model, path):
        try:
            torch.save(model, path)
        except Exception as e:
            print(f"Error saving model object: {e}")