import os
from datetime import datetime
from itertools import chain
from pathlib import Path

import hydra
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, Dataset

import stable_worldmodel as swm

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from stable_worldmodel.diffusion import (
    ConditionalUnet1D,
    ResNet18ObsEncoder,
)
from stable_worldmodel.policy import DiffusionPolicy
import stable_pretraining as spt
from utils import get_img_preprocessor, get_column_normalizer
from tqdm import tqdm




class DiffusionPolicyDataset(Dataset):
    def __init__(
        self,
        dataset,
        action_key,
        obs_horizon,
        image_transform,
        action_transform,
    ):
        self.dataset = dataset
        self.action_key = action_key
        self.obs_horizon = obs_horizon
        self.image_transform = image_transform
        self.action_transform = action_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        pixels = sample["pixels"][:self.obs_horizon]
        action = sample[self.action_key]

        data = {
            "pixels": pixels,
            self.action_key: action,
        }

        data = self.image_transform(data)
        data = self.action_transform(data)

        return {
            "pixels": data["pixels"],
            "action": data[self.action_key],
        }



def get_action_stats(
    dataset,
    action_key: str,
):
    """
    Get the same mean/std used by get_column_normalizer().
    Used only for checkpointing / inference.
    """

    action_np = dataset.get_col_data(action_key)

    action = torch.from_numpy(np.asarray(action_np)).float()

    action = action[~torch.isnan(action).any(dim=1)]

    mean = action.mean(dim=0, keepdim=True,).clone()

    std = action.std(dim=0, keepdim=True,).clone()

    eps = 1e-4

    std_safe = torch.where(std < eps, torch.ones_like(std), std,)

    return mean, std_safe



# ============================================================
# checkpoint
# ============================================================


def save_checkpoint(
    path: Path,
    policy: DiffusionPolicy,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    action_mean: torch.Tensor,
    action_std: torch.Tensor,
    cfg,
):
    """
    Save everything necessary to reconstruct DiffusionPolicy.
    """

    checkpoint = {
        "epoch": epoch,
        "val_loss": val_loss,

        # Networks
        "model_state_dict":
            policy.model.state_dict(),

        "obs_encoder_state_dict":
            policy.obs_encoder.state_dict(),

        # Optimizer
        "optimizer_state_dict":
            optimizer.state_dict(),

        # Normalization
        "action_mean":
            action_mean.cpu(),

        "action_std":
            action_std.cpu(),

        "action_key":
            cfg.policy.action_key,

        # Policy parameters
        "pred_horizon":
            cfg.policy.pred_horizon,

        "obs_horizon":
            cfg.policy.obs_horizon,

        "action_horizon":
            cfg.policy.action_horizon,

        "action_dim":
            policy.action_dim,

        "obs_feature_dim":
            policy.obs_encoder.output_shape()[0],

        # Noise scheduler parameters
        "num_train_timesteps":
            cfg.diffusion.num_train_timesteps,

        "beta_schedule":
            cfg.diffusion.beta_schedule,

        "prediction_type":
            cfg.diffusion.prediction_type,

        "num_inference_steps":
            policy.num_inference_steps,

        # Full Hydra config
        "config":
            OmegaConf.to_container(
                cfg,
                resolve=True,
            ),
    }

    torch.save(checkpoint, path,)


# ============================================================
# train / validation
# ============================================================


def train_one_epoch(
    policy,
    loader,
    optimizer,
    device,
    gradient_clip_val,
    wandb_run = None,
):
    policy.model.train()
    policy.obs_encoder.train()

    total_loss = 0.0
    num_batches = 0

    params = list(
        chain(
            policy.model.parameters(),
            policy.obs_encoder.parameters(),
        )
    )

    for batch_idx, batch in enumerate(tqdm(loader, desc="train")):

        # compute_loss() handles device transfer,
        # but explicit movement here makes behavior clear.
        batch = {
            k: v.to(
                device,
                non_blocking=True,
            )
            for k, v in batch.items()
        }

        optimizer.zero_grad(
            set_to_none=True
        )

        loss = policy.compute_loss(
            batch
        )

        loss.backward()

        if gradient_clip_val is not None:
            torch.nn.utils.clip_grad_norm_(
                params,
                max_norm=gradient_clip_val,
            )

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if wandb_run is not None:
            wandb_run.log({
                "train/step_loss": loss.item(),
            })

    return total_loss / max(num_batches, 1,)


@torch.no_grad()
def validate(
    policy,
    loader,
    device,
    wandb_run = None,
):
    policy.model.eval()
    policy.obs_encoder.eval()

    total_loss = 0.0
    num_batches = 0

    for batch in loader:

        batch = {
            k: v.to(
                device,
                non_blocking=True,
            )
            for k, v in batch.items()
        }

        loss = policy.compute_loss(
            batch
        )

        total_loss += loss.item()
        num_batches += 1

        if wandb_run is not None:
            wandb_run.log({
                "val/step_loss": loss.item(),
            })

    return total_loss / max(
        num_batches,
        1,
    )


# ============================================================
# main
# ============================================================


@hydra.main(
    version_base=None,
    config_path="./config/train/policy",
    config_name="diffusion_policy",
)
def run(cfg):

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    torch.manual_seed(cfg.learning_seed)

    np.random.seed(cfg.learning_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.learning_seed)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    if (cfg.device == "cuda" and torch.cuda.is_available()):
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print("device:", device,)



    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset_cfg = dict(cfg.data.dataset)

    dataset_cfg["num_steps"] = (cfg.policy.pred_horizon)

    dataset_cfg["keys_to_load"] = ["pixels", cfg.policy.action_key,]

    dataset_cfg["keys_to_cache"] = [cfg.policy.action_key,]

    dataset = swm.data.HDF5Dataset(
        **dataset_cfg,
        transform=None,
    )

    action_dim = dataset.get_dim(cfg.policy.action_key)

    print("dataset:", cfg.data.dataset.name,)
    print("num samples:", len(dataset),)
    print("action_dim:", action_dim,)



    # --------------------------------------------------------
    # Preprocessing
    # Same preprocessing pipeline as World Model
    # --------------------------------------------------------
    image_transform = get_img_preprocessor(
        source="pixels",
        target="pixels",
        img_size=cfg.img_size,
    )

    action_transform = get_column_normalizer(
        dataset,
        source=cfg.policy.action_key,
        target=cfg.policy.action_key,
    )

    print("image transform:", image_transform)
    print("action transform:", action_transform)



    # --------------------------------------------------------
    # Save normalization statistics for inference
    # --------------------------------------------------------

    action_mean, action_std = get_action_stats(dataset, cfg.policy.action_key,)

    print("action mean:",action_mean,)

    print("action std:", action_std,)


    # --------------------------------------------------------
    # Diffusion Policy dataset wrapper
    # --------------------------------------------------------

    dataset = DiffusionPolicyDataset(
        dataset=dataset,
        action_key=cfg.policy.action_key,
        obs_horizon=cfg.policy.obs_horizon,
        image_transform=image_transform,
        action_transform=action_transform,
    )



    # --------------------------------------------------------
    # Train / validation split
    # --------------------------------------------------------

    generator = torch.Generator()
    generator.manual_seed(cfg.seed)

    train_size = int(len(dataset) * cfg.train_split)

    val_size = (len(dataset) - train_size)

    train_set, val_set = (
        torch.utils.data.random_split(
            dataset,
            [
                train_size,
                val_size,
            ],
            generator=generator,
        )
    )

    train_loader = DataLoader(
        train_set,
        batch_size=cfg.loader.batch_size,
        shuffle=True,
        num_workers=cfg.loader.num_workers,
        pin_memory=cfg.loader.pin_memory,
        persistent_workers=(cfg.loader.persistent_workers),
        drop_last=True,
        generator=generator,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=cfg.loader.batch_size,
        shuffle=False,
        num_workers=cfg.loader.num_workers,
        pin_memory=cfg.loader.pin_memory,
        persistent_workers=(cfg.loader.persistent_workers),
        drop_last=False,
    )

    # --------------------------------------------------------
    # Observation encoder
    # --------------------------------------------------------

    obs_encoder = (
        ResNet18ObsEncoder(
            pretrained=(
                cfg.encoder.pretrained
            )
        )
    )

    obs_feature_dim = (obs_encoder.output_shape()[0])

    print("obs_feature_dim:", obs_feature_dim,)

    # --------------------------------------------------------
    # Conditional U-Net
    # --------------------------------------------------------

    model = ConditionalUnet1D(
        input_dim=action_dim,

        global_cond_dim=(
            obs_feature_dim
            * cfg.policy.obs_horizon
        ),

        diffusion_step_embed_dim=(
            cfg.model.diffusion_step_embed_dim
        ),

        down_dims=tuple(
            cfg.model.down_dims
        ),

        kernel_size=(
            cfg.model.kernel_size
        ),

        n_groups=(
            cfg.model.n_groups
        ),

        cond_predict_scale=(
            cfg.model.cond_predict_scale
        ),
    )

    # --------------------------------------------------------
    # DDPM scheduler
    # --------------------------------------------------------

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=(
            cfg.diffusion.num_train_timesteps
        ),

        beta_schedule=(
            cfg.diffusion.beta_schedule
        ),

        clip_sample=(
            cfg.diffusion.clip_sample
        ),

        prediction_type=(
            cfg.diffusion.prediction_type
        ),
    )

    # --------------------------------------------------------
    # Diffusion Policy
    # --------------------------------------------------------

    policy = DiffusionPolicy(
        model=model,
        obs_encoder=obs_encoder,
        noise_scheduler=noise_scheduler,

        pred_horizon=(
            cfg.policy.pred_horizon
        ),

        obs_horizon=(
            cfg.policy.obs_horizon
        ),

        action_horizon=(
            cfg.policy.action_horizon
        ),

        action_dim=action_dim,

        num_inference_steps=(
            cfg.diffusion.num_inference_steps
        ),

        # Normalization is already done by Dataset.
        process=None,

        # Image preprocessing is already done by Dataset.
        transform=None,
    )

    policy.model.to(device)
    policy.obs_encoder.to(device)

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    parameters = list(
        chain(
            policy.model.parameters(),
            policy.obs_encoder.parameters(),
        )
    )

    optimizer = torch.optim.AdamW(
        parameters,
        lr=cfg.optimizer.lr,
        weight_decay=(
            cfg.optimizer.weight_decay
        ),
    )

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    task_name = (
        cfg.data.dataset.name.split("/")[0]
    )

    run_dir = Path(
        swm.data.utils.get_cache_dir(),
        "checkpoints",
        task_name,
        "policy",
        cfg.output_model_name,
        f"seed_{cfg.learning_seed}",
        timestamp,
    )

    run_dir.mkdir(parents=True, exist_ok=True,)

    print("run_dir:", run_dir,)

    OmegaConf.save(cfg, run_dir / "config.yaml",)

    # Normalizer saved separately as well
    torch.save(
        {
            "mean": action_mean,
            "std": action_std,
            "action_key":
                cfg.policy.action_key,
        },
        run_dir / "action_normalizer.pt",
    )

    # --------------------------------------------------------
    # WandB
    # --------------------------------------------------------

    wandb_run = None

    if cfg.wandb.enabled:
        import wandb

        wandb_name = (f"{cfg.output_model_name}_"f"{timestamp}")

        wandb_run = wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            name=wandb_name,

            config=OmegaConf.to_container(cfg, resolve=True,),
        )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_val_loss = float("inf")

    for epoch in range(
        cfg.trainer.max_epochs
    ):

        train_loss = train_one_epoch(
            policy=policy,
            loader=train_loader,
            optimizer=optimizer,
            device=device,
            gradient_clip_val=(
                cfg.trainer.gradient_clip_val
            ),
            wandb_run = wandb_run
        )

        val_loss = validate(
            policy=policy,
            loader=val_loader,
            device=device,
            wandb_run = wandb_run,
        )

        print(
            f"[Epoch "
            f"{epoch + 1:03d}/"
            f"{cfg.trainer.max_epochs:03d}] "
            f"train_loss="
            f"{train_loss:.6f} "
            f"val_loss="
            f"{val_loss:.6f}"
        )

        if wandb_run is not None:

            wandb_run.log(
                {
                    "epoch":
                        epoch + 1,

                    "train/loss":
                        train_loss,

                    "val/loss":
                        val_loss,
                }
            )

        # ----------------------------------------------------
        # Latest checkpoint
        # ----------------------------------------------------

        save_checkpoint(
            path=(
                run_dir
                / "latest.ckpt"
            ),

            policy=policy,
            optimizer=optimizer,

            epoch=epoch + 1,
            val_loss=val_loss,

            action_mean=action_mean,
            action_std=action_std,

            cfg=cfg,
        )

        # ----------------------------------------------------
        # Best checkpoint
        # ----------------------------------------------------

        save_checkpoint(
            path=run_dir / f"epoch_{epoch + 1:03d}.ckpt",
            policy=policy,
            optimizer=optimizer,
            epoch=epoch + 1,
            val_loss=val_loss,
            action_mean=action_mean,
            action_std=action_std,
            cfg=cfg,
        )

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            save_checkpoint(
                path=(
                    run_dir
                    / "best.ckpt"
                ),

                policy=policy,
                optimizer=optimizer,

                epoch=epoch + 1,
                val_loss=val_loss,

                action_mean=action_mean,
                action_std=action_std,

                cfg=cfg,
            )

            print(
                "saved new best checkpoint:"
                f" val_loss="
                f"{best_val_loss:.6f}"
            )

    # --------------------------------------------------------
    # Finish
    # --------------------------------------------------------

    if wandb_run is not None:
        wandb_run.finish()

    print("\nDiffusion Policy training finished.")

    print("best validation loss:", best_val_loss,)

    print("checkpoint:", run_dir / "best.ckpt",)


if __name__ == "__main__":
    run()