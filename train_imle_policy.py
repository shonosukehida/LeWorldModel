"""Independent IMLE trainer: s3fmp RS loss/gripper/EMA with the DP data pipeline."""

from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
import hydra
import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, random_split
from torch_ema import ExponentialMovingAverage
import stable_worldmodel as swm
import stable_pretraining as spt
from train_policy import DiffusionPolicyDataset
from utils import get_img_preprocessor
from action_normalization import fit_action_stats, ActionTransform
from imle_checkpoint import build_policy, save_checkpoint, modules, parameters


def run_epoch(policy, loader, device, optimizer=None, ema=None):
    training = optimizer is not None
    for module in modules(policy):
        module.train(training)
    total, count = 0.0, 0
    logs = {}
    with torch.set_grad_enabled(training):
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            if training:
                optimizer.zero_grad(set_to_none=True)
            loss = policy.compute_loss(batch)
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite IMLE loss")
            if training:
                loss.backward()
                optimizer.step()
                if ema is not None:
                    ema.update(parameters(policy))
            size = batch["action"].shape[0]
            total += loss.item() * size
            count += size
            for key, value in policy.loss_metrics.items():
                logs[key] = logs.get(key, 0.0) + float(value) * size
    if count == 0:
        raise ValueError(
            "Empty IMLE data loader; reduce batch size or supply more samples"
        )
    return total / count, {key: value / count for key, value in logs.items()}


@hydra.main(
    version_base=None, config_path="./config/train/policy", config_name="imle_policy"
)
def run(cfg):
    torch.manual_seed(cfg.learning_seed)
    np.random.seed(cfg.learning_seed)
    device = torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA requested but unavailable; use device=cpu for a CPU run"
        )
    dataset_cfg = dict(cfg.data.dataset)
    dataset_cfg.update(
        num_steps=cfg.policy.pred_horizon,
        keys_to_load=["pixels", "wrist_pixels", cfg.policy.action_key],
        keys_to_cache=[cfg.policy.action_key],
    )
    if dataset_cfg.get("path") is not None:
        path = Path(dataset_cfg["path"]).expanduser()
        if not path.is_absolute():
            raise ValueError("Dataset path must be absolute")
        if not path.suffix:
            path = path.with_suffix(".h5")
        if not path.is_file():
            raise FileNotFoundError(path)
        dataset_cfg["path"] = str(path)
    else:
        dataset_cfg.pop("path", None)
    raw = swm.data.HDF5Dataset(**dataset_cfg, transform=None)
    action_dim = raw.get_dim(cfg.policy.action_key)
    stats = fit_action_stats(
        raw, cfg.policy.action_key, cfg.policy.action_normalization
    )
    images = spt.data.transforms.Compose(
        *[
            get_img_preprocessor(source=k, target=k, img_size=cfg.img_size)
            for k in ("pixels", "wrist_pixels")
        ]
    )
    dataset = DiffusionPolicyDataset(
        raw,
        cfg.policy.action_key,
        cfg.policy.obs_horizon,
        images,
        ActionTransform(stats, cfg.policy.action_key),
    )
    generator = torch.Generator().manual_seed(cfg.seed)
    train_size = int(len(dataset) * cfg.train_split)
    if not 0 < train_size < len(dataset):
        raise ValueError("Train and validation splits must both be nonempty")
    train, val = random_split(
        dataset, [train_size, len(dataset) - train_size], generator=generator
    )
    options = dict(
        batch_size=cfg.loader.batch_size,
        num_workers=cfg.loader.num_workers,
        pin_memory=cfg.loader.pin_memory,
        persistent_workers=cfg.loader.persistent_workers,
    )
    train_loader = DataLoader(
        train, shuffle=True, drop_last=True, generator=generator, **options
    )
    val_loader = DataLoader(val, shuffle=False, drop_last=False, **options)
    policy = build_policy(cfg, action_dim)
    for module in modules(policy):
        module.to(device)
    optimizer = torch.optim.AdamW(
        parameters(policy), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay
    )
    ema = ExponentialMovingAverage(
        parameters(policy), decay=cfg.imle.ema_decay, use_num_updates=True
    )
    ema.to(device)
    output = Path(cfg.output_dir).expanduser() / datetime.now().strftime(
        "%Y%m%d_%H%M%S_%f"
    )
    output.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(cfg, output / "config.yaml")
    run_wandb = None
    if cfg.wandb.enabled:
        import wandb

        run_wandb = wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            name=f"imle_seed_{cfg.learning_seed}",
            config=OmegaConf.to_container(cfg, resolve=True),
        )
    best, stale = float("inf"), 0
    try:
        for epoch in range(cfg.trainer.max_epochs):
            train_loss, train_logs = run_epoch(
                policy, train_loader, device, optimizer, ema
            )
            # s3fmp validates EMA weights and preserves the training RNG stream.
            validation_rng = (
                torch.random.fork_rng(devices=list(range(torch.cuda.device_count())))
                if cfg.trainer.validation_seed is not None
                else nullcontext()
            )
            with validation_rng:
                if cfg.trainer.validation_seed is not None:
                    torch.manual_seed(cfg.trainer.validation_seed)
                with ema.average_parameters(parameters(policy)):
                    val_loss, val_logs = run_epoch(policy, val_loader, device)
            save_checkpoint(
                output / "latest.ckpt",
                policy,
                optimizer,
                epoch,
                val_loss,
                cfg,
                stats,
                ema,
            )
            if val_loss < best:
                best, stale = val_loss, 0
                save_checkpoint(
                    output / "best.ckpt",
                    policy,
                    optimizer,
                    epoch,
                    val_loss,
                    cfg,
                    stats,
                    ema,
                )
            else:
                stale += 1
            print(
                f"epoch={epoch} train_loss={train_loss:.6g} val_loss={val_loss:.6g} output={output}"
            )
            if run_wandb:
                run_wandb.log(
                    dict(
                        epoch=epoch,
                        train_loss=train_loss,
                        val_loss=val_loss,
                        **{"train/" + k: v for k, v in train_logs.items()},
                        **{"val/" + k: v for k, v in val_logs.items()},
                    )
                )
            if stale >= cfg.trainer.patience:
                break
    finally:
        if run_wandb:
            run_wandb.finish()


if __name__ == "__main__":
    run()
