"""Train an independent UpperJEPA; no lower checkpoint or GPC integration."""
from datetime import datetime
from pathlib import Path

import hydra
import lightning as pl
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger, WandbLogger
from omegaconf import OmegaConf
import torch

from upper_data import prepare_upper_data
from upper_training import UpperTrainingModule, serialize_process


@hydra.main(version_base=None, config_path="./config/train", config_name="upper_jepa")
def run(cfg):
    pl.seed_everything(cfg.learning_seed, workers=True)
    config = OmegaConf.to_container(cfg, resolve=True)
    dataset_cfg = config["data"]["dataset"]
    if dataset_cfg.get("path"):
        path = Path(hydra.utils.to_absolute_path(dataset_cfg["path"])).expanduser()
    else:
        from stable_worldmodel.data.utils import get_cache_dir
        path = Path(get_cache_dir(), "datasets", dataset_cfg["name"]).with_suffix(".h5")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    run_dir = Path(hydra.utils.to_absolute_path(config["output_dir"])) / f"upper_jepa_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    config["normalization_stats_path"] = str(run_dir / "process_stats.npz")
    config["data"]["dataset"]["path"] = str(path.resolve())
    wm = config["wm"]
    train, val, process, split = prepare_upper_data(
        path, run_dir, horizon=wm["horizon"], history_size=wm["history_size"],
        train_split=config["train_split"], seed=config["seed"], img_size=config["img_size"])
    OmegaConf.save(OmegaConf.create(config), run_dir / "config.yaml")
    generator = torch.Generator().manual_seed(config["seed"])
    train_loader = torch.utils.data.DataLoader(train, **config["loader"], shuffle=True, generator=generator)
    val_loader = torch.utils.data.DataLoader(val, **config["loader"], shuffle=False)
    module = UpperTrainingModule(config, serialize_process(process), split)
    loggers = [CSVLogger(str(run_dir), name="metrics", version="")]
    if config["wandb"]["enabled"]:
        options = dict(config["wandb"]["config"])
        options.update(name=f"{config['output_model_name']}_{stamp}", id=f"upper_jepa_{stamp}")
        wandb = WandbLogger(**options)
        wandb.log_hyperparams(config)
        loggers.append(wandb)
    checkpoint = ModelCheckpoint(dirpath=run_dir, filename="upper_jepa-{epoch:03d}",
                                 every_n_epochs=1, save_top_k=-1, save_last=True)
    trainer = pl.Trainer(**config["trainer"], default_root_dir=str(run_dir),
                         callbacks=[checkpoint], logger=loggers, num_sanity_val_steps=1)
    trainer.fit(module, train_dataloaders=train_loader, val_dataloaders=val_loader)
    print(f"UpperJEPA run directory: {run_dir}")
    return run_dir


if __name__ == "__main__":
    run()
