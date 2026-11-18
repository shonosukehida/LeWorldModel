"""Offline six-frame DIAMOND training. No collector or robot dependencies."""
import json
from datetime import datetime
from zoneinfo import ZoneInfo
import math
import time
from pathlib import Path
import random

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import default_collate

from diamond_monitor import Monitor, legacy_totals, preview
from diamond_run import prepare_run
from diamond_dataset import DiamondDataset, inspect_file, load_wm_stats, split_episodes
from stable_worldmodel.wm.diamond.checkpoint import read_checkpoint, restore_rng, save_checkpoint
from stable_worldmodel.wm.diamond.model import ALIGNMENT, DiamondWorldModel
from stable_worldmodel.wm.diamond.upstream.optim import configure_opt, get_lr_sched
from stable_worldmodel.wm.diamond.upstream.types import Batch


def batch_to(samples, device):
    return Batch(*(x.to(device) for x in default_collate(samples)))


@torch.no_grad()
def probe(model, dataset, cfg, device, full=False):
    """Fixed windows and noise make losses comparable without consuming training randomness."""
    model.eval()
    losses = []
    windows = torch.Generator().manual_seed(cfg.seed + 101)
    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(cfg.seed + 102)
        if devices:
            torch.cuda.default_generators[device.index].manual_seed(cfg.seed + 102)
        if full:
            batches = (range(i, min(i + cfg.batch_size, len(dataset)))
                       for i in range(0, len(dataset), cfg.batch_size))
        else:
            batches = (torch.randint(len(dataset), (cfg.batch_size,), generator=windows).tolist()
                       for _ in range(cfg.val_batches))
        for ids in batches:
            loss, _ = model(batch_to([dataset[i] for i in ids], device))
            if not loss.isfinite():
                raise FloatingPointError("Non-finite probe loss")
            losses.append(loss.item() * len(ids) if full else loss.item())
    model.train()
    return sum(losses)/(len(dataset) if full else len(losses))


@hydra.main(version_base=None, config_path="config/train", config_name="diamond")
def main(cfg):
    started = datetime.now(ZoneInfo("Asia/Tokyo"))
    previous = None
    config = OmegaConf.to_container(cfg, resolve=True)
    if cfg.action_alignment != ALIGNMENT["action"]:
        raise ValueError("Unsupported action alignment")
    if min(cfg.steps, cfg.batch_size, cfg.val_batches, cfg.save_every, cfg.validate_every) < 1:
        raise ValueError("Counts and intervals must be positive")
    if cfg.warmup_steps < 0 or min(cfg.grad_clip, cfg.lr, cfg.eps) <= 0 or cfg.weight_decay < 0:
        raise ValueError("Invalid optimizer settings")
    device = torch.device(cfg.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Supported devices: cpu and cuda")
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.use_deterministic_algorithms(True)
    path = Path(cfg.dataset).expanduser()
    normalizer = load_wm_stats(cfg.stats or path.with_name("process_stats.npz"))
    dataset, episodes = inspect_file(path, cfg.views)
    split = split_episodes(episodes, cfg.seed, cfg.val_fraction)
    train = DiamondDataset(path, episodes, split["train"], normalizer, cfg.views, limit=cfg.train_windows)
    val = DiamondDataset(path, episodes, split["val"], normalizer, cfg.views)
    model = DiamondWorldModel(cfg.views).to(device)
    optimizer = configure_opt(model, cfg.lr, cfg.weight_decay, cfg.eps)
    scheduler = get_lr_sched(optimizer, cfg.warmup_steps)
    windows = torch.Generator().manual_seed(cfg.seed)
    length = math.ceil(len(train) / cfg.batch_size)
    if (cfg.validation_mode not in ("fixed", "full") or cfg.validate_every_epochs < 1
            or cfg.wandb.mode not in ("online","offline","disabled")
            or min(cfg.wandb.image_horizon, cfg.wandb.image_every_epochs) < 1):
        raise ValueError("Invalid validation or W&B configuration")
    state = {}
    step = 0
    if cfg.resume:
        cp = read_checkpoint(Path(cfg.resume).expanduser())
        for key, current in dict(spec=model.spec, dataset=dataset, split=split, normalizer=normalizer).items():
            if cp[key] != current:
                raise ValueError(f"Resume {key} mismatch")
        mutable = {"output", "resume", "steps", "save_every", "validate_every", "dataset", "stats"}
        mutable.update({"wandb", "validation_mode", "validate_every_epochs"})
        if any(cp["config"][k] != config[k] for k in config.keys() - mutable):
            raise ValueError("Resume training configuration mismatch")
        if cp["versions"]["torch"] != str(torch.__version__):
            raise ValueError("Resume Torch version mismatch")
        model.load_state_dict(cp["model"], strict=True)
        optimizer.load_state_dict(cp["optimizer"])
        scheduler.load_state_dict(cp["scheduler"])
        restore_rng(cp["rng"], windows, device)
        step = cp["global_step"]
        state = cp["config"].get("epoch_monitor")
        if state is None:
            state = legacy_totals(Path(cfg.resume).expanduser().with_name("metrics.jsonl"),
                                  step, length, cfg.batch_size)
        previous = cp["config"]
        del cp
    if cfg.steps <= step:
        raise ValueError("steps must exceed the restored global_step")
    # Each invocation has its own output, including resumes. No prior logs/checkpoints overwritten.
    out = prepare_run(config, previous, state, started)
    (out/"config.yaml").write_text(OmegaConf.to_yaml(OmegaConf.create(config)))
    (out/"data.json").write_text(json.dumps(dict(dataset=dataset, split=split, alignment=ALIGNMENT,
                                                normalizer=normalizer), indent=2))
    print("Recorded rows at stride 1. Acquisition period and physical action causality unverified.", flush=True)
    # A one-window diagnostic repeatedly reads identical data; cache just that preprocessed window.
    if len(train) == 1:
        single = [train[0]]
        train.close()
        train = single
    monitor = Monitor(config, out, device, length, state)
    image_ds = None
    if monitor.run and cfg.wandb.log_images:
        try:
            image_ds = DiamondDataset(path, episodes, split["val"], normalizer, cfg.views,
                                      frames=4+cfg.wandb.image_horizon, limit=1)
        except ValueError as exc:
            import warnings
            warnings.warn(f"DIAMOND image logging unavailable: {exc}")
    with (out/"metrics.jsonl").open("x", buffering=1) as log:
        def record(**fields):
            line = json.dumps(dict(step=step, **fields), allow_nan=False)
            print(line, flush=True)
            print(line, file=log)
        if cfg.train_windows is not None:
            record(fixed_train_loss=probe(model, train, cfg, device))
        while step < cfg.steps:
            start = time.perf_counter()
            ids = torch.randint(len(train), (cfg.batch_size,), generator=windows).tolist()
            batch = batch_to([train[i] for i in ids], device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = model(batch)
            if not loss.isfinite():
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip, error_if_nonfinite=True)
            lr = optimizer.param_groups[0]["lr"]
            optimizer.step()
            scheduler.step()
            step += 1
            metrics = dict(loss=loss.item(), grad_norm=norm.item(), lr=lr)
            elapsed = time.perf_counter() - start
            if step % cfg.validate_every == 0 or step == cfg.steps:
                metrics["val_loss"] = probe(model, val, cfg, device, cfg.validation_mode == "full")
                if cfg.train_windows is not None:
                    metrics["fixed_train_loss"] = probe(model, train, cfg, device)
            record(**metrics)
            monitor.total += metrics["loss"] * len(ids)
            monitor.samples += len(ids)
            epoch = (step-1)//length + 1
            row = {"global_step": step, "epoch_progress": step/length,
                   "train/loss_step": metrics["loss"], "train/learning_rate": lr,
                   "train/grad_norm": metrics["grad_norm"], "system/step_time_sec": elapsed}
            image = None
            if step % length == 0 or step == cfg.steps:
                expected = ((step-1) % length + 1) * cfg.batch_size
                row.update(epoch=epoch, epoch_partial=bool(step % length),
                           epoch_samples=monitor.samples, epoch_history_complete=monitor.samples == expected)
                if monitor.samples == expected:
                    row["train/loss_epoch"] = monitor.total / monitor.samples
                if monitor.run and (epoch % cfg.validate_every_epochs == 0 or step == cfg.steps):
                    row["val/loss_epoch"] = (metrics["val_loss"] if "val_loss" in metrics else
                                             probe(model, val, cfg, device, cfg.validation_mode == "full"))
                if image_ds is not None and epoch % cfg.wandb.image_every_epochs == 0:
                    image = lambda sdk: preview(sdk, model, image_ds, cfg, device)
                if step % length == 0:
                    monitor.total = 0.
                    monitor.samples = 0
            monitor.send(row, image)
            if step % cfg.save_every == 0 or step == cfg.steps:
                save_checkpoint(out/f"step-{step:06d}.pt", model, optimizer, scheduler, windows,
                                config=dict(config, epoch_monitor=monitor.state()), dataset=dataset, split=split, normalizer=normalizer,
                                global_step=step, epoch=step*cfg.batch_size//len(train))
    monitor.finish()
    if image_ds is not None:
        image_ds.close()
    val.close()
    if isinstance(train, DiamondDataset):
        train.close()


if __name__ == "__main__":
    main()
