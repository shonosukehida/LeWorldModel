"""Offline six-frame DIAMOND training. No collector or robot dependencies."""
import json
from pathlib import Path
import random

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.utils.data import default_collate

from diamond_dataset import DiamondDataset, inspect_file, load_wm_stats, split_episodes
from stable_worldmodel.wm.diamond.checkpoint import read_checkpoint, restore_rng, save_checkpoint
from stable_worldmodel.wm.diamond.model import ALIGNMENT, DiamondWorldModel
from stable_worldmodel.wm.diamond.upstream.optim import configure_opt, get_lr_sched
from stable_worldmodel.wm.diamond.upstream.types import Batch


def batch_to(samples, device):
    return Batch(*(x.to(device) for x in default_collate(samples)))


@torch.no_grad()
def probe(model, dataset, cfg, device):
    """Fixed windows and noise make losses comparable without consuming training randomness."""
    model.eval()
    losses = []
    windows = torch.Generator().manual_seed(cfg.seed + 101)
    devices = [device] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(cfg.seed + 102)
        if devices:
            torch.cuda.default_generators[device.index].manual_seed(cfg.seed + 102)
        for _ in range(cfg.val_batches):
            ids = torch.randint(len(dataset), (cfg.batch_size,), generator=windows)
            loss, _ = model(batch_to([dataset[i] for i in ids.tolist()], device))
            if not loss.isfinite():
                raise FloatingPointError("Non-finite probe loss")
            losses.append(loss.item())
    model.train()
    return sum(losses)/len(losses)


@hydra.main(version_base=None, config_path="config/train", config_name="diamond")
def main(cfg):
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
    step = 0
    if cfg.resume:
        cp = read_checkpoint(Path(cfg.resume).expanduser())
        for key, current in dict(spec=model.spec, dataset=dataset, split=split, normalizer=normalizer).items():
            if cp[key] != current:
                raise ValueError(f"Resume {key} mismatch")
        mutable = {"output", "resume", "steps", "save_every", "validate_every", "dataset", "stats"}
        if any(cp["config"][k] != config[k] for k in config.keys() - mutable):
            raise ValueError("Resume training configuration mismatch")
        if cp["versions"]["torch"] != str(torch.__version__):
            raise ValueError("Resume Torch version mismatch")
        model.load_state_dict(cp["model"], strict=True)
        optimizer.load_state_dict(cp["optimizer"])
        scheduler.load_state_dict(cp["scheduler"])
        restore_rng(cp["rng"], windows, device)
        step = cp["global_step"]
        del cp
    if cfg.steps <= step:
        raise ValueError("steps must exceed the restored global_step")
    # Each invocation has its own output, including resumes. No prior logs/checkpoints overwritten.
    out = Path(config["output"]).expanduser()
    out.mkdir(parents=True, exist_ok=False)
    (out/"config.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))
    (out/"data.json").write_text(json.dumps(dict(dataset=dataset, split=split, alignment=ALIGNMENT,
                                                normalizer=normalizer), indent=2))
    print("Recorded rows at stride 1. Acquisition period and physical action causality unverified.", flush=True)
    # A one-window diagnostic repeatedly reads identical data; cache just that preprocessed window.
    if len(train) == 1:
        single = [train[0]]
        train.close()
        train = single
    with (out/"metrics.jsonl").open("x", buffering=1) as log:
        def record(**fields):
            line = json.dumps(dict(step=step, **fields), allow_nan=False)
            print(line, flush=True)
            print(line, file=log)
        if cfg.train_windows is not None:
            record(fixed_train_loss=probe(model, train, cfg, device))
        while step < cfg.steps:
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
            if step % cfg.validate_every == 0 or step == cfg.steps:
                metrics["val_loss"] = probe(model, val, cfg, device)
                if cfg.train_windows is not None:
                    metrics["fixed_train_loss"] = probe(model, train, cfg, device)
            record(**metrics)
            if step % cfg.save_every == 0 or step == cfg.steps:
                save_checkpoint(out/f"step-{step:06d}.pt", model, optimizer, scheduler, windows,
                                config=config, dataset=dataset, split=split, normalizer=normalizer,
                                global_step=step, epoch=step*cfg.batch_size//len(train))
    val.close()
    if isinstance(train, DiamondDataset):
        train.close()


if __name__ == "__main__":
    main()
