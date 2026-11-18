"""Optional telemetry for replacement-sampled (equivalent) epochs."""
from contextlib import contextmanager
import json
from pathlib import Path
import random
import subprocess
import warnings

import numpy as np
import torch


@contextmanager
def isolated_rng(device):
    python, numpy = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            yield
    finally:
        random.setstate(python)
        np.random.set_state(numpy)


def legacy_totals(path, step, length, batch_size):
    """Recover available steps from the sibling JSONL of a pre-monitor checkpoint."""
    start = step - step % length
    losses = {}
    if step != start and path.is_file():
        with path.open() as stream:
            for line in stream:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if start < row["step"] <= step and "loss" in row:
                    losses[row["step"]] = row["loss"]
    return dict(total=sum(losses.values()) * batch_size, samples=len(losses)*batch_size)


class Monitor:
    def __init__(self, config, out, device, length, state):
        self.device = device
        self.run = None
        self.run_id = state.get("run_id")
        self.total = state.get("total", 0.)
        self.samples = state.get("samples", 0)
        self.length = length
        options = config["wandb"]
        if options["enabled"] and options["mode"] != "disabled":
            with isolated_rng(device):
                try:
                    import wandb
                    self.sdk = wandb
                    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=Path(__file__).parent,
                                         text=True, capture_output=True)
                    args = {k: options[k] for k in ("project","entity","name","group","tags","mode")}
                    args.update(name=config["run_name"], id=config["run_id"])
                    if options["mode"] == "online" and self.run_id:
                        args.update(id=self.run_id, resume="must")
                    self.run = wandb.init(**args, dir=str(out.resolve()), config={
                        **config, "steps_per_epoch": length, "git_commit": git.stdout.strip() or None})
                    if options["mode"] == "online":
                        self.run_id = self.run.id
                    for metric in ("train/loss_epoch", "val/loss_epoch"):
                        self.run.define_metric(metric, step_metric="epoch")
                    for metric in ("train/loss_step","train/learning_rate","train/grad_norm",
                                   "system/step_time_sec","epoch_progress"):
                        self.run.define_metric(metric, step_metric="global_step")
                except Exception as exc:
                    self.disable(exc)

    def disable(self, exc):
        warnings.warn(f"DIAMOND W&B disabled after {type(exc).__name__}: {exc}")
        self.run = None

    def state(self):
        return dict(total=self.total, samples=self.samples, run_id=self.run_id)

    def send(self, row, image=None):
        if self.run is None:
            return
        with isolated_rng(self.device):
            try:
                if image:
                    try:
                        row["val/predictions"] = image(self.sdk)
                    except Exception as exc:
                        warnings.warn(f"DIAMOND preview skipped: {exc}")
                self.run.log(row, step=row["global_step"])
            except Exception as exc:
                self.disable(exc)

    def finish(self):
        if self.run is not None:
            with isolated_rng(self.device):
                try:
                    self.run.finish()
                except Exception as exc:
                    self.disable(exc)


@torch.inference_mode()
def preview(sdk, model, ds, cfg, device):
    sample = ds[0]
    obs = sample.obs[:4][None].to(device)
    actions = sample.act[None].to(device)
    previous = model.training
    try:
        model.eval()
        frames = model.rollout_images(obs, actions[:, :3], actions[:, 3:],
                                      torch.Generator(device=device).manual_seed(cfg.seed+103))[0].cpu()
    finally:
        model.train(previous)
    images = []
    for j, view in enumerate(cfg.views):
        ch = slice(3*j,3*j+3)
        rows = [torch.cat((sample.obs[3,ch], gt[ch], pred[ch]), dim=2)
                for gt,pred in zip(sample.obs[4:], frames)]
        grid = torch.cat(rows, dim=1).clamp(-1,1).add(1).mul(127.5).round().byte().permute(1,2,0)
        images.append(sdk.Image(grid.numpy(), caption=f"{view}: current | GT | prediction; rows t+1..t+{len(rows)}"))
    return images
