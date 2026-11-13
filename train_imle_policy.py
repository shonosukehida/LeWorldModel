"""Independent IMLE trainer: s3fmp RS loss/gripper/EMA with the DP data pipeline."""

from contextlib import nullcontext
from datetime import datetime
from pathlib import Path
from time import perf_counter
import json
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


def collect_proprio_windows(raw, indices, obs_horizon):
    """Match raw[i]['proprio'][:obs_horizon] without loading other columns.

    Keep split order and repeated rows exactly as in the original torch.cat.
    The trainer constructs raw with transform=None. Dataset clip indices already
    encode episode boundaries; offsets and frameskip map them to stored rows.
    """
    proprio = raw.get_col_data("proprio")
    clips = np.asarray(raw.clip_indices)[np.asarray(indices, dtype=np.int64)]
    starts = np.asarray(raw.offsets)[clips[:, 0]] + clips[:, 1]
    steps = np.arange(raw.num_steps)[:obs_horizon] * raw.frameskip
    rows = starts[:, None] + steps[None, :]
    return torch.from_numpy(proprio[rows.reshape(-1)])


def log_preprocessing_time(stage, started):
    print(f"[IMLE preprocessing] {stage}: {perf_counter() - started:.6f}s", flush=True)


class IMLEProprioDataset(DiffusionPolicyDataset):
    """Same-row image/state history; leave the shared DP dataset unchanged."""

    def __getitem__(self, idx):
        # Fetch one raw window so stochastic data sources cannot desynchronize views.
        sample = self.dataset[idx]
        data = {k: sample[k][:self.obs_horizon] for k in ('pixels', 'wrist_pixels')}
        data[self.action_key] = sample[self.action_key]
        data = self.action_transform(self.image_transform(data))
        proprio = torch.as_tensor(sample['proprio'][:self.obs_horizon]).float()
        if proprio.shape != (self.obs_horizon, 8) or not torch.isfinite(proprio).all():
            raise ValueError('IMLE dataset proprio requires finite (obs_horizon,8)')
        return dict(pixels=data['pixels'], wrist_pixels=data['wrist_pixels'],
                    proprio=proprio, action=data[self.action_key])


def checkpoint_directory(cfg):
    """Use the DP task convention, with a flat timestamped IMLE run directory."""
    if cfg.data.dataset.name is not None:
        task_name = Path(cfg.data.dataset.name).parts[0]
    else:
        task_name = Path(cfg.data.dataset.path).parent.parent.name
    return (
        Path(swm.data.utils.get_cache_dir())
        / "checkpoints" / task_name / "policy" / "imle"
        / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    )


class StepLogger:
    """Keep the optimizer-step axis across epochs, even with W&B disabled."""

    def __init__(self, run):
        self.run = run
        self.global_step = 0
        self.metric_names = set()
        if run is not None:
            run.define_metric("global_step")
            run.define_metric("epoch")
            for name in (
                "train_loss", "train_eval_loss", "val_loss",
                "train/*", "train_eval/*", "val/*",
            ):
                run.define_metric(name, step_metric="epoch")

    def __call__(self, loss, metrics):
        self.global_step += 1
        if self.run is None:
            return
        values = {"train/step_" + k: v for k, v in metrics.items()}
        values["train/step_loss"] = loss
        for name in values:
            if name not in self.metric_names:
                self.run.define_metric(name, step_metric="global_step")
                self.metric_names.add(name)
        self.run.log(dict(global_step=self.global_step, **values))


def run_epoch(policy, loader, device, optimizer=None, ema=None, step_logger=None):
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
            loss_value = loss.item()
            total += loss_value * size
            count += size
            metrics = {key: float(value) for key, value in policy.loss_metrics.items()}
            for key, value in metrics.items():
                logs[key] = logs.get(key, 0.0) + value * size
            if training and step_logger is not None:
                step_logger(loss_value, metrics)
    if count == 0:
        raise ValueError(
            "Empty IMLE data loader; reduce batch size or supply more samples"
        )
    return total / count, {key: value / count for key, value in logs.items()}


def evaluate_ema(policy, train_eval_loader, val_loader, device, ema, seed):
    """Evaluate both splits at one EMA snapshot with isolated train-eval RNG.

    With seed=None, validation retains its historical RNG consumption. The added
    train pass is always isolated; a fixed seed restarts each split independently.
    """
    def evaluate(loader, preserve_rng):
        rng = (
            torch.random.fork_rng(devices=list(range(torch.cuda.device_count())))
            if preserve_rng else nullcontext()
        )
        with rng:
            if seed is not None:
                torch.manual_seed(seed)
            return run_epoch(policy, loader, device)

    with ema.average_parameters(parameters(policy)):
        train_result = evaluate(train_eval_loader, preserve_rng=True)
        val_result = evaluate(val_loader, preserve_rng=seed is not None)
    return train_result, val_result


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
    use_proprio = cfg.imle.get("use_proprio", False)
    dataset_cfg = dict(cfg.data.dataset)
    dataset_cfg.update(
        num_steps=cfg.policy.pred_horizon,
        keys_to_load=["pixels", "wrist_pixels", cfg.policy.action_key],
        keys_to_cache=[cfg.policy.action_key],
    )
    if use_proprio:
        dataset_cfg['keys_to_load'].append('proprio')
        dataset_cfg['keys_to_cache'].append('proprio')
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
    if use_proprio:
        preprocessing_started = stage_started = perf_counter()
        print("[IMLE preprocessing] starting dataset initialization", flush=True)
    raw = swm.data.HDF5Dataset(**dataset_cfg, transform=None)
    if use_proprio:
        log_preprocessing_time("dataset initialization/cache", stage_started)
        stage_started = perf_counter()
    action_dim = raw.get_dim(cfg.policy.action_key)
    stats = fit_action_stats(
        raw, cfg.policy.action_key, cfg.policy.action_normalization
    )
    if use_proprio:
        log_preprocessing_time("action statistics", stage_started)
        stage_started = perf_counter()
    images = spt.data.transforms.Compose(
        *[
            get_img_preprocessor(source=k, target=k, img_size=cfg.img_size)
            for k in ("pixels", "wrist_pixels")
        ]
    )
    dataset_class = IMLEProprioDataset if use_proprio else DiffusionPolicyDataset
    dataset = dataset_class(
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
    train_eval_loader = DataLoader(train, shuffle=False, drop_last=False, **options)
    val_loader = DataLoader(val, shuffle=False, drop_last=False, **options)
    if use_proprio:
        log_preprocessing_time("transforms/split/loaders", stage_started)
        stage_started = perf_counter()
    proprio_stats = None
    if use_proprio:
        from stable_worldmodel.imle.proprio import fit_proprio_stats
        # Only selected Train observation windows contribute; validation is untouched.
        train_proprio = collect_proprio_windows(raw, train.indices, cfg.policy.obs_horizon)
        proprio_stats = fit_proprio_stats(train_proprio)
        log_preprocessing_time("Train proprio collection/statistics", stage_started)
        stage_started = perf_counter()
    policy = build_policy(cfg, action_dim, proprio_stats=proprio_stats)
    for module in modules(policy):
        module.to(device)
    optimizer = torch.optim.AdamW(
        parameters(policy), lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay
    )
    ema = ExponentialMovingAverage(
        parameters(policy), decay=cfg.imle.ema_decay, use_num_updates=True
    )
    ema.to(device)
    if use_proprio:
        log_preprocessing_time("policy/device/optimizer/EMA", stage_started)
        stage_started = perf_counter()
    output = checkpoint_directory(cfg)
    output.mkdir(parents=True, exist_ok=False)
    if use_proprio:
        from stable_worldmodel.imle.proprio import normalize_proprio
        def distribution(values):
            x = values.double()
            return {name: value.tolist() for name, value in dict(
                min=x.amin(0), max=x.amax(0), mean=x.mean(0),
                std=x.std(0, correction=0),
            ).items()}
        diagnostics = {"layout": proprio_stats["proprio_layout"],
                       "fit_source": "Train observation windows only; repeated rows retain window weights"}
        for name, values in (("train", train_proprio), ("validation", collect_proprio_windows(
            raw, val.indices, cfg.policy.obs_horizon
        ))):
            normalized = normalize_proprio(values, proprio_stats)
            diagnostics[name] = dict(count=len(values), raw=distribution(values),
                                     normalized=distribution(normalized))
        (output / "proprio_diagnostics.json").write_text(json.dumps(diagnostics, indent=2, allow_nan=False) + "\n")
        log_preprocessing_time("proprio diagnostics/output directory", stage_started)
        stage_started = perf_counter()
    print("run_dir:", output)
    OmegaConf.save(cfg, output / "config.yaml")
    run_wandb = None
    if cfg.wandb.enabled:
        import wandb
        run_name = f"imle_policy_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        run_wandb = wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            name=run_name,
            config=OmegaConf.to_container(cfg, resolve=True),
        )
    step_logger = StepLogger(run_wandb)
    best, stale = float("inf"), 0
    if use_proprio:
        log_preprocessing_time("config/logging setup", stage_started)
        log_preprocessing_time("total before first epoch", preprocessing_started)
    try:
        for epoch in range(cfg.trainer.max_epochs):
            train_loss, train_logs = run_epoch(
                policy, train_loader, device, optimizer, ema, step_logger
            )
            (train_eval_loss, train_eval_logs), (val_loss, val_logs) = evaluate_ema(
                policy, train_eval_loader, val_loader, device, ema,
                cfg.trainer.validation_seed,
            )
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
                f"epoch={epoch} train_loss={train_loss:.6g} "
                f"train_eval_loss={train_eval_loss:.6g} val_loss={val_loss:.6g} output={output}"
            )
            if run_wandb:
                run_wandb.log(
                    dict(
                        epoch=epoch,
                        train_loss=train_loss,
                        train_eval_loss=train_eval_loss,
                        val_loss=val_loss,
                        **{"train/" + k: v for k, v in train_logs.items()},
                        **{"train_eval/" + k: v for k, v in train_eval_logs.items()},
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
