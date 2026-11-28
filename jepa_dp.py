"""JEPA-DP checkpoint integration. No robot imports or random initialization of JEPA."""
from copy import deepcopy
import hashlib
import math
import os
from pathlib import Path
from tempfile import mkdtemp, TemporaryDirectory

import torch
from torch import nn
from omegaconf import OmegaConf
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from stable_worldmodel.diffusion import ConditionalUnet1D
from stable_worldmodel.diffusion.jepa import (
    FEATURE_MODULES, FrozenJEPAEncoder, EmptyWristEncoder, JEPADiffusionPolicy,
)
from action_normalization import restore_action_processor
from normalization_stats import load_normalization_process
from utils import get_eval_img_preprocessor


def finetune_config(settings):
    options = {"mode": "frozen", "unfreeze_last_n_blocks": 3, "lr": 1e-5}
    options.update(settings.get("finetune", {}))
    lr = options["lr"]
    if isinstance(lr, bool) or not isinstance(lr, (float, int)) or not math.isfinite(lr) or lr < 0:
        raise ValueError(f"encoder.finetune.lr must be a nonnegative finite number, got {lr!r}")
    return options


def fingerprint(path):
    with Path(path).open('rb') as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
        return digest.hexdigest()


def verify_jepa_source(source):
    for key, artifact in source.items():
        actual = fingerprint(artifact["path"])
        if actual != artifact["sha256"]:
            raise RuntimeError(f"JEPA {key} changed: {artifact['path']}, "
                               f"expected SHA-256 {artifact['sha256']}, got {actual}")


def create_jepa_run_dir(path, policy):
    try:
        path.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        # Concurrent runs may have the same second-resolution timestamp.
        path = Path(mkdtemp(prefix=path.name + "_", dir=path.parent))
    policy.jepa_rolling_checkpoints = {path.resolve() / name: None for name in ("best.ckpt", "latest.ckpt")}
    return path


def save_jepa_checkpoint(path, checkpoint, policy):
    path = Path(path)
    path = path.parent.resolve() / path.name
    for artifact in policy.jepa_source.values():
        source = Path(artifact["path"])
        if path.resolve() == source or (path.exists() and source.exists() and path.samefile(source)):
            raise ValueError(f"Refusing to overwrite protected JEPA source: {path}")

    # Only the training process which created this fresh directory may update
    # its rolling checkpoints. Direct saves and epoch snapshots are create-only.
    rolling = getattr(policy, "jepa_rolling_checkpoints", {})
    previous = rolling.get(path)
    if path.is_symlink() or (previous is None and path.exists()):
        raise FileExistsError(f"Refusing to overwrite existing checkpoint: {path}")
    with TemporaryDirectory(dir=path.parent) as staging:
        staged = Path(staging) / "checkpoint.pt"
        torch.save(checkpoint, staged)
        identity = staged.stat()
        if previous is not None and path.exists():
            if not os.path.samestat(previous, path.lstat()):
                raise FileExistsError(f"Rolling checkpoint was replaced by another file: {path}")
            os.replace(staged, path)
        else:
            os.link(staged, path)  # Atomically fail if another writer claimed the name.
        if path in rolling:
            rolling[path] = identity


def load_pretrained_encoder(settings):
    """Load the model-object format produced by ModelObjectCallBack.

    Lightning state_dict-only files cannot describe their architecture or training
    normalization, and are deliberately rejected rather than partially loaded.
    """
    paths = {}
    for key in ("checkpoint", "world_config", "process_stats"):
        if not settings.get(key):
            raise ValueError(f"encoder.{key} must reference the JEPA training artifact")
        paths[key] = Path(settings[key]).expanduser().resolve(strict=True)
    provenance = {key: {'path': str(path), 'sha256': fingerprint(path)}
                  for key, path in paths.items()}
    options = finetune_config(settings)
    world_config = OmegaConf.load(paths['world_config'])
    image_size = int(world_config.img_size)
    root = torch.load(paths['checkpoint'], map_location='cpu', weights_only=False)
    if not isinstance(root, nn.Module):
        raise ValueError("JEPA requires a *_object.ckpt model object, not a state_dict checkpoint")
    matches = [m for m in root.modules() if callable(getattr(m, 'encode', None))
               and all(isinstance(getattr(m, key, None), nn.Module) for key in FEATURE_MODULES)]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one multiview JEPA model, found {len(matches)}")
    world = deepcopy(matches[0]).cpu()
    # Retain JEPA.encode itself and all five trained feature modules only.
    for name in ('predictor', 'action_encoder', 'pred_proj'):
        setattr(world, name, nn.Identity())
    process, _ = load_normalization_process(paths['process_stats'])
    if 'proprio' not in process:
        raise ValueError("JEPA process_stats must include world-training proprio statistics")
    prop = process['proprio']
    encoder = FrozenJEPAEncoder(world, prop.mean_, prop.scale_, image_size, mode=options["mode"],
                                unfreeze_last_n_blocks=options["unfreeze_last_n_blocks"])
    expected = (int(world_config.wm.embed_dim), int(world_config.wm.embed_dim),
                int(world_config.wm.prop_embed_dim))
    if encoder.feature_dims != expected or encoder.proprio_mean.shape[-1] != world_config.wm.prop_dim:
        raise ValueError("JEPA checkpoint architecture does not match world_config")
    provenance['world_config'] = dict(provenance['world_config'],
                                     config=OmegaConf.to_container(world_config, resolve=False))
    return encoder, provenance


def checkpoint_artifact(policy):
    return dict(jepa_format_version=1,
                # Object stores the trained architecture (including source JEPA.encode).
                # State dict is separately strict-loaded, detecting missing/incompatible weights.
                jepa_encoder=deepcopy(policy.obs_encoder).cpu(),
                jepa_source=policy.jepa_source,
                jepa_feature_dims=policy.obs_encoder.feature_dims,
                jepa_image_size=policy.obs_encoder.image_size)


def load_jepa_policy(checkpoint, checkpoint_path, device):
    # The existing offline runners request CUDA; match training's CPU fallback.
    if str(device).startswith('cuda') and not torch.cuda.is_available():
        device = 'cpu'
    if checkpoint.get('jepa_format_version') != 1:
        raise ValueError('Unsupported JEPA-DP checkpoint version')
    encoder = checkpoint['jepa_encoder']
    if not isinstance(encoder, FrozenJEPAEncoder):
        raise ValueError('Missing JEPA encoder architecture')
    if (tuple(checkpoint['jepa_feature_dims']) != encoder.feature_dims
            or checkpoint['obs_feature_dim'] != encoder.output_shape()[0]
            or checkpoint['wrist_obs_feature_dim'] != 0
            or checkpoint['jepa_image_size'] != encoder.image_size):
        raise ValueError('Incompatible JEPA feature dimensions or preprocessing')
    encoder.load_state_dict(checkpoint['obs_encoder_state_dict'], strict=True)
    encoder.requires_grad_(False).eval()
    config = checkpoint['config']
    model = ConditionalUnet1D(
        input_dim=checkpoint['action_dim'],
        global_cond_dim=encoder.output_shape()[0] * checkpoint['obs_horizon'],
        **dict(config['model']))
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    scheduler = DDPMScheduler(**{key: checkpoint[key] for key in
        ('num_train_timesteps', 'beta_schedule', 'prediction_type')},
        clip_sample=config['diffusion']['clip_sample'])
    transform = get_eval_img_preprocessor(encoder.image_size)
    policy = JEPADiffusionPolicy(
        model=model, obs_encoder=encoder, wrist_obs_encoder=EmptyWristEncoder(),
        noise_scheduler=scheduler,
        **{key: checkpoint[key] for key in ('pred_horizon', 'obs_horizon',
           'action_horizon', 'action_dim', 'num_inference_steps')},
        process={checkpoint['action_key']: restore_action_processor(checkpoint, checkpoint['action_dim'])},
        transform={'pixels': transform, 'wrist_pixels': transform})
    policy.wrist_obs_encoder.load_state_dict(checkpoint['wrist_obs_encoder_state_dict'], strict=True)
    policy.model.to(device).eval().requires_grad_(False)
    policy.obs_encoder.to(device)
    policy.jepa_source = checkpoint['jepa_source']
    policy.offline_checkpoint_metadata = dict(
        checkpoint_path=str(Path(checkpoint_path).resolve()), encoder_type='jepa',
        jepa_source=policy.jepa_source, jepa_feature_dims=encoder.feature_dims,
        image_size=encoder.image_size, checkpoint_config=config,
        action_normalization=checkpoint.get('action_normalization', 'zscore'),
        device=str(device),
        scheduler=dict(scheduler.config))
    return policy
