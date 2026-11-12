"""IMLE-only construction and checkpoint IO. No legacy checkpoint changes."""

from itertools import chain
from pathlib import Path
import torch
from omegaconf import OmegaConf
from stable_worldmodel.diffusion import ResNet18ObsEncoder
from stable_worldmodel.imle import ActionGenerator, IMLEPolicy
from action_normalization import validate_action_stats, restore_action_processor

if not OmegaConf.has_resolver("imle_gripper_weight"):
    OmegaConf.register_new_resolver(
        "imle_gripper_weight", lambda horizon: 0.1 / int(horizon)
    )

ACTION_LAYOUT = ["x", "y", "z", "qx", "qy", "qz", "qw", "gripper"]


def parameters(policy):
    return list(
        chain(
            policy.model.parameters(),
            policy.obs_encoder.parameters(),
            policy.wrist_obs_encoder.parameters(),
        )
    )


def modules(policy):
    return (policy.model, policy.obs_encoder, policy.wrist_obs_encoder)


def encoder_architecture(cfg):
    """Self-contained architecture; missing type means historical ResNet18."""
    kind = cfg.encoder.get("type", "resnet18")
    feature_dim = int(cfg.imle.feature_dim)
    if feature_dim < 1:
        raise ValueError("IMLE feature_dim must be positive")
    if kind == "resnet18":
        return dict(
            type=kind,
            feature_dim=feature_dim,
            backbone="resnet18",
            backbone_feature_dim=512,
            projection_bias=True,
        )
    if kind == "convnet":
        from stable_worldmodel.imle.obs_encoder import convnet_parameters

        if cfg.encoder.pretrained:
            raise ValueError("IMLE ConvNet does not provide pretrained weights")
        return dict(
            type=kind,
            feature_dim=feature_dim,
            convnet=convnet_parameters(cfg.encoder.get("convnet", {})),
            normalization="none",
            dropout=0.0,
            attention=False,
            conv_bias=True,
            projection_bias=True,
        )
    raise ValueError(
        f"Unknown IMLE encoder.type: {kind!r}; expected resnet18 or convnet"
    )


def build_policy(cfg, action_dim=8, process=None, transform=None):
    if action_dim != 8:
        raise ValueError("IMLE robot adapter requires 8D Cartesian actions")
    architecture = encoder_architecture(cfg)
    if architecture["type"] == "convnet":
        from stable_worldmodel.imle.obs_encoder import ConvNetObsEncoder

        feature_dim = int(cfg.imle.feature_dim)
        enc = ConvNetObsEncoder(feature_dim, **architecture["convnet"])
        wrist = ConvNetObsEncoder(feature_dim, **architecture["convnet"])
    else:
        enc = ResNet18ObsEncoder(pretrained=cfg.encoder.pretrained)
        wrist = ResNet18ObsEncoder(pretrained=cfg.encoder.pretrained)
        feature_dim = int(cfg.imle.feature_dim)

        # Explicit projection to s3fmp's conditioning width; DP encoders stay untouched.
        class ProjectedEncoder(torch.nn.Sequential):
            def output_shape(self):
                return (feature_dim,)

        enc = ProjectedEncoder(enc, torch.nn.Linear(enc.output_shape()[0], feature_dim))
        wrist = ProjectedEncoder(
            wrist, torch.nn.Linear(wrist.output_shape()[0], feature_dim)
        )
    model = ActionGenerator(
        action_dim=action_dim,
        pred_horizon=cfg.policy.pred_horizon,
        obs_horizon=cfg.policy.obs_horizon,
        feature_dim=feature_dim,
        channels=cfg.imle.channels,
        gripper_channels=cfg.imle.gripper_channels,
        kernel_size=cfg.imle.kernel_size,
        heads=cfg.imle.heads,
        separate_gripper=cfg.imle.separate_gripper,
    )
    policy = IMLEPolicy(
        model=model,
        obs_encoder=enc,
        wrist_obs_encoder=wrist,
        pred_horizon=cfg.policy.pred_horizon,
        obs_horizon=cfg.policy.obs_horizon,
        action_horizon=cfg.policy.action_horizon,
        action_dim=action_dim,
        noise_scale=cfg.imle.noise_scale,
        n_samples_per_condition=cfg.imle.n_samples_per_condition,
        epsilon=cfg.imle.epsilon,
        epsilon_scale_by_dim=cfg.imle.epsilon_scale_by_dim,
        gripper_loss_weight=cfg.imle.gripper_loss_weight,
        process=process,
        transform=transform,
    )

    policy.encoder_architecture = architecture
    return policy


def save_checkpoint(
    path, policy, optimizer, epoch, val_loss, cfg, action_stats, ema=None
):
    validate_action_stats(action_stats, policy.action_dim)
    architecture = encoder_architecture(cfg)
    if architecture != policy.encoder_architecture:
        raise ValueError("IMLE encoder config disagrees with the constructed policy")
    config = OmegaConf.to_container(cfg, resolve=True)
    config["encoder"]["type"] = architecture["type"]
    if architecture["type"] == "convnet":
        config["encoder"]["convnet"] = architecture["convnet"]
    state = dict(
        policy_type="imle",
        epoch=epoch,
        val_loss=float(val_loss),
        generator_state_dict=policy.model.state_dict(),
        obs_encoder_state_dict=policy.obs_encoder.state_dict(),
        wrist_obs_encoder_state_dict=policy.wrist_obs_encoder.state_dict(),
        optimizer_state_dict=optimizer.state_dict(),
        config=config,
        architecture=config["imle"],
        encoder_architecture=architecture,
        noise_dim=policy.noise_dim,
        pred_horizon=policy.pred_horizon,
        obs_horizon=policy.obs_horizon,
        action_horizon=policy.action_horizon,
        action_dim=policy.action_dim,
        action_key=cfg.policy.action_key,
        action_layout=ACTION_LAYOUT,
        position_units="m",
        quaternion_order="xyzw",
        gripper_convention="0=open,1=closed",
        **{k: v.cpu() if torch.is_tensor(v) else v for k, v in action_stats.items()},
    )
    if ema is not None:
        state["ema"] = ema.state_dict()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_imle_policy(checkpoint_path, image_transform, device="cuda"):
    if checkpoint_path is None or str(checkpoint_path).strip().lower() in (
        "",
        "null",
        "none",
    ):
        raise ValueError(
            "An IMLE checkpoint path is required; use YAML null for missing paths"
        )
    path = Path(checkpoint_path).expanduser()
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("policy_type") != "imle":
        raise ValueError("Expected policy_type=imle checkpoint")
    required = (
        "config",
        "architecture",
        "noise_dim",
        "pred_horizon",
        "obs_horizon",
        "action_horizon",
        "action_dim",
        "generator_state_dict",
        "obs_encoder_state_dict",
        "wrist_obs_encoder_state_dict",
        "action_key",
        "action_layout",
        "position_units",
        "quaternion_order",
        "gripper_convention",
    )
    missing = [k for k in required if k not in state]
    if missing:
        raise ValueError(f"IMLE checkpoint missing fields: {missing}")
    if (
        state["action_key"] != "action_cartesian"
        or state["action_layout"] != ACTION_LAYOUT
        or state["position_units"] != "m"
        or state["quaternion_order"] != "xyzw"
        or state["gripper_convention"] != "0=open,1=closed"
    ):
        raise ValueError(
            "IMLE action layout/units do not match the robot Cartesian interface"
        )
    mode = validate_action_stats(state, state["action_dim"])
    cfg = OmegaConf.create(state["config"])
    architecture = encoder_architecture(cfg)
    if state.get("encoder_architecture", architecture) != architecture:
        raise ValueError(
            "IMLE checkpoint encoder architecture disagrees with saved config"
        )
    cfg.encoder.pretrained = (
        False  # Saved weights are authoritative; no network download.
    )
    for key in ("pred_horizon", "obs_horizon", "action_horizon"):
        if cfg.policy[key] != state[key]:
            raise ValueError(f"IMLE checkpoint {key} disagrees with saved config")
    if state["architecture"] != state["config"]["imle"]:
        raise ValueError("IMLE checkpoint architecture disagrees with saved config")
    policy = build_policy(
        cfg,
        state["action_dim"],
        process={
            "action_cartesian": restore_action_processor(state, state["action_dim"])
        },
        transform={"pixels": image_transform, "wrist_pixels": image_transform},
    )
    if policy.noise_dim != state["noise_dim"]:
        raise ValueError("IMLE noise dimension disagrees with saved architecture")
    for module, key in zip(
        modules(policy),
        (
            "generator_state_dict",
            "obs_encoder_state_dict",
            "wrist_obs_encoder_state_dict",
        ),
    ):
        module.load_state_dict(state[key])
        module.to(device)
        module.eval()
    if "ema" in state:
        from torch_ema import ExponentialMovingAverage

        ema = ExponentialMovingAverage(
            parameters(policy), decay=cfg.imle.ema_decay, use_num_updates=True
        )
        ema.load_state_dict(state["ema"])
        ema.to(device)
        ema.copy_to(parameters(policy))
    for module in modules(policy):
        module.requires_grad_(False)
    policy.offline_checkpoint_metadata = dict(
        policy_type="imle",
        checkpoint_path=str(path.resolve()),
        action_normalization=mode,
        checkpoint_config=state["config"],
        noise_dim=policy.noise_dim,
        sampling="one-shot Gaussian noise; s3fmp IMLE",
        uses_ema="ema" in state,
        encoder_architecture=architecture,
    )
    return policy
