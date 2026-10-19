"""Existing multiview DP checkpoint loader, independent of robot/GUI imports."""
from pathlib import Path

import torch
import stable_worldmodel as swm
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from stable_worldmodel.diffusion import ConditionalUnet1D, ResNet18ObsEncoder
from normalization_stats import SafeStandardScaler


def load_diffusion_policy(
    checkpoint_path,
    image_transform,
    device="cuda",
):
    checkpoint_path = Path(checkpoint_path).expanduser()

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    train_cfg = checkpoint["config"]

    # --------------------------------------------------
    # Observation encoder
    # --------------------------------------------------

    obs_encoder = ResNet18ObsEncoder(
        pretrained=False,
    )

    wrist_obs_encoder = ResNet18ObsEncoder(
        pretrained=False,
    )

    # --------------------------------------------------
    # Conditional U-Net
    # --------------------------------------------------

    model = ConditionalUnet1D(
        input_dim=checkpoint["action_dim"],

        global_cond_dim=(
            (
                checkpoint["obs_feature_dim"]
                + checkpoint["wrist_obs_feature_dim"]
            )
            * checkpoint["obs_horizon"]
        ),

        diffusion_step_embed_dim=(
            train_cfg["model"]["diffusion_step_embed_dim"]
        ),

        down_dims=tuple(
            train_cfg["model"]["down_dims"]
        ),

        kernel_size=(
            train_cfg["model"]["kernel_size"]
        ),

        n_groups=(
            train_cfg["model"]["n_groups"]
        ),

        cond_predict_scale=(
            train_cfg["model"]["cond_predict_scale"]
        ),
    )

    # --------------------------------------------------
    # DDPM scheduler
    # --------------------------------------------------

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=(
            checkpoint["num_train_timesteps"]
        ),

        beta_schedule=(
            checkpoint["beta_schedule"]
        ),

        clip_sample=(
            train_cfg["diffusion"]["clip_sample"]
        ),

        prediction_type=(
            checkpoint["prediction_type"]
        ),
    )

    # --------------------------------------------------
    # DP normalization
    # --------------------------------------------------

    action_processor = SafeStandardScaler(
        eps=1e-4
    )

    action_processor.mean_ = (checkpoint["action_mean"].cpu().numpy())
    action_processor.scale_ = (checkpoint["action_std"].cpu().numpy())
    action_key = checkpoint["action_key"]

    dp_process = {action_key: action_processor,}

    # --------------------------------------------------
    # DiffusionPolicy
    # --------------------------------------------------

    diffusion_policy = swm.policy.DiffusionPolicy(
        model=model,

        obs_encoder=obs_encoder,
        wrist_obs_encoder=wrist_obs_encoder,

        noise_scheduler=noise_scheduler,

        pred_horizon=checkpoint["pred_horizon"],
        obs_horizon=checkpoint["obs_horizon"],
        action_horizon=checkpoint["action_horizon"],
        action_dim=checkpoint["action_dim"],

        num_inference_steps=(
            checkpoint["num_inference_steps"]
        ),

        process=dp_process,

        transform={
            "pixels": image_transform,
            "wrist_pixels": image_transform,
        },
    )

    # --------------------------------------------------
    # Load weights
    # --------------------------------------------------

    diffusion_policy.model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    diffusion_policy.obs_encoder.load_state_dict(
        checkpoint["obs_encoder_state_dict"]
    )

    diffusion_policy.wrist_obs_encoder.load_state_dict(
        checkpoint[
            "wrist_obs_encoder_state_dict"
        ]
    )

    diffusion_policy.model.to(device)
    diffusion_policy.obs_encoder.to(device)
    diffusion_policy.wrist_obs_encoder.to(device)

    diffusion_policy.model.eval()
    diffusion_policy.obs_encoder.eval()
    diffusion_policy.wrist_obs_encoder.eval()

    diffusion_policy.model.requires_grad_(False)
    diffusion_policy.obs_encoder.requires_grad_(False)
    diffusion_policy.wrist_obs_encoder.requires_grad_(False)

    return diffusion_policy



