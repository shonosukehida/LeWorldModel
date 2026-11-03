# Frozen loader from real_robot_multiview before encoder selection.
def load_diffusion_policy(checkpoint_path, image_transform, device='cuda'):
    checkpoint_path = Path(checkpoint_path).expanduser()
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    required = ('action_dim', 'config', 'pred_horizon', 'obs_horizon', 'action_horizon', 'obs_feature_dim', 'wrist_obs_feature_dim', 'num_inference_steps', 'num_train_timesteps', 'beta_schedule', 'prediction_type', 'action_key', 'model_state_dict', 'obs_encoder_state_dict', 'wrist_obs_encoder_state_dict')
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError(f'Diffusion checkpoint missing configuration: {missing}')
    for (section, keys) in {'model': ('diffusion_step_embed_dim', 'down_dims', 'kernel_size', 'n_groups', 'cond_predict_scale'), 'diffusion': ('clip_sample',)}.items():
        for key in keys:
            if key not in checkpoint['config'].get(section, {}):
                raise ValueError(f'Diffusion checkpoint missing config.{section}.{key}')
    normalization_mode = validate_action_stats(checkpoint, checkpoint['action_dim'])
    train_cfg = checkpoint['config']
    obs_encoder = ResNet18ObsEncoder(pretrained=False)
    wrist_obs_encoder = ResNet18ObsEncoder(pretrained=False)
    for (key, encoder) in (('obs_feature_dim', obs_encoder), ('wrist_obs_feature_dim', wrist_obs_encoder)):
        if encoder.output_shape()[0] != checkpoint[key]:
            raise ValueError(f'Checkpoint {key} incompatible with saved ResNet18 encoder architecture')
    model = ConditionalUnet1D(input_dim=checkpoint['action_dim'], global_cond_dim=(checkpoint['obs_feature_dim'] + checkpoint['wrist_obs_feature_dim']) * checkpoint['obs_horizon'], diffusion_step_embed_dim=train_cfg['model']['diffusion_step_embed_dim'], down_dims=tuple(train_cfg['model']['down_dims']), kernel_size=train_cfg['model']['kernel_size'], n_groups=train_cfg['model']['n_groups'], cond_predict_scale=train_cfg['model']['cond_predict_scale'])
    noise_scheduler = DDPMScheduler(num_train_timesteps=checkpoint['num_train_timesteps'], beta_schedule=checkpoint['beta_schedule'], clip_sample=train_cfg['diffusion']['clip_sample'], prediction_type=checkpoint['prediction_type'])
    if normalization_mode == 'min_max':
        print('normalization_mode: min_max')
        action_processor = restore_action_processor(checkpoint, checkpoint['action_dim'])
    else:
        print('normalization_mode: zscore')
        action_processor = SafeStandardScaler(eps=0.0001)
        action_processor.mean_ = checkpoint['action_mean'].cpu().numpy()
        action_processor.scale_ = checkpoint['action_std'].cpu().numpy()
    action_key = checkpoint['action_key']
    dp_process = {action_key: action_processor}
    diffusion_policy = swm.policy.DiffusionPolicy(model=model, obs_encoder=obs_encoder, wrist_obs_encoder=wrist_obs_encoder, noise_scheduler=noise_scheduler, pred_horizon=checkpoint['pred_horizon'], obs_horizon=checkpoint['obs_horizon'], action_horizon=checkpoint['action_horizon'], action_dim=checkpoint['action_dim'], num_inference_steps=checkpoint['num_inference_steps'], process=dp_process, transform={'pixels': image_transform, 'wrist_pixels': image_transform})
    diffusion_policy.model.load_state_dict(checkpoint['model_state_dict'])
    diffusion_policy.obs_encoder.load_state_dict(checkpoint['obs_encoder_state_dict'])
    diffusion_policy.wrist_obs_encoder.load_state_dict(checkpoint['wrist_obs_encoder_state_dict'])
    diffusion_policy.model.to(device)
    diffusion_policy.obs_encoder.to(device)
    diffusion_policy.wrist_obs_encoder.to(device)
    diffusion_policy.model.eval()
    diffusion_policy.obs_encoder.eval()
    diffusion_policy.wrist_obs_encoder.eval()
    diffusion_policy.model.requires_grad_(False)
    diffusion_policy.obs_encoder.requires_grad_(False)
    diffusion_policy.wrist_obs_encoder.requires_grad_(False)
    diffusion_policy.offline_checkpoint_metadata = {'checkpoint_path': str(checkpoint_path.resolve()), 'action_normalization': normalization_mode, 'obs_feature_dim': checkpoint['obs_feature_dim'], 'wrist_obs_feature_dim': checkpoint['wrist_obs_feature_dim'], 'checkpoint_config': train_cfg, 'scheduler': dict(noise_scheduler.config)}
    return diffusion_policy
