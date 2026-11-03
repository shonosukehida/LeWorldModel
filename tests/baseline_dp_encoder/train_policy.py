# Frozen pre-selection checkpoint writer.
def save_checkpoint(path: Path, policy: DiffusionPolicy, optimizer: torch.optim.Optimizer, epoch: int, val_loss: float, action_mean: torch.Tensor, action_std: torch.Tensor, cfg, action_stats=None):
    """
    Save everything necessary to reconstruct DiffusionPolicy.
    """
    checkpoint = {'epoch': epoch, 'val_loss': val_loss, 'model_state_dict': policy.model.state_dict(), 'obs_encoder_state_dict': policy.obs_encoder.state_dict(), 'wrist_obs_encoder_state_dict': policy.wrist_obs_encoder.state_dict(), 'multiview': policy.wrist_obs_encoder is not None, 'wrist_obs_feature_dim': policy.wrist_obs_encoder.output_shape()[0], 'optimizer_state_dict': optimizer.state_dict(), 'action_mean': action_mean.cpu() if action_mean is not None else None, 'action_std': action_std.cpu() if action_std is not None else None, 'action_key': cfg.policy.action_key, 'pred_horizon': cfg.policy.pred_horizon, 'obs_horizon': cfg.policy.obs_horizon, 'action_horizon': cfg.policy.action_horizon, 'action_dim': policy.action_dim, 'obs_feature_dim': policy.obs_encoder.output_shape()[0], 'num_train_timesteps': cfg.diffusion.num_train_timesteps, 'beta_schedule': cfg.diffusion.beta_schedule, 'prediction_type': cfg.diffusion.prediction_type, 'num_inference_steps': policy.num_inference_steps, 'config': OmegaConf.to_container(cfg, resolve=True)}
    if action_stats is None:
        action_stats = dict(action_normalization='zscore', action_mean=action_mean, action_std=action_std)
    validate_action_stats(action_stats, policy.action_dim)
    checkpoint.pop('action_mean')
    checkpoint.pop('action_std')
    checkpoint.update({k: v.cpu() if torch.is_tensor(v) else v for (k, v) in action_stats.items()})
    torch.save(checkpoint, path)
