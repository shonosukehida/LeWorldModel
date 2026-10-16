"""Standalone upper JEPA losses, compatible logging, and portable checkpoints."""
from copy import deepcopy

import lightning as pl
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from module import ARPredictor, IDM, MLP, SIGReg
from normalization_stats import SafeStandardScaler
from upper_data import STAT_KEYS, UpperObservationTransform, validate_process
from upper_jepa import UpperJEPA


def serialize_process(process):
    validate_process(process)
    return {key: {name: np.asarray(getattr(process[key], name)).tolist()
                  for name in ("mean_", "scale_", "raw_min_", "raw_max_", "normed_min_", "normed_max_", "eps")}
            for key in STAT_KEYS}


def deserialize_process(values):
    process = {}
    for key in STAT_KEYS:
        if key not in values:
            raise ValueError(f"Missing required checkpoint normalization processor: {key}")
        scaler = SafeStandardScaler()
        for name in ("mean_", "scale_", "raw_min_", "raw_max_", "normed_min_", "normed_max_", "eps"):
            if name not in values[key]:
                raise ValueError(f"Missing checkpoint normalization value: {key}.{name}")
            setattr(scaler, name, np.asarray(values[key][name]))
        process[key] = scaler
    validate_process(process)
    return process


def build_upper_model(config, process):
    """Fresh ViT-tiny modules only; no lower model/checkpoint is accepted."""
    import stable_pretraining as spt
    validate_process(process)
    wm = config["wm"]
    if (config["encoder_scale"], wm["embed_dim"], wm["prop_embed_dim"], wm["prop_dim"]) != ("tiny", 192, 32, 8):
        raise ValueError("UpperJEPA requires ViT-tiny and the 192+192+32 latent layout with 8D proprio")
    encoders = [spt.backbone.utils.vit_hf(
        "tiny", patch_size=config["patch_size"], image_size=config["img_size"],
        pretrained=False, use_mask_token=False) for _ in range(2)]
    hidden = encoders[0].config.hidden_size
    projection_hidden = config.get("projector_hidden_dim", 2048)
    def projector(output_dim):
        return MLP(hidden, projection_hidden, output_dim, norm_fn=nn.BatchNorm1d)
    return UpperJEPA(
        horizon=wm["horizon"], history_size=wm["history_size"],
        use_action_delta=wm["use_action_delta"], actions_normalized=False,
        action_mean=process["action_cartesian"].mean_, action_scale=process["action_cartesian"].scale_,
        delta_mean=process["action_delta"].mean_, delta_scale=process["action_delta"].scale_,
        overhead_encoder=encoders[0], wrist_encoder=encoders[1],
        overhead_projector=projector(192), wrist_projector=projector(192),
        prop_encoder=MLP(8, 64, 32), pred_proj=projector(416),
        predictor=ARPredictor(num_frames=wm["history_size"], input_dim=416,
                              hidden_dim=hidden, output_dim=hidden, **config["predictor"]))


def compute_upper_losses(model, sigreg, idm, batch, *, sigreg_weight, idm_weight):
    """Use K context states/actions; supervise only the final H-step prediction.

    Cartesian actions must be physical (B,K,H,8). Observations have K+1 frames
    and already use the shared image/proprio preprocessing. Like lejepa_forward,
    target latents are produced by the same encoder without stop-gradient/EMA.
    """
    if model.actions_normalized:
        raise ValueError("Upper training expects physical action input; actions_normalized must be False")
    actions = batch["action_cartesian"]
    k, h = model.history_size, model.horizon
    if actions.ndim != 4 or actions.shape[1:] != (k, h, 8):
        raise ValueError("action_cartesian must have shape (B, history_size, horizon, 8)")
    # Shared macro generation is the sole action-standardization operation.
    macros = model.build_macro_action(actions)
    encoded = model.encode({key: batch[key] for key in ("pixels", "wrist_pixels", "proprio")})
    emb = encoded["emb"]
    if emb.shape[1:] != (k + 1, 416):
        raise ValueError("Observations must contain history_size+1 upper frames with 416D latents")
    predicted = model.predict(emb[:, :-1], model.action_encoder(macros))
    final_prediction = predicted[:, -1:]
    target = emb[:, -1:]
    output = {"pred_loss": F.mse_loss(final_prediction, target)}
    for name, start, end in (("overhead", 0, 192), ("wrist", 192, 384), ("prop", 384, 416)):
        output[f"{name}_pred_loss"] = F.mse_loss(final_prediction[..., start:end], target[..., start:end])
    # Keep the sketch/statistic in float32 under mixed precision.
    with torch.autocast(device_type=emb.device.type, enabled=False):
        output["sigreg_loss"] = sigreg(emb.float().transpose(0, 1))
    output["loss"] = output["pred_loss"] + sigreg_weight * output["sigreg_loss"]
    if idm is not None:
        inverse = idm(emb[:, :-1], emb[:, 1:])
        output["idm_loss"] = F.mse_loss(inverse, macros)
        output["loss"] = output["loss"] + idm_weight * output["idm_loss"]
        output["idm_prediction"] = inverse
    output.update(emb=emb, predicted_emb=predicted, macro_action=macros)
    return output


class UpperTrainingModule(pl.LightningModule):
    """Optimize the upper model through a randomly initialized, frozen IDM.

    spt.Module calls forward with stages 'fit' and 'validate'; use those exact
    stages and the two lejepa_forward log_dict calls here. Automatic Lightning
    optimization supplies gradient clipping/mixed precision/checkpoint resume.
    """
    def __init__(self, config, normalization_statistics, split=None):
        super().__init__()
        self.config = deepcopy(config)
        self.process = deserialize_process(normalization_statistics)
        self.save_hyperparameters(dict(config=self.config,
                                      normalization_statistics=normalization_statistics, split=split))
        self.model = build_upper_model(self.config, self.process)
        self.sigreg = SIGReg(**self.config["loss"]["sigreg"]["kwargs"])
        idm_cfg = self.config["loss"]["idm"]
        self.idm = IDM(416, self.model.macro_action_dim, idm_cfg["hidden_dim"]) if idm_cfg["use"] else None
        if self.idm is not None:
            self.idm.requires_grad_(False)
        self.observation_transform = UpperObservationTransform(self.process, self.config["img_size"])

    def forward(self, batch, stage=None):
        loss_cfg = self.config["loss"]
        output = compute_upper_losses(self.model, self.sigreg, self.idm, batch,
                                      sigreg_weight=loss_cfg["sigreg"]["weight"],
                                      idm_weight=loss_cfg["idm"]["weight"])
        if stage is not None:
            losses_dict = {f"{stage}/{key}": value.detach() for key, value in output.items() if "loss" in key}
            step_losses_dict = {f"{stage}/{key}_step": value.detach() for key, value in output.items() if "loss" in key}
            self.log_dict(losses_dict, on_step=False, on_epoch=True, sync_dist=True)
            self.log_dict(step_losses_dict, on_step=True, on_epoch=False, sync_dist=True)
        return output

    def training_step(self, batch, batch_idx):
        return self(batch, stage="fit")["loss"]

    def validation_step(self, batch, batch_idx):
        return self(batch, stage="validate")["loss"]

    def configure_optimizers(self):
        from stable_pretraining.optim import create_optimizer, create_scheduler
        named_params = [(name, parameter) for name, parameter in self.named_parameters() if parameter.requires_grad]
        optimizer = create_optimizer([parameter for _, parameter in named_params], self.config["optimizer"],
                                     named_params=named_params)
        scheduler = create_scheduler(optimizer, self.config["scheduler"], module=self)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}

    def on_save_checkpoint(self, checkpoint):
        checkpoint["upper_jepa_metadata"] = {
            "version": 1, "horizon": self.model.horizon, "history_size": self.model.history_size,
            "use_action_delta": self.model.use_action_delta, "actions_normalized": False,
            "normalization_stats_path": "process_stats.npz", "statistics_embedded": True,
            "observation_preprocessing": "ImageNet images; train-stat standardized proprio",
            "target_encoder": "shared encoder; no stop-gradient", "latent_dims": [192, 192, 32],
        }


def load_upper_checkpoint(path, map_location="cpu"):
    """Rebuild architecture and preprocessing from embedded stats, without HDF5.

    Returned module is in eval mode. module.observation_transform accepts raw
    images/proprio; module.model.build_macro_action accepts physical actions.
    No external statistics path is needed after copying/moving the checkpoint.
    """
    module = UpperTrainingModule.load_from_checkpoint(path, map_location=map_location, weights_only=True)
    module.model._validate_statistics()
    return module.eval()
