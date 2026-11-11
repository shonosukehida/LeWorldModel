# GPC proprio cost weight

`gpc.lambda_prop` defaults to `1.0` when omitted. This changes only terminal
candidate scoring, not DP generation, WM inputs/rollout, action normalization,
execution stride, robot control, Encoder/Predictor, or checkpoint formats.

```yaml
gpc:
  lambda_prop: 1.0
```

The reward is `-(image_cost + lambda_prop * prop_cost)`. Finite nonnegative
numbers are accepted; `0..1` is the recommended range. NaN/Inf, negative values,
booleans and strings are rejected. At `1.0` the original all-dimension squared
error reduction is used, without splitting/recombining sums, for exact numerical
compatibility. The legacy two-argument `latent_goal_reward(output, info)` remains
valid for models with any latent dimension and needs no boundary information.

Configured real/offline GPC uses `LatentGoalReward`. Image/proprio dimensions come
from the final Linear layers in the current JEPA overhead projector, wrist
projector and proprio MLP. For the current WM they are 192+192 and 32. Weighted
scoring rejects unknown architectures or output dimensions inconsistent with the
inferred structure. Default full-state scoring remains available for unknown
architectures. It does not guess an image boundary of 384.

All three offline modes record `lambda_prop` and the available image boundary in
metrics metadata. With a known boundary, HDF5 outputs add `image_cost`, `prop_cost`
and `weighted_total_cost`, aligned with candidate rewards/indices. Existing fields
and figures remain supported. `weighted_total_cost` is exactly the negative of the
actual reward, so at weight 1 it may differ by floating-point rounding from adding
the two separately reduced diagnostic costs. The existing selected candidate index
fields remain authoritative.

When GPC candidate recording is enabled, `candidates.npz` additionally records the
same costs per inference record and `selected_candidate_index`; legacy `best_idx`
and action arrays are retained. The top-level archive includes `lambda_prop`.
Cost logging uses the existing reward output; no additional WM rollout or Encoder
call is introduced.

`latent_dist` validates the configured weighted reward. Its observed/terminal
latent distances and WM prediction error are still ordinary full-latent L2
metrics. They are not redefined as weighted costs. A different selection can
naturally change selected-candidate distance metrics.

## Offline comparison

Run from LeWorldModel, supplying the actual Demo dataset if it is stored elsewhere.
Pin both checkpoints and use identical candidate count/seed for the two runs:

```bash
WM="$HOME/.stable_worldmodel/checkpoints/flip_mug/ep100_tm300_multiview_play_normal_brightness/seed_41/lewm20261103_155756/lewm_epoch_10"
DP="$HOME/.stable_worldmodel/checkpoints/flip_mug/policy/diffusion_policy/ep200_tm300_multiview_demo/seed41/20261021_135400/latest.ckpt"
DATA="$HOME/.stable_worldmodel/datasets/flip_mug/ep200_tm300_multiview_demo/push.h5"
for WEIGHT in 1.0 0.0; do
  .venv/bin/python eval_real_robot.py --config-name flip_mug \
    policy_type=gpc policy="$WM" gpc.diffusion_checkpoint="$DP" \
    gpc.lambda_prop="$WEIGHT" \
    eval.normalization_stats_path="$HOME/.stable_worldmodel/datasets/flip_mug/ep100_tm300_multiview_play_normal_brightness/process_stats.npz" \
    eval.real_robot.execute=false eval.real_robot.dry_run=true \
    eval.real_robot.dry_run_mode=latent_dist \
    eval.real_robot.dry_run_dataset_path="$DATA" \
    eval.real_robot.dry_run_episode_idx=0 \
    eval.real_robot.dry_run_num_candidates=20 eval.real_robot.dry_run_seed=42 \
    eval.real_robot.dp_execute_steps=8 \
    eval.real_robot.output_dir="outputs/gpc_lambda_${WEIGHT}" \
    eval.probing.exe_probe=false
 done
```

These CLI commands use the current CUDA loader. They do not connect to hardware
because dry-run routing occurs before environment initialization. Replace
`latent_dist` with `chunk_rollout`, or `multi_sample` plus a valid stored
`eval.real_robot.dry_run_timestep`, to compare those modes. Output directories are
`outputs/gpc_lambda_<weight>/offline_gpc/<mode>/episode_.../`.

The above production CLI was not executed: the referenced Demo dataset is not
present in this workspace. Software tests use synthetic data and real GPC APIs.
Before robot trials, inspect rank changes and selected chunks offline. Weight zero
removes the direct proprio goal penalty; current proprio still conditions WM
predictions and the camera images can still encode robot posture.
