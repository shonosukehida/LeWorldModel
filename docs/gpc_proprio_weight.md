# GPC multiview latent goal cost weights

`gpc.lambda_overhead`, `gpc.lambda_wrist` and `gpc.lambda_prop` each default to
`1.0` when omitted. Existing configurations that specify only `lambda_prop`
retain their scoring behavior, including the image-block reduction order. This changes only terminal
candidate scoring, not DP generation, WM inputs/rollout, action normalization,
execution stride, robot control, Encoder/Predictor, or checkpoint formats.

```yaml
gpc:
  lambda_overhead: 1.0
  lambda_wrist: 1.0
  lambda_prop: 1.0
```

The reward is
`-(lambda_overhead * overhead_cost + lambda_wrist * wrist_cost + lambda_prop * prop_cost)`.
Each cost is the squared L2 distance between the corresponding features in the
last WM prediction and the last goal latent. Finite nonnegative numbers are accepted. NaN/Inf, negative values,
booleans and strings are rejected. When all three weights are `1.0`, the original all-dimension squared
error reduction is used, without splitting/recombining sums, for exact numerical
compatibility. The legacy two-argument `latent_goal_reward(output, info)` remains
valid for models with any latent dimension and needs no boundary information.

Configured real/offline GPC uses `LatentGoalReward`. Image/proprio dimensions come
from the final Linear layers in the current JEPA overhead projector, wrist
projector and proprio MLP. For the current WM they are 192+192 and 32. Weighted
scoring rejects unknown architectures or output dimensions inconsistent with the
inferred structure. Default full-state scoring remains available for unknown
architectures. It does not guess an image boundary of 384.

All three offline modes record the three weights and inferred modality dimensions
in metrics metadata. With known boundaries, HDF5 outputs include `overhead_cost`,
`wrist_cost`, `prop_cost`, `weighted_total_cost` and the existing `image_cost`,
which is `overhead_cost + wrist_cost`, aligned with candidate rewards/indices. Existing fields
and figures remain supported. `weighted_total_cost` is exactly the negative of the
actual reward, so at weight 1 it may differ by floating-point rounding from adding
the separately reduced diagnostic costs (also for legacy proprio-only weighting). The existing selected candidate index
fields remain authoritative.

When GPC candidate recording is enabled, `candidates.npz` additionally records the
same costs per inference record and `selected_candidate_index`; legacy `best_idx`
and action arrays are retained. The top-level archive includes all three weights.
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
    gpc.lambda_overhead=1.0 gpc.lambda_wrist=1.0 gpc.lambda_prop="$WEIGHT" \
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

Software tests cover synthetic candidates through the real GPC APIs and CPU
evaluation with locally available production DP/WM checkpoints. The production
CLI above and physical robot trials are separate from these software tests.
Before robot trials, inspect rank changes and selected chunks offline. Weight zero
removes the direct proprio goal penalty; current proprio still conditions WM
predictions and the camera images can still encode robot posture.

## Multiview extension: changed files

Both repositories use branch `real_robot_multiview`.

In `stable-worldmodel`:

- `stable_worldmodel/reward.py`: independent weights, model-derived boundaries,
  modality costs and metadata; original public functions remain available.
- `tests/test_latent_goal_reward.py`: exact legacy reductions, dynamic boundaries,
  independent candidate ranking, invalid coefficients and dimension mismatches.

In `LeWorldModel`:

- `config/eval/flip_mug.yaml`: all three weights default to `1.0`.
- `eval_real_robot.py`, `offline_imle_eval.py`: propagate camera weights into
  configured GPC rewards for live and dataset inference.
- `offline_gpc_eval.py`, `offline_latent_dist_eval.py`: record modality costs in
  all three offline modes and validate the configured weighted reward.
- `utils/gpc_visualization.py`: preserve existing NPZ fields and add camera costs
  and weights per recording, plus top-level weights.
- `real_robot_latent_visualization.py`: include all configured weights in metadata.
- `tests/test_gpc_proprio_weight.py`: exercise camera weighting in HDF5/NPZ logs
  and selection through the actual GPC API with identical action proposals.
- `tests/test_gpc_chunk_execute_steps.py`: account for added metadata while
  retaining exact legacy artifact and PNG comparisons.
- `tests/test_imle_dispatch.py`: verify weight propagation and make the legacy
  image-conditioned test checkpoint configuration explicit.
- `docs/gpc_proprio_weight.md`: configuration, compatibility and validation notes.

No changes are made to candidate generation, WM rollout, normalization or action
execution implementations. Physical robot operation was not performed.

## Validation result (2026-11-16)

The final regression run completed with **109 passed, 2 failed, 30 subtests
passed**, including the production-checkpoint CPU evaluations. All new reward
and weighting integration tests passed. `git diff --check` passed in both repos.

```bash
PYTHONDONTWRITEBYTECODE=1 MPLCONFIGDIR=/tmp/gpc-matplotlib \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 .venv/bin/python -m pytest -q \
  -o cache_dir=/tmp/gpc-pytest-final \
  ../stable-worldmodel/tests/test_latent_goal_reward.py \
  tests/test_gpc_proprio_weight.py tests/test_offline_gpc_eval.py \
  tests/test_offline_gpc_chunk_rollout.py tests/test_offline_latent_dist_eval.py \
  tests/test_gpc_visualization.py tests/test_gpc_chunk_execute_steps.py \
  tests/test_imle_dispatch.py tests/test_real_robot_latent_visualization.py
```

The two failures are pre-existing historical source comparisons:

- `tests/test_offline_gpc_eval.py::test_existing_apis_and_robot_implementation_unchanged`
- `tests/test_offline_latent_dist_eval.py::test_existing_evaluators_and_robot_methods_unchanged`

Both compare `_run_xarm_task_with_env` against an older pinned revision. Both
also failed when the relevant source reads were replaced with pre-change HEAD
source, without altering the working tree. The current function's AST is
identical to pre-change HEAD. These tests were left intact.

Exact all-one results were checked at 416 dimensions and with alternate model
sizes, in float32, float64 and bfloat16. Legacy proprio-only reductions were
checked exactly. Independent camera/proprio weights change selection for fixed
candidates. HDF5/NPZ logs retain legacy fields, add the requested modality costs,
and store `weighted_total_cost` equal to the negative of the actual reward.
