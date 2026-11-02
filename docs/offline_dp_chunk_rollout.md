# Diffusion Policy chunk rollout dry run

From the LeWorldModel directory, use the existing checkpoint and dataset settings:

```bash
.venv311/bin/python eval_real_robot.py --config-name flip_mug \
  policy_type=diffusion eval.real_robot.dry_run=true \
  eval.real_robot.dry_run_mode=chunk_rollout \
  eval.real_robot.dry_run_num_candidates=20 eval.real_robot.dry_run_seed=42
```

Override `gpc.diffusion_checkpoint`, `eval.real_robot.dry_run_dataset_path`,
`eval.real_robot.dry_run_episode_idx`, and `eval.real_robot.output_dir` as needed.
`dry_run_timestep` is unused in this mode. Omitted `dry_run_mode` still selects
`episode`; the existing `multi_sample` mode is unchanged.

The checkpoint supplies all horizons. Inference runs every `action_horizon`
rows starting at the episode's actual first `step_idx`. Each inference uses
recorded observations (with first-frame padding), not predicted observations.
Only prediction indices `[obs_horizon - 1 : obs_horizon - 1 + action_horizon]`
are displayed. The sampler inverse-transforms once using checkpoint statistics;
no smoothing, clipping, or quaternion correction is applied to saved actions.

Each run creates a unique `offline_dp/chunk_rollout/episode_<id>_<run_id>/`
under the configured output directory, containing position, quaternion component,
and gripper PNGs, `chunk_rollout_metrics.json`, and `chunk_rollout.h5`.
GT is black; candidate 0 is blue; other candidates are light gray. Each chunk
is a separate line and dashed vertical lines mark chunk starts.

HDF5 `predicted_chunks` has shape `(chunks, candidates, action_horizon, 8)`.
Unused terminal values are NaN; `chunk_valid_mask` marks valid slots.
`chunk_step_idx` maps each slot to the stored step, using -1 for padding.
`step_idx` maps the full `ground_truth_actions` array. Metadata records the
checkpoint, action normalization, horizons, seed rule (`seed + chunk_index`),
actual chunk seeds, and valid step lists.

Metrics use candidate 0 without GT-based selection. They report count, mean,
max, and p95 separately within chunks and across boundaries, in m/step and
degree/step. Empty difference sets have count 0 and null summaries. Rotation
metrics normalize quaternion copies and account for q/-q equivalence; zero-norm
quaternions raise an error. Padding contributes to neither plots nor metrics.
These are action-command differences, not measured robot velocity/acceleration
or a guarantee of real-robot safety. Dry run returns before robot initialization.

CPU-only regression checks:

```bash
MPLCONFIGDIR=/tmp/codex-mpl .venv311/bin/python -m unittest discover -s tests -p 'test_offline_dp*.py'
```
