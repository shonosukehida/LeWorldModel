# GPC candidate-0 execution experiment

On `real_robot_multiview`, set `gpc.execute_first_candidate: true` in
`config/eval/flip_mug.yaml` (default: `false`). The setting is passed only to GPC;
`policy_type: diffusion` keeps its existing behavior. It also applies to dataset
GPC evaluation (`multi_sample`, `chunk_rollout`, `latent_dist`, and the IMLE-GPC
`episode` path).

GPC still generates K proposals, normalizes the same action interval, rolls out
all K with the WM, computes all rewards, validates their shape, and computes
`argmax`. Only the final index used to retrieve the physical action chunk changes.
The single-action API uses the first action of that same selected chunk. Robot
safety checks, buffering, and control periods are unchanged.

- `best_idx`: the first argmax of rewards, regardless of execution mode.
- `selected_candidate_index`: the actual executed candidate (0 in this mode).
- Offline chunk/latent files retain the existing plural
  `selected_candidate_indices` array for actual execution and add a `best_idx`
  array with one entry per evaluation. Multi-sample HDF5/JSON uses scalar indices.
- Real-robot candidate NPZ files retain `record_NNNN_best_idx` and always record
  `record_NNNN_selected_candidate_index`, even without modality cost diagnostics.
  The red curve denotes execution; WM ranks and the best index remain available.
  Old archives without the execution index still render with their saved best index.
- GPC policy instances also expose the last indices as detached `(B,)` tensors,
  avoiding a new CUDA-to-host synchronization in selection.

No extra stage timing or CUDA synchronization is added. Existing recorder capture
time remains recorder overhead, not a measurement of DP/WM/reward latency.

## Changed files

`stable-worldmodel`:

- `stable_worldmodel/policy.py`
- `tests/test_gpc_execute_first_candidate.py`

`LeWorldModel`:

- `config/eval/flip_mug.yaml`
- `eval_real_robot.py`
- `offline_gpc_eval.py`
- `offline_latent_dist_eval.py`
- `offline_imle_eval.py`
- `utils/gpc_visualization.py`
- `tests/test_gpc_first_candidate_integration.py`
- `tests/test_gpc_chunk_execute_steps.py`
- `tests/test_offline_gpc_eval.py`
- `tests/test_offline_latent_dist_eval.py`
- `docs/gpc_execute_first_candidate.md`

## Software validation

Final CPU runs (no robot connection or execution):

- `stable-worldmodel`: `test_gpc_execute_first_candidate.py` and
  `test_latent_goal_reward.py`: **13 passed, 28 subtests passed**.
- `LeWorldModel`: the ten files below: **131 passed, 1 skipped, 21 subtests passed**.
  This includes actual DP/WM checkpoints and dataset slices on CPU, exact
  pre-change action/DP artifact comparisons, all-candidate evaluation, execution
  indices, real-inference recorder rendering, and unchanged robot control/safety
  definitions. CUDA execution was not tested.

```sh
PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/gpc-mpl \
.venv/bin/python -m pytest -q \
  tests/test_gpc_first_candidate_integration.py \
  tests/test_offline_gpc_eval.py tests/test_gpc_visualization.py \
  tests/test_offline_gpc_chunk_rollout.py tests/test_gpc_chunk_execute_steps.py \
  tests/test_gpc_proprio_weight.py tests/test_offline_latent_dist_eval.py \
  tests/test_offline_dp_eval.py tests/test_dp_action_chunks.py tests/test_imle_proprio.py
```

A broader exploratory run also exercised `tests/test_imle_integration.py`.
Four existing tests fail in their shared fixture before reaching GPC:
`IMLE use_proprio=true requires checkpoint proprio statistics`. The same failure
was reproduced using `imle_checkpoint.py` from the pre-change HEAD. These tests
are not included in the clean final run above; their fixture was not changed.
