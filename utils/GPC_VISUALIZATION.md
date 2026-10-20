GPC candidate visualization
===========================

Enable with the Hydra override `gpc.visualization_enabled=true` on the existing
real-robot inference command. The default in `config/eval/flip_mug.yaml` is false.
The helper loads only for an enabled GPC instance; standalone DP and CEM are
unaffected. This feature does not establish whether WM improves success rates.

The recorder wraps `diffusion_policy.sample_action_sequences` and `policy.reward_fn`
on the selected instances. Each original runs once and its exact result is returned.
It copies `(1,K,T,8)` proposals after slicing `obs_horizon-1` through
`min(start+action_horizon,T)`, and records the existing `(1,K)` rewards and their
`torch.argmax`. Real-robot recording deliberately requires B=1 and D=8.
`begin_step` clears pending data; reinforcement holds do not produce duplicate
records. `close`/the context manager restores inherited methods and instance
attributes, including on exceptions. The rollout restores hooks before closing
the environment and generating visualizations.

During control, only synchronous CPU copies, argmax and in-memory recording run.
After the environment closes, the loaded DP processor's `transform` normalizes
flattened `(K*H,8)` physical candidates, then restores `(K,H,8)`. The processor is
selected in the same key order as DP inverse normalization: `action_cartesian`,
`action`, `action_joint`. In this entrypoint it is `SafeStandardScaler` loaded
from the DP checkpoint's `action_mean`/`action_std`. WM normalization statistics
are not used. No model is evaluated during saving or replotting.

Outputs under the existing `run_dir`:

- `gpc/data/candidates.npz`
- `gpc/gpc_actions/step_0000.png`, etc. Step numbers are control-loop indices;
  gaps during reinforcement holds are intentional.

NPZ loads with `allow_pickle=False`. Top-level `step_idx` maps record positions
to control ticks. `record_count`, `normalization_space`, `action_key`,
`configured_action_horizon`, and total `capture_seconds` describe the archive.
Each `record_0000_`, `record_0001_`, ... prefix contains:

- `step_idx`, `candidate_actions` (DP normalized, K/H/D), `rewards` (K),
  `best_idx` (zero based), `num_candidates`, `action_horizon` (actual sliced H).
- `candidate_actions_physical` for auditing against the GPC return before robot
  safety clipping or gripper reinforcement, and per-record `capture_seconds`.

Variable horizons/candidate counts can be stored without object arrays. Red is
the entire selected target sequence, not measured motion or final robot commands.
Stable descending reward sorting breaks ties by lower candidate index, matching
argmax. Rank 1 is darkest, Rank K is light gray; the chosen sequence is dark red.

Replot a saved archive without checkpoints or hardware:

```bash
.venv/bin/python utils/gpc_visualization.py /path/to/run_dir/gpc/data/candidates.npz
# Optional alternate PNG destination:
.venv/bin/python utils/gpc_visualization.py /path/to/candidates.npz --output-dir /tmp/gpc_replot
```

Hardware-free checks:

```bash
MPLCONFIGDIR=/tmp/gpc-mpl .venv/bin/python -m unittest discover -s tests -p test_gpc_visualization.py
MPLCONFIGDIR=/tmp/gpc-mpl .venv/bin/python -m unittest discover -s tests -p test_gripper_reinforcement.py
```

Tests execute the installed GPC class source with mock proposals/rollouts to
avoid optional robot/simulation imports. They also exercise the existing rollout
with mock hardware, including enabled/disabled capture and reinforcement holds.
The recorder retains candidates in RAM until shutdown. GPU-to-CPU copies may
synchronize CUDA; this environment has no available CUDA device, so 10 Hz timing
on the target GPU is not established. Per-record/total capture timings are saved
for measurement on that machine. Rendering and compressed archive writes happen
only after control ends. Existing `run_dir` naming and overwrite behavior apply.
