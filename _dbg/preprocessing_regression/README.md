# Flip Mug preprocessing validation

Worktrees: LeWorldModel-add-prop (`real_robot_add_prop`, ea1d631),
stable-worldmodel-main (`main`, 9fc1f3f). No commit, push or retraining.

Changed production functions: extracted `scale_to_unit_range` in utils.py;
`get_img_preprocessor` uses that function; new `get_eval_img_preprocessor`
adapts the full training dictionary pipeline to a single image.
`img_transform` now uses that adapter for pixels/goal. `dp_img_transform`
preserves the prior diffusion policy pipeline. No wrist image keys added.
The existing policy-loading test mock was updated for the new diffusion factory.

Before: ToImage -> ToDtype(float32, scale=True) -> ImageNet Normalize -> Resize.
Float32 [0,255] did not scale. After: shared range conversion (float32,
[0,1] unchanged, [0,255] /255, invalid range raises) -> ImageNet Normalize -> Resize.
Training behavior is preserved. stable-worldmodel/main is unmodified;
_prepare_pixels applies the supplied transform once, with no extra scaling.
The evaluation dataset is constructed without a dataset transform.

Synthetic comparisons (same image, uint8 [0,255], float32 [0,255], float32 [0,1]):
all max_abs_diff=0, mean_abs_diff=0. Old float conversion range [0,255]; new [0,1].
All 10 image preprocessing/policy-loading tests passed. Full discovery additionally
encountered three existing hardware test import errors from missing digit_interface
and dynamixel_sdk. git diff --check passed.

Real sample: configured validation dataset val_ep10_tm300_prop/push.h5,
index 0, float32 [0,218]. Both outputs float32, min=-2.1179044247,
max=1.9919260740, mean=0.2486262172, std=0.8925757408;
max_abs_diff=mean_abs_diff=0.

Forward: ep200_tm300/lewm_object.ckpt, 16 one-step transitions on CPU,
using unmodified main ProbingEvaluator. Train and fixed pred MSE both
3.5058710575; old pred MSE 2.8514378071. Initial copied latent excluded from MSE.
current_z, true_z, pred_z each have max_abs_diff=mean_abs_diff=0 between
train and fixed. one_step_pca.png generated using main's native PCA plotter.
This verifies preprocessing equivalence and PCA generation, not model quality.

Limitation: configured training dataset ep200_tm300/push.h5 and corresponding
normalization statistics were absent. Configured validation data and available
ep2_tm300/process_stats.npz were used for all comparison paths. Absolute MSE
is for this explicit replacement setup, not the unavailable original setup.

Reproduce from /home/shonosukehida/work:

```bash
MPLCONFIGDIR=/tmp/lewm-mpl PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/home/shonosukehida/work/stable-worldmodel-main:/home/shonosukehida/work/LeWorldModel-add-prop:/home/shonosukehida/work/LeWorldModel-add-prop/tests \
LeWorldModel/.venv/bin/python -m unittest test_image_preprocessing test_policy_loading -v

MPLCONFIGDIR=/tmp/lewm-mpl PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=/home/shonosukehida/work/stable-worldmodel-main:/home/shonosukehida/work/LeWorldModel-add-prop \
LeWorldModel/.venv/bin/python LeWorldModel-add-prop/_dbg/check_world_model_preprocessing.py \
  --checkpoint /home/shonosukehida/.stable_worldmodel/checkpoints/flip_mug/ep200_tm300/lewm_object.ckpt \
  --dataset /home/shonosukehida/.stable_worldmodel/datasets/flip_mug/val_ep10_tm300_prop/push.h5 \
  --stats /home/shonosukehida/.stable_worldmodel/datasets/flip_mug/ep2_tm300/process_stats.npz
```
