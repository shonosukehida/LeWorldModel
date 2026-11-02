"""Frozen episode evaluator before dp_execute_steps support (regression oracle)."""
from collections import deque
import json
from pathlib import Path
import h5py
import numpy as np
from offline_dp_eval import load_episode, compute_metrics

def evaluate_episode(policy, dataset_path, episode_id, output_dir, metadata):
    if policy.action_dim != 8 or 'action_cartesian' not in policy.process:
        raise ValueError('Offline evaluation requires an 8D action_cartesian checkpoint')
    if policy.obs_horizon < 1 or policy.action_horizon < 1 or policy.pred_horizon < policy.obs_horizon:
        raise ValueError('Invalid checkpoint horizons')
    data, steps, order_source = load_episode(dataset_path, episode_id)
    histories = {k: deque(maxlen=policy.obs_horizon) for k in ('pixels', 'wrist_pixels')}
    predicted = []
    for t in range(len(steps)):
        for key, history in histories.items():
            if not history:
                history.extend([data[key][t]] * policy.obs_horizon)
            else:
                history.append(data[key][t])
        info = {k: np.stack(v)[None] for k, v in histories.items()}
        # get_action selects predict_action[:, obs_horizon-1], i.e. current
        # observation's action in train_policy's same-row observation/action windows.
        # It also inverse-transforms using checkpoint statistics: do not do it twice.
        action = np.asarray(policy.get_action(info))
        if action.shape != (1, 8) or not np.isfinite(action).all():
            raise ValueError(f'Invalid DP output at step {steps[t]}: expected finite (1,8)')
        predicted.append(action[0].copy())
    predicted = np.stack(predicted)
    gt = data['action_cartesian']
    metrics = compute_metrics(predicted, gt)
    metadata = dict(metadata, dataset_path=str(Path(dataset_path).expanduser().resolve()),
                    episode_idx=int(episode_id), num_steps=len(steps), temporal_order=order_source,
                    alignment='observation t -> action_cartesian t; get_action uses obs_horizon-1',
                    model={k: int(getattr(policy, k)) for k in
                           ('pred_horizon', 'obs_horizon', 'action_horizon', 'action_dim', 'num_inference_steps')})
    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    with h5py.File(out / 'offline_rollout.h5', 'w') as f:
        for k, v in dict(predicted_action=predicted, ground_truth_action=gt, step_idx=steps,
                         episode_idx=np.full(len(steps), episode_id, dtype=np.int64)).items():
            f.create_dataset(k, data=v)
        f.attrs['metadata'] = json.dumps(metadata, allow_nan=False)
    (out / 'metrics.json').write_text(json.dumps(dict(metrics, metadata=metadata), indent=2, allow_nan=False) + '\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # for name, dims, labels, ylabel in (
    #     ('position', range(3), ('x', 'y', 'z'), 'Position [m]'),
    #     ('orientation', range(3, 7), ('qx', 'qy', 'qz', 'qw'), 'Quaternion component'),
    #     ('gripper', (7,), ('gripper',), 'Gripper (0=open, 1=closed)'),
    # ):
    #     fig, axes = plt.subplots(len(labels), 1, sharex=True, squeeze=False, figsize=(10, 2.6 * len(labels)))
    #     for ax, dim, label in zip(axes[:, 0], dims, labels):
    #         ax.plot(steps, predicted[:, dim], color='tab:blue', label='Predicted')
    #         ax.plot(steps, gt[:, dim], color='tab:orange', label='Ground Truth')
    #         ax.set_ylabel(ylabel)
    #         ax.set_title(label)
    #         ax.grid(alpha=0.3)
    #         ax.legend()
    #     axes[-1, 0].set_xlabel('Step')
    #     fig.tight_layout()
    #     fig.savefig(out / f'predicted_vs_gt_{name}.png')
    #     plt.close(fig)


    # Checkpointと同じ正規化方式・統計を使用
    processor = policy.process["action_cartesian"]
    predicted_norm = processor.transform(predicted.copy())
    gt_norm = processor.transform(gt.copy())

    plot_specs = (
        ("position", range(3), ("x", "y", "z"), "Position [m]"),
        ("orientation", range(3, 7), ("qx", "qy", "qz", "qw"), "Quaternion component"),
        ("gripper", (7,), ("gripper",), "Gripper (0=open, 1=closed)"),
    )

    for normalized, pred_values, gt_values in (
        (False, predicted, gt),
        (True, predicted_norm, gt_norm),
    ):
        for name, dims, labels, ylabel in plot_specs:
            fig, axes = plt.subplots(
                len(labels), 1, sharex=True, squeeze=False,
                figsize=(10, 2.6 * len(labels))
            )

            for ax, dim, label in zip(axes[:, 0], dims, labels):
                ax.plot(
                    steps, pred_values[:, dim],
                    color="tab:blue", label="Predicted"
                )
                ax.plot(
                    steps, gt_values[:, dim],
                    color="tab:orange", label="Ground Truth"
                )

                if normalized:
                    ax.axhline(
                        -1.0, color="red", linestyle="--",
                        linewidth=1.0, alpha=0.25,
                        label="Normalized -1"
                    )
                    ax.axhline(
                        1.0, color="red", linestyle="--",
                        linewidth=1.0, alpha=0.25,
                        label="Normalized +1"
                    )

                ax.set_ylabel(
                    "Normalized action value" if normalized else ylabel
                )
                ax.set_title(label)
                ax.grid(alpha=0.3)
                ax.legend()

            axes[-1, 0].set_xlabel("Step")
            fig.tight_layout()

            suffix = "_normalized" if normalized else ""
            fig.savefig(out / f"predicted_vs_gt_{name}{suffix}.png")
            plt.close(fig)



    return metrics

