"""Instance-local GPC recording; no model inference or robot dependencies.

NPZ candidate_actions are in DP training normalization space. Red curves are
selected target sequences, never measured or safety-limited robot trajectories.
"""
from pathlib import Path
import time

import numpy as np
import torch


def evaluation_ranks(rewards):
    """Rank 1 is best; ties use candidate index (as torch.argmax does)."""
    rewards = np.asarray(rewards)
    if rewards.ndim != 1 or not np.isfinite(rewards).all():
        raise ValueError("Expected finite one-dimensional rewards")
    order = np.argsort(-rewards, kind="stable")
    ranks = np.empty(len(order), dtype=np.int64)
    ranks[order] = np.arange(1, len(order) + 1)
    return ranks


def plot_candidates(actions, rewards, best_idx, step_idx, save_path=None):
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, Normalize
    from matplotlib.cm import ScalarMappable

    ranks = evaluation_ranks(rewards)
    k, horizon, dim = actions.shape
    if dim != 8 or ranks[best_idx] != 1:
        raise ValueError("Expected 8D Cartesian candidates and argmax selection")
    cmap = ListedColormap(plt.get_cmap("Greys")(np.linspace(.95, .25, 256)))
    norm = Normalize(vmin=1, vmax=max(k, 2))
    fig, axes = plt.subplots(3, 1, sharex=True, figsize=(10, 9), layout="constrained")
    for axis_idx, ax in enumerate(axes):
        for candidate_idx in range(k):
            if candidate_idx != best_idx:
                ax.plot(np.arange(horizon), actions[candidate_idx, :, axis_idx],
                        color=cmap(norm(ranks[candidate_idx])), linewidth=1)
        ax.plot(np.arange(horizon), actions[best_idx, :, axis_idx],
                color="#8B0000", linewidth=2.8, label="WM selected target sequence")
        ax.set_ylabel(f"{'XYZ'[axis_idx]} (normalized)")
        ax.set_ylim(-1, 1)
        ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
        ax.yaxis.set_major_formatter("{x:.1f}")
        ax.grid(alpha=.2)
        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), ax=ax)
        cb.set_label("WM Evaluation Rank")
        cb.set_ticks([1] if k == 1 else [1, k])
        cb.ax.invert_yaxis()
    axes[0].legend()
    axes[-1].set_xlabel("Prediction time (step)")
    fig.suptitle(f"Step {step_idx} | num_candidates={k} | selected index={best_idx} (Rank 1)\n"
                 "DP-normalized target actions; not measured robot trajectories")
    if save_path is not None:
        try:
            fig.savefig(save_path, dpi=150)
        finally:
            plt.close(fig)
    return fig, axes


class GPCRecorder:
    """Wrap only this GPC instance and its DP; restore exact instance attributes.

    begin_step must precede each control tick. No reward call means no record,
    including reinforcement holds. CPU transfer is synchronous and timed.
    """
    def __init__(self, policy):
        self.policy = policy
        self.records = []
        self.step_idx = None
        self.pending = None
        self._originals = []
        self.capture_seconds = 0.0
        self._sample_seconds = 0.0
        dp = policy.diffusion_policy
        self.action_key = next((key for key in ("action_cartesian", "action", "action_joint")
                                if key in dp.process), None)
        if self.action_key is None:
            raise ValueError("DP action normalization processor is required")
        self.processor = dp.process[self.action_key]

    def _wrap(self, obj, name, wrapper):
        own = name in vars(obj)
        value = vars(obj).get(name)
        original = getattr(obj, name)
        self._originals.append((obj, name, own, value))
        setattr(obj, name, wrapper(original))

    def __enter__(self):
        if self._originals:
            raise RuntimeError("Recorder already installed")
        try:
            self._wrap(self.policy.diffusion_policy, "sample_action_sequences", self._sample_wrapper)
            self._wrap(self.policy, "reward_fn", self._reward_wrapper)
        except BaseException:
            self.close()
            raise
        return self

    def begin_step(self, step_idx):
        self.step_idx = int(step_idx)
        self.pending = None

    def _sample_wrapper(self, original):
        def sample(*args, **kwargs):
            self.pending = None
            result = original(*args, **kwargs)
            started = time.perf_counter()
            if self.step_idx is None:
                raise RuntimeError("begin_step must precede GPC inference")
            if result.ndim != 4 or result.shape[0] != 1 or result.shape[-1] != 8:
                raise ValueError("Real-robot recording requires (1, K, T, 8)")
            start = self.policy.diffusion_policy.obs_horizon - 1
            end = min(start + self.policy.action_horizon, result.shape[2])
            if end <= start:
                raise ValueError("Empty GPC action horizon")
            self.pending = result[:, :, start:end].detach().cpu().numpy().copy()[0]
            self._sample_seconds = time.perf_counter() - started
            self.capture_seconds += self._sample_seconds
            return result
        return sample

    def _reward_wrapper(self, original):
        def reward(*args, **kwargs):
            result = original(*args, **kwargs)
            started = time.perf_counter()
            if self.pending is None or result.shape != (1, len(self.pending)):
                raise ValueError("Reward/candidate recording mismatch")
            best = int(torch.argmax(result, dim=1).detach().cpu()[0])
            values = result.detach().cpu().numpy().copy()[0]
            self.records.append(dict(step_idx=self.step_idx,
                                     candidate_actions_physical=self.pending,
                                     rewards=values, best_idx=best,
                                     num_candidates=len(values),
                                     action_horizon=self.pending.shape[1]))
            self.pending = None
            elapsed = time.perf_counter() - started
            self.records[-1]["capture_seconds"] = self._sample_seconds + elapsed
            self.capture_seconds += elapsed
            return result
        return reward

    def close(self):
        for obj, name, own, value in reversed(self._originals):
            if own:
                setattr(obj, name, value)
            else:
                delattr(obj, name)
        self._originals.clear()
        self.pending = None

    def __exit__(self, *exc):
        self.close()

    def save(self, run_dir):
        """Normalize with the loaded DP processor and persist before plotting."""
        root = Path(run_dir) / "gpc"
        (root / "data").mkdir(parents=True, exist_ok=True)
        (root / "gpc_actions").mkdir(exist_ok=True)
        payload = dict(normalization_space=np.array("DP training normalization"),
                       action_key=np.array(self.action_key),
                       configured_action_horizon=np.array(self.policy.action_horizon),
                       step_idx=np.array([r["step_idx"] for r in self.records], dtype=np.int64),
                       capture_seconds=np.array(self.capture_seconds),
                       record_count=np.array(len(self.records)))
        # Named arrays support variable K/H without pickle or lost step mapping.
        for index, record in enumerate(self.records):
            physical = record["candidate_actions_physical"]
            normalized = np.asarray(self.processor.transform(
                physical.reshape(-1, physical.shape[-1]).copy())).reshape(physical.shape)
            record["candidate_actions"] = normalized
            prefix = f"record_{index:04d}_"
            for key, value in record.items():
                payload[prefix + key] = np.asarray(value)
        np.savez_compressed(root / "data" / "candidates.npz", **payload)
        for record in self.records:
            plot_candidates(record["candidate_actions"], record["rewards"],
                            record["best_idx"], record["step_idx"],
                            root / "gpc_actions" / f"step_{record['step_idx']:04d}.png")
        return root


def render_archive(archive_path, output_dir=None):
    """Regenerate PNGs from saved normalized data without DP/WM or a checkpoint."""
    archive_path = Path(archive_path)
    output_dir = Path(output_dir) if output_dir else archive_path.parent.parent / "gpc_actions"
    output_dir.mkdir(parents=True, exist_ok=True)
    with np.load(archive_path, allow_pickle=False) as data:
        if str(data["normalization_space"]) != "DP training normalization":
            raise ValueError("Archive must contain DP-normalized actions")
        for index in range(int(data["record_count"])):
            prefix = f"record_{index:04d}_"
            step = int(data[prefix + "step_idx"])
            plot_candidates(data[prefix + "candidate_actions"], data[prefix + "rewards"],
                            int(data[prefix + "best_idx"]), step,
                            output_dir / f"step_{step:04d}.png")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    render_archive(args.archive, args.output_dir)
