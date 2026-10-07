"""Compare real-data preprocessing and one-step probing without robot execution."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

import numpy as np
import torch
import stable_worldmodel as swm
from omegaconf import OmegaConf
from normalization_stats import load_normalization_process
from utils import get_img_preprocessor
from test_image_preprocessing import load_transforms
from stable_worldmodel.probing.flip_mug.probe_evaluator import ProbingEvaluator
from stable_worldmodel.probing.flip_mug.plot import plot_one_step_rollout_pca


def stats(x):
    return dict(dtype=str(x.dtype), min=x.min().item(), max=x.max().item(),
                mean=x.float().mean().item(), std=x.float().std().item())


class TrainingPreprocessingProber(ProbingEvaluator):
    def _prepare_pixel_sequence(self, pixels, pixel_key="pixels"):
        transform = get_img_preprocessor(pixel_key, pixel_key, 224)
        return transform({pixel_key: torch.as_tensor(pixels)})[pixel_key].unsqueeze(0).to(self.device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Existing object checkpoint; never modified")
    parser.add_argument("--cache-dir", default="/home/shonosukehida/.stable_worldmodel")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--output", default=str(ROOT / "_dbg/preprocessing_regression"))
    args = parser.parse_args()
    torch.set_num_threads(4)
    cfg = OmegaConf.load(ROOT / "config/eval/flip_mug.yaml")
    factories = load_transforms()
    model = torch.load(args.checkpoint, map_location="cpu", weights_only=False).eval()
    history = ProbingEvaluator.resolve_history_size(cfg.eval.probing, model)
    dataset = swm.data.HDF5Dataset(
        cfg.eval.dataset_name, num_steps=history + 1,
        keys_to_cache=list(cfg.dataset.keys_to_cache), cache_dir=args.cache_dir,
    )
    process, _ = load_normalization_process(Path(args.cache_dir) / "datasets/flip_mug/ep100_tm300_multiview_play/process_stats.npz")
    sample = dataset[0]
    report = dict(checkpoint=args.checkpoint, history_size=history, images={})
    for key in ("pixels", "wrist_pixels"):
        x = sample[key]
        train = get_img_preprocessor(key, key, cfg.eval.img_size)({key: x.clone()})[key]
        probe = torch.stack([factories["img_transform"](cfg)(frame) for frame in x])
        diff = (train - probe).abs()
        report["images"][key] = dict(input=stats(x), output=stats(probe),
            max_abs_diff=diff.max().item(), mean_abs_diff=diff.mean().item())
        print(key, report["images"][key], flush=True)
        assert diff.max() < 1e-5
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for label, factory in [("fixed", factories["img_transform"]),
                           ("train", factories["img_transform"]),
                           ("old", factories["dp_img_transform"])]:
        transform = {key: factory(cfg) for key in ("pixels", "wrist_pixels", "goal", "goal_wrist_pixels")}
        prober_class = TrainingPreprocessingProber if label == "train" else ProbingEvaluator
        prober = prober_class(dataset, model, config=cfg.eval.probing, device="cpu",
                                 transform=transform, process=process, results_path=output)
        rollout = prober.collect_one_step_rollout_latents(max_horizon=args.samples)
        error = rollout["pred_z"] - rollout["true_z"]
        raw = float(np.mean(error ** 2))
        offset = float(np.mean(error.mean(axis=0) ** 2))
        report[label] = dict(samples=len(error), raw_mse=raw,
                            centered_mse=float(np.mean((error - error.mean(axis=0)) ** 2)),
                            offset_ratio=offset / raw)
        print(label, report[label], flush=True)
        if label == "fixed":
            fixed_rollout = rollout
            plot_one_step_rollout_pca(rollout, save_path=output / "train_one_step_pca.png")
            np.savez(output / "train_one_step_rollout_data.npz", **{k: v for k, v in rollout.items() if k != "targets"})
        if label == "train":
            for key in ("current_z", "true_z", "pred_z"):
                diff = np.abs(rollout[key] - fixed_rollout[key])
                report["train"][key + "_max_abs_diff"] = float(diff.max())
                assert diff.max() < 1e-5
    (output / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
