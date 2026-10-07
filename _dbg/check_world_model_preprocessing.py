"""Check real Flip Mug data and checkpoint with the production probing path.

Run with stable-worldmodel/main first on PYTHONPATH; never runs a robot or trains.
Required arguments keep dataset/checkpoint provenance explicit.
"""
import argparse
import ast
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'tests'))

import numpy as np
import torch
import stable_worldmodel as swm
from omegaconf import OmegaConf
from utils import get_img_preprocessor
from test_image_preprocessing import load_transforms
from stable_worldmodel.probing.flip_mug.probe_evaluator import ProbingEvaluator
from stable_worldmodel.probing.flip_mug.plot import plot_one_step_rollout_pca


def stats(x):
    return dict(dtype=str(x.dtype), min=x.min().item(), max=x.max().item(),
                mean=x.float().mean().item(), std=x.float().std().item())


class TrainingPreprocessingProber(ProbingEvaluator):
    def _prepare_pixels(self, pixels, pixel_key='pixels'):
        pixels = torch.as_tensor(pixels)
        if pixels.ndim == 4:
            pixels = pixels[-1]
        return get_img_preprocessor(pixel_key, pixel_key, 224)(
            {pixel_key: pixels})[pixel_key].unsqueeze(0).to(self.device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--stats', required=True)
    parser.add_argument('--samples', type=int, default=16)
    parser.add_argument('--output', default=str(ROOT / '_dbg/preprocessing_regression'))
    args = parser.parse_args()
    torch.set_num_threads(4)
    cfg = OmegaConf.load(ROOT / 'config/eval/flip_mug.yaml')
    factories = load_transforms()
    # Load only normalization helpers, avoiding robot/environment initialization.
    tree = ast.parse((ROOT / 'eval_real_robot.py').read_text())
    nodes = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))
             and n.name in ('SafeStandardScaler', 'load_normalization_process')]
    ns = dict(np=np, json=json, Path=Path)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), 'eval_real_robot.py', 'exec'), ns)
    process, _ = ns['load_normalization_process'](args.stats)
    model = torch.load(args.checkpoint, map_location='cpu', weights_only=False).eval()
    dataset = swm.data.HDF5Dataset(path=args.dataset, keys_to_cache=['action_cartesian', 'proprio'])
    sample = dataset[0]
    x = sample['pixels']
    train = get_img_preprocessor('pixels', 'pixels', cfg.eval.img_size)({'pixels': x.clone()})['pixels']
    probe = torch.stack([factories['img_transform'](cfg)(frame) for frame in x])
    diff = (train - probe).abs()
    report = dict(checkpoint=args.checkpoint, dataset=args.dataset, stats=args.stats,
                  swm_source=swm.__file__, input=stats(x), train_output=stats(train),
                  probe_output=stats(probe), max_abs_diff=diff.max().item(),
                  mean_abs_diff=diff.mean().item())
    print(json.dumps(report, indent=2), flush=True)
    assert diff.max() < 1e-5
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    rollouts = {}
    for label, factory in [('fixed', factories['img_transform']),
                           ('train', factories['img_transform']),
                           ('old', factories['dp_img_transform'])]:
        transform = {key: factory(cfg) for key in ('pixels', 'goal')}
        cls = TrainingPreprocessingProber if label == 'train' else ProbingEvaluator
        prober = cls(dataset, model, config=cfg.eval.probing, device='cpu',
                     transform=transform, process=process, results_path=output)
        rollout = prober.collect_one_step_rollout_latents(max_horizon=args.samples)
        rollouts[label] = rollout
        # Exclude copied initial latent, which is not a prediction.
        error = rollout['pred_z'][1:] - rollout['true_z'][1:]
        assert np.isfinite(error).all()
        report[label] = dict(transitions=len(error), pred_mse=float(np.mean(error ** 2)))
        print(label, report[label], flush=True)
    report['forward_differences'] = {}
    for key in ('current_z', 'true_z', 'pred_z'):
        diff = np.abs(rollouts['train'][key] - rollouts['fixed'][key])
        report['forward_differences'][key] = dict(max_abs_diff=float(diff.max()),
                                                mean_abs_diff=float(diff.mean()))
        assert diff.max() < 1e-5
    plot_one_step_rollout_pca(rollouts['fixed'], save_path=output / 'one_step_pca.png')
    report['pca_generated'] = (output / 'one_step_pca.png').is_file()
    (output / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
