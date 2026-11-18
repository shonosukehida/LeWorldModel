"""Measure only DIAMOND rollout, including candidate batching and output assembly."""
import argparse
from itertools import product
import json
from pathlib import Path
import platform
import time

import numpy as np
import torch

from stable_worldmodel.wm.diamond.checkpoint import load_for_inference
from stable_worldmodel.wm.diamond.model import DiamondWorldModel


def measure(model, device, k, h, steps, microbatch, args):
    chunk = microbatch or k
    obs = torch.zeros(1, 4, 6, 64, 64, device=device)
    past = torch.zeros(1, 3, 8, device=device)
    actions = torch.zeros(1, k, h, 8, device=device)
    rng = torch.Generator(device=device).manual_seed(args.seed)
    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)
    def rollout():
        if chunk >= k:
            return model.rollout_images(obs, past, actions, rng, denoising_steps=steps)
        chunks = [model.rollout_images(obs, past, part, rng, denoising_steps=steps)
                  for part in actions.split(chunk, dim=1)]
        return torch.cat(chunks, dim=1)
    for _ in range(args.warmup):
        rollout()
    sync()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    times = []
    for _ in range(args.repeats):
        sync()
        start = time.perf_counter()
        result = rollout()
        sync()
        times.append((time.perf_counter()-start)*1000)
        assert result.shape == (1, k, h, 6, 64, 64)
        del result
    p50, p95, p99 = np.percentile(times, [50, 95, 99]).tolist()
    return dict(k=k, horizon=h, denoising_steps=steps, microbatch=chunk, samples_ms=times,
                mean_ms=float(np.mean(times)), p50_ms=p50, p95_ms=p95, p99_ms=p99,
                candidates_per_second=k*1000/np.mean(times),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type=="cuda" else None,
                peak_reserved_bytes=torch.cuda.max_memory_reserved(device) if device.type=="cuda" else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--k", type=int, nargs="+", default=[1, 5, 20, 50])
    parser.add_argument("--horizon", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--steps", type=int, nargs="+", default=[1, 3, 5])
    parser.add_argument("--microbatch", type=int, nargs="+", default=[0], help="0 runs all K together")
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    if min(*args.k,*args.horizon,*args.steps,args.repeats)<1 or min(*args.microbatch,args.warmup)<0:
        parser.error("Invalid sweep or repetition counts")
    device = torch.device(args.device)
    gpu = torch.cuda.get_device_properties(device) if device.type=="cuda" else None
    torch.manual_seed(args.seed)
    if args.checkpoint:
        model, _ = load_for_inference(args.checkpoint, device)
    else:
        model = DiamondWorldModel().eval().to(device)
    if model.spec["unet"]["img_channels"] != 6:
        raise ValueError("Benchmark expects both cameras")
    report = dict(environment=dict(torch=str(torch.__version__), cuda_build=torch.version.cuda,
                                   cuda_available=torch.cuda.is_available(), gpu=gpu.name if gpu else None,
                                   total_vram_bytes=gpu.total_memory if gpu else None, device=str(device),
                                   cpu=platform.processor() or platform.machine(), threads=torch.get_num_threads(),
                                   precision="fp32", resolution=[64,64], views=model.spec["image"]["views"]),
                  config=vars(args), results=[])
    for k, h, n in product(args.k, args.horizon, args.steps):
        for micro in dict.fromkeys(min(m or k,k) for m in args.microbatch):
            row = measure(model, device, k, h, n, micro, args)
            report["results"].append(row)
            print(f"K={k} H={h} N={n} batch={micro} p50={row['p50_ms']:.1f} ms", flush=True)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as f:
        json.dump(report, f, indent=2, allow_nan=False)


if __name__ == "__main__":
    main()
