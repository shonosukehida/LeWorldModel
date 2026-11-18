"""Compare recorded future frames to standalone DIAMOND rollouts, by view and horizon."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw
import torch

from diamond_dataset import DiamondDataset, inspect_file
from stable_worldmodel.wm.diamond.checkpoint import load_for_inference
from stable_worldmodel.wm.diamond.preprocessing import pack_views, unpack_views


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def rgb(tensor):
    return Image.fromarray(tensor.permute(1, 2, 0).contiguous().cpu().numpy())


def panel(columns, views):
    """Column labels describe image sources; row labels identify cameras."""
    sample = next(iter(columns[0][1].values()))
    h, w = sample.shape[-2:]
    result = Image.new("RGB", (80 + w*len(columns), 20 + h*len(views)))
    draw = ImageDraw.Draw(result)
    for j, (label, images) in enumerate(columns):
        x = 80 + j*w
        draw.text((x+2, 2), label, fill="white")
        for i, k in enumerate(views):
            y = 20 + i*h
            draw.text((2, y+2), "Overhead" if k == "pixels" else "Wrist", fill="white")
            result.paste(rgb(images[k]), (x, y))
    return result


def at(images, i):
    return {k:v[i] for k, v in images.items()}


def mse(a, b):
    return (a.float()-b.float()).square().mean().item()


def save_video(path, frames, fps):
    import cv2
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, frames[0].size)
    if not writer.isOpened():
        raise RuntimeError("Unable to open MP4 writer")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(np.asarray(frame), cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def evaluate(args):
    if args.horizon < 1 or min(args.starts) < 0 or args.num_episodes < 1 or args.display_fps <= 0:
        raise ValueError("Invalid horizon, start or episode count")
    device = torch.device(args.device)
    model, meta = load_for_inference(args.checkpoint, device)
    views = meta["spec"]["image"]["views"]
    path = Path(args.dataset or meta["config"]["dataset"]).expanduser()
    identity, episodes = inspect_file(path, views)
    if identity != meta["dataset"]:
        raise ValueError("Evaluation expects the dataset identified by the checkpoint split")
    eligible = [ep for ep in meta["split"]["val"] if episodes[ep][1] >= 4+args.horizon+max(args.starts)]
    selected = args.episodes if args.episodes is not None else eligible[:args.num_episodes]
    if not selected or not set(selected) <= set(eligible):
        raise ValueError("Select held-out episodes with complete evaluation windows")
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    records, deltas, windows = [], [], []
    for ep in selected:
        ds = DiamondDataset(path, episodes, [ep], meta["normalizer"], views, frames=4+args.horizon)
        for start in args.starts:
            target_dir = out/f"ep-{ep:03d}-start-{start:04d}"
            target_dir.mkdir()
            raw, actions = ds.read(start)
            packed = pack_views(raw, views)
            obs = packed[:4].unsqueeze(0).to(device)
            past = actions[:3].unsqueeze(0).to(device)
            future = actions[3:].unsqueeze(0).to(device)
            seed = args.seed + ep*1000003 + start
            def predict(f):
                gen = torch.Generator(device=device).manual_seed(seed)
                sync(device)
                begin = time.perf_counter()
                result = model.rollout_images(obs, past, f, gen)
                sync(device)
                return result[0].cpu(), time.perf_counter()-begin
            pred, elapsed = predict(future)
            windows.append(dict(episode=ep, start=start, shape=list(pred.shape), seconds=elapsed,
                                finite=bool(pred.isfinite().all()), minimum=pred.min().item(), maximum=pred.max().item()))
            gt = packed[4:]
            gt_views, pred_views = unpack_views(gt, views), unpack_views(pred, views)
            hw = raw[views[0]].shape[-2:]
            full_pred = unpack_views(pred, views, hw)
            raw_gt = {k:v[4:].round().byte() for k,v in raw.items()}
            variants = {}
            if args.diagnostics:
                # A seeded non-identity permutation when H>1; H=1 has only the identity.
                perm = torch.randperm(args.horizon, generator=torch.Generator().manual_seed(seed))
                if args.horizon > 1 and torch.equal(perm, torch.arange(args.horizon)):
                    perm = perm.roll(1)
                other = next((i for i in eligible if i != ep), None)
                if other is None:
                    raise ValueError("Another eligible validation episode is needed for diagnostics")
                alt = DiamondDataset(path, episodes, [other], meta["normalizer"], views, frames=4+args.horizon)
                other_actions = alt.read_actions(start)
                alt.close()
                windows[-1].update(other_episode=other, permutation=perm.tolist(), noise_seed=seed)
                for name, changed in (("shuffle", future[:, perm]), ("norm-zero", torch.zeros_like(future)),
                                      ("other-ep", other_actions[None, 3:].to(device))):
                    prediction, _ = predict(changed)
                    variants[name] = unpack_views(prediction, views)
                    for t in range(args.horizon):
                        for j,k in enumerate(views):
                            chan = slice(j*3, (j+1)*3)
                            deltas.append(dict(episode=ep, start=start, step=t+1, view=k, variant=name,
                                               pred_delta_mse=mse(prediction[t,chan]/2, pred[t,chan]/2)))
            video = []
            for t in range(args.horizon):
                prefix = f"t{t+1:03d}"
                columns = [("GT 64", at(gt_views, t)), ("Pred 64", at(pred_views, t))]
                comparison = panel(columns, views)
                comparison.save(target_dir/f"{prefix}-64.png")
                video.append(comparison)
                panel([("Original GT", at(raw_gt,t)), ("Restored prediction", at(full_pred,t))],
                      views).save(target_dir/f"{prefix}-original.png")
                if variants:
                    panel([columns[1]]+[(name, at(images,t)) for name,images in variants.items()],
                          views).save(target_dir/f"{prefix}-actions.png")
                for j,k in enumerate(views):
                    if t == 0:
                        for name, tensor in (("gt", gt_views[k][0]), ("prediction", pred_views[k][0])):
                            rgb(tensor).save(target_dir/f"next-{k}-{name}.png")
                    chan = slice(j*3, (j+1)*3)
                    records.append(dict(episode=ep, start=start, step=t+1, view=k,
                                        mse64=mse(gt[t,chan]/2, pred[t,chan]/2),
                                        mse_original=mse(raw_gt[k][t].float()/255, full_pred[k][t].float()/255)))
            if args.video:
                save_video(target_dir/"comparison.mp4", video, args.display_fps)
        ds.close()
    per_step = []
    for t in range(1,args.horizon+1):
        for k in views:
            rows = [r for r in records if r["step"]==t and r["view"]==k]
            per_step.append(dict(step=t, view=k, **{m:np.mean([r[m] for r in rows]) for m in ("mse64","mse_original")}))
    result = dict(checkpoint=str(Path(args.checkpoint).resolve()), global_step=meta["global_step"],
                  dataset=identity["id"], split="val", config=vars(args), windows=windows, per_window=records,
                  per_step=per_step, mse64_one_step=np.mean([r["mse64"] for r in records if r["step"]==1]),
                  mse64_rollout=np.mean([r["mse64"] for r in records]), action_diagnostics=deltas,
                  units="pixel MSE on [0,1]; diagnostics compare predictions using shared noise; FPS is playback only")
    (out/"metrics.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    print(out/"metrics.json", result["mse64_one_step"], result["mse64_rollout"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dataset", help="Relocated copy of the checkpoint dataset")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument("--starts", type=int, nargs="+", default=[0])
    parser.add_argument("--episodes", type=int, nargs="+")
    parser.add_argument("--num-episodes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--diagnostics", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--display-fps", type=float, default=4.0)
    evaluate(parser.parse_args())
