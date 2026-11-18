import json
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import h5py
import numpy as np
from omegaconf import OmegaConf
import torch

from normalization_stats import SafeStandardScaler, save_normalization_process
from train_diamond import main
from test_diamond_checkpoint import identical
from stable_worldmodel.wm.diamond.checkpoint import read_checkpoint


def test_equivalent_epochs_and_partial_resume(tmp_path, monkeypatch):
    runs = []
    class Run:
        id = "mock-run"
        def __init__(self, **kw):
            self.args = kw
            self.rows = []
            self.axes = {}
            runs.append(self)
        def define_metric(self, name, step_metric):
            self.axes[name] = step_metric
        def log(self, row, step):
            assert step == row["global_step"]
            self.rows.append(row)
            torch.rand(3)
            np.random.rand()
            random.random()
        def finish(self):
            pass
    monkeypatch.setitem(sys.modules, "wandb", SimpleNamespace(init=Run, Image=lambda x, caption: x))
    path = tmp_path/"tiny.h5"
    rng = np.random.default_rng(42)
    actions = rng.normal(size=(16, 8)).astype("float32")
    with h5py.File(path,"w") as f:
        f["ep_len"], f["ep_offset"] = [8,8],[0,8]
        f["episode_idx"], f["step_idx"] = np.repeat([0,1],8), np.tile(np.arange(8),2)
        f["proprio"], f["action_cartesian"] = actions, actions
        for key in ("pixels","wrist_pixels"):
            f[key] = rng.integers(256,size=(16,3,12,16),dtype=np.uint8)
    save_normalization_process(tmp_path/"process_stats.npz", {"action_cartesian":SafeStandardScaler().fit(actions)}, "action_cartesian")
    cfg = OmegaConf.load(Path(__file__).resolve().parents[1]/"config/train/diamond.yaml")
    del cfg.hydra
    cfg.merge_with(dict(dataset=str(path),batch_size=2,steps=5,validate_every=5,save_every=5,
                        output=str(tmp_path/"off")))
    main.__wrapped__(cfg)
    cfg.merge_with(dict(steps=1,output=str(tmp_path/"part"),validate_every_epochs=2,
                        validation_mode="full",wandb=dict(enabled=True,log_images=True,image_every_epochs=2)))
    main.__wrapped__(cfg)
    cp = tmp_path/"part/step-000001.pt"
    assert runs[0].rows[0]["epoch_partial"]
    cfg.merge_with(dict(steps=5,output=str(tmp_path/"on"),resume=str(cp)))
    main.__wrapped__(cfg)
    rows = runs[0].rows + runs[1].rows
    assert runs[1].args["id"] == "mock-run" and runs[1].args["resume"] == "must"
    assert runs[1].axes["val/loss_epoch"] == runs[1].axes["train/loss_epoch"] == "epoch"
    assert [r["global_step"] for r in rows] == list(range(1,6))
    assert [r["epoch_progress"] for r in rows] == [.5,1.,1.5,2.,2.5]
    assert [r["global_step"] for r in rows if "val/loss_epoch" in r] == [1,4,5]
    assert len(rows[3]["val/predictions"]) == 2
    assert rows[1]["train/loss_epoch"] == np.mean([r["train/loss_step"] for r in rows[:2]])
    assert rows[4]["epoch_partial"] and rows[4]["epoch"] == 3
    logs = [json.loads(s) for s in (tmp_path/"off/metrics.jsonl").read_text().splitlines()]
    for r,l in zip(rows,logs,strict=True):
        assert (r["train/loss_step"],r["train/grad_norm"],r["train/learning_rate"]) == (l["loss"],l["grad_norm"],l["lr"])
        assert r["system/step_time_sec"] > 0
    a,b = [read_checkpoint(tmp_path/k/"step-000005.pt") for k in ("on","off")]
    for k in ("model","optimizer","scheduler","rng","global_step"):
        identical(a[k],b[k])
