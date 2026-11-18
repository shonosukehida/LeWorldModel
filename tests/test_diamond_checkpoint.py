"""CPU end-to-end checkpoint, continuation and strict metadata regression."""
from copy import deepcopy
from pathlib import Path

import h5py
import numpy as np
from omegaconf import OmegaConf
import pytest
import torch
import train_diamond

from normalization_stats import SafeStandardScaler, save_normalization_process
from train_diamond import main
from stable_worldmodel.wm.diamond.checkpoint import load_for_inference, read_checkpoint, validate


def identical(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            identical(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        for x, y in zip(a, b, strict=True):
            identical(x,y)
    else:
        assert a == b


def test_checkpoint_resume_and_sampling(tmp_path, monkeypatch):
    obs, act = torch.zeros(1,4,6,64,64), torch.zeros(1,4,8)
    def predict(net):
        return net.predict_next(obs, act, torch.Generator().manual_seed(3))
    before_save = {}
    original_save = train_diamond.save_checkpoint
    def observed_save(path, model, *args, **kwargs):
        before_save[path] = predict(model.eval())
        model.train()
        return original_save(path, model, *args, **kwargs)
    monkeypatch.setattr(train_diamond, "save_checkpoint", observed_save)
    data = tmp_path/"mini.h5"
    rng = np.random.default_rng(43)
    actions = rng.normal(size=(16,8)).astype(np.float32)
    with h5py.File(data, "w") as f:
        f["ep_len"], f["ep_offset"] = [8,8], [0,8]
        f["step_idx"], f["episode_idx"] = np.tile(np.arange(8),2), np.repeat([4,9],8)
        f["action_cartesian"], f["proprio"] = actions, actions
        for k in ("pixels", "wrist_pixels"):
            f[k] = rng.integers(256, size=(16,3,12,16), dtype=np.uint8)
    save_normalization_process(tmp_path/"process_stats.npz", {"action_cartesian":SafeStandardScaler().fit(actions)},
                               "action_cartesian")
    cfg = OmegaConf.load(Path(__file__).resolve().parents[1]/"config/train/diamond.yaml")
    del cfg.hydra
    cfg.merge_with(dict(dataset=str(data), batch_size=1, steps=2, save_every=1, validate_every=2,
                        output=str(tmp_path/"continuous"), warmup_steps=0))
    main.__wrapped__(cfg)
    orig = tmp_path/"continuous/step-000002.pt"
    cfg.merge_with(dict(resume=str(tmp_path/"continuous/step-000001.pt"), output=str(tmp_path/"resumed")))
    main.__wrapped__(cfg)
    resumed = tmp_path/"resumed/step-000002.pt"
    a, b = read_checkpoint(orig), read_checkpoint(resumed)
    for k in ("model", "optimizer", "scheduler", "rng", "epoch", "global_step", "spec", "normalizer"):
        identical(a[k], b[k])
    initial = torch.get_rng_state()
    model, meta = load_for_inference(orig)
    other, _ = load_for_inference(resumed, expected_normalizer=meta["normalizer"])
    assert torch.equal(torch.get_rng_state(), initial)
    assert torch.equal(before_save[orig], predict(model))
    assert torch.equal(predict(model), predict(other))
    assert torch.equal(torch.get_rng_state(), initial)
    with pytest.raises(ValueError, match="normalization mismatch"):
        load_for_inference(orig, expected_normalizer=dict(meta["normalizer"], mean=[0.]*8))
    for replace in ({"version":2}, {"alignment":{}}, {"normalizer":dict(meta["normalizer"], scale=[0]*8)}):
        with pytest.raises(ValueError):
            validate(dict(a, **replace))
    for name, value in (("resolution", [32,32]), ("views", ["pixels"])):
        spec = deepcopy(a["spec"])
        spec["image"][name] = value
        with pytest.raises(ValueError):
            validate(dict(a, spec=spec))
    bad = dict(a, model=dict(a["model"]))
    bad["model"].pop(next(iter(a["model"])))
    torch.save(bad, tmp_path/"bad.pt")
    with pytest.raises(RuntimeError, match="Missing key"):
        load_for_inference(tmp_path/"bad.pt")
