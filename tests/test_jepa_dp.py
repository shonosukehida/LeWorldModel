"""Synthetic and real JEPA integration, plus exact legacy DP regression."""
import ast
from copy import deepcopy
from pathlib import Path
import h5py
import numpy as np
import pytest
import torch
from torch import nn
from transformers import ViTConfig, ViTModel
from omegaconf import OmegaConf
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from jepa import JEPA
from jepa_dp import (create_jepa_run_dir, fingerprint, load_pretrained_encoder,
                     load_jepa_policy, verify_jepa_source)
from stable_worldmodel.diffusion import ConditionalUnet1D, create_obs_encoder
from stable_worldmodel.diffusion.jepa import JEPADiffusionPolicy, EmptyWristEncoder
from stable_worldmodel.policy import DiffusionPolicy, BasePolicy
from train_policy import build_optimizer, evaluate, save_checkpoint, train_one_epoch
from normalization_stats import build_normalization_process, save_normalization_process
from offline_dp_eval import evaluate_episode, evaluate_multi_sample, evaluate_chunk_rollout

ROOT = Path(__file__).resolve().parents[1]
REAL = Path('/home/shonosukehida/.stable_worldmodel')
RUN = REAL/'checkpoints/flip_mug/ep100_tm300_multiview_play_normal_brightness/seed_41/lewm20261103_155756'
STATS = REAL/'datasets/ep100_tm300_multiview_play_normal_brightness/process_stats.npz'


def cfg():
    return OmegaConf.create(dict(encoder=dict(type='jepa'),
        optimizer=dict(lr=1e-3, weight_decay=1e-6),
        policy=dict(action_key='action_cartesian', pred_horizon=8, obs_horizon=2, action_horizon=3),
        model=dict(diffusion_step_embed_dim=16, down_dims=[16,32], kernel_size=3,n_groups=8, cond_predict_scale=True),
        diffusion=dict(num_train_timesteps=4, beta_schedule='linear', prediction_type='epsilon', clip_sample=True)))


@pytest.fixture
def synthetic(tmp_path):
    torch.set_num_threads(2)
    def vit():
        return ViTModel(ViTConfig(image_size=16,patch_size=8,hidden_size=24,
            num_hidden_layers=3,num_attention_heads=4,intermediate_size=32,
            hidden_dropout_prob=0.1), use_mask_token=True)
    world = JEPA(vit(), vit(), nn.Linear(7,7), nn.Linear(8,7), nn.Linear(8,5),
                 nn.Sequential(nn.Linear(24,7),nn.BatchNorm1d(7)), nn.Linear(24,7), nn.Linear(7,7))
    torch.save(world, tmp_path/'world_object.ckpt')
    OmegaConf.save(OmegaConf.create(dict(img_size=16,wm=dict(embed_dim=7,prop_embed_dim=5,prop_dim=8))),tmp_path/'config.yaml')
    class Columns:
        def get_col_data(self, key): return np.arange(80,dtype=np.float32).reshape(10,8)/10
    process, key = build_normalization_process(Columns(), ['action_cartesian','proprio'])
    save_normalization_process(tmp_path/'process_stats.npz',process,key)
    settings=dict(checkpoint=str(tmp_path/'world_object.ckpt'),world_config=str(tmp_path/'config.yaml'),process_stats=str(tmp_path/'process_stats.npz'))
    return load_pretrained_encoder(settings), settings


def make_policy(encoder, source):
    config=cfg()
    p=JEPADiffusionPolicy(ConditionalUnet1D(8,global_cond_dim=encoder.output_shape()[0]*2,**dict(config.model)),
        encoder,DDPMScheduler(**dict(config.diffusion)),8,2,3,8,num_inference_steps=2,wrist_obs_encoder=EmptyWristEncoder())
    p.jepa_source=source
    return p


def observations(encoder):
    obs=dict(pixels=torch.randn(2,2,3,encoder.image_size,encoder.image_size),
             wrist_pixels=torch.randn(2,2,3,encoder.image_size,encoder.image_size),proprio=torch.randn(2,2,8))
    return obs


def roundtrip(encoder,source,tmp_path):
    config=cfg()
    p=make_policy(encoder, source)
    obs=observations(encoder)
    expected=encoder(obs).clone()
    raw=dict(obs,proprio=(obs['proprio']-encoder.proprio_mean)/encoder.proprio_std)
    encoded=encoder.world_model.encode(raw)
    assert torch.equal(expected,torch.cat([encoded[k] for k in ('overhead_img_emb','wrist_img_emb','prop_emb')],-1))
    cond,b,t=p._encode_observation_info(obs)
    assert (b,t)==(2,2) and cond.shape==(2,2*sum(encoder.feature_dims))
    before={k:v.clone() for k,v in encoder.state_dict().items()}
    optimizer=build_optimizer(p, config)
    assert {id(x) for g in optimizer.param_groups for x in g['params']}=={id(x) for x in p.model.parameters()}
    initial=next(p.model.parameters()).clone()
    loss,_=train_one_epoch(p,[dict(obs,action=torch.randn(2,8,8))],optimizer,'cpu',1.,0,0)
    assert np.isfinite(loss) and not torch.equal(initial,next(p.model.parameters()))
    assert not encoder.training and all(not m.training for m in encoder.modules())
    assert all(not v.requires_grad and v.grad is None for v in encoder.parameters())
    assert all(torch.equal(v,encoder.state_dict()[k]) for k,v in before.items())
    ckpt=tmp_path/'jepa.ckpt'
    save_checkpoint(ckpt,p,optimizer,1,loss,torch.zeros(1,8),torch.ones(1,8),config)
    saved=torch.load(ckpt,weights_only=False)
    from test_dp_encoder_selection import loader
    loaded=loader()(ckpt,None,'cpu')
    assert torch.equal(expected,loaded.obs_encoder(obs))
    loaded.transform={}
    for method in ('predict_action','sample_action_sequences'):
        torch.manual_seed(123); a=getattr(p,method)(obs)
        torch.manual_seed(123); b=getattr(loaded,method)(obs)
        if isinstance(a,dict): assert all(torch.equal(a[k],b[k]) for k in a)
        else: assert torch.equal(a,b)
    broken=deepcopy(saved); broken['obs_encoder_state_dict'].pop(next(iter(broken['obs_encoder_state_dict'])))
    with pytest.raises(RuntimeError): load_jepa_policy(broken,ckpt,'cpu')
    with pytest.raises(KeyError): p.compute_loss(dict(pixels=obs['pixels'],action=torch.zeros(2,8,8)))
    return load_jepa_policy(saved,ckpt,'cpu')


def test_synthetic_training_checkpoint_all_dry_runs(synthetic,tmp_path):
    (encoder,source),_=synthetic
    assert encoder.feature_dims==(7,7,5)
    loaded=roundtrip(encoder,source,tmp_path)
    path=tmp_path/'episode.h5'
    with h5py.File(path,'w') as f:
        f['episode_idx']=np.zeros(4,dtype=np.int64); f['step_idx']=np.arange(4)
        for key in ('pixels','wrist_pixels'):
            f[key]=np.random.default_rng(42).integers(0,256,(4,3,16,16),dtype=np.uint8)
        f['proprio']=np.zeros((4,8),dtype=np.float32)
        actions=np.zeros((4,8),dtype=np.float32); actions[:,6]=1; f['action_cartesian']=actions
    evaluate_episode(loaded,path,0,tmp_path/'episode',{})
    evaluate_multi_sample(loaded,path,0,1,2,42,tmp_path/'multi',{})
    evaluate_chunk_rollout(loaded,path,0,2,42,tmp_path/'chunk',{})
    assert len(list(tmp_path.rglob('*.png')))==12
    with h5py.File(path,'a') as f: del f['proprio']
    with pytest.raises(ValueError,match='proprio'): evaluate_episode(loaded,path,0,tmp_path/'missing',{})


def test_reject_bad_source(synthetic,tmp_path):
    _,settings=synthetic
    for options, field in (({"mode": "invalid"}, "mode"), ({"lr": -1e-5}, "lr")):
        with pytest.raises(ValueError, match=f"encoder.finetune.{field}"):
            load_pretrained_encoder(dict(settings, finetune=options))
    torch.save({'state_dict':{}},tmp_path/'state.ckpt')
    with pytest.raises(ValueError,match='model object'): load_pretrained_encoder(dict(settings,checkpoint=str(tmp_path/'state.ckpt')))
    with pytest.raises(ValueError,match='process_stats'): load_pretrained_encoder(dict(settings,process_stats=None))
    config=OmegaConf.load(settings['world_config']); config.wm.embed_dim=99; OmegaConf.save(config,settings['world_config'])
    with pytest.raises(ValueError,match='architecture'): load_pretrained_encoder(settings)


@pytest.mark.parametrize(("mode", "n"), [("frozen", 3), ("partial", 1), ("partial", 2), ("full", 3)])
def test_finetune_step_and_restore(synthetic, tmp_path, monkeypatch, mode, n):
    _, settings = synthetic
    config = cfg()
    config.encoder.finetune = dict(mode=mode, unfreeze_last_n_blocks=n, lr=2e-4)
    world = torch.load(settings["checkpoint"], weights_only=False)
    with monkeypatch.context() as patch:
        patch.setattr(torch, "load", lambda *args, **kwargs: world)
        encoder, source = load_pretrained_encoder(dict(settings, finetune=config.encoder.finetune))
    assert {p.data_ptr() for p in world.parameters()}.isdisjoint(
        p.data_ptr() for p in encoder.parameters())
    policy = make_policy(encoder, source)
    optimizer = build_optimizer(policy, config)
    before = deepcopy(encoder.state_dict())
    unet_before = next(policy.model.parameters()).detach().clone()

    expected = set()
    cameras = ("world_model.overhead_encoder", "world_model.wrist_encoder")
    for name, param in encoder.named_parameters():
        if mode == "partial" and any(
            name.startswith(f"{view}.encoder.layer.{i}.") for view in cameras for i in range(3 - n, 3)
        ):
            expected.add(name)
        if mode == "full" and ".pooler." not in name and not name.endswith(".mask_token"):
            expected.add(name)
        assert param.requires_grad == (name in expected)
    for name in ("predictor", "action_encoder", "pred_proj"):
        assert isinstance(getattr(encoder.world_model, name), nn.Identity)
    assert [g["lr"] for g in optimizer.param_groups] == ([1e-3, 2e-4] if expected else [1e-3])
    assert all(g["weight_decay"] == config.optimizer.weight_decay for g in optimizer.param_groups)
    assert {id(p) for p in optimizer.param_groups[0]["params"]} == {id(p) for p in policy.model.parameters()}
    if expected:
        actual = {id(p) for p in optimizer.param_groups[1]["params"]}
        assert actual == {id(p) for k, p in encoder.named_parameters() if k in expected}

    obs = observations(encoder)
    batch = dict(obs, action=torch.randn(2, 8, 8))
    loss, _ = train_one_epoch(policy, [batch], optimizer, "cpu", 1., 0, 0)
    assert np.isfinite(loss) and not torch.equal(unet_before, next(policy.model.parameters()))
    for name, param in encoder.named_parameters():
        assert (param.grad is not None) == (name in expected)
    changed = {k for k, v in encoder.state_dict().items() if not torch.equal(before[k], v)}
    if mode == "full":
        assert "world_model.overhead_projector.1.num_batches_tracked" in changed
        for module in ("overhead_encoder", "wrist_encoder", "overhead_projector", "wrist_projector", "prop_encoder"):
            assert any(k.startswith(f"world_model.{module}.") for k in changed & expected)
    else:
        assert changed <= expected
    if mode == "partial":
        for view in cameras:
            for i in range(3 - n, 3):
                assert any(k.startswith(f"{view}.encoder.layer.{i}.") for k in changed)
        for vit in (encoder.world_model.overhead_encoder, encoder.world_model.wrist_encoder):
            assert [block.training for block in vit.encoder.layer] == [False] * (3 - n) + [True] * n
            assert not vit.embeddings.training
        assert not encoder.world_model.overhead_projector.training

    after = deepcopy(encoder.state_dict())
    evaluate(policy, [batch], "cpu")
    assert all(not module.training for module in encoder.modules())
    assert all(torch.equal(v, after[k]) for k, v in encoder.state_dict().items())
    feature = encoder(obs)
    assert not feature.requires_grad

    path = tmp_path / "policy.ckpt"
    save_checkpoint(path, policy, optimizer, 1, loss, torch.zeros(1, 8), torch.ones(1, 8), config)
    saved = torch.load(path, weights_only=False)
    assert saved["encoder_config"]["finetune"] == dict(config.encoder.finetune)
    assert saved["config"]["encoder"]["finetune"] == dict(config.encoder.finetune)
    assert saved["optimizer_state_dict"]["param_groups"] == optimizer.state_dict()["param_groups"]
    assert len(saved["optimizer_state_dict"]["state"]) == len(expected) + len(list(policy.model.parameters()))
    assert saved["jepa_source"] == source
    verify_jepa_source(source)

    from test_dp_encoder_selection import loader
    read = torch.load
    def read_policy_only(file, *args, **kwargs):
        assert Path(file) == path
        return read(file, *args, **kwargs)
    with monkeypatch.context() as patch:
        patch.setattr(torch, "load", read_policy_only)
        restored = loader()(path, None, "cpu")
    assert all(torch.equal(v, restored.obs_encoder.state_dict()[k]) for k, v in after.items())
    assert torch.equal(feature, restored.obs_encoder(obs))
    assert all(not p.requires_grad for p in restored.obs_encoder.parameters())
    restored.transform = {}
    for method in ("predict_action", "sample_action_sequences"):
        torch.manual_seed(123)
        a = getattr(policy, method)(obs)
        torch.manual_seed(123)
        b = getattr(restored, method)(obs)
        if isinstance(a, dict):
            assert all(torch.equal(a[k], b[k]) for k in a)
        else:
            assert torch.equal(a, b)


def test_frozen_exact_legacy_update_and_pickle(synthetic, tmp_path, monkeypatch):
    (encoder, source), _ = synthetic
    policy = make_policy(encoder, source)
    reference = deepcopy(policy)
    config = cfg()  # Legacy config has no finetune section.
    batch = dict(observations(encoder), action=torch.randn(2, 8, 8))
    @torch.no_grad()
    def old_forward(self, info):
        raw = {k: info[k].to(self.proprio_mean) for k in ("pixels", "wrist_pixels", "proprio")}
        raw["proprio"] = (raw["proprio"] - self.proprio_mean) / self.proprio_std
        return self.world_model.encode(raw)["emb"]
    with monkeypatch.context() as patch:
        patch.setattr(type(encoder), "train", lambda self, mode=True: nn.Module.train(self, False))
        patch.setattr(type(encoder), "forward", old_forward)
        old_optimizer = torch.optim.AdamW(reference.model.parameters(), **dict(config.optimizer))
        torch.manual_seed(42)
        baseline, _ = train_one_epoch(reference, [batch], old_optimizer, "cpu", 1., 0, 0)
    optimizer = build_optimizer(policy, config)
    torch.manual_seed(42)
    loss, _ = train_one_epoch(policy, [batch], optimizer, "cpu", 1., 0, 0)
    assert loss == baseline
    for name in ("model", "obs_encoder"):
        expected = getattr(reference, name).state_dict()
        assert all(torch.equal(v, expected[k]) for k, v in getattr(policy, name).state_dict().items())

    # Version 1 pickles written before fine-tuning have no mode attributes.
    del encoder.finetune_mode, encoder.unfreeze_last_n_blocks
    path = tmp_path / "legacy.ckpt"
    save_checkpoint(path, policy, optimizer, 1, loss, torch.zeros(1, 8), torch.ones(1, 8), config)
    saved = torch.load(path, weights_only=False)
    del saved["encoder_config"]["finetune"]
    restored = load_jepa_policy(saved, path, "cpu")
    assert torch.equal(restored.obs_encoder(batch), reference.obs_encoder(batch))


@pytest.mark.parametrize("n", [0, -1, 4, True, 1.5, "2"])
def test_invalid_partial_depth(synthetic, n):
    _, settings = synthetic
    with pytest.raises(ValueError, match="unfreeze_last_n_blocks"):
        load_pretrained_encoder(dict(settings, finetune=dict(mode="partial", unfreeze_last_n_blocks=n)))


@pytest.mark.parametrize("mode", ["frozen", "partial", "full"])
def test_hydra_training_run(synthetic, tmp_path, monkeypatch, mode):
    from hydra import compose, initialize_config_dir
    from train_policy import run
    (_, source), settings = synthetic
    dataset = tmp_path / "observations.h5"
    rng = np.random.default_rng(42)
    with h5py.File(dataset, "w") as f:
        f["ep_len"], f["ep_offset"] = [12], [0]
        for view in ("pixels", "wrist_pixels"):
            f[view] = rng.integers(256, size=(12, 16, 16, 3), dtype=np.uint8)
        for key in ("proprio", "action_cartesian"):
            f[key] = rng.standard_normal((12, 8)).astype("float32")
    cache = tmp_path / "cache"
    monkeypatch.setenv("STABLEWM_HOME", str(cache))
    overrides = [f"encoder.{key}={path}" for key, path in settings.items()]
    overrides += [
        "~encoder.finetune" if mode == "frozen" else f"encoder.finetune.mode={mode}",
        f"data.dataset.path={dataset}", "device=cpu", "wandb.enabled=false",
        "loader.num_workers=0", "loader.batch_size=2", "loader.pin_memory=false",
        "trainer.max_epochs=2", "model.down_dims=[16,32]", "model.diffusion_step_embed_dim=16",
        "model.kernel_size=3", "policy.pred_horizon=8", "policy.action_horizon=3",
        "diffusion.num_train_timesteps=4", "diffusion.num_inference_steps=2"]
    with initialize_config_dir(version_base=None, config_dir=str(ROOT / "config/train/policy")):
        config = compose(config_name="jepa_diffusion_policy", overrides=overrides)
    run.__wrapped__(config)
    directory, = [p.parent for p in cache.rglob("latest.ckpt")]
    latest = torch.load(directory / "latest.ckpt", weights_only=False)
    assert latest["epoch"] == 2
    assert latest["encoder_config"]["finetune"]["mode"] == mode
    assert all((directory / f).is_file() for f in ("best.ckpt", "epoch_001.ckpt", "epoch_002.ckpt",
                                                    "action_normalizer.pt"))
    saved_config = OmegaConf.load(directory / "config.yaml")
    assert saved_config.encoder.finetune.mode == mode
    assert saved_config.encoder.source_sha256 == {k: v["sha256"] for k, v in source.items()}
    verify_jepa_source(source)


def test_source_and_previous_runs_protected(synthetic, tmp_path):
    (encoder, source), settings = synthetic
    policy = make_policy(encoder, source)
    config = cfg()
    optimizer = build_optimizer(policy, config)
    def save(path):
        save_checkpoint(path, policy, optimizer, 0, 0., torch.zeros(1, 8), torch.ones(1, 8), config)

    for key, artifact in source.items():
        path = Path(settings[key])
        alias = tmp_path / f"{key}.symlink"
        link = tmp_path / f"{key}.hardlink"
        alias.symlink_to(path)
        link.hardlink_to(path)
        for dest in (path, alias, link):
            with pytest.raises(ValueError, match="protected"):
                save(dest)
        assert fingerprint(path) == artifact["sha256"]
    with pytest.raises(RuntimeError, match="changed"):
        verify_jepa_source(dict(source, checkpoint=dict(source["checkpoint"], sha256="incorrect")))

    old_run = create_jepa_run_dir(tmp_path / "run", policy)
    old = old_run / "best.ckpt"
    save(old)
    old_hash = fingerprint(old)
    new_run = create_jepa_run_dir(old_run, policy)
    assert new_run != old_run
    with pytest.raises(FileExistsError):
        save(old)
    best = new_run / "best.ckpt"
    best.symlink_to(old)
    with pytest.raises(FileExistsError):
        save(best)
    best.unlink()
    save(best)
    save(best)
    best.unlink()
    best.hardlink_to(old)
    with pytest.raises(FileExistsError):
        save(best)
    epoch = new_run / "epoch_001.ckpt"
    save(epoch)
    with pytest.raises(FileExistsError):
        save(epoch)
    assert fingerprint(old) == old_hash
    verify_jepa_source(source)


@pytest.mark.skipif(not (RUN/'lewm_epoch_10_object.ckpt').exists() or not STATS.exists(),reason='Local real JEPA artifacts unavailable')
def test_real_checkpoint_train_roundtrip(tmp_path):
    encoder,source=load_pretrained_encoder(dict(checkpoint=str(RUN/'lewm_epoch_10_object.ckpt'),world_config=str(RUN/'config.yaml'),process_stats=str(STATS)))
    assert encoder.feature_dims==(192,192,32)
    roundtrip(encoder,source,tmp_path)


@pytest.mark.parametrize('kind',['resnet18','conv'])
def test_legacy_exact_loss_actions_and_rng(kind):
    torch.set_num_threads(2)
    tree=ast.parse((ROOT/'tests/baseline_dp_chunks/policy.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DiffusionPolicy')
    ns=dict(BasePolicy=BasePolicy,torch=torch,np=np)
    exec(compile(ast.Module(body=[cls],type_ignores=[]),'baseline_dp_chunks','exec'),ns)
    enc=create_obs_encoder(kind,{'feature_dim':17} if kind=='conv' else {}).eval()
    model=ConditionalUnet1D(8,global_cond_dim=enc.output_shape()[0]*4,**dict(cfg().model)).eval()
    p=DiffusionPolicy(model,enc,DDPMScheduler(**dict(cfg().diffusion)),8,2,3,8,num_inference_steps=2,wrist_obs_encoder=deepcopy(enc))
    old=ns['DiffusionPolicy'](deepcopy(model),deepcopy(enc),DDPMScheduler(**dict(cfg().diffusion)),8,2,3,8,num_inference_steps=2,wrist_obs_encoder=deepcopy(enc))
    batch=dict(pixels=torch.randn(2,2,3,32,32),wrist_pixels=torch.randn(2,2,3,32,32),action=torch.randn(2,8,8))
    for method in ('compute_loss','predict_action','sample_action_sequences'):
        torch.manual_seed(42); a=getattr(old,method)(batch); state=torch.get_rng_state()
        torch.manual_seed(42); b=getattr(p,method)(batch)
        assert torch.equal(state,torch.get_rng_state())
        if isinstance(a,dict): assert all(torch.equal(a[k],b[k]) for k in a)
        else: assert torch.equal(a,b)
    # Compare actual training updates as well as forward/loss outputs.
    config = cfg()
    config.encoder.type = kind
    config.optimizer = {"lr": 1e-4, "weight_decay": 0.01}
    for policy in (old,p):
        for module in (policy.model,policy.obs_encoder,policy.wrist_obs_encoder): module.train()
        params=[v for module in (policy.model,policy.obs_encoder,policy.wrist_obs_encoder) for v in module.parameters()]
        optimizer = torch.optim.AdamW(params, lr=1e-4) if policy is old else build_optimizer(policy, config)
        torch.manual_seed(123)
        policy.compute_loss(batch).backward()
        optimizer.step()
    for name in ('model','obs_encoder','wrist_obs_encoder'):
        assert all(torch.equal(v,getattr(old,name).state_dict()[k]) for k,v in getattr(p,name).state_dict().items())
