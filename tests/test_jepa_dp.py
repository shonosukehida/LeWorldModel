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
from jepa_dp import load_pretrained_encoder, load_jepa_policy
from stable_worldmodel.diffusion import ConditionalUnet1D, create_obs_encoder
from stable_worldmodel.diffusion.jepa import JEPADiffusionPolicy, EmptyWristEncoder
from stable_worldmodel.policy import DiffusionPolicy, BasePolicy
from train_policy import save_checkpoint, train_one_epoch
from normalization_stats import build_normalization_process, save_normalization_process
from offline_dp_eval import evaluate_episode, evaluate_multi_sample, evaluate_chunk_rollout

ROOT = Path(__file__).resolve().parents[1]
REAL = Path('/home/shonosukehida/.stable_worldmodel')
RUN = REAL/'checkpoints/flip_mug/ep100_tm300_multiview_play_normal_brightness/seed_41/lewm20261103_155756'
STATS = REAL/'datasets/ep100_tm300_multiview_play_normal_brightness/process_stats.npz'


def cfg():
    return OmegaConf.create(dict(encoder=dict(type='jepa'),
        policy=dict(action_key='action_cartesian', pred_horizon=8, obs_horizon=2, action_horizon=3),
        model=dict(diffusion_step_embed_dim=16, down_dims=[16,32], kernel_size=3,n_groups=8, cond_predict_scale=True),
        diffusion=dict(num_train_timesteps=4, beta_schedule='linear', prediction_type='epsilon', clip_sample=True)))


@pytest.fixture
def synthetic(tmp_path):
    torch.set_num_threads(2)
    def vit():
        return ViTModel(ViTConfig(image_size=16,patch_size=8,hidden_size=24,
            num_hidden_layers=1,num_attention_heads=4,intermediate_size=32))
    world = JEPA(vit(), vit(), nn.Identity(), nn.Identity(), nn.Linear(8,5),
                 nn.Sequential(nn.Linear(24,7),nn.BatchNorm1d(7)), nn.Linear(24,7))
    torch.save(world, tmp_path/'world_object.ckpt')
    OmegaConf.save(OmegaConf.create(dict(img_size=16,wm=dict(embed_dim=7,prop_embed_dim=5,prop_dim=8))),tmp_path/'config.yaml')
    class Columns:
        def get_col_data(self, key): return np.arange(80,dtype=np.float32).reshape(10,8)/10
    process, key = build_normalization_process(Columns(), ['action_cartesian','proprio'])
    save_normalization_process(tmp_path/'process_stats.npz',process,key)
    settings=dict(checkpoint=str(tmp_path/'world_object.ckpt'),world_config=str(tmp_path/'config.yaml'),process_stats=str(tmp_path/'process_stats.npz'))
    return load_pretrained_encoder(settings), settings


def roundtrip(encoder,source,tmp_path):
    config=cfg()
    p=JEPADiffusionPolicy(ConditionalUnet1D(8,global_cond_dim=encoder.output_shape()[0]*2,**dict(config.model)),
        encoder,DDPMScheduler(**dict(config.diffusion)),8,2,3,8,num_inference_steps=2,wrist_obs_encoder=EmptyWristEncoder())
    p.jepa_source=source
    obs=dict(pixels=torch.randn(2,2,3,encoder.image_size,encoder.image_size),
             wrist_pixels=torch.randn(2,2,3,encoder.image_size,encoder.image_size),proprio=torch.randn(2,2,8))
    expected=encoder(obs).clone()
    raw=dict(obs,proprio=(obs['proprio']-encoder.proprio_mean)/encoder.proprio_std)
    encoded=encoder.world_model.encode(raw)
    assert torch.equal(expected,torch.cat([encoded[k] for k in ('overhead_img_emb','wrist_img_emb','prop_emb')],-1))
    cond,b,t=p._encode_observation_info(obs)
    assert (b,t)==(2,2) and cond.shape==(2,2*sum(encoder.feature_dims))
    before={k:v.clone() for k,v in encoder.state_dict().items()}
    optimizer=torch.optim.AdamW([v for module in (p.model,p.obs_encoder,p.wrist_obs_encoder)
                                for v in module.parameters() if v.requires_grad],lr=1e-3)
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
    torch.save({'state_dict':{}},tmp_path/'state.ckpt')
    with pytest.raises(ValueError,match='model object'): load_pretrained_encoder(dict(settings,checkpoint=str(tmp_path/'state.ckpt')))
    with pytest.raises(ValueError,match='process_stats'): load_pretrained_encoder(dict(settings,process_stats=None))
    config=OmegaConf.load(settings['world_config']); config.wm.embed_dim=99; OmegaConf.save(config,settings['world_config'])
    with pytest.raises(ValueError,match='architecture'): load_pretrained_encoder(settings)


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
    for policy in (old,p):
        for module in (policy.model,policy.obs_encoder,policy.wrist_obs_encoder): module.train()
        params=[v for module in (policy.model,policy.obs_encoder,policy.wrist_obs_encoder) for v in module.parameters()]
        optimizer=torch.optim.AdamW(params,lr=1e-4)
        torch.manual_seed(123)
        policy.compute_loss(batch).backward()
        optimizer.step()
    for name in ('model','obs_encoder','wrist_obs_encoder'):
        assert all(torch.equal(v,getattr(old,name).state_dict()[k]) for k,v in getattr(p,name).state_dict().items())
