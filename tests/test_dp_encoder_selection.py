"""Real CNN/U-Net/DDPM compatibility checks, without importing robot hardware."""
import gc
import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch
import stable_worldmodel as swm
from stable_worldmodel.diffusion import (
    ResNet18ObsEncoder, ConvObsEncoder, ConditionalUnet1D,
    create_obs_encoder, resolve_obs_encoder_config,
)
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from omegaconf import OmegaConf
from action_normalization import validate_action_stats, restore_action_processor
from normalization_stats import SafeStandardScaler
from test_action_normalization import extract

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'tests/baseline_dp_encoder'
REAL = Path('/home/shonosukehida/.stable_worldmodel/checkpoints/flip_mug/policy/diffusion_policy/ep200_tm300_multiview_demo/seed_41/20261021_135400/latest.ckpt')
spec = importlib.util.spec_from_file_location('frozen_encoder', BASE/'obs_encoder.py')
frozen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(frozen)


def loader(old=False):
    namespace = dict(Path=Path, torch=torch, swm=swm,
        ResNet18ObsEncoder=frozen.ResNet18ObsEncoder, create_obs_encoder=create_obs_encoder,
        ConditionalUnet1D=ConditionalUnet1D, DDPMScheduler=DDPMScheduler,
        validate_action_stats=validate_action_stats, restore_action_processor=restore_action_processor,
        SafeStandardScaler=SafeStandardScaler)
    path = BASE/'eval_real_robot.py' if old else ROOT/'eval_real_robot.py'
    return extract(path.read_text(), 'load_diffusion_policy', namespace)


def config(kind='resnet18', dim=512):
    return OmegaConf.create(dict(encoder=dict(type=kind, pretrained=False, feature_dim=dim),
        policy=dict(action_key='action_cartesian', pred_horizon=8, obs_horizon=2, action_horizon=3),
        model=dict(diffusion_step_embed_dim=16, down_dims=[16,32], kernel_size=3,
                   n_groups=8, cond_predict_scale=True),
        diffusion=dict(num_train_timesteps=8, beta_schedule='linear', prediction_type='epsilon', clip_sample=True)))


def policy(cfg, old=False):
    kind, settings = resolve_obs_encoder_config(cfg.encoder)
    enc = lambda: frozen.ResNet18ObsEncoder() if old else create_obs_encoder(kind, settings)
    overhead, wrist = enc(), enc()
    model = ConditionalUnet1D(8, global_cond_dim=(overhead.output_shape()[0]+wrist.output_shape()[0])*2,
                             **dict(cfg.model))
    return swm.policy.DiffusionPolicy(model, overhead, DDPMScheduler(**dict(cfg.diffusion)),
        8, 2, 3, 8, num_inference_steps=4, wrist_obs_encoder=wrist, transform={'pixels':lambda x:x, 'wrist_pixels':lambda x:x})


def save(path, p, cfg):
    fn = extract((ROOT/'train_policy.py').read_text(), 'save_checkpoint',
        dict(Path=Path, DiffusionPolicy=swm.policy.DiffusionPolicy, torch=torch,
             OmegaConf=OmegaConf, validate_action_stats=validate_action_stats,
             resolve_obs_encoder_config=resolve_obs_encoder_config))
    fn(path, p, torch.optim.Adam(list(p.model.parameters())+list(p.obs_encoder.parameters())+
       list(p.wrist_obs_encoder.parameters())), 0, 0., torch.zeros(1,8), torch.ones(1,8), cfg)


@torch.no_grad()
def outputs(p):
    for net in (p.model, p.obs_encoder, p.wrist_obs_encoder): net.eval()
    n = p.obs_horizon
    pixels = torch.linspace(0, 1, n*3*32*32).reshape(1,n,3,32,32)
    wrist = pixels.flip(-1)
    a = p.obs_encoder(pixels.flatten(0,1))
    b = p.wrist_obs_encoder(wrist.flatten(0,1))
    cond, _, _ = p._encode_observation(pixels, wrist)
    # Capture the actual U-Net conditioning as well as encoder features.
    received = []
    handle = p.model.register_forward_pre_hook(lambda m,args,kw: received.append(kw['global_cond'].clone()), with_kwargs=True)
    torch.manual_seed(42)
    result = p.predict_action({'pixels':pixels.permute(0,1,3,4,2), 'wrist_pixels':wrist.permute(0,1,3,4,2)})['action_pred']
    handle.remove()
    assert all(torch.equal(cond, x) for x in received)
    return a, b, cond, result, torch.get_rng_state()


class EncoderSelectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls): torch.set_num_threads(2)

    def compare(self, a, b, label):
        errors = [float((x.float()-y.float()).abs().max()) for x,y in zip(a,b)]
        for x,y in zip(a,b): self.assertTrue(torch.equal(x,y))
        print(f'{label}: overhead/wrist/U-Net condition/actions/RNG max_abs={errors}; tolerance=0')

    def test_structure_and_defaults(self):
        old, new = frozen.ResNet18ObsEncoder().eval(), create_obs_encoder().eval()
        self.assertEqual({k:tuple(v.shape) for k,v in old.state_dict().items()},
                         {k:tuple(v.shape) for k,v in new.state_dict().items()})
        self.assertEqual(old.output_shape(), new.output_shape())
        self.assertEqual(old(torch.zeros(2,3,32,32)).shape, new(torch.zeros(2,3,32,32)).shape)
        self.assertEqual(resolve_obs_encoder_config({'pretrained':False}), ('resnet18', {'pretrained':False}))
        for kind in ('resnet18','conv'):
            net=create_obs_encoder(kind)
            print(kind, 'parameters per view=',sum(p.numel() for p in net.parameters()), 'features=', net.output_shape())
        for fn in (lambda: create_obs_encoder('typo'), lambda: resolve_obs_encoder_config({'type':'typo'})):
            with self.assertRaisesRegex(ValueError, 'Unknown encoder_type'): fn()

    def test_legacy_checkpoint_and_exact_inference(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'legacy.ckpt'
            cfg=config(); p=policy(cfg, old=True)
            # Save the old format using the unchanged pre-selection save function.
            source=(BASE/'train_policy.py').read_text()
            fn=extract(source,'save_checkpoint',dict(Path=Path, DiffusionPolicy=swm.policy.DiffusionPolicy,
                torch=torch, OmegaConf=OmegaConf, validate_action_stats=validate_action_stats))
            fn(path,p,torch.optim.Adam(p.model.parameters()),0,0.,torch.zeros(1,8),torch.ones(1,8),cfg)
            self.assertNotIn('encoder_type',torch.load(path,weights_only=False))
            del p; gc.collect()
            old=loader(True)(path,lambda x: x,'cpu'); expected=outputs(old); del old; gc.collect()
            new=loader()(path,lambda x: x,'cpu')
            self.assertIsInstance(new.obs_encoder,ResNet18ObsEncoder)
            self.compare(expected, outputs(new), 'synthetic legacy checkpoint')

    def test_new_checkpoints_and_conv_optimizer(self):
        for kind,dim in [('resnet18',512),('conv',37),('conv',512)]:
            with self.subTest(kind=kind,dim=dim), tempfile.TemporaryDirectory() as tmp:
                cfg=config(kind,dim); p=policy(cfg)
                if kind=='conv':
                    optimizer=torch.optim.Adam(list(p.model.parameters())+list(p.obs_encoder.parameters())+list(p.wrist_obs_encoder.parameters()))
                    image=torch.randn(2,3,32,32)
                    cond=torch.cat([p.obs_encoder(image),p.wrist_obs_encoder(image)],-1).reshape(1,-1)
                    before=p.obs_encoder.backbone[0].weight.detach().clone()
                    p.model(torch.randn(1,8,8),torch.tensor(2),global_cond=cond).square().mean().backward()
                    self.assertIsNotNone(p.wrist_obs_encoder.backbone[0].weight.grad)
                    optimizer.step()
                    self.assertFalse(torch.equal(before,p.obs_encoder.backbone[0].weight))
                expected=outputs(p); path=Path(tmp)/'new.ckpt'; save(path,p,cfg)
                ckpt=torch.load(path,weights_only=False)
                self.assertEqual(ckpt['encoder_type'],kind)
                if kind=='conv': self.assertEqual(ckpt['encoder_config'],{'feature_dim':dim})
                # The full training config cannot override reconstruction metadata.
                ckpt['config']['encoder']={'type':'conv' if kind=='resnet18' else 'resnet18', 'feature_dim':999}
                torch.save(ckpt,path)
                restored=loader()(path,lambda x: x,'cpu')
                self.assertIsNot(restored.obs_encoder,restored.wrist_obs_encoder)
                self.compare(expected,outputs(restored),f'new {kind}/{dim}')
                for key,net in [('model_state_dict',restored.model),('obs_encoder_state_dict',restored.obs_encoder),('wrist_obs_encoder_state_dict',restored.wrist_obs_encoder)]:
                    net.load_state_dict(ckpt[key],strict=True)
                ckpt['encoder_type']='typo'; torch.save(ckpt,path)
                with self.assertRaisesRegex(ValueError,'Unknown encoder_type'): loader()(path,lambda x: x,'cpu')

    @unittest.skipUnless(REAL.exists(), 'Production checkpoint unavailable; synthetic test covers legacy path')
    def test_production_checkpoint_exact_inference(self):
        # Sequential loads bound memory use; never modify the production file.
        metadata=torch.load(REAL, map_location='cpu', weights_only=False, mmap=True)
        self.assertNotIn('encoder_type', metadata)
        del metadata
        old=loader(True)(REAL,lambda x: x,'cpu'); expected=outputs(old); del old; gc.collect()
        new=loader()(REAL,lambda x: x,'cpu')
        self.compare(expected,outputs(new),'production legacy checkpoint')
        print('Production checkpoint:', REAL)

    def test_conv_offline_episode_and_multi_sample(self):
        from test_offline_dp_eval import OfflineTests
        from offline_dp_eval import evaluate_episode, evaluate_multi_sample
        with tempfile.TemporaryDirectory() as tmp:
            directory=Path(tmp)
            checkpoint=directory/'conv.ckpt'
            save(checkpoint,policy(config('conv',37)),config('conv',37))
            restored=loader()(checkpoint,lambda x:x,'cpu')
            dataset=directory/'data.h5'
            OfflineTests().dataset(dataset)
            evaluate_episode(restored,dataset,7,directory/'episode',{})
            evaluate_multi_sample(restored,dataset,7,1,2,42,directory/'multi_sample',{})
            self.assertTrue((directory/'episode/metrics.json').exists())
            self.assertTrue(list((directory/'multi_sample').rglob('multi_sample_metrics.json')))
