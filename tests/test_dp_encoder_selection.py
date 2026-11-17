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
from stable_worldmodel.diffusion.obs_encoder import SpatialSoftmax
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
def outputs(p, image_size=32):
    for net in (p.model, p.obs_encoder, p.wrist_obs_encoder): net.eval()
    n = p.obs_horizon
    pixels = torch.linspace(0, 1, n*3*image_size*image_size).reshape(1,n,3,image_size,image_size)
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

    def test_gap_initialization_and_training_exact(self):
        torch.manual_seed(314)
        old = policy(config(), old=True)
        old_rng = torch.get_rng_state()
        torch.manual_seed(314)
        cfg = config(); cfg.encoder.pooling = 'gap'
        new = policy(cfg)
        self.assertTrue(torch.equal(old_rng, torch.get_rng_state()))
        batch = {'pixels':torch.rand(1,2,3,64,64), 'wrist_pixels':torch.rand(1,2,3,64,64),
                 'action':torch.randn(1,8,8)}
        for a,b in ((old.model,new.model), (old.obs_encoder,new.obs_encoder),
                    (old.wrist_obs_encoder,new.wrist_obs_encoder)):
            self.assertEqual(list(a.state_dict()),list(b.state_dict()))
            for key,value in a.state_dict().items(): self.assertTrue(torch.equal(value,b.state_dict()[key]))
        results=[]
        for p in (old,new):
            nets=(p.model,p.obs_encoder,p.wrist_obs_encoder)
            params=[param for net in nets for param in net.parameters()]
            optimizer=torch.optim.Adam(params)
            torch.manual_seed(271)
            loss=p.compute_loss(batch); loss.backward(); optimizer.step()
            results.append((loss.detach(),torch.get_rng_state()))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(*results)))
        for a,b in ((old.model,new.model), (old.obs_encoder,new.obs_encoder),
                    (old.wrist_obs_encoder,new.wrist_obs_encoder)):
            for key,value in a.state_dict().items(): self.assertTrue(torch.equal(value,b.state_dict()[key]))

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
                else: self.assertEqual(ckpt['encoder_config'],{'pretrained':False})
                # The full training config cannot override reconstruction metadata.
                ckpt['config']['encoder']={'type':'conv' if kind=='resnet18' else 'resnet18',
                                           'feature_dim':999, 'pooling':'spatial_softmax'}
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


class SpatialSoftmaxTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    compare = EncoderSelectionTests.compare

    def test_expected_coordinates_and_gradients(self):
        pool = SpatialSoftmax()
        # Channel 0 weights [[1, 2], [3, 4]]; channel 1 reverses them.
        x = torch.tensor([[[[1., 2.], [3., 4.]], [[4., 3.], [2., 1.]]]],
                         dtype=torch.float64).log().requires_grad_()
        result = pool(x)
        torch.testing.assert_close(result, torch.tensor([[.2, .4, -.2, -.4]], dtype=x.dtype))
        self.assertEqual(result.shape, (1, 4))
        self.assertTrue(torch.isfinite(result).all())
        self.assertTrue((result.abs() <= 1).all())
        result.square().sum().backward()
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertGreater(x.grad.abs().sum().item(), 0)
        self.assertEqual(dict(pool.state_dict()), {})
        self.assertEqual(list(pool.parameters()), [])
        for shape in ((2,3,1,1), (2,3,3,7), (2,3,7,3)):
            self.assertTrue(torch.allclose(pool(torch.zeros(shape)), torch.zeros(shape[0],shape[1]*2), atol=1e-6))
        with self.assertRaisesRegex(ValueError, 'feature map'): pool(torch.zeros(2,3,4))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda(self):
        x = torch.randn(2,3,7,9, device='cuda', requires_grad=True)
        y = SpatialSoftmax()(x)
        self.assertTrue(y.is_cuda and torch.isfinite(y).all())
        y.square().sum().backward()
        self.assertGreater(x.grad.abs().sum().item(), 0)

    def test_config_validation_and_gap_conv_compatibility(self):
        for kind in ('resnet18', 'conv'):
            old_settings = {'pretrained':False} if kind == 'resnet18' else {'feature_dim':512}
            for setting in ({'type':kind}, {'type':kind, 'pooling':'gap'}):
                self.assertEqual(resolve_obs_encoder_config(setting), (kind,old_settings))
            self.assertEqual(resolve_obs_encoder_config({'type':kind, 'pooling':'spatial_softmax'}),
                             (kind,dict(old_settings,pooling='spatial_softmax')))
            for invalid in ('typo', None, 1):
                with self.assertRaisesRegex(ValueError, 'Unknown pooling'):
                    resolve_obs_encoder_config({'type':kind, 'pooling':invalid})
                with self.assertRaisesRegex(ValueError, 'Unknown pooling'):
                    create_obs_encoder(kind, {'pooling':invalid})
            cls = ResNet18ObsEncoder if kind == 'resnet18' else ConvObsEncoder
            with self.assertRaisesRegex(ValueError, 'Unknown pooling'): cls(pooling='typo')
        # Reconstruct the exact original ConvNet sequence, including initialization RNG.
        torch.manual_seed(17)
        old = torch.nn.Sequential(
            torch.nn.Conv2d(3,32,5,2,2), torch.nn.ReLU(),
            torch.nn.Conv2d(32,64,3,2,1), torch.nn.ReLU(),
            torch.nn.Conv2d(64,128,3,2,1), torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d((1,1)), torch.nn.Flatten(), torch.nn.Linear(128,37))
        old_rng = torch.get_rng_state()
        torch.manual_seed(17)
        new = create_obs_encoder('conv', {'feature_dim':37})
        self.assertTrue(torch.equal(old_rng, torch.get_rng_state()))
        self.assertEqual({k:v.shape for k,v in old.state_dict().items()},
                         {k:v.shape for k,v in new.backbone.state_dict().items()})
        new.backbone.load_state_dict(old.state_dict(), strict=True)
        image = torch.randn(2,3,64,64)
        self.assertTrue(torch.equal(old(image), new(image)))

    def test_encoder_spatial_maps_and_sizes(self):
        for kind, channels, dim in (('resnet18',512,512), ('conv',128,37)):
            net = create_obs_encoder(kind, {'pooling':'spatial_softmax', **({'feature_dim':dim} if kind=='conv' else {})}).eval()
            pool = net.backbone.avgpool if kind=='resnet18' else net.backbone[6]
            projection = net.backbone.fc if kind=='resnet18' else net.backbone[8]
            self.assertEqual((projection.in_features,projection.out_features), (2*channels,dim))
            for size in (64,224):
                captured=[]
                hook=pool.register_forward_hook(lambda m,args,out: captured.append((args[0].shape,out)))
                image=torch.randn(2,3,size,size, requires_grad=True)
                features=net(image)
                hook.remove()
                shape, coordinates=captured[0]
                self.assertGreater(shape[-1],1)
                self.assertEqual(coordinates.shape,(2,2*channels))
                self.assertTrue((coordinates.abs()<=1).all())
                self.assertEqual(features.shape,(2,dim))
                self.assertTrue(torch.isfinite(features).all())
                features.square().mean().backward()
                self.assertGreater(image.grad.abs().sum().item(),0)
                net.zero_grad()

    def test_spatial_training_checkpoint_offline_and_gpc(self):
        from test_offline_dp_eval import OfflineTests
        from test_offline_gpc_eval import WorldModel
        from offline_dp_eval import evaluate_episode, evaluate_multi_sample
        from stable_worldmodel.reward import latent_goal_reward
        import h5py
        import numpy as np
        for kind in ('resnet18','conv'):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as tmp:
                cfg=config(kind,37)
                cfg.encoder.pooling='spatial_softmax'
                p=policy(cfg)
                nets=(p.model,p.obs_encoder,p.wrist_obs_encoder)
                optimizer=torch.optim.Adam([param for net in nets for param in net.parameters()], lr=1e-3)
                before=[next(net.parameters()).detach().clone() for net in nets]
                batch={'pixels':torch.rand(1,2,3,64,64), 'wrist_pixels':torch.rand(1,2,3,64,64),
                       'action':torch.randn(1,8,8)}
                loss=p.compute_loss(batch)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                for net in nets:
                    grad=next(net.parameters()).grad
                    self.assertTrue(torch.isfinite(grad).all())
                    self.assertGreater(grad.abs().sum().item(),0)
                optimizer.step()
                for net,prev in zip(nets,before): self.assertFalse(torch.equal(prev,next(net.parameters())))
                expected=outputs(p,64)
                directory=Path(tmp); checkpoint=directory/'spatial.ckpt'
                save(checkpoint,p,cfg)
                ckpt=torch.load(checkpoint,weights_only=False)
                self.assertEqual(ckpt['encoder_config']['pooling'],'spatial_softmax')
                # Contradict both type and pooling in full config; metadata must win.
                ckpt['config']['encoder']={'type':'conv' if kind=='resnet18' else 'resnet18','pooling':'gap'}
                torch.save(ckpt,checkpoint)
                restored=loader()(checkpoint,lambda x:x,'cpu')
                self.compare(expected, outputs(restored,64), f'spatial {kind}')
                for key,net in [('model_state_dict',restored.model),('obs_encoder_state_dict',restored.obs_encoder),
                                ('wrist_obs_encoder_state_dict',restored.wrist_obs_encoder)]:
                    net.load_state_dict(ckpt[key],strict=True)
                # Removing pooling must fail strict restoration rather than convert weights.
                del ckpt['encoder_config']['pooling']; torch.save(ckpt,checkpoint)
                with self.assertRaises(RuntimeError): loader()(checkpoint,lambda x:x,'cpu')
                dataset=directory/'data.h5'; OfflineTests().dataset(dataset)
                with h5py.File(dataset, 'a') as data:
                    for key in ('pixels', 'wrist_pixels'):
                        images = data[key][:].repeat(16, axis=1).repeat(16, axis=2)
                        del data[key]
                        data[key] = images
                evaluate_episode(restored,dataset,7,directory/'episode',{})
                evaluate_multi_sample(restored,dataset,7,1,2,42,directory/'multi_sample',{})
                self.assertTrue((directory/'episode/metrics.json').exists())
                self.assertTrue(list((directory/'multi_sample').rglob('multi_sample_metrics.json')))
                # GPC uses this same checkpoint loader in eval_real_robot.main.
                gpc=swm.policy.GPCPolicy(restored,WorldModel(),latent_goal_reward,num_candidates=2,
                                         process=restored.process,transform={})
                info={'pixels':np.zeros((1,2,64,64,3),np.float32),
                      'wrist_pixels':np.zeros((1,2,64,64,3),np.float32),
                      'goal':np.zeros((1,1,64,64,3),np.float32)}
                chunk=gpc.get_action_chunk(info)
                self.assertEqual(chunk.shape,(1,3,8))
                self.assertTrue(np.isfinite(chunk).all())
