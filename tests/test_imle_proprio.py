"""CPU state-conditioning contracts. No hardware or camera imports."""
from pathlib import Path
import importlib.util
import subprocess
import tempfile
import unittest
import h5py
import numpy as np
import torch
from torch_ema import ExponentialMovingAverage
from stable_worldmodel.imle.proprio import (
    fit_proprio_stats, normalize_proprio, inverse_proprio, validate_proprio_stats,
)
from imle_checkpoint import build_policy, save_checkpoint, load_imle_policy, parameters, modules
from offline_imle_eval import load_proprio_episode, proprio_evaluation_options
from offline_imle_core import evaluate_episode, evaluate_multi_sample, evaluate_chunk_rollout
from test_imle_integration import config, stats, dataset

ROOT = Path(__file__).resolve().parents[1]
import stable_worldmodel
SWM = Path(stable_worldmodel.__file__).resolve().parents[1]


def state_stats():
    return fit_proprio_stats(torch.arange(80).reshape(10, 8).float() / 100)


class ProprioTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def policy(self):
        cfg = config()
        cfg.imle.use_proprio = True
        return cfg, build_policy(cfg, proprio_stats=state_stats())

    def test_train_val_normalization_and_invalid(self):
        train = torch.tensor([[1., 2, 3, 0, 0, 0, 1, 0], [3., 2, 3, 0, 0, 0, 1, 1]])
        st = fit_proprio_stats(train)
        n = normalize_proprio(train, st)
        torch.testing.assert_close(n.mean(0), torch.zeros(8), rtol=0, atol=0)
        torch.testing.assert_close(n[:, 0], torch.tensor([-1., 1.]))
        val = train + 10
        torch.testing.assert_close(normalize_proprio(val, st), (val-st['proprio_mean'])/st['proprio_std'])
        torch.testing.assert_close(inverse_proprio(n, st), train)
        for invalid in (float('nan'), float('inf')):
            x = train.clone(); x[0, 0] = invalid
            with self.assertRaises(ValueError): fit_proprio_stats(x)
            with self.assertRaises(ValueError): normalize_proprio(x, st)
        with self.assertRaises(ValueError): validate_proprio_stats(None)

    def test_shapes_loss_step_ema_checkpoint_and_raw_normalization_exact(self):
        cfg, p = self.policy()
        b = dict(pixels=torch.randn(2, 2, 3, 32, 32),
                 wrist_pixels=torch.randn(2, 2, 3, 32, 32),
                 proprio=torch.randn(2, 2, 8), action=torch.randn(2, 16, 8))
        cond = p._condition(b['pixels'], b['wrist_pixels'], b['proprio'])
        self.assertEqual(cond.shape, (2, 2, 24))
        expected = normalize_proprio(b['proprio'], p.proprio_stats)
        torch.testing.assert_close(cond[..., -8:], expected, rtol=0, atol=0)
        info = p._prepare_info({k: v.detach().numpy() for k,v in b.items() if k != 'action'})
        torch.testing.assert_close(p._condition(info['pixels'],info['wrist_pixels'],info['proprio'])[..., -8:], expected, rtol=0, atol=0)
        optimizer = torch.optim.AdamW(parameters(p), lr=1e-4)
        ema = ExponentialMovingAverage(parameters(p), decay=.9)
        loss = p.compute_loss(b); self.assertTrue(torch.isfinite(loss)); loss.backward()
        self.assertTrue(all(v.grad is not None and torch.isfinite(v.grad).all() for v in parameters(p)))
        optimizer.step(); ema.update(parameters(p))
        for m in modules(p): m.eval()
        torch.manual_seed(19); a = p.sample_action_sequences(b, 3); rng = torch.get_rng_state()
        torch.manual_seed(19); torch.testing.assert_close(a, p.sample_action_sequences(b,3),rtol=0,atol=0)
        self.assertEqual(a.shape,(2,3,16,8)); self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        self.assertFalse(torch.equal(a[:,0],a[:,1]))
        altered = dict(b, proprio=b['proprio']+1)
        torch.manual_seed(19); self.assertFalse(torch.equal(a, p.sample_action_sequences(altered,3)))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'new.ckpt'
            save_checkpoint(path,p,optimizer,0,loss.item(),cfg,stats())
            q=load_imle_policy(path,lambda x:x,'cpu'); q.transform={}
            torch.manual_seed(19); torch.testing.assert_close(a,q.sample_action_sequences(b,3),rtol=0,atol=0)
            torch.testing.assert_close(expected,normalize_proprio(b['proprio'],q.proprio_stats),rtol=0,atol=0)
            save_checkpoint(path,p,optimizer,0,loss.item(),cfg,stats(),ema)
            q=load_imle_policy(path,lambda x:x,'cpu'); q.transform={}
            with ema.average_parameters(parameters(p)):
                torch.manual_seed(19); expected_ema=p.sample_action_sequences(b,3)
            torch.manual_seed(19); torch.testing.assert_close(expected_ema,q.sample_action_sequences(b,3),rtol=0,atol=0)
            state=torch.load(path,weights_only=False); del state['proprio_stats']; torch.save(state,path)
            with self.assertRaisesRegex(ValueError,'requires checkpoint proprio'):load_imle_policy(path,lambda x:x,'cpu')
        for value in (None,torch.zeros(2,1,8),torch.full((2,2,8),float('nan'))):
            with self.assertRaises(ValueError):p._condition(b['pixels'],b['wrist_pixels'],value)

    def test_legacy_same_weights_condition_output_rng_exact(self):
        cfg=config(); cfg.imle.pop('use_proprio',None)
        # Execute exact pre-change policy class; reuse the same model/encoders/weights.
        source=subprocess.check_output(['git','-C',str(SWM),'show','cfe90c7:stable_worldmodel/imle/policy.py'],text=True)
        env={'__name__':'stable_worldmodel.imle.legacy','__package__':'stable_worldmodel.imle'}
        exec(compile(source,'legacy_policy.py','exec'),env)
        p=build_policy(cfg)
        old=env['IMLEPolicy'](p.model,p.obs_encoder,p.wrist_obs_encoder,16,2,2,8)
        info=dict(pixels=torch.randn(1,2,3,32,32),wrist_pixels=torch.randn(1,2,3,32,32))
        for seed in (0,42,99):
            torch.manual_seed(seed); expected=old.sample_action_sequences(info,3); rng=torch.get_rng_state()
            torch.manual_seed(seed); actual=p.sample_action_sequences(info,3)
            torch.testing.assert_close(expected,actual,rtol=0,atol=0)
            self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'old.ckpt'; opt=torch.optim.AdamW(parameters(p))
            save_checkpoint(path,p,opt,0,0.,cfg,stats())
            state=torch.load(path,weights_only=False); state.pop('use_proprio'); torch.save(state,path)
            q=load_imle_policy(path,lambda x:x,'cpu'); q.transform={}; self.assertFalse(q.use_proprio)
            torch.manual_seed(99); torch.testing.assert_close(actual,q.sample_action_sequences(info,3),rtol=0,atol=0)

    def test_training_entrypoint_train_only_stats(self):
        from unittest.mock import patch
        from torch.utils.data import random_split
        from train_imle_policy import run
        import stable_worldmodel as swm
        with tempfile.TemporaryDirectory() as tmp:
            cfg = config(); cfg.imle.use_proprio = True
            path = Path(tmp)/'data.h5'; dataset(path)
            cfg.data.dataset.path = str(path)
            with patch('train_imle_policy.swm.data.utils.get_cache_dir', return_value=tmp):
                run.__wrapped__(cfg)
            checkpoint = next((Path(tmp)/'checkpoints').rglob('latest.ckpt'))
            state = torch.load(checkpoint,weights_only=False)
            self.assertTrue(state['use_proprio']); self.assertIn('ema',state)
            self.assertTrue(np.isfinite(state['val_loss']))
            raw = swm.data.HDF5Dataset(path=str(path),num_steps=16,frameskip=1,
                                      keys_to_load=['proprio'],keys_to_cache=['proprio'])
            train,_ = random_split(raw,[int(len(raw)*cfg.train_split),len(raw)-int(len(raw)*cfg.train_split)],
                                   generator=torch.Generator().manual_seed(cfg.seed))
            expected = fit_proprio_stats(torch.cat([raw[i]['proprio'][:2] for i in train.indices]))
            for key in ('proprio_mean','proprio_std'):
                torch.testing.assert_close(expected[key],state['proprio_stats'][key],rtol=0,atol=0)

    def test_gpc_dry_run_with_proprio_candidates(self):
        from types import SimpleNamespace as NS
        from test_imle_integration import TinyWorldModel
        from stable_worldmodel.policy import GPCPolicy
        from stable_worldmodel.reward import latent_goal_reward
        from action_normalization import restore_action_processor
        from offline_imle_eval import evaluate_imle_gpc
        cfg,p=self.policy()
        scaler=restore_action_processor(stats(),8)
        p.process={'action_cartesian':scaler}
        p.transform={k:lambda x:x.float()/255 for k in ('pixels','wrist_pixels')}
        identity=NS(transform=lambda x:x,inverse_transform=lambda x:x)
        gpc=GPCPolicy(action_policy=p,world_model=TinyWorldModel(),reward_fn=latent_goal_reward,
                      num_candidates=3,process={'action_cartesian':scaler,'proprio':identity,'goal_proprio':identity})
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'data.h5'; dataset(path,3)
            output=evaluate_imle_gpc(gpc,path,0,Path(tmp)/'out',2)
            self.assertEqual(output.shape,(3,8))
            self.assertTrue((Path(tmp)/'out/proprio_histories.npz').exists())
            self.assertTrue((Path(tmp)/'out/gpc/data/candidates.npz').exists())

    def test_current_multiview_dp_gpc_numerical_reference(self):
        import test_dp_action_chunks as helper
        old_path = helper.POLICY
        try:
            helper.POLICY = SWM/'stable_worldmodel/policy.py'
            helper.NumericalTests.setUpClass()
            reference = helper.NumericalTests()
            with tempfile.TemporaryDirectory() as tmp:
                path=Path(tmp)/'policy.py'
                path.write_bytes(subprocess.check_output(['git','-C',str(SWM),'show','cfe90c7:stable_worldmodel/policy.py']))
                reference.old=helper.classes(path)
            reference.test_gpc_candidates_rollout_rewards_selection()
            old,new=(reference.dp(v) for v in (reference.old,reference.new))
            info={'pixels':torch.linspace(0,1,2*3*32*32).reshape(1,2,3,32,32)}
            for seed in (0,42,99):
                torch.manual_seed(seed); expected=old.get_action_chunk(info); rng=torch.get_rng_state()
                torch.manual_seed(seed); actual=new.get_action_chunk(info)
                np.testing.assert_array_equal(actual,expected)
                self.assertTrue(torch.equal(rng,torch.get_rng_state()))
        finally:
            helper.POLICY=old_path

    def test_real_loop_state_history_mock_only(self):
        from unittest.mock import patch
        from test_dp_action_chunks import LoopTests
        from test_gripper_reinforcement import Diffusion, GPC
        class IMLE(Diffusion):
            type='imle'
            use_proprio=True
        class IMLEGPC(GPC):
            def __init__(self,commands):
                self.diffusion_policy=IMLE(commands)
                self.action_policy=self.diffusion_policy
                self.observations=self.action_policy.observations
        source=(ROOT/'eval_real_robot.py').read_text()
        for kind in (IMLE,IMLEGPC):
            for execute in (1,3):
                # Execution safety is tested separately; this mock test inspects histories.
                with patch('imle_runtime.validate_imle_action'):
                    env,policy,_,logs=LoopTests().run_loop(kind,source,steps=execute,enabled=False)
                for index,history in enumerate(policy.observations):
                    row=index*execute
                    expected_rows=np.maximum(np.arange(row-1,row+1),0)
                    expected=logs['proprio'][expected_rows][None]
                    np.testing.assert_array_equal(history['proprio'],expected)
                    self.assertEqual(history['pixels'].shape[:2],history['proprio'].shape[:2])
                    np.testing.assert_array_equal(history['pixels'][0,:,0,0,0],expected_rows)

    def test_actual_jepa_with_state_conditioned_proposals(self):
        from types import SimpleNamespace as NS
        from transformers import ViTConfig, ViTModel
        from jepa import JEPA
        from stable_worldmodel.policy import GPCPolicy
        from stable_worldmodel.reward import latent_goal_reward
        from action_normalization import restore_action_processor
        class Predictor(torch.nn.Module):
            def forward(self,emb,act):return emb+act
        vit=ViTConfig(image_size=32,patch_size=16,hidden_size=16,
                      num_hidden_layers=1,num_attention_heads=4,intermediate_size=32)
        world=JEPA(ViTModel(vit),ViTModel(vit),Predictor(),torch.nn.Linear(8,48),torch.nn.Linear(8,16)).eval()
        cfg,p=self.policy(); scaler=restore_action_processor(stats(),8)
        p.process={'action_cartesian':scaler}
        image_transform=lambda x:x.float()/255
        p.transform={k:image_transform for k in ('pixels','wrist_pixels')}
        identity=NS(transform=lambda x:x,inverse_transform=lambda x:x)
        gpc=GPCPolicy(action_policy=p,world_model=world,reward_fn=latent_goal_reward,num_candidates=3,
                      process={'action_cartesian':scaler,'proprio':identity,'goal_proprio':identity},
                      transform={k:image_transform for k in ('pixels','wrist_pixels','goal','goal_wrist_pixels')})
        image=np.zeros((1,1,32,32,3),dtype='u1')
        info=dict(pixels=image,wrist_pixels=image,goal=image,goal_wrist_pixels=image,
                  proprio=np.zeros((1,1,8),dtype='f4'),goal_proprio=np.ones((1,1,8),dtype='f4'))
        history={k:np.repeat(image,2,axis=1) for k in ('pixels','wrist_pixels')}
        history['proprio']=np.ones((1,2,8),dtype='f4')
        with torch.no_grad():
            torch.manual_seed(7); proposals=p.sample_action_sequences(history,3,True)
            wm_actions=gpc.normalize_action_for_world_model(proposals[:,:,1:3])
            output=gpc.rollout(info,wm_actions)
            # Reconstruct the original JEPA class with identical module weights.
            import copy
            old_ns={}
            exec(subprocess.check_output(['git','-C',str(ROOT),'show','e6048c9:jepa.py'],text=True),old_ns)
            old_world=old_ns['JEPA'](*[copy.deepcopy(module) for module in (
                world.overhead_encoder,world.wrist_encoder,world.predictor,
                world.action_encoder,world.prop_encoder)]).eval()
            gpc.world_model=old_world
            baseline_output=gpc.rollout(info,wm_actions)
            for key,value in output.items():
                if torch.is_tensor(value):
                    torch.testing.assert_close(value,baseline_output[key],rtol=0,atol=0)
            gpc.world_model=world
            rewards=latent_goal_reward(output,info)
            torch.manual_seed(7); selected=gpc.get_action_chunk(info,dp_info_dict=history)
        np.testing.assert_array_equal(selected,proposals[:,rewards.argmax(1).item(),1:3].numpy())

    def test_offline_modes_and_padding_alignment(self):
        cfg,p=self.policy()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); data_path=root/'data.h5'; dataset(data_path)
            with h5py.File(data_path,'a') as f:
                f['proprio'][:]=np.arange(24*8,dtype='f4').reshape(24,8)/100
            ckpt=root/'p.ckpt'; save_checkpoint(ckpt,p,torch.optim.AdamW(parameters(p)),0,0.,cfg,stats())
            # Native raw HWC -> training-compatible CHW tensor.
            transform=lambda x:x.float()/255
            q=load_imle_policy(ckpt,transform,'cpu')
            data,steps,_=load_proprio_episode(data_path,0)
            options=proprio_evaluation_options(q,data_path,0,root)
            histories=np.load(root/'proprio_histories.npz')
            np.testing.assert_array_equal(histories['raw'][0],np.repeat(data['proprio'][0:1],2,0))
            np.testing.assert_array_equal(histories['raw'][1],data['proprio'][:2])
            evaluate_episode(q,data_path,0,root/'episode',{},**options)
            evaluate_episode(q,data_path,0,root/'episode_chunk',{},dp_execute_steps=2,**options)
            evaluate_multi_sample(q,data_path,0,int(steps[2]),2,1,root/'multi',{},**options)
            evaluate_chunk_rollout(q,data_path,0,2,1,root/'chunks',{},**options)

if __name__=='__main__':unittest.main()
