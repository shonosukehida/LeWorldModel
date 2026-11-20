"""Exercise actual evaluation dispatch without importing robot entrypoint."""
import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import pytest
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('kind', ['imle', 'gpc'])
def test_imle_dry_run_returns_before_normalization_and_robot_setup(kind):
    node = next(n for n in ast.parse((ROOT / 'eval_real_robot.py').read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'run')
    node.decorator_list = []
    ns = dict(DictConfig=object, img_transform=lambda cfg: 'training-preprocessing',
              swm=SimpleNamespace(policy=SimpleNamespace(AutoCostModel=Mock())))
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'dispatch', 'exec'), ns)
    cfg = OmegaConf.load(ROOT / 'config/eval/flip_mug.yaml')
    cfg.policy_type = kind
    cfg.gpc.diffusion_checkpoint = None
    cfg.gpc.imle_checkpoint = 'synthetic.ckpt'
    cfg.eval.real_robot.dry_run = True
    with patch('offline_imle_eval.run_offline_imle') as standalone, patch(
            'offline_imle_eval.run_offline_imle_gpc_from_config') as ranked:
        ns['run'](cfg)
    if kind == 'imle':
        standalone.assert_called_once_with(cfg, 'training-preprocessing')
        ranked.assert_not_called()
    else:
        ranked.assert_called_once_with(cfg, 'training-preprocessing', ns['swm'].policy.AutoCostModel)
        standalone.assert_not_called()


def test_legacy_yaml_values_unchanged_and_imle_disabled():
    import subprocess
    old = OmegaConf.create(subprocess.check_output(
        ['git', '-C', str(ROOT), 'show', 'cc06f7d:config/eval/flip_mug.yaml'], text=True))
    new = OmegaConf.load(ROOT / 'config/eval/flip_mug.yaml')
    assert new.gpc.imle_checkpoint is None
    del new.gpc.imle_checkpoint
    del new.gpc.imle_device
    assert new.gpc.goal_mode == 'final_goal'
    del new.gpc.goal_mode
    assert new.gpc.execute_first_candidate is False
    del new.gpc.execute_first_candidate
    for key in ('lambda_overhead', 'lambda_wrist'):
        assert new.gpc[key] == 1.0
        del new.gpc[key]
    assert OmegaConf.to_container(old, resolve=False) == OmegaConf.to_container(new, resolve=False)


@pytest.mark.parametrize('proposal', ['diffusion', 'imle'])
@pytest.mark.parametrize('goal_mode', [None, 'current_observation'])
def test_robot_gpc_dispatch_and_saved_config(tmp_path, proposal, goal_mode):
    from hydra import compose, initialize_config_dir
    from omegaconf import open_dict
    from stable_worldmodel.policy import GPCPolicy
    from stable_worldmodel.reward import LatentGoalReward

    with initialize_config_dir(version_base=None, config_dir=str(ROOT / 'config/eval')):
        overrides = [f'gpc.goal_mode={goal_mode}'] if goal_mode else []
        cfg = compose(config_name='flip_mug', overrides=overrides)
    if goal_mode is None:
        with open_dict(cfg.gpc):
            del cfg.gpc.goal_mode  # Legacy config saved before this setting existed.
    if proposal == 'imle':
        cfg.gpc.diffusion_checkpoint = None
        cfg.gpc.imle_checkpoint = 'imle.ckpt'
    stats = tmp_path / 'stats.npz'
    stats.touch()
    cfg.eval.normalization_stats_path = str(stats)
    process = {k: object() for k in ('action_cartesian', 'proprio', 'goal_proprio')}
    model = Mock(spec=['to', 'eval', 'requires_grad_'])
    model.to.return_value = model.eval.return_value = model
    dp_loader = Mock(return_value=SimpleNamespace(action_horizon=8))

    # Stop at the entry to the control loop. Exercise both real GPC constructors.
    class ReachedRobotLoop(Exception):
        pass

    robot = Mock(side_effect=ReachedRobotLoop)
    ns = dict(DictConfig=object, Path=Path, OmegaConf=OmegaConf, LatentGoalReward=LatentGoalReward,
              swm=SimpleNamespace(data=SimpleNamespace(utils=SimpleNamespace(get_cache_dir=lambda: tmp_path)),
                                  policy=SimpleNamespace(AutoCostModel=lambda _: model,
                                                         GPCPolicy=GPCPolicy, RandomPolicy=type(None))),
              img_transform=lambda _: None, dp_img_transform=lambda _: None,
              load_normalization_process=lambda _: (process, 'action_cartesian'),
              load_diffusion_policy=dp_loader, run_xarm_task=robot)
    node = next(n for n in ast.parse((ROOT / 'eval_real_robot.py').read_text()).body
                if isinstance(n, ast.FunctionDef) and n.name == 'run')
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), 'dispatch', 'exec'), ns)
    provenance = {'world_model_checkpoint': 'saved WM'}
    with (patch('real_robot_latent_visualization.checkpoint_path', return_value=tmp_path / 'wm_object.ckpt'),
          patch('real_robot_latent_visualization.capture_metadata', return_value=provenance),
          patch('imle_checkpoint.load_imle_policy', return_value=SimpleNamespace(action_horizon=8)) as imle_loader):
        with pytest.raises(ReachedRobotLoop):
            ns['run'](cfg)
    stored_config, policy = robot.call_args.args[:2]
    assert stored_config.gpc.goal_mode == policy.goal_mode == (goal_mode or 'final_goal')
    assert policy._latent_visualization_metadata == provenance
    assert dp_loader.called is (proposal == 'diffusion')
    assert imle_loader.called is (proposal == 'imle')


@pytest.mark.parametrize('goal_mode', [None, 'current_observation'])
@pytest.mark.parametrize('saved_stats', [True, False])
def test_imle_gpc_config_loads_checkpoint_and_wm_statistics_read_only(tmp_path, saved_stats, goal_mode):
    import h5py
    import torch
    import offline_imle_eval as offline
    from test_imle_integration import config, dataset, stats, TinyWorldModel
    from imle_checkpoint import build_policy, save_checkpoint, parameters
    from create_process_stats import create_process_stats
    torch.set_num_threads(1)
    train = config()
    # This legacy image-conditioned fixture has no checkpoint proprio statistics.
    train.imle.use_proprio = False
    proposal = build_policy(train)
    checkpoint = tmp_path / 'imle.ckpt'
    save_checkpoint(checkpoint, proposal, torch.optim.AdamW(parameters(proposal)),
                    0, 1.0, train, stats())
    data = tmp_path / 'push.h5'
    dataset(data, 3)
    stat_path = tmp_path / 'stats/process_stats.npz'
    if saved_stats:
        create_process_stats(data, stat_path.parent)
        before = stat_path.read_bytes()
    cfg = OmegaConf.load(ROOT / 'config/eval/flip_mug.yaml')
    cfg.policy_type = 'gpc'
    cfg.policy = 'synthetic-world'
    cfg.gpc.diffusion_checkpoint = None
    cfg.gpc.imle_checkpoint = str(checkpoint)
    if goal_mode is None:
        del cfg.gpc.goal_mode
    else:
        cfg.gpc.goal_mode = goal_mode
    cfg.gpc.imle_device = 'cpu'
    cfg.gpc.num_candidates = 3
    cfg.gpc.lambda_prop = 0.0
    cfg.gpc.lambda_overhead = 0.25
    cfg.gpc.lambda_wrist = 2.0
    cfg.eval.dataset_name = str(data)
    cfg.eval.normalization_stats_path = str(stat_path)
    cfg.eval.real_robot.dry_run_mode = "episode"
    cfg.eval.real_robot.dry_run_dataset_path = str(data)
    cfg.eval.real_robot.dry_run_episode_idx = 0
    cfg.eval.real_robot.dp_execute_steps = 2
    cfg.eval.real_robot.output_dir = str(tmp_path / 'out')
    world = TinyWorldModel()
    world.overhead_projector = torch.nn.Linear(1, 2)
    world.wrist_projector = torch.nn.Linear(1, 2)
    world.prop_encoder = torch.nn.Linear(1, 4)
    with patch.object(offline, 'evaluate_imle_gpc', wraps=offline.evaluate_imle_gpc) as evaluate:
        result = offline.run_offline_imle_gpc_from_config(
            cfg, lambda x: x.float() / 255, lambda path: world)
    gpc = evaluate.call_args.args[0]
    assert gpc.goal_mode == (goal_mode or 'final_goal')
    with h5py.File(tmp_path / 'out/offline_imle_gpc/imle_gpc_rollout.h5') as rollout:
        assert rollout.attrs['goal_mode'] == gpc.goal_mode
    assert gpc.reward_fn.lambda_prop == 0.0
    assert gpc.reward_fn.lambda_overhead == 0.25
    assert gpc.reward_fn.lambda_wrist == 2.0
    assert gpc.num_candidates == 3
    assert result.shape == (3, 8)
    if saved_stats:
        assert stat_path.read_bytes() == before
    else:
        assert not stat_path.exists()



def test_imle_gpc_rejects_unsupported_mode_before_loading_models():
    from offline_imle_eval import run_offline_imle_gpc
    cfg = OmegaConf.load(ROOT / 'config/eval/flip_mug.yaml')
    cfg.eval.real_robot.dry_run_mode = 'latent_dist'
    loader = Mock()
    with pytest.raises(ValueError, match='dry_run_mode=episode'):
        run_offline_imle_gpc(cfg, None, loader, {}, {})
    loader.assert_not_called()
