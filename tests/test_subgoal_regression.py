"""Opt-in exact inference comparison using the two pre-subgoal git trees and real checkpoints.

Set SUBGOAL_REGRESSION_IMLE to the IMLE checkpoint. The other asset paths
come from flip_mug.yaml and the matching multiview WM run. Set
SUBGOAL_REGRESSION_DEVICE=cuda to compare on GPU (the default is CPU).
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
from unittest.mock import Mock, patch

import pytest

ROOT = Path(__file__).resolve().parents[1]
SWM = ROOT.parent / "stable-worldmodel"
BASE = {ROOT: "5bf2474ae3a34fa1dbfebc293fbc540771bebcae",
        SWM: "795a18ce94e57d8b1a862e77078e8ae37e4d5606"}


def snapshot(repo, target):
    target.mkdir()
    with tempfile.TemporaryFile() as archive:
        subprocess.run(["git", "archive", BASE[repo], "--", "*.py", "*.yaml", "*.png"],
                       cwd=repo, stdout=archive, check=True)
        archive.seek(0)
        with tarfile.open(fileobj=archive) as files:
            files.extractall(target)


@pytest.mark.parametrize("mode", ["final_goal", "current_observation", "latent_smoothness"])
def test_small_wm_against_pre_subgoal_commit(mode):
    import torch
    from stable_worldmodel import policy as policies
    from test_gpc_opt_integration import components, info, make

    source = subprocess.check_output(
        ["git", "show", f"{BASE[SWM]}:stable_worldmodel/policy.py"], cwd=SWM)
    namespace = vars(policies).copy()
    exec(compile(source, "<pre-subgoal GPC>", "exec"), namespace)

    def run(cls):
        proposal, wm = components()
        torch.manual_seed(42)
        policy = make(cls, proposal, wm, {"goal_mode": mode})
        with patch.object(policy, "reward_fn", wraps=policy.reward_fn) as reward:
            chunk = policy.get_action_chunk(info())
            output, observation = reward.call_args.args
            return (proposal.actions, output["predicted_emb"], reward(output, observation),
                    policy.selected_candidate_index, torch.from_numpy(chunk), torch.get_rng_state())

    old = run(namespace["GPCPolicy"])
    new = run(policies.GPCPolicy)
    assert all(a.dtype == b.dtype and a.shape == b.shape and a.numpy().tobytes() == b.numpy().tobytes()
               for a, b in zip(old, new))


@pytest.mark.skipif(not os.getenv("SUBGOAL_REGRESSION_IMLE"), reason="requires local flip-mug assets")
def test_trained_checkpoints_against_pre_subgoal_commits(tmp_path):
    for repo in BASE:
        snapshot(repo, tmp_path / repo.name)
    for kind in ("imle", "diffusion"):
        traces = []
        for root in (tmp_path, ROOT.parent):
            output = tmp_path / f"{kind}.json"
            env = dict(os.environ, PYTHONPATH=os.pathsep.join(str(root / p.name) for p in BASE),
                       CUBLAS_WORKSPACE_CONFIG=":4096:8", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
            if env.get("SUBGOAL_REGRESSION_DEVICE", "cpu") == "cpu":
                env["CUDA_VISIBLE_DEVICES"] = ""
            subprocess.run([sys.executable, __file__, kind, str(output)], cwd=root / ROOT.name,
                           env=env, check=True)
            traces.append(json.loads(output.read_text()))
        assert traces[0].keys() == traces[1].keys()
        for key in traces[0]:
            assert traces[0][key] == traces[1][key], (kind, key)


def trace(kind, output):
    import types
    import h5py
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from stable_worldmodel.policy import GPCPolicy, AutoCostModel
    from stable_worldmodel.reward import gpc_scoring_options
    from stable_worldmodel.smoothness import MeasuredHistory
    from imle_checkpoint import load_imle_policy
    from normalization_stats import load_normalization_process
    from offline_dp_eval import dataset_images_to_hwc
    from offline_imle_core import build_observation_history

    # The CLI imports an unused MuJoCo environment at module scope.
    # Exercise the actual robot observation, action buffer and dry execution.
    sim = types.ModuleType("env.franka.env")
    sim.FrankaSimEnv = None
    with patch.dict(sys.modules, {"env.franka.env": sim}):
        import eval_real_robot as robot

    device = os.environ.get("SUBGOAL_REGRESSION_DEVICE", "cpu")
    torch.set_num_threads(1)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    cfg = OmegaConf.load("config/eval/flip_mug.yaml")
    rc = cfg.eval.real_robot
    cfg.gpc.num_candidates = 2
    transform = robot.img_transform(cfg)
    dp = (load_imle_policy(os.environ["SUBGOAL_REGRESSION_IMLE"], transform, device) if kind == "imle"
          else robot.load_diffusion_policy(cfg.gpc.diffusion_checkpoint, transform, device))
    process, _ = load_normalization_process(Path(cfg.eval.normalization_stats_path))
    model = AutoCostModel("flip_mug/ep100_tm300_multiview_play_normal_brightness/lewm").to(device).eval().requires_grad_(False)
    model.interpolate_pos_encoding = True
    row = dp.obs_horizon - 1
    with h5py.File(rc.dry_run_dataset_path) as f:
        data = {k: dataset_images_to_hwc(f[k][:row + 1], k) for k in ("pixels", "wrist_pixels")}
        data["proprio"] = f["proprio"][:row + 1]
        steps = f["step_idx"][:row + 1]
        assert np.unique(f["episode_idx"][:row + 1]).size == 1
    history_keys = tuple(data) if getattr(dp, "use_proprio", False) else ("pixels", "wrist_pixels")
    history = build_observation_history(data, row, dp.obs_horizon, history_keys)
    goal, wrist, prop = robot._load_or_capture_goal(None, rc)
    obs = robot._policy_observation(data["pixels"][row], data["wrist_pixels"][row], goal, wrist,
                                    data["proprio"][row, :7], data["proprio"][row, 7], prop, int(steps[row]), process)
    rc.dry_run = True
    rc.dry_run_image_path = rc.dry_run_wrist_image_path = None
    results = {}

    def record(name, value):
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, (np.ndarray, torch.Tensor)):
                    record(f"{name}/{key}", item)
            return
        array = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
        digest = hashlib.sha256(array.tobytes()).hexdigest()
        results.setdefault(f"{mode}/{name}", []).append((array.dtype.str, array.shape, digest))

    def tap(name, fn):
        def call(*args, **kwargs):
            value = fn(*args, **kwargs)
            record(name, value)
            return value
        return call

    dp._prepare_info = tap("policy_input", dp._prepare_info)
    dp.predict_action = tap("policy_prediction", dp.predict_action)
    if kind == "imle":
        dp._condition = tap("imle_condition", dp._condition)
    dp.sample_action_sequences = tap("physical_candidates", dp.sample_action_sequences)
    for mode in ("standalone", "final_goal", "current_observation", "latent_smoothness"):
        torch.manual_seed(cfg.seed)
        record("observation", obs)
        options = {}
        inference_input = history
        policy = dp
        if mode != "standalone":
            cfg.gpc.goal_mode = mode
            policy = GPCPolicy(dp, model, process=process, num_candidates=cfg.gpc.num_candidates,
                               transform={k: transform for k in ("pixels", "wrist_pixels", "goal", "goal_wrist_pixels")},
                               **gpc_scoring_options(cfg.gpc, model))
            policy._prepare_info = tap("wm_inputs", policy._prepare_info)
            policy.normalize_action_for_world_model = tap("normalized_actions", policy.normalize_action_for_world_model)
            policy.rollout = tap("wm_rollout", policy.rollout)
            policy.reward_fn = tap("rewards", policy.reward_fn)
            inference_input = dict(obs)
            if mode == "latent_smoothness":
                inference_input["smoothness_history"] = MeasuredHistory.from_episode(
                    policy.smoothness_horizon, data, data["proprio"], steps, row)
            options["dp_info_dict"] = history
        policy.get_action_chunk = tap("selected_chunk", policy.get_action_chunk)
        env = robot.XArmInferenceEnv(rc, cfg.plan_config, use_camera=False)
        # Also exercise the SDK-facing branch. Every hardware boundary is
        # already replaced before switching this second instance out of dry run.
        hardware = robot.XArmInferenceEnv(rc, cfg.plan_config, use_camera=False)
        state = hardware.get_robot_state()
        hardware.get_robot_state = Mock(return_value=state)
        hardware._ik_solver = Mock()
        hardware._ik_solver.solve.return_value = np.arange(7, dtype=np.float32) / 10
        hardware.forward_kinematics = Mock(return_value=state[2])
        hardware._fk_solver = Mock()
        hardware._fk_solver.solve.return_value = state[2]
        hardware._follower = Mock()
        hardware.dry_run = False
        queue = robot._DPActionBuffer(policy, rc.dp_execute_steps)
        for _ in range(queue.execute_steps):
            command = queue.get_action(inference_input, **options)
            record("command", command)
            record("executed", env.execute(command, "cartesian"))
            record("robot_target", hardware.execute(command, "cartesian"))
            record("follower_command", hardware._follower.command_joint_state.call_args.args[0])
        if mode != "standalone":
            record("selected_index", policy.selected_candidate_index)
            assert f"{mode}/wm_rollout/predicted_emb" in results
        record("rng", torch.get_rng_state())
        if torch.device(device).type == "cuda":
            record("cuda_rng", torch.cuda.get_rng_state(device))
    Path(output).write_text(json.dumps(results))


if __name__ == "__main__":
    trace(*sys.argv[1:])
