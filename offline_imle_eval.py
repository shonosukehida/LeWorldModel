"""Dataset-only IMLE evaluations using the unchanged DP plotting/time axis."""

from pathlib import Path
import numpy as np
import torch
from imle_checkpoint import load_imle_policy
from offline_dp_eval import (
    evaluate_episode,
    evaluate_multi_sample,
    evaluate_chunk_rollout,
)


def run_offline_imle(cfg, image_transform):
    robot = cfg.eval.real_robot
    policy = load_imle_policy(
        cfg.gpc.imle_checkpoint,
        image_transform,
        device=cfg.gpc.get("imle_device", "cuda"),
    )
    metadata = dict(
        policy.offline_checkpoint_metadata,
        image_size=cfg.eval.img_size,
        preprocessing="training get_img_preprocessor (range scaling, ImageNet normalization, resize)",
    )
    root = Path(robot.output_dir) / "offline_imle"
    mode = robot.get("dry_run_mode", "episode")
    if mode == "episode":
        return evaluate_episode(
            policy,
            robot.dry_run_dataset_path,
            int(robot.dry_run_episode_idx),
            root,
            metadata,
            dp_execute_steps=robot.get("dp_execute_steps", 1),
        )
    if mode == "multi_sample":
        return evaluate_multi_sample(
            policy,
            robot.dry_run_dataset_path,
            int(robot.dry_run_episode_idx),
            robot.dry_run_timestep,
            robot.dry_run_num_candidates,
            robot.dry_run_seed,
            root / "multi_sample",
            metadata,
        )
    if mode == "chunk_rollout":
        return evaluate_chunk_rollout(
            policy,
            robot.dry_run_dataset_path,
            int(robot.dry_run_episode_idx),
            robot.dry_run_num_candidates,
            robot.dry_run_seed,
            root / "chunk_rollout",
            metadata,
        )
    raise ValueError(f"Unknown dry_run_mode: {mode}")


@torch.no_grad()
def evaluate_imle_gpc(policy, dataset_path, episode_id, output_dir, execute_steps=1):
    """Exercise actual GPC/WM selection and recorder without robot initialization.

    Observations come from the dataset, never from simulated action feedback.
    The last stored observation is the goal. This evaluates proposals/selection,
    not closed-loop task success.
    """
    import h5py
    import importlib.util
    from offline_dp_eval import load_episode, build_observation_history

    spec = importlib.util.spec_from_file_location(
        "imle_gpc_visualization", Path(__file__).parent / "utils/gpc_visualization.py"
    )
    vis = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(vis)
    data, steps, _ = load_episode(dataset_path, episode_id)
    with h5py.File(Path(dataset_path).expanduser(), "r") as f:
        if "proprio" not in f:
            raise ValueError("IMLE-GPC offline evaluation requires stored proprio")
        rows = np.flatnonzero(np.asarray(f["episode_idx"]).reshape(-1) == episode_id)
        proprio = np.asarray(f["proprio"][rows])
    if proprio.shape != (len(steps), 8) or not np.isfinite(proprio).all():
        raise ValueError("IMLE-GPC requires finite Cartesian proprio (T,8)")
    if not 1 <= execute_steps <= policy.action_horizon:
        raise ValueError("dp_execute_steps must fit the GPC action horizon")
    predicted = []
    recorder = vis.GPCRecorder(policy)
    with recorder:
        for row in range(0, len(steps), execute_steps):
            obs = {
                key: data[key][row][None, None] for key in ("pixels", "wrist_pixels")
            }
            obs.update(
                proprio=proprio[row][None, None],
                goal=data["pixels"][-1][None, None],
                goal_wrist_pixels=data["wrist_pixels"][-1][None, None],
                goal_proprio=proprio[-1][None, None],
            )
            history = build_observation_history(
                data, row, policy.action_policy.obs_horizon
            )
            recorder.begin_step(int(steps[row]))
            chunk = policy.get_action_chunk(obs, dp_info_dict=history)
            predicted.extend(chunk[0, : min(execute_steps, len(steps) - row)])
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    recorder.save(root)
    with h5py.File(root / "imle_gpc_rollout.h5", "w") as f:
        f.create_dataset("predicted_action", data=np.asarray(predicted))
        f.create_dataset("ground_truth_action", data=data["action_cartesian"])
        f.create_dataset("step_idx", data=steps)
        f.attrs["evaluation"] = (
            "dataset observations; final dataset observation goal; no hardware"
        )
    return np.asarray(predicted)


def run_offline_imle_gpc(cfg, image_transform, world_model_loader, process, transform):
    from stable_worldmodel.policy import GPCPolicy
    from stable_worldmodel.reward import LatentGoalReward

    mode = cfg.eval.real_robot.get("dry_run_mode", "episode")
    if mode != "episode":
        raise ValueError("IMLE-GPC dataset dry_run supports dry_run_mode=episode only")
    device = cfg.gpc.get("imle_device", "cuda")
    if cfg.eval.probing.get("use_random_encoder", False):
        raise ValueError(
            "IMLE-GPC dataset dry_run requires the saved World Model encoder"
        )
    world_model = world_model_loader(cfg.policy).to(device).eval()
    world_model.requires_grad_(False)
    world_model.interpolate_pos_encoding = True
    action_policy = load_imle_policy(cfg.gpc.imle_checkpoint, image_transform, device)
    policy = GPCPolicy(
        action_policy=action_policy,
        world_model=world_model,
        reward_fn=LatentGoalReward(world_model, cfg.gpc.get("lambda_prop", 1.0)),
        num_candidates=cfg.gpc.num_candidates,
        process=process,
        transform=transform,
    )
    robot = cfg.eval.real_robot
    return evaluate_imle_gpc(
        policy,
        robot.dry_run_dataset_path,
        int(robot.dry_run_episode_idx),
        Path(robot.output_dir) / "offline_imle_gpc",
        robot.get("dp_execute_steps", 1),
    )


def run_offline_imle_gpc_from_config(cfg, image_transform, world_model_loader):
    """Load WM statistics read-only before any robot/environment initialization."""
    import stable_worldmodel as swm
    from normalization_stats import (
        load_normalization_process,
        build_normalization_process,
    )

    path = Path(cfg.eval.normalization_stats_path).expanduser()
    if path.is_file():
        process, _ = load_normalization_process(path)
    else:
        name = cfg.eval.dataset_name
        data_path = Path(name).expanduser()
        if not data_path.is_absolute():
            cache = Path(cfg.cache_dir or swm.data.utils.get_cache_dir()).expanduser()
            data_path = cache / "datasets" / f"{name}.h5"
        dataset = swm.data.HDF5Dataset(
            path=data_path,
            keys_to_load=list(cfg.dataset.keys_to_cache),
            keys_to_cache=list(cfg.dataset.keys_to_cache),
        )
        process, _ = build_normalization_process(dataset, cfg.dataset.keys_to_cache)
    missing = {"action_cartesian", "proprio", "goal_proprio"} - set(process)
    if missing:
        raise ValueError(f"World Model normalization missing keys: {sorted(missing)}")
    transform = {
        key: image_transform
        for key in ("pixels", "wrist_pixels", "goal", "goal_wrist_pixels")
    }
    return run_offline_imle_gpc(
        cfg, image_transform, world_model_loader, process, transform
    )
