import os

os.environ["MUJOCO_GL"] = "egl"

import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm
import env.franka

def img_transform(cfg):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"

    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )
    return dataset

@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    """Run evaluation of dinowm vs random policy."""

    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"
    

    # print("cfg.policy:", cfg.policy)
    # print("swm.data.utils.get_cache_dir():", swm.data.utils.get_cache_dir())
    results_path = (
        Path(swm.data.utils.get_cache_dir(), "eval", cfg.policy).parent
        if cfg.policy != "random"
        else Path(__file__).parent
    ) 
    print("results_path:", results_path) #/home/shonosukehida/.stable_worldmodel/franka_push/pairs_100_ep_1_timestep_500_sample_mix_direction_towards_bluebox_1p00_1p00_view_top_reverse
    


    # create world environment
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    world = swm.World(**cfg.world, image_shape=(cfg.world.height, cfg.world.width))
    
    

    # create the transform
    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset  # get_dataset(cfg, cfg.dataset.stats)
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(stats_dataset.get_col_data(col_name), return_index=True)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != "action":
            process[f"goal_{col}"] = process[col]

    # -- run evaluation
    policy = cfg.get("policy", "random") #franka_push/pairs_100_ep_1_timestep_500_sample_mix_direction_towards_bluebox_1p00_1p00_view_top_reverse/lewm


    if policy != "random":
        model = swm.policy.AutoCostModel(cfg.policy)
        model = model.to("cuda")
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        # print("solver:", solver) #solver: <stable_worldmodel.solver.cem.CEMSolver object at 0x7f881e73a140>
        # print("policy config:", config) #PlanConfig(horizon=5, receding_horizon=5, history_len=1, action_block=1, warm_start=True)
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform
        )

    else:
        policy = swm.policy.RandomPolicy()

    print("world.set_policy(policy)...")
    world.set_policy(policy)        
    if cfg.eval.eval_zeroshot.execute:
    
        st_ps = list(cfg.eval.eval_zeroshot.start_positions)
        gl_ps = list(cfg.eval.eval_zeroshot.goal_positions)
        init_ee_ps = list(cfg.eval.eval_zeroshot.init_ee_positions)
        
        start_positions = np.repeat([st_ps], cfg.eval.num_eval, axis=0)
        goal_positions = np.repeat([gl_ps], cfg.eval.num_eval, axis=0)
        init_ee_positions = np.repeat([init_ee_ps], cfg.eval.num_eval, axis=0)
        
        video_dir = results_path / "zeroshot"
        video_dir.mkdir(parents=True, exist_ok=True)

        start_time = time.time()
        metrics = world.evaluate_zeroshot(
            start_positions=start_positions,
            goal_positions=goal_positions,
            init_ee_poses=init_ee_positions,
            eval_budget=cfg.eval.eval_budget,
            start_option_name="box_pos",
            goal_option_name="goal_marker_pos",
            start_info_name="bluebox_pos",
            goal_info_name="goal_pos",
            callables=[
                {
                    "method": "set_bluebox_pos",
                    "args": {
                        "bluebox_pos": {
                            "value": "start_positions",
                            "in_positions": True,
                        },
                    },
                },
                {
                    "method": "set_goal_pos",
                    "args": {
                        "goal_pos": {
                            "value": "goal_positions",
                            "in_positions": True,
                        },
                    },
                },
            ],
            video_path=video_dir,
        )
        end_time = time.time()
        
        print("==RESULTS==")
        print(f"metrics: {metrics}")
        print(f"evaluation_time: {end_time - start_time} seconds\n")
        
        log_path = video_dir / "zeroshot_results.txt"
        with log_path.open('a') as f:
            f.write("\n")  # separate from previous runs

            f.write("==== CONFIG ====\n")
            f.write(OmegaConf.to_yaml(cfg))
            f.write("\n")

            f.write("==== RESULTS ====\n")
            f.write(f"metrics: {metrics}\n")
            f.write(f"evaluation_time: {end_time - start_time} seconds\n")
            



    

    # sample the episodes and the starting indices
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.eval_tr_ds.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    # Map each dataset row’s episode_idx to its max_start_idx
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )

    # remove all the lines of dataset for which dataset['step_idx'] > max_start_per_row
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    print(valid_mask.sum(), "valid starting points found for evaluation.")

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
    )

    # sort increasingly to avoid issues with HDF5Dataset indexing
    random_episode_indices = np.sort(valid_indices[random_episode_indices])

    print(random_episode_indices)

    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")


    world.set_policy(policy)


    if cfg.eval.eval_tr_ds.execute:
        
    
        start_time = time.time()
        
        #学習データセット分布内テスト
        metrics = world.evaluate_from_dataset(
            dataset,
            start_steps=eval_start_idx.tolist(),
            goal_offset_steps=cfg.eval.eval_tr_ds.goal_offset_steps,
            eval_budget=cfg.eval.eval_budget,
            episodes_idx=eval_episodes.tolist(),
            callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
            video_path=results_path,
        )
        end_time = time.time()
        
        print(metrics)

        results_path = results_path / cfg.output.filename
        results_path.parent.mkdir(parents=True, exist_ok=True)

        with results_path.open("a") as f:
            f.write("\n")  # separate from previous runs

            f.write("==== CONFIG ====\n")
            f.write(OmegaConf.to_yaml(cfg))
            f.write("\n")

            f.write("==== RESULTS ====\n")
            f.write(f"metrics: {metrics}\n")
            f.write(f"evaluation_time: {end_time - start_time} seconds\n")
        
        


if __name__ == "__main__":
    run()
