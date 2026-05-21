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

from stable_worldmodel.probing.probe_evaluator import ProbingEvaluator
from env.franka.env import FrankaSimEnv


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
    
    keys_to_load = list(cfg.dataset.keys_to_cache)
    if "pixels" not in keys_to_load:
        keys_to_load.append("pixels")
    if "step_idx" not in keys_to_load:
        keys_to_load.append("step_idx")
    if "ep_idx" not in keys_to_load:
        keys_to_load.append("ep_idx")
    if "bluebox_pos" not in keys_to_load:
        keys_to_load.append("bluebox_pos")
    if "ee_pos" not in keys_to_load:
        keys_to_load.append("ee_pos")
    if "qpos" not in keys_to_load:
        keys_to_load.append("qpos") 
    if "qvel" not in keys_to_load:
        keys_to_load.append("qvel")
        
        
    dataset = swm.data.HDF5Dataset(
        dataset_name,
        keys_to_load=keys_to_load,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )
    return dataset

class SafeStandardScaler:
    def __init__(self, eps=1e-4):
        self.eps = eps
        self.mean_ = None
        self.scale_ = None
        self.raw_min_ = 1000000000000.
        self.raw_max_ = -1000000000000.
        self.normed_min_ = 1000000000000. 
        self.normed_max_ = -1000000000000.
        

    def fit(self, x):
        self.mean_ = np.mean(x, axis=0, keepdims=True)
        std = np.std(x, axis=0, keepdims=True)
        self.scale_ = np.where(std < self.eps, 1.0, std)
        return self

    def transform(self, x):
        return (x - self.mean_) / self.scale_

    def inverse_transform(self, x):
        return x * self.scale_ + self.mean_

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

    # print("cfg:", cfg)
    # print("cfg(keys_to_cache): ", cfg["dataset"]["keys_to_cache"])
    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    
    print("dataset.column_names:", dataset.column_names)
    
    
    
    
    stats_dataset = dataset  # get_dataset(cfg, cfg.dataset.stats)
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(stats_dataset.get_col_data(col_name), return_index=True)

    print("cfg.dataset.keys_to_cache:", cfg.dataset.keys_to_cache)
    
    process = {} #action.min, maxも持たせる?
    action_key = ""
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        # processor = preprocessing.StandardScaler()
        processor = SafeStandardScaler(eps=1e-4)
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        
        processor.raw_min_ = col_data.min(axis=0, keepdims=True)
        processor.raw_max_ = col_data.max(axis=0, keepdims=True)
        processor.normed_min_ = processor.transform(processor.raw_min_)
        processor.normed_max_ = processor.transform(processor.raw_max_)
        
        process[col] = processor

        action_keys = {"action", "action_cartesian", "action_joint"}
        if col not in action_keys:
            process[f"goal_{col}"] = process[col]
        else:
            action_key = col
            
        # if col in action_keys:
        #     print(f"\n================ {col} =================")

        #     print("raw mean:")
        #     print(col_data.mean(axis=0))

        #     print("raw std:")
        #     print(col_data.std(axis=0))

        #     print("raw min:")
        #     print(col_data.min(axis=0))

        #     print("raw max:")
        #     print(col_data.max(axis=0))

        #     # ============================================================
        #     # normalized
        #     # ============================================================
        #     normed = processor.transform(col_data)

        #     print("\n[normed]")

        #     print("mean:")
        #     print(normed.mean(axis=0))

        #     print("std:")
        #     print(normed.std(axis=0))

        #     print("min:")
        #     print(normed.min(axis=0))

        #     print("max:")
        #     print(normed.max(axis=0))

        #     # ============================================================
        #     # inverse transform check
        #     # ============================================================
        #     restored = processor.inverse_transform(normed)

        #     print("\n[restored diff]")

        #     print("mean abs diff:")
        #     print(np.abs(restored - col_data).mean(axis=0))

        #     print("max abs diff:")
        #     print(np.abs(restored - col_data).max(axis=0))
        # print("action_key:", action_key)
            
            
                

                        

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

        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform
        )
        
        # print("process.raw_min_:", policy.process["action"].raw_min_)
        # print("process.raw_max_:", policy.process["action"].raw_max_)
        # print("process.normed_min_:", policy.process["action"].normed_min_)
        # print("process.normed_max_:", policy.process["action"].normed_max_)

    else:
        policy = swm.policy.RandomPolicy()





    world.set_policy(policy, results_path)        
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
            plot_joint_compare_normed=cfg.eval.eval_zeroshot.plot_joint_compare_normed,
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


    if cfg.eval.compute_opt_action_cost.execute:
        dataset_for_action_cost = get_dataset(cfg, cfg.eval.ac_cost_dataset_name)

        col_name_cost = "episode_idx" if "episode_idx" in dataset_for_action_cost.column_names else "ep_idx"
        ep_indices_cost, _ = np.unique(
            dataset_for_action_cost.get_col_data(col_name_cost),
            return_index=True,
        )

        episode_len_cost = get_episodes_length(dataset_for_action_cost, ep_indices_cost)
        horizon = cfg.eval.compute_opt_action_cost.horizon

        max_start_idx_cost = episode_len_cost - horizon - 1
        max_start_idx_dict_cost = {
            ep_id: max_start_idx_cost[i]
            for i, ep_id in enumerate(ep_indices_cost)
        }

        max_start_per_row_cost = np.array([
            max_start_idx_dict_cost[ep_id]
            for ep_id in dataset_for_action_cost.get_col_data(col_name_cost)
        ])

        valid_mask_cost = dataset_for_action_cost.get_col_data("step_idx") <= max_start_per_row_cost
        valid_indices_cost = np.nonzero(valid_mask_cost)[0]

        print(valid_mask_cost.sum(), "valid starting points found for action-cost evaluation.")

        g = np.random.default_rng(cfg.seed)
        sampled_indices_cost = g.choice(
            valid_indices_cost,
            size=min(cfg.eval.num_eval, len(valid_indices_cost)),
            replace=False,
        )
        sampled_indices_cost = np.sort(sampled_indices_cost)

        eval_episodes_for_action_cost = dataset_for_action_cost.get_row_data(sampled_indices_cost)[col_name_cost]
        eval_start_idx_for_action_cost = dataset_for_action_cost.get_row_data(sampled_indices_cost)["step_idx"]

        cost_results = compute_action_costs(
            model=model,
            dataset=dataset_for_action_cost,
            eval_episodes=eval_episodes_for_action_cost.tolist(),
            eval_start_idx=eval_start_idx_for_action_cost.tolist(),
            horizon=horizon,
            transform=transform,
            process=process,
            action_key=action_key,
        )





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
    


    # print("cfg.probing.dataset_name:", cfg.probing.dataset_name)
    dataset = get_dataset(cfg, cfg.eval.probing.dataset_name)
    val_dataset = get_dataset(cfg, cfg.eval.probing.val_dataset_name)
    
    
    # print("model:", model)
    # print("model.predictor:", model.predictor)
    if cfg.eval.probing.exe_probe:
        results_path = (
            Path(swm.data.utils.get_cache_dir(), "eval", cfg.policy).parent
        ) 
        
        task_env_cfg = cfg.world.config 
        env = FrankaSimEnv(task_env_cfg)


        prober = ProbingEvaluator(
            dataset,
            model,
            transform = transform,
            process = process,
            results_path = results_path,
            val_dataset = val_dataset,
            plot_all_train_data = cfg.eval.probing.plot_all_train_data,
            plot_all_val_data = cfg.eval.probing.plot_all_val_data,
            plot_closed_data = cfg.eval.probing.plot_closed_data, 
            plot_open_data = cfg.eval.probing.plot_open_data,
            closed_pred_step = cfg.eval.probing.closed_pred_step,
            max_samples = cfg.eval.probing.max_samples,
            plot_line = cfg.eval.probing.plot_line,
            env = env
            
        )
        
        prober.run()
        
        


@torch.no_grad()
def compute_action_costs(
    model,
    dataset,
    eval_episodes,
    eval_start_idx,
    horizon,
    transform,
    process,
    action_key="action",  # or "action_cartesian", "action_joint"
):
    device = next(model.parameters()).device

    ep_idx_arr = np.array(eval_episodes)
    start_steps_arr = np.array(eval_start_idx)
    end_steps = start_steps_arr + horizon + 1
    
    print("ep_idx_arr:", ep_idx_arr) #[0]
    print("start_steps_arr:", start_steps_arr) #[88]

    data = dataset.load_chunk(ep_idx_arr, start_steps_arr, end_steps)

    dataset_costs = []
    random_costs = []
    zero_costs = []

    # random action 用pool
    action_pool = dataset.get_col_data(action_key)

    # print("type(data):", type(data))
    # print("len(data):", len(data))
    for i, ep in enumerate(data):

        # ============================================================
        # pixels
        # ============================================================
        pixels = ep["pixels"]
        if isinstance(pixels, torch.Tensor):
            pixels = pixels.cpu()

        # ============================================================
        # actions
        # ============================================================
        actions = ep[action_key]
        if isinstance(actions, torch.Tensor):
            actions = actions.cpu()

        H = 1

        # ============================================================
        # init / goal image
        # ============================================================
        init_pixels = pixels[:H]
        goal_pixels = pixels[-1:]

        init_pixels = torch.stack([
            transform["pixels"](p)
            for p in init_pixels
        ])

        goal_pixels = torch.stack([
            transform["goal"](p)
            for p in goal_pixels
        ])

        info = {
            "pixels": init_pixels[None, None].to(device),
            "goal": goal_pixels[None, None].to(device),
        }

        # ============================================================
        # dataset action
        # ============================================================
        dataset_action_seq = actions[:horizon]

        dataset_action_seq = (
            dataset_action_seq.numpy()
            if isinstance(dataset_action_seq, torch.Tensor)
            else np.asarray(dataset_action_seq)
        )
        

        # ============================================================
        # random action
        # ============================================================
        # shuffle_action_seq = dataset_action_seq.copy()
        # np.random.shuffle(shuffle_action_seq)


        # random_action_seq = shuffle_action_seq.astype(np.float32)
        # print("random_action_seq:", random_action_seq)
        if action_key == "action" or action_key == "action_joint":
            low = np.array([
                -2.8973,
                -1.7628,
                -2.8973,
                -3.0718,
                -2.8973,
                -0.0175,
                -2.8973,
            ], dtype=np.float32)

            high = np.array([
                2.8973,
                1.7628,
                2.8973,
                -0.0698,
                2.8973,
                3.7525,
                2.8973,
            ], dtype=np.float32)
            random_action_seq = np.random.uniform(
                low=low,
                high=high,
                size=(horizon, low.shape[0]),
            ).astype(np.float32)
        else:
            low = np.array([
                0.315,   # x
               -0.2,     # y
                0.1,     # z
            ], dtype=np.float32)

            high = np.array([
                0.715,   # x
                0.2,     # y
                0.1,     # z
            ], dtype=np.float32)

            random_action_seq = np.random.uniform(
                low=low,
                high=high,
                size=(horizon, low.shape[0]),
            ).astype(np.float32)
            

        # ============================================================
        # zero action
        # ============================================================
        zero_action_seq = np.zeros_like(dataset_action_seq)

        # ============================================================
        # normalize
        # ============================================================
        def normalize_action(action_seq):
            if action_key in process:
                return process[action_key].transform(action_seq)
            return action_seq

        dataset_action_seq = normalize_action(dataset_action_seq)
        random_action_seq = normalize_action(random_action_seq)
        zero_action_seq = normalize_action(zero_action_seq)

        # ============================================================
        # tensor helper
        # ============================================================
        def to_action_candidates(action_seq):
            return torch.as_tensor(
                action_seq[None, None],
                dtype=torch.float32,
                device=device,
            )

        dataset_candidates = to_action_candidates(dataset_action_seq)
        random_candidates = to_action_candidates(random_action_seq)
        zero_candidates = to_action_candidates(zero_action_seq)

        # ============================================================
        # cost
        # ============================================================
        dataset_cost = model.get_cost(info, dataset_candidates).item()
        random_cost = model.get_cost(info, random_candidates).item()
        zero_cost = model.get_cost(info, zero_candidates).item()

        dataset_costs.append(dataset_cost)
        random_costs.append(random_cost)
        zero_costs.append(zero_cost)

        # ============================================================
        # per-episode print
        # ============================================================
        print(f"\n[Episode {i}]")
        print(f"dataset_cost : {dataset_cost:.6f}")
        print(f"random_cost  : {random_cost:.6f}")
        print(f"zero_cost    : {zero_cost:.6f}")

    # ============================================================
    # summary
    # ============================================================
    dataset_costs = np.array(dataset_costs)
    random_costs = np.array(random_costs)
    zero_costs = np.array(zero_costs)

    print("\n==================== SUMMARY ====================")

    print("\n[DATASET ACTION COST]")
    print("mean:", dataset_costs.mean())
    print("min :", dataset_costs.min())
    print("max :", dataset_costs.max())
    print("std :", dataset_costs.std())

    print("\n[RANDOM ACTION COST]")
    print("mean:", random_costs.mean())
    print("min :", random_costs.min())
    print("max :", random_costs.max())
    print("std :", random_costs.std())

    print("\n[ZERO ACTION COST]")
    print("mean:", zero_costs.mean())
    print("min :", zero_costs.min())
    print("max :", zero_costs.max())
    print("std :", zero_costs.std())

    # ============================================================
    # comparison
    # ============================================================
    dataset_better_than_random = np.mean(dataset_costs < random_costs)
    dataset_better_than_zero = np.mean(dataset_costs < zero_costs)

    print("\n==================== COMPARISON ====================")
    print(
        f"dataset < random : "
        f"{dataset_better_than_random * 100:.2f}%"
    )

    print(
        f"dataset < zero   : "
        f"{dataset_better_than_zero * 100:.2f}%"
    )

    return {
        "dataset": dataset_costs,
        "random": random_costs,
        "zero": zero_costs,
    }

        
        


if __name__ == "__main__":
    run()
