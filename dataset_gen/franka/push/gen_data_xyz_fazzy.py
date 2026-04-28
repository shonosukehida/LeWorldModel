
import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)



import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import torch
import yaml
import pandas as pd
import imageio
from tqdm import tqdm
from dm_control import mujoco

from env.franka.env import FrankaSimEnv
from PIL import Image 
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import glob

from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt

import h5py

from stable_worldmodel.data.utils import get_cache_dir





class FrankaDatasetGenerator:
    def __init__(self, config):
        self.config = config
        self.PAIRS = config["pairs"]
        self.MIN_DIST = config["min_dist"]
        self.EPISODES_PER_PAIR = config["episodes_per_pair"]
        self.STEPS_PER_EPISODE = config["steps_per_episode"]
        self.IMAGE_SIZE = tuple(config["image_size"])
        self.MODEL_PATH = config["model_path"]
        self.CAMERA_NAME = config.get("camera_name", "")
        self.IS_VAL = config["is_val"]
        self.START_GOAL_X_RANGE = tuple(config["start_goal_x_range"])
        self.START_GOAL_Y_RANGE = tuple(config["start_goal_y_range"])
        self.START_GOAL_Z_RANGE = tuple(config["start_goal_z_range"])
        self.X_RANGE = tuple(config["x_range"])
        self.Y_RANGE = tuple(config["y_range"])
        self.Z_RANGE = tuple(config["z_range"])
        self.CONFIRM_IK = config["confirm_ik_result"]
        self.CONFIRM_DIST = config["confirm_dist"]
        # self.STEPS = config["steps"]
        self.MAX_DQ = config["max_dq"]
        self.SETTLE_STEPS = config["settle_steps"]
        self.box_start_pos = config.get("box_start_pos", None)
        if self.box_start_pos is not None: self.box_start_pos = np.array(self.box_start_pos)
        self.box_goal_pos = config.get("box_goal_pos", None)
        if self.box_goal_pos is not None: self.box_goal_pos = np.array(self.box_goal_pos)
        # print("[DBG] self.box_start_pos:", self.box_start_pos)
        # print("[DBG] self.box_start_pos.type:", type(self.box_start_pos))
        # print("[DBG] self.box_goal_pos:", self.box_goal_pos)
        # print("[DBG] self.box_goal_pos.type:", type(self.box_goal_pos))


        #単一/ミックスかを判定
        self.sample_method_list = config.get("sample_method_list", None)
        self.sample_method_ratio = config.get("sample_method_ratio", None)

        if self.sample_method_list is None:
            # 旧仕様: 単一メソッド
            self.single_method = config["sample_method"]
            print("sample_method(single):", self.single_method)
            self.method_schedule_per_pair = None
            sample_tag = self.single_method
        else:
            # 新仕様: 複数メソッド＋比率
            assert self.sample_method_ratio is not None, "sample_method_ratio is required when sample_method_list is set"
            assert len(self.sample_method_list) == len(self.sample_method_ratio), "sample_method_list and sample_method_ratio length mismatch"

            self.single_method = None
            self._build_method_schedules()
            print("sample_method_list:", self.sample_method_list)
            print("sample_method_ratio:", self.sample_method_ratio)
            sample_tag = "mix_" + "_".join(self.sample_method_list)
        self.sample_tag = sample_tag
        
        # self.SAMPLE_METHOD = config['sample_method']

        self.specify_init_position = config['specify_init_position']
        self.init_joint_method = config['init_joint_method']
        print("[DBG] init_joint_method:", self.init_joint_method)
        
        self.eval_only = self.config['eval_only']

        # self.SAVE_PATH = (
        #     f"pldm_envs/franka/presaved_datasets/val_pairs_{self.PAIRS}_ep_{self.EPISODES_PER_PAIR}_timestep_{self.STEPS_PER_EPISODE}_sample_{self.sample_tag}"
        #     if self.IS_VAL else
        #     f"pldm_envs/franka/presaved_datasets/pairs_{self.PAIRS}_ep_{self.EPISODES_PER_PAIR}_timestep_{self.STEPS_PER_EPISODE}_sample_{self.sample_tag}"
        # )
        
        
        # sample_tag を決めたあと
        if self.sample_method_ratio is None:
            ratio_tag = ""
        else:
            ratio_tag = "_".join(f"{r:.2f}".replace(".", "p") for r in self.sample_method_ratio)


        # if self.IS_VAL:
        #     if "mix" not in self.sample_tag:
        #         self.SAVE_PATH = f".../val_pairs_{...}_sample_{self.sample_tag}"
        #     else:
        #         self.SAVE_PATH = f".../val_pairs_{...}_sample_{self.sample_tag}_{ratio_tag}"
        # else:
        #     if "mix" not in self.sample_tag:
        #         self.SAVE_PATH = f".../pairs_{...}_sample_{self.sample_tag}"
        #     else:
        #         self.SAVE_PATH = f".../pairs_{...}_sample_{self.sample_tag}_{ratio_tag}"

        base = self.config.get("save_dir", "pldm_envs/franka/presaved_datasets")

        if self.sample_method_list is None:
            prefix = f"{'val_' if self.IS_VAL else ''}pairs_{self.PAIRS}_ep_{self.EPISODES_PER_PAIR}_timestep_{self.STEPS_PER_EPISODE}_sample_{self.sample_tag}_view_{self.CAMERA_NAME}"
        else:
            ratio_tag = "_".join(f"{r:.2f}".replace(".", "p") for r in self.sample_method_ratio)
            prefix = f"{'val_' if self.IS_VAL else ''}pairs_{self.PAIRS}_ep_{self.EPISODES_PER_PAIR}_timestep_{self.STEPS_PER_EPISODE}_sample_{self.sample_tag}_{ratio_tag}_view_{self.CAMERA_NAME}"

        self.SAVE_PATH = os.path.join(base, prefix)
        print("SAVE_PATH =", os.path.abspath(self.SAVE_PATH))
        self.prefix = prefix



        
        #データ確認のみの場合, 確認パスを指定
        if self.eval_only:
            self.SAVE_PATH = self.config['data_dir']
            
        
        
        os.makedirs(self.SAVE_PATH, exist_ok=True)

        os.environ["MUJOCO_GL"] = "egl"
        self.env = FrankaSimEnv(config)

        # 制御周波数をセット
        control_hz = config.get("control_hz", None)
        if control_hz is not None:
            self.actual_control_hz = self.set_control_frequency_by_substeps(self.env, control_hz)
        else:
            dt = float(self.env.physics.model.opt.timestep)
            self.actual_control_hz = 1.0 / (self.env.substeps * dt)
        
        # データセット周波数をセット
        dataset_hz = config.get("dataset_hz", 5.0)
        self.actual_dataset_hz = self.set_dataset_frequency_by_steps(
            self.actual_control_hz, dataset_hz
        )

        self.bluebox_geom_id = self.env.physics.model.name2id("blue_box", mujoco.mjtObj.mjOBJ_GEOM)
        
        self.franka_geom_ids = [
            self.env.physics.model.name2id("link0_c", mujoco.mjtObj.mjOBJ_GEOM),
            self.env.physics.model.name2id("link1_c", mujoco.mjtObj.mjOBJ_GEOM),
            self.env.physics.model.name2id("link2_c", mujoco.mjtObj.mjOBJ_GEOM),
            self.env.physics.model.name2id("link3_c", mujoco.mjtObj.mjOBJ_GEOM),
            self.env.physics.model.name2id("link4_c", mujoco.mjtObj.mjOBJ_GEOM),
            self.env.physics.model.name2id("link5_c0", mujoco.mjtObj.mjOBJ_GEOM),
            self.env.physics.model.name2id("link5_c1", mujoco.mjtObj.mjOBJ_GEOM),
            self.env.physics.model.name2id("link5_c2", mujoco.mjtObj.mjOBJ_GEOM),
            self.env.physics.model.name2id("link6_c", mujoco.mjtObj.mjOBJ_GEOM),
            self.env.physics.model.name2id("link7_c", mujoco.mjtObj.mjOBJ_GEOM),
            self.env.physics.model.name2id("hand_c", mujoco.mjtObj.mjOBJ_GEOM),
            self.env.physics.model.name2id("left_finger_0", mujoco.mjtObj.mjOBJ_GEOM),
            self.env.physics.model.name2id("right_finger_0", mujoco.mjtObj.mjOBJ_GEOM),
            # fingertip pads も入れる
            self.env.physics.model.name2id("fingertip_pad_collision_1", mujoco.mjtObj.mjOBJ_GEOM),
            self.env.physics.model.name2id("fingertip_pad_collision_2", mujoco.mjtObj.mjOBJ_GEOM),
            self.env.physics.model.name2id("fingertip_pad_collision_3", mujoco.mjtObj.mjOBJ_GEOM),
            self.env.physics.model.name2id("fingertip_pad_collision_4", mujoco.mjtObj.mjOBJ_GEOM),
            self.env.physics.model.name2id("fingertip_pad_collision_5", mujoco.mjtObj.mjOBJ_GEOM),
        ]

        print("[DBG] self.CAMERA_NAME:", self.CAMERA_NAME)
        if self.CAMERA_NAME == 'default':
            self.camera_id = -1
        else:
            self.camera_id = self.env.physics.model.name2id(self.CAMERA_NAME, mujoco.mjtObj.mjOBJ_CAMERA)
        
        self.count_objective_reached = 0 #手先が目標位置に到達した回数をカウント
        self.count_command_robot = 0 #ロボットに目標位置を指定した回数をカウント
        self.pair_list = self.get_start_goal_pairs()
        self.goal_obs_list = self.get_goal_obs_list()
        self.all_images = []
        self.data_list = []
        self.data_enums = {'target_pos':[], 'contact_count':[]} #目標直交座標を格納
        
        # self.mjc_steps = config['steps']
        self.target_sampling_step = config.get('target_sampling_step', 1)
        self.rot_weight = config['rot_weight']
        
        self.margin_ratio = self.config['margin_ratio']
        self.mgn_x_range = self.shrink_range(self.X_RANGE, ratio=self.margin_ratio)
        self.mgn_y_range = self.shrink_range(self.Y_RANGE, ratio=self.margin_ratio)
        self.mgn_z_range = self.shrink_range(self.Z_RANGE, ratio=self.margin_ratio)
        
        
        self.target_rotmat = None
        print('freeze_quat:', self.config["freeze_quat"])
        if self.config["freeze_quat"]:
            # 姿勢制御: hand を下に向けたい
            x = np.array(self.config['ee_x'])         
            y = np.array(self.config['ee_y'])
            z = np.array(self.config['ee_z'])        

            # 回転行列を構成（各軸を列に並べる）
            self.target_rotmat = np.stack([x, y, z], axis=1)
        
        #ロボットの周波数
        dt = float(self.env.physics.model.opt.timestep)
        control_dt = self.env.substeps * dt           # 1 control-step の時間

        # データセット1ステップの時間
        dataset_dt = self.mjc_steps * control_dt

        # target_xyz 更新の周期（秒）
        period_target = dataset_dt * self.target_sampling_step
        freq_target = 1.0 / period_target

        print(f'control frequency (env.step): {self.actual_control_hz:.3f} Hz')
        print(f'dataset frequency           : {self.actual_dataset_hz:.3f} Hz')
        print(f'target update frequency      : {freq_target:.3f} Hz')

        
        #逆運動学計算の確認
        loop = 100
        success_cnt = 0
        for _ in range(loop):
            target_xyz = self.sample_uniform_xyz(self.X_RANGE, self.Y_RANGE, self.Z_RANGE)
            success_cnt += int(self.env.check_ik_accuracy(target_xyz))
        success_rate = success_cnt / loop * 100 
        print(f'ik-calculation success rate: {success_rate:2f}')
        

        
        self.episode_chunk_size = config['episode_chunk_size']
        self.chunk_idx = 0
        
        #順運動学計算用env
        self.env_for_fk = FrankaSimEnv(config)


    def _get_center_of_cube(self):
        x_center = (self.X_RANGE[0] + self.X_RANGE[1]) / 2
        y_center = (self.Y_RANGE[0] + self.Y_RANGE[1]) / 2
        z_center = (self.Z_RANGE[0] + self.Z_RANGE[1]) / 2
        return np.array([x_center, y_center, z_center])
    
    def generate(self):
        #ik の計算結果の確認
        ik_idx = 0
        d_idx = 0
        count_all_timestep = 0
        out_of_bound_count = 0
        for pair_idx, (start_xyz, goal_xyz) in enumerate(tqdm(self.pair_list)):
            ep_idx = 0
            pbar = tqdm(total=self.EPISODES_PER_PAIR, desc=f"Episode Progress (pair {pair_idx})")

            while ep_idx < self.EPISODES_PER_PAIR + 1:
                valid_episode = True
                # 🔹 このエピソードで使うサンプリング手法を決める
                if self.sample_method_list is None:
                    # 単一モード
                    current_method = self.single_method
                else:
                    if ep_idx == 0:
                        # ウォームアップ用には、とりあえずこの pair の 0番目の method を使う
                        current_method = self.method_schedule_per_pair[pair_idx][0]
                    else:
                        current_method = self.method_schedule_per_pair[pair_idx][ep_idx - 1]

                
                if self.init_joint_method == 'random':
                    init_xyz = self.sample_uniform_xyz(self.mgn_x_range, self.mgn_y_range, self.mgn_z_range)
                elif self.init_joint_method == 'center':
                    init_xyz = self._get_center_of_cube()

                try:
                    result = self.env.calc_inverse_kinematic(init_xyz, target_rotmat=self.target_rotmat)
                except Exception as e:
                    print(f'IK失敗: {e}')
                    valid_episode = False 
                    continue
                
                init_joint = result.qpos[:7]
                self.env.reset_and_place_all(
                    box_pos=start_xyz, 
                    start_marker_pos=start_xyz, 
                    goal_marker_pos=goal_xyz, 
                    init_position=init_joint if self.specify_init_position else None
                    )
                self.env.physics.forward()
                for _ in range(10):
                    self.env.physics.forward()

                episode_images = []
                episode_obs = []
                episode_actions_joint = []
                episode_actions_cartesian = []
                ik_log = []
                dist_log = []
                dist_xyz_log = []
                
                episode_target_xyz = [] #指定直交座標を記録
                bluebox_contact_count = 0 #blue-box との衝突回数
                
                if ep_idx != 0:
                    self.env.physics.forward()
                    bluebox_geom_id = self.bluebox_geom_id
                    bluebox_pos = self.env.physics.data.geom_xpos[bluebox_geom_id]
                    
                    obs = np.concatenate(
                            [
                                self.env.physics.data.qpos[:7], 
                                self.env.physics.data.qvel[:7], 
                                self.env.get_ee_position(),
                                bluebox_pos,
                                ]
                        )
                    episode_obs.append(obs.copy())
                    img = self.env.render_image(size = self.IMAGE_SIZE)
                    # self.all_images.append(img)
                    episode_images.append(img)
                    
                    episode_target_xyz.append(self.env.get_ee_position())
                
                cur_yaw = None #self.SAMPLE_METHOD == 'direction'でvonmises分布の場合の水平角の初期化
                for idx in range(self.STEPS_PER_EPISODE):
                    if idx % self.target_sampling_step == 0:
                        self.count_command_robot += 1
                        if idx == 0: #初期IKの
                            target_xyz = init_xyz.copy()
                        else:
                            if current_method == 'uniform':
                                target_xyz = self.sample_uniform_xyz(self.mgn_x_range, self.mgn_y_range, self.mgn_z_range)
                                
                            elif current_method == 'direction':
                                current_pos = self.env.get_ee_position()
                                dist_range = tuple(self.config['sample_direction']['dist_range'])
                                max_loop = self.config['sample_direction']['max_loop']
                                freeze_z = self.config['sample_direction']['freeze_z']
                                kappa = self.config['sample_direction']['vonmises']['kappa']

                                target_xyz, cur_yaw = self.sample_direc_xyz(current_pos, dist_range, freeze_z=freeze_z, max_loop=max_loop, prev_yaw=cur_yaw, kappa=kappa)
                                
                            elif current_method == 'uniform_constrain':
                                current_pos = self.env.get_ee_position()
                                
                                max_dist = self.config['sample_uniform_constrain']['max_dist']
                                max_loop = self.config['sample_uniform_constrain']['max_loop']
                                target_xyz = self.sample_uniform_constrain_xyz(current_pos, max_dist, max_loop = max_loop)


                            elif current_method == 'towards_bluebox':
                                current_pos = self.env.get_ee_position()

                                # 現在の bluebox 位置を取得
                                bluebox_geom_id = self.bluebox_geom_id
                                bluebox_pos = self.env.physics.data.geom_xpos[bluebox_geom_id].copy()

                                tb_cfg = self.config.get('sample_towards_bluebox', {})
                                step_range = tuple(tb_cfg.get('step_range', (0.0025, 0.0025)))
                                near_threshold = tb_cfg.get('near_threshold', 0.05)
                                lateral_noise_std = tb_cfg.get('lateral_noise_std', 0.0)

                                target_xyz = self.sample_towards_bluebox_xyz(
                                    current_pos,
                                    bluebox_pos,
                                    step_range=step_range,
                                    near_threshold=near_threshold,
                                    lateral_noise_std=lateral_noise_std,
                                )
                            else:
                                raise ValueError(
                                    f"Unknown sampling method '{current_method}'. "
                                    f"Expected one of ['uniform', 'direction', 'uniform_constrain', 'towards_bluebox']."
                                    )
                    
                    try:
                        mjc_tol = float(self.config['tol'])
                        joint_angles, ee_pos, dist_steps, objective_reached = self.env.step_xyz(
                            target_xyz, 
                            target_rotmat=self.target_rotmat,
                            steps=self.mjc_steps, 
                            tol=mjc_tol,
                            rot_weight=self.rot_weight,
                            max_dq=self.MAX_DQ,
                            )

                    except Exception as e:
                        print(f'IK失敗: {e}')
                        valid_episode = False
                        if idx % self.target_sampling_step == 0:
                            self.count_command_robot -= 1
                        continue
                    
                    #衝突をカウント
                    ncon = self.env.physics.data.ncon
                    for i in range(ncon):
                        contact = self.env.physics.data.contact[i]
                        geom1 = contact.geom1
                        geom2 = contact.geom2

                        if (
                            (geom1 == self.bluebox_geom_id and geom2 in self.franka_geom_ids) or
                            (geom2 == self.bluebox_geom_id and geom1 in self.franka_geom_ids)
                        ):
                            bluebox_contact_count += 1
                    

                    
                    # 目標到達回数をカウント(/ 目標指示回数)
                    if (idx + 1) % self.target_sampling_step == 0:
                        if objective_reached:
                            self.count_objective_reached += 1
                            
                    if ep_idx != 0 and valid_episode:
                        command_target = int(idx % self.target_sampling_step == 0)
                        ik_log.append({'target_xyz': target_xyz, 'ee_pos': ee_pos})
                        dist_log.append(
                            {
                                'dist_start' : np.linalg.norm(dist_steps[0]), 
                                'dist_end' : np.linalg.norm(dist_steps[-1]), 
                                'command_target' : command_target
                                }
                            )
                        dist_xyz_log.append(
                                {
                                    'dist_start_x' : dist_steps[0][0], 
                                    'dist_goal_x' : dist_steps[-1][0], 
                                    'dist_start_y' : dist_steps[0][1],
                                    'dist_goal_y' : dist_steps[-1][1],
                                    'dist_start_z' : dist_steps[0][2],
                                    'dist_goal_z' : dist_steps[-1][2],
                                }
                            )

                        #手先が範囲外に出る回数をカウント
                        out_of_bound_count += int(not self.is_within_bounds(self.env.get_ee_position(), self.X_RANGE, self.Y_RANGE))
                        count_all_timestep += 1
                        
                        #bluebox の位置を取得
                        self.env.physics.forward()
                        bluebox_geom_id = self.bluebox_geom_id
                        bluebox_pos = self.env.physics.data.geom_xpos[bluebox_geom_id]
                        
                        
                        
                        
                        action_joint = joint_angles.copy()
                        action_cartesian = target_xyz.copy()
                        
                        obs = np.concatenate(
                                [
                                    self.env.physics.data.qpos[:7], 
                                    self.env.physics.data.qvel[:7], 
                                    self.env.get_ee_position(),
                                    bluebox_pos,
                                ]
                            )

                        episode_obs.append(obs.copy())
                        episode_actions_joint.append(action_joint.copy())
                        episode_actions_cartesian.append(action_cartesian)
                        img = self.env.physics.render(height=self.IMAGE_SIZE[0], width=self.IMAGE_SIZE[1], camera_id=self.camera_id)
                        # self.all_images.append(img)
                        episode_images.append(img)
                        
                        episode_target_xyz.append(target_xyz.copy())
                        
                if ep_idx != 0 and valid_episode:
                    self.data_list.append(
                                {
                                "observations": np.array(episode_obs),
                                "actions": np.array(episode_actions_joint),
                                "action_joint": np.array(episode_actions_joint),
                                "action_cartesian": np.array(episode_actions_cartesian),
                                "goal_obs": self.goal_obs_list[pair_idx][0].copy(),
                                "map_idx": pair_idx,
                            }
                        )
                    self.all_images.extend(episode_images)
                    
                    if (
                        self.episode_chunk_size is not None
                        and len(self.data_list) >= self.episode_chunk_size
                    ):
                        self._save_chunk()

                    
                    self.data_enums['target_pos'].append(np.array(episode_target_xyz))
                    self.data_enums['contact_count'].append(bluebox_contact_count)
                
                    if self.CONFIRM_IK:
                        self.confirm_target_actual_pos(ik_log, ik_idx)
                        ik_idx += 1
                    if self.CONFIRM_DIST:
                        self.confirm_target_actual_dist(dist_log, d_idx)
                        self.confirm_target_actual_dist_xyz(dist_xyz_log, d_idx)
                        d_idx += 1
                if ep_idx == 0: ep_idx += 1
                else:
                    if valid_episode: 
                        ep_idx += 1
                        pbar.update(1)
                    else: 
                        valid_episode = True
            pbar.close()



        reach_success_rate = (
            self.count_objective_reached / self.count_command_robot * 100 
            if self.count_command_robot > 0 
            else 0
            )
        out_of_bound_rate = (
            out_of_bound_count / count_all_timestep * 100
            if count_all_timestep > 0 
            else 0
        )
        print(f'percentage of targets reached : {reach_success_rate:2f}')
        print(f'percentage of out-of-bound : {out_of_bound_rate:2f}')

        
    def get_start_goal_pairs(self):
        print("📦 Sampling XYZ pairs...")
        pair_list = []

        
        while len(pair_list) < self.PAIRS:
            if (self.box_start_pos is None) or (self.box_goal_pos is None):
                start = self.sample_goal_xyz()
                goal = self.sample_goal_xyz()
                if np.linalg.norm(start - goal) >= self.MIN_DIST:
                    pair_list.append((start, goal))
            else:
                start = self.box_start_pos 
                goal = self.box_goal_pos
                pair_list.append((start, goal))
        
        return pair_list

    def get_goal_obs_list(self):
        print("🎯 Computing goal observations...")
        goal_obs_list = []
        for i, (start_pos, goal_pos) in enumerate(self.pair_list):
            print(f"{i}: start={start_pos}, goal={goal_pos}")
            self.env.reset_and_place_all(box_pos=goal_pos, start_marker_pos=start_pos, goal_marker_pos=goal_pos)
            bluebox_geom_id = self.bluebox_geom_id
            bluebox_pos = self.env.physics.data.geom_xpos[bluebox_geom_id]
            print(i, "goal_pos =", goal_pos, "actual bluebox_pos =", bluebox_pos)

            goal_arm_pos_center = bool(self.config['specify_goal_position'])

            if goal_arm_pos_center:
                self.env.set_xyz(
                    target_pos = self._get_center_of_cube(),
                    settle_steps=self.SETTLE_STEPS,
                    )
            else:
                offset = np.array(self.config['goal_offset'])
                self.env.set_xyz(
                    target_pos = goal_pos + offset,
                    settle_steps=self.SETTLE_STEPS,
                    )    

            self.env.physics.forward() 
            img = self.env.render_image(size=self.IMAGE_SIZE)
            goal_obs = np.concatenate(
                [
                    self.env.physics.data.qpos[:7], 
                    self.env.physics.data.qvel[:7], 
                    self.env.get_ee_position(),
                    bluebox_pos,
                    ]
                )
            goal_obs_list.append((goal_obs.copy(), img.copy()))

        return goal_obs_list

    def sample_xyz(self, range_tuple):
        return np.array([np.random.uniform(*r) for r in range_tuple])
    
    def sample_uniform_xyz(self, x_range, y_range, z_range):
        return self.sample_xyz([x_range, y_range, z_range])

    def sample_start_xyz(self):
        return self.sample_xyz([self.START_GOAL_X_RANGE, self.START_GOAL_Y_RANGE, self.START_GOAL_Z_RANGE])
    
    def sample_goal_xyz(self):
        return self.sample_xyz([self.START_GOAL_X_RANGE, self.START_GOAL_Y_RANGE, self.START_GOAL_Z_RANGE])
    
    def sample_direc_xyz(self, current_pos, dist_range, freeze_z=True, max_loop=100, prev_yaw=None, kappa=3.0):
        prob_dist = self.config['sample_direction']['prob_dist']

        flag = False
        for _ in range(max_loop):
            
            if prob_dist == 'uniform':
                cur_yaw = np.random.uniform(- np.pi, np.pi)
            elif prob_dist == 'vonmises':
                if prev_yaw is None:
                    cur_yaw = np.random.uniform(- np.pi, np.pi)
                else:
                    cur_yaw = np.random.vonmises(mu=prev_yaw, kappa=kappa)
            else:
                raise ValueError(f"Unknown prob_dist '{prob_dist}'. Expected 'uniform' or 'vonmises'.")
            
            # cur_yaw = (cur_yaw + np.pi) % (2 * np.pi) - np.pi
            
            if not freeze_z:
                pitch = np.random.uniform(- np.pi / 2, np.pi / 2)
            else:
                pitch = 0
            
            
            dist = np.random.uniform(*dist_range)
            
            dx = dist * np.cos(pitch) * np.cos(cur_yaw)
            dy = dist * np.cos(pitch) * np.sin(cur_yaw)
            dz = dist * np.sin(pitch)
            
            delta = np.array([dx, dy, dz])
            new_pos = current_pos + delta
            

            if self.is_within_bounds(new_pos, self.mgn_x_range, self.mgn_y_range, self.mgn_z_range):
                flag = True
                return new_pos, cur_yaw

        if not flag:
            # print('max_loop reached in sample_direc_xyz')
            if not self.is_within_bounds(new_pos, self.X_RANGE, self.Y_RANGE):
                cur_yaw = (cur_yaw + np.pi) % (2 * np.pi) - np.pi
                # cur_yaw = np.random.uniform(- np.pi, np.pi)
                pass
            x = np.clip(new_pos[0], *self.mgn_x_range)
            y = np.clip(new_pos[1], *self.mgn_y_range)
            z = np.clip(new_pos[2], *self.mgn_z_range)
            new_pos = np.array([x, y, z])
        
        return new_pos, cur_yaw

    def shrink_range(self, range_tuple, ratio=0.8):
        min_val, max_val = range_tuple
        center = (min_val + max_val) / 2 
        half_width = (max_val - min_val) / 2
        new_half_width = half_width * ratio 
        return (center - new_half_width, center + new_half_width)

    
    def sample_uniform_constrain_xyz(self, current_pos, max_dist = 0.1, max_loop = 100):
        for _ in range(max_loop):
            target_xyz = self.sample_xyz([self.X_RANGE, self.Y_RANGE, self.Z_RANGE])
            dist = np.linalg.norm(target_xyz - current_pos)
            if dist < max_dist:
                break
            else:
                target_xyz = current_pos.copy()
                x = np.clip(target_xyz[0], *self.X_RANGE)
                y = np.clip(target_xyz[1], *self.Y_RANGE)
                z = np.clip(target_xyz[2], *self.Z_RANGE)
                target_xyz = np.array([x, y, z])
            return target_xyz

    
    def is_within_bounds(self, pos, x_range, y_range, z_range=None):
        x, y, z = pos 
        if z_range is not None:
            return (x_range[0] <= x <= x_range[1] and
                    y_range[0] <= y <= y_range[1] and
                    z_range[0] <= z <= z_range[1])
        else:
            return (x_range[0] <= x <= x_range[1] and
                    y_range[0] <= y <= y_range[1])


    def sample_towards_bluebox_xyz(
        self,
        current_pos,
        bluebox_pos,
        step_range=(0.0025, 0.0025),
        near_threshold=0.05,
        lateral_noise_std=0.0,
    ):
        """
        EE から bluebox 中心方向へ距離を縮める target をサンプルする。
        current_pos, bluebox_pos: np.array shape (3,)
        """

        current_pos = np.asarray(current_pos, dtype=np.float32)
        bluebox_pos = np.asarray(bluebox_pos, dtype=np.float32)

        dir_vec = bluebox_pos - current_pos
        dist = np.linalg.norm(dir_vec)

        # すでに十分近いときは，その場＋少しノイズにする
        if dist < near_threshold:
            target = current_pos.copy()
            if lateral_noise_std > 0.0:
                noise = np.random.normal(scale=lateral_noise_std, size=3)
                target = target + noise
        else:
            # 中心を飛び越えないように step をクリップ
            step_min, step_max = step_range
            step = np.random.uniform(step_min, step_max)
            step = min(step, dist)  # center を超えない

            dir_unit = dir_vec / (dist + 1e-8)
            target = current_pos + step * dir_unit

            # 必要なら軽い横ノイズ
            if lateral_noise_std > 0.0:
                noise = np.random.normal(scale=lateral_noise_std, size=3)
                target = target + noise

        # ワークスペース内にクリップ
        x = np.clip(target[0], *self.mgn_x_range)
        y = np.clip(target[1], *self.mgn_y_range)
        z = np.clip(target[2], *self.mgn_z_range)
        target = np.array([x, y, z], dtype=np.float32)

        return target

    def _build_method_schedules(self):
        """
        sample_method_list と ratio に基づいて，
        各 pair ごとに「エピソードごとのサンプリング手法の並び」を作る。

        method_schedule_per_pair[ pair_idx ] は長さ EPISODES_PER_PAIR のリストで，
        各要素が 'direction' や 'towards_bluebox' などのメソッド名。
        """
        methods = list(self.sample_method_list)
        ratios = np.array(self.sample_method_ratio, dtype=np.float32)
        ratios = ratios / ratios.sum()

        # まず floor でベースを計算
        base_counts = np.floor(ratios * self.EPISODES_PER_PAIR).astype(int)
        # 端数を埋める
        remainder = self.EPISODES_PER_PAIR - int(base_counts.sum())
        for i in range(remainder):
            base_counts[i % len(base_counts)] += 1

        print("per-pair episode counts:", dict(zip(methods, base_counts)))

        self.method_schedule_per_pair = []
        for _ in range(self.PAIRS):
            schedule = []
            for m, c in zip(methods, base_counts):
                schedule += [m] * c
            np.random.shuffle(schedule)  # エピソード順はランダムに
            assert len(schedule) == self.EPISODES_PER_PAIR
            self.method_schedule_per_pair.append(schedule)

    
    def confirm_target_actual_pos(self, ik_log, ik_idx):
        os.makedirs('robot_sim/data_value/ik_value', exist_ok=True)
        df = pd.DataFrame(ik_log)
        df.to_csv(f'robot_sim/data_value/ik_value/ik_log{ik_idx}.csv', index=False)
    
    
    def confirm_target_actual_dist(self, dist_log, d_idx):
        os.makedirs('robot_sim/data_value/dist_value', exist_ok=True)
        df = pd.DataFrame(dist_log)
        df.to_csv(f'robot_sim/data_value/dist_value/dist_log{d_idx}.csv', index=False)
    
    def confirm_target_actual_dist_xyz(self, dist_xyz_log, d_idx):
        os.makedirs('robot_sim/data_value/dist_xyz_value', exist_ok=True)
        df = pd.DataFrame(dist_xyz_log)
        df.to_csv(f'robot_sim/data_value/dist_xyz_value/dist_xyz_log{d_idx}.csv', index=False)

    def set_control_frequency_by_substeps(self, env, control_hz: float):
        m = env.physics.model
        dt = float(m.opt.timestep)           # 物理の基本タイムステップ [s]
        # 制御周期 1/control_hz を dt の整数倍で近似
        sub = max(1, int(round((1.0 / control_hz) / dt)))
        env.substeps = sub
        env.control_dt = sub * dt            # 1 control step あたり時間 [s]

        actual_hz = 1.0 / (sub * dt)
        print(f"[CTRL FREQ] target={control_hz:.3f} Hz -> substeps={sub}, actual≈{actual_hz:.3f} Hz")
        return actual_hz

    def set_dataset_frequency_by_steps(self, control_hz: float, dataset_hz: float):
        """
        control_hz: Franka の制御周波数 [Hz]
        dataset_hz: データとして記録したい周波数 [Hz]
        """
        if dataset_hz > control_hz:
            # 制御ループより速くサンプリングはできないので clamp
            print(f"[WARN] dataset_hz={dataset_hz} > control_hz={control_hz}. "
                f"Clamping dataset_hz to control_hz.")
            dataset_hz = control_hz

        # steps = control_hz / dataset_hz
        raw_steps = control_hz / dataset_hz
        steps = max(1, int(round(raw_steps)))

        actual_dataset_hz = control_hz / steps
        print(f"[DATASET FREQ] target={dataset_hz:.3f} Hz -> steps={steps}, "
            f"actual≈{actual_dataset_hz:.3f} Hz")

        self.mjc_steps = steps  
        return actual_dataset_hz




    def _save_chunk(self):
        chunk_dir = os.path.join(self.SAVE_PATH, "chunks")
        os.makedirs(chunk_dir, exist_ok=True)
        data_chunk_dir = os.path.join(chunk_dir, "data")
        os.makedirs(data_chunk_dir, exist_ok=True)
        image_chunk_dir = os.path.join(chunk_dir, "image")
        os.makedirs(image_chunk_dir, exist_ok=True)
        

        # --- save data_list ---
        data_path = os.path.join(
            data_chunk_dir, f"data_chunk_{self.chunk_idx}.pt"
        )
        torch.save(self.data_list, data_path)

        # --- save images ---
        if len(self.all_images) > 0:
            images_arr = np.array(self.all_images, dtype=np.uint8)
            img_path = os.path.join(
                image_chunk_dir, f"images_chunk_{self.chunk_idx}.npy"
            )
            np.save(img_path, images_arr)

        print(f"✅ Saved chunk {self.chunk_idx} ({len(self.data_list)} episodes)")

        # --- flush ---
        self.data_list = []
        self.all_images = []
        self.chunk_idx += 1


    def merge_chunks(self, h5_name=f"push"):
        """
        chunks/data/data_chunk_*.pt と chunks/image/images_chunk_*.npy を読み込み,
        LeWM / HDF5Dataset が読める .h5 を保存する。

        保存される主な key:
        - pixels      : (N, H, W, C) uint8
        - action      : (N, 7) float32
        - action_joint : (N, 7) float32
        - action_cartesian : (N, 3) float32
        - ep_len      : (E,) int32
        - ep_offset   : (E,) int64

        optional:
        - qpos        : (N, 7) float32
        - qvel        : (N, 7) float32
        - ee_pos      : (N, 3) float32
        - bluebox_pos : (N, 3) float32
        - ep_idx      : (N,) int32
        - step_idx    : (N,) int32
        """
        chunk_dir = os.path.join(self.SAVE_PATH, "chunks")
        data_chunk_dir = os.path.join(chunk_dir, "data")
        image_chunk_dir = os.path.join(chunk_dir, "image")

        data_files = sorted(glob.glob(os.path.join(data_chunk_dir, "data_chunk_*.pt")))
        image_files = sorted(glob.glob(os.path.join(image_chunk_dir, "images_chunk_*.npy")))

        if len(data_files) == 0:
            raise FileNotFoundError(f"No data chunks found in {data_chunk_dir}")
        if len(image_files) == 0:
            raise FileNotFoundError(f"No image chunks found in {image_chunk_dir}")

        # ---- chunk を順に読む ----
        all_episodes = []
        for f in data_files:
            chunk_data = torch.load(f, weights_only=False)
            all_episodes.extend(chunk_data)
            print(f"Loaded {f} with {len(chunk_data)} episodes")

        images_list = []
        for f in image_files:
            arr = np.load(f)
            images_list.append(arr)
            print(f"Loaded {f} with shape {arr.shape}")

        merged_images = np.concatenate(images_list, axis=0) if len(images_list) > 0 else np.array([], dtype=np.uint8)

        # ---- episode ごとに整列して HDF5 用に flatten ----
        pixels_all = []
        actions_all = []
        action_joint_all = []
        action_cartesian_all = []
        qpos_all = []
        qvel_all = []
        ee_pos_all = []
        bluebox_pos_all = []
        ep_len = []
        ep_offset = []

        ep_idx_all = []
        step_idx_all = []

        img_cursor = 0
        global_offset = 0

        for ep_idx, ep in enumerate(all_episodes):
            obs = np.asarray(ep["observations"])   # shape: (T+1, obs_dim)
            acts = np.asarray(ep["actions"])       # shape: (T, 7)
            acts_joint = np.asarray(ep["action_joint"])
            acts_cartesian = np.asarray(ep["action_cartesian"])
            
            # print("acts.shape:", acts.shape)
            # print("acts_joint.shape: ", acts_joint.shape)
            # print("acts_cartesian.shape: ", acts_cartesian.shape)

            T = acts.shape[0]

            # 今の生成コードでは observations は T+1, actions は T のはず
            if obs.shape[0] != T + 1:
                raise ValueError(
                    f"Episode {ep_idx}: observations length {obs.shape[0]} != actions length+1 {T+1}"
                )

            # images も T+1 枚ある前提
            ep_images = merged_images[img_cursor: img_cursor + (T + 1)]
            if ep_images.shape[0] != T + 1:
                raise ValueError(
                    f"Episode {ep_idx}: images length {ep_images.shape[0]} != expected {T+1}"
                )
            img_cursor += (T + 1)

            # LeWM 用には pixels[t], action[t] で次を予測したいので
            # images[:-1] と actions[:] を対応させる
            pixels_ep = ep_images[:-1]              # (T, H, W, C)
            action_ep = acts.astype(np.float32)     # (T, 7)
            action_joint_ep = acts_joint.astype(np.float32)
            action_cartesian_ep = acts_cartesian.astype(np.float32)
            
            # print("action_ep.shape: ", action_ep.shape)
            # print("action_joint_ep.shape:", action_joint_ep.shape)
            # print("action_cartesian_ep.shape:", action_cartesian_ep.shape)

            # observations の中身:
            # [qpos(7), qvel(7), ee_pos(3), bluebox_pos(3)] を想定
            # これも pixels と揃えて obs[:-1] 側を使う
            obs_ep = obs[:-1]

            qpos_ep = obs_ep[:, 0:7].astype(np.float32)
            qvel_ep = obs_ep[:, 7:14].astype(np.float32)
            ee_pos_ep = obs_ep[:, 14:17].astype(np.float32)
            bluebox_pos_ep = obs_ep[:, 17:20].astype(np.float32)

            pixels_all.append(pixels_ep.astype(np.uint8))
            actions_all.append(action_ep)
            action_joint_all.append(action_joint_ep)
            action_cartesian_all.append(action_cartesian_ep)
            qpos_all.append(qpos_ep)
            qvel_all.append(qvel_ep)
            ee_pos_all.append(ee_pos_ep)
            bluebox_pos_all.append(bluebox_pos_ep)

            ep_len.append(T)
            ep_offset.append(global_offset)

            ep_idx_all.append(np.full((T,), ep_idx, dtype=np.int32))
            step_idx_all.append(np.arange(T, dtype=np.int32))

            global_offset += T

        # 連結
        pixels_all = np.concatenate(pixels_all, axis=0)
        actions_all = np.concatenate(actions_all, axis=0)
        action_joint_all = np.concatenate(action_joint_all, axis=0)
        action_cartesian_all = np.concatenate(action_cartesian_all, axis=0)
        qpos_all = np.concatenate(qpos_all, axis=0)
        qvel_all = np.concatenate(qvel_all, axis=0)
        ee_pos_all = np.concatenate(ee_pos_all, axis=0)
        bluebox_pos_all = np.concatenate(bluebox_pos_all, axis=0)
        ep_len = np.asarray(ep_len, dtype=np.int32)
        ep_offset = np.asarray(ep_offset, dtype=np.int64)
        ep_idx_all = np.concatenate(ep_idx_all, axis=0)
        step_idx_all = np.concatenate(step_idx_all, axis=0)

        # ---- 保存先 ----
        # 既存の stable_worldmodel 流儀に合わせて ~/.stable_worldmodel/datasets/<name>.h5 に置く
        from stable_worldmodel.data.utils import get_cache_dir

        datasets_dir = get_cache_dir(sub_folder="datasets")
        h5_path = os.path.join(datasets_dir, f"franka/{self.prefix}/{h5_name}.h5")
        os.makedirs(os.path.dirname(h5_path), exist_ok=True)

        # ---- HDF5 書き出し ----
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("pixels", data=pixels_all, compression="gzip")
            f.create_dataset("action", data=actions_all, compression="gzip")
            f.create_dataset("action_joint", data=action_joint_all, compression="gzip")
            f.create_dataset("action_cartesian", data=action_cartesian_all, compression="gzip")

            f.create_dataset("ep_len", data=ep_len)
            f.create_dataset("ep_offset", data=ep_offset)

            # optional keys
            f.create_dataset("qpos", data=qpos_all)
            f.create_dataset("qvel", data=qvel_all)
            f.create_dataset("ee_pos", data=ee_pos_all)
            f.create_dataset("bluebox_pos", data=bluebox_pos_all)
            f.create_dataset("ep_idx", data=ep_idx_all)
            f.create_dataset("step_idx", data=step_idx_all)

        print("✅ Saved LeWM-style HDF5 dataset!")
        print(f"   path: {h5_path}")
        print(f"   pixels: {pixels_all.shape}, dtype={pixels_all.dtype}")
        print(f"   action: {actions_all.shape}, dtype={actions_all.dtype}")
        print(f"   action_joint: {action_joint_all.shape}, dtype={actions_all.dtype}")
        print(f"   action_cartesian: {action_cartesian_all.shape}, dtype={actions_all.dtype}")
        print(f"   ep_len: {ep_len.shape}, total episodes={len(ep_len)}")


    def _resolve_h5_path(self, h5_name="push.h5"):
        

        candidates = []

        data_dir = self.config.get("data_dir", None)
        if data_dir is not None:
            candidates.append(os.path.join(data_dir, h5_name))

            base_name = os.path.basename(os.path.normpath(data_dir))
            datasets_dir = get_cache_dir(sub_folder="datasets")
            candidates.append(os.path.join(datasets_dir, "franka", base_name, h5_name))

        datasets_dir = get_cache_dir(sub_folder="datasets")
        candidates.append(os.path.join(datasets_dir, "franka", self.prefix, h5_name))

        for p in candidates:
            if os.path.exists(p):
                return p

        raise FileNotFoundError("push.h5 not found. candidates:\n" + "\n".join(candidates))


    def add_action_field(self, h5_name="push.h5", overwrite=False, batch_size=10000):
        h5_path = self._resolve_h5_path(h5_name)
        print(f"[ADD ACTION FIELD] target h5: {h5_path}")

        with h5py.File(h5_path, "a") as f:
            if "action" not in f:
                raise KeyError("Existing dataset must have key 'action'.")

            action = f["action"]
            if action.ndim != 2 or action.shape[1] != 7:
                raise ValueError(f"'action' must be joint action with shape (N, 7), got {action.shape}")

            N = action.shape[0]

            if "action_joint" in f:
                if overwrite:
                    del f["action_joint"]
                else:
                    print("[SKIP] action_joint already exists")
            if "action_cartesian" in f:
                if overwrite:
                    del f["action_cartesian"]
                else:
                    print("[SKIP] action_cartesian already exists")

            if "action_joint" not in f:
                f.create_dataset(
                    "action_joint",
                    data=action[:].astype(np.float32),
                    compression="gzip",
                )
                print("✅ created action_joint")

            if "action_cartesian" not in f:
                dset = f.create_dataset(
                    "action_cartesian",
                    shape=(N, 3),
                    dtype=np.float32,
                    compression="gzip",
                )

                for s in tqdm(range(0, N, batch_size), desc="Computing FK action_cartesian"):
                    e = min(s + batch_size, N)
                    joints = action[s:e]

                    xyz = np.zeros((e - s, 3), dtype=np.float32)
                    for i, q in enumerate(joints):
                        xyz[i] = self.env_for_fk.calc_forward_kinematics(q)

                    dset[s:e] = xyz

                print("✅ created action_cartesian")

        print("✅ add_action_field finished")
    
    # def _test_fk(self, target_xyz):
        
    #     print("type(target_xyz): ", type(target_xyz))
    #     print("target_xyz:", target_xyz)
        
    #     result = self.env_for_fk.calc_inverse_kinematic(target_xyz)
    #     target_joint = result.qpos[:7]
        
    #     print("result:", result)
    #     print("target_joint:", target_joint)
        
    #     recon_xyz = self.env_for_fk.calc_forward_kinematics(target_joint)
        
    #     print("type(recon_xyz):", type(recon_xyz))
    #     print("recon_xyz:", recon_xyz)
        
        
    #     diff = np.linalg.norm(target_xyz - recon_xyz, ord=2)
    #     print("diff(L2):", diff)
        
    #     return 
    
    # def _test_loop(self, loop = 10):
    #     success = 0
        
    #     # 範囲
    #     x_range = [0.315, 0.515]
    #     y_range = [-0.2, 0.2]
    #     z_range = [0.05, 0.05]
        
    #     eps = 1e-3
        
    #     diffs = []
        
    #     for _ in range(loop):

    #         # 1サンプル取得
    #         x = np.random.uniform(*x_range)
    #         y = np.random.uniform(*y_range)
    #         z = np.random.uniform(*z_range)

    #         target_xyz = np.array([x, y, z])
            
            
    #         result = self.env_for_fk.calc_inverse_kinematic(target_xyz)
    #         target_joint = result.qpos[:7]
            
    #         recon_xyz = self.env_for_fk.calc_forward_kinematics(target_joint)
    #         diff = np.linalg.norm(target_xyz - recon_xyz, ord=2)
    #         diffs.append(diff)
            
            
    #         if diff < eps:
    #             success += 1
        
    #     print("success rate:", success / loop * 100. , "%")
    #     print("num_success: ", success, "num_trials:", loop)
        
        
    #     print("diff log")
    #     for i, d in enumerate(diffs):
    #         print("i: ", i, "d: ", d)
            
            
    







if __name__ == "__main__":
    
    with open("dataset_gen/franka/push/config.yaml", "r") as f:
        config = yaml.safe_load(f)


    dataset_generator = FrankaDatasetGenerator(config)
    if not config['eval_only']:
        dataset_generator.generate()

        if len(dataset_generator.data_list) > 0:
            dataset_generator._save_chunk()
        dataset_generator.merge_chunks()
    
    if config['add_action_field']:
        dataset_generator.add_action_field()
    
    
        
    # dataset_generator.confirm_data_architecture()
    # dataset_generator.confirm_data()
    # if config['make_video']: 
    #     dataset_generator.make_video()
    # if config['confirm_ee_trajectory']:
    #     vis_target_traj = (not config['eval_only']) and config['visualize_target_trajectory']
    #     dataset_generator.confirm_endeffector_trajectory('xy', vis_target_traj)
    #     if config['confirm_ee_traj_xz']: 
    #         dataset_generator.confirm_endeffector_trajectory('xz', vis_target_traj)  
    # if config.get("plot_all_ee_traj", False):
    #     dataset_generator.plot_all_endeffector_trajectories(
    #         axes="xy",
    #         stride=1,         # 重いなら 5 や 10 に
    #         max_episodes=None, # 重すぎるなら 2000 とか
    #         alpha = 1.0,          # 重ね描きの濃さ
    #         lw = 1.8,
    #     )
    
    # if config.get("plot_all_bluebox_traj", False):
    #     dataset_generator.plot_all_bluebox_trajectories(
    #         axes="xy", 
    #         stride=1, 
    #         alpha=1.0, 
    #         lw=1.8,
    #         )
 
    print("finished!!")