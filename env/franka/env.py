#データセット生成用に作った環境

import os
os.environ["MUJOCO_GL"] = "egl"
import numpy as np
import torch
import yaml
import pandas as pd
import imageio
from tqdm import tqdm
from dm_control import mujoco
# from dm_control.utils.inverse_kinematics import qpos_from_site_pose
from env.franka.ik_with_limits import qpos_from_site_pose
from scipy.spatial.transform import Rotation as R




class FrankaSimEnv:
    def __init__(self, config):
        self.config = config

        self.MODEL_PATH = config["model_path"]
        self.CAMERA_NAME = config.get("camera_name", "")
        self.IMAGE_SIZE = tuple(config["image_size"])

        self.CONFIRM_IK = config["confirm_ik_result"]


        self.TOL = float(config["tol"])


        os.environ["MUJOCO_GL"] = "egl"
        self.physics = mujoco.Physics.from_xml_path(self.MODEL_PATH)

        if self.CAMERA_NAME == 'default':
            self.camera_id = -1
        else:
            self.camera_id = self.physics.model.name2id(self.CAMERA_NAME, mujoco.mjtObj.mjOBJ_CAMERA)
        
        self.target_sampling_step = config['target_sampling_step']


        arm_ids = []
        for i in range(1, 8):
            arm_ids.append(self.physics.model.name2id(f"actuator{i}", "actuator"))
        self.arm_actuator_ids = np.array(arm_ids, dtype=int)

        self.ctrlrange = self.physics.model.actuator_ctrlrange[self.arm_actuator_ids].copy()
        self.n_arm_act = len(self.arm_actuator_ids)

        # substeps: なければ envs と同じ 200 をデフォルトに
        self.substeps = config.get("substeps", 200)

        self.control_dt = float(self.physics.model.opt.timestep) * int(self.substeps)
        
        self.bluebox_geom_id = self.physics.model.name2id("blue_box", mujoco.mjtObj.mjOBJ_GEOM)




    def reset_and_place_all(self, box_pos, start_marker_pos=None, goal_marker_pos=None, init_position=None):
        self.physics.reset()
        
        
        ## reset直後
        blue = self.bluebox_geom_id
        self.physics.forward()
        p0 = self.physics.data.geom_xpos[blue].copy()
        print("box pos after reset:", p0)
        ###

        
        if init_position is not None:
            self.physics.data.qpos[:7] = init_position
            self.physics.data.qvel[:7] = 0

        joint_id = self.physics.model.name2id("free_joint_blue_box", "joint")
        start_idx = self.physics.model.jnt_qposadr[joint_id]
        self.physics.data.qpos[start_idx:start_idx+3] = box_pos
        self.physics.data.qpos[start_idx+3:start_idx+7] = np.array([1, 0, 0, 0])
        self.physics.data.qvel[start_idx:start_idx+6] = 0


        # 3) セット直後の確認
        self.physics.forward()
        p_set = self.physics.data.geom_xpos[blue].copy()
        # print("[DBG] box after set:", p_set, "delta_from_target:", p_set - box_pos)

        # # 4) 1 step後に動くか
        # self.physics.step()
        # self.physics.forward()
        # p1 = self.physics.data.geom_xpos[blue].copy()
        # print("[DBG] box after 1 step:", p1, "delta:", p1 - p_set)



        if start_marker_pos is not None:
            model_id = self.physics.model.name2id('start_marker', 'geom')
            self.physics.model.geom_pos[model_id][:3] = start_marker_pos
        if goal_marker_pos is not None:
            model_id = self.physics.model.name2id('goal_marker', 'geom')
            self.physics.model.geom_pos[model_id][:3] = goal_marker_pos
        
        self.physics.forward()
    
    def calc_inverse_kinematic(self, target_xyz, target_rotmat=None, rot_weight=1.0):
        target_quat = None
        if target_rotmat is not None:
            target_quat = R.from_matrix(target_rotmat).as_quat(scalar_first=True)
        # print('target_quat:', target_quat)
        
        joint_names = [f"joint{i}" for i in range(1, 8)]
        result = qpos_from_site_pose(
            self.physics,
            site_name="ee_target",
            target_pos=target_xyz,
            target_quat=target_quat,
            joint_names=joint_names,
            tol = 1e-4,
            rot_weight=rot_weight
        )
        return result
    
    
    def get_ee_position(self):
        try:
            sid = self.physics.model.name2id("ee_target", "site")
            return self.physics.data.site_xpos[sid].copy()
        except Exception:
            pass

        bid = self.physics.model.name2id("hand", "body")
        return self.physics.data.xpos[bid].copy()


    def step(self, action, max_dq=0.01):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        qpos = self.physics.data.qpos[:7].copy()

        dq_raw = action - qpos
        dq = np.clip(dq_raw, -max_dq, max_dq)

        
        target = qpos + dq

        low, high = self.ctrlrange[:, 0], self.ctrlrange[:, 1]
        target = np.clip(target, low, high)

        self.physics.data.ctrl[:] = 0.0
        self.physics.data.ctrl[self.arm_actuator_ids] = target

        for _ in range(self.substeps):
            self.physics.step()

        ee_pos = self.get_ee_position()
        return ee_pos

    def step_xyz(self, target_pos, target_rotmat=None,
                steps=200, tol=1e-3, rot_weight=1.0, max_dq=0.01):
        result = self.calc_inverse_kinematic(
            target_pos,
            target_rotmat=target_rotmat,
            rot_weight=rot_weight,
        )
        if not result.success:
            raise ValueError("IK failed!")
        joint_angles = result.qpos[:7].copy()

        objective_reached = False
        dist_steps = []

        for _ in range(steps):
            ee_pos = self.step(joint_angles, max_dq=max_dq)  # ← envs と同じ制御パイプライン
            dist_steps.append(np.abs(ee_pos - target_pos))

            dist = np.linalg.norm(ee_pos - target_pos)
            # if dist < tol:
            #     objective_reached = True
            #     break

        site_pos = ee_pos.copy()
        return joint_angles, site_pos, dist_steps, objective_reached


    def set_xyz(self, target_pos, target_rotmat=None, rot_weight=0.1,
                settle_steps=10000, sync_ctrl=True):
        """
        envs.FrankaSimEnv.set_xyz にできるだけ合わせた set_xyz（データセット用簡略版）

        - IK で q_des を求める
        - qpos, qvel, act, qacc_warmstart をきれいに初期化してから q_des を適用
        - sync_ctrl=True なら actuator の ctrl にも q_des をセット
        - settle_steps > 0 なら、その状態で physics.step() を何ステップか回して「落ち着かせる」
        """
        
        result = self.calc_inverse_kinematic(
            target_pos,
            target_rotmat=target_rotmat,
            rot_weight=rot_weight,
        )
        if not result.success:
            raise ValueError("IK failed!")

        q_des = result.qpos[:7].copy()



        self.physics.data.qpos[:7] = q_des
        self.physics.data.qvel[:]  = 0.0         
        self.physics.data.act[:]   = 0.0          
        self.physics.data.qacc_warmstart[:] = 0.0 
        self.physics.forward()


        if sync_ctrl:
            low, high = self.ctrlrange[:, 0], self.ctrlrange[:, 1]
            target = np.clip(q_des, low, high)
            self.physics.data.ctrl[:] = 0.0
            self.physics.data.ctrl[self.arm_actuator_ids] = target


        for _ in range(settle_steps):
            self.physics.step()

        ee_pos = self.get_ee_position()
        return q_des, ee_pos



    
    def render_image(self, size = (64, 64)):
        return self.physics.render(*size, camera_id = self.camera_id)
    

    def get_obs(self):
        qpos = self.physics.data.qpos[:7]
        qvel = self.physics.data.qvel[:7]
        ee = self.get_ee_position()
        return np.concatenate([qpos, qvel, ee])


    def check_ik_accuracy(self, target_xyz):
        self.physics.reset()
        
        # target_xyz = self.sample_uniform_xyz()
        result = self.calc_inverse_kinematic(target_xyz)
        
        if not result.success:
            print("⚠️ IK失敗しました!")
            return False

        self.physics.data.qpos[:7] = result.qpos[:7]
        self.physics.data.qvel[:7] = 0
        self.physics.forward() 

        ee_pos = self.get_ee_position()

        dx, dy, dz = ee_pos - target_xyz
        dist = np.linalg.norm([dx, dy, dz])
        tol = 1e-4
        flag = dist < tol


        self.physics.reset()
        return flag
