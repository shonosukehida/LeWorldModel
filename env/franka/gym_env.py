import copy
from typing import Any, Dict, Optional

import gymnasium as gym
from gymnasium import spaces
import numpy as np

from env.franka.env import FrankaSimEnv

from env.franka.ik_with_limits import *

DEFAULT_CONFIG: Dict[str, Any] = {
    # 必要に応じてあなたの環境に合わせて修正
    "model_path": "",
    "camera_name": "default",
    "image_size": [64, 64],
    "confirm_ik_result": False,
    "tol": 1e-4,
    "target_sampling_step": 1,
    "substeps": 200,
}


class FrankaPushEnv(gym.Env):
    """
    Gymnasium wrapper for the user's custom Franka MuJoCo environment.

    目的:
    - gym.make("swm/FrankaPush-v0") で生成できるようにする
    - LeWorldModel / stable-worldmodel の eval から呼べる最低限の API を揃える
    - 既存の FrankaSimEnv を再利用する

    観測:
    - Dict observation
      {
        "pixels": uint8 image, shape=(H, W, 3)
        "state":  float32 vector, shape=(17,)  # qpos(7) + qvel(7) + ee_pos(3)
      }

    行動:
    - 7次元の関節目標値（FrankaSimEnv.step にそのまま渡す）
    """

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        max_episode_steps: int = 50,
        render_mode: Optional[str] = None,
        model_path: Optional[str] = None,
        camera_name: Optional[str] = None,
        image_size: Optional[list] = None,
        confirm_ik_result: Optional[bool] = None,
        tol: Optional[float] = None,
        target_sampling_step: Optional[int] = None,
        substeps: Optional[int] = None,
        **kwargs,
    ):
        super().__init__()

        self.config = copy.deepcopy(DEFAULT_CONFIG)
        if config is not None:
            self.config.update(config)

        if model_path is not None:
            self.config["model_path"] = model_path
        if camera_name is not None:
            self.config["camera_name"] = camera_name
        if image_size is not None:
            self.config["image_size"] = image_size
        if confirm_ik_result is not None:
            self.config["confirm_ik_result"] = confirm_ik_result
        if tol is not None:
            self.config["tol"] = tol
        if target_sampling_step is not None:
            self.config["target_sampling_step"] = target_sampling_step
        if substeps is not None:
            self.config["substeps"] = substeps

        if "image_shape" in kwargs:
            image_shape = kwargs["image_shape"]
            if isinstance(image_shape, (tuple, list)) and len(image_shape) == 2:
                self.config["image_size"] = [int(image_shape[0]), int(image_shape[1])]

        if not self.config["model_path"]:
            raise ValueError(
                "FrankaPushEnv requires config['model_path'] to be set. "
                "Please pass it from the env registration or world config."
            )

        self.sim = FrankaSimEnv(self.config)
        self.max_episode_steps = int(max_episode_steps)
        self.render_mode = render_mode

        self.image_size = tuple(self.config["image_size"])
        self._step_count = 0

        # Goal は最初は optional
        self.goal_pos: Optional[np.ndarray] = None

        # Action space:
        # FrankaSimEnv.step は absolute joint target を期待している
        low = self.sim.ctrlrange[:, 0].astype(np.float32)
        high = self.sim.ctrlrange[:, 1].astype(np.float32)
        self.action_space = spaces.Box(
            low=low,
            high=high,
            shape=(7,),
            dtype=np.float32,
        )

        # Observation space:
        # get_obs() = qpos(7) + qvel(7) + ee(3) -> 17
        h, w = self.image_size
        self.observation_space = spaces.Dict(
            {
                "pixels": spaces.Box(
                    low=0,
                    high=255,
                    shape=(h, w, 3),
                    dtype=np.uint8,
                ),
                "state": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(17,),
                    dtype=np.float32,
                ),
            }
        )
        


    # -----------------------------
    # Internal helpers
    # -----------------------------
    def _get_obs(self) -> Dict[str, np.ndarray]:
        pixels = self.sim.render_image(self.image_size)
        state = self.sim.get_obs().astype(np.float32)
        return {
            "pixels": pixels,
            "state": state,
        }

    def _get_bluebox_pos(self) -> np.ndarray:
        """
        HDF5 の bluebox_pos と揃えたいので、env 内からも取れるようにする。
        geom_xpos を使って blue box の world position を取得。
        """
        self.sim.physics.forward()
        pos = self.sim.physics.data.geom_xpos[self.sim.bluebox_geom_id].copy()
        return pos.astype(np.float32)

    def _compute_reward(self) -> float:
        """
        最初は簡易報酬:
        goal_pos があれば、bluebox と goal の距離の負値
        goal_pos がなければ 0
        """
        if self.goal_pos is None:
            return 0.0

        bluebox_pos = self._get_bluebox_pos()
        dist = np.linalg.norm(bluebox_pos - self.goal_pos)
        return float(-dist)

    def _check_success(self) -> bool:
        """
        簡易 success 判定:
        blue box が goal_pos から tol 以内
        """

        if self.goal_pos is None:
            return False

        bluebox_pos = self._get_bluebox_pos()
        dist = np.linalg.norm(bluebox_pos - self.goal_pos)
        tol = float(self.config.get("tol", 1e-4))
        return bool(dist < tol)

    # -----------------------------
    # Gymnasium API
    # -----------------------------
    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        print("[env/franka/gym_env.py] options:", options)
        super().reset(seed=seed)

        self._step_count = 0
        self.goal_pos = None

        # options から初期状態を受け取れるようにしておく
        options = options or {}

        box_pos = np.asarray(
            options.get("box_pos", [0.5, 0.0, 0.02]),
            dtype=np.float32,
        )

        start_marker_pos = options.get("start_marker_pos", None)
        goal_marker_pos = options.get("goal_marker_pos", None)
        init_ee_pos = options.get("init_ee_pos", None)
        
        
        # init_position = options.get("init_position", None)
        

        self.sim.reset_and_place_all(
            box_pos=box_pos,
            start_marker_pos=start_marker_pos,
            goal_marker_pos=goal_marker_pos,
            init_ee_pos=init_ee_pos,
        )

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action: np.ndarray):
        self._step_count += 1

        action = np.asarray(action, dtype=np.float32).reshape(7,)
        self.sim.step(action)

        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._check_success()
        truncated = self._step_count >= self.max_episode_steps

        info = {
            "step_count": self._step_count,
            "bluebox_pos": self._get_bluebox_pos(),
            "ee_pos" : obs["state"][-3:].copy(),
            "goal_pos": None if self.goal_pos is None else self.goal_pos.copy(),
            "is_success": terminated,
        }

        return obs, reward, terminated, truncated, info

    def render(self):
        return self.sim.render_image(self.image_size)

    def close(self):
        # dm_control physics には明示 close が必須ではないことが多い
        # 必要ならここに後で追記
        pass

    # -----------------------------
    # Setters for evaluation
    # -----------------------------
    def set_state(self, qpos: np.ndarray, qvel: np.ndarray):
        """
        Robot arm state を dataset から直接流し込む。
        """
        qpos = np.asarray(qpos, dtype=np.float32).reshape(7,)
        qvel = np.asarray(qvel, dtype=np.float32).reshape(7,)

        self.sim.physics.data.qpos[:7] = qpos
        self.sim.physics.data.qvel[:7] = qvel
        self.sim.physics.forward()

    def set_bluebox_pos(self, bluebox_pos: np.ndarray):
        """
        blue box の自由関節位置を直接セット。
        """
        bluebox_pos = np.asarray(bluebox_pos, dtype=np.float32).reshape(3,)

        joint_id = self.sim.physics.model.name2id("free_joint_blue_box", "joint")
        start_idx = self.sim.physics.model.jnt_qposadr[joint_id]

        self.sim.physics.data.qpos[start_idx:start_idx + 3] = bluebox_pos
        self.sim.physics.data.qpos[start_idx + 3:start_idx + 7] = np.array(
            [1.0, 0.0, 0.0, 0.0], dtype=np.float32
        )
        self.sim.physics.data.qvel[start_idx:start_idx + 6] = 0.0
        self.sim.physics.forward()

    def set_goal_pos(self, goal_pos: np.ndarray):
        """
        評価時の目標位置を保存し、goal marker があれば見た目も更新する。
        """
        goal_pos = np.asarray(goal_pos, dtype=np.float32).reshape(3,)
        self.goal_pos = goal_pos.copy()

        # goal marker が存在すれば見た目も更新
        try:
            model_id = self.sim.physics.model.name2id("goal_marker", "geom")
            self.sim.physics.model.geom_pos[model_id][:3] = goal_pos
            self.sim.physics.forward()
        except Exception:
            # marker がない環境でも落とさない
            pass

    def set_start_marker_pos(self, start_marker_pos: np.ndarray):
        """
        必要なら start marker も更新できるようにする。
        """
        start_marker_pos = np.asarray(start_marker_pos, dtype=np.float32).reshape(3,)
        try:
            model_id = self.sim.physics.model.name2id("start_marker", "geom")
            self.sim.physics.model.geom_pos[model_id][:3] = start_marker_pos
            self.sim.physics.forward()
        except Exception:
            pass