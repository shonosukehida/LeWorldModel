from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from collections.abc import Callable

import numpy as np
import torch
from loguru import logger as logging
from torchvision import tv_tensors

import stable_worldmodel as swm
from stable_worldmodel.solver import Solver
import gymnasium as gym




from stable_worldmodel.plot.plot import plot_cem_cost_convergence, plot_cem_sequence_transition_colormap
from stable_worldmodel.reward import latent_goal_reward


@dataclass(frozen=True)
class PlanConfig:
    """Configuration for the MPC planning loop.

    Attributes:
        horizon: Planning horizon in number of steps.
        receding_horizon: Number of steps to execute before re-planning.
        history_len: Number of past observations to consider.
        action_block: Number of times each action is repeated (frameskip).
        warm_start: Whether to use the previous plan to initialize the next one.
    """

    horizon: int
    receding_horizon: int
    history_len: int = 1
    action_block: int = 1
    warm_start: bool = True
    action_space: str = ""
    clip_action: bool = True

    @property
    def plan_len(self) -> int:
        """Total plan length in environment steps."""
        return self.horizon * self.action_block


class Transformable(Protocol):
    """Protocol for reversible data transformations (e.g., normalizers, scalers)."""

    def transform(self, x: np.ndarray) -> np.ndarray:  # pragma: no cover
        """Apply preprocessing to input data.

        Args:
            x: Input data as a numpy array.

        Returns:
            Preprocessed data as a numpy array.
        """
        ...

    def inverse_transform(
        self, x: np.ndarray
    ) -> np.ndarray:  # pragma: no cover
        """Reverse the preprocessing transformation.

        Args:
            x: Preprocessed data as a numpy array.

        Returns:
            Original data as a numpy array.
        """
        ...


class Actionable(Protocol):
    """Protocol for model action computation."""

    def get_action(info) -> torch.Tensor:  # pragma: no cover
        """Compute action from observation and goal"""
        ...


class BasePolicy:
    """Base class for agent policies.

    Attributes:
        env: The environment the policy is associated with.
        type: A string identifier for the policy type.
    """

    env: Any
    type: str

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the base policy.

        Args:
            **kwargs: Additional configuration parameters.
        """
        self.env = None
        self.type = 'base'
        for arg, value in kwargs.items():
            setattr(self, arg, value)

    def get_action(self, obs: Any, **kwargs: Any) -> np.ndarray:
        """Get action from the policy given the observation.

        Args:
            obs: The current observation from the environment.
            **kwargs: Additional parameters for action selection.

        Returns:
            Selected action as a numpy array.

        Raises:
            NotImplementedError: If not implemented by a subclass.
        """
        raise NotImplementedError

    def set_env(self, env: Any) -> None:
        """Associate this policy with an environment.

        Args:
            env: The environment to associate.
        """
        self.env = env

    def _prepare_info(self, info_dict: dict) -> dict[str, torch.Tensor]:
        """Pre-process and transform observations.

        Applies preprocessing (via `self.process`) and transformations (via `self.transform`)
        to observation data. Used by subclasses like FeedForwardPolicy and WorldModelPolicy.

        Args:
            info_dict: Raw observation dictionary from the environment.

        Returns:
            A dictionary of processed tensors.

        Raises:
            ValueError: If an expected numpy array is missing for processing.
        """
        for k, v in info_dict.items():
            is_numpy = isinstance(v, (np.ndarray | np.generic))

            if hasattr(self, 'process') and k in self.process:
                if not is_numpy:
                    raise ValueError(
                        f"Expected numpy array for key '{k}' in process, got {type(v)}"
                    )

                # flatten extra dimensions if needed
                shape = v.shape
                if len(shape) > 2:
                    v = v.reshape(-1, *shape[2:])

                # process and reshape back
                v = self.process[k].transform(v)
                v = v.reshape(shape)

            # collapse env and time dimensions for transform (e, t, ...) -> (e * t, ...)
            # then restore after transform
            image_keys = {"pixels", "wrist_pixels", "goal", "goal_wrist_pixels",}
            
            if hasattr(self, 'transform') and k in self.transform:
                shape = None
                if is_numpy or torch.is_tensor(v):
                    if v.ndim > 2:
                        shape = v.shape
                        v = v.reshape(-1, *shape[2:])
                if k in image_keys:
                    # permute channel first for transform
                    if is_numpy:
                        v = np.transpose(v, (0, 3, 1, 2))
                    else:
                        v = v.permute(0, 3, 1, 2)
                v = torch.stack(
                    [self.transform[k](tv_tensors.Image(x)) for x in v]
                )
                is_numpy = isinstance(v, (np.ndarray | np.generic))

                if shape is not None:
                    v = v.reshape(*shape[:2], *v.shape[1:])

            if is_numpy and v.dtype.kind not in 'USO':
                v = torch.from_numpy(v)

            info_dict[k] = v

        return info_dict


class RandomPolicy(BasePolicy):
    """Policy that samples random actions from the action space."""

    def __init__(self, seed: int | None = None, **kwargs: Any) -> None:
        """Initialize the random policy.

        Args:
            seed: Optional random seed for the action space.
            **kwargs: Additional configuration parameters.
        """
        super().__init__(**kwargs)
        self.type = 'random'
        self.seed = seed

    def get_action(self, obs: Any, **kwargs: Any) -> np.ndarray:
        """Get a random action from the environment's action space.

        Args:
            obs: The current observation (ignored).
            **kwargs: Additional parameters (ignored).

        Returns:
            A randomly sampled action.
        """
        return self.env.action_space.sample()

    def set_seed(self, seed: int) -> None:
        """Set the random seed for action sampling.

        Args:
            seed: The seed value.
        """
        if self.env is not None:
            self.env.action_space.seed(seed)


class ExpertPolicy(BasePolicy):
    """Policy using expert demonstrations or heuristics."""

    def __init__(self, **kwargs: Any) -> None:
        """Initialize the expert policy.

        Args:
            **kwargs: Additional configuration parameters.
        """
        super().__init__(**kwargs)
        self.type = 'expert'

    def get_action(
        self, obs: Any, goal_obs: Any, **kwargs: Any
    ) -> np.ndarray | None:
        """Get action from the expert policy.

        Args:
            obs: The current observation.
            goal_obs: The goal observation.
            **kwargs: Additional parameters.

        Returns:
            The expert action, or None if not available.
        """
        # Implement expert policy logic here
        pass


class FeedForwardPolicy(BasePolicy):
    """Feed-Forward Policy using a neural network model.

    Actions are computed via a single forward pass through the model.
    Useful for imitation learning policies like Goal-Conditioned Behavioral Cloning (GCBC).

    Attributes:
        model: Neural network model implementing the Actionable protocol.
        process: Dictionary of data preprocessors for specific keys.
        transform: Dictionary of tensor transformations (e.g., image transforms).
    """

    def __init__(
        self,
        model: Actionable,
        process: dict[str, Transformable] | None = None,
        transform: dict[str, Callable[[torch.Tensor], torch.Tensor]]
        | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize the feed-forward policy.

        Args:
            model: Neural network model with a `get_action` method.
            process: Dictionary of data preprocessors for specific keys.
            transform: Dictionary of tensor transformations (e.g., image transforms).
            **kwargs: Additional configuration parameters.
        """
        super().__init__(**kwargs)
        self.type = 'feed_forward'
        self.model = model.eval()
        self.process = process or {}
        self.transform = transform or {}

    def get_action(self, info_dict: dict, **kwargs: Any) -> np.ndarray:
        """Get action via a forward pass through the neural network model.

        Args:
            info_dict: Current state information containing at minimum a 'goal' key.
            **kwargs: Additional parameters (unused).

        Returns:
            The selected action as a numpy array.

        Raises:
            AssertionError: If environment not set or 'goal' not in info_dict.
        """
        assert hasattr(self, 'env'), 'Environment not set for the policy'
        assert 'goal' in info_dict, "'goal' must be provided in info_dict"

        # Prepare the info dict (transforms and normalizes inputs)
        info_dict = self._prepare_info(info_dict)

        # Add goal_pixels key for GCBC model
        if 'goal' in info_dict:
            info_dict['goal_pixels'] = info_dict['goal']

        # Move all tensors to the model's device
        device = next(self.model.parameters()).device
        for k, v in info_dict.items():
            if torch.is_tensor(v):
                info_dict[k] = v.to(device)

        # Get action from model
        with torch.no_grad():
            action = self.model.get_action(info_dict)

        # Convert to numpy
        if torch.is_tensor(action):
            action = action.cpu().detach().numpy()

        # post-process action
        if 'action' in self.process:
            action = self.process['action'].inverse_transform(action)

        return action


class DiffusionPolicy(BasePolicy):
    def __init__(
        self,
        model,
        obs_encoder,
        noise_scheduler,
        pred_horizon: int,
        obs_horizon: int,
        action_horizon: int,
        action_dim: int,
        num_inference_steps: int | None = None,
        process=None,
        transform=None,
        wrist_obs_encoder=None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.type = "diffusion"

        self.model = model
        self.obs_encoder = obs_encoder
        self.wrist_obs_encoder = wrist_obs_encoder

        self.noise_scheduler = noise_scheduler

        self.pred_horizon = pred_horizon
        self.obs_horizon = obs_horizon
        self.action_horizon = action_horizon
        self.action_dim = action_dim

        self.process = process or {}
        self.transform = transform or {}

        if num_inference_steps is None:
            num_inference_steps = (
                noise_scheduler.config.num_train_timesteps
            )

        self.num_inference_steps = num_inference_steps

    @property
    def device(self):
        return next(self.model.parameters()).device

    @property
    def dtype(self):
        return next(self.model.parameters()).dtype

    def conditional_sample(
        self,
        condition_data: torch.Tensor,
        global_cond: torch.Tensor,
        generator=None,
    ) -> torch.Tensor:
        """
        Run reverse diffusion and generate an action trajectory.

        Args:
            condition_data:
                Initial trajectory tensor.
                Shape: (B, pred_horizon, action_dim)

            global_cond:
                Encoded observation condition.
                Shape: (B, obs_feature_dim * obs_horizon)

        Returns:
            trajectory:
                Normalized action trajectory.
                Shape: (B, pred_horizon, action_dim)
        """

        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )

        self.noise_scheduler.set_timesteps(
            self.num_inference_steps
        )

        for t in self.noise_scheduler.timesteps:

            # predict noise
            model_output = self.model(
                trajectory,
                t,
                local_cond=None,
                global_cond=global_cond,
            )

            # x_t -> x_(t-1)
            trajectory = self.noise_scheduler.step(
                model_output,
                t,
                trajectory,
                generator=generator,
            ).prev_sample

        return trajectory

    def predict_action(
        self,
        info_dict: dict,
    ) -> dict[str, torch.Tensor]:
        """
        Predict one action trajectory from the current observation.

        Returns:
            {
                "action":      actions actually used by the policy,
                "action_pred": full predicted trajectory
            }
        """
        
        info_dict = dict(info_dict)
        info_dict = self._prepare_info(info_dict)

        if "pixels" not in info_dict:
            raise KeyError(
                "'pixels' must be provided for DiffusionPolicy"
            )

        global_cond, B, To = self._encode_observation(
            pixels=info_dict["pixels"],
            wrist_pixels=info_dict.get(
                "wrist_pixels",
                None,
            ),
        )


        # --------------------------------------------------
        # Reverse diffusion
        # --------------------------------------------------

        condition_data = torch.zeros(
            (
                B,
                self.pred_horizon,
                self.action_dim,
            ),
            device=self.device,
            dtype=self.dtype,
        )

        naction_pred = self.conditional_sample(
            condition_data=condition_data,
            global_cond=global_cond,
        )

        # --------------------------------------------------
        # Action chunk
        # --------------------------------------------------

        start = To - 1
        end = start + self.action_horizon

        action = naction_pred[:, start:end]

        return {"action": action, "action_pred": naction_pred,}

    def get_action(
        self,
        info_dict: dict,
        **kwargs,
    ):
        """
        Standalone Diffusion Policy inference.
        """

        with torch.no_grad():
            result = self.predict_action(
                info_dict
            )

        action = result["action"]

        # For standalone execution:
        # currently return first action of action chunk
        action = action[:, 0]

        action = action.detach().cpu().numpy()

        # --------------------------------------------------
        # Denormalize
        # --------------------------------------------------

        if "action_cartesian" in self.process:
            action = self.process[
                "action_cartesian"
            ].inverse_transform(action)

        elif "action" in self.process:
            action = self.process[
                "action"
            ].inverse_transform(action)

        elif "action_joint" in self.process:
            action = self.process[
                "action_joint"
            ].inverse_transform(action)

        return action

    def sample_action_sequences(
        self,
        info_dict: dict,
        num_samples: int = 1,
        denormalize: bool = False,
    ) -> torch.Tensor:
        """
        Sample multiple candidate action trajectories.

        Mainly used by GPCPolicy.

        Returns:
            Tensor:
                (B, K, pred_horizon, action_dim)
        """

        info_dict = dict(info_dict)
        info_dict = self._prepare_info(info_dict)

        if "pixels" not in info_dict:
            raise KeyError(
                "'pixels' must be provided for DiffusionPolicy"
            )

        global_cond, B, To = self._encode_observation(
            pixels=info_dict["pixels"],
            wrist_pixels=info_dict.get(
                "wrist_pixels",
                None,
            ),
        )

        global_cond = (
            global_cond[:, None, :]
            .expand(
                B,
                num_samples,
                global_cond.shape[-1],
            )
            .reshape(
                B * num_samples,
                -1,
            )
        )
        



        # --------------------------------------------------
        # Generate K trajectories
        # --------------------------------------------------

        condition_data = torch.zeros(
            (
                B * num_samples,
                self.pred_horizon,
                self.action_dim,
            ),
            device=self.device,
            dtype=self.dtype,
        )

        with torch.no_grad():
            trajectories = self.conditional_sample(
                condition_data=condition_data,
                global_cond=global_cond,
            )

        trajectories = trajectories.reshape(
            B,
            num_samples,
            self.pred_horizon,
            self.action_dim,
        )


        if denormalize:
            shape = trajectories.shape
            traj_np = trajectories.detach().cpu().numpy()
            traj_np = traj_np.reshape(-1, shape[-1])

            if "action_cartesian" in self.process:
                traj_np = self.process["action_cartesian"].inverse_transform(traj_np)
            elif "action" in self.process:
                traj_np = self.process["action"].inverse_transform(traj_np)
            elif "action_joint" in self.process:
                traj_np = self.process["action_joint"].inverse_transform(traj_np)

            trajectories = torch.from_numpy(traj_np.reshape(shape)).to(device=self.device, dtype=self.dtype,)

        return trajectories

    def compute_loss(
        self,
        batch: dict,
    ) -> torch.Tensor:
        """
        Diffusion Policy training loss.

        Expected:
            batch["pixels"]:
                (B, To, C, H, W)

            batch["action"]:
                (B, pred_horizon, action_dim)

        action is assumed to already be normalized.
        """
        pixels = batch["pixels"]

        wrist_pixels = batch.get(
            "wrist_pixels",
            None,
        )

        actions = batch["action"].to(
            device=self.device,
            dtype=self.dtype,
        )

        global_cond, B, To = self._encode_observation(
            pixels=batch["pixels"],
            wrist_pixels=batch.get(
                "wrist_pixels",
                None,
            ),
        )

        # --------------------------------------------------
        # Sample diffusion noise
        # --------------------------------------------------

        noise = torch.randn_like(actions)

        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (B,),
            device=self.device,
        ).long()

        noisy_actions = (
            self.noise_scheduler.add_noise(
                actions,
                noise,
                timesteps,
            )
        )

        # --------------------------------------------------
        # Predict noise
        # --------------------------------------------------

        noise_pred = self.model(
            noisy_actions,
            timesteps,
            local_cond=None,
            global_cond=global_cond,
        )

        prediction_type = (
            self.noise_scheduler.config.prediction_type
        )

        if prediction_type == "epsilon":
            target = noise

        elif prediction_type == "sample":
            target = actions

        else:
            raise ValueError(
                "Unsupported prediction type: "
                f"{prediction_type}"
            )

        loss = torch.nn.functional.mse_loss(
            noise_pred,
            target,
        )

        return loss


    def _encode_observation(
        self,
        pixels,
        wrist_pixels=None,
    ):
        pixels = pixels.to(
            device=self.device,
            dtype=self.dtype,
        )

        B = pixels.shape[0]

        if self.wrist_obs_encoder is None:
            To = min(
                self.obs_horizon,
                pixels.shape[1],
            )

            pixels = pixels[:, :To]

            pixels_flat = pixels.reshape(
                B * To,
                *pixels.shape[2:],
            )

            obs_features = self.obs_encoder(
                pixels_flat
            )

            if not torch.is_tensor(obs_features):
                raise TypeError(
                    "obs_encoder must return a torch.Tensor"
                )

            global_cond = obs_features.reshape(
                B,
                -1,
            )

            return global_cond, B, To

        if wrist_pixels is None:
            raise KeyError(
                "'wrist_pixels' is required for multiview DiffusionPolicy"
            )

        wrist_pixels = wrist_pixels.to(
            device=self.device,
            dtype=self.dtype,
        )

        if pixels.shape[:2] != wrist_pixels.shape[:2]:
            raise ValueError(
                "pixels and wrist_pixels must have the same "
                f"(B, T), got {pixels.shape[:2]} "
                f"and {wrist_pixels.shape[:2]}"
            )

        To = min(
            self.obs_horizon,
            pixels.shape[1],
            wrist_pixels.shape[1],
        )

        pixels = pixels[:, :To]
        wrist_pixels = wrist_pixels[:, :To]

        pixels_flat = pixels.reshape(
            B * To,
            *pixels.shape[2:],
        )

        wrist_pixels_flat = wrist_pixels.reshape(
            B * To,
            *wrist_pixels.shape[2:],
        )

        overhead_features = self.obs_encoder(
            pixels_flat
        )

        wrist_features = self.wrist_obs_encoder(
            wrist_pixels_flat
        )

        if not torch.is_tensor(overhead_features):
            raise TypeError(
                "obs_encoder must return a torch.Tensor"
            )

        if not torch.is_tensor(wrist_features):
            raise TypeError(
                "wrist_obs_encoder must return a torch.Tensor"
            )

        obs_features = torch.cat(
            [
                overhead_features,
                wrist_features,
            ],
            dim=-1,
        )

        global_cond = obs_features.reshape(
            B,
            -1,
        )

        return global_cond, B, To





class GPCPolicy(BasePolicy):
    """
    GPC-RANK policy.

    1. DiffusionPolicy generates K candidate action sequences.
    2. World Model rolls out each candidate.
    3. reward_fn evaluates the predicted trajectories.
    4. The highest-reward candidate is selected.

    GPC-OPT is not implemented here.
    """

    def __init__(
        self,
        diffusion_policy: DiffusionPolicy,
        world_model,
        reward_fn,
        num_candidates: int = 50,
        action_horizon: int | None = None,
        process = None,
        transform = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.type = "gpc"

        self.diffusion_policy = diffusion_policy
        self.world_model = world_model

        # reward_fn(
        #     rollout_output,
        #     info_dict,
        # ) -> Tensor of shape (B, K)
        self.reward_fn = reward_fn

        self.num_candidates = num_candidates

        if action_horizon is None:
            action_horizon = diffusion_policy.action_horizon

        self.action_horizon = action_horizon
        
        
        #LeWM用統計
        self.process = process or {}
        self.transform = transform or {}

    @property
    def device(self):
        return self.diffusion_policy.device


    def rollout(
        self,
        info_dict: dict,
        action_sequences: torch.Tensor,
    ):
        """
        Roll out K candidate action sequences with LeWM.

        Args:
            info_dict:
                Current observation information.

            action_sequences:
                Candidate actions for LeWM.
                Shape: (B, K, T, action_dim)

        Returns:
            rollout_output:
                Dictionary containing predicted embeddings.
                rollout_output["predicted_emb"]:
                    (B, K, T_pred, latent_dim)
        """

        wm_info = dict(info_dict)
        wm_info = self._prepare_info(wm_info)

        B, K, T, D = action_sequences.shape


        for key, value in list(wm_info.items()):
            if not torch.is_tensor(value): continue

            if value.shape[0] != B: continue

            wm_info[key] = (value[:, None].expand(B, K, *value.shape[1:],))


        device = next(self.world_model.parameters()).device

        for key, value in wm_info.items():
            if torch.is_tensor(value):
                wm_info[key] = value.to(device)

        action_sequences = action_sequences.to(device)

        goal_emb = self.encode_goal(info_dict)

        rollout_output = self.world_model.rollout(wm_info, action_sequences,)
        
        rollout_output["goal_emb"] = goal_emb

        return rollout_output



    @torch.no_grad()
    def get_action(
        self,
        info_dict: dict,
        **kwargs,
    ) -> np.ndarray:
        """
        Select an action with GPC-RANK.

        info_dict:
            LeWM用の観測。

        kwargs["dp_info_dict"]:
            DiffusionPolicy用の観測履歴。
            指定されなければinfo_dictを使用する。
        """

        # --------------------------------------------------
        # 0. Diffusion Policy observation
        # --------------------------------------------------

        dp_info_dict = kwargs.get(
            "dp_info_dict",
            info_dict,
        )

        # --------------------------------------------------
        # 1. Generate Diffusion Policy proposals
        # --------------------------------------------------

        action_sequences = (
            self.diffusion_policy.sample_action_sequences(
                dp_info_dict,
                num_samples=self.num_candidates,
                denormalize=True,
            )
        )

        if action_sequences.ndim != 4:
            raise ValueError(
                "Expected candidate actions with shape "
                "(B, K, T, D), "
                f"but got {action_sequences.shape}"
            )

        B, K, T, D = action_sequences.shape

        if K != self.num_candidates:
            raise ValueError(
                f"Expected K={self.num_candidates}, got K={K}"
            )

        # --------------------------------------------------
        # 2. Extract actions corresponding to future
        # --------------------------------------------------

        start = self.diffusion_policy.obs_horizon - 1

        end = min(
            start + self.action_horizon,
            T,
        )

        if end <= start:
            raise ValueError(
                f"Invalid action range: start={start}, end={end}"
            )

        candidate_actions = action_sequences[
            :,
            :,
            start:end,
        ]

        # --------------------------------------------------
        # 3. Normalize for World Model
        # --------------------------------------------------

        wm_actions = self.normalize_action_for_world_model(
            candidate_actions
        )

        # --------------------------------------------------
        # 4. World Model rollout
        # --------------------------------------------------

        rollout_output = self.rollout(
            info_dict=info_dict,
            action_sequences=wm_actions,
        )

        # --------------------------------------------------
        # 5. Evaluate
        # --------------------------------------------------

        rewards = self.reward_fn(
            rollout_output,
            info_dict,
        )

        if not torch.is_tensor(rewards):
            raise TypeError(
                "reward_fn must return a torch.Tensor"
            )

        if rewards.shape != (B, K):
            raise ValueError(
                "reward_fn must return shape "
                f"(B, K)=({B}, {K}), "
                f"but got {rewards.shape}"
            )

        # --------------------------------------------------
        # 6. Select best candidate
        # --------------------------------------------------

        best_idx = torch.argmax(
            rewards,
            dim=1,
        )

        batch_idx = torch.arange(
            B,
            device=best_idx.device,
        )

        best_sequence = candidate_actions[
            batch_idx,
            best_idx,
        ]

        # Execute first action only
        action = best_sequence[:, 0]

        return (
            action
            .detach()
            .cpu()
            .numpy()
        )


    def encode_goal(
        self,
        info_dict: dict,
    ) -> torch.Tensor:
        """
        Encode goal observation with LeWM.

        Args:
            info_dict:
                Observation dictionary containing
                "goal" and corresponding goal_* entries.

        Returns:
            goal_emb:
                Encoded goal latent.
                Shape: (B, T_goal, latent_dim)
        """

        if "goal" not in info_dict:
            raise KeyError(
                "'goal' must be provided in info_dict"
            )

        info = dict(info_dict)

        # --------------------------------------------------
        # Build goal observation
        # --------------------------------------------------

        goal_info = {k: v for k, v in info.items() if k == "goal" or k.startswith("goal_")}

        # goal image -> pixels
        goal_info["pixels"] = goal_info["goal"]

        # goal_proprio -> proprio
        # goal_xxx     -> xxx
        for key in list(goal_info.keys()):
            if key.startswith("goal_"):
                new_key = key[len("goal_"):]
                goal_info[new_key] = goal_info.pop(key)

        goal_info.pop("goal", None)

        # Goal encoding does not require actions
        for key in ["action", "action_joint", "action_cartesian",]:
            goal_info.pop(key, None)

        # --------------------------------------------------
        # LeWM preprocessing
        # --------------------------------------------------
        goal_info = self._prepare_info(goal_info)

        # --------------------------------------------------
        # Move to World Model device
        # --------------------------------------------------

        device = next(self.world_model.parameters()).device

        for key, value in goal_info.items():
            if torch.is_tensor(value):
                goal_info[key] = value.to(device)

        # --------------------------------------------------
        # Encode goal
        # --------------------------------------------------
        goal_output = self.world_model.encode(goal_info)

        return goal_output["emb"]


    def normalize_action_for_world_model(self, action_sequences: torch.Tensor,) -> torch.Tensor:
        shape = action_sequences.shape

        action_np = (action_sequences.detach().cpu().numpy().reshape(-1, shape[-1]))

        if "action_cartesian" in self.process:
            action_np = self.process["action_cartesian"].transform(action_np)

        elif "action" in self.process:
            action_np = self.process["action"].transform(action_np)

        elif "action_joint" in self.process:
            action_np = self.process["action_joint"].transform(action_np)

        else:
            raise KeyError("No action processor found for World Model")

        action = torch.from_numpy(action_np.reshape(shape)).to(
            device=action_sequences.device,
            dtype=action_sequences.dtype,
        )

        return action





class WorldModelPolicy(BasePolicy):
    """Policy using a world model and planning solver for action selection."""

    def __init__(
        self,
        solver: Solver,
        config: PlanConfig,
        process: dict[str, Transformable] | None = None,
        transform: dict[str, Callable[[torch.Tensor], torch.Tensor]]
        | None = None,
        action_projector=None,
        **kwargs: Any,
    ) -> None:
        """Initialize the world model policy.

        Args:
            solver: The planning solver to use.
            config: MPC planning configuration.
            process: Dictionary of data preprocessors for specific keys.
            transform: Dictionary of tensor transformations (e.g., image transforms).
            **kwargs: Additional configuration parameters.
        """
        super().__init__(**kwargs)

        self.type = 'world_model'
        self.cfg = config
        self.solver = solver
        
        self.action_projector = action_projector
        
        
        self.action_buffer: deque[torch.Tensor] = deque(
            maxlen=self.flatten_receding_horizon
        )
        self.process = process or {}
        self.transform = transform or {}
        self._action_buffer: deque[torch.Tensor] | None = None
        self._next_init: torch.Tensor | None = None
        
        # print("self.process:", self.process)
        
        
        self.results_path = None



    @property
    def flatten_receding_horizon(self) -> int:
        """Receding horizon in environment steps (with frameskip)."""
        return self.cfg.receding_horizon * self.cfg.action_block


    def set_action_projector(self, action_projector) -> None:
        self.action_projector = action_projector


    def set_env(self, env: Any) -> None:
        """Configure the policy and solver for the given environment.

        Args:
            env: The environment to associate with the policy.
        """
        self.env = env

        if self.cfg.action_space == "joint":
            self.action_space = self.env.action_space #実空間
            
            if ("action" in self.process.keys()):
                self.action_processor = self.process["action"] 
                # action_processor = self.process["action_joint"] 
            elif ("action_joint" in self.process.keys()): 
                self.action_processor = self.process["action_joint"] 
            
        elif self.cfg.action_space == "cartesian":
            self.action_space = gym.spaces.Box(
                low=np.array([[0.315, -0.2, 0.095]], dtype=np.float32),
                high=np.array([[0.715,  0.2, 0.105]], dtype=np.float32),
                dtype=np.float32,
            ) #実空間
            self.action_processor = self.process["action_cartesian"]
        else:
            raise ValueError(
                f"Unknown action_space: {self.cfg.action_space}. "
                "Expected 'joint' or 'cartesian'."
            )
        
        # print("self.process.keys():", self.process.keys())
            
        
        
        
        n_envs = getattr(env, 'num_envs', 1)
        
        self.solver.configure(
            n_envs=n_envs, config=self.cfg, action_processor=self.action_processor, action_space=self.action_space, 
        )

        self._action_buffer = deque(maxlen=self.flatten_receding_horizon)

        assert isinstance(self.solver, Solver), (
            'Solver must implement the Solver protocol'
        )

    def get_action(self, info_dict: dict, **kwargs: Any) -> np.ndarray:
        """Get action via planning with the world model.

        Args:
            info_dict: Current state information from the environment.
            **kwargs: Additional parameters for planning.

        Returns:
            The selected action(s) as a numpy array.
        """
        
        assert hasattr(self, 'env'), 'Environment not set for the policy'
        assert 'pixels' in info_dict, "'pixels' must be provided in info_dict"
        assert 'goal' in info_dict, "'goal' must be provided in info_dict"


        # ActionProjector用の物理状態。(qpos, ee, gripper)
        # _prepare_info()には通さない。
        projection_state = kwargs.get(
            "projection_state",
            None,
        )

        info_dict = self._prepare_info(info_dict)
        # print("[stable-worldmodel/stable_worldmodel/policy.py] info_dict.keys() : ", info_dict.keys())
        # print("info_dict(step_idx)", info_dict["step_idx"][0][0])

        outputs = None
        # need to replan if action buffer is empty
        if len(self._action_buffer) == 0:
            outputs = self.solver(
                info_dict, 
                init_action=self._next_init,
                action_projector=self.action_projector,
                projection_state=projection_state,
            )
            
            
            #cem の内部状況をログ
            timestep = int(info_dict["step_idx"][0][0])
            save_dir = self.results_path / "cem"
            save_dir.mkdir(parents=True, exist_ok=True)
            print("save_dir:", save_dir)
            
            plot_cem_cost_convergence(outputs, env_idx=0, save_dir=save_dir, timestep=timestep)
            plot_cem_sequence_transition_colormap(outputs, env_idx=0, save_dir=save_dir, timestep=timestep, action_processor=self.action_processor)

            actions = outputs['actions']  # (num_envs, horizon, action_dim)
            
            
            actions_np = actions.cpu().numpy() if torch.is_tensor(actions) else actions

            for d in range(actions_np.shape[-1]):
                print(f"dim {d}: min={actions_np[..., d].min():.3f}, max={actions_np[..., d].max():.3f}") 
            


            keep_horizon = self.cfg.receding_horizon
            plan = actions[:, :keep_horizon]
            rest = actions[:, keep_horizon:]
            self._next_init = rest if self.cfg.warm_start else None

            # frameskip back to timestep
            plan = plan.reshape(
                self.env.num_envs, self.flatten_receding_horizon, -1
            )

            self._action_buffer.extend(plan.transpose(0, 1))

        action = self._action_buffer.popleft()
        action = action.reshape(*self.action_space.shape)
        action = action.numpy()


        # post-process action
        
        if 'action' in self.process: ##
            action = self.process['action'].inverse_transform(action)
            # action_joint = action
        elif "action_cartesian" in self.process:
            action = self.process["action_cartesian"].inverse_transform(action)
        elif "action_joint" in self.process:
            action = self.process["action_joint"].inverse_transform(action)

        return action, outputs  # (num_envs, action_dim)


def _load_model_with_attribute(run_name, attribute_name, cache_dir=None):
    """Helper function to load a model checkpoint and find a module with the specified attribute.

    Args:
        run_name: Path or name of the model run
        attribute_name: Name of the attribute to look for in the module (e.g., 'get_action', 'get_cost')
        cache_dir: Optional cache directory path

    Returns:
        The module with the specified attribute

    Raises:
        RuntimeError: If no module with the specified attribute is found
    """
    if Path(run_name).exists():
        run_path = Path(run_name)
    else:
        run_path = Path(
            cache_dir
            or swm.data.utils.get_cache_dir(sub_folder='checkpoints'),
            run_name,
        )

    if run_path.is_dir():
        ckpt_files = list(run_path.glob('*_object.ckpt'))
        ckpt_files.sort(key=lambda x: x.stat().st_ctime, reverse=True)
        path = ckpt_files[0]
        logging.info(f'Loading model from checkpoint: {path}')
    else:
        path = Path(f'{run_path}_object.ckpt')
        assert path.exists(), (
            f'Checkpoint path does not exist: {path}. Launch pretraining first.'
        )

    spt_module = torch.load(path, weights_only=False, map_location='cpu')

    def scan_module(module):
        if hasattr(module, attribute_name):
            if isinstance(module, torch.nn.Module):
                module = module.eval()
            return module
        for child in module.children():
            result = scan_module(child)
            if result is not None:
                return result
        return None

    result = scan_module(spt_module)
    if result is not None:
        return result

    raise RuntimeError(
        f"No module with '{attribute_name}' found in the loaded world model."
    )


def AutoActionableModel(
    run_name: str, cache_dir: str | Path | None = None
) -> torch.nn.Module:
    """Load a model checkpoint and return the module with a `get_action` method.

    Automatically scans the checkpoint for a module implementing the Actionable
    protocol (i.e., has a `get_action` method).

    Args:
        run_name: Path or name of the model run/checkpoint.
        cache_dir: Optional cache directory path. Defaults to STABLEWM_HOME.

    Returns:
        The module with a `get_action` method, set to eval mode.

    Raises:
        RuntimeError: If no module with `get_action` is found in the checkpoint.
    """
    return _load_model_with_attribute(run_name, 'get_action', cache_dir)


def AutoCostModel(
    run_name: str, cache_dir: str | Path | None = None
) -> torch.nn.Module:
    """Load a model checkpoint and return the module with a `get_cost` method.

    Automatically scans the checkpoint for a module implementing a cost function
    (i.e., has a `get_cost` method) for use with planning solvers.

    Args:
        run_name: Path or name of the model run/checkpoint.
        cache_dir: Optional cache directory path. Defaults to STABLEWM_HOME.

    Returns:
        The module with a `get_cost` method, set to eval mode.

    Raises:
        RuntimeError: If no module with `get_cost` is found in the checkpoint.
    """
    return _load_model_with_attribute(run_name, 'get_cost', cache_dir)


# Alias for backward compatibility and type hinting
Policy = BasePolicy