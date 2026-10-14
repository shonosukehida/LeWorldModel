"""Frozen reinforcement functions from HEAD 8e0e8e6 for exact regression checks."""
import numpy as np
import time

def _reinforced_diffusion_action(policy, observation, ee, state, config, gripper_config):
    """Select a Cartesian command using a monotonic deadline, not step count."""
    now = time.monotonic()
    if state.get("deadline") is not None and now >= state["deadline"]:
        state.clear()
    if not state:
        result = policy.get_action(observation)
        action = result[0] if isinstance(result, tuple) else result
        action = np.asarray(action).reshape(-1)
        if action.size != 8:
            raise ValueError("Gripper reinforcement requires an 8D Cartesian action")
        policy_command = float(action[7])  # Already inverse-transformed by DP.
        if policy_command > float(config.threshold):
            state["pose"] = np.asarray(ee, dtype=np.float32).copy()
            state["deadline"] = time.monotonic() + float(config.duration_sec)
        else:
            return result, False, policy_command
    else:
        policy_command = float("nan")
    command = (float(config.target_opening_mm) - float(gripper_config.open_position)) / (
        float(gripper_config.closed_position) - float(gripper_config.open_position)
    )
    # execute() retains Cartesian/IK safety and gripper.max_delta clipping.
    # Its 8D follower command replaces the persistent background target;
    # Robopy only reissues G2 commands when that target changes. gripper.wait
    # controls SDK waiting in that thread, not this monotonic deadline.
    return np.concatenate([state["pose"], [command]]), True, policy_command

def _reinforced_gpc_action(policy, observation, dp_info, projection_state, ee,
                           state, config, gripper_config):
    """Hold the observed EE pose after GPC selects a physical closing action."""
    values = [float(config.threshold), float(config.target_opening_mm),
              float(config.duration_sec)]
    open_mm = float(gripper_config.open_position)
    closed_mm = float(gripper_config.closed_position)
    if (not np.isfinite(values).all() or not 0 <= values[0] <= 1
            or values[2] <= 0 or not np.isfinite([open_mm, closed_mm]).all()
            or open_mm <= closed_mm or not closed_mm <= values[1] <= open_mm):
        raise ValueError("Invalid GPC gripper reinforcement configuration")
    now = time.monotonic()
    if state.get("deadline") is not None and now >= state["deadline"]:
        state.clear()
    if not state:
        result = policy.get_action(
            observation, dp_info_dict=dp_info, projection_state=projection_state,
        )
        action = np.asarray(result[0] if isinstance(result, tuple) else result)
        if action.shape not in ((8,), (1, 8)) or not np.isfinite(action).all():
            raise ValueError("GPC reinforcement requires a finite 8D Cartesian action")
        policy_command = float(action.reshape(-1)[7])
        if policy_command <= values[0]:
            return result, False, policy_command
        pose = np.asarray(ee, dtype=np.float32)
        if pose.shape != (7,) or not np.isfinite(pose).all():
            raise ValueError("GPC reinforcement requires a finite 7D EE pose")
        state["pose"] = pose.copy()
        state["deadline"] = time.monotonic() + values[2]
    else:
        policy_command = float("nan")
    command = (values[1] - open_mm) / (closed_mm - open_mm)
    return np.concatenate([state["pose"], [command]]), True, policy_command
