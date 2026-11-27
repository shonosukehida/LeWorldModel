"""Read the fixed flip-mug goal without importing the robot or simulator."""
import cv2
import numpy as np


def validate_goal_config(config):
    if not config.goal_image_path or not config.goal_wrist_image_path or config.get("goal_proprio") is None:
        raise ValueError("Specify goal_image_path, goal_wrist_image_path and goal_proprio for subgoal")


def load_saved_goal(config):
    def rgb(path, label):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Could not read {label}goal image: {path}")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    overhead = rgb(config.goal_image_path, "")
    wrist = rgb(config.goal_wrist_image_path, "wrist ")
    if config.get("goal_proprio") is None:
        raise ValueError("goal_proprio must be specified when using a saved goal image")
    proprio = np.asarray(config.goal_proprio, dtype=np.float32)
    if proprio.shape != (8,):
        raise ValueError(f"goal_proprio must have shape (8,), got {proprio.shape}")
    quat = proprio[3:7]
    norm = float(np.linalg.norm(quat))
    if norm < 1e-8:
        raise ValueError("goal_proprio contains a zero quaternion")
    proprio[3:7] = quat / norm
    if proprio[6] < 0:
        proprio[3:7] *= -1.0
    return overhead, wrist, proprio


def goal_observation(config):
    validate_goal_config(config)
    overhead, wrist, proprio = load_saved_goal(config)
    return dict(goal=overhead[None, None], goal_wrist_pixels=wrist[None, None],
                goal_proprio=proprio[None, None])
