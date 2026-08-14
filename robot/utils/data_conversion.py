"""Merge per-episode xArm HDF5 files into a single LeWM dataset.

The output additionally contains ``leader_ee_pos_quat``, calculated from
the xArm7 target joint angles stored in ``arms/leader`` using the xArm SDK's
forward kinematics.

Output:
    ee_pos_quat          : follower EE pose [x, y, z, qx, qy, qz, qw]
    proprio              : follower proprioception
                           [x, y, z, qx, qy, qz, qw, gripper]
    leader_ee_pos_quat   : leader target EE pose [x, y, z, qx, qy, qz, qw]
    action_cartesian     : target EE pose + gripper
                           [x, y, z, qx, qy, qz, qw, gripper]
    follower             : follower joint state
    leader               : leader target joint state
    pixels               : camera observations
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from scipy.spatial.transform import Rotation
from xarm.wrapper import XArmAPI


SOURCE_KEYS = {
    "ee_pos_quat": "arms/ee_pos_quat",
    "follower": "arms/follower",
    "leader": "arms/leader",
    "pixels": "sensors/cameras/main",
}

LEADER_EE_KEY = "leader_ee_pos_quat"
ACTION_CARTESIAN_KEY = "action_cartesian"
PROPRIO_KEY = "proprio"


def get_episode_files(input_dir: Path) -> list[Path]:
    """Return sorted per-episode HDF5 files."""
    episode_files = sorted(input_dir.glob("episode_*.h5"))

    if not episode_files:
        raise FileNotFoundError(
            f"No episode_*.h5 files were found in: {input_dir}"
        )

    return episode_files


def inspect_episode(
    episode_path: Path,
) -> dict[str, tuple[tuple[int, ...], np.dtype]]:
    """Read shapes and dtypes of one episode."""
    result: dict[str, tuple[tuple[int, ...], np.dtype]] = {}

    with h5py.File(episode_path, "r") as file:
        for output_key, source_key in SOURCE_KEYS.items():
            if source_key not in file:
                raise KeyError(
                    f"Missing dataset '{source_key}' in {episode_path}"
                )

            dataset = file[source_key]
            result[output_key] = (dataset.shape, dataset.dtype)

    ee_shape, _ = result["ee_pos_quat"]
    follower_shape, _ = result["follower"]

    if len(ee_shape) != 2 or ee_shape[1] != 7:
        raise ValueError(
            "ee_pos_quat must have shape (T, 7), "
            f"but got {ee_shape} in {episode_path}"
        )

    if len(follower_shape) != 2 or follower_shape[1] < 8:
        raise ValueError(
            "Follower dataset must contain 7 joints and "
            "1 gripper value, "
            f"but got {follower_shape} in {episode_path}"
        )

    if ee_shape[0] != follower_shape[0]:
        raise ValueError(
            "ee_pos_quat and follower must have the same "
            "time dimension, "
            f"but got {ee_shape[0]} and {follower_shape[0]} "
            f"in {episode_path}"
        )

    result[PROPRIO_KEY] = (
        (ee_shape[0], 8),
        np.dtype(np.float32),
    )
    leader_shape, _ = result["leader"]




    if len(leader_shape) != 2 or leader_shape[1] < 7:
        raise ValueError(
            "Leader dataset must have shape (T, >=7), "
            f"but got {leader_shape} in {episode_path}"
        )

    # leader joint -> [x, y, z, qx, qy, qz, qw]
    result[LEADER_EE_KEY] = (
        (leader_shape[0], 7),
        np.dtype(np.float32),
    )
    
    if leader_shape[1] < 8:
        raise ValueError(
            "Leader dataset must contain 7 joints and 1 gripper value, "
            f"but got shape={leader_shape} in {episode_path}"
        )

    result[ACTION_CARTESIAN_KEY] = (
        (leader_shape[0], 8),
        np.dtype(np.float32),
    )


    return result


def validate_episode(
    episode_path: Path,
    expected_info: dict[str, tuple[tuple[int, ...], np.dtype]],
) -> None:
    """Check that an episode has the expected source keys and shapes."""
    with h5py.File(episode_path, "r") as file:
        for output_key, source_key in SOURCE_KEYS.items():
            if source_key not in file:
                raise KeyError(
                    f"Missing dataset '{source_key}' in {episode_path}"
                )

            actual_shape = file[source_key].shape
            expected_shape, _ = expected_info[output_key]

            if actual_shape != expected_shape:
                raise ValueError(
                    f"Shape mismatch in {episode_path}\n"
                    f"  key: {source_key}\n"
                    f"  expected: {expected_shape}\n"
                    f"  actual: {actual_shape}"
                )


def create_output_datasets(
    output_file: h5py.File,
    num_episodes: int,
    dataset_info: dict[str, tuple[tuple[int, ...], np.dtype]],
    compression: str | None,
) -> None:
    """Create flattened LeWM datasets.

    Each temporal dataset is stored as:

        (total_steps, ...)

    Episode boundaries are represented separately by:
        ep_len
        ep_offset
    """
    episode_length: int | None = None

    for output_key, (episode_shape, dtype) in dataset_info.items():
        if len(episode_shape) < 1:
            raise ValueError(
                f"Dataset '{output_key}' must have a time dimension, "
                f"got shape={episode_shape}"
            )

        current_length = episode_shape[0]

        if episode_length is None:
            episode_length = current_length
        elif current_length != episode_length:
            raise ValueError(
                "All temporal datasets must have the same T dimension.\n"
                f"  expected: {episode_length}\n"
                f"  {output_key}: {current_length}"
            )

        total_steps = num_episodes * current_length
        output_shape = (total_steps, *episode_shape[1:])

        # 過度に大きなchunkを避ける
        chunk_steps = min(current_length, 32)
        chunks = (chunk_steps, *episode_shape[1:])

        output_file.create_dataset(
            name=output_key,
            shape=output_shape,
            dtype=dtype,
            chunks=chunks,
            compression=compression,
        )

    if episode_length is None:
        raise ValueError("No temporal datasets were found")

    ep_len = np.full(
        shape=(num_episodes,),
        fill_value=episode_length,
        dtype=np.int64,
    )

    ep_offset = np.arange(
        num_episodes,
        dtype=np.int64,
    ) * episode_length

    output_file.create_dataset(
        "ep_len",
        data=ep_len,
        dtype=np.int64,
    )

    output_file.create_dataset(
        "ep_offset",
        data=ep_offset,
        dtype=np.int64,
    )
    
    
    episode_idx = np.repeat(
        np.arange(num_episodes, dtype=np.int32),
        episode_length,
    )

    step_idx = np.tile(
        np.arange(episode_length, dtype=np.int32),
        num_episodes,
    )

    output_file.create_dataset(
        "episode_idx",
        data=episode_idx,
        dtype=np.int32,
    )

    output_file.create_dataset(
        "step_idx",
        data=step_idx,
        dtype=np.int32,
    )


def canonicalize_quaternion_sequence(
    quaternions: np.ndarray,
) -> np.ndarray:
    """Make quaternion signs continuous within one episode.

    Args:
        quaternions:
            Shape (T, 4), ordered as [qx, qy, qz, qw].

    Returns:
        Shape (T, 4), normalized and sign-continuous quaternions.
    """
    quaternions = np.asarray(
        quaternions,
        dtype=np.float32,
    ).copy()

    if quaternions.ndim != 2 or quaternions.shape[1] != 4:
        raise ValueError(
            "quaternions must have shape (T, 4), "
            f"got {quaternions.shape}"
        )

    if quaternions.shape[0] == 0:
        raise ValueError("quaternions must not be empty")

    if not np.all(np.isfinite(quaternions)):
        raise ValueError(
            "quaternions contain NaN or Inf"
        )

    norms = np.linalg.norm(
        quaternions,
        axis=1,
        keepdims=True,
    )

    if np.any(norms < 1e-8):
        raise ValueError(
            "Zero-length quaternion was found"
        )

    quaternions /= norms

    if quaternions[0, 3] < 0:
        quaternions[0] *= -1.0

    for step in range(1, quaternions.shape[0]):
        dot = np.dot(
            quaternions[step - 1],
            quaternions[step],
        )

        if dot < 0:
            quaternions[step] *= -1.0

    return quaternions

def fk_pose_to_pos_quat(
    fk_pose: list[float] | np.ndarray,
) -> np.ndarray:
    """Convert xArm FK pose to [x_m, y_m, z_m, qx, qy, qz, qw].

    xArm ``get_forward_kinematics`` returns:

        [x_mm, y_mm, z_mm, roll_rad, pitch_rad, yaw_rad]

    SciPy returns quaternion in:

        [qx, qy, qz, qw]
    """
    pose = np.asarray(fk_pose, dtype=np.float64)

    if pose.shape != (6,):
        raise ValueError(
            f"FK pose must have shape (6,), got {pose.shape}"
        )

    position_m = pose[:3] / 1000.0

    quaternion = Rotation.from_euler(
        "xyz",
        pose[3:6],
        degrees=False,
    ).as_quat()

    result = np.concatenate(
        [position_m, quaternion]
    ).astype(np.float32)

    if not np.all(np.isfinite(result)):
        raise ValueError(
            f"FK result contains NaN or Inf: {result}"
        )

    return result


def calculate_leader_ee_trajectory(
    arm: XArmAPI,
    leader: np.ndarray,
    episode_name: str,
) -> np.ndarray:
    """Calculate leader EE poses for one episode.

    Args:
        arm:
            Connected xArm SDK object used for FK calculation.
        leader:
            Shape (T, 8). First seven values are xArm7 target joints.
        episode_name:
            Episode name used in error messages.

    Returns:
        Shape (T, 7):
        [x_m, y_m, z_m, qx, qy, qz, qw]
    """
    leader = np.asarray(leader, dtype=np.float32)

    if leader.ndim != 2 or leader.shape[1] < 7:
        raise ValueError(
            f"leader must have shape (T, >=7), got {leader.shape}"
        )

    num_steps = leader.shape[0]
    leader_ee = np.empty(
        (num_steps, 7),
        dtype=np.float32,
    )

    for step in range(num_steps):
        joints = leader[step, :7]

        if not np.all(np.isfinite(joints)):
            raise ValueError(
                f"Invalid leader joints in {episode_name}, "
                f"step={step}: {joints}"
            )

        code, fk_pose = arm.get_forward_kinematics(
            joints.tolist(),
            input_is_radian=True,
        )

        if code != 0 or fk_pose is None:
            raise RuntimeError(
                "get_forward_kinematics failed\n"
                f"  episode: {episode_name}\n"
                f"  step: {step}\n"
                f"  code: {code}\n"
                f"  joints: {joints}"
            )

        leader_ee[step] = fk_pose_to_pos_quat(fk_pose)

    return leader_ee



def calculate_proprio(
    ee_pos_quat: np.ndarray,
    follower: np.ndarray,
) -> np.ndarray:
    """Create follower proprioception.

    Args:
        ee_pos_quat:
            Shape (T, 7):
            [x, y, z, qx, qy, qz, qw]

        follower:
            Shape (T, >=8):
            [j1, ..., j7, gripper]

    Returns:
        Shape (T, 8):
        [x, y, z, qx, qy, qz, qw, gripper]
    """
    ee_pos_quat = np.asarray(
        ee_pos_quat,
        dtype=np.float32,
    )
    follower = np.asarray(
        follower,
        dtype=np.float32,
    )

    if ee_pos_quat.ndim != 2:
        raise ValueError(
            "ee_pos_quat must be a 2D array, "
            f"got shape={ee_pos_quat.shape}"
        )

    if ee_pos_quat.shape[1] != 7:
        raise ValueError(
            "ee_pos_quat must have shape (T, 7), "
            f"got shape={ee_pos_quat.shape}"
        )

    if follower.ndim != 2 or follower.shape[1] < 8:
        raise ValueError(
            "follower must have shape (T, >=8), "
            f"got shape={follower.shape}"
        )

    if ee_pos_quat.shape[0] != follower.shape[0]:
        raise ValueError(
            "ee_pos_quat and follower must have the "
            "same T dimension, "
            f"got {ee_pos_quat.shape[0]} and "
            f"{follower.shape[0]}"
        )

    if ee_pos_quat.shape[0] == 0:
        raise ValueError(
            "ee_pos_quat must not be empty"
        )

    if not np.all(np.isfinite(ee_pos_quat)):
        raise ValueError(
            "ee_pos_quat contains NaN or Inf"
        )

    if not np.all(np.isfinite(follower[:, 7])):
        raise ValueError(
            "Follower gripper contains NaN or Inf"
        )

    gripper = follower[:, 7:8]

    proprio = np.concatenate(
        [ee_pos_quat, gripper],
        axis=1,
    ).astype(np.float32)

    return proprio

def calculate_action_cartesian(
    leader_ee_pos_quat: np.ndarray,
    leader: np.ndarray,
) -> np.ndarray:
    """Create Cartesian actions with gripper.

    Each target is:
        [x, y, z, qx, qy, qz, qw, gripper]

    For t < T - 1:
        action[t] = target[t + 1]

    For t = T - 1:
        action[t] = target[T - 1]
    """
    leader_ee_pos_quat = np.asarray(
        leader_ee_pos_quat,
        dtype=np.float32,
    )
    leader = np.asarray(
        leader,
        dtype=np.float32,
    )

    if leader_ee_pos_quat.ndim != 2:
        raise ValueError(
            "leader_ee_pos_quat must be a 2D array, "
            f"got shape={leader_ee_pos_quat.shape}"
        )

    if leader_ee_pos_quat.shape[1] != 7:
        raise ValueError(
            "leader_ee_pos_quat must have shape (T, 7), "
            f"got shape={leader_ee_pos_quat.shape}"
        )

    if leader.ndim != 2 or leader.shape[1] < 8:
        raise ValueError(
            "leader must have shape (T, >=8), "
            f"got shape={leader.shape}"
        )

    if leader.shape[0] != leader_ee_pos_quat.shape[0]:
        raise ValueError(
            "leader and leader_ee_pos_quat must have the same T dimension, "
            f"got {leader.shape[0]} and {leader_ee_pos_quat.shape[0]}"
        )

    if leader_ee_pos_quat.shape[0] == 0:
        raise ValueError("leader_ee_pos_quat must not be empty")

    # leaderの8列目をグリッパー値として使用
    gripper = leader[:, 7:8]

    # Shape: (T, 8)
    cartesian_target = np.concatenate(
        [leader_ee_pos_quat, gripper],
        axis=1,
    ).astype(np.float32)

    action_cartesian = np.empty_like(
        cartesian_target,
        dtype=np.float32,
    )

    action_cartesian[:-1] = cartesian_target[1:]
    action_cartesian[-1] = cartesian_target[-1]

    return action_cartesian


def merge_episodes(
    input_dir: Path,
    output_path: Path,
    robot_ip: str,
    num_episodes: int | None = None,
    compression: str | None = "gzip",
) -> None:
    """Merge selected per-episode files into one HDF5 file."""
    episode_files = get_episode_files(input_dir)
    available_episodes = len(episode_files)

    if num_episodes is None:
        num_episodes = available_episodes

    if num_episodes <= 0:
        raise ValueError(
            f"num_episodes must be positive, got {num_episodes}"
        )

    if num_episodes > available_episodes:
        raise ValueError(
            "Requested more episodes than available.\n"
            f"  requested: {num_episodes}\n"
            f"  available: {available_episodes}"
        )

    episode_files = episode_files[:num_episodes]

    print(f"Found {available_episodes} episodes")
    print(f"Using {num_episodes} episodes")
    print(f"First episode: {episode_files[0].name}")
    print(f"Last episode : {episode_files[-1].name}")

    dataset_info = inspect_episode(episode_files[0])

    print("\nExpected per-episode structure")
    print("=" * 72)

    for output_key, (shape, dtype) in dataset_info.items():
        if output_key == LEADER_EE_KEY:
            source_key = "arms/leader -> xArm7 FK"
        elif output_key == ACTION_CARTESIAN_KEY:
            source_key = (
                "leader_ee_pos_quat shifted by 1"
            )
        elif output_key == PROPRIO_KEY:
            source_key = (
                "ee_pos_quat + follower gripper"
            )
        else:
            source_key = SOURCE_KEYS[output_key]

        print(
            f"{source_key:<34} -> "
            f"{output_key:<20} "
            f"shape={shape}, dtype={dtype}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        raise FileExistsError(
            f"Output file already exists: {output_path}\n"
            "Delete it explicitly before running this script again."
        )

    arm: Any | None = None

    try:
        print(f"\nConnecting to xArm SDK for FK: {robot_ip}")
        arm = XArmAPI(robot_ip, is_radian=True)

        with h5py.File(output_path, "w") as output_file:
            create_output_datasets(
                output_file=output_file,
                num_episodes=num_episodes,
                dataset_info=dataset_info,
                compression=compression,
            )

            output_file.attrs["num_episodes"] = num_episodes
            output_file.attrs["source_directory"] = str(
                input_dir.resolve()
            )
            output_file.attrs["format"] = "xarm_flip_mug"
            
            output_file.attrs["storage_layout"] = "flattened_time"
            output_file.attrs["time_axis"] = 0
            output_file.attrs["episode_boundaries"] = "ep_len_and_ep_offset"

            output_file.attrs["robot_ip_used_for_fk"] = robot_ip
            output_file.attrs["leader_ee_position_unit"] = "metre"
            output_file.attrs["leader_ee_quaternion_order"] = "qx_qy_qz_qw"
            output_file.attrs["leader_fk_orientation_input"] = (
                "roll_pitch_yaw_xyz_radians"
            )
            output_file.attrs["action_cartesian_layout"] = (
                "x_y_z_qx_qy_qz_qw_gripper"
            )
            output_file.attrs["action_cartesian_dimension"] = 8
            output_file.attrs["action_cartesian_gripper_source"] = "leader_column_7"
            output_file.attrs["proprio_layout"] = (
                "x_y_z_qx_qy_qz_qw_gripper"
            )
            output_file.attrs["proprio_dimension"] = 8
            output_file.attrs["proprio_position_unit"] = "metre"
            output_file.attrs["proprio_quaternion_order"] = (
                "qx_qy_qz_qw"
            )
            output_file.attrs["proprio_gripper_source"] = (
                "follower_column_7"
            )



            string_dtype = h5py.string_dtype(encoding="utf-8")
            episode_names = output_file.create_dataset(
                "episode_names",
                shape=(num_episodes,),
                dtype=string_dtype,
            )

            episode_length = dataset_info["pixels"][0][0]

            for episode_index, episode_path in enumerate(episode_files):
                validate_episode(
                    episode_path=episode_path,
                    expected_info=dataset_info,
                )

                start = episode_index * episode_length
                end = start + episode_length

                with h5py.File(episode_path, "r") as episode_file:
                    for output_key, source_key in SOURCE_KEYS.items():
                        episode_data = episode_file[source_key][...]

                        if episode_data.shape[0] != episode_length:
                            raise ValueError(
                                f"Unexpected T dimension for {source_key}: "
                                f"{episode_data.shape}"
                            )

                        if output_key == "ee_pos_quat":
                            episode_data = np.asarray(
                                episode_data,
                                dtype=np.float32,
                            ).copy()

                            if (
                                episode_data.ndim != 2
                                or episode_data.shape[1] != 7
                            ):
                                raise ValueError(
                                    "ee_pos_quat must have shape (T, 7), "
                                    f"got {episode_data.shape}"
                                )

                            episode_data[:, 3:7] = (
                                canonicalize_quaternion_sequence(
                                    episode_data[:, 3:7]
                                )
                            )

                        output_file[output_key][start:end] = episode_data


                    leader = np.asarray(
                        episode_file[SOURCE_KEYS["leader"]][...],
                        dtype=np.float32,
                    )

                    follower = np.asarray(
                        episode_file[SOURCE_KEYS["follower"]][...],
                        dtype=np.float32,
                    )

                    # push.h5へ実際に保存されたee_pos_quatを使用する。
                    # クォータニオン連続化処理を行っている場合も、
                    # その修正後の値がここから取得される。
                    stored_ee_pos_quat = np.asarray(
                        output_file["ee_pos_quat"][start:end],
                        dtype=np.float32,
                    )

                    proprio = calculate_proprio(
                        ee_pos_quat=stored_ee_pos_quat,
                        follower=follower,
                    )
                    
                    if proprio.shape != (episode_length, 8):
                        raise ValueError(
                            "proprio shape mismatch: "
                            f"expected {(episode_length, 8)}, "
                            f"got {proprio.shape}"
                        )

                    leader_ee_pos_quat = calculate_leader_ee_trajectory(
                        arm=arm,
                        leader=leader,
                        episode_name=episode_path.name,
                    )
                    
                    leader_ee_pos_quat[:, 3:7] = (
                        canonicalize_quaternion_sequence(
                            leader_ee_pos_quat[:, 3:7]
                        )
                    )

                    action_cartesian = calculate_action_cartesian(
                        leader_ee_pos_quat=leader_ee_pos_quat,
                        leader=leader,
                    )

                    if leader_ee_pos_quat.shape[0] != episode_length:
                        raise ValueError(
                            f"leader_ee_pos_quat length mismatch: "
                            f"{leader_ee_pos_quat.shape}"
                        )

                    if action_cartesian.shape[0] != episode_length:
                        raise ValueError(
                            f"action_cartesian length mismatch: "
                            f"{action_cartesian.shape}"
                        )

                    output_file[LEADER_EE_KEY][start:end] = (
                        leader_ee_pos_quat
                    )

                    output_file[ACTION_CARTESIAN_KEY][start:end] = (
                        action_cartesian
                    )
                    
                    output_file[PROPRIO_KEY][start:end] = proprio

                episode_names[episode_index] = episode_path.name

                print(
                    f"\rMerging + FK: "
                    f"{episode_index + 1:4d}/{num_episodes:4d} "
                    f"{episode_path.name}",
                    end="",
                    flush=True,
                )

            output_file.flush()

    except Exception:
        # 失敗時に不完全なpush.h5を残さない
        if output_path.exists():
            output_path.unlink()
        raise

    finally:
        if arm is not None:
            try:
                arm.disconnect()
            except Exception:
                pass

    print()
    print(f"\nSaved merged dataset to:\n{output_path}")


def print_h5_structure(file_path: Path) -> None:
    """Print the resulting HDF5 structure."""
    print("\nMerged HDF5 Structure")
    print("=" * 72)

    with h5py.File(file_path, "r") as file:
        for key, dataset in file.items():
            if isinstance(dataset, h5py.Dataset):
                print(
                    f"{key:<24} "
                    f"shape={dataset.shape}, "
                    f"dtype={dataset.dtype}"
                )
        if "ep_len" in file:
            print(f"\nep_len   : {file['ep_len'][:]}")

        if "ep_offset" in file:
            print(f"ep_offset: {file['ep_offset'][:]}")

        print("\nAttributes")
        print("-" * 72)

        for key, value in file.attrs.items():
            print(f"{key}: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge xArm per-episode HDF5 files into one push.h5 "
            "and calculate leader EE poses."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(
            "/home/hida/.stable_worldmodel/datasets/flip_mug/ep200_tm300/per_episode"
        ),
        help="Directory containing episode_*.h5 files.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "/home/hida/.stable_worldmodel/datasets/flip_mug/ep200_tm300"
        ),
        help="Directory where push.h5 will be created.",
    )

    parser.add_argument(
        "--num_episodes",
        type=int,
        default=None,
        help="Number of episodes to merge. Uses all episodes if omitted.",
    ) 

    parser.add_argument(
        "--robot-ip",
        type=str,
        default="192.168.1.240",
        help="xArm IP address used by the SDK for FK calculation.",
    )

    parser.add_argument(
        "--compression",
        choices=["gzip", "lzf", "none"],
        default="gzip",
        help="HDF5 compression method.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    compression = (
        None
        if args.compression == "none"
        else args.compression
    )

    input_dir = args.input_dir
    output_path = args.output_dir / "push.h5"
    num_episodes = args.num_episodes

    merge_episodes(
        input_dir=input_dir,
        output_path=output_path,
        robot_ip=args.robot_ip,
        num_episodes=num_episodes,
        compression=compression,
    )

    print_h5_structure(output_path)


if __name__ == "__main__":
    # 実行例:
    #
    # uv run robot/utils/data_conversion.py \
    #     --input-dir /path/to/per_episode \
    #     --output-dir /path/to/output \
    #.    --num_episodes None
    #
    main()