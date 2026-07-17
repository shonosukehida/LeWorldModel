"""Merge per-episode xArm HDF5 files into a single LeWM dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


SOURCE_KEYS = {
    "ee_pos_quat": "arms/ee_pos_quat",
    "follower": "arms/follower",
    "leader": "arms/leader",
    "pixels": "sensors/cameras/main",
}


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

    return result


def validate_episode(
    episode_path: Path,
    expected_info: dict[str, tuple[tuple[int, ...], np.dtype]],
) -> None:
    """Check that an episode has the expected keys and shapes."""
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
    """Create output datasets with an episode dimension."""
    for output_key, (episode_shape, dtype) in dataset_info.items():
        output_shape = (num_episodes, *episode_shape)

        # エピソード単位で読み書きしやすいチャンク構造
        chunks = (1, *episode_shape)

        output_file.create_dataset(
            name=output_key,
            shape=output_shape,
            dtype=dtype,
            chunks=chunks,
            compression=compression,
        )


def merge_episodes(
    input_dir: Path,
    output_path: Path,
    compression: str | None = "gzip",
) -> None:
    """Merge all per-episode files into one HDF5 file."""
    episode_files = get_episode_files(input_dir)
    num_episodes = len(episode_files)

    print(f"Found {num_episodes} episodes")
    print(f"First episode: {episode_files[0].name}")
    print(f"Last episode : {episode_files[-1].name}")

    dataset_info = inspect_episode(episode_files[0])

    print("\nExpected per-episode structure")
    print("=" * 60)

    for output_key, (shape, dtype) in dataset_info.items():
        source_key = SOURCE_KEYS[output_key]
        print(
            f"{source_key:<30} -> "
            f"{output_key:<12} shape={shape}, dtype={dtype}"
        )

    # 出力先の親ディレクトリを作成
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        raise FileExistsError(
            f"Output file already exists: {output_path}\n"
            "Delete it explicitly before running this script again."
        )

    try:
        with h5py.File(output_path, "w") as output_file:
            create_output_datasets(
                output_file=output_file,
                num_episodes=num_episodes,
                dataset_info=dataset_info,
                compression=compression,
            )

            # データセット全体に関するメタデータ
            output_file.attrs["num_episodes"] = num_episodes
            output_file.attrs["source_directory"] = str(
                input_dir.resolve()
            )
            output_file.attrs["format"] = "xarm_flip_mug"
            output_file.attrs["episode_axis"] = 0
            output_file.attrs["time_axis"] = 1

            string_dtype = h5py.string_dtype(encoding="utf-8")
            episode_names = output_file.create_dataset(
                "episode_names",
                shape=(num_episodes,),
                dtype=string_dtype,
            )

            for episode_index, episode_path in enumerate(episode_files):
                validate_episode(
                    episode_path=episode_path,
                    expected_info=dataset_info,
                )

                with h5py.File(episode_path, "r") as episode_file:
                    for output_key, source_key in SOURCE_KEYS.items():
                        # 1エピソード分だけメモリに読み込む
                        output_file[output_key][episode_index] = (
                            episode_file[source_key][...]
                        )

                episode_names[episode_index] = episode_path.name

                print(
                    f"\rMerging: "
                    f"{episode_index + 1:4d}/{num_episodes:4d} "
                    f"{episode_path.name}",
                    end="",
                    flush=True,
                )

            output_file.flush()

    except Exception:
        # 途中で失敗した不完全なファイルを残さない
        if output_path.exists():
            output_path.unlink()
        raise

    print()
    print(f"\nSaved merged dataset to:\n{output_path}")


def print_h5_structure(file_path: Path) -> None:
    """Print the resulting HDF5 structure."""
    print("\nMerged HDF5 Structure")
    print("=" * 60)

    with h5py.File(file_path, "r") as file:
        for key, dataset in file.items():
            if isinstance(dataset, h5py.Dataset):
                print(
                    f"{key:<16} "
                    f"shape={dataset.shape}, "
                    f"dtype={dataset.dtype}"
                )

        print("\nAttributes")
        print("-" * 60)

        for key, value in file.attrs.items():
            print(f"{key}: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge xArm per-episode HDF5 files into one push.h5."
        )
    )
    
    parser.add_argument(
        "--dataset-dir",
        type=Path, 
        default=Path(
            "/home/shonosukehida/.stable_worldmodel/"
            "datasets/flip_mug/ep200_tm300"
        ),
        
    )

    # parser.add_argument(
    #     "--input-dir",
    #     type=Path,
    #     default=Path(
    #         "/home/shonosukehida/.stable_worldmodel/"
    #         "datasets/flip_mug/ep200_tm300/per_episode"
    #     ),
    #     help="Directory containing episode_*.h5 files.",
    # )

    # parser.add_argument(
    #     "--output",
    #     type=Path,
    #     default=Path(
    #         "/home/shonosukehida/.stable_worldmodel/"
    #         "datasets/flip_mug/ep200_tm300/push.h5"
    #     ),
    #     help="Output HDF5 path.",
    # )

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
        None if args.compression == "none" else args.compression
    )
    
    input_dir = args.dataset_dir/"per_episode"
    output_path = args.dataset_dir/"push.h5"

    merge_episodes(
        input_dir=input_dir,
        output_path=output_path,
        compression=compression,
    )

    print_h5_structure(output_path)


if __name__ == "__main__":
    
    #uv run robot/utils/data_conversion.py --dataset-dir /home/shonosukehida/.stable_worldmodel/datasets/flip_mug/ep200_tm300 のように実行
    main()