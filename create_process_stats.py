"""Generate evaluation normalization statistics directly from an HDF5 file."""
import argparse
from pathlib import Path

import h5py
import numpy as np

from normalization_stats import build_normalization_process, save_normalization_process


class HDF5Columns:
    def __init__(self, file, path):
        self.file = file
        self.path = path

    def get_col_data(self, column):
        if column not in self.file:
            raise KeyError(f"Missing dataset column {column!r} in {self.path}")
        return self.file[column][:]


def create_process_stats(dataset, output, keys=None, *, overwrite=True):
    dataset = Path(dataset).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    if dataset == output or (
        dataset.exists() and output.exists() and dataset.samefile(output)
    ):
        raise ValueError(f"Output must differ from input dataset: {dataset}")
    if not dataset.is_file():
        raise FileNotFoundError(f"Dataset file was not found: {dataset}")
    if output.exists() and not overwrite:
        raise FileExistsError(f"Overwriting normalization statistics is forbidden: {output}")
    keys = list(keys if keys is not None else ("action_cartesian", "proprio"))
    if "action_cartesian" not in keys:
        raise ValueError(f"Required column 'action_cartesian' must be included in --keys: {dataset}")
    if len(keys) != len(set(keys)):
        raise ValueError("Duplicate columns in --keys")
    if any(key == "pixels" or key.startswith("goal_") for key in keys):
        raise ValueError(f"--keys must contain base numeric columns, not pixels or goal aliases: {keys}")
    with h5py.File(dataset, "r") as file:
        columns = HDF5Columns(file, dataset)
        # Validate every requested column before fitting or creating an output.
        for key in keys:
            values = np.asarray(columns.get_col_data(key))
            if values.ndim not in (1, 2) or (values.ndim == 2 and values.shape[1] == 0):
                raise ValueError(f"Column {key!r} in {dataset} must have shape (N,) or (N, D)")
            if not np.issubdtype(values.dtype, np.number):
                raise ValueError(f"Column {key!r} in {dataset} must be numeric")
            valid = values[~np.isnan(values).reshape(len(values), -1).any(axis=1)] if len(values) else values
            if len(valid) == 0:
                raise ValueError(f"No valid samples for column {key!r} in {dataset}")
            if len(valid) < 2:
                raise ValueError(f"Column {key!r} in {dataset} needs at least two valid rows for ddof=1")
            if not np.isfinite(valid).all():
                raise ValueError(f"Column {key!r} in {dataset} contains infinite values")
        process, _ = build_normalization_process(columns, keys)
    for key in keys:
        processor = process[key]
        for name in ("mean_", "scale_", "raw_min_", "raw_max_", "normed_min_", "normed_max_"):
            if not np.isfinite(getattr(processor, name)).all():
                raise ValueError(f"Invalid {name} for column {key!r} in {dataset}")
    save_normalization_process(output, process, "action_cartesian")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="Input HDF5 dataset")
    parser.add_argument("--output", required=True, help="Output statistics path")
    parser.add_argument("--keys", nargs="+", default=["action_cartesian", "proprio"])
    parser.add_argument("--no-overwrite", action="store_true", help="Reject an existing output")
    args = parser.parse_args()
    create_process_stats(args.dataset, args.output, args.keys, overwrite=not args.no_overwrite)


if __name__ == "__main__":
    main()
