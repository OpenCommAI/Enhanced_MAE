#!/usr/bin/env python3
"""Validate the NumPy dataset contract used by the CSI prediction trainers."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


REQUIRED_FILES = (
    "train_inputs.npy",
    "train_labels.npy",
    "test_inputs.npy",
    "test_labels.npy",
)


def canonical_shape(array: np.ndarray, name: str) -> tuple[int, int, int, int]:
    """Return the logical channel-first shape without copying the array."""
    if array.ndim == 3:
        return (array.shape[0], 2, array.shape[1], array.shape[2])
    if array.ndim == 4 and array.shape[1] == 2:
        return tuple(array.shape)
    if array.ndim == 4 and array.shape[-1] == 2:
        return (array.shape[0], 2, array.shape[1], array.shape[2])
    raise ValueError(
        f"{name}: expected (N,64,W), (N,2,64,W), or (N,64,W,2); "
        f"got {array.shape}"
    )


def finite_status(array: np.ndarray) -> str:
    if np.issubdtype(array.dtype, np.number):
        return "yes" if np.isfinite(array).all() else "NO"
    return "not numeric"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check filenames, shapes, dtypes, sample counts, and finite values."
    )
    parser.add_argument("--data", type=Path, required=True, help="Dataset directory.")
    parser.add_argument("--history", type=int, default=8, help="Historical TTIs.")
    parser.add_argument("--prediction", type=int, default=1, help="Future TTIs.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_dir = args.data.expanduser().resolve()
    if not data_dir.is_dir():
        raise SystemExit(f"Dataset directory does not exist: {data_dir}")
    if args.history < 1 or args.prediction < 1:
        raise SystemExit("--history and --prediction must be positive integers.")

    missing = [name for name in REQUIRED_FILES if not (data_dir / name).is_file()]
    if missing:
        names = "\n  - ".join(missing)
        raise SystemExit(f"Missing required files:\n  - {names}")

    arrays: dict[str, np.ndarray] = {}
    shapes: dict[str, tuple[int, int, int, int]] = {}
    print(f"Dataset: {data_dir}")
    for name in REQUIRED_FILES:
        array = np.load(data_dir / name, mmap_mode="r", allow_pickle=False)
        shape = canonical_shape(array, name)
        arrays[name] = array
        shapes[name] = shape
        print(
            f"  {name:18s} raw={str(array.shape):18s} "
            f"logical={str(shape):20s} dtype={str(array.dtype):10s} "
            f"finite={finite_status(array)}"
        )

    expected_input = (2, 64, 32 * args.history)
    expected_label = (2, 64, 32 * args.prediction)
    errors: list[str] = []
    for split in ("train", "test"):
        input_name = f"{split}_inputs.npy"
        label_name = f"{split}_labels.npy"
        if shapes[input_name][1:] != expected_input:
            errors.append(
                f"{input_name}: logical sample shape {shapes[input_name][1:]}, "
                f"expected {expected_input}"
            )
        if shapes[label_name][1:] != expected_label:
            errors.append(
                f"{label_name}: logical sample shape {shapes[label_name][1:]}, "
                f"expected {expected_label}"
            )
        if shapes[input_name][0] != shapes[label_name][0]:
            errors.append(
                f"{split}: {shapes[input_name][0]} inputs but "
                f"{shapes[label_name][0]} labels"
            )

    for name, array in arrays.items():
        if np.issubdtype(array.dtype, np.number) and not np.isfinite(array).all():
            errors.append(f"{name}: contains NaN or infinity")

    if errors:
        print("\nValidation failed:")
        for error in errors:
            print(f"  - {error}")
        return 1

    print(
        f"\nOK: valid {args.history}-to-{args.prediction} CSI dataset "
        f"({shapes['train_inputs.npy'][0]} train, "
        f"{shapes['test_inputs.npy'][0]} test samples)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
