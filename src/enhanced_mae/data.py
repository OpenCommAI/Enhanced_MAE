"""Dataset contract and validation utilities for CSI prediction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


REQUIRED_FILES = (
    "train_inputs.npy",
    "train_labels.npy",
    "test_inputs.npy",
    "test_labels.npy",
)


@dataclass(frozen=True)
class ArrayReport:
    """Metadata collected for one memory-mapped NumPy array."""

    name: str
    raw_shape: tuple[int, ...]
    logical_shape: tuple[int, int, int, int]
    dtype: str
    finite: bool


@dataclass(frozen=True)
class DatasetReport:
    """Validation result for one processed dataset directory."""

    data_dir: Path
    history_tti: int
    pred_tti: int
    arrays: tuple[ArrayReport, ...]

    @property
    def train_samples(self) -> int:
        return self.arrays[0].logical_shape[0]

    @property
    def test_samples(self) -> int:
        return self.arrays[2].logical_shape[0]


def canonical_shape(array: np.ndarray, name: str) -> tuple[int, int, int, int]:
    """Return the logical ``(N, 2, 64, W)`` shape without copying data."""
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


def to_channel_first(array: np.ndarray, name: str = "array") -> np.ndarray:
    """Convert a supported CSI array to float32 ``(N, 2, 64, W)`` format."""
    canonical_shape(array, name)
    if array.ndim == 3:
        real = np.real(array)
        imag = np.imag(array) if np.iscomplexobj(array) else np.zeros_like(array)
        return np.stack((real, imag), axis=1).astype(np.float32, copy=False)
    if array.shape[1] == 2:
        return array.astype(np.float32, copy=False)
    return np.transpose(array, (0, 3, 1, 2)).astype(np.float32, copy=False)


def validate_dataset(
    data_dir: str | Path,
    *,
    history_tti: int = 8,
    pred_tti: int = 1,
    check_finite: bool = True,
) -> DatasetReport:
    """Validate all files, layouts, shapes, sample counts, and numeric values."""
    path = Path(data_dir).expanduser().resolve()
    if not path.is_dir():
        raise ValueError(f"Dataset directory does not exist: {path}")
    if history_tti < 1 or pred_tti < 1:
        raise ValueError("history_tti and pred_tti must be positive")

    missing = [name for name in REQUIRED_FILES if not (path / name).is_file()]
    if missing:
        raise ValueError(f"Missing required files: {', '.join(missing)}")

    arrays: dict[str, np.ndarray] = {}
    reports: list[ArrayReport] = []
    for name in REQUIRED_FILES:
        array = np.load(path / name, mmap_mode="r", allow_pickle=False)
        logical = canonical_shape(array, name)
        finite = bool(np.isfinite(array).all()) if check_finite else True
        arrays[name] = array
        reports.append(
            ArrayReport(
                name=name,
                raw_shape=tuple(array.shape),
                logical_shape=logical,
                dtype=str(array.dtype),
                finite=finite,
            )
        )

    expected_input = (2, 64, 32 * history_tti)
    expected_label = (2, 64, 32 * pred_tti)
    errors: list[str] = []
    for split in ("train", "test"):
        input_name = f"{split}_inputs.npy"
        label_name = f"{split}_labels.npy"
        input_shape = canonical_shape(arrays[input_name], input_name)
        label_shape = canonical_shape(arrays[label_name], label_name)
        if input_shape[1:] != expected_input:
            errors.append(f"{input_name}: got {input_shape[1:]}, expected {expected_input}")
        if label_shape[1:] != expected_label:
            errors.append(f"{label_name}: got {label_shape[1:]}, expected {expected_label}")
        if input_shape[0] != label_shape[0]:
            errors.append(
                f"{split}: {input_shape[0]} input samples but {label_shape[0]} labels"
            )

    errors.extend(report.name for report in reports if not report.finite)
    if errors:
        raise ValueError("Dataset validation failed: " + "; ".join(errors))

    return DatasetReport(
        data_dir=path,
        history_tti=history_tti,
        pred_tti=pred_tti,
        arrays=tuple(reports),
    )
