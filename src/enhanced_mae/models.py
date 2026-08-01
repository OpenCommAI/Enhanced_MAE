"""Stable model imports backed by the original paper implementation."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
_TRAINING_ROOT = _REPOSITORY_ROOT / "experiments" / "training"


def _ensure_research_path() -> None:
    path = str(_TRAINING_ROOT)
    if path not in sys.path:
        sys.path.insert(0, path)


def build_model(history_tti: int = 8, pred_tti: int = 1, **kwargs: Any):
    """Build Enhanced MAE using the architecture reported in the paper."""
    _ensure_research_path()
    from models_mae_multi_hop_random_v1 import build_mae_channel_target_hop3

    return build_mae_channel_target_hop3(
        history_tti=history_tti,
        pred_tti=pred_tti,
        **kwargs,
    )
