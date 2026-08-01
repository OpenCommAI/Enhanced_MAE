"""Public interface for the Enhanced MAE CSI prediction implementation."""

from __future__ import annotations

from typing import Any

__all__ = ["__version__", "build_model"]
__version__ = "0.1.0"


def build_model(history_tti: int = 8, pred_tti: int = 1, **kwargs: Any):
    """Build the paper's Enhanced MAE model.

    PyTorch and the research model are imported lazily so that commands such as
    dataset validation and ``--help`` remain usable without loading CUDA.
    """
    from enhanced_mae.models import build_model as _build_model

    return _build_model(history_tti=history_tti, pred_tti=pred_tti, **kwargs)
