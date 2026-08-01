#!/usr/bin/env python3
"""Run one synthetic Enhanced MAE forward pass without external artifacts."""

from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = REPO_ROOT / "experiments" / "training"
sys.path.insert(0, str(MODEL_DIR))

import torch  # noqa: E402
from models_mae_multi_hop_random_v1 import build_mae_channel_target_hop3  # noqa: E402


def main() -> int:
    torch.manual_seed(42)
    model = build_mae_channel_target_hop3(history_tti=8, pred_tti=1)
    model.eval()
    inputs = torch.zeros(1, 2, 64, 32 * 8)
    with torch.inference_mode():
        outputs = model(inputs, force_hop_step=1)
    expected = (1, 2, 64, 32)
    if tuple(outputs.shape) != expected:
        raise RuntimeError(f"Unexpected output shape {tuple(outputs.shape)}; expected {expected}")
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"OK: input {tuple(inputs.shape)} -> output {tuple(outputs.shape)}")
    print(f"Model parameters: {parameters:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
