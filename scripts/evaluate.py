#!/usr/bin/env python3
"""Portable evaluator for the paper's Enhanced MAE metrics."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = REPO_ROOT / "experiments" / "evaluation" / "metrics"
EVAL_SCRIPT = EVAL_DIR / "eval_enhanced_mae_multi_hop_random_v1.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Enhanced MAE with NMSE, R2, EVM, parameters, and FLOPs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--prediction", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--force-hop", type=int, choices=(1, 3, 5, 7))
    parser.add_argument("--save-predictions", action="store_true")
    return parser.parse_args()


def load_evaluator():
    sys.path.insert(0, str(EVAL_DIR))
    module_name = "_enhanced_mae_evaluator"
    spec = importlib.util.spec_from_file_location(module_name, EVAL_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load evaluator: {EVAL_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    args = parse_args()
    data_dir = args.data.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    if not data_dir.is_dir():
        raise SystemExit(f"Dataset directory does not exist: {data_dir}")
    if not checkpoint.is_file():
        raise SystemExit(f"Checkpoint does not exist: {checkpoint}")
    for name in ("test_inputs.npy", "test_labels.npy"):
        if not (data_dir / name).is_file():
            raise SystemExit(f"Missing evaluation file: {data_dir / name}")

    module = load_evaluator()
    cfg = module.cfg
    cfg.data_dir = str(data_dir)
    cfg.ckpt_path = str(checkpoint)
    cfg.output_dir = str(output_dir)
    cfg.history_tti = args.history
    cfg.pred_tti = args.prediction
    cfg.batch_size = args.batch_size
    cfg.num_workers = args.workers
    cfg.device = args.device
    cfg.seed = args.seed
    cfg.force_hop_step = args.force_hop
    cfg.save_predictions = args.save_predictions
    module.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
