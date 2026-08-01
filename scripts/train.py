#!/usr/bin/env python3
"""Portable command-line entry point for the original research trainers."""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = REPO_ROOT / "experiments" / "training"
MODEL_SPECS = {
    "enhanced-mae": ("main_train_8TTi_multi_NMSE_hop_random.py", None),
    "mae": ("train_plain_mae_original.py", None),
    "rnn": ("RNN_train.py", None),
    "lstm": ("LSTM_train.py", None),
    "transformer": ("Transformer_train_8TTi.py", None),
    "llm4cp": ("LLM4CP_train.py", None),
    "ablation-hop": ("train_ablation.py", "hop_only"),
    "ablation-multiscale": ("train_ablation.py", "multiscale_only"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Enhanced MAE or a paper baseline with portable paths.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", choices=MODEL_SPECS, default="enhanced-mae")
    parser.add_argument("--data", type=Path, required=True, help="Directory with four .npy files.")
    parser.add_argument("--output", type=Path, required=True, help="Root directory for runs.")
    parser.add_argument("--history", type=int, default=8)
    parser.add_argument("--prediction", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda", help="'cuda', 'cuda:N', or 'cpu'.")
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--gpt-path", type=Path, help="Local GPT-2 directory for LLM4CP.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the resolved configuration without importing the trainer.",
    )
    return parser.parse_args()


def load_module(path: Path) -> ModuleType:
    sys.path.insert(0, str(path.parent))
    module_name = f"_enhanced_mae_{path.stem}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load trainer: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def resolved_settings(args: argparse.Namespace) -> dict[str, object]:
    data_dir = args.data.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()
    script_name, ablation_type = MODEL_SPECS[args.model]
    return {
        "model": args.model,
        "trainer": str(TRAIN_DIR / script_name),
        "data": str(data_dir),
        "output": str(output_dir),
        "history_tti": args.history,
        "pred_tti": args.prediction,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "lr": args.lr,
        "seed": args.seed,
        "device": args.device,
        "save_every": args.save_every,
        "ablation_type": ablation_type,
        "gpt_path": str(args.gpt_path.expanduser().resolve()) if args.gpt_path else None,
    }


def validate_args(args: argparse.Namespace, settings: dict[str, object]) -> None:
    if args.history < 1 or args.prediction < 1:
        raise SystemExit("--history and --prediction must be positive.")
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0:
        raise SystemExit("Invalid epochs, batch size, or worker count.")
    data_dir = Path(str(settings["data"]))
    required = ("train_inputs.npy", "train_labels.npy", "test_inputs.npy", "test_labels.npy")
    missing = [name for name in required if not (data_dir / name).is_file()]
    if missing and not args.dry_run:
        raise SystemExit(
            f"Dataset is missing {', '.join(missing)}. "
            "Run scripts/check_dataset.py for a detailed report."
        )
    if args.model == "llm4cp" and args.gpt_path is None and not args.dry_run:
        raise SystemExit("--gpt-path is required when --model llm4cp is selected.")


def apply_settings(module: ModuleType, args: argparse.Namespace, settings: dict[str, object]) -> None:
    cfg = module.cfg
    data_dir = Path(str(settings["data"]))

    if hasattr(cfg, "data_root"):
        cfg.data_root = str(data_dir.parent)
        if hasattr(cfg, "all_dataset_name"):
            cfg.all_dataset_name = data_dir.name
    elif hasattr(cfg, "data_dir"):
        cfg.data_dir = str(data_dir)
    else:
        raise RuntimeError("Selected trainer has no supported data path field.")

    shared = {
        "out_root": settings["output"],
        "history_tti": args.history,
        "pred_tti": args.prediction,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "lr": args.lr,
        "seed": args.seed,
        "device": args.device,
        "save_every": args.save_every,
    }
    for field, value in shared.items():
        if hasattr(cfg, field):
            setattr(cfg, field, value)

    ablation_type = settings["ablation_type"]
    if ablation_type is not None:
        cfg.model_type = ablation_type
    if args.gpt_path is not None and hasattr(cfg, "gpt_path"):
        cfg.gpt_path = str(args.gpt_path.expanduser().resolve())
        cfg.local_files_only = True
    if hasattr(cfg, "__post_init__"):
        cfg.__post_init__()


def main() -> int:
    args = parse_args()
    settings = resolved_settings(args)
    validate_args(args, settings)
    print(json.dumps(settings, indent=2, ensure_ascii=False))
    if args.dry_run:
        return 0

    script_name, _ = MODEL_SPECS[args.model]
    module = load_module(TRAIN_DIR / script_name)
    apply_settings(module, args, settings)
    module.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
