"""Command-line interface for training, evaluation, and validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from enhanced_mae.data import validate_dataset
from enhanced_mae.runner import (
    TRAINERS,
    EvaluateOptions,
    TrainOptions,
    run_evaluation,
    run_training,
)


def _add_shape_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--history", type=int, default=8, help="Historical TTIs.")
    parser.add_argument("--prediction", type=int, default=1, help="Future TTIs.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="enhanced-mae",
        description="Enhanced MAE experiments for CSI prediction.",
    )
    parser.add_argument("--version", action="version", version="%(prog)s 0.1.0")
    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="Train the main model, a baseline, or an ablation.")
    train.add_argument("--model", choices=TRAINERS, default="enhanced-mae")
    train.add_argument("--data", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    _add_shape_arguments(train)
    train.add_argument("--epochs", type=int, default=300)
    train.add_argument("--batch-size", type=int, default=128)
    train.add_argument("--workers", type=int, default=1)
    train.add_argument("--lr", type=float, default=3e-4)
    train.add_argument("--seed", type=int, default=42)
    train.add_argument("--device", default="cuda")
    train.add_argument("--save-every", type=int, default=50)
    train.add_argument("--gpt-path", type=Path)
    train.add_argument("--dry-run", action="store_true")

    evaluate = commands.add_parser("evaluate", help="Evaluate an Enhanced MAE checkpoint.")
    evaluate.add_argument("--data", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    _add_shape_arguments(evaluate)
    evaluate.add_argument("--batch-size", type=int, default=128)
    evaluate.add_argument("--workers", type=int, default=0)
    evaluate.add_argument("--device", default="cuda")
    evaluate.add_argument("--seed", type=int, default=2026)
    evaluate.add_argument("--force-hop", type=int, choices=(1, 3, 5, 7))
    evaluate.add_argument("--save-predictions", action="store_true")

    check = commands.add_parser("check-data", help="Validate a processed NumPy dataset.")
    check.add_argument("--data", type=Path, required=True)
    _add_shape_arguments(check)
    check.add_argument(
        "--skip-finite-check",
        action="store_true",
        help="Skip the full NaN/Inf scan for a faster metadata-only check.",
    )

    commands.add_parser("smoke-test", help="Run one synthetic CPU forward pass.")
    return parser


def _train(args: argparse.Namespace) -> int:
    options = TrainOptions(
        model=args.model,
        data=args.data,
        output=args.output,
        history_tti=args.history,
        pred_tti=args.prediction,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.workers,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        save_every=args.save_every,
        gpt_path=args.gpt_path,
    )
    print(json.dumps(options.to_dict(), indent=2, ensure_ascii=False))
    if args.dry_run:
        return 0
    validate_dataset(
        options.data,
        history_tti=options.history_tti,
        pred_tti=options.pred_tti,
    )
    if options.model == "llm4cp" and options.gpt_path is None:
        raise ValueError("--gpt-path is required for the LLM4CP baseline")
    run_training(options)
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    options = EvaluateOptions(
        data=args.data,
        checkpoint=args.checkpoint,
        output=args.output,
        history_tti=args.history,
        pred_tti=args.prediction,
        batch_size=args.batch_size,
        num_workers=args.workers,
        device=args.device,
        seed=args.seed,
        force_hop_step=args.force_hop,
        save_predictions=args.save_predictions,
    )
    validate_dataset(
        options.data,
        history_tti=options.history_tti,
        pred_tti=options.pred_tti,
    )
    if not options.checkpoint.expanduser().is_file():
        raise ValueError(f"Checkpoint does not exist: {options.checkpoint}")
    run_evaluation(options)
    return 0


def _check_data(args: argparse.Namespace) -> int:
    report = validate_dataset(
        args.data,
        history_tti=args.history,
        pred_tti=args.prediction,
        check_finite=not args.skip_finite_check,
    )
    print(f"Dataset: {report.data_dir}")
    for array in report.arrays:
        print(
            f"  {array.name:18s} raw={str(array.raw_shape):18s} "
            f"logical={str(array.logical_shape):20s} "
            f"dtype={array.dtype:10s} finite={'yes' if array.finite else 'NO'}"
        )
    print(
        f"OK: valid {report.history_tti}-to-{report.pred_tti} dataset "
        f"({report.train_samples} train, {report.test_samples} test samples)."
    )
    return 0


def _smoke_test() -> int:
    import torch

    from enhanced_mae import build_model

    torch.manual_seed(42)
    model = build_model(history_tti=8, pred_tti=1)
    model.eval()
    inputs = torch.zeros(1, 2, 64, 256)
    with torch.inference_mode():
        outputs = model(inputs, force_hop_step=1)
    expected = (1, 2, 64, 32)
    if tuple(outputs.shape) != expected:
        raise RuntimeError(f"Got output shape {tuple(outputs.shape)}, expected {expected}")
    parameters = sum(parameter.numel() for parameter in model.parameters())
    print(f"OK: input {tuple(inputs.shape)} -> output {tuple(outputs.shape)}")
    print(f"Model parameters: {parameters:,}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "train":
        return _train(args)
    if args.command == "evaluate":
        return _evaluate(args)
    if args.command == "check-data":
        return _check_data(args)
    if args.command == "smoke-test":
        return _smoke_test()
    raise RuntimeError(f"Unhandled command: {args.command}")
