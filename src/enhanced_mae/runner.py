"""Adapters that run the original paper trainers with portable configuration."""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import ModuleType


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
TRAINING_ROOT = REPOSITORY_ROOT / "experiments" / "training"
EVALUATION_ROOT = (
    REPOSITORY_ROOT / "experiments" / "evaluation" / "metrics"
)


@dataclass(frozen=True)
class TrainerSpec:
    script: str
    description: str
    ablation_type: str | None = None


TRAINERS = {
    "enhanced-mae": TrainerSpec(
        "main_train_8TTi_multi_NMSE_hop_random.py",
        "Proposed hopping and multi-scale Enhanced MAE",
    ),
    "mae": TrainerSpec("train_plain_mae_original.py", "Plain MAE baseline"),
    "rnn": TrainerSpec("RNN_train.py", "RNN baseline"),
    "lstm": TrainerSpec("LSTM_train.py", "LSTM baseline"),
    "transformer": TrainerSpec("Transformer_train_8TTi.py", "Transformer baseline"),
    "llm4cp": TrainerSpec("LLM4CP_train.py", "LLM4CP baseline"),
    "ablation-hop": TrainerSpec(
        "train_ablation.py",
        "Time-frequency hopping ablation",
        "hop_only",
    ),
    "ablation-multiscale": TrainerSpec(
        "train_ablation.py",
        "Multi-scale fusion ablation",
        "multiscale_only",
    ),
}


@dataclass(frozen=True)
class TrainOptions:
    model: str
    data: Path
    output: Path
    history_tti: int = 8
    pred_tti: int = 1
    epochs: int = 300
    batch_size: int = 128
    num_workers: int = 1
    lr: float = 3e-4
    seed: int = 42
    device: str = "cuda"
    save_every: int = 50
    gpt_path: Path | None = None

    def to_dict(self) -> dict[str, object]:
        values = asdict(self)
        return {key: str(value) if isinstance(value, Path) else value for key, value in values.items()}


@dataclass(frozen=True)
class EvaluateOptions:
    data: Path
    checkpoint: Path
    output: Path
    history_tti: int = 8
    pred_tti: int = 1
    batch_size: int = 128
    num_workers: int = 0
    device: str = "cuda"
    seed: int = 2026
    force_hop_step: int | None = None
    save_predictions: bool = False


def _load_module(path: Path, name: str) -> ModuleType:
    parent = str(path.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Python module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def configure_trainer(module: ModuleType, options: TrainOptions) -> None:
    """Map portable options onto a legacy trainer's configuration object."""
    cfg = module.cfg
    data = options.data.expanduser().resolve()
    if hasattr(cfg, "data_root"):
        cfg.data_root = str(data.parent)
        if hasattr(cfg, "all_dataset_name"):
            cfg.all_dataset_name = data.name
    elif hasattr(cfg, "data_dir"):
        cfg.data_dir = str(data)
    else:
        raise RuntimeError("Trainer has no supported data path field")

    values = {
        "out_root": str(options.output.expanduser().resolve()),
        "history_tti": options.history_tti,
        "pred_tti": options.pred_tti,
        "epochs": options.epochs,
        "batch_size": options.batch_size,
        "num_workers": options.num_workers,
        "lr": options.lr,
        "seed": options.seed,
        "device": options.device,
        "save_every": options.save_every,
    }
    for field, value in values.items():
        if hasattr(cfg, field):
            setattr(cfg, field, value)

    trainer = TRAINERS[options.model]
    if trainer.ablation_type is not None:
        cfg.model_type = trainer.ablation_type
    if options.gpt_path is not None and hasattr(cfg, "gpt_path"):
        cfg.gpt_path = str(options.gpt_path.expanduser().resolve())
        cfg.local_files_only = True
    if hasattr(cfg, "__post_init__"):
        cfg.__post_init__()


def run_training(options: TrainOptions) -> None:
    """Load and execute the selected original trainer."""
    if options.model not in TRAINERS:
        raise ValueError(f"Unknown trainer: {options.model}")
    spec = TRAINERS[options.model]
    module = _load_module(TRAINING_ROOT / spec.script, f"_enhanced_mae_train_{options.model}")
    configure_trainer(module, options)
    module.main()


def run_evaluation(options: EvaluateOptions) -> None:
    """Run the paper's Enhanced MAE metric evaluator."""
    script = EVALUATION_ROOT / "eval_enhanced_mae_multi_hop_random_v1.py"
    module = _load_module(script, "_enhanced_mae_evaluation")
    cfg = module.cfg
    cfg.data_dir = str(options.data.expanduser().resolve())
    cfg.ckpt_path = str(options.checkpoint.expanduser().resolve())
    cfg.output_dir = str(options.output.expanduser().resolve())
    cfg.history_tti = options.history_tti
    cfg.pred_tti = options.pred_tti
    cfg.batch_size = options.batch_size
    cfg.num_workers = options.num_workers
    cfg.device = options.device
    cfg.seed = options.seed
    cfg.force_hop_step = options.force_hop_step
    cfg.save_predictions = options.save_predictions
    module.main()
