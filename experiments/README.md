# Experiment implementations

This directory contains the scripts used to produce the paper's training runs, baselines, ablations, and evaluation results. Most users should use the installable `enhanced_mae` package under `src/` rather than invoke these files directly.

## Layout

```text
experiments/
├── training/                   # Main model, baselines, and ablation trainers
└── evaluation/
    ├── ablations/      # Component ablations across UE speeds
    ├── metrics/     # Accuracy and complexity metrics
    ├── spectral_efficiency/                # Spectral-efficiency evaluation
    └── history_length/     # Historical-window experiments
```

The original experiment filenames are retained to preserve the mapping between paper results, saved checkpoints, and research logs.

## Recommended interface

Install the repository once:

```bash
pip install -e .
```

Then use the unified command:

```bash
enhanced-mae --help
enhanced-mae check-data --data data/25dB_svd090_norm_8to1_dataset
enhanced-mae train --model enhanced-mae --data data/25dB_svd090_norm_8to1_dataset --output outputs/main
enhanced-mae evaluate --data data/25dB_svd090_norm_8to1_dataset --checkpoint checkpoints/best.pt --output outputs/eval
```

The stable Python API is:

```python
from enhanced_mae import build_model

model = build_model(history_tti=8, pred_tti=1)
```

Some evaluation directories intentionally contain frozen copies of model definitions. They are retained for checkpoint compatibility and historical reproducibility; new development should target `src/enhanced_mae/`.
