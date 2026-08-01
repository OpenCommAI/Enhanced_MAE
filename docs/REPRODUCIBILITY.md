# Reproducibility guide

This guide separates a quick implementation check from full paper reproduction. Start with the smallest tier that answers your question.

## Tier 0: validate the installation

No dataset or checkpoint is required.

```bash
conda env create -f environment.yml
conda activate enhanced-mae
enhanced-mae smoke-test
```

Expected outcome: the Enhanced MAE builds on CPU and maps a synthetic `(1, 2, 64, 256)` input to a `(1, 2, 64, 32)` prediction.

## Tier 1: validate a processed dataset

```bash
enhanced-mae check-data \
  --data data/25dB_svd090_norm_8to1_dataset \
  --history 8 \
  --prediction 1
```

The command checks filenames, array ranks, channel layout, time-frequency dimensions, sample counts, dtypes, and non-finite values.

## Tier 2: train the main model

The paper uses AdamW, batch size 128, initial learning rate `3e-4`, at most 300 epochs, early stopping, and one NVIDIA RTX 3090.

```bash
enhanced-mae train \
  --model enhanced-mae \
  --data data/25dB_svd090_norm_8to1_dataset \
  --output outputs/enhanced-mae-25db \
  --history 8 \
  --prediction 1 \
  --batch-size 128 \
  --epochs 300 \
  --lr 3e-4 \
  --seed 42 \
  --device cuda
```

The research trainer creates a timestamped run directory and stores `best.pt`, `last.pt`, periodic checkpoints, configuration metadata, and training logs.

Use `--dry-run` to inspect the resolved configuration without importing PyTorch or starting a run:

```bash
enhanced-mae train \
  --model enhanced-mae \
  --data data/25dB_svd090_norm_8to1_dataset \
  --output outputs/debug \
  --dry-run
```

## Tier 3: evaluate the main checkpoint

```bash
enhanced-mae evaluate \
  --data data/25dB_svd090_norm_8to1_dataset \
  --checkpoint outputs/enhanced-mae-25db/<run-name>/best.pt \
  --output outputs/evaluation \
  --batch-size 128 \
  --seed 2026 \
  --device cuda
```

The evaluator reports global NMSE, R², EVM, trainable and total parameters, FLOPs, checkpoint size, hop distribution, and theoretical inference time. It saves a text summary and per-sample metric arrays.

## Baselines and ablations

The same portable training entry point dispatches to the original research scripts:

| CLI model | Research implementation |
|---|---|
| `enhanced-mae` | Proposed SVD + hopping + multi-scale model |
| `mae` | Plain MAE |
| `rnn` | RNN baseline |
| `lstm` | LSTM baseline |
| `transformer` | Transformer baseline |
| `llm4cp` | LLM4CP baseline |
| `ablation-hop` | MAE with hopping only |
| `ablation-multiscale` | MAE with multi-scale fusion only |

Example:

```bash
enhanced-mae train \
  --model lstm \
  --data data/25dB_zscore_norm_8to1_dataset \
  --output outputs/lstm-25db \
  --device cuda
```

LLM4CP additionally needs a local GPT-2 directory:

```bash
enhanced-mae train \
  --model llm4cp \
  --data data/25dB_zscore_norm_8to1_dataset \
  --output outputs/llm4cp-25db \
  --gpt-path pretrained/gpt2 \
  --device cuda
```

## Paper result map

| Paper result | Code location | Additional artifacts |
|---|---|---|
| Table I: NMSE/R²/EVM/complexity | `experiments/evaluation/metrics/` | 25 dB test split and one checkpoint per model |
| Fig. 3: time-window length | `experiments/evaluation/history_length/` | Processed datasets/checkpoints for each window |
| Fig. 4: SVD threshold and SNR | Main trainer plus threshold-specific datasets | Preprocessing metadata for every threshold/SNR |
| Fig. 5: model ablations vs. speed | `experiments/evaluation/ablations/` | Per-speed SVD and non-SVD test splits |
| Fig. 6: NMSE vs. SNR | Training/evaluation scripts | Per-SNR processed data and checkpoints |
| Fig. 7: NMSE vs. UE speed | `experiments/evaluation/ablations/` | Ten fixed-speed test splits |
| Spectral efficiency experiments | `experiments/evaluation/spectral_efficiency/` | Raw-scale statistics and all model checkpoints |

## Determinism notes

The supplied trainers seed Python, NumPy, and PyTorch. The main evaluator also disables cuDNN benchmarking and requests deterministic cuDNN behavior. Exact bitwise agreement can still depend on the GPU, CUDA, cuDNN, PyTorch build, and data preprocessing version. For an archival release, record:

```bash
python --version
python -c "import torch; print(torch.__version__, torch.version.cuda)"
nvidia-smi
```

Store this output with the experiment logs and publish SHA-256 checksums for all datasets and checkpoints.
