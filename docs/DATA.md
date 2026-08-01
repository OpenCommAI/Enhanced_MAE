# Data preparation

## Paper setting

The paper uses QuaDRiGa to generate time-varying CSI following the 3GPP channel model.

| Item | Setting |
|---|---|
| Scenario | 3GPP UMa NLOS |
| BS array | Dual-polarized UPA, `Nh = 4`, `Nv = 4` |
| Effective subcarriers | 64 |
| Bandwidth | 11.52 MHz |
| Carrier frequency | 2.4 GHz |
| CSI sampling interval | 0.5 ms |
| Training samples | 16,000 |
| Training UE speeds | Uniformly sampled from 10 to 100 km/h |
| Test speeds | 10, 20, ..., 100 km/h |
| Test samples | 1,000 per speed |
| Channel noise | Complex AWGN at the evaluated SNR |
| Default prediction task | 8 historical TTIs to 1 future TTI |

The Enhanced MAE experiments apply energy-threshold SVD reconstruction and z-score normalization before training. The threshold is selected on the validation data for each SNR and then held fixed for the corresponding experiment.

## Files expected by the code

Each processed dataset directory must contain:

```text
<dataset>/
├── train_inputs.npy
├── train_labels.npy
├── test_inputs.npy
└── test_labels.npy
```

For an `H`-to-`P` task, the loader accepts any of the following representations:

| Representation | Input shape | Label shape |
|---|---|---|
| Complex NumPy | `(N, 64, 32H)` | `(N, 64, 32P)` |
| Channel first | `(N, 2, 64, 32H)` | `(N, 2, 64, 32P)` |
| Channel last | `(N, 64, 32H, 2)` | `(N, 64, 32P, 2)` |

The two real-valued channels store the real and imaginary parts. The default `H = 8`, `P = 1` setting therefore becomes `(N, 2, 64, 256)` for inputs and `(N, 2, 64, 32)` for labels.

Run the repository validator before training:

```bash
enhanced-mae check-data --data data/25dB_svd090_norm_8to1_dataset
```

For a different prediction horizon:

```bash
enhanced-mae check-data --data data/8to4_dataset --history 8 --prediction 4
```

## Normalization metadata

The training loaders consume the four arrays above directly. Some spectral-efficiency tests additionally invert z-score normalization and therefore require the mean and standard-deviation arrays referenced by the corresponding scripts in `experiments/evaluation/spectral_efficiency/`. Release those statistics beside the SE test data and record their exact filenames in the artifact download instructions.

## Current artifact status

Processed datasets, normalization statistics, and checkpoints are not included in this local repository snapshot. This means:

- the model architecture can be smoke-tested without external artifacts;
- training can be reproduced once the four processed arrays are provided;
- exact table and figure reproduction additionally needs the original dataset splits, preprocessing metadata, and paper checkpoints.

Before making the repository public, upload these artifacts to a stable host (for example Zenodo, Hugging Face, or an institutional archive), add checksums, and replace this section with direct download commands. The required items are enumerated in [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md).
