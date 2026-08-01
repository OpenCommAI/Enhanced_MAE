# Public-release checklist

The repository structure and portable entry points are ready. Complete the items below before announcing exact paper reproducibility.

## Required

- [ ] Upload the processed training/test arrays for every reported SNR and speed.
- [ ] Upload the ten fixed-speed test splits used in the mobility figures.
- [ ] Upload the SVD-threshold datasets or the preprocessing/generation code needed to create them.
- [ ] Upload normalization statistics used by the spectral-efficiency tests.
- [ ] Upload `best.pt` for Enhanced MAE and every reported baseline.
- [ ] Add direct artifact URLs and SHA-256 checksums to `docs/DATA.md`.
- [ ] Record the exact Python, PyTorch, CUDA, cuDNN, GPU, and QuaDRiGa versions used for the final results.
- [ ] Run every command in `docs/REPRODUCIBILITY.md` in a clean environment.
- [ ] Choose and add a software license. This template intentionally does not select legal terms on the authors' behalf.
- [ ] Replace the provisional BibTeX/CFF record with the final venue, volume, pages, and DOI.

## Recommended

- [ ] Publish one small sample dataset that completes in minutes for CI/tutorial use.
- [ ] Add a single command that downloads and verifies public artifacts.
- [ ] Save machine-readable CSV/JSON summaries for all paper tables and figures.
- [ ] Add GitHub issue templates for installation and reproduction reports.
- [ ] Create a tagged release and archive it on Zenodo to obtain a software DOI.
- [ ] Add the paper's arXiv or publisher URL to both READMEs when available.
