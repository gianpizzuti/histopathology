# Reliability-Aware Benchmarking of Self-Supervised Learning in Histopathology Under 1–10% Label Regimes

Code for the CIBB 2026 paper. This repository contains the experiment notebooks behind every
figure and table in the paper. The work is a **reliability-aware benchmark** of self-supervised
learning (SSL) for low-label histopathology classification: beyond ranking metrics (AUROC,
accuracy, F1), we report **calibration** (ECE, Brier), **OOD confidence diagnostics**, a
**federated-vs-centralized** comparison, and **cross-dataset transfer**.

> **Methods:** SimCLR, BYOL, an MAE-inspired masked-reconstruction baseline
> **Backbones:** ResNet-18, ViT-B/16 (frozen-encoder linear probing)
> **Datasets:** PatchCamelyon (PCam), PANDA (prostate), colorectal histology tiles
> **Label regimes:** 1%, 5%, 10%

## Repository layout

```
.
├── README.md
├── LICENSE                 # MIT
├── requirements.txt
├── .gitignore
└── notebooks/
    ├── 01_resnet18_benchmark.ipynb
    ├── 02_vit_pcam_simclr.ipynb
    ├── 03_vit_pcam_byol.ipynb
    ├── 04_vit_panda_simclr.ipynb
    ├── 05_vit_panda_byol.ipynb
    ├── 06_supervised_sanity_baselines.ipynb
    ├── 07_mae_baseline_pcam.ipynb
    ├── 08_federated_simclr_pcam.ipynb
    ├── 09_ood_diagnostics.ipynb
    └── 10_transfer_pcam_to_colorectal.ipynb
```

## Notebook → paper map

| Notebook | Experiment | Paper element |
|---|---|---|
| `01_resnet18_benchmark.ipynb` | ResNet-18, SimCLR + BYOL, PCam + PANDA, 3 splits × 3 seeds (n = 9) | Fig. 1, Table 2 (ResNet rows), §3.1 |
| `02_vit_pcam_simclr.ipynb` | ViT-B/16, PCam, SimCLR, 1 split × 3 seeds | Fig. 1, Table 2 (ViT) |
| `03_vit_pcam_byol.ipynb` | ViT-B/16, PCam, BYOL, 1 split × 3 seeds | Fig. 1, Table 2 (ViT) |
| `04_vit_panda_simclr.ipynb` | ViT-B/16, PANDA, SimCLR, 1 split × 3 seeds | Fig. 1, Table 2 (ViT) |
| `05_vit_panda_byol.ipynb` | ViT-B/16, PANDA, BYOL, 1 split × 3 seeds | Fig. 1, Table 2 (ViT, PANDA 0.7176 row) |
| `06_supervised_sanity_baselines.ipynb` | Supervised ResNet-18 (scratch / ImageNet init) on the same splits | §3.1 "Sanity baselines (supervised)" |
| `07_mae_baseline_pcam.ipynb` | MAE-inspired masked-reconstruction baseline on PCam | Table 3 (MAE row), §3.3 |
| `08_federated_simclr_pcam.ipynb` | Federated SimCLR on PCam (3 clients, FedAvg, 5 rounds) | Table 3 (Federated rows), §3.3 |
| `09_ood_diagnostics.ipynb` | OOD confidence diagnostics, PCam ↔ PANDA (MSP / uncertainty) | Fig. 3, Table 3 (OOD rows), §3.3 |
| `10_transfer_pcam_to_colorectal.ipynb` | Transfer: PCam SSL features → colorectal 8-class linear probe | Table 3 (Transfer row), §3.3 |

The PCA representation-space visualization (Fig. 4) is produced from the embeddings exported by the
benchmark / OOD notebooks.

## Datasets

Data is **not** included in this repository. The notebooks were developed on
[Kaggle](https://www.kaggle.com/) and read from `/kaggle/input/...`; adjust the data paths if you
run them elsewhere.

| Dataset | Source |
|---|---|
| **PatchCamelyon (PCam)** — tumor / non-tumor breast patches (96×96) | <https://github.com/basveeling/pcam> |
| **PANDA** — prostate cancer WSIs (patch proxy from ISUP grades) | <https://www.kaggle.com/c/prostate-cancer-grade-assessment> |
| **Colorectal histology** — 8 tissue classes (transfer target) | <https://www.kaggle.com/datasets/kmader/colorectal-histology-mnist> |

See the paper's references [5] (PCam), [6] (PANDA), [7] (colorectal) for citations.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab   # or jupyter notebook
```

A CUDA-capable GPU is recommended (the notebooks use mixed precision when CUDA is available).

## Reproducibility notes

- **Cell outputs were stripped** from the notebooks to keep the repository small and diffs readable.
  Re-run a notebook end-to-end to regenerate its figures and result CSVs.
- ResNet-18 results are averaged over **3 splits × 3 seeds (n = 9)**; ViT-B/16 uses **1 fixed split
  × 3 seeds (n = 3)** due to compute constraints, with identical splits across methods.
- Feature standardization is fit on **train only** and applied to validation with the same
  parameters.
- The linear probe is trained on **frozen** SSL features (10 epochs, Adam, lr 1e-3, batch 128).

## Citation

```bibtex
@inproceedings{ssl_histopath_reliability_cibb2026,
  title     = {Reliability-Aware Benchmarking of Self-Supervised Learning in Histopathology Under 1--10\% Label Regimes},
  author    = {TODO: author list for camera-ready},
  booktitle = {Proceedings of CIBB 2026},
  year      = {2026}
}
```

## License

Code released under the [MIT License](LICENSE).
