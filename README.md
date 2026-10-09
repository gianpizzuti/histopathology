# SSL reliability in histopathology: LNBI extension

Code for the journal version (LNBI) of the conference paper. The conference
notebooks are kept unchanged in `notebooks/legacy/`. New experiments use the
`sslhist` package, which reproduces their protocol exactly (checked by
`tests/test_protocol.py` against verbatim copies of the notebook functions).

```
configs/            base.yaml (conference protocol) + one YAML per experiment
src/sslhist/        data, models, ssl (SimCLR/BYOL/Barlow Twins), probe, metrics, runner, report, plotting,
                    ood + supervised (E6)
scripts/            check_setup.py · run_unit.py · launch.py · status.py · aggregate.py · make_splits.py · prefetch_weights.py
                    (E6: ood.py and supervised.py in src/sslhist, same scripts)
tests/              protocol equivalence with the notebooks + CPU smoke tests
results/lnbi/       committed: per-run JSON, summary CSV/LaTeX, PDF figures
artifacts/          NOT committed: encoders, features, logs
notebooks/legacy/   conference notebooks (Kaggle)
```

## Setup (GPU server)

Everything lives in one folder: code, data, the conda environment `hist`, caches and
credentials. GPUs are chosen when launching (`--gpus 2,3`, `--gpu 2`), numbered as in
`nvidia-smi`; nothing is fixed in the environment.
The server driver is CUDA 12.4: install a matching PyTorch build **before** the package
(the default PyPI wheel may target a newer CUDA).

```bash
G=/opt/scratch/labs/vap/gianluca/ssl-histo
mkdir -p $G/data $G/envs $G/.kaggle $G/.cache && cd $G
git clone -b claude/experiment-package-extensions-1nwg8n https://github.com/gianpizzuti/histopathology.git

# conda environment "hist", stored in $G/envs (other environments are not touched)
conda config --append envs_dirs $G/envs          # so that `conda activate hist` finds it
export CONDA_PKGS_DIRS=$G/.cache/conda-pkgs
conda create -y -p $G/envs/hist python=3.11
conda env config vars set -p $G/envs/hist \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    KAGGLE_CONFIG_DIR=$G/.kaggle TORCH_HOME=$G/.cache/torch \
    PIP_CACHE_DIR=$G/.cache/pip MPLCONFIGDIR=$G/.cache/matplotlib
conda activate hist

cd $G/histopathology
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[dev]" kaggle
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0))"
# expected: 2.6.0+cu124 True 4 NVIDIA H100 NVL
pytest -q            # 39 passed, a few minutes; the tests never use a GPU
```

Later sessions: `conda activate hist && cd /opt/scratch/labs/vap/gianluca/ssl-histo/histopathology`.

## Data

Use **the same Kaggle datasets as the conference runs**: the universe
(`df.sample(frac=0.2, random_state=42)`) and therefore every split depend on
the rows of the CSV files and on which PANDA PNGs exist. The Kaggle metadata of
the legacy notebooks lists the inputs that were attached:

| Kaggle source (notebook metadata) | Dataset |
|---|---|
| competition 11848 (data bundle 862157) | `histopathologic-cancer-detection` (PCam) |
| competition 18647 (data bundle 1126921) | `prostate-cancer-grade-assessment` (PANDA, only `train.csv` is used) |
| dataset id 615046 (version 1101206) | `xhlulu/panda-resized-train-data-512x512`: one 512×512 PNG per slide in `train_images/train_images/<image_id>.png` |

```bash
# needs $G/.kaggle/kaggle.json (chmod 600) and the competition rules accepted on kaggle.com
kaggle competitions download -c histopathologic-cancer-detection -p $G/data/histopathologic-cancer-detection
kaggle competitions download -c prostate-cancer-grade-assessment -f train.csv -p $G/data/prostate-cancer-grade-assessment
kaggle datasets download -d xhlulu/panda-resized-train-data-512x512 -p $G/data/panda-resized
# unzip the archives, then:
cp configs/paths.example.yaml configs/paths.yaml   # paths already set for the server layout above; check them with ls
```

Check the data partition (and compare the printed fingerprints across machines).
For PANDA the script also checks the class counts against Table 1 of the paper
(universe 1055/1068, every split 844/854 train and 211/214 val):

```bash
python scripts/make_splits.py --config configs/e1_vit_matched.yaml --dataset pcam
python scripts/make_splits.py --config configs/e1_vit_matched.yaml --dataset panda
```

## Running an experiment (E1 as example)

```bash
# 1. quick check (a few minutes): environment, data (incl. PANDA vs Table 1 of the paper),
#    the whole E1 grid with 1 split x 1 seed and 60 SSL steps per unit through the launcher,
#    outputs and aggregation, plus an estimate of the full run. Writes only under artifacts/.
python scripts/check_setup.py --gpus 2,3 --per-gpu 3

# 2. whole grid on GPUs 2 and 3, 3 units per GPU (re-run the same command to resume).
#    --gpus (required) takes the ids shown by nvidia-smi: only those GPUs are used. Run it inside tmux.
python scripts/launch.py --config configs/e1_vit_matched.yaml --gpus 2,3 --per-gpu 3 --dry-run
python scripts/launch.py --config configs/e1_vit_matched.yaml --gpus 2,3 --per-gpu 3 2>&1 | tee e1_launch.log

#    Progress at any time (units done, epoch of the running ones, failures, time left):
python scripts/status.py --config configs/e1_vit_matched.yaml
watch -n 60 python scripts/status.py --config configs/e1_vit_matched.yaml   # refreshes every minute

# 3. tables, paired comparison vs ResNet-18, figures
python scripts/aggregate.py --config configs/e1_vit_matched.yaml --reference resnet18
#    experiments on the same splits can be analysed together (written to <experiment>/with_<other>/):
python scripts/aggregate.py --config configs/e2_resnet50.yaml --include e1_vit_matched --reference resnet18
#    --reference backbone:method compares every series with one fixed series (e.g. frozen DINOv2):
python scripts/aggregate.py --config configs/e3_dinov2.yaml --include e1_vit_matched e2_resnet50 e3_barlow \
    --reference resnet18 resnet50:simclr
git add results/lnbi/e1_vit_matched && git commit -m "E1 results" && git push
```

Experiments so far (one config each, same splits and labelled subsets):

| Config | What | Units |
|---|---|---|
| `e1_vit_matched.yaml` | ViT-B/16 and ResNet-18, SimCLR + BYOL | 72 |
| `e2_resnet50.yaml` | ResNet-50, SimCLR + BYOL | 36 |
| `e3_dinov2.yaml` | DINOv2 ViT-S/14 and ViT-B/14, frozen (method `frozen`, ImageNet normalisation); run `python scripts/prefetch_weights.py` once first | 36 |
| `e3_barlow.yaml` | ResNet-18, Barlow Twins | 18 |
| `e4_federated.yaml` | Federated SimCLR ResNet-18 on PCam (FedAvg; 5/10 clients; IID, Dirichlet α 0.5/0.1), paired with the centralised SimCLR of E1 (see below) | 54 |
| `e6_ood.yaml` | OOD analysis PCam ↔ PANDA of all the encoders above + supervised ResNet-18 baselines (see below) | 198 |

A *unit* is one SSL pretraining (dataset, method, backbone, split, seed) followed
by the linear probes at 1/5/10% labels. Logs: `artifacts/<experiment>/logs/<unit>.log`.

Single unit by hand (e.g. to debug), or a timing test of its first 200 SSL steps:

```bash
python scripts/run_unit.py --gpu 2 --config configs/e1_vit_matched.yaml \
    --dataset pcam --method simclr --backbone vit_b_16 --split 0 --seed 0 --timing 200
```

## E4: federated SimCLR

Same protocol as the centralised ResNet-18 SimCLR of E1 (encoder, loss, augmentations,
Adam, AMP, batch 128, splits, seeds, labelled subsets, probe), which is the paired
reference: a federated unit differs from its E1 counterpart only by the federation
(`src/sslhist/federated.py`):
- the training part of the split is divided among K = 5 or 10 clients of equal size, IID or
  with label skew (each client's label distribution ~ Dir(α·C·p), p = overall class proportions,
  Hsu et al. 2019; α = 0.5 and 0.1). Labels define the partition only, never the training;
- FedAvg (McMahan et al. 2017): each round every client trains 1 local epoch from the global model
  (Adam, new optimiser state each round), then all parameters and buffers, BatchNorm statistics
  included, are averaged weighted by client size; 10 rounds, so the data are seen as often as in E1;
- methods are named `simclr-fed-k<K>-<iid|a<α>>`; the result files also hold the client sizes and
  positive rates (`client_sizes`, `client_pos_rates`, `label_skew`).

```bash
python scripts/launch.py --config configs/e4_federated.yaml --gpus 2,3 --per-gpu 3 --dry-run   # 54 units
python scripts/launch.py --config configs/e4_federated.yaml --gpus 2,3 --per-gpu 3 2>&1 | tee e4_launch.log
python scripts/status.py --config configs/e4_federated.yaml
# tables and figures with the centralised SimCLR of E1, paired comparison federated - centralised:
python scripts/aggregate.py --config configs/e4_federated.yaml --include e1_vit_matched:resnet18:simclr \
    --reference resnet18:simclr
# control: the evaluation of notebook 08 vs ours on the same encoders (CPU, a few minutes)
python scripts/legacy08_control.py --config configs/e4_federated.yaml --sources e1_vit_matched:resnet18:simclr e4_federated
```

Why E4 replaces the federated diagnostic of the conference paper (Table 3, notebook 08): its rows
were not comparable with the centralised ones. Notebook 08 used another split (StratifiedKFold,
1 split x 3 seeds), NT-Xent temperature 0.2, no image normalisation, 5 rounds, a probe on raw
features with Adam lr 1e-3 and early stopping on a validation set, and the ECE of the predicted
class (max(p, 1−p)) instead of the ECE of the positive-class probability used everywhere else;
its "non-IID" partition sorted the images by label and then shuffled them, i.e. it was IID; the
paper reports 3 clients, the notebook has 5. `scripts/legacy08_control.py` scores the same
encoders with both evaluations (`results/lnbi/e4_federated/legacy08_control/`).

## E6: is OOD overconfidence specific to SSL?

Run after E1, E2 and E3: no SSL pretraining, it reads their encoders, probes and
features from `artifacts/`. For every unit (ID dataset, method, backbone, split, seed)
and label fraction it scores the ID validation images and 2000 images of the other
dataset (PCam → PANDA, PANDA → PCam; notebook 09: `RandomState(123)`, OOD images
resized to the ID image size). The supervised ResNet-18 baselines of notebook 06
(from scratch and ImageNet init) are trained on the same labelled subsets.

```bash
python scripts/prefetch_weights.py --backbones resnet18      # ImageNet ResNet-18, once
python scripts/launch.py --config configs/e6_ood.yaml --gpus 2,3 --per-gpu 4 --dry-run   # 198 units
python scripts/launch.py --config configs/e6_ood.yaml --gpus 2,3 --per-gpu 4 2>&1 | tee e6_launch.log
python scripts/status.py --config configs/e6_ood.yaml
python scripts/aggregate.py --config configs/e6_ood.yaml --reference resnet18:sup_imagenet resnet18:sup_scratch
```

What is computed (`src/sslhist/ood.py`), per run:
- ID metrics on the whole validation set (`auroc`, `ece`, ...: for the SSL encoders identical to E1-E3, checked);
- temperature scaling: T fitted (NLL) on a stratified half of the validation set (seed `2000 + split`,
  the same for every method), every other E6 number on the other half, before (`_raw`) and after (`_ts`);
- confidence on ID and OOD images: mean MSP (`id_msp_*`, `ood_msp_*`), mean binary entropy in bits
  (`*_entropy_*`), share of OOD images with MSP ≥ 0.9 (`ood_conf90_*`), share predicted positive;
- OOD detection with score 1 − MSP, OOD = positive class, not flipped (`ood_auroc_raw` < 0.5: more
  confident on OOD than on ID), and FPR at 95% ID retained. A single temperature does not change the
  ranking, so these are the same after scaling; for a binary classifier MSP and entropy rank alike;
- feature-space control without the classifier: cosine distance to the 10th nearest ID training feature
  (`knn_ood_auroc`, `knn_fpr95`): does the representation separate the datasets even when the
  classifier is confident on both?
- checks for the saved encoders: the loaded encoder reproduces the saved features (`encoder_check_rel_diff`),
  and the saved probe its logits (`probe_check_max_abs_diff`); a unit stops if the encoder does not match.

Supervised baselines: backbone + linear head trained end to end on the labelled subset (AdamW lr 1e-3,
wd 1e-4, 10 epochs, batch 128, no augmentation, gradient clipping at 1). Unlike notebook 06, the AMP
gradients are unscaled before clipping (`supervised.unscale_before_clip: false` reproduces the notebook,
which clipped the scaled gradients), and every label fraction starts from the run seed.

Outputs in `results/lnbi/e6_ood/`: `raw/*.json` (one per run), `summary.txt` (three tables:
calibration, confidence, detection), `summary.csv`, `summary.tex` + `summary_detection.tex`,
`paired_vs_resnet18-sup_imagenet.csv` (every series minus the ImageNet supervised baseline, same
split/seed/labelled subset, Wilcoxon), and in `figures/`: label-efficiency curves of the OOD metrics,
`*_confidence_bars.pdf` (ID vs OOD MSP before/after scaling, 10% labels) and `*_msp_hist.pdf`
(MSP distributions, from `artifacts/e6_ood/scores/`).

## Outputs

`results/lnbi/<experiment>/`
- `raw/<dataset>__<method>__<backbone>__split<s>__seed<k>__frac<pp>.json`: one file per run with
  split, seed, method, backbone, label fraction, AUROC, accuracy, F1, ECE, Brier, plus
  hyperparameters, split fingerprints, git commit, GPU and timing
- `all_runs.csv`, `summary.csv` (mean, std with ddof=1, n), `summary.tex`, `summary.txt` (readable table, also printed by `aggregate.py`)
- `paired_vs_<ref>.csv`: per-setting mean ± std of the paired difference vs a reference
  backbone (same split, seed and labelled subset) and Wilcoxon signed-rank p-value
- `figures/<experiment>_<dataset>_<metric>.pdf`, in the style of the camera-ready figures

`artifacts/<experiment>/`: `encoders/<unit>.pt` (backbone state dict),
`features/<unit>.npz` (fp16 features of the full train part and of val, the labelled
subset indices, val logits and probe weights per fraction), `logs/`.

## Protocol (conference paper, unchanged)

| | ResNet family | ViT family |
|---|---|---|
| image size PCam / PANDA | 96 / 224 | 224 / 224 |
| SSL batch / probe-feature batch | 128 / 128 | 16 / 32 |
| SimCLR head (hidden → proj) | 2048 → 128 | 1024 → 128 |
| BYOL head (hidden → proj), EMA | 4096 → 256, 0.996 | 2048 → 256, 0.996 |

Common: 20% universe per dataset; 3 splits (80/20 stratified, `random_state=1000+split`);
3 seeds; SSL on the train part of the split only, 10 epochs, Adam lr 1e-3, fp16 AMP,
NT-Xent temperature 0.5; labelled subsets stratified with seed `42+seed+10*split`;
frozen features standardized with train-subset statistics; linear probe 10 epochs,
AdamW lr 1e-4, wd 1e-4, batch 256; metrics at threshold 0.5, ECE with 15 bins.

Differences from the notebooks (none changes the protocol):
- explicit data paths instead of searching `/kaggle/input`;
- the ViT SSL batch is fixed at 16 (the notebooks silently fell back to smaller batches on OOM);
- the run seed is set again before the probe phase, so probes are reproducible from a saved encoder;
- `num_workers` is configurable (8 instead of 4);
- encoders, features and probe outputs are saved for E5/E6.
