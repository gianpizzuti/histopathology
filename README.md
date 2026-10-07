# SSL reliability in histopathology: LNBI extension

Code for the journal version (LNBI) of the conference paper. The conference
notebooks are kept unchanged in `notebooks/legacy/`. New experiments use the
`sslhist` package, which reproduces their protocol exactly (checked by
`tests/test_protocol.py` against verbatim copies of the notebook functions).

```
configs/            base.yaml (conference protocol) + one YAML per experiment
src/sslhist/        data, models, ssl (SimCLR/BYOL), probe, metrics, runner, report, plotting
scripts/            check_setup.py · run_unit.py · launch.py · aggregate.py · make_splits.py
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
pytest -q            # 17 passed, ~2 min; the tests never use a GPU
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

# 3. tables, paired comparison vs ResNet-18, figures
python scripts/aggregate.py --config configs/e1_vit_matched.yaml --reference resnet18
git add results/lnbi/e1_vit_matched && git commit -m "E1 results" && git push
```

A *unit* is one SSL pretraining (dataset, method, backbone, split, seed) followed
by the linear probes at 1/5/10% labels. Logs: `artifacts/<experiment>/logs/<unit>.log`.

Single unit by hand (e.g. to debug), or a timing test of its first 200 SSL steps:

```bash
python scripts/run_unit.py --gpu 2 --config configs/e1_vit_matched.yaml \
    --dataset pcam --method simclr --backbone vit_b_16 --split 0 --seed 0 --timing 200
```

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
