# SSL reliability in histopathology: LNBI extension

Code for the journal version (LNBI) of the conference paper. The conference
notebooks are kept unchanged in `notebooks/legacy/`. New experiments use the
`sslhist` package, which reproduces their protocol exactly (checked by
`tests/test_protocol.py` against verbatim copies of the notebook functions).

```
configs/            base.yaml (conference protocol) + one YAML per experiment
src/sslhist/        data, models, ssl (SimCLR/BYOL), probe, metrics, runner, report, plotting
scripts/            run_unit.py · launch.py · aggregate.py · make_splits.py
tests/              protocol equivalence with the notebooks + CPU smoke tests
results/lnbi/       committed: per-run JSON, summary CSV/LaTeX, PDF figures
artifacts/          NOT committed: encoders, features, logs
notebooks/legacy/   conference notebooks (Kaggle)
```

## Setup (GPU server)

The server driver is CUDA 12.4: install a matching PyTorch build **before** the package
(the default PyPI wheel may target a newer CUDA).

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[dev]"
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.device_count())"
pytest -q            # ~1 min on CPU
```

## Data

Use **the same Kaggle datasets as the conference runs**: the universe
(`df.sample(frac=0.2, random_state=42)`) and therefore every split depend on
the rows of the CSV files and on which PANDA PNGs exist.

```bash
pip install kaggle   # needs ~/.kaggle/kaggle.json and the competition rules accepted
kaggle competitions download -c histopathologic-cancer-detection -p /data/histopathologic-cancer-detection
kaggle competitions download -c prostate-cancer-grade-assessment -f train.csv -p /data/prostate-cancer-grade-assessment
kaggle datasets download -d <resized-panda-dataset-used-in-the-paper> -p /data/panda-resized
# unzip the archives, then:
cp configs/paths.example.yaml configs/paths.yaml   # and edit the three paths
```

Check the data partition (and compare the printed fingerprints across machines):

```bash
python scripts/make_splits.py --config configs/e1_vit_matched.yaml --dataset pcam
python scripts/make_splits.py --config configs/e1_vit_matched.yaml --dataset panda
```

## Running an experiment (E1 as example)

```bash
# 1. timing test: 200 SSL steps, prints the estimated duration of a full unit
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 python scripts/run_unit.py --config configs/e1_vit_matched.yaml \
    --dataset pcam --method simclr --backbone vit_b_16 --split 0 --seed 0 --timing 200

# 2. whole grid on GPUs 2 and 3 (the ones we may use), 3 units per GPU (re-run the same command to resume).
#    --gpus takes the ids shown by nvidia-smi: only those GPUs are used.
python scripts/launch.py --config configs/e1_vit_matched.yaml --gpus 2,3 --per-gpu 3 --dry-run
nohup python scripts/launch.py --config configs/e1_vit_matched.yaml --gpus 2,3 --per-gpu 3 > e1_launch.log 2>&1 &

# 3. tables, paired comparison vs ResNet-18, figures
python scripts/aggregate.py --config configs/e1_vit_matched.yaml --reference resnet18
git add results/lnbi/e1_vit_matched && git commit -m "E1 results" && git push
```

A *unit* is one SSL pretraining (dataset, method, backbone, split, seed) followed
by the linear probes at 1/5/10% labels. Logs: `artifacts/<experiment>/logs/<unit>.log`.

## Outputs

`results/lnbi/<experiment>/`
- `raw/<dataset>__<method>__<backbone>__split<s>__seed<k>__frac<pp>.json`: one file per run with
  split, seed, method, backbone, label fraction, AUROC, accuracy, F1, ECE, Brier, plus
  hyperparameters, split fingerprints, git commit, GPU and timing
- `all_runs.csv`, `summary.csv` (mean, std with ddof=1, n), `summary.tex`
- `paired_vs_<ref>.csv`: per-setting mean ± std of the paired difference vs a reference
  backbone (same split, seed and labelled subset) and Wilcoxon signed-rank p-value
- `figures/*.pdf`

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
