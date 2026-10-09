"""Control for E4: the evaluation pipeline of notebook 08 (federated diagnostic of the
conference paper, Table 3) applied to saved encoders, next to ours.

In the paper the federated rows came from notebook 08 and the centralised rows from
the main benchmark, whose evaluations differ: notebook 08 trains the probe on raw
(not standardised) features with Adam lr 1e-3, keeps the epoch with the lowest
validation loss (early stopping) and reports the ECE of the predicted class,
max(p, 1-p) (``compute_ece_toplabel``); the benchmark standardises the features,
trains 10 epochs with AdamW lr 1e-4 and reports the ECE of the positive-class
probability (``compute_ece``). Scoring the same encoders both ways shows how much
of the conference federated-vs-centralised difference comes from the evaluation.

Both evaluations use the same labelled subsets and the same images: the validation
part of the split is halved as in E6 (seed 2000 + split); notebook 08 stops early on
the first half, and every number is computed on the second half.
"""
import glob
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .config import resolve
from .io import frac_tag
from .metrics import compute_ece, compute_ece_toplabel, sanitize_np_probs
from .ood import calibration_split
from .probe import logits_to_probs
from .utils import fingerprint

def legacy08_probe(tr_x: np.ndarray, tr_y: np.ndarray, va_x: np.ndarray, va_y: np.ndarray, te_x: np.ndarray,
                   seed: int, epochs: int = 10, lr: float = 1e-3, batch: int = 256) -> np.ndarray:
    """Probabilities on ``te_x`` from the linear probe of notebook 08 (``eval_linear_probe``)."""
    torch.manual_seed(seed)
    x, y = torch.tensor(tr_x, dtype=torch.float32), torch.tensor(tr_y, dtype=torch.float32)
    xv, yv = torch.tensor(va_x, dtype=torch.float32), torch.tensor(va_y, dtype=torch.float32)
    probe = nn.Linear(x.size(1), 1)
    opt = torch.optim.Adam(probe.parameters(), lr=lr)
    bce = nn.BCEWithLogitsLoss()
    best_val, best_state = float("inf"), None
    for _ in range(epochs):
        perm = torch.randperm(x.size(0))
        for i in range(0, x.size(0), batch):
            idx = perm[i:i + batch]
            loss = bce(probe(x[idx]).squeeze(1), y[idx])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        with torch.no_grad():
            val_loss = float(bce(probe(xv).squeeze(1), yv))
        if val_loss < best_val:
            best_val, best_state = val_loss, {k: v.clone() for k, v in probe.state_dict().items()}
    probe.load_state_dict(best_state)
    with torch.no_grad():
        logits = probe(torch.tensor(te_x, dtype=torch.float32)).squeeze(1).numpy()
    return 1 / (1 + np.exp(-logits))


def _scores(probs: np.ndarray, y: np.ndarray) -> dict:
    from sklearn.metrics import roc_auc_score
    p = sanitize_np_probs(probs)
    return {"auroc": float(roc_auc_score(y, p)), "ece": compute_ece(p, y), "ece_toplabel": compute_ece_toplabel(p, y)}


def run_control(cfg: dict, sources: List[str], dataset: str, log=print) -> List[dict]:
    """``sources``: experiment names, optionally restricted to one series as
    ``experiment:backbone:method``. Reads artifacts/<experiment>/features/*.npz."""
    rows = []
    for src in sources:
        exp, _, series = src.partition(":")
        backbone, _, method = series.partition(":")
        for path in sorted(glob.glob(str(resolve(cfg["artifacts_dir"]) / exp / "features" / f"{dataset}__*.npz"))):
            ds_, meth, bb, sp, sd = Path(path).stem.split("__")  # Unit.tag
            if (backbone and bb != backbone) or (method and meth != method):
                continue
            f = np.load(path)
            split, seed = int(sp[len("split"):]), int(sd[len("seed"):])
            val_y = f["val_y"]
            cal, ev = calibration_split(val_y, split, 0.5, 2000)
            pos_of = {int(v): i for i, v in enumerate(f["train_idx"])}
            tr_f = f["train_feats"].astype(np.float32)
            va_f = f["val_feats"].astype(np.float32)
            for frac in cfg["protocol"]["label_fracs"]:
                ft = frac_tag(frac)
                pos = np.array([pos_of[int(v)] for v in f[f"{ft}_subset_idx"]])
                ours = _scores(logits_to_probs(f[f"{ft}_val_logits"])[ev], val_y[ev])
                nb08 = _scores(legacy08_probe(tr_f[pos], f["train_y"][pos], va_f[cal], val_y[cal], va_f[ev],
                                              seed=10_000 + 100 * split + seed), val_y[ev])
                rows.append({"experiment": exp, "dataset": dataset, "method": meth, "backbone": bb,
                             "split": split, "seed": seed, "label_frac": frac, "n_labeled": int(len(pos)),
                             "n_eval": int(len(ev)), "subset_fp": fingerprint(f[f"{ft}_subset_idx"]),
                             **{f"ours_{k}": v for k, v in ours.items()},
                             **{f"nb08_{k}": v for k, v in nb08.items()}})
            log(f"  [CONTROL] {exp}/{Path(path).stem} at {frac * 100:g}% labels: ECE ours={ours['ece']:.3f} "
                f"(top-label {ours['ece_toplabel']:.3f}) | notebook 08 ECE={nb08['ece']:.3f} "
                f"(top-label {nb08['ece_toplabel']:.3f})")
    return rows


CONTROL_METRICS = ["ours_auroc", "ours_ece", "ours_ece_toplabel", "nb08_auroc", "nb08_ece", "nb08_ece_toplabel"]


def summarize_control(rows: List[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    g = df.groupby(["experiment", "dataset", "backbone", "method", "label_frac"])
    out = g[CONTROL_METRICS].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out["n_runs"] = g.size()
    return out.reset_index()
