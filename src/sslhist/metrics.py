"""Classification and calibration metrics (identical to the legacy notebooks)."""
from typing import Dict

import numpy as np
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score


def sanitize_np_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs).reshape(-1)
    probs = np.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
    return np.clip(probs, 1e-7, 1 - 1e-7)


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    """Binary ECE on the positive-class probability, 15 equal-width bins."""
    probs = sanitize_np_probs(probs)
    labels = np.asarray(labels).reshape(-1).astype(int)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece, n = 0.0, len(probs)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (probs >= lo) & (probs < hi)
        if m.sum() == 0:
            continue
        ece += (m.sum() / n) * abs(labels[m].mean() - probs[m].mean())
    return float(ece)


def compute_metrics_binary(probs: np.ndarray, labels: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    probs = sanitize_np_probs(probs)
    labels = np.asarray(labels).reshape(-1).astype(int)
    preds = (probs >= thr).astype(int)
    try:
        auroc = float(roc_auc_score(labels, probs))
    except ValueError:
        auroc = float("nan")
    return {
        "auroc": auroc,
        "accuracy": float((preds == labels).mean()),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "ece": compute_ece(probs, labels),
        "brier": float(brier_score_loss(labels, probs)),
    }


METRICS = ["auroc", "accuracy", "f1", "ece", "brier"]
