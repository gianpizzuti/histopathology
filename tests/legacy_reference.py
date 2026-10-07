"""Functions copied verbatim from notebooks/legacy/01_resnet18_benchmark.ipynb
(the ViT notebooks 02-05 contain the same code). Only the dataset loaders are
adapted: the Kaggle path search is replaced by explicit paths. Used by the
tests to prove that the new package reproduces the conference protocol."""
import os
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit


def sanitize_np_probs(probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(probs).reshape(-1)
    probs = np.nan_to_num(probs, nan=0.5, posinf=1.0, neginf=0.0)
    probs = np.clip(probs, 1e-7, 1 - 1e-7)
    return probs


def compute_ece(probs: np.ndarray, labels: np.ndarray, n_bins: int = 15) -> float:
    probs = sanitize_np_probs(probs)
    labels = np.asarray(labels).reshape(-1).astype(int)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(probs)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (probs >= lo) & (probs < hi)
        if m.sum() == 0:
            continue
        bin_conf = probs[m].mean()
        bin_acc = labels[m].mean()
        ece += (m.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def compute_metrics_binary(probs: np.ndarray, labels: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    probs = sanitize_np_probs(probs)
    labels = np.asarray(labels).reshape(-1).astype(int)
    preds = (probs >= thr).astype(int)

    acc = float((preds == labels).mean())
    try:
        auc = float(roc_auc_score(labels, probs))
    except ValueError:
        auc = float("nan")
    f1 = float(f1_score(labels, preds, zero_division=0))
    brier = float(brier_score_loss(labels, probs))
    ece = compute_ece(probs, labels, n_bins=15)
    return {"acc": acc, "auc": auc, "f1": f1, "ece": ece, "brier": brier}


def make_splits_indices(universe_indices: np.ndarray, labels: np.ndarray, n_splits: int, val_ratio: float, base_seed: int):
    labels = np.asarray(labels, dtype=np.int64)
    splits = []
    for split_id in range(n_splits):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=base_seed + split_id)
        tr_rel, va_rel = next(sss.split(np.zeros(len(universe_indices)), labels[universe_indices]))
        splits.append((universe_indices[tr_rel], universe_indices[va_rel]))
    return splits


def stratified_label_subset_indices(train_indices: np.ndarray, labels: np.ndarray, frac: float, seed: int):
    labels = np.asarray(labels, dtype=np.int64)
    n = len(train_indices)
    k = max(1, int(n * frac))
    sss = StratifiedShuffleSplit(n_splits=1, train_size=k, random_state=seed)
    sel_rel, _ = next(sss.split(np.zeros(n), labels[train_indices]))
    return train_indices[sel_rel]


def load_pcam_paths_labels(pcam_root: str, sample_frac: float):
    """Body of legacy ``load_pcam_datasource`` (CSV branch), explicit root."""
    labels_path = os.path.join(pcam_root, "train_labels.csv")
    df = pd.read_csv(labels_path)
    if 0 < sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)

    def make_path(img_id: str):
        tif = os.path.join(pcam_root, "train", f"{img_id}.tif")
        if os.path.exists(tif):
            return tif
        for ext in ["png", "jpg", "jpeg"]:
            p = os.path.join(pcam_root, "train", f"{img_id}.{ext}")
            if os.path.exists(p):
                return p
        return tif

    df["image_path"] = df["id"].astype(str).apply(make_path)
    df = df[df["image_path"].apply(os.path.exists)].reset_index(drop=True)
    return df["image_path"].tolist(), df["label"].astype(int).tolist()


def load_panda_paths_labels(train_csv: str, resized_dir: str, sample_frac: float):
    """Body of legacy ``load_panda_datasource``, explicit paths."""
    df = pd.read_csv(train_csv)[["image_id", "isup_grade"]].dropna()
    if 0 < sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)

    df["image_path"] = df["image_id"].astype(str).apply(lambda x: os.path.join(resized_dir, f"{x}.png"))
    df = df[df["image_path"].apply(os.path.exists)].reset_index(drop=True)

    labels = [1 if int(g) >= 2 else 0 for g in df["isup_grade"].astype(int).tolist()]
    paths = df["image_path"].tolist()
    return paths, labels
