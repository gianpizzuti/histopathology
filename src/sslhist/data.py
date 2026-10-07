"""Datasets, loaders, splits and label subsets.

The loaders reproduce the legacy Kaggle notebooks exactly, but take explicit
paths instead of searching ``/kaggle/input``. With the same input files, the
universe (``df.sample(frac, random_state=42)``), the train/val splits and the
labelled subsets are identical to the conference runs, for every backbone.
"""
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset
from torchvision import transforms

UNIVERSE_SEED = 42


# -----------------------
# Seed conventions (legacy)
# -----------------------
def run_seed(split_id: int, seed: int) -> int:
    """Global seed set before SSL pretraining of one (split, seed) run."""
    return 10_000 + 100 * split_id + seed


def subset_seed(split_id: int, seed: int) -> int:
    """Seed of the stratified labelled subset of one (split, seed) run."""
    return 42 + seed + 10 * split_id


# -----------------------
# Transforms (legacy)
# -----------------------
def ssl_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def sup_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


class TwoViewDataset(Dataset):
    def __init__(self, image_paths: List[str], img_size: int):
        self.image_paths = image_paths
        self.t = ssl_transform(img_size)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.t(img), self.t(img)


class LabeledDataset(Dataset):
    def __init__(self, image_paths: List[str], labels: List[int], img_size: int):
        assert len(image_paths) == len(labels)
        self.image_paths = image_paths
        self.labels = labels
        self.t = sup_transform(img_size)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.t(img), int(self.labels[idx])


# -----------------------
# Data sources
# -----------------------
@dataclass
class DataSource:
    name: str
    meta: pd.DataFrame  # one row per image: id, image_path, label (+ dataset-specific columns)

    def __len__(self) -> int:
        return len(self.meta)

    @property
    def paths(self) -> List[str]:
        return self.meta["image_path"].tolist()

    @property
    def ids(self) -> List[str]:
        return self.meta["id"].astype(str).tolist()

    def labels_all(self) -> np.ndarray:
        return self.meta["label"].to_numpy(dtype=np.int64)

    def make_ssl_dataset(self, indices: np.ndarray, img_size: int) -> Dataset:
        paths = self.paths
        return TwoViewDataset([paths[i] for i in indices], img_size)

    def make_sup_dataset(self, indices: np.ndarray, img_size: int) -> Dataset:
        paths, labels = self.paths, self.labels_all()
        return LabeledDataset([paths[i] for i in indices], [int(labels[i]) for i in indices], img_size)


def _subsample(df: pd.DataFrame, sample_frac: float) -> pd.DataFrame:
    if 0 < sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=UNIVERSE_SEED).reset_index(drop=True)
    return df


def load_pcam(root: str, sample_frac: float) -> DataSource:
    """PCam from the Kaggle 'histopathologic-cancer-detection' layout:
    ``<root>/train_labels.csv`` and ``<root>/train/<id>.tif``."""
    df = _subsample(pd.read_csv(os.path.join(root, "train_labels.csv")), sample_frac)

    def make_path(img_id: str) -> str:
        tif = os.path.join(root, "train", f"{img_id}.tif")
        if os.path.exists(tif):
            return tif
        for ext in ["png", "jpg", "jpeg"]:
            p = os.path.join(root, "train", f"{img_id}.{ext}")
            if os.path.exists(p):
                return p
        return tif

    df["image_path"] = df["id"].astype(str).apply(make_path)
    df = df[df["image_path"].apply(os.path.exists)].reset_index(drop=True)
    df["label"] = df["label"].astype(int)
    return DataSource("pcam", df[["id", "image_path", "label"]].copy())


def load_panda(train_csv: str, images_dir: str, sample_frac: float) -> DataSource:
    """PANDA: competition ``train.csv`` + one resized PNG per slide in ``images_dir``.
    Binary label: ISUP grade >= 2."""
    cols = ["image_id", "isup_grade"]
    raw = pd.read_csv(train_csv)
    extra = [c for c in ["data_provider", "gleason_score"] if c in raw.columns]
    # dropna only on the legacy columns, so the universe matches the conference runs
    df = raw[cols + extra].dropna(subset=cols)
    df = _subsample(df, sample_frac)
    df["image_path"] = df["image_id"].astype(str).apply(lambda x: os.path.join(images_dir, f"{x}.png"))
    df = df[df["image_path"].apply(os.path.exists)].reset_index(drop=True)
    df["label"] = (df["isup_grade"].astype(int) >= 2).astype(int)
    df = df.rename(columns={"image_id": "id"})
    return DataSource("panda", df[["id", "image_path", "label", "isup_grade"] + extra].copy())


def load_datasource(name: str, data_cfg: dict) -> DataSource:
    frac = float(data_cfg["sample_frac"][name])
    if name == "pcam":
        return load_pcam(data_cfg["pcam_root"], frac)
    if name == "panda":
        return load_panda(data_cfg["panda_train_csv"], data_cfg["panda_images_dir"], frac)
    raise ValueError(f"Unknown dataset: {name}")


# -----------------------
# Splits + stratified label subsets (legacy)
# -----------------------
def make_split(labels: np.ndarray, split_id: int, val_ratio: float, base_seed: int,
               universe: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
    """Train/val split ``split_id``. Each split is drawn independently with
    ``random_state = base_seed + split_id``, as in ``make_splits_indices``."""
    labels = np.asarray(labels, dtype=np.int64)
    if universe is None:
        universe = np.arange(len(labels), dtype=np.int64)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_ratio, random_state=base_seed + split_id)
    tr_rel, va_rel = next(sss.split(np.zeros(len(universe)), labels[universe]))
    return universe[tr_rel], universe[va_rel]


def label_subset(train_indices: np.ndarray, labels: np.ndarray, frac: float, seed: int) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.int64)
    n = len(train_indices)
    k = max(1, int(n * frac))
    sss = StratifiedShuffleSplit(n_splits=1, train_size=k, random_state=seed)
    sel_rel, _ = next(sss.split(np.zeros(n), labels[train_indices]))
    return train_indices[sel_rel]
