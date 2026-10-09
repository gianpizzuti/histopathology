"""Frozen-feature extraction and linear probe (legacy protocol).

Standardization statistics are fitted on the labelled TRAIN subset only and
applied unchanged to validation (and to OOD data in E6).
"""
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .metrics import sanitize_np_probs


def sanitize_tensor(x: torch.Tensor, clamp: float = 1e4) -> torch.Tensor:
    x = x.float()
    x = torch.nan_to_num(x, nan=0.0, posinf=clamp, neginf=-clamp)
    return torch.clamp(x, -clamp, clamp)


@torch.no_grad()
def extract_features(backbone: nn.Module, dl: DataLoader, device: torch.device, amp: bool
                     ) -> Tuple[torch.Tensor, np.ndarray]:
    backbone.eval()
    backbone = backbone.to(device)
    feats, ys = [], []
    for x, y in dl:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp):
            z = backbone(x)
        if isinstance(z, (tuple, list)):
            z = z[0]
        if z.ndim > 2:
            z = torch.flatten(z, 1)
        feats.append(sanitize_tensor(z).detach().cpu())
        ys.append(y.numpy())
    return sanitize_tensor(torch.cat(feats, dim=0)), np.concatenate(ys, axis=0).astype(int)


@torch.no_grad()
def fit_standardizer(train_feats: torch.Tensor, eps: float = 1e-6) -> Tuple[torch.Tensor, torch.Tensor]:
    x = train_feats.float()
    return x.mean(dim=0, keepdim=True), x.std(dim=0, keepdim=True).clamp_min(eps)


@torch.no_grad()
def apply_standardizer(feats: torch.Tensor, mu: torch.Tensor, sd: torch.Tensor) -> torch.Tensor:
    return (feats.float() - mu) / sd


class LinearProbe(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.fc(x).squeeze(1)


def train_probe(feats_std: torch.Tensor, labels: np.ndarray, epochs: int, lr: float,
                device: torch.device, batch_size: int = 256) -> LinearProbe:
    X = sanitize_tensor(feats_std).float()
    y = torch.tensor(labels, dtype=torch.float32)
    dl = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)
    clf = LinearProbe(X.size(1)).to(device)
    opt = torch.optim.AdamW(clf.parameters(), lr=lr, weight_decay=1e-4)
    crit = nn.BCEWithLogitsLoss()
    clf.train()
    for _ in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            loss = crit(sanitize_tensor(clf(xb)), yb)
            if not torch.isfinite(loss):
                continue
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(clf.parameters(), 1.0)
            opt.step()
    return clf


@torch.no_grad()
def predict_logits(clf: LinearProbe, feats_std: torch.Tensor, device: torch.device) -> np.ndarray:
    clf.eval()
    X = sanitize_tensor(feats_std).float().to(device)
    return sanitize_tensor(clf(X).detach().cpu()).numpy()


def logits_to_probs(logits: np.ndarray) -> np.ndarray:
    return sanitize_np_probs(torch.sigmoid(torch.as_tensor(logits)).numpy())
