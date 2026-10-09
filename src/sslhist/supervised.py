"""Supervised baselines for E6: a backbone + linear head trained end to end on the
labelled subset only, as ``train_supervised_resnet18`` in notebook 06 (AdamW,
BCE, fp16 AMP, gradient clipping at norm 1, no augmentation).

One documented difference: with AMP the notebook clipped the *scaled* gradients
(``clip_grad_norm_`` before ``scaler.step``), which shrinks every update by the loss
scale. Here the gradients are unscaled first, as in the PyTorch AMP recipe;
``unscale_before_clip: false`` in the config reproduces the notebook.
"""
import time
from typing import Callable, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from .models import build_backbone
from .probe import sanitize_tensor


class SupervisedNet(nn.Module):
    def __init__(self, backbone: nn.Module, feat_dim: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(feat_dim, 1)

    def forward(self, x):
        return self.head(self.backbone(x)).squeeze(1)


def train_supervised(backbone_name: str, pretrained: bool, train_ds: Dataset, epochs: int, lr: float,
                     weight_decay: float, batch_size: int, num_workers: int, device: torch.device, amp: bool,
                     unscale_before_clip: bool = True, log: Callable[[str], None] = print) -> SupervisedNet:
    backbone, feat_dim = build_backbone(backbone_name, pretrained=pretrained)
    model = SupervisedNet(backbone, feat_dim).to(device)
    dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                    pin_memory=device.type == "cuda")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    for ep in range(1, epochs + 1):
        model.train()
        t0, total, n = time.time(), 0.0, 0
        for xb, yb in dl:
            xb, yb = xb.to(device, non_blocking=True), yb.float().to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                loss = crit(model(xb), yb)
            if not torch.isfinite(loss):
                continue
            scaler.scale(loss).backward()
            if unscale_before_clip:
                scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            total += float(loss.item()) * xb.size(0)
            n += xb.size(0)
        log(f"[SUP] ep {ep}/{epochs} loss={total / max(1, n):.4f} time={time.time() - t0:.1f}s")
    return model


@torch.no_grad()
def features_and_logits(model: SupervisedNet, dl: DataLoader, device: torch.device, amp: bool
                        ) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
    """Penultimate features and logits in one pass (logits as in the notebook evaluation)."""
    model.eval()
    feats, logits, ys = [], [], []
    for x, y in dl:
        x = x.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", enabled=amp):
            f = model.backbone(x)
            z = model.head(f).squeeze(1)
        feats.append(sanitize_tensor(f).cpu())
        logits.append(sanitize_tensor(z).cpu())
        ys.append(y.numpy())
    return (torch.cat(feats), torch.cat(logits).numpy(), np.concatenate(ys).astype(int))
