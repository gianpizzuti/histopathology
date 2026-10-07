"""SimCLR and BYOL pretraining, ported from the legacy notebooks.

The training loops keep the legacy details: Adam without weight decay, fp16
autocast + GradScaler, NT-Xent computed in fp32 outside autocast (SimCLR),
BYOL loss inside autocast, EMA update of target encoder + projector after
every step, non-finite batches skipped, ``drop_last=True``.
"""
import copy
import time
from typing import Callable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .models import build_backbone


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, proj_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim),
        )

    def forward(self, x):
        return self.net(x)


def build_simclr(backbone_name: str, proj_dim: int, hidden_dim: int) -> nn.Sequential:
    backbone, feat_dim = build_backbone(backbone_name, pretrained=False)
    return nn.Sequential(backbone, ProjectionHead(feat_dim, proj_dim, hidden_dim))


def nt_xent_loss_fp32(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    z1, z2 = z1.float(), z2.float()
    b = z1.size(0)
    z1, z2 = F.normalize(z1, dim=1), F.normalize(z2, dim=1)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.T) / float(temperature)
    sim = sim.masked_fill(torch.eye(2 * b, dtype=torch.bool, device=z.device), -1e4)
    labels = torch.arange(b, device=z.device)
    labels = torch.cat([labels + b, labels], dim=0)
    return F.cross_entropy(sim, labels)


class BYOL(nn.Module):
    def __init__(self, backbone_name: str, proj_dim: int, hidden_dim: int, moving_avg_decay: float = 0.996):
        super().__init__()
        backbone, feat_dim = build_backbone(backbone_name, pretrained=False)
        self.online_encoder = backbone
        self.online_proj = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim),
        )
        self.online_pred = nn.Sequential(
            nn.Linear(proj_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim),
        )
        self.target_encoder = copy.deepcopy(self.online_encoder)
        self.target_proj = copy.deepcopy(self.online_proj)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_proj.parameters():
            p.requires_grad = False
        self.m = moving_avg_decay

    @torch.no_grad()
    def update_target(self):
        for op, tp in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            tp.data = self.m * tp.data + (1 - self.m) * op.data
        for op, tp in zip(self.online_proj.parameters(), self.target_proj.parameters()):
            tp.data = self.m * tp.data + (1 - self.m) * op.data

    def forward(self, x1, x2):
        z1 = self.online_proj(self.online_encoder(x1))
        z2 = self.online_proj(self.online_encoder(x2))
        p1, p2 = self.online_pred(z1), self.online_pred(z2)
        with torch.no_grad():
            t1 = self.target_proj(self.target_encoder(x1))
            t2 = self.target_proj(self.target_encoder(x2))
        return p1, p2, t1.detach(), t2.detach()

    @staticmethod
    def loss_fn(p, z):
        p = F.normalize(p.float(), dim=1)
        z = F.normalize(z.float(), dim=1)
        return 2 - 2 * (p * z).sum(dim=1).mean()


def pretrain_ssl(
    method: str,
    backbone_name: str,
    ssl_ds: Dataset,
    head_cfg: dict,
    batch_size: int,
    epochs: int,
    lr: float,
    num_workers: int,
    device: torch.device,
    amp: bool,
    max_steps: Optional[int] = None,
    log: Callable[[str], None] = print,
) -> Tuple[nn.Module, dict]:
    """Pretrain one encoder. Returns ``(backbone, history)``.

    The model is built here, after the caller has set the run seed, matching
    the legacy order (seed -> model init -> optimizer -> DataLoader).
    ``max_steps`` stops early (timing runs only).
    """
    method = method.lower()
    if method == "simclr":
        model = build_simclr(backbone_name, head_cfg["proj_dim"], head_cfg["hidden_dim"])
    elif method == "byol":
        model = BYOL(backbone_name, head_cfg["proj_dim"], head_cfg["hidden_dim"], head_cfg["ema"])
    else:
        raise ValueError(f"Unknown SSL method: {method}")

    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    dl = DataLoader(ssl_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                    pin_memory=(device.type == "cuda"), drop_last=True)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    history = {"epoch_loss": [], "epoch_time_s": [], "skipped_batches": 0, "steps": 0}
    warmup_steps, t_warm, t_last = 10, None, None  # steady-state speed, without start-up costs
    model.train()
    for ep in range(1, epochs + 1):
        t0 = time.time()
        total, n = 0.0, 0
        for x1, x2 in dl:
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            if method == "simclr":
                with torch.amp.autocast("cuda", enabled=amp):
                    z1, z2 = model(x1), model(x2)
                loss = nt_xent_loss_fp32(z1, z2, temperature=0.5)
            else:
                with torch.amp.autocast("cuda", enabled=amp):
                    p1, p2, t1, t2 = model(x1, x2)
                    loss = BYOL.loss_fn(p1, t2) + BYOL.loss_fn(p2, t1)
            if not torch.isfinite(loss):
                history["skipped_batches"] += 1
                continue
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            if method == "byol":
                model.update_target()
            total += float(loss.item()) * x1.size(0)
            n += x1.size(0)
            history["steps"] += 1
            t_last = time.time()  # loss.item() above synchronizes with the GPU
            if history["steps"] == warmup_steps:
                t_warm = t_last
            if max_steps is not None and history["steps"] >= max_steps:
                break
        dt = time.time() - t0
        history["epoch_loss"].append(total / max(1, n))
        history["epoch_time_s"].append(dt)
        log(f"[{method}/{backbone_name}] ep {ep}/{epochs} loss={total / max(1, n):.4f} "
            f"time={dt:.1f}s ({n / max(dt, 1e-9):.0f} samples/s)")
        if max_steps is not None and history["steps"] >= max_steps:
            break

    if t_warm is not None and history["steps"] > warmup_steps:
        history["sec_per_step"] = (t_last - t_warm) / (history["steps"] - warmup_steps)
    else:
        history["sec_per_step"] = sum(history["epoch_time_s"]) / max(1, history["steps"])
    backbone = model[0] if method == "simclr" else model.online_encoder
    return backbone, history
