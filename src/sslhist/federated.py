"""E4: federated SSL pretraining (FedAvg), simulated on one GPU.

Same encoder, SSL loss, augmentations, optimiser, AMP and batch size as the
centralised runs of E1, so that a federated unit differs from its E1 counterpart
(same split, seed and labelled subsets) only by the federation:

- the training part of the split is divided among K clients, either IID or with
  label skew: each client's label distribution is drawn from Dir(alpha * C * p), with
  p the overall class proportions (Hsu et al., 2019), and its images are sampled from
  it until the client holds n/K images (sampling as in Acar et al., 2021). All clients
  have the same size, so the skew is in the labels only; alpha = 0.1 gives nearly
  single-class clients. Labels are used only to build the partition, never for training;
- each round, every client starts from the global model, trains ``local_epochs``
  epochs on its images (Adam, new optimiser state each round), and the server
  averages all parameters and buffers, BatchNorm statistics included, weighted by
  client size (FedAvg, McMahan et al., 2017);
- rounds = ssl_epochs / local_epochs, so the data are seen as many times as in
  the centralised runs (10 epochs).

Federated methods are named ``<ssl method>-fed-k<clients>-<iid | a<alpha>>``,
e.g. ``simclr-fed-k10-a0.1``.
"""
import copy
import re
import time
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

from .ssl import build_ssl_model, ssl_backbone, ssl_loss

_FED_RE = re.compile(r"^(simclr|byol|barlow)-fed-k(\d+)-(iid|a(\d+(?:\.\d+)?))$")


def parse_fed_method(method: str) -> Optional[dict]:
    """``simclr-fed-k10-a0.1`` -> {"ssl": "simclr", "clients": 10, "partition": "dirichlet",
    "alpha": 0.1}; None for a centralised method."""
    m = _FED_RE.match(method)
    if not m:
        return None
    ssl, k, part, alpha = m.groups()
    return {"ssl": ssl, "clients": int(k), "partition": "iid" if part == "iid" else "dirichlet",
            "alpha": float(alpha) if alpha else None}


def fed_label(method: str) -> str:
    """Legend label, e.g. 'FED-SIMCLR (10 clients, Dir α=0.1)'."""
    f = parse_fed_method(method)
    part = "IID" if f["partition"] == "iid" else f"Dir α={f['alpha']:g}"
    return f"FED-{f['ssl'].upper()} ({f['clients']} clients, {part})"


# -----------------------
# Partitions
# -----------------------
def partition_clients(labels: np.ndarray, n_clients: int, partition: str, alpha: Optional[float],
                      seed: int) -> List[np.ndarray]:
    """Positions (into ``labels``) of the images of each client; disjoint, covering all
    images, client sizes differing by at most one."""
    labels = np.asarray(labels)
    n = len(labels)
    rng = np.random.RandomState(seed)
    sizes = np.full(n_clients, n // n_clients)
    sizes[: n % n_clients] += 1
    if partition == "iid":
        perm = rng.permutation(n)
        return [np.sort(p) for p in np.split(perm, np.cumsum(sizes)[:-1])]
    if partition != "dirichlet":
        raise ValueError(f"Unknown partition: {partition}")

    classes, counts = np.unique(labels, return_counts=True)
    # Label distribution of each client ~ Dir(alpha * C * p), p = overall class proportions
    # (Hsu et al., 2019; equals Dir(alpha, ..., alpha) for balanced classes). Centring the
    # prior on p keeps the clients' demand close to what is available, so the classes are
    # not used up early, which would undo the skew with two unbalanced classes.
    priors = rng.dirichlet(alpha * len(classes) * counts / counts.sum(), size=n_clients)
    pools = [list(rng.permutation(np.where(labels == c)[0])) for c in classes]
    clients: List[List[int]] = [[] for _ in range(n_clients)]
    open_clients = [k for k in range(n_clients) if sizes[k] > 0]
    while open_clients:
        k = open_clients[rng.randint(len(open_clients))]
        c = rng.choice(len(classes), p=priors[k])
        clients[k].append(pools[c].pop())
        if not pools[c] and any(pools):  # class used up: the clients sample from the remaining classes
            priors[:, c] = 0.0
            empty = priors.sum(axis=1) == 0
            priors[empty] = [len(p) > 0 for p in pools]
            priors /= priors.sum(axis=1, keepdims=True)
        if len(clients[k]) == sizes[k]:
            open_clients.remove(k)
    return [np.sort(np.asarray(c, dtype=np.int64)) for c in clients]


def partition_stats(labels: np.ndarray, parts: List[np.ndarray]) -> dict:
    labels = np.asarray(labels)
    pos = [float(labels[p].mean()) if len(p) else float("nan") for p in parts]
    return {"client_sizes": [int(len(p)) for p in parts], "client_pos_rates": [round(r, 4) for r in pos],
            # mean absolute difference between a client's positive rate and the overall one
            "label_skew": float(np.nanmean(np.abs(np.asarray(pos) - labels.mean())))}


# -----------------------
# FedAvg
# -----------------------
def fedavg(states: List[Dict[str, torch.Tensor]], weights: List[float]) -> Dict[str, torch.Tensor]:
    """Weighted average of floating-point entries; integer buffers (BatchNorm
    ``num_batches_tracked``) are taken from the first client."""
    w = [float(x) / float(sum(weights)) for x in weights]
    out = {}
    for k, v in states[0].items():
        if v.is_floating_point():
            out[k] = sum(wi * s[k].double() for wi, s in zip(w, states)).to(v.dtype)
        else:
            out[k] = v.clone()
    return out


def pretrain_federated(
    method: str,
    backbone_name: str,
    ssl_ds: Dataset,
    labels: np.ndarray,
    head_cfg: dict,
    batch_size: int,
    epochs: int,
    local_epochs: int,
    lr: float,
    num_workers: int,
    device: torch.device,
    amp: bool,
    partition_seed: int,
    max_steps: Optional[int] = None,
    log: Callable[[str], None] = print,
) -> Tuple[nn.Module, dict]:
    """Federated counterpart of ssl.pretrain_ssl; same return value. ``labels`` (of the
    images of ``ssl_ds``, in order) only define the label-skewed partition."""
    fed = parse_fed_method(method)
    if fed is None:
        raise ValueError(f"Not a federated method: {method}")
    if epochs % local_epochs:
        raise ValueError(f"ssl_epochs ({epochs}) must be a multiple of local_epochs ({local_epochs})")
    ssl, k = fed["ssl"], fed["clients"]
    rounds = epochs // local_epochs

    # Built first, right after the run seed: same initial weights as the centralised run.
    global_model = build_ssl_model(ssl, backbone_name, head_cfg).to(device)
    parts = partition_clients(labels, k, fed["partition"], fed["alpha"], partition_seed)
    stats = partition_stats(labels, parts)
    log(f"[FED] {k} clients, {fed['partition']}" + (f" alpha={fed['alpha']:g}" if fed["alpha"] else "") +
        f", {rounds} rounds x {local_epochs} local epoch(s); sizes={stats['client_sizes']} "
        f"positive rates={stats['client_pos_rates']} label skew={stats['label_skew']:.3f}")

    local_model = copy.deepcopy(global_model)
    history = {"epoch_loss": [], "epoch_time_s": [], "skipped_batches": 0, "steps": 0,
               "federated": {"n_clients": k, "partition": fed["partition"], "alpha": fed["alpha"],
                             "rounds": rounds, "local_epochs": local_epochs, **stats}}
    stop = False
    for rnd in range(1, rounds + 1):
        t0 = time.time()
        states, weights, total, n = [], [], 0.0, 0
        for part in parts:
            local_model.load_state_dict(global_model.state_dict())
            local_model.train()
            opt = torch.optim.Adam(local_model.parameters(), lr=lr)
            scaler = torch.amp.GradScaler("cuda", enabled=amp)
            dl = DataLoader(Subset(ssl_ds, part.tolist()), batch_size=batch_size, shuffle=True,
                            num_workers=num_workers, pin_memory=(device.type == "cuda"), drop_last=True)
            for _ in range(local_epochs):
                for x1, x2 in dl:
                    x1, x2 = x1.to(device, non_blocking=True), x2.to(device, non_blocking=True)
                    opt.zero_grad(set_to_none=True)
                    loss = ssl_loss(ssl, local_model, x1, x2, amp, head_cfg)
                    if not torch.isfinite(loss):
                        history["skipped_batches"] += 1
                        continue
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                    if ssl == "byol":
                        local_model.update_target()
                    total += float(loss.item()) * x1.size(0)
                    n += x1.size(0)
                    history["steps"] += 1
                    if max_steps is not None and history["steps"] >= max_steps:
                        stop = True
                        break
                if stop:
                    break
            states.append({key: v.detach().clone() for key, v in local_model.state_dict().items()})
            weights.append(len(part))
            if stop:
                break
        global_model.load_state_dict(fedavg(states, weights))
        dt = time.time() - t0
        history["epoch_loss"].append(total / max(1, n))
        history["epoch_time_s"].append(dt)
        # "ep r/R" (one round = one pass over all clients) keeps scripts/status.py working
        log(f"[{method}/{backbone_name}] ep {rnd}/{rounds} loss={total / max(1, n):.4f} time={dt:.1f}s "
            f"({n / max(dt, 1e-9):.0f} samples/s)")
        if stop:
            break
    history["sec_per_step"] = sum(history["epoch_time_s"]) / max(1, history["steps"])
    return ssl_backbone(ssl, global_model), history
