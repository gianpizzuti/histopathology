"""One unit = SSL pretraining for one (dataset, method, backbone, split, seed),
followed by a linear probe for every label fraction.

Follows ``run_config_resnet18`` and the ViT notebooks, with one documented
difference: the global seed is set again (to the same run seed) right before
the probe phase. This makes the probes reproducible from a saved encoder,
independently of how many random numbers the pretraining consumed.
"""
import time
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from . import data as D
from .config import family_cfg
from .io import Unit, exp_artifacts_dir, frac_tag, result_path, unit_complete, write_json
from .metrics import compute_metrics_binary
from .probe import (apply_standardizer, extract_features, fit_standardizer, logits_to_probs,
                    predict_logits, train_probe)
from .ssl import pretrain_ssl
from .utils import fingerprint, get_device, run_metadata, set_seed


def log(msg: str) -> None:
    print(msg, flush=True)


def run_unit(cfg: dict, unit: Unit, force: bool = False, timing_steps: Optional[int] = None) -> Optional[dict]:
    """Run one unit and write its result files. With ``timing_steps`` only the
    first SSL steps are run and a time estimate is returned; nothing is written."""
    prot, rt = cfg["protocol"], cfg["runtime"]
    fam = family_cfg(cfg, unit.backbone)
    if timing_steps is None and not force and unit_complete(cfg, unit):
        log(f"[SKIP] {unit.tag} already complete")
        return None

    torch.set_num_threads(int(rt.get("torch_threads", 4)))
    device = get_device()
    amp = bool(prot["amp"]) and device.type == "cuda"
    nw = int(rt["num_workers"])
    pin = device.type == "cuda"

    ds = D.load_datasource(unit.dataset, cfg["data"])
    labels, ids = ds.labels_all(), np.asarray(ds.ids)
    tr_idx, va_idx = D.make_split(labels, unit.split, prot["val_ratio"], prot["split_base_seed"])
    img = int(fam["img_size"][unit.dataset])
    rs, ss = D.run_seed(unit.split, unit.seed), D.subset_seed(unit.split, unit.seed)
    log("=" * 90)
    log(f"[RUN] {unit.tag} img={img} n_universe={len(ds)} n_tr={len(tr_idx)} n_va={len(va_idx)} "
        f"device={device} amp={amp}")
    log("=" * 90)

    # ---- SSL pretraining (on the TRAIN part of this split only)
    set_seed(rs)
    ssl_ds = ds.make_ssl_dataset(tr_idx, img_size=img)
    t0 = time.time()
    backbone, hist = pretrain_ssl(
        unit.method, unit.backbone, ssl_ds, fam[unit.method],
        batch_size=int(fam["batch_ssl"]), epochs=int(prot["ssl_epochs"]), lr=float(prot["ssl_lr"]),
        num_workers=nw, device=device, amp=amp, max_steps=timing_steps, log=log,
    )
    ssl_time = time.time() - t0

    if timing_steps is not None:
        steps_per_epoch = len(ssl_ds) // int(fam["batch_ssl"])
        sec_per_step = ssl_time / max(1, hist["steps"])
        est = {
            "unit": unit.tag,
            "steps_measured": hist["steps"],
            "sec_per_step": sec_per_step,
            "est_epoch_min": steps_per_epoch * sec_per_step / 60,
            "est_ssl_total_min": steps_per_epoch * sec_per_step * int(prot["ssl_epochs"]) / 60,
            "peak_gpu_mem_gb": (torch.cuda.max_memory_allocated() / 1024 ** 3) if device.type == "cuda" else 0.0,
        }
        log(f"[TIMING] {est}")
        return est

    art = exp_artifacts_dir(cfg)
    if rt.get("save_encoder", True):
        (art / "encoders").mkdir(parents=True, exist_ok=True)
        torch.save({k: v.detach().cpu() for k, v in backbone.state_dict().items()},
                   art / "encoders" / f"{unit.tag}.pt")

    # ---- Linear probes
    set_seed(rs)
    val_dl = DataLoader(ds.make_sup_dataset(va_idx, img_size=img), batch_size=int(fam["batch_sup"]),
                        shuffle=False, num_workers=nw, pin_memory=pin)
    rows, probe_art = [], {}
    t1 = time.time()
    for frac in prot["label_fracs"]:
        sub_idx = D.label_subset(tr_idx, labels, frac, seed=ss)
        train_dl = DataLoader(ds.make_sup_dataset(sub_idx, img_size=img), batch_size=int(fam["batch_sup"]),
                              shuffle=True, num_workers=nw, pin_memory=pin)
        tr_f, tr_y = extract_features(backbone, train_dl, device, amp)
        va_f, va_y = extract_features(backbone, val_dl, device, amp)
        mu, sd = fit_standardizer(tr_f)  # TRAIN only
        clf = train_probe(apply_standardizer(tr_f, mu, sd), tr_y, int(prot["probe_epochs"]),
                          float(prot["probe_lr"]), device, batch_size=int(prot["probe_batch"]))
        val_logits = predict_logits(clf, apply_standardizer(va_f, mu, sd), device)
        metrics = compute_metrics_binary(logits_to_probs(val_logits), va_y)
        log(f"  [PROBE] frac={frac:.2f} n_lab={len(sub_idx)} " +
            " ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

        rows.append({
            "experiment": cfg["experiment"],
            "dataset": unit.dataset, "method": unit.method, "backbone": unit.backbone,
            "split": unit.split, "seed": unit.seed, "label_frac": frac,
            **metrics,
            "n_labeled": int(len(sub_idx)), "n_train": int(len(tr_idx)), "n_val": int(len(va_idx)),
            "img_size": img, "sample_frac": float(cfg["data"]["sample_frac"][unit.dataset]),
            "ssl_epochs": int(prot["ssl_epochs"]), "ssl_batch": int(fam["batch_ssl"]),
            "ssl_lr": float(prot["ssl_lr"]), "probe_epochs": int(prot["probe_epochs"]),
            "probe_lr": float(prot["probe_lr"]), "num_workers": nw,
            "run_seed": rs, "subset_seed": ss,
            "val_ids_fp": fingerprint(ids[va_idx]), "subset_ids_fp": fingerprint(ids[sub_idx]),
            "ssl_final_loss": hist["epoch_loss"][-1] if hist["epoch_loss"] else float("nan"),
            "ssl_skipped_batches": hist["skipped_batches"], "ssl_time_s": round(ssl_time, 1),
        })
        ft = frac_tag(frac)
        probe_art[f"{ft}_subset_idx"] = sub_idx
        probe_art[f"{ft}_val_logits"] = val_logits.astype(np.float32)
        probe_art[f"{ft}_probe_w"] = clf.fc.weight.detach().cpu().numpy().ravel()
        probe_art[f"{ft}_probe_b"] = clf.fc.bias.detach().cpu().numpy()
        probe_art[f"{ft}_mu"] = mu.numpy().ravel()
        probe_art[f"{ft}_sd"] = sd.numpy().ravel()
    probe_time = time.time() - t1

    # ---- Artifacts for later analyses (E5/E6): features of the full train part and val
    if rt.get("save_features", True):
        full_tr_dl = DataLoader(ds.make_sup_dataset(tr_idx, img_size=img), batch_size=int(fam["batch_sup"]),
                                shuffle=False, num_workers=nw, pin_memory=pin)
        tr_f, tr_y = extract_features(backbone, full_tr_dl, device, amp)
        va_f, va_y = extract_features(backbone, val_dl, device, amp)
        (art / "features").mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            art / "features" / f"{unit.tag}.npz",
            train_idx=tr_idx, train_feats=tr_f.numpy().astype(np.float16), train_y=tr_y,
            val_idx=va_idx, val_feats=va_f.numpy().astype(np.float16), val_y=va_y,
            **probe_art,
        )

    meta = run_metadata()
    write_json(art / "logs" / f"{unit.tag}.json", {
        "unit": unit.tag, "ssl_history": hist, "ssl_time_s": ssl_time, "probe_time_s": probe_time, **meta,
    })
    # Result files are written last: a unit counts as complete only once everything is saved.
    for row in rows:
        write_json(result_path(cfg, unit, row["label_frac"]), {**row, **meta})
    log(f"[DONE] {unit.tag} ssl={ssl_time / 60:.1f} min probes={probe_time / 60:.1f} min")
    return {"rows": rows}
