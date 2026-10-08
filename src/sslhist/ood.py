"""E6: is OOD overconfidence specific to SSL? PCam <-> PANDA, as in notebook 09.

One E6 unit = one (ID dataset, method, backbone, split, seed), scored at every label
fraction. ID = validation part of the split; OOD = 2000 images of the other dataset
(``RandomState(123)`` on its universe, as in notebook 09), resized to the ID image size
and normalised as the ID images (notebook 09 fix: no resolution confound).

- SSL / frozen encoders (E1-E3): nothing is retrained. The saved encoder computes the
  OOD features; the saved standardizer and linear probe of each label fraction turn
  them into logits. ID logits are the saved validation logits (identical to E1-E3).
- Supervised baselines (``sup_scratch``, ``sup_imagenet``): backbone + head trained end
  to end on the same labelled subset (see supervised.py), then the same scoring.

Scores (binary classifier, p = sigmoid(logit / T)):
- confidence = MSP = max(p, 1-p); uncertainty = binary entropy in bits (0..1);
- OOD detection: score 1 - MSP (computed from -|logit| so that clipped probabilities do
  not create ties), OOD = positive class, not flipped (``ood_auroc_raw``: < 0.5 means the
  classifier is MORE confident on OOD than on ID); FPR at 95% ID retained;
- temperature scaling: T fitted (NLL) on a stratified half of the ID validation set
  (calibration half); every ID metric of E6 and the OOD scores use the other half
  (evaluation half), before (``_raw``) and after (``_ts``) scaling. A single temperature
  does not change the ranking of |logit|, hence ``ood_auroc_raw`` and ``ood_fpr95_raw``
  are the same after scaling: TS changes how confident the model is on OOD, not how
  well confidence separates ID from OOD. For a binary classifier MSP and entropy rank
  samples identically, so one AUROC covers both;
- feature-space control, independent of the classifier: distance to the k-th nearest
  neighbour (cosine, k=10) in a bank of ID training features (``knn_ood_auroc``). High
  values with low ``ood_auroc_raw`` mean that the representation separates the two
  datasets but the classifier's confidence does not.
"""
import json
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize_scalar
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import DataLoader

from . import data as D
from .config import family_cfg, resolve
from .io import Unit, exp_artifacts_dir, frac_tag, result_path, unit_complete, write_json
from .metrics import compute_metrics_binary, sanitize_np_probs
from .models import build_backbone
from .probe import extract_features, logits_to_probs, sanitize_tensor
from .supervised import features_and_logits, train_supervised
from .utils import fingerprint, get_device, run_metadata, set_seed

# method -> ImageNet initialisation of the supervised baselines
SUPERVISED = {"sup_scratch": False, "sup_imagenet": True}
ENCODER_LABELS = {"sup_scratch": "supervised from scratch", "sup_imagenet": "supervised, ImageNet init",
                  "frozen": "pretrained, frozen"}


def log(msg: str) -> None:
    print(msg, flush=True)


# -----------------------
# Samples
# -----------------------
def ood_sample(n_universe: int, n: int, seed: int) -> np.ndarray:
    """Indices of the OOD images, as notebook 09 (same for every split and seed)."""
    rng = np.random.RandomState(seed)
    return rng.choice(np.arange(n_universe, dtype=np.int64), size=min(n, n_universe), replace=False)


def calibration_split(y_val: np.ndarray, split_id: int, calib_frac: float, base_seed: int
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """Positions (in validation order) of the calibration and evaluation halves; stratified,
    the same for every method, backbone and seed of a split, so comparisons stay paired."""
    sss = StratifiedShuffleSplit(n_splits=1, train_size=calib_frac, random_state=base_seed + split_id)
    cal, ev = next(sss.split(np.zeros(len(y_val)), y_val))
    return np.sort(cal), np.sort(ev)


def knn_bank_positions(n_train: int, n_bank: int, seed: int) -> np.ndarray:
    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(n_train, size=min(n_bank, n_train), replace=False))


# -----------------------
# Scores
# -----------------------
def probs_at(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    return logits_to_probs(np.asarray(logits, dtype=np.float32) / np.float32(temperature))


def msp(p: np.ndarray) -> np.ndarray:
    """Maximum softmax probability of a binary classifier."""
    return np.maximum(p, 1.0 - p)


def binary_entropy(p: np.ndarray) -> np.ndarray:
    """In bits: 0 = certain, 1 = p of 0.5."""
    p = sanitize_np_probs(p).astype(np.float64)
    return -(p * np.log2(p) + (1 - p) * np.log2(1 - p))


def nll(p: np.ndarray, y: np.ndarray) -> float:
    p = sanitize_np_probs(p).astype(np.float64)
    y = np.asarray(y, dtype=np.float64)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def fit_temperature(logits: np.ndarray, y: np.ndarray, bounds=(0.01, 100.0)) -> float:
    """Temperature minimising the NLL of sigmoid(logit / T) (Guo et al., 2017). It reaches the
    upper bound when the logits carry no information on the calibration half (T -> inf: p = 0.5)."""
    z, y = np.asarray(logits, dtype=np.float64), np.asarray(y, dtype=np.float64)

    def loss(log_t):
        p = np.clip(expit(z / np.exp(log_t)), 1e-7, 1 - 1e-7)
        return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

    res = minimize_scalar(loss, bounds=(np.log(bounds[0]), np.log(bounds[1])), method="bounded",
                          options={"xatol": 1e-5})
    return float(np.exp(res.x))


def detection(id_score: np.ndarray, ood_score: np.ndarray) -> Tuple[float, float]:
    """AUROC with OOD as the positive class (higher score = more OOD-like, no flipping) and
    FPR95: fraction of OOD samples kept as ID at the threshold that keeps 95% of ID."""
    y = np.r_[np.zeros(len(id_score)), np.ones(len(ood_score))]
    try:
        auroc = float(roc_auc_score(y, np.r_[id_score, ood_score]))
    except ValueError:
        auroc = float("nan")
    fpr95 = float(np.mean(ood_score <= np.quantile(id_score, 0.95)))
    return auroc, fpr95


@torch.no_grad()
def knn_distance(bank: torch.Tensor, queries: torch.Tensor, k: int, device: torch.device,
                 chunk: int = 4096) -> np.ndarray:
    """Cosine distance to the k-th nearest neighbour in the bank (Sun et al., ICML 2022)."""
    bank = F.normalize(bank.float().to(device), dim=1)
    k = min(k, bank.size(0))
    out = []
    for q in queries.split(chunk):
        sim = F.normalize(q.float().to(device), dim=1) @ bank.T
        out.append((1.0 - sim.topk(k, dim=1).values[:, -1]).cpu())
    return torch.cat(out).numpy()


def knn_detection(bank, id_feats, ood_feats, k: int, device) -> Tuple[float, float]:
    as_t = lambda x: x if isinstance(x, torch.Tensor) else torch.from_numpy(np.asarray(x, dtype=np.float32))  # noqa: E731
    return detection(knn_distance(as_t(bank), as_t(id_feats), k, device),
                     knn_distance(as_t(bank), as_t(ood_feats), k, device))


def score_fraction(val_logits: np.ndarray, val_y: np.ndarray, ood_logits: np.ndarray,
                   cal_pos: np.ndarray, eval_pos: np.ndarray) -> Dict[str, float]:
    """Temperature (fitted on the calibration half) and every ID/OOD score on the evaluation half."""
    t = fit_temperature(val_logits[cal_pos], val_y[cal_pos])
    z_id, y_id = np.asarray(val_logits)[eval_pos], np.asarray(val_y)[eval_pos]
    z_ood = np.asarray(ood_logits)
    out = {"temperature": t}
    for tag, temp in [("raw", 1.0), ("ts", t)]:
        p_id, p_ood = probs_at(z_id, temp), probs_at(z_ood, temp)
        m = compute_metrics_binary(p_id, y_id)
        out[f"id_ece_{tag}"], out[f"id_brier_{tag}"], out[f"id_nll_{tag}"] = m["ece"], m["brier"], nll(p_id, y_id)
        for name, p in [("id", p_id), ("ood", p_ood)]:
            c = msp(p)
            out[f"{name}_msp_{tag}"] = float(c.mean())
            out[f"{name}_entropy_{tag}"] = float(binary_entropy(p).mean())
            out[f"{name}_conf90_{tag}"] = float((c >= 0.9).mean())
        if tag == "raw":
            out["id_eval_auroc"] = m["auroc"]
            out["ood_auroc_raw"], out["ood_fpr95_raw"] = detection(-np.abs(z_id), -np.abs(z_ood))
            out["ood_pos_rate"] = float((p_ood >= 0.5).mean())
    return out


# -----------------------
# Saved SSL / frozen encoders (E1-E3)
# -----------------------
def find_source(cfg: dict, unit: Unit) -> str:
    """First experiment in ``ood.source_experiments`` holding the features of this unit."""
    tried = []
    for exp in cfg["ood"]["source_experiments"]:
        p = resolve(cfg["artifacts_dir"]) / exp / "features" / f"{unit.tag}.npz"
        if p.exists():
            return exp
        tried.append(str(p))
    raise FileNotFoundError(f"No saved features for {unit.tag}; looked for:\n  " + "\n  ".join(tried))


def load_encoder(cfg: dict, source_exp: str, unit: Unit) -> torch.nn.Module:
    if unit.method == "frozen":
        backbone, _ = build_backbone(unit.backbone, pretrained=True)
        return backbone
    backbone, _ = build_backbone(unit.backbone, pretrained=False)
    path = resolve(cfg["artifacts_dir"]) / source_exp / "encoders" / f"{unit.tag}.pt"
    backbone.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    return backbone


def probe_logits(feats: torch.Tensor, w, b, mu, sd) -> np.ndarray:
    """Saved linear probe applied as in probe.predict_logits (standardise, then w.x + b)."""
    as_t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32))  # noqa: E731
    x = sanitize_tensor((feats.float() - as_t(mu)) / as_t(sd))
    return sanitize_tensor(x @ as_t(w) + as_t(b)).numpy()


# -----------------------
# Unit
# -----------------------
def run_ood_unit(cfg: dict, unit: Unit, force: bool = False) -> Optional[dict]:
    prot, rt, oc = cfg["protocol"], cfg["runtime"], cfg["ood"]
    if not force and unit_complete(cfg, unit):
        log(f"[SKIP] {unit.tag} already complete")
        return None

    torch.set_num_threads(int(rt.get("torch_threads", 4)))
    device = get_device()
    amp = bool(prot["amp"]) and device.type == "cuda"
    nw, pin = int(rt["num_workers"]), device.type == "cuda"
    fam = family_cfg(cfg, unit.backbone)
    img, norm, bs = int(fam["img_size"][unit.dataset]), D.norm_from_cfg(fam), int(fam["batch_sup"])
    ood_name = oc["pairs"][unit.dataset]

    ds, ods = D.load_datasource(unit.dataset, cfg["data"]), D.load_datasource(ood_name, cfg["data"])
    labels, ids = ds.labels_all(), np.asarray(ds.ids)
    tr_idx, va_idx = D.make_split(labels, unit.split, prot["val_ratio"], prot["split_base_seed"])
    rs, ss = D.run_seed(unit.split, unit.seed), D.subset_seed(unit.split, unit.seed)
    ood_idx = ood_sample(len(ods), int(oc["n_samples"]), int(oc["sample_seed"]))
    cal_pos, eval_pos = calibration_split(labels[va_idx], unit.split, float(oc["calib_frac"]),
                                          int(oc["calib_seed_base"]))
    bank_pos = knn_bank_positions(len(tr_idx), int(oc["knn_bank"]), int(oc["knn_seed"]))
    k = int(oc["knn_k"])
    log("=" * 90)
    log(f"[RUN] {unit.tag} img={img} n_tr={len(tr_idx)} n_va={len(va_idx)} device={device} amp={amp}")
    log(f"[E6] ID={unit.dataset} OOD={ood_name} n_ood={len(ood_idx)} calibration/evaluation "
        f"halves of val={len(cal_pos)}/{len(eval_pos)} knn bank={len(bank_pos)} k={k}")
    log("=" * 90)

    def loader(source, idx):
        return DataLoader(source.make_sup_dataset(idx, img_size=img, norm=norm), batch_size=bs,
                          shuffle=False, num_workers=nw, pin_memory=pin)

    ood_dl = loader(ods, ood_idx)
    t0 = time.time()
    per_frac, checks = {}, {}
    if unit.method in SUPERVISED:
        source_exp, encoder = "", ENCODER_LABELS[unit.method]
        sc = cfg["supervised"]
        val_dl, bank_dl = loader(ds, va_idx), loader(ds, tr_idx[bank_pos])
        for frac in prot["label_fracs"]:
            sub_idx = D.label_subset(tr_idx, labels, frac, seed=ss)
            log(f"  [SUP] frac={frac:.2f} n_lab={len(sub_idx)} init={'ImageNet' if SUPERVISED[unit.method] else 'random'}")
            set_seed(rs)  # every label fraction starts from the same seed
            model = train_supervised(
                unit.backbone, SUPERVISED[unit.method], ds.make_sup_dataset(sub_idx, img_size=img, norm=norm),
                epochs=int(sc["epochs"]), lr=float(sc["lr"]), weight_decay=float(sc["weight_decay"]),
                batch_size=bs, num_workers=nw, device=device, amp=amp,
                unscale_before_clip=bool(sc.get("unscale_before_clip", True)), log=log)
            va_f, va_logits, val_y = features_and_logits(model, val_dl, device, amp)
            ood_f, ood_logits, _ = features_and_logits(model, ood_dl, device, amp)
            bank_f, _, _ = features_and_logits(model, bank_dl, device, amp)
            per_frac[frac] = {"sub_idx": sub_idx, "val_logits": va_logits, "ood_logits": ood_logits,
                              "knn": knn_detection(bank_f, va_f[eval_pos], ood_f, k, device)}
    else:
        source_exp = find_source(cfg, unit)
        src = np.load(resolve(cfg["artifacts_dir"]) / source_exp / "features" / f"{unit.tag}.npz")
        if not (np.array_equal(src["train_idx"], tr_idx) and np.array_equal(src["val_idx"], va_idx)):
            raise RuntimeError(f"{unit.tag}: the split saved in {source_exp} differs from this one")
        with open(resolve(cfg["results_dir"]) / source_exp / "raw" / f"{unit.tag}__frac01.json") as f:
            src_row = json.load(f)
        if int(src_row["img_size"]) != img:
            raise RuntimeError(f"{unit.tag}: image size {img} here, {src_row['img_size']} in {source_exp}")
        # results written before E3 have no "encoder" field: they are all SSL from scratch
        encoder = src_row.get("encoder", "SSL from scratch")
        val_y = src["val_y"]
        set_seed(rs)  # as run_unit before building the encoder (only matters for randomly initialised test stand-ins)
        backbone = load_encoder(cfg, source_exp, unit).to(device)

        # The loaded encoder must reproduce the saved validation features (same weights,
        # image size and normalisation); fp16 storage and AMP give differences of ~1e-3.
        n = min(int(oc["encoder_check_n"]), len(va_idx))
        chk, _ = extract_features(backbone, loader(ds, va_idx[:n]), device, amp)
        saved = torch.from_numpy(src["val_feats"][:n].astype(np.float32))
        checks["encoder_check_rel_diff"] = float((chk - saved).abs().max() / saved.abs().max().clamp_min(1e-6))
        if checks["encoder_check_rel_diff"] > 0.05:
            raise RuntimeError(f"{unit.tag}: the saved encoder does not reproduce the saved features "
                               f"(relative difference {checks['encoder_check_rel_diff']:.3g})")

        ood_f, _ = extract_features(backbone, ood_dl, device, amp)
        # same fp16 rounding as the saved ID features
        knn = knn_detection(src["train_feats"][bank_pos], src["val_feats"][eval_pos], ood_f.half(), k, device)
        checks["probe_check_max_abs_diff"] = 0.0
        for frac in prot["label_fracs"]:
            ft = frac_tag(frac)
            sub_idx = D.label_subset(tr_idx, labels, frac, seed=ss)
            if not np.array_equal(src[f"{ft}_subset_idx"], sub_idx):
                raise RuntimeError(f"{unit.tag}: labelled subset {ft} differs from the saved one")
            w, b, mu, sd = (src[f"{ft}_{x}"] for x in ["probe_w", "probe_b", "mu", "sd"])
            # the saved probe applied to the re-encoded validation images must give the saved logits
            # (exact on CPU; on GPU, AMP with other batch sizes gives small differences)
            rebuilt = probe_logits(chk, w, b, mu, sd)
            checks["probe_check_max_abs_diff"] = max(checks["probe_check_max_abs_diff"],
                                                     float(np.abs(rebuilt - src[f"{ft}_val_logits"][:n]).max()))
            per_frac[frac] = {"sub_idx": sub_idx, "val_logits": src[f"{ft}_val_logits"],
                              "ood_logits": probe_logits(ood_f, w, b, mu, sd), "knn": knn}
        log(f"  [CHECK] encoder rel. diff={checks['encoder_check_rel_diff']:.2e} "
            f"probe max diff={checks['probe_check_max_abs_diff']:.2e} (source: {source_exp})")

    rows, scores = [], {"cal_pos": cal_pos, "eval_pos": eval_pos, "ood_idx": ood_idx, "val_y": val_y}
    for frac, r in per_frac.items():
        full = compute_metrics_binary(logits_to_probs(r["val_logits"]), val_y)  # whole val, as E1-E3
        if source_exp and abs(full["auroc"] - src_row_auroc(cfg, source_exp, unit, frac)) > 1e-5:
            raise RuntimeError(f"{unit.tag}: validation AUROC differs from {source_exp}")
        sc = score_fraction(r["val_logits"], val_y, r["ood_logits"], cal_pos, eval_pos)
        sc["knn_ood_auroc"], sc["knn_fpr95"] = r["knn"]
        log(f"  [OOD] frac={frac:.2f} T={sc['temperature']:.2f} MSP id/ood raw={sc['id_msp_raw']:.3f}/"
            f"{sc['ood_msp_raw']:.3f} ts={sc['id_msp_ts']:.3f}/{sc['ood_msp_ts']:.3f} "
            f"ood_auroc_raw={sc['ood_auroc_raw']:.3f} knn_auroc={sc['knn_ood_auroc']:.3f} "
            f"ECE raw/ts={sc['id_ece_raw']:.3f}/{sc['id_ece_ts']:.3f}")
        rows.append({
            "experiment": cfg["experiment"],
            "dataset": unit.dataset, "ood_dataset": ood_name, "method": unit.method, "backbone": unit.backbone,
            "split": unit.split, "seed": unit.seed, "label_frac": frac,
            **full, **sc,
            "n_labeled": int(len(r["sub_idx"])), "n_val": int(len(va_idx)), "n_cal": int(len(cal_pos)),
            "n_eval": int(len(eval_pos)), "n_ood": int(len(ood_idx)), "knn_k": k, "knn_bank_n": int(len(bank_pos)),
            "img_size": img, "normalize": "custom" if fam.get("normalize") else "legacy (0.5, 0.5)",
            "encoder": encoder, "source_experiment": source_exp or "trained here",
            **({f"sup_{k_}": float(cfg["supervised"][k_]) for k_ in ["epochs", "lr", "weight_decay"]}
               if unit.method in SUPERVISED else {}),
            "run_seed": rs, "subset_seed": ss,
            "val_ids_fp": fingerprint(ids[va_idx]), "subset_ids_fp": fingerprint(ids[r["sub_idx"]]),
            "ood_ids_fp": fingerprint(np.asarray(ods.ids)[ood_idx]),
            **checks,
        })
        ft = frac_tag(frac)
        scores.update({f"{ft}_val_logits": np.asarray(r["val_logits"], dtype=np.float32),
                       f"{ft}_ood_logits": np.asarray(r["ood_logits"], dtype=np.float32),
                       f"{ft}_temperature": np.float32(sc["temperature"])})

    art = exp_artifacts_dir(cfg)
    (art / "scores").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(art / "scores" / f"{unit.tag}.npz", **scores)
    meta = run_metadata()
    write_json(art / "logs" / f"{unit.tag}.json", {"unit": unit.tag, "total_time_s": time.time() - t0, **meta})
    for row in rows:  # written last: a unit counts as complete only once everything is saved
        write_json(result_path(cfg, unit, row["label_frac"]), {**row, **meta})
    log(f"[DONE] {unit.tag} in {(time.time() - t0) / 60:.1f} min")
    return {"rows": rows}


def src_row_auroc(cfg: dict, source_exp: str, unit: Unit, frac: float) -> float:
    p = Path(resolve(cfg["results_dir"])) / source_exp / "raw" / f"{unit.tag}__{frac_tag(frac)}.json"
    with open(p) as f:
        return float(json.load(f)["auroc"])
