#!/usr/bin/env python
"""Probe robustness: retrain the linear probe of every saved unit with a converged,
cross-validated logistic regression (src/sslhist/probe_variants.py) on the same features, labelled subsets and validation
images, and compare with the protocol probe ("legacy"). No GPU, no pretraining: reads
artifacts/<experiment>/features/*.npz. With --ood-experiment, the E6 scores (OOD
confidence, temperature scaling) are recomputed for every probe from the OOD features
saved by E6 (artifacts/e6_ood/scores/*.npz, written by E6 runs of this version).

Writes results/lnbi/probe_robustness/:
  raw.csv, summary.csv, summary.txt     ID metrics (whole validation set) per probe
  paired_vs_legacy.csv                  difference probe - legacy, same runs, Wilcoxon
  e6_raw.csv, e6_summary.csv, e6_summary.txt   E6 scores per probe (+ supervised baselines)
  figures/probe_<probe>_<dataset>_<auroc|ece>.pdf

Example (CPU, 16 processes):
  python scripts/probe_robustness.py --experiments e1_vit_matched e2_resnet50 e3_dinov2 e3_barlow e4_federated \\
      --ood-experiment e6_ood --workers 16
"""
import argparse
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only: set before torch initialises CUDA
for _var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ.setdefault(_var, "1")

import multiprocessing as mp  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import wilcoxon  # noqa: E402

from sslhist import data as D  # noqa: E402
from sslhist.config import load_config, resolve  # noqa: E402
from sslhist.io import frac_tag, load_raw  # noqa: E402
from sslhist.metrics import METRICS, compute_metrics_binary  # noqa: E402
from sslhist.ood import score_fraction  # noqa: E402
from sslhist.plotting import plot_label_efficiency  # noqa: E402
from sslhist.probe import logits_to_probs  # noqa: E402
from sslhist.probe_variants import FITTERS, VARIANTS  # noqa: E402

KEYS = ["experiment", "dataset", "backbone", "method", "split", "seed", "label_frac"]
E6_METRICS = ["ood_auroc_raw", "id_msp_raw", "ood_msp_raw", "id_msp_ts", "ood_msp_ts", "ood_entropy_raw",
              "ood_entropy_ts", "id_ece_raw", "id_ece_ts", "temperature", "ood_conf90_raw", "ood_conf90_ts"]


def process_unit(job):
    import torch
    torch.set_num_threads(1)
    exp, npz_path, fracs, e6_path = job
    dataset, method, backbone, sp, sd = Path(npz_path).stem.split("__")  # Unit.tag
    split, seed = int(sp[len("split"):]), int(sd[len("seed"):])
    f = np.load(npz_path)
    tr_f, va_f, val_y = f["train_feats"].astype(np.float32), f["val_feats"].astype(np.float32), f["val_y"]
    pos_of = {int(v): i for i, v in enumerate(f["train_idx"])}
    e6 = np.load(e6_path) if e6_path else None
    if e6 is not None and ("ood_feats" not in e6.files or not np.array_equal(e6["val_y"], val_y)):
        e6 = None  # E6 run by an older version (no OOD features): ID metrics only
    ood_f = e6["ood_feats"].astype(np.float32) if e6 is not None else None
    base = {"experiment": exp, "dataset": dataset, "backbone": backbone, "method": method, "split": split,
            "seed": seed}
    rows, e6_rows = [], []
    for frac in fracs:
        ft = frac_tag(frac)
        pos = np.array([pos_of[int(v)] for v in f[f"{ft}_subset_idx"]])
        x, y = tr_f[pos], f["train_y"][pos]
        probes = {"legacy": (f[f"{ft}_val_logits"].astype(np.float64),
                             e6[f"{ft}_ood_logits"].astype(np.float64) if e6 is not None else None, {})}
        for name, fit in FITTERS.items():
            m = fit(x, y, seed=D.run_seed(split, seed))
            probes[name] = (m["logits"](va_f), m["logits"](ood_f) if ood_f is not None else None,
                            {k: v for k, v in m.items() if k != "logits"})
        for name, (z_val, z_ood, extra) in probes.items():
            row = {**base, "label_frac": frac, "probe": name, "n_labeled": int(len(pos)),
                   **compute_metrics_binary(logits_to_probs(z_val), val_y),
                   "mean_abs_logit": float(np.mean(np.abs(z_val))), **extra}
            rows.append(row)
            if z_ood is not None:
                sc = score_fraction(z_val, val_y, z_ood, e6["cal_pos"], e6["eval_pos"])
                e6_rows.append({**base, "label_frac": frac, "probe": name, **sc})
    return rows, e6_rows


def summarize(df: pd.DataFrame, metrics, by) -> pd.DataFrame:
    g = df.groupby(by, dropna=False)
    out = g[metrics].agg(["mean", "std"])
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    out["n_runs"] = g.size()
    return out.reset_index()


def pivot_text(summary: pd.DataFrame, metric: str, probes, by) -> str:
    rows = []
    for key, grp in summary.groupby(by, sort=True):
        r = dict(zip(by, key))
        r["label_frac"] = f"{r['label_frac'] * 100:g}%"
        for p in probes:
            s = grp[grp.probe == p]
            r[p] = f"{s[f'{metric}_mean'].iloc[0]:.3f} ± {s[f'{metric}_std'].iloc[0]:.3f}" if len(s) else "--"
        rows.append(r)
    return pd.DataFrame(rows).to_string(index=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/base.yaml", help="for paths and label fractions")
    ap.add_argument("--experiments", nargs="+", required=True)
    ap.add_argument("--ood-experiment", default=None, help="E6 experiment whose scores hold OOD features")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    cfg = load_config(args.config)
    fracs = cfg["protocol"]["label_fracs"]
    art = resolve(cfg["artifacts_dir"])
    jobs = []
    for exp in args.experiments:
        for p in sorted((art / exp / "features").glob("*.npz")):
            e6 = art / args.ood_experiment / "scores" / p.name if args.ood_experiment else None
            jobs.append((exp, str(p), fracs, str(e6) if e6 is not None and e6.exists() else None))
    if not jobs:
        print("No saved features found")
        return 1
    print(f"[probe] {len(jobs)} units x {len(fracs)} label fractions x {len(VARIANTS)} probes, "
          f"{args.workers} processes", flush=True)
    rows, e6_rows = [], []
    with mp.get_context("fork").Pool(args.workers) as pool:
        for i, (r, e) in enumerate(pool.imap_unordered(process_unit, jobs), 1):
            rows += r
            e6_rows += e
            if i % 20 == 0 or i == len(jobs):
                print(f"[probe] {i}/{len(jobs)} units", flush=True)

    out = resolve(cfg["results_dir"]) / "probe_robustness"
    (out / "figures").mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(rows)
    raw.to_csv(out / "raw.csv", index=False)
    by = ["experiment", "dataset", "backbone", "method", "label_frac", "probe"]
    summary = summarize(raw, METRICS + ["mean_abs_logit"], by)
    summary.to_csv(out / "summary.csv", index=False)

    # sanity: the legacy probe reproduces the result files of the experiments
    res = load_raw([resolve(cfg["results_dir"]) / e for e in args.experiments])
    chk = raw[raw.probe == "legacy"].merge(res[KEYS + ["auroc"]], on=KEYS, suffixes=("", "_result"))
    max_diff = float((chk["auroc"] - chk["auroc_result"]).abs().max()) if len(chk) else float("nan")

    # paired differences probe - legacy (same run, same labelled subset)
    grp = ["experiment", "dataset", "backbone", "method", "label_frac"]
    wide = {m: raw.pivot_table(index=KEYS, columns="probe", values=m) for m in ["auroc", "ece"]}
    paired = []
    for p in FITTERS:
        diffs = pd.DataFrame({m: wide[m][p] - wide[m]["legacy"] for m in wide}).reset_index()
        for key, g in diffs.groupby(grp):
            row = {**dict(zip(grp, key)), "probe": p, "n_pairs": len(g)}
            for m in wide:
                d = g[m].to_numpy()
                row[f"{m}_diff_mean"] = float(d.mean())
                row[f"{m}_diff_std"] = float(d.std(ddof=1)) if len(d) > 1 else np.nan
                row[f"{m}_wilcoxon_p"] = float(wilcoxon(d).pvalue) if len(d) > 1 and np.any(d != 0) else np.nan
            paired.append(row)
    pd.DataFrame(paired).to_csv(out / "paired_vs_legacy.csv", index=False)

    text = (f"Linear probe robustness (whole validation set). legacy = protocol probe (AdamW 1e-4, 10 epochs); "
            f"logreg_cv = L2 logistic regression, C by CV on the labelled subset.\nCheck: legacy vs result files, max |AUROC difference| = {max_diff:.2e}\n\n"
            "== AUROC\n" + pivot_text(summary, "auroc", VARIANTS, grp) +
            "\n\n== ECE\n" + pivot_text(summary, "ece", VARIANTS, grp) +
            "\n\n== mean |logit| on validation (scale of the probe output)\n" +
            pivot_text(summary, "mean_abs_logit", VARIANTS, grp) + "\n")
    (out / "summary.txt").write_text(text)
    print(text)

    for p in VARIANTS:
        s = summary[(summary.probe == p) & ~summary.method.str.contains("-fed-")]
        for ds in sorted(s["dataset"].unique()):
            for m in ["auroc", "ece"]:
                plot_label_efficiency(s, ds, m, out / "figures" / f"probe_{p}_{ds}_{m}.pdf",
                                      title=f"{ds.upper()} label efficiency: {m.upper()}, probe {p} (mean±std)")

    if e6_rows:
        e6 = pd.DataFrame(e6_rows)
        sup = load_raw([resolve(cfg["results_dir"]) / args.ood_experiment])
        sup = sup[sup.method.str.startswith("sup_")].assign(probe="end-to-end")
        e6 = pd.concat([e6, sup[KEYS + ["probe"] + E6_METRICS]], ignore_index=True)
        e6.to_csv(out / "e6_raw.csv", index=False)
        e6_sum = summarize(e6, E6_METRICS, ["dataset", "backbone", "method", "label_frac", "probe"])
        e6_sum.to_csv(out / "e6_summary.csv", index=False)
        probes = VARIANTS + ["end-to-end"]
        g6 = ["dataset", "backbone", "method", "label_frac"]
        text6 = ("E6 scores per probe (ID = evaluation half of val, OOD = other dataset; TS = temperature "
                 "scaling on the calibration half). end-to-end = supervised baselines of E6.\n\n" +
                 "\n\n".join(f"== {m}\n" + pivot_text(e6_sum, m, probes, g6)
                             for m in ["ood_auroc_raw", "id_msp_raw", "ood_msp_raw", "ood_msp_ts", "temperature",
                                       "id_ece_raw", "id_ece_ts"]) + "\n")
        (out / "e6_summary.txt").write_text(text6)
        print(text6)
    elif args.ood_experiment:
        print(f"[WARN] no OOD features in {art / args.ood_experiment / 'scores'}: re-run the SSL units of E6 "
              "with --force (see README) to score OOD with the other probes")
    print(f"[probe] {len(raw)} rows -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
