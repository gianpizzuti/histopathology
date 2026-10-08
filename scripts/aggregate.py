#!/usr/bin/env python
"""Collect the per-run JSON files of an experiment and write:

  results/lnbi/<experiment>/all_runs.csv       every run, one row per label fraction
  results/lnbi/<experiment>/summary.csv        mean, std (ddof=1), n per setting
  results/lnbi/<experiment>/summary.tex        same, as a LaTeX booktabs table
  results/lnbi/<experiment>/summary.txt        same, as a plain-text table (also printed)
  results/lnbi/<experiment>/paired_vs_<ref>.csv  paired differences vs a reference backbone
  results/lnbi/<experiment>/figures/<experiment>_<dataset>_<metric>.pdf  (camera-ready style)

With --include, the results of other experiments run on the same splits are
added (e.g. E2 + E1: ResNet-50 vs ResNet-18 vs ViT-B/16) and everything is
written to results/lnbi/<experiment>/with_<included>/ instead.

With the E6 config (kind: ood) the tables are about confidence and OOD detection
(summary.txt: three tables; summary.tex + summary_detection.tex), the paired
comparisons use the OOD metrics, and the figures are label-efficiency curves of the
OOD metrics, confidence bars and MSP histograms (the last read artifacts/e6_ood/scores).

Examples:
  python scripts/aggregate.py --config configs/e1_vit_matched.yaml --reference resnet18
  python scripts/aggregate.py --config configs/e2_resnet50.yaml --include e1_vit_matched --reference resnet18
  python scripts/aggregate.py --config configs/e3_dinov2.yaml --include e1_vit_matched e2_resnet50 \
      --reference resnet50:simclr
  python scripts/aggregate.py --config configs/e6_ood.yaml --reference resnet18:sup_imagenet resnet18:sup_scratch
"""
import argparse
import sys

import numpy as np
import pandas as pd

from sslhist.config import load_config, resolve
from sslhist.io import exp_artifacts_dir, exp_results_dir, grid_units, load_legacy_csv, load_raw
from sslhist.metrics import METRICS
from sslhist.plotting import (METRIC_LABELS, plot_confidence_bars, plot_label_efficiency, plot_msp_histograms,
                              series_label)
from sslhist.report import check_completeness, paired_comparison, summarize, to_latex

# E6 tables (summary.txt), paired comparisons and figures
OOD_TABLES = {
    "ID classification and calibration (AUROC on the whole val; ECE/NLL on its evaluation half; "
    "TS = after temperature scaling)": ["auroc", "id_ece_raw", "id_ece_ts", "temperature", "id_nll_raw", "id_nll_ts"],
    "Confidence: mean MSP and entropy (bits) on ID (evaluation half) and OOD images, "
    "share of OOD images with MSP >= 0.9": ["id_msp_raw", "ood_msp_raw", "id_msp_ts", "ood_msp_ts", "id_entropy_raw",
                                            "ood_entropy_raw", "id_entropy_ts", "ood_entropy_ts", "ood_conf90_raw",
                                            "ood_conf90_ts"],
    "OOD detection, OOD = positive, not flipped (MSP scores rank samples the same after TS); "
    "kNN = feature space": ["ood_auroc_raw", "ood_fpr95_raw", "knn_ood_auroc", "knn_fpr95", "ood_pos_rate"],
}
OOD_PAIRED = ["ood_msp_raw", "ood_msp_ts", "ood_entropy_raw", "ood_entropy_ts", "ood_conf90_raw", "ood_conf90_ts",
              "ood_auroc_raw", "knn_ood_auroc", "id_ece_raw", "id_ece_ts", "auroc"]
OOD_FIGURES = ["ood_msp_raw", "ood_msp_ts", "ood_entropy_raw", "ood_entropy_ts", "ood_auroc_raw", "knn_ood_auroc",
               "id_ece_raw", "id_ece_ts", "auroc"]
OOD_TEX = {"summary.tex": ["id_ece_raw", "id_ece_ts", "temperature", "id_msp_raw", "ood_msp_raw", "ood_msp_ts",
                           "ood_entropy_raw", "ood_entropy_ts"],
           "summary_detection.tex": ["auroc", "ood_auroc_raw", "ood_fpr95_raw", "knn_ood_auroc", "knn_fpr95",
                                     "ood_pos_rate"]}


def readable_table(summary: pd.DataFrame, metrics=METRICS, digits: int = 4) -> str:
    """Plain-text mean +/- std table, for the terminal and summary.txt."""
    t = summary[["dataset", "backbone", "method", "label_frac", "n_runs"]].copy()
    t["label_frac"] = t["label_frac"].map(lambda f: f"{f * 100:g}%")
    for m in metrics:
        t[m] = [f"{a:.{digits}f} ± {b:.{digits}f}" for a, b in zip(summary[f"{m}_mean"], summary[f"{m}_std"])]
    return t.to_string(index=False)


def msp_panels(cfg: dict, dataset: str, frac: float):
    """MSP of every run of each series (pooled over splits x seeds) from artifacts/<exp>/scores."""
    from sslhist.io import frac_tag
    from sslhist.ood import msp, probs_at
    ft, panels = frac_tag(frac), {}
    for u in grid_units(cfg, datasets=[dataset]):
        p = exp_artifacts_dir(cfg) / "scores" / f"{u.tag}.npz"
        if not p.exists():
            continue
        s = np.load(p)
        z_id, z_ood, t = s[f"{ft}_val_logits"][s["eval_pos"]], s[f"{ft}_ood_logits"], float(s[f"{ft}_temperature"])
        d = panels.setdefault((u.method, u.backbone), {"id_raw": [], "ood_raw": [], "id_ts": [], "ood_ts": []})
        for key, z, temp in [("id_raw", z_id, 1.0), ("ood_raw", z_ood, 1.0), ("id_ts", z_id, t), ("ood_ts", z_ood, t)]:
            d[key].append(msp(probs_at(z, temp)))
    order = sorted(panels, key=lambda s: (s[0].startswith("sup"), s))
    return [(series_label(*k), {key: np.concatenate(v) for key, v in panels[k].items()}) for k in order]


def aggregate_ood(cfg: dict, args) -> int:
    exp_dir = exp_results_dir(cfg)
    raw = load_raw([exp_dir])
    if raw.empty:
        print(f"No results found in {exp_dir / 'raw'}")
        return 1
    raw.to_csv(exp_dir / "all_runs.csv", index=False)
    metrics = list(dict.fromkeys(m for ms in OOD_TABLES.values() for m in ms))
    summary = summarize(raw, metrics).sort_values(["dataset", "method", "backbone", "label_frac"])
    summary.to_csv(exp_dir / "summary.csv", index=False)
    expected = len(cfg["protocol"]["splits"]) * len(cfg["protocol"]["seeds"])
    for msg in check_completeness(summary, expected):
        print("[WARN] incomplete:", msg)

    pairs = cfg["ood"]["pairs"]
    for name, cols in OOD_TEX.items():
        to_latex(summary, exp_dir / name, metrics=cols,
                 caption="E6, " + ", ".join(f"{d} (ID) vs {o} (OOD)" for d, o in pairs.items()) +
                 ": mean $\\pm$ std over splits $\\times$ seeds; TS = after temperature scaling.")
    text = "\n\n".join(f"== {title}\n" + readable_table(summary, cols, digits=3) for title, cols in OOD_TABLES.items())
    (exp_dir / "summary.txt").write_text(text + "\n")
    print(text)

    for ref in args.reference:
        paired = paired_comparison(raw, ref, OOD_PAIRED)
        paired.to_csv(exp_dir / f"paired_vs_{ref.replace(':', '-')}.csv", index=False)

    figs = exp_dir / "figures"
    figs.mkdir(exist_ok=True)
    for ds in sorted(summary["dataset"].unique()):
        ood = pairs[ds]
        for m in OOD_FIGURES:
            plot_label_efficiency(summary, ds, m, figs / f"{cfg['experiment']}_{ds}_{m}.pdf",
                                  title=f"{ds.upper()} (ID) vs {ood.upper()} (OOD): {METRIC_LABELS[m]} (mean±std)",
                                  ylabel=METRIC_LABELS[m])
        frac = max(summary["label_frac"])
        plot_confidence_bars(summary, ds, ood, frac, figs / f"{cfg['experiment']}_{ds}_confidence_bars.pdf")
        panels = msp_panels(cfg, ds, frac)
        if panels:
            plot_msp_histograms(panels, f"{ds.upper()} (ID) vs {ood.upper()} (OOD), {frac * 100:g}% labels: "
                                "MSP over all splits x seeds", figs / f"{cfg['experiment']}_{ds}_msp_hist.pdf")
        else:
            print(f"[WARN] no scores in {exp_artifacts_dir(cfg) / 'scores'}: MSP histograms skipped")
    print(f"[aggregate] {len(raw)} rows -> {exp_dir}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--reference", nargs="*", default=[],
                    help="paired comparison: a backbone (resnet18: same method) or backbone:method "
                         "(resnet50:simclr: every series vs that one); several allowed")
    ap.add_argument("--legacy-csv", nargs="*", default=[],
                    help="conference results_raw_*.csv to include as experiment 'conference'")
    ap.add_argument("--include", nargs="*", default=[], metavar="EXPERIMENT",
                    help="other experiments (folder names in results/lnbi) to analyse together with this one")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if cfg.get("kind") == "ood":
        if args.include or args.legacy_csv:
            ap.error("--include / --legacy-csv are not used with the E6 config")
        return aggregate_ood(cfg, args)
    exp_dir = exp_results_dir(cfg)
    exps = [cfg["experiment"]] + args.include
    raw = load_raw([exp_dir] + [resolve(cfg["results_dir"]) / e for e in args.include])
    if raw.empty or not raw["experiment"].isin([cfg["experiment"]]).any():
        print(f"No results found in {exp_dir / 'raw'}")
        return 1
    missing = [e for e in args.include if not raw["experiment"].isin([e]).any()]
    if missing:
        print(f"No results found for included experiment(s): {missing}")
        return 1
    if args.legacy_csv:
        raw = pd.concat([raw] + [load_legacy_csv(p) for p in args.legacy_csv], ignore_index=True)

    out = exp_dir / ("with_" + "_".join(args.include)) if args.include else exp_dir
    out.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out / "all_runs.csv", index=False)
    summary = summarize(raw)
    summary.to_csv(out / "summary.csv", index=False)
    cur = summary[summary.experiment.isin(exps)].sort_values(["dataset", "backbone", "method", "label_frac"])
    to_latex(cur, out / "summary.tex",
             caption=f"{' + '.join(exps)}: mean $\\pm$ std over splits $\\times$ seeds.")

    expected = len(cfg["protocol"]["splits"]) * len(cfg["protocol"]["seeds"])
    for msg in check_completeness(cur, expected):
        print("[WARN] incomplete:", msg)

    for ref in args.reference:
        paired = paired_comparison(raw[raw.experiment.isin(exps)], ref)
        paired.to_csv(out / f"paired_vs_{ref.replace(':', '-')}.csv", index=False)

    table = readable_table(cur)
    (out / "summary.txt").write_text(table + "\n")
    print(table)

    figs = out / "figures"
    figs.mkdir(exist_ok=True)
    prefix = "_".join(exps)
    for ds in sorted(cur["dataset"].unique()):
        for m in METRICS:
            plot_label_efficiency(cur, ds, m, figs / f"{prefix}_{ds}_{m}.pdf")

    print(f"[aggregate] {len(raw)} rows -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
