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

Examples:
  python scripts/aggregate.py --config configs/e1_vit_matched.yaml --reference resnet18
  python scripts/aggregate.py --config configs/e2_resnet50.yaml --include e1_vit_matched --reference resnet18
"""
import argparse
import sys

import pandas as pd

from sslhist.config import load_config, resolve
from sslhist.io import exp_results_dir, load_legacy_csv, load_raw
from sslhist.metrics import METRICS
from sslhist.plotting import plot_label_efficiency
from sslhist.report import check_completeness, paired_comparison, summarize, to_latex


def readable_table(summary: pd.DataFrame) -> str:
    """Plain-text mean +/- std table, for the terminal and summary.txt."""
    t = summary[["dataset", "backbone", "method", "label_frac", "n_runs"]].copy()
    t["label_frac"] = t["label_frac"].map(lambda f: f"{f * 100:g}%")
    for m in METRICS:
        t[m] = [f"{a:.4f} ± {b:.4f}" for a, b in zip(summary[f"{m}_mean"], summary[f"{m}_std"])]
    return t.to_string(index=False)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--reference", default=None, help="backbone for the paired comparison, e.g. resnet18")
    ap.add_argument("--legacy-csv", nargs="*", default=[],
                    help="conference results_raw_*.csv to include as experiment 'conference'")
    ap.add_argument("--include", nargs="*", default=[], metavar="EXPERIMENT",
                    help="other experiments (folder names in results/lnbi) to analyse together with this one")
    args = ap.parse_args()

    cfg = load_config(args.config)
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

    if args.reference:
        paired = paired_comparison(raw[raw.experiment.isin(exps)], args.reference)
        paired.to_csv(out / f"paired_vs_{args.reference}.csv", index=False)

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
