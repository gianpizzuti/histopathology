#!/usr/bin/env python
"""Collect the per-run JSON files of an experiment and write:

  results/lnbi/<experiment>/all_runs.csv       every run, one row per label fraction
  results/lnbi/<experiment>/summary.csv        mean, std (ddof=1), n per setting
  results/lnbi/<experiment>/summary.tex        same, as a LaTeX booktabs table
  results/lnbi/<experiment>/paired_vs_<ref>.csv  paired differences vs a reference backbone
  results/lnbi/<experiment>/figures/<experiment>_<dataset>_<metric>.pdf  (camera-ready style)

Example:
  python scripts/aggregate.py --config configs/e1_vit_matched.yaml --reference resnet18
"""
import argparse
import sys

import pandas as pd

from sslhist.config import load_config
from sslhist.io import exp_results_dir, load_legacy_csv, load_raw
from sslhist.metrics import METRICS
from sslhist.plotting import plot_label_efficiency
from sslhist.report import check_completeness, paired_comparison, summarize, to_latex


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--reference", default=None, help="backbone for the paired comparison, e.g. resnet18")
    ap.add_argument("--legacy-csv", nargs="*", default=[],
                    help="conference results_raw_*.csv to include as experiment 'conference'")
    args = ap.parse_args()

    cfg = load_config(args.config)
    out = exp_results_dir(cfg)
    raw = load_raw([out])
    if raw.empty:
        print(f"No results found in {out / 'raw'}")
        return 1
    if args.legacy_csv:
        raw = pd.concat([raw] + [load_legacy_csv(p) for p in args.legacy_csv], ignore_index=True)

    raw.to_csv(out / "all_runs.csv", index=False)
    summary = summarize(raw)
    summary.to_csv(out / "summary.csv", index=False)
    to_latex(summary[summary.experiment == cfg["experiment"]], out / "summary.tex",
             caption=f"{cfg['experiment']}: mean $\\pm$ std over splits $\\times$ seeds.")

    expected = len(cfg["protocol"]["splits"]) * len(cfg["protocol"]["seeds"])
    for msg in check_completeness(summary[summary.experiment == cfg["experiment"]], expected):
        print("[WARN] incomplete:", msg)

    if args.reference:
        paired = paired_comparison(raw[raw.experiment == cfg["experiment"]], args.reference)
        paired.to_csv(out / f"paired_vs_{args.reference}.csv", index=False)

    figs = out / "figures"
    figs.mkdir(exist_ok=True)
    cur = summary[summary.experiment == cfg["experiment"]]
    for ds in sorted(cur["dataset"].unique()):
        for m in METRICS:
            plot_label_efficiency(cur, ds, m, figs / f"{cfg['experiment']}_{ds}_{m}.pdf")

    print(f"[aggregate] {len(raw)} rows -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
