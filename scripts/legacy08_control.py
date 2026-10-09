#!/usr/bin/env python
"""E4 control: score saved encoders with the evaluation of notebook 08 (probe and ECE of
the federated diagnostic in the conference paper) and with ours, on the same images.

Shows how much of the conference federated-vs-centralised difference (Table 3: ECE
0.004 federated vs 0.101 centralised) comes from the two evaluation pipelines rather
than from federation. Runs on CPU from artifacts/<experiment>/features (no GPU, a few
minutes). Writes results/lnbi/<experiment>/legacy08_control/{raw.csv, summary.csv, summary.txt}.

Example (after E4):
  python scripts/legacy08_control.py --config configs/e4_federated.yaml \\
      --sources e1_vit_matched:resnet18:simclr e4_federated
"""
import argparse
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only: set before torch initialises CUDA
for _var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ.setdefault(_var, "4")

import pandas as pd  # noqa: E402

from sslhist.config import load_config  # noqa: E402
from sslhist.io import exp_results_dir  # noqa: E402
from sslhist.legacy08 import CONTROL_METRICS, run_control, summarize_control  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="output goes to this experiment's results folder")
    ap.add_argument("--sources", nargs="+", required=True, metavar="EXPERIMENT[:BACKBONE:METHOD]")
    ap.add_argument("--dataset", default="pcam")
    args = ap.parse_args()

    cfg = load_config(args.config)
    rows = run_control(cfg, args.sources, args.dataset)
    if not rows:
        print("No saved features found for", args.sources)
        return 1
    out = exp_results_dir(cfg) / "legacy08_control"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out / "raw.csv", index=False)
    summary = summarize_control(rows)
    summary.to_csv(out / "summary.csv", index=False)

    t = summary[["experiment", "method", "label_frac", "n_runs"]].copy()
    t["label_frac"] = t["label_frac"].map(lambda f: f"{f * 100:g}%")
    for m in CONTROL_METRICS:
        t[m] = [f"{a:.3f} ± {b:.3f}" for a, b in zip(summary[f"{m}_mean"], summary[f"{m}_std"])]
    text = ("Same encoders, labelled subsets and evaluation images; ours = protocol of E1-E4, "
            "nb08 = probe of notebook 08 (raw features, Adam 1e-3, early stopping);\n"
            "ece = ECE of the positive-class probability (all experiments), ece_toplabel = ECE of "
            "max(p, 1-p) (notebook 08)\n\n" + t.to_string(index=False))
    (out / "summary.txt").write_text(text + "\n")
    print(text)
    print(f"[control] {len(rows)} rows -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
