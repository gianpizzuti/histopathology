#!/usr/bin/env python
"""Export the universe, train/val splits and labelled subsets of a dataset as
image ids, so that every experiment (and every co-author) can check that it
uses exactly the same data partition.

Writes results/lnbi/splits/<dataset>_splits.csv.gz with one row per image:
  id, label, split{s} in {train, val}, sub_s{s}_seed{k}_frac{pp} in {0, 1}
and prints a short fingerprint per split, to compare across machines.

Example:
  python scripts/make_splits.py --config configs/e1_vit_matched.yaml --dataset pcam
"""
import argparse
import sys

import numpy as np
import pandas as pd

from sslhist import data as D
from sslhist.config import load_config, resolve
from sslhist.io import frac_tag
from sslhist.data import PAPER_TABLE1, class_counts
from sslhist.utils import fingerprint


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--paths", default=None)
    ap.add_argument("--dataset", required=True, choices=["pcam", "panda"])
    args = ap.parse_args()

    cfg = load_config(args.config, args.paths)
    prot = cfg["protocol"]
    ds = D.load_datasource(args.dataset, cfg["data"])
    labels, ids = ds.labels_all(), np.asarray(ds.ids)
    table = pd.DataFrame({"id": ids, "label": labels})
    print(f"[{args.dataset}] universe n={len(ds)} class0/1={class_counts(labels)} fp={fingerprint(ids)}")
    ref = PAPER_TABLE1.get(args.dataset) if cfg["data"]["sample_frac"][args.dataset] == 0.2 else None
    mismatches = []
    if ref and class_counts(labels) != ref["universe"]:
        mismatches.append(f"universe {class_counts(labels)} != paper {ref['universe']}")

    for s in prot["splits"]:
        tr, va = D.make_split(labels, s, prot["val_ratio"], prot["split_base_seed"])
        col = np.full(len(ds), "", dtype=object)
        col[tr], col[va] = "train", "val"
        table[f"split{s}"] = col
        print(f"  split {s}: train class0/1={class_counts(labels[tr])} val class0/1={class_counts(labels[va])} "
              f"val_fp={fingerprint(ids[va])}")
        if ref:
            for part, idx in [("train", tr), ("val", va)]:
                if class_counts(labels[idx]) != ref[part]:
                    mismatches.append(f"split {s} {part} {class_counts(labels[idx])} != paper {ref[part]}")
        for k in prot["seeds"]:
            for f in prot["label_fracs"]:
                sub = D.label_subset(tr, labels, f, D.subset_seed(s, k))
                flag = np.zeros(len(ds), dtype=np.int8)
                flag[sub] = 1
                table[f"sub_s{s}_seed{k}_{frac_tag(f)}"] = flag

    out = resolve(cfg["results_dir"]) / "splits"
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.dataset}_splits.csv.gz"
    table.to_csv(path, index=False)
    print(f"Saved {path}")
    if ref:
        if mismatches:
            print("[CHECK] MISMATCH with the paper (Table 1):\n  " + "\n  ".join(mismatches))
            return 1
        print("[CHECK] class counts match Table 1 of the paper")
    return 0


if __name__ == "__main__":
    sys.exit(main())
