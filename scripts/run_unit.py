#!/usr/bin/env python
"""Run one SSL pretraining unit + linear probes for all label fractions.

Examples:
  # timing test: 200 SSL steps, prints the estimated time of a full run
  python scripts/run_unit.py --config configs/e1_vit_matched.yaml \
      --dataset pcam --method simclr --backbone vit_b_16 --split 0 --seed 0 --timing 200

  # full run (skipped if its result files already exist, unless --force)
  CUDA_VISIBLE_DEVICES=0 python scripts/run_unit.py --config configs/e1_vit_matched.yaml \
      --dataset pcam --method simclr --backbone vit_b_16 --split 1 --seed 0
"""
import argparse
import sys

from sslhist.config import load_config
from sslhist.io import Unit
from sslhist.runner import run_unit


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--paths", default=None, help="paths YAML (default: configs/paths.yaml)")
    ap.add_argument("--dataset", required=True, choices=["pcam", "panda"])
    ap.add_argument("--method", required=True, choices=["simclr", "byol"])
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--split", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--force", action="store_true", help="re-run even if results exist")
    ap.add_argument("--timing", type=int, default=None, metavar="STEPS",
                    help="run only STEPS SSL steps and print a time estimate; writes nothing")
    args = ap.parse_args()

    cfg = load_config(args.config, args.paths)
    unit = Unit(args.dataset, args.method, args.backbone, args.split, args.seed)
    run_unit(cfg, unit, force=args.force, timing_steps=args.timing)
    return 0


if __name__ == "__main__":
    sys.exit(main())
