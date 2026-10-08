#!/usr/bin/env python
"""Run one SSL pretraining unit + linear probes for all label fractions.

Examples:
  # timing test on GPU 2 (numbered as in nvidia-smi): 200 SSL steps, prints the estimated time of a full run
  python scripts/run_unit.py --gpu 2 --config configs/e1_vit_matched.yaml \
      --dataset pcam --method simclr --backbone vit_b_16 --split 0 --seed 0 --timing 200

  # full run (skipped if its result files already exist, unless --force)
  python scripts/run_unit.py --gpu 2 --config configs/e1_vit_matched.yaml \
      --dataset pcam --method simclr --backbone vit_b_16 --split 1 --seed 0
"""
import argparse
import os
import sys

# Same CPU thread limits that launch.py sets for every unit, also when a unit is started
# by hand (set before numpy/torch are imported; one thread per core is very slow here).
for _var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"]:
    os.environ.setdefault(_var, "2")

from sslhist.config import load_config  # noqa: E402
from sslhist.io import Unit  # noqa: E402
from sslhist.runner import run_unit  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpu", default=None,
                    help="GPU id as shown by nvidia-smi ('cpu': no GPU); default: CUDA_VISIBLE_DEVICES")
    ap.add_argument("--config", required=True)
    ap.add_argument("--paths", default=None, help="paths YAML (default: configs/paths.yaml)")
    ap.add_argument("--dataset", required=True, choices=["pcam", "panda"])
    ap.add_argument("--method", required=True, choices=["simclr", "byol", "barlow", "frozen"])
    ap.add_argument("--backbone", required=True)
    ap.add_argument("--split", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--force", action="store_true", help="re-run even if results exist")
    ap.add_argument("--timing", type=int, default=None, metavar="STEPS",
                    help="run only STEPS SSL steps and print a time estimate; writes nothing")
    args = ap.parse_args()

    # Choose the GPU before CUDA is initialised; ids are the same as in nvidia-smi.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = "" if args.gpu == "cpu" else args.gpu
    elif "CUDA_VISIBLE_DEVICES" not in os.environ:
        ap.error("choose the GPU with --gpu <id as in nvidia-smi> (or set CUDA_VISIBLE_DEVICES)")

    cfg = load_config(args.config, args.paths)
    unit = Unit(args.dataset, args.method, args.backbone, args.split, args.seed)
    run_unit(cfg, unit, force=args.force, timing_steps=args.timing)
    return 0


if __name__ == "__main__":
    sys.exit(main())
