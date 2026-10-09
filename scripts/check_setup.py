#!/usr/bin/env python
"""Check that everything works before launching an experiment.

  1. environment  PyTorch, CUDA, the chosen GPUs (and whether someone else is using them), CPU cores
  2. data         the three paths, PCam and PANDA universes, one image per dataset,
                  PANDA class counts against Table 1 of the paper
  3. quick run    the whole E1 grid (both datasets, methods and backbones) with
                  1 split x 1 seed and 60 SSL steps per unit, through scripts/launch.py
  4. outputs      result JSON, encoders, features, logs, scripts/aggregate.py
  5. estimate     duration of the full experiment, measured on this machine

Everything is written under artifacts/ (git-ignored) and deleted at the next check.
Takes a few minutes. Exit code 0 only if every check passed.

Example:
  python scripts/check_setup.py --gpus 2,3 --per-gpu 3     # GPUs chosen at launch time
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys

from PIL import Image

from sslhist import data as D
from sslhist.config import REPO_ROOT, load_config
from sslhist.io import exp_artifacts_dir, exp_results_dir, grid_units, result_path

FAILURES, WARNINGS = [], []


def ok(msg):
    print(f"  [ok]   {msg}")


def warn(msg):
    WARNINGS.append(msg)
    print(f"  [WARN] {msg}")


def fail(msg):
    FAILURES.append(msg)
    print(f"  [FAIL] {msg}")


def section(title):
    print(f"\n=== {title}", flush=True)


def check_environment(cfg, gpus, per_gpu, allow_cpu):
    section("1/5 Environment")
    import torch
    print(f"  python {sys.version.split()[0]} | torch {torch.__version__} (CUDA {torch.version.cuda})")
    if torch.cuda.is_available():
        names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        if len(names) == len(gpus.split(",")):
            ok(f"CUDA available, GPU(s) {gpus}: {', '.join(names)}")
        else:
            fail(f"--gpus {gpus} but only {len(names)} of them exist")
    elif allow_cpu:
        warn("CUDA not available: running on CPU (--allow-cpu)")
    else:
        fail(f"no usable GPU among --gpus {gpus}: check the ids (nvidia-smi), "
             f"the PyTorch build (README: cu124 wheel) and the driver")
    rt = cfg["runtime"]
    n_gpus = 1 if gpus == "cpu" else len(gpus.split(","))
    need = n_gpus * per_gpu * (int(rt["num_workers"]) + int(rt.get("torch_threads", 4)))
    cores = os.cpu_count() or 1
    msg = f"CPU: {cores} cores, the launcher will use about {need} ({n_gpus} GPU(s) x {per_gpu} units)"
    (ok if need <= cores else warn)(msg)

    if gpus == "cpu":
        return
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.used,memory.total,utilization.gpu",
                              "--format=csv,noheader,nounits", "-i", gpus],
                             capture_output=True, text=True, timeout=30)
        for line in out.stdout.strip().splitlines():
            idx, name, used, total, util = [x.strip() for x in line.split(",")]
            print(f"  GPU {idx} ({name}): {used}/{total} MiB used, {util}% util")
            if float(used) > 2000 or float(util) > 10:
                warn(f"GPU {idx} is already in use by another process")
        if out.returncode != 0:
            warn(f"nvidia-smi -i {gpus} failed: {out.stderr.strip()}")
    except (FileNotFoundError, subprocess.TimeoutExpired):
        warn("nvidia-smi not available: cannot check the load of the GPUs")



def check_data(cfg):
    section("2/5 Data")
    data = cfg["data"]
    paths_ok = True
    for key in ["pcam_root", "panda_train_csv", "panda_images_dir"]:
        p = data.get(key)
        if p and os.path.exists(p):
            ok(f"{key}: {p}")
        else:
            fail(f"{key} missing or not found: {p!r} (edit configs/paths.yaml)")
            paths_ok = False
    if not paths_ok:
        return
    prot = cfg["protocol"]
    for name in cfg["grid"]["datasets"]:
        try:
            ds = D.load_datasource(name, data)
        except Exception as e:  # noqa: BLE001
            fail(f"{name}: cannot load ({e})")
            continue
        labels = ds.labels_all()
        if len(ds) == 0:
            fail(f"{name}: no image found (check the image folder in configs/paths.yaml)")
            continue
        with Image.open(ds.paths[0]) as im:
            ok(f"{name}: universe n={len(ds)} class0/1={D.class_counts(labels)}, "
               f"first image {im.size[0]}x{im.size[1]} {im.mode}")
        ref = D.PAPER_TABLE1.get(name)
        if ref and float(data["sample_frac"][name]) == 0.2:
            tr, va = D.make_split(labels, 0, prot["val_ratio"], prot["split_base_seed"])
            got = {"universe": D.class_counts(labels), "train": D.class_counts(labels[tr]),
                   "val": D.class_counts(labels[va])}
            if got == ref:
                ok(f"{name}: class counts match Table 1 of the paper {ref['universe']}")
            else:
                fail(f"{name}: class counts {got} differ from Table 1 of the paper {ref} "
                     f"(different dataset version?)")


def quick_run(args, cfg):
    section("3/5 Quick run of the whole grid")
    if not cfg["experiment"].startswith("check"):
        sys.exit(f"Refusing to delete the outputs of experiment '{cfg['experiment']}': "
                 f"the check config must have an experiment name starting with 'check'.")
    shutil.rmtree(exp_results_dir(cfg), ignore_errors=True)
    shutil.rmtree(exp_artifacts_dir(cfg), ignore_errors=True)
    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "launch.py"), "--config", cfg["_config_path"],
           "--gpus", args.gpus, "--per-gpu", str(args.per_gpu)]
    if args.paths:
        cmd += ["--paths", args.paths]
    res = subprocess.run(cmd, cwd=REPO_ROOT)
    (ok if res.returncode == 0 else fail)(f"launch.py exit code {res.returncode}")


def units_of(cfg):
    return grid_units(cfg)


def check_outputs(args, cfg):
    section("4/5 Outputs")
    art = exp_artifacts_dir(cfg)
    for u in units_of(cfg):
        missing = [str(f) for f in cfg["protocol"]["label_fracs"] if not result_path(cfg, u, f).exists()]
        missing += [x for x in [f"encoders/{u.tag}.pt", f"features/{u.tag}.npz", f"logs/{u.tag}.json"]
                    if not (art / x).exists()]
        if missing:
            fail(f"{u.tag}: missing {', '.join(missing)}")
            log = art / "logs" / f"{u.tag}.log"
            if log.exists():
                print("         last lines of " + str(log) + ":")
                for line in log.read_text().splitlines()[-15:]:
                    print("         | " + line)
            continue
        rows = [json.load(open(result_path(cfg, u, f))) for f in cfg["protocol"]["label_fracs"]]
        bad = [r["label_frac"] for r in rows if not all(math.isfinite(r[m]) for m in ["auroc", "ece", "brier"])]
        if bad:
            fail(f"{u.tag}: non-finite metrics at label fractions {bad}")
        else:
            aurocs = ", ".join("{:g}%={:.3f}".format(r["label_frac"] * 100, r["auroc"]) for r in rows)
            ok(f"{u.tag}: AUROC at {aurocs}")
        if rows[0]["ssl_skipped_batches"]:
            warn(f"{u.tag}: {rows[0]['ssl_skipped_batches']} SSL batches skipped (non-finite loss)")

    cmd = [sys.executable, str(REPO_ROOT / "scripts" / "aggregate.py"), "--config", cfg["_config_path"]]
    if "resnet18" in cfg["grid"]["backbones"]:
        cmd += ["--reference", "resnet18"]
    res = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    out = exp_results_dir(cfg)
    figs = list((out / "figures").glob("*.pdf")) if (out / "figures").exists() else []
    if res.returncode == 0 and (out / "summary.csv").exists() and figs:
        ok(f"aggregate.py: summary tables and {len(figs)} figures in {out}")
    else:
        fail(f"aggregate.py failed (exit {res.returncode}):\n{res.stdout[-2000:]}\n{res.stderr[-2000:]}")


def estimate(args, cfg):
    section("5/5 Estimated duration of the full experiment")
    target = load_config(args.target, args.paths)
    n_runs = len(target["protocol"]["splits"]) * len(target["protocol"]["seeds"])
    epochs = int(target["protocol"]["ssl_epochs"])
    art = exp_artifacts_dir(cfg)
    per_unit, mems = [], []
    print(f"  {'unit (split 0, seed 0)':<42}{'s/step':>8}{'SSL min':>9}{'rest min':>9}{'GPU GB':>8}")
    for u in units_of(cfg):
        p = art / "logs" / f"{u.tag}.json"
        if not p.exists():
            continue
        lg = json.load(open(p))
        ssl_min = lg["ssl_history"]["sec_per_step"] * lg["steps_per_epoch"] * epochs / 60
        rest_min = (lg["total_time_s"] - lg["ssl_time_s"]) / 60
        per_unit += [ssl_min + rest_min] * n_runs
        mems.append(lg["peak_gpu_mem_gb"])
        name = f"{u.dataset}/{u.method}/{u.backbone}"
        print(f"  {name:<42}{lg['ssl_history']['sec_per_step']:>8.3f}{ssl_min:>9.1f}{rest_min:>9.1f}"
              f"{lg['peak_gpu_mem_gb']:>8.1f}")
    if not per_unit:
        warn("no timing available")
        return
    slots = len(args.gpus.split(",")) * args.per_gpu
    wall_h = max(sum(per_unit) / slots, max(per_unit)) / 60
    print(f"\n  {target['experiment']}: {len(per_unit)} units, {sum(per_unit) / 60:.1f} unit-hours; "
          f"with --gpus {args.gpus} --per-gpu {args.per_gpu} about {wall_h:.1f} h wall time.")
    print("  (Speeds were measured with the check units running side by side, so they include\n"
          "   the sharing of GPUs and CPU; treat the total as an order of magnitude.)")
    if mems and max(mems) * args.per_gpu > 80:
        warn(f"peak GPU memory {max(mems):.1f} GB x {args.per_gpu} units per GPU is close to the 94 GB of an H100")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/check.yaml", help="quick-check config")
    ap.add_argument("--target", default="configs/e1_vit_matched.yaml", help="experiment to estimate")
    ap.add_argument("--paths", default=None)
    ap.add_argument("--gpus", required=True, help="GPU ids as shown by nvidia-smi, e.g. 2,3 ('cpu' for tests)")
    ap.add_argument("--per-gpu", type=int, default=3)
    ap.add_argument("--allow-cpu", action="store_true", help="for tests without a GPU")
    args = ap.parse_args()

    # This process only looks at the chosen GPUs (numbered as in nvidia-smi); launch.py
    # then gives each unit one of them.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "" if args.gpus == "cpu" else args.gpus
    cfg = load_config(args.config, args.paths)
    check_environment(cfg, args.gpus, args.per_gpu, args.allow_cpu)
    check_data(cfg)
    if FAILURES:
        print("\nStopping before the quick run: fix the failures above first.")
        return 1
    quick_run(args, cfg)
    check_outputs(args, cfg)
    estimate(args, cfg)

    print("\n" + "=" * 70)
    if FAILURES:
        print(f"CHECK FAILED ({len(FAILURES)} problem(s)):")
        for f in FAILURES:
            print("  -", f.splitlines()[0])
        return 1
    print(f"ALL CHECKS PASSED ({len(WARNINGS)} warning(s)). To launch the full experiment:")
    print(f"  python scripts/launch.py --config {os.path.relpath(load_config(args.target)['_config_path'], REPO_ROOT)} "
          f"--gpus {args.gpus} --per-gpu {args.per_gpu}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
