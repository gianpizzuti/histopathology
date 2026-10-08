#!/usr/bin/env python
"""Run a grid of units on a few GPUs, several units per GPU at the same time.

Each unit runs in its own process with ``CUDA_VISIBLE_DEVICES`` set to one of the
GPUs given with --gpus (ids as in nvidia-smi); no other GPU is touched.
Units whose result files already exist are skipped, so the launcher can be
stopped and started again at any time. Logs go to artifacts/<experiment>/logs/.

Examples:
  python scripts/launch.py --config configs/e1_vit_matched.yaml --gpus 2,3 --per-gpu 3 --dry-run
  nohup python scripts/launch.py --config configs/e1_vit_matched.yaml --gpus 2,3 --per-gpu 3 > launch.log 2>&1 &
"""
import argparse
import itertools
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from sslhist.config import REPO_ROOT, load_config
from sslhist.io import Unit, exp_artifacts_dir, unit_complete

# Relative cost, only used to start the longest units first.
_COST = {"pcam": 20, "panda": 1, "vit_b_16": 8, "vit_s_16": 3, "resnet50": 3, "resnet18": 1,
         "dinov2_vitb14": 1, "dinov2_vits14": 0.5, "byol": 1.3, "simclr": 1.0, "barlow": 1.0, "frozen": 0.1}


def cost(u: Unit) -> float:
    return _COST.get(u.dataset, 1) * _COST.get(u.backbone, 1) * _COST.get(u.method, 1)


def csv_list(s, cast=str):
    return [cast(x) for x in s.split(",")] if s else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--paths", default=None)
    ap.add_argument("--datasets", help="comma list (default: grid in config)")
    ap.add_argument("--methods")
    ap.add_argument("--backbones")
    ap.add_argument("--splits", help="comma list of ints (default: protocol.splits)")
    ap.add_argument("--seeds", help="comma list of ints (default: protocol.seeds)")
    ap.add_argument("--gpus", required=True,
                    help="comma list of GPU ids as shown by nvidia-smi, e.g. 2,3 ('cpu': no GPU, for tests)")
    ap.add_argument("--per-gpu", type=int, default=1, help="units running at the same time on each GPU")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, args.paths)
    grid, prot = cfg.get("grid", {}), cfg["protocol"]
    units = [Unit(*u) for u in itertools.product(
        csv_list(args.datasets) or grid["datasets"],
        csv_list(args.methods) or grid["methods"],
        csv_list(args.backbones) or grid["backbones"],
        csv_list(args.splits, int) or prot["splits"],
        csv_list(args.seeds, int) or prot["seeds"],
    )]
    done = [u for u in units if unit_complete(cfg, u)] if not args.force else []
    todo = sorted([u for u in units if u not in done], key=cost, reverse=True)
    print(f"[launch] experiment={cfg['experiment']} units={len(units)} complete={len(done)} to run={len(todo)}")
    if args.dry_run:
        for u in todo:
            print("  ", u.tag)
        return 0

    log_dir = exp_artifacts_dir(cfg) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    gpu_ids = [""] if args.gpus == "cpu" else csv_list(args.gpus)  # "" hides every GPU
    free_slots = [g for g in gpu_ids for _ in range(args.per_gpu)]
    threads = str(cfg["runtime"].get("torch_threads", 4))
    running, failed, ok = {}, [], []

    def stop_all(*_):
        for p in running:
            p.terminate()
        sys.exit(130)

    signal.signal(signal.SIGINT, stop_all)
    signal.signal(signal.SIGTERM, stop_all)

    t_start = time.time()
    while todo or running:
        while todo and free_slots:
            u, gpu = todo.pop(0), free_slots.pop(0)
            cmd = [sys.executable, str(REPO_ROOT / "scripts" / "run_unit.py"), "--config", cfg["_config_path"],
                   "--dataset", u.dataset, "--method", u.method, "--backbone", u.backbone,
                   "--split", str(u.split), "--seed", str(u.seed)]
            if args.paths:
                cmd += ["--paths", args.paths]
            if args.force:
                cmd.append("--force")
            # PCI_BUS_ID: GPU ids are the same as in nvidia-smi (CUDA's default order may differ)
            env = {**os.environ, "CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": gpu,
                   "OMP_NUM_THREADS": threads, "MKL_NUM_THREADS": threads,
                   "OPENBLAS_NUM_THREADS": threads, "PYTHONUNBUFFERED": "1"}
            logf = open(log_dir / f"{u.tag}.log", "a")
            p = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, env=env, cwd=REPO_ROOT)
            running[p] = (u, gpu, time.time(), logf)
            print(f"[launch] start {u.tag} on {'GPU ' + gpu if gpu else 'CPU'} (pid {p.pid})", flush=True)

        time.sleep(5)
        for p in [p for p in running if p.poll() is not None]:
            u, gpu, t0, logf = running.pop(p)
            logf.close()
            free_slots.append(gpu)
            mins = (time.time() - t0) / 60
            if p.returncode == 0:
                ok.append(u)
                total = len(ok) + len(failed) + len(todo) + len(running)
                print(f"[launch] done  {u.tag} in {mins:.1f} min | completed {len(ok)}/{total} "
                      f"| waiting={len(todo)} running={len(running)}", flush=True)
            else:
                failed.append(u)
                print(f"[launch] FAIL  {u.tag} (exit {p.returncode}) see {log_dir / (u.tag + '.log')}", flush=True)

    print(f"[launch] finished in {(time.time() - t_start) / 3600:.2f} h: ok={len(ok)} failed={len(failed)}")
    for u in failed:
        print("   failed:", u.tag)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
