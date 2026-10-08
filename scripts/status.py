#!/usr/bin/env python
"""Progress of an experiment launched with scripts/launch.py: units completed,
running (with their current epoch), failed and pending, and an estimate of the
time left. It only reads files, so it can run at any time without disturbing
the experiment.

Examples:
  python scripts/status.py --config configs/e1_vit_matched.yaml
  watch -n 60 python scripts/status.py --config configs/e1_vit_matched.yaml   # refresh every minute
"""
import argparse
import itertools
import json
import re
import sys
import time

from sslhist.config import load_config, resolve
from sslhist.io import Unit, exp_artifacts_dir, exp_results_dir, unit_complete

EPOCH_RE = re.compile(r"\] ep (\d+)/(\d+) loss=\S+ time=([\d.]+)s")


def parse_log(text: str) -> dict:
    """State of a unit from its log; only the part after the last [RUN] counts
    (launch.py appends to the same log when a unit is started again)."""
    last = text.rfind("[RUN]")
    if last < 0:
        return {"state": "starting"}
    seg = text[last:]
    if "Traceback" in seg:
        return {"state": "failed", "error": [l for l in seg.splitlines() if l.strip()][-1][:120]}
    epochs = [(int(a), int(b), float(t)) for a, b, t in EPOCH_RE.findall(seg)]
    info = {"state": "running", "epoch": 0, "epochs": None, "epoch_times": [t for _, _, t in epochs]}
    if epochs:
        info["epoch"], info["epochs"] = epochs[-1][0], epochs[-1][1]
    if "[PROBE]" in seg or (epochs and epochs[-1][0] == epochs[-1][1]):
        info["phase"] = "probes / saving"
    elif "[FROZEN]" in seg:
        info["phase"] = "features (frozen)"
    else:
        info["phase"] = "SSL pretraining"
    return info


def fmt_min(m: float) -> str:
    return f"{m / 60:.1f} h" if m >= 90 else f"{m:.0f} min"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--slots", type=int, default=None,
                    help="units running in parallel (default: as many as are running now)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    grid, prot = cfg["grid"], cfg["protocol"]
    units = [Unit(*u) for u in itertools.product(grid["datasets"], grid["methods"], grid["backbones"],
                                                 prot["splits"], prot["seeds"])]
    logs = exp_artifacts_dir(cfg) / "logs"
    n_epochs = int(prot["ssl_epochs"])

    # Durations of completed units, per kind of unit, for the estimate
    done_min = {}
    for u in units:
        p = logs / f"{u.tag}.json"
        if unit_complete(cfg, u) and p.exists():
            done_min.setdefault((u.dataset, u.method, u.backbone), []).append(json.load(open(p))["total_time_s"] / 60)
    # Fallback for kinds not completed yet: speeds measured by scripts/check_setup.py
    check_min = {}
    for p in (resolve(cfg["artifacts_dir"]) / "check" / "logs").glob("*__split0__seed0.json"):
        lg = json.load(open(p))
        kind = tuple(lg["unit"].split("__")[:3])
        check_min[kind] = (lg["ssl_history"]["sec_per_step"] * lg["steps_per_epoch"] * n_epochs
                           + lg["total_time_s"] - lg["ssl_time_s"]) / 60

    complete, running, failed, pending = [], [], [], []
    for u in units:
        if unit_complete(cfg, u):
            complete.append(u)
            continue
        p = logs / f"{u.tag}.log"
        if not p.exists():
            pending.append(u)
            continue
        info = parse_log(p.read_text(errors="replace"))
        info["idle_min"] = (time.time() - p.stat().st_mtime) / 60
        (failed if info["state"] == "failed" else running).append((u, info))

    n_res = sum(1 for _ in (exp_results_dir(cfg) / "raw").glob("*.json")) if (exp_results_dir(cfg) / "raw").exists() else 0
    print(f"{cfg['experiment']}  {time.strftime('%Y-%m-%d %H:%M')}")
    print(f"  complete {len(complete)}/{len(units)} units ({n_res} result files) | running {len(running)} | "
          f"failed {len(failed)} | pending {len(pending)}")

    remaining = 0.0
    unknown = 0
    if running:
        print("\n  running:")
    for u, info in sorted(running, key=lambda x: x[0].tag):
        ep_t = info["epoch_times"]
        per_ep = sum(ep_t) / len(ep_t) / 60 if ep_t else None
        kind = done_min.get((u.dataset, u.method, u.backbone))
        if per_ep is not None:
            left = (n_epochs - info["epoch"]) * per_ep + (sum(kind) / len(kind) - n_epochs * per_ep if kind else 1.0)
            left = max(left, 0.5)
            remaining += left
            eta = f"~{fmt_min(left)} left"
        elif info["phase"] != "SSL pretraining":  # frozen encoder or probes: a few minutes at most
            remaining += 2.0
            eta = ""
        else:
            unknown += 1
            eta = "first epoch in progress"
        stale = f"  [no log output for {info['idle_min']:.0f} min]" if info["idle_min"] > 15 else ""
        print(f"    {u.tag:<45} epoch {info['epoch']}/{n_epochs}  {info['phase']:<16} {eta}{stale}")

    for u in pending:
        kind = done_min.get((u.dataset, u.method, u.backbone))
        if kind:
            remaining += sum(kind) / len(kind)
        elif (u.dataset, u.method, u.backbone) in check_min:
            remaining += check_min[(u.dataset, u.method, u.backbone)]
        else:
            # same kind of unit still running: use its epoch time
            same = [i for v, i in running if (v.dataset, v.method, v.backbone) == (u.dataset, u.method, u.backbone)
                    and i["epoch_times"]]
            if same:
                remaining += sum(same[0]["epoch_times"]) / len(same[0]["epoch_times"]) / 60 * n_epochs + 1.0
            else:
                unknown += 1

    if failed:
        print("\n  FAILED (re-run the launch command to retry them; see artifacts/<experiment>/logs/<unit>.log):")
        for u, info in failed:
            print(f"    {u.tag:<45} {info['error']}")

    if running or pending:
        slots = args.slots or max(1, len(running))
        msg = f"\n  estimated time left: ~{fmt_min(remaining / slots)} with {slots} units in parallel"
        if unknown:
            msg += f" ({unknown} units not estimated yet: no timing for their kind so far)"
        print(msg)
    elif not failed:
        print("\n  all units complete: run scripts/aggregate.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
