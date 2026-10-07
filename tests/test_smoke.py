"""End-to-end runs on tiny fake data (CPU): pretraining, probes, result files,
resume, launcher and aggregation."""
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from conftest import REPO
from sslhist.config import load_config
from sslhist.io import Unit, exp_artifacts_dir, exp_results_dir, result_path
from sslhist.metrics import METRICS
from sslhist.runner import run_unit


@pytest.mark.parametrize("unit", [
    Unit("pcam", "simclr", "vit_tiny_test", 0, 0),
    Unit("pcam", "byol", "resnet18", 1, 2),
    Unit("panda", "byol", "vit_tiny_test", 2, 1),
])
def test_run_unit_writes_results_and_artifacts(tiny_config, unit):
    cfg = load_config(str(tiny_config))
    run_unit(cfg, unit)

    for frac in cfg["protocol"]["label_fracs"]:
        with open(result_path(cfg, unit, frac)) as f:
            row = json.load(f)
        for key in ["split", "seed", "method", "backbone", "label_frac", *METRICS]:
            assert key in row
        assert (row["split"], row["seed"], row["label_frac"]) == (unit.split, unit.seed, frac)
        for m in ["accuracy", "f1", "ece", "brier"]:
            assert 0.0 <= row[m] <= 1.0
        assert row["n_labeled"] >= 1

    art = exp_artifacts_dir(cfg)
    assert (art / "encoders" / f"{unit.tag}.pt").exists()
    feats = np.load(art / "features" / f"{unit.tag}.npz")
    assert feats["train_feats"].shape[0] == len(feats["train_idx"])
    assert feats["val_feats"].shape[0] == len(feats["val_idx"])
    assert set(feats["train_idx"]).isdisjoint(feats["val_idx"])
    assert len(feats["frac01_val_logits"]) == len(feats["val_idx"])

    # Resume: a complete unit is skipped
    assert run_unit(cfg, unit) is None


def test_timing_mode_writes_nothing(tiny_config):
    cfg = load_config(str(tiny_config))
    unit = Unit("pcam", "simclr", "resnet18", 0, 0)
    est = run_unit(cfg, unit, timing_steps=2)
    assert est["steps_measured"] == 2 and est["est_ssl_total_min"] > 0
    assert not exp_results_dir(cfg).exists()


def test_launcher_and_aggregate(tiny_config, subprocess_env):
    run = lambda *a: subprocess.run([sys.executable, *a], cwd=REPO, env=subprocess_env,  # noqa: E731
                                    capture_output=True, text=True)
    common = ["--config", str(tiny_config), "--splits", "0,1", "--seeds", "0"]

    dry = run("scripts/launch.py", *common, "--dry-run")
    assert dry.returncode == 0, dry.stderr
    assert "to run=4" in dry.stdout

    res = run("scripts/launch.py", *common, "--gpus", "0", "--per-gpu", "2")
    assert res.returncode == 0, res.stdout + res.stderr
    again = run("scripts/launch.py", *common, "--dry-run")
    assert "to run=0" in again.stdout

    agg = run("scripts/aggregate.py", "--config", str(tiny_config), "--reference", "resnet18")
    assert agg.returncode == 0, agg.stdout + agg.stderr

    cfg = load_config(str(tiny_config))
    out = exp_results_dir(cfg)
    summary = pd.read_csv(out / "summary.csv")
    assert len(summary) == 2 * 3  # 2 backbones x 3 label fractions
    assert (summary["n_runs"] == 2).all()
    paired = pd.read_csv(out / "paired_vs_resnet18.csv")
    assert set(paired["backbone"]) == {"vit_tiny_test"} and (paired["n_pairs"] == 2).all()
    assert "\\toprule" in (out / "summary.tex").read_text()
    for m in METRICS:
        assert (out / "figures" / f"smoke_panda_{m}.pdf").stat().st_size > 1000
