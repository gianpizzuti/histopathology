"""Probe robustness analysis on tiny fake data (CPU)."""
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score

from conftest import REPO
from sslhist.config import load_config
from sslhist.io import Unit, result_path
from sslhist.ood import run_ood_unit
from sslhist.probe_variants import FITTERS, standardizer
from sslhist.runner import run_unit


def test_probe_variants_learn_a_separable_task():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2000, 32)).astype(np.float32)
    y = (x[:, 0] + 0.3 * rng.normal(size=2000) > 0).astype(int)
    for name, fit in FITTERS.items():
        m = fit(x[:200], y[:200], seed=1)
        assert roc_auc_score(y[1000:], m["logits"](x[1000:])) > 0.9, name
    assert "C" in FITTERS["logreg_cv"](x[:60], y[:60], seed=1)
    std = standardizer(x[:60])
    assert np.allclose(std(x[:60]).std(axis=0, ddof=1), 1, atol=1e-4)


def test_probe_robustness_end_to_end(tmp_path, tiny_config, subprocess_env):
    base = yaml.safe_load(tiny_config.read_text())
    prot = dict(base["protocol"], splits=[0], seeds=[0, 1])
    src = dict(base, experiment="src", protocol=prot)
    e6 = dict(base, experiment="e6", kind="ood", protocol=prot,
              ood={"source_experiments": ["src"], "pairs": {"pcam": "panda", "panda": "pcam"}, "n_samples": 40,
                   "sample_seed": 123, "calib_frac": 0.5, "calib_seed_base": 2000, "knn_k": 5, "knn_bank": 50,
                   "knn_seed": 0, "encoder_check_n": 8},
              supervised={"epochs": 1, "lr": 1.0e-3, "weight_decay": 1.0e-4},
              grid={"datasets": ["pcam"], "series": [["simclr", "resnet18"], ["sup_scratch", "resnet18"]]})
    paths = {}
    for name, c in [("src", src), ("e6", e6)]:
        paths[name] = tmp_path / f"{name}.yaml"
        paths[name].write_text(yaml.safe_dump(c))
    src_cfg, e6_cfg = load_config(str(paths["src"])), load_config(str(paths["e6"]))
    for seed in (0, 1):
        run_unit(src_cfg, Unit("pcam", "simclr", "resnet18", 0, seed))
        run_ood_unit(e6_cfg, Unit("pcam", "simclr", "resnet18", 0, seed))
        run_ood_unit(e6_cfg, Unit("pcam", "sup_scratch", "resnet18", 0, seed))

    res = subprocess.run([sys.executable, "scripts/probe_robustness.py", "--config", str(paths["e6"]),
                          "--experiments", "src", "--ood-experiment", "e6", "--workers", "2"],
                         cwd=REPO, env=subprocess_env, capture_output=True, text=True)
    assert res.returncode == 0, res.stdout[-3000:] + res.stderr[-3000:]
    assert "max |AUROC difference| = 0.00e+00" in res.stdout  # legacy probe = result files
    out = tmp_path / "results" / "probe_robustness"
    raw = pd.read_csv(out / "raw.csv")
    assert len(raw) == 2 * 3 * 2 and set(raw.probe) == {"legacy", "logreg_cv"}
    for m in ["auroc", "ece", "brier"]:
        assert raw[m].between(0, 1).all()
    assert raw[raw.probe == "logreg_cv"]["C"].notna().all()

    e6_raw = pd.read_csv(out / "e6_raw.csv")
    assert set(e6_raw.probe) == {"legacy", "logreg_cv", "end-to-end"}
    # with the legacy probe, the E6 scores are those of E6 itself
    leg = e6_raw[(e6_raw.probe == "legacy") & (e6_raw.seed == 1) & (e6_raw.label_frac == 0.05)].iloc[0]
    e6_row = json.loads(result_path(e6_cfg, Unit("pcam", "simclr", "resnet18", 0, 1), 0.05).read_text())
    for m in ["ood_auroc_raw", "ood_msp_raw", "ood_msp_ts", "temperature", "id_ece_ts"]:
        assert abs(leg[m] - e6_row[m]) < 1e-6, m
    assert (out / "paired_vs_legacy.csv").exists() and "== ood_msp_ts" in (out / "e6_summary.txt").read_text()
    assert (out / "figures" / "probe_logreg_cv_pcam_auroc.pdf").stat().st_size > 1000
