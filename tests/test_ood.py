"""E6 (OOD analysis) on tiny fake data, CPU: scores, temperature scaling, OOD scoring of
saved encoders, supervised baselines, launcher, status and aggregation."""
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import yaml

from conftest import REPO
from sslhist.config import load_config
from sslhist.io import Unit, exp_artifacts_dir, exp_results_dir, grid_units, result_path
from sslhist.runner import run_unit


def test_temperature_scaling_and_scores():
    from sslhist.ood import detection, fit_temperature, score_fraction
    rng = np.random.default_rng(0)
    z = rng.normal(0, 2, 20000)
    y = (rng.random(20000) < 1 / (1 + np.exp(-z))).astype(int)
    assert abs(fit_temperature(3 * z, y) - 3) < 0.15      # over-confident logits -> T ~ 3
    assert abs(fit_temperature(z, y) - 1) < 0.05

    assert detection(np.zeros(100), np.ones(100)) == (1.0, 0.0)   # OOD always more uncertain
    auroc, fpr = detection(np.ones(100), np.zeros(100))           # OOD always more confident: not flipped
    assert auroc == 0.0 and fpr == 1.0

    ood_z = rng.normal(0, 8, 500)                                  # very confident on OOD
    s = score_fraction(3 * z[:2000], y[:2000], ood_z, np.arange(0, 2000, 2), np.arange(1, 2000, 2))
    assert s["temperature"] > 2 and s["ood_msp_ts"] < s["ood_msp_raw"] and s["id_ece_ts"] < s["id_ece_raw"]
    assert s["ood_auroc_raw"] < 0.5 and 0 <= s["ood_conf90_ts"] <= s["ood_conf90_raw"] <= 1
    assert 0 <= s["ood_entropy_raw"] <= s["ood_entropy_ts"] <= 1


def test_grid_units_series_and_product(tiny_config):
    cfg = load_config(str(tiny_config))
    assert len(grid_units(cfg)) == 1 * 1 * 2 * 3 * 3
    assert {u.method for u in grid_units(cfg, methods=["byol"])} == {"byol"}  # replaces the config methods
    cfg["grid"] = {"datasets": ["pcam"], "series": [["simclr", "resnet18"], ["frozen", "dinov2_tiny_test"]]}
    units = grid_units(cfg, splits=[0], seeds=[0])
    assert [(u.method, u.backbone) for u in units] == [("simclr", "resnet18"), ("frozen", "dinov2_tiny_test")]
    assert [u.method for u in grid_units(cfg, methods=["frozen"])] == ["frozen"] * 9  # filters the series


def test_sup_imagenet_uses_pretrained_weights(monkeypatch):
    import torch
    from torch.utils.data import TensorDataset
    from sslhist import models, supervised
    calls = []

    def fake_build(name, pretrained=False):
        calls.append((name, pretrained))
        return models.build_backbone("vit_tiny_test")

    monkeypatch.setattr(supervised, "build_backbone", fake_build)
    ds = TensorDataset(torch.randn(8, 3, 32, 32), torch.tensor([0, 1] * 4))
    model = supervised.train_supervised("resnet18", True, ds, epochs=1, lr=1e-3, weight_decay=1e-4, batch_size=4,
                                        num_workers=0, device=torch.device("cpu"), amp=False, log=lambda m: None)
    assert calls == [("resnet18", True)] and isinstance(model, supervised.SupervisedNet)


def test_status_phases_of_e6_units():
    import importlib.util
    spec = importlib.util.spec_from_file_location("status", REPO / "scripts" / "status.py")
    status = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(status)
    head = "[RUN] pcam__simclr__resnet18__split0__seed0 img=96\n[E6] ID=pcam OOD=panda\n"
    assert status.parse_log(head)["phase"] == "OOD scoring"
    sup = head + "  [SUP] frac=0.01 n_lab=352\n[SUP] ep 1/10 loss=0.6 time=2.0s\n  [SUP] frac=0.05 n_lab=1760\n"
    assert status.parse_log(sup)["phase"] == "supervised, label fraction 2"


@pytest.fixture
def e6_setup(tmp_path, tiny_config):
    """Source experiment with SSL and frozen encoders (as E1-E3), and an E6 config reading it."""
    base = yaml.safe_load(tiny_config.read_text())
    src = dict(base, experiment="src_ssl")
    src["protocol"] = dict(base["protocol"], splits=[0], seeds=[0, 1])
    src_path = tmp_path / "src_ssl.yaml"
    src_path.write_text(yaml.safe_dump(src))
    src_cfg = load_config(str(src_path))
    for ds in ["pcam", "panda"]:
        for method, backbone in [("simclr", "resnet18"), ("frozen", "dinov2_tiny_test")]:
            for seed in (0, 1):
                run_unit(src_cfg, Unit(ds, method, backbone, 0, seed))
    # E1/E2 result files were written before the "encoder" field existed
    for p in (tmp_path / "results" / "src_ssl" / "raw").glob("*__simclr__*.json"):
        row = json.loads(p.read_text())
        del row["encoder"]
        p.write_text(json.dumps(row))

    e6 = dict(base, experiment="e6_test", kind="ood")
    e6["protocol"] = dict(src["protocol"])
    e6["ood"] = {"source_experiments": ["missing_exp", "src_ssl"], "pairs": {"pcam": "panda", "panda": "pcam"},
                 "n_samples": 60, "sample_seed": 123, "calib_frac": 0.5, "calib_seed_base": 2000,
                 "knn_k": 5, "knn_bank": 80, "knn_seed": 0, "encoder_check_n": 16}
    e6["supervised"] = {"epochs": 1, "lr": 1.0e-3, "weight_decay": 1.0e-4, "unscale_before_clip": True}
    e6["grid"] = {"datasets": ["pcam", "panda"],
                  "series": [["simclr", "resnet18"], ["frozen", "dinov2_tiny_test"], ["sup_scratch", "resnet18"]]}
    e6_path = tmp_path / "e6_test.yaml"
    e6_path.write_text(yaml.safe_dump(e6))
    return e6_path


def test_e6_end_to_end(e6_setup, subprocess_env):
    run = lambda *a: subprocess.run([sys.executable, *a], cwd=REPO, env=subprocess_env,  # noqa: E731
                                    capture_output=True, text=True)
    res = run("scripts/launch.py", "--config", str(e6_setup), "--gpus", "cpu", "--per-gpu", "4")
    assert res.returncode == 0, res.stdout + res.stderr
    cfg = load_config(str(e6_setup))
    units = grid_units(cfg)
    assert len(units) == 12

    for u in units:
        for frac in cfg["protocol"]["label_fracs"]:
            row = json.loads(result_path(cfg, u, frac).read_text())
            assert row["ood_dataset"] == {"pcam": "panda", "panda": "pcam"}[u.dataset]
            for key in ["ood_auroc_raw", "knn_ood_auroc", "id_msp_raw", "ood_msp_raw", "id_msp_ts", "ood_msp_ts",
                        "id_entropy_raw", "ood_entropy_ts", "id_ece_raw", "id_ece_ts", "auroc", "ece"]:
                assert 0.0 <= row[key] <= 1.0, (u.tag, key, row[key])
            for key in ["id_msp_raw", "ood_msp_raw", "id_msp_ts", "ood_msp_ts"]:
                assert row[key] >= 0.5
            assert row["temperature"] > 0 and row["n_cal"] + row["n_eval"] == row["n_val"]
            assert row["n_ood"] == 60
            if u.method == "sup_scratch":
                assert row["source_experiment"] == "trained here" and row["encoder"] == "supervised from scratch"
            else:
                assert row["source_experiment"] == "src_ssl"
                assert row["encoder"] == {"simclr": "SSL from scratch", "frozen": "pretrained, frozen"}[u.method]
                assert row["encoder_check_rel_diff"] < 0.05 and row["probe_check_max_abs_diff"] < 1e-4
                # same labelled subset and validation set as the source run
                src = json.loads((exp_results_dir(cfg).parent / "src_ssl" / "raw" /
                                  result_path(cfg, u, frac).name).read_text())
                assert (row["subset_ids_fp"], row["val_ids_fp"], row["auroc"]) == \
                    (src["subset_ids_fp"], src["val_ids_fp"], src["auroc"])
    assert (exp_artifacts_dir(cfg) / "scores" / f"{units[0].tag}.npz").exists()

    status = run("scripts/status.py", "--config", str(e6_setup))
    assert status.returncode == 0 and "complete 12/12 units (36 result files)" in status.stdout, status.stdout

    agg = run("scripts/aggregate.py", "--config", str(e6_setup), "--reference", "resnet18:sup_scratch")
    assert agg.returncode == 0, agg.stdout + agg.stderr
    out = exp_results_dir(cfg)
    summary = pd.read_csv(out / "summary.csv")
    assert len(summary) == 2 * 3 * 3 and (summary["n_runs"] == 2).all()
    paired = pd.read_csv(out / "paired_vs_resnet18-sup_scratch.csv")
    assert set(paired["method"]) == {"simclr", "frozen"} and (paired["n_pairs"] == 2).all()
    assert "ood_msp_ts_diff_mean" in paired.columns
    assert "OOD detection" in (out / "summary.txt").read_text()
    assert "\\toprule" in (out / "summary.tex").read_text() and (out / "summary_detection.tex").exists()
    for ds in ["pcam", "panda"]:
        for name in ["ood_msp_raw", "ood_msp_ts", "knn_ood_auroc", "confidence_bars", "msp_hist"]:
            assert (out / "figures" / f"e6_test_{ds}_{name}.pdf").stat().st_size > 1000


def test_e6_unit_without_saved_encoder_fails_clearly(e6_setup):
    from sslhist.ood import run_ood_unit
    cfg = load_config(str(e6_setup))
    with pytest.raises(FileNotFoundError, match="No saved features"):
        run_ood_unit(cfg, Unit("pcam", "byol", "resnet18", 0, 0))
