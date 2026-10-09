"""E4 (federated SimCLR) on tiny fake data, CPU: partitions, FedAvg, a federated unit,
the notebook-08 control and the aggregation with the centralised reference."""
import json
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from conftest import REPO
from sslhist.config import load_config
from sslhist.federated import fed_label, fedavg, parse_fed_method, partition_clients, partition_stats
from sslhist.io import Unit, exp_artifacts_dir, exp_results_dir, result_path
from sslhist.runner import run_unit


def test_parse_federated_method_names():
    assert parse_fed_method("simclr-fed-k10-a0.1") == {"ssl": "simclr", "clients": 10, "partition": "dirichlet",
                                                       "alpha": 0.1}
    assert parse_fed_method("byol-fed-k5-iid") == {"ssl": "byol", "clients": 5, "partition": "iid", "alpha": None}
    for name in ["simclr", "frozen", "simclr-fed-k10", "simclr-fed-kx-iid", "mae-fed-k5-iid"]:
        assert parse_fed_method(name) is None
    assert fed_label("simclr-fed-k10-a0.5") == "FED-SIMCLR (10 clients, Dir α=0.5)"


@pytest.mark.parametrize("k", [5, 10])
def test_partitions_cover_all_images_with_equal_sizes_and_growing_skew(k):
    rng = np.random.default_rng(0)
    y = (rng.random(3001) < 0.4).astype(int)
    skew = {}
    for part, alpha in [("iid", None), ("dirichlet", 0.5), ("dirichlet", 0.1)]:
        values = []
        for seed in range(10):
            parts = partition_clients(y, k, part, alpha, seed)
            allidx = np.concatenate(parts)
            assert len(allidx) == len(y) and len(np.unique(allidx)) == len(y)       # disjoint, complete
            assert max(map(len, parts)) - min(map(len, parts)) <= 1                 # equal sizes
            values.append(partition_stats(y, parts)["label_skew"])
        skew[alpha] = np.mean(values)
    assert skew[None] < 0.05 < skew[0.5] < skew[0.1]
    assert np.array_equal(np.concatenate(partition_clients(y, k, "dirichlet", 0.1, 3)),
                          np.concatenate(partition_clients(y, k, "dirichlet", 0.1, 3)))  # reproducible


def test_fedavg_weights_clients_by_size():
    a = {"w": torch.tensor([1.0, 2.0]), "n": torch.tensor(5)}
    b = {"w": torch.tensor([3.0, 6.0]), "n": torch.tensor(7)}
    avg = fedavg([a, b], [1, 3])
    assert torch.allclose(avg["w"], torch.tensor([2.5, 5.0])) and avg["n"].item() == 5


@pytest.fixture
def e4_setup(tmp_path, tiny_config):
    """A centralised SimCLR run (as E1) and a federated config on the same splits."""
    base = yaml.safe_load(tiny_config.read_text())
    prot = dict(base["protocol"], splits=[0], seeds=[0, 1])
    central = dict(base, experiment="central", protocol=prot,
                   grid={"datasets": ["pcam"], "methods": ["simclr", "byol"], "backbones": ["resnet18"]})
    fed = dict(base, experiment="fed", protocol=prot, federated={"local_epochs": 1},
               grid={"datasets": ["pcam"], "backbones": ["resnet18"],
                     "methods": ["simclr-fed-k3-iid", "simclr-fed-k3-a0.1"]})
    paths = {}
    for name, c in [("central", central), ("fed", fed)]:
        paths[name] = tmp_path / f"{name}.yaml"
        paths[name].write_text(yaml.safe_dump(c))
    ccfg = load_config(str(paths["central"]))
    for seed in (0, 1):
        run_unit(ccfg, Unit("pcam", "simclr", "resnet18", 0, seed))
    return paths


def test_federated_units_aggregate_and_control(e4_setup, subprocess_env):
    run = lambda *a: subprocess.run([sys.executable, *a], cwd=REPO, env=subprocess_env,  # noqa: E731
                                    capture_output=True, text=True)
    res = run("scripts/launch.py", "--config", str(e4_setup["fed"]), "--gpus", "cpu", "--per-gpu", "2")
    assert res.returncode == 0, res.stdout + res.stderr
    cfg = load_config(str(e4_setup["fed"]))
    for method in ["simclr-fed-k3-iid", "simclr-fed-k3-a0.1"]:
        unit = Unit("pcam", method, "resnet18", 0, 0)
        row = json.loads(result_path(cfg, unit, 0.01).read_text())
        assert row["n_clients"] == 3 and row["rounds"] == 1 and row["encoder"] == "federated SSL from scratch"
        assert sum(row["client_sizes"]) == row["n_train"] and row["ssl_steps"] > 0
        assert row["partition"] == ("iid" if method.endswith("iid") else "dirichlet")
        assert (exp_artifacts_dir(cfg) / "encoders" / f"{unit.tag}.pt").exists()
        assert "] ep 1/1 loss=" in (exp_artifacts_dir(cfg) / "logs" / f"{unit.tag}.log").read_text()

    status = run("scripts/status.py", "--config", str(e4_setup["fed"]))
    assert "complete 4/4 units" in status.stdout, status.stdout

    # paired with the centralised run: only its SimCLR series is taken
    agg = run("scripts/aggregate.py", "--config", str(e4_setup["fed"]), "--include", "central:resnet18:simclr",
              "--reference", "resnet18:simclr")
    assert agg.returncode == 0, agg.stdout + agg.stderr
    out = exp_results_dir(cfg) / "with_central-resnet18-simclr"
    summary = pd.read_csv(out / "summary.csv")
    assert set(summary["method"]) == {"simclr", "simclr-fed-k3-iid", "simclr-fed-k3-a0.1"}
    paired = pd.read_csv(out / "paired_vs_resnet18-simclr.csv")
    assert set(paired["method"]) == {"simclr-fed-k3-iid", "simclr-fed-k3-a0.1"} and (paired["n_pairs"] == 2).all()
    assert "Fed-SimCLR (3 clients, Dir $\\alpha$=0.1)" in (out / "summary.tex").read_text()
    assert (out / "figures" / "fed_central-resnet18-simclr_pcam_ece.pdf").exists()

    ctrl = run("scripts/legacy08_control.py", "--config", str(e4_setup["fed"]),
               "--sources", "central:resnet18:simclr", "fed")
    assert ctrl.returncode == 0, ctrl.stdout + ctrl.stderr
    raw = pd.read_csv(exp_results_dir(cfg) / "legacy08_control" / "raw.csv")
    assert len(raw) == 3 * 2 * 3  # 3 series x 2 seeds x 3 label fractions
    for c in ["ours_auroc", "ours_ece", "ours_ece_toplabel", "nb08_auroc", "nb08_ece", "nb08_ece_toplabel"]:
        assert raw[c].between(0, 1).all(), c
    assert "notebook 08" in (exp_results_dir(cfg) / "legacy08_control" / "summary.txt").read_text()


def test_run_unit_rejects_unknown_methods(subprocess_env):
    res = subprocess.run([sys.executable, "scripts/run_unit.py", "--gpu", "cpu", "--config", "configs/e4_federated.yaml",
                          "--dataset", "pcam", "--method", "simclr-fed-k5", "--backbone", "resnet18",
                          "--split", "0", "--seed", "0"], cwd=REPO, env=subprocess_env, capture_output=True, text=True)
    assert res.returncode == 2 and "unknown method" in res.stderr
