"""The new package must reproduce the conference protocol exactly: same
universe, same splits, same labelled subsets, same metric values."""
import numpy as np
import pytest

import legacy_reference as L
from sslhist import data as D
from sslhist.metrics import compute_metrics_binary


@pytest.mark.parametrize("frac", [0.2, 0.5, 1.0])
def test_pcam_universe_matches_legacy(fake_pcam, frac):
    ds = D.load_pcam(str(fake_pcam), frac)
    paths, labels = L.load_pcam_paths_labels(str(fake_pcam), frac)
    assert ds.paths == paths
    assert ds.labels_all().tolist() == labels


@pytest.mark.parametrize("frac", [0.2, 0.5, 1.0])
def test_panda_universe_matches_legacy(fake_panda, frac):
    csv, img_dir = str(fake_panda / "train.csv"), str(fake_panda / "train_images" / "train_images")
    ds = D.load_panda(csv, img_dir, frac)
    paths, labels = L.load_panda_paths_labels(csv, img_dir, frac)
    assert ds.paths == paths
    assert ds.labels_all().tolist() == labels


@pytest.mark.parametrize("n,pos", [(5000, 0.4), (2123, 0.55), (300, 0.1)])
def test_splits_and_subsets_match_legacy(n, pos):
    rng = np.random.default_rng(n)
    labels = (rng.random(n) < pos).astype(np.int64)
    universe = np.arange(n, dtype=np.int64)
    legacy = L.make_splits_indices(universe, labels, n_splits=3, val_ratio=0.2, base_seed=1000)
    for split_id in range(3):
        tr, va = D.make_split(labels, split_id, val_ratio=0.2, base_seed=1000)
        np.testing.assert_array_equal(tr, legacy[split_id][0])
        np.testing.assert_array_equal(va, legacy[split_id][1])
        for seed in range(3):
            for frac in [0.01, 0.05, 0.10]:
                legacy_seed = 42 + seed + 10 * split_id  # as written in the notebooks
                np.testing.assert_array_equal(
                    D.label_subset(tr, labels, frac, D.subset_seed(split_id, seed)),
                    L.stratified_label_subset_indices(tr, labels, frac, legacy_seed),
                )


def test_seed_formulas():
    assert D.run_seed(0, 0) == 10_000
    assert D.run_seed(2, 1) == 10_201
    assert D.subset_seed(2, 1) == 63


def test_metrics_match_legacy():
    rng = np.random.default_rng(0)
    for _ in range(20):
        y = rng.integers(0, 2, size=500)
        p = np.clip(rng.beta(2, 2, size=500) + 0.3 * (y - 0.5), 0, 1)
        p[:3] = [np.nan, 1.0, 0.0]
        new, old = compute_metrics_binary(p, y), L.compute_metrics_binary(p, y)
        assert new["auroc"] == old["auc"]
        assert new["accuracy"] == old["acc"]
        assert new["f1"] == old["f1"]
        assert new["ece"] == old["ece"]
        assert new["brier"] == old["brier"]


def test_toplabel_ece_matches_notebook_08():
    from sslhist.metrics import compute_ece, compute_ece_toplabel
    rng = np.random.default_rng(1)
    for _ in range(20):
        y = rng.integers(0, 2, size=500)
        p = np.clip(rng.beta(2, 2, size=500) + 0.3 * (y - 0.5), 0.001, 0.999)
        assert compute_ece_toplabel(p, y) == L.expected_calibration_error(p, y)
    # the two definitions are different numbers (why the conference federated ECE is not comparable)
    assert compute_ece_toplabel(p, y) != compute_ece(p, y)
