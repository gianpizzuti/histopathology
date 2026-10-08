import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
from PIL import Image

# Tests never use a GPU, also on the shared server: hide every GPU from this process
# and from the scripts it starts (set before torch initialises CUDA).
os.environ["CUDA_VISIBLE_DEVICES"] = ""

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))


def _img(rng, size):
    return Image.fromarray(rng.integers(0, 255, size=(size, size, 3), dtype=np.uint8))


@pytest.fixture(scope="session")
def fake_pcam(tmp_path_factory):
    """Kaggle PCam layout: train_labels.csv + train/<id>.tif (a few files missing)."""
    root = tmp_path_factory.mktemp("pcam")
    rng = np.random.default_rng(0)
    n = 300
    ids = [f"{i:040x}" for i in rng.integers(0, 2**62, size=n)]
    labels = (rng.random(n) < 0.4).astype(int)
    pd.DataFrame({"id": ids, "label": labels}).to_csv(root / "train_labels.csv", index=False)
    (root / "train").mkdir()
    for k, i in enumerate(ids):
        if k % 97 == 5:  # missing image -> dropped by the existence filter
            continue
        _img(rng, 32).save(root / "train" / f"{i}.tif")
    return root


@pytest.fixture(scope="session")
def fake_panda(tmp_path_factory):
    """PANDA train.csv (with a few NaN grades) + one resized PNG per slide."""
    root = tmp_path_factory.mktemp("panda")
    rng = np.random.default_rng(1)
    n = 300
    ids = [f"{i:032x}" for i in rng.integers(0, 2**62, size=n)]
    grades = rng.integers(0, 6, size=n).astype(float)
    grades[[3, 50]] = np.nan
    pd.DataFrame({
        "image_id": ids,
        "data_provider": rng.choice(["karolinska", "radboud"], size=n),
        "isup_grade": grades,
        "gleason_score": "3+3",
    }).to_csv(root / "train.csv", index=False)
    img_dir = root / "train_images" / "train_images"
    img_dir.mkdir(parents=True)
    for k, i in enumerate(ids):
        if k % 41 == 7:
            continue
        _img(rng, 40).save(img_dir / f"{i}.png")
    return root


@pytest.fixture
def tiny_config(tmp_path, fake_pcam, fake_panda):
    """Experiment config inheriting the real base.yaml, shrunk to run on CPU in seconds."""
    cfg = {
        "inherit": str(REPO / "configs" / "base.yaml"),
        "experiment": "smoke",
        "results_dir": str(tmp_path / "results"),
        "artifacts_dir": str(tmp_path / "artifacts"),
        "data": {
            "sample_frac": {"pcam": 1.0, "panda": 1.0},
            "pcam_root": str(fake_pcam),
            "panda_train_csv": str(fake_panda / "train.csv"),
            "panda_images_dir": str(fake_panda / "train_images" / "train_images"),
        },
        "protocol": {"ssl_epochs": 1, "probe_epochs": 2},
        "families": {
            "resnet": {"img_size": {"pcam": 32, "panda": 32}, "batch_ssl": 32, "batch_sup": 64},
            "vit": {"img_size": {"pcam": 32, "panda": 32}, "batch_ssl": 16, "batch_sup": 64},
            "dinov2": {"img_size": {"pcam": 28, "panda": 28}, "batch_sup": 64},
        },
        "runtime": {"num_workers": 0, "torch_threads": 2},
        "grid": {"datasets": ["panda"], "methods": ["simclr"], "backbones": ["resnet18", "vit_tiny_test"]},
    }
    path = tmp_path / "smoke.yaml"
    path.write_text(yaml.safe_dump(cfg))
    return path


@pytest.fixture
def subprocess_env():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env["SSLHIST_PATHS"] = "/nonexistent/paths.yaml"
    return env
