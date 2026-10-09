"""Result files: one JSON per (dataset, method, backbone, split, seed, label
fraction) under ``results/lnbi/<experiment>/raw/``; bulky artifacts (encoders,
features, logs) under ``artifacts/<experiment>/``, which is not committed."""
import glob
import itertools
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import pandas as pd

from .config import resolve

ID_COLS = ["experiment", "dataset", "method", "backbone", "split", "seed", "label_frac"]


@dataclass(frozen=True)
class Unit:
    """One SSL pretraining run; it produces one result file per label fraction."""
    dataset: str
    method: str
    backbone: str
    split: int
    seed: int

    @property
    def tag(self) -> str:
        return f"{self.dataset}__{self.method}__{self.backbone}__split{self.split}__seed{self.seed}"


def grid_units(cfg: dict, datasets: Optional[Sequence[str]] = None, methods: Optional[Sequence[str]] = None,
               backbones: Optional[Sequence[str]] = None, splits: Optional[Sequence[int]] = None,
               seeds: Optional[Sequence[int]] = None) -> List[Unit]:
    """Units of an experiment. The grid is either methods x backbones or, when only some
    pairs exist (E6), an explicit list ``series: [[method, backbone], ...]``. Arguments
    that are given replace (methods x backbones) or filter (series) the config values."""
    grid, prot = cfg.get("grid", {}), cfg["protocol"]
    if "series" in grid:
        series = [tuple(s) for s in grid["series"]
                  if (not methods or s[0] in methods) and (not backbones or s[1] in backbones)]
    else:
        series = list(itertools.product(methods or grid["methods"], backbones or grid["backbones"]))
    return [Unit(d, m, b, s, k) for d in (datasets or grid["datasets"]) for m, b in series
            for s in (splits or prot["splits"]) for k in (seeds or prot["seeds"])]


def frac_tag(frac: float) -> str:
    return f"frac{round(frac * 100):02d}"


def exp_results_dir(cfg: dict) -> Path:
    return resolve(cfg["results_dir"]) / cfg["experiment"]


def exp_artifacts_dir(cfg: dict) -> Path:
    return resolve(cfg["artifacts_dir"]) / cfg["experiment"]


def result_path(cfg: dict, unit: Unit, frac: float) -> Path:
    return exp_results_dir(cfg) / "raw" / f"{unit.tag}__{frac_tag(frac)}.json"


def unit_complete(cfg: dict, unit: Unit) -> bool:
    return all(result_path(cfg, unit, f).exists() for f in cfg["protocol"]["label_fracs"])


def write_json(path: Path, obj: dict) -> None:
    """Atomic write, so an interrupted run never leaves a half-written result."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
    os.replace(tmp, path)


def load_raw(results_dirs: Iterable[Path]) -> pd.DataFrame:
    rows: List[dict] = []
    for d in results_dirs:
        for p in sorted(glob.glob(str(Path(d) / "raw" / "*.json"))):
            with open(p) as f:
                rows.append(json.load(f))
    return pd.DataFrame(rows)


# Column names used by the conference notebooks -> names used here
LEGACY_RENAME = {"split_id": "split", "acc": "accuracy", "auc": "auroc"}


def load_legacy_csv(path: str, experiment: str = "conference") -> pd.DataFrame:
    """Read a ``results_raw_*.csv`` from the conference runs into the current schema."""
    df = pd.read_csv(path).rename(columns=LEGACY_RENAME)
    df["experiment"] = experiment
    df["source"] = os.path.basename(path)
    return df
