"""YAML configs with single inheritance (``inherit: base.yaml``) and a
machine-specific paths file that is not committed (``configs/paths.yaml``)."""
import copy
import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _load_with_inherit(path: Path) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    parent = cfg.pop("inherit", None)
    if parent:
        cfg = _deep_merge(_load_with_inherit(path.parent / parent), cfg)
    return cfg


def resolve(p: str) -> Path:
    """Paths in configs are relative to the repository root."""
    p = Path(os.path.expanduser(str(p)))
    return p if p.is_absolute() else REPO_ROOT / p


def load_config(path: str, paths_file: str = None) -> dict:
    """Load an experiment config and merge the data paths into ``cfg['data']``.

    The paths file is looked up in this order: the ``paths_file`` argument,
    the ``SSLHIST_PATHS`` environment variable, ``configs/paths.yaml``.
    """
    cfg = _load_with_inherit(resolve(path))
    pf = paths_file or os.environ.get("SSLHIST_PATHS") or "configs/paths.yaml"
    pf = resolve(pf)
    if pf.exists():
        with open(pf) as f:
            cfg["data"] = _deep_merge(cfg.get("data", {}), yaml.safe_load(f) or {})
    cfg["_config_path"] = str(resolve(path))
    cfg["_paths_file"] = str(pf) if pf.exists() else None
    return cfg


def family_cfg(cfg: dict, backbone: str) -> dict:
    from .models import backbone_family
    return cfg["families"][backbone_family(backbone)]
