import hashlib
import os
import platform
import random
import subprocess
from datetime import datetime, timezone
from typing import Iterable

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Same as the legacy notebooks (cuDNN deterministic, no benchmark)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fingerprint(items: Iterable) -> str:
    """Short hash of an ordered collection (e.g. image ids of a split)."""
    h = hashlib.sha1()
    for x in items:
        h.update(str(x).encode())
        h.update(b"\0")
    return h.hexdigest()[:12]


def git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        ).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def run_metadata() -> dict:
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return {
        "git_commit": git_commit(),
        "hostname": platform.node(),
        "gpu": gpu,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
