#!/usr/bin/env python
"""Download once, and check, pretrained weights: DINOv2 (E3, frozen encoders) and the
ImageNet ResNet-18 of the E6 supervised baseline.

Run it before launching E3 / E6, so that the units started in parallel find the weights
in the torch.hub cache ($TORCH_HOME/hub) instead of all downloading them at once.
Runs on CPU (no GPU is used).

Examples:
  python scripts/prefetch_weights.py                          # E3
  python scripts/prefetch_weights.py --backbones resnet18     # E6
"""
import argparse
import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only: set before torch initialises CUDA

import torch  # noqa: E402

from sslhist.models import build_backbone  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbones", default="dinov2_vits14,dinov2_vitb14")
    args = ap.parse_args()

    print(f"torch.hub cache: {torch.hub.get_dir()}")
    for name in args.backbones.split(","):
        model, dim = build_backbone(name, pretrained=True)
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(2, 3, 224, 224))
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        assert tuple(out.shape) == (2, dim), f"{name}: unexpected output {tuple(out.shape)}"
        print(f"[ok] {name}: {n_params:.1f}M parameters, feature dim {dim}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
