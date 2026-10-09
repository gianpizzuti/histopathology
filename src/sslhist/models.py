"""Backbones. Each builder returns ``(module, feat_dim)`` with the classifier
head removed, so ``module(x)`` returns the pooled feature / CLS token."""
from typing import Tuple

import torch.nn as nn
from torchvision import models
from torchvision.models.vision_transformer import VisionTransformer


def _resnet(name: str, pretrained: bool) -> Tuple[nn.Module, int]:
    weights = {"resnet18": models.ResNet18_Weights, "resnet50": models.ResNet50_Weights}[name]
    m = getattr(models, name)(weights=weights.DEFAULT if pretrained else None)
    feat_dim = m.fc.in_features
    m.fc = nn.Identity()
    return m, feat_dim


def _vit_b_16(pretrained: bool) -> Tuple[nn.Module, int]:
    m = models.vit_b_16(weights=models.ViT_B_16_Weights.DEFAULT if pretrained else None)
    m.heads = nn.Identity()
    return m, m.hidden_dim


def _vit_s_16(pretrained: bool) -> Tuple[nn.Module, int]:
    # torchvision ships no ViT-S; same configuration as DeiT-S / DINO ViT-S/16.
    if pretrained:
        raise ValueError("No torchvision ImageNet weights for vit_s_16.")
    m = VisionTransformer(image_size=224, patch_size=16, num_layers=12, num_heads=6,
                          hidden_dim=384, mlp_dim=1536)
    m.heads = nn.Identity()
    return m, m.hidden_dim


# DINOv2 (E3): official checkpoints from facebookresearch/dinov2 via torch.hub, used frozen.
# The branch is given explicitly so torch.hub does not need to probe GitHub for it.
DINOV2_HUB = "facebookresearch/dinov2:main"


def _dinov2(name: str, pretrained: bool) -> Tuple[nn.Module, int]:
    if not pretrained:
        raise ValueError(f"{name} is only used with its pretrained weights (method 'frozen').")
    import torch
    m = torch.hub.load(DINOV2_HUB, name, pretrained=True, trust_repo=True)
    # forward(x) returns the normalised CLS token (embed_dim); inputs must be a multiple of 14 px
    return m, int(m.embed_dim)


def _dinov2_tiny_test(pretrained: bool) -> Tuple[nn.Module, int]:
    """Stand-in for DINOv2 in the CPU tests (no download): a tiny ViT on 28x28 inputs."""
    m = VisionTransformer(image_size=28, patch_size=14, num_layers=1, num_heads=2,
                          hidden_dim=32, mlp_dim=64)
    m.heads = nn.Identity()
    return m, m.hidden_dim


def _vit_tiny_test(pretrained: bool) -> Tuple[nn.Module, int]:
    """Very small ViT used only by the CPU smoke tests (32x32 input)."""
    m = VisionTransformer(image_size=32, patch_size=8, num_layers=2, num_heads=2,
                          hidden_dim=32, mlp_dim=64)
    m.heads = nn.Identity()
    return m, m.hidden_dim


_BUILDERS = {
    "resnet18": lambda p: _resnet("resnet18", p),
    "resnet50": lambda p: _resnet("resnet50", p),
    "vit_b_16": _vit_b_16,
    "vit_s_16": _vit_s_16,
    "vit_tiny_test": _vit_tiny_test,
    "dinov2_vits14": lambda p: _dinov2("dinov2_vits14", p),
    "dinov2_vitb14": lambda p: _dinov2("dinov2_vitb14", p),
    "dinov2_tiny_test": _dinov2_tiny_test,
}


def build_backbone(name: str, pretrained: bool = False) -> Tuple[nn.Module, int]:
    if name not in _BUILDERS:
        raise ValueError(f"Unknown backbone '{name}'. Available: {sorted(_BUILDERS)}")
    return _BUILDERS[name](pretrained)


def backbone_family(name: str) -> str:
    """Protocol family: batch sizes, image sizes and SSL head widths depend on it."""
    if name.startswith("dinov2"):
        return "dinov2"
    if name.startswith("resnet"):
        return "resnet"
    if name.startswith("vit"):
        return "vit"
    raise ValueError(f"Unknown backbone family for '{name}'")
