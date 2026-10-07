"""Figures in the style of the CIBB 2026 camera-ready (Fig. 1a, Fig. 2):
matplotlib defaults, one panel per dataset and metric, one line per
"METHOD + backbone" with 'o' markers and a shaded mean +/- std band,
log-scaled label-fraction axis with ticks 1%, 5%, 10%. Saved as PDF.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# Fixed series -> colour mapping, so a series keeps its colour in every figure.
# The first four reproduce the camera-ready (tab10 C0-C3 in alphabetical order).
SERIES_COLORS = {
    ("byol", "resnet18"): "tab:blue",
    ("byol", "vit_b_16"): "tab:orange",
    ("simclr", "resnet18"): "tab:green",
    ("simclr", "vit_b_16"): "tab:red",
    ("byol", "resnet50"): "tab:purple",
    ("simclr", "resnet50"): "tab:brown",
    ("byol", "vit_s_16"): "tab:pink",
    ("simclr", "vit_s_16"): "tab:olive",
    ("barlow", "resnet18"): "tab:cyan",
    ("frozen", "dinov2_vits14"): "tab:gray",
}
_EXTRA = ["#1a55a3", "#b3541e", "#2e7d32", "#7b1fa2", "#00838f", "#5d4037"]

# Used by the LaTeX tables (report.py)
BACKBONE_LABELS = {
    "resnet18": "ResNet-18", "resnet50": "ResNet-50",
    "vit_b_16": "ViT-B/16", "vit_s_16": "ViT-S/16",
    "dinov2_vits14": "DINOv2 ViT-S/14", "dinov2_vitb14": "DINOv2 ViT-B/14",
}
METHOD_STYLES = {
    "simclr": {"label": "SimCLR"}, "byol": {"label": "BYOL"},
    "barlow": {"label": "Barlow Twins"}, "frozen": {"label": "frozen"},
}
METRIC_LABELS = {"auroc": "AUROC", "accuracy": "Accuracy", "f1": "F1", "ece": "ECE", "brier": "Brier"}
DATASET_LABELS = {"pcam": "PCam", "panda": "PANDA"}

# Titles and axis labels as in the camera-ready
_TITLE_METRIC = {"auroc": "AUC", "accuracy": "ACCURACY", "f1": "F1", "ece": "ECE", "brier": "BRIER"}
_YLABEL = {"auroc": "AUC", "accuracy": "Accuracy", "f1": "F1", "ece": "ECE", "brier": "Brier score"}


def series_color(method: str, backbone: str, used: dict) -> str:
    key = (method, backbone)
    if key in SERIES_COLORS:
        return SERIES_COLORS[key]
    if key not in used:
        used[key] = _EXTRA[len(used) % len(_EXTRA)]
    return used[key]


def plot_label_efficiency(summary, dataset: str, metric: str, out_pdf, figsize=(8, 4.5)) -> None:
    """``summary`` needs columns dataset, backbone, method, label_frac, <metric>_mean, <metric>_std."""
    plt.style.use("default")
    df = summary[summary["dataset"] == dataset]
    fracs = sorted(df["label_frac"].unique())
    used = {}
    fig, ax = plt.subplots(figsize=figsize)
    for method, backbone in sorted(set(zip(df["method"], df["backbone"]))):
        s = df[(df.method == method) & (df.backbone == backbone)].set_index("label_frac").reindex(fracs)
        mean, std = s[f"{metric}_mean"].to_numpy(), s[f"{metric}_std"].fillna(0).to_numpy()
        color = series_color(method, backbone, used)
        ax.plot(fracs, mean, marker="o", color=color, label=f"{method.upper()} + {backbone}")
        ax.fill_between(fracs, mean - std, mean + std, color=color, alpha=0.2, linewidth=0)
    ax.set_xscale("log")
    ax.set_xticks(fracs)
    ax.set_xticklabels([f"{f * 100:g}%" for f in fracs])
    ax.minorticks_off()
    ax.set_xlabel("Label fraction")
    ax.set_ylabel(_YLABEL.get(metric, metric))
    ax.set_title(f"{dataset.upper()} label efficiency: {_TITLE_METRIC.get(metric, metric.upper())} (mean±std)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)
