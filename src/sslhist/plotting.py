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
    ("frozen", "dinov2_vitb14"): "black",
    ("sup_scratch", "resnet18"): "#e6ab02",
    ("sup_imagenet", "resnet18"): "#1b9e77",
    # E4 federated SimCLR: blues (5 clients) and oranges (10 clients), darker = more label skew
    ("simclr-fed-k5-iid", "resnet18"): "#9ecae1",
    ("simclr-fed-k5-a0.5", "resnet18"): "#4292c6",
    ("simclr-fed-k5-a0.1", "resnet18"): "#08519c",
    ("simclr-fed-k10-iid", "resnet18"): "#fdae6b",
    ("simclr-fed-k10-a0.5", "resnet18"): "#f16913",
    ("simclr-fed-k10-a0.1", "resnet18"): "#a63603",
}
# Supervised baselines (E6) are dashed, so they stand out from the SSL encoders
METHOD_LINESTYLE = {"sup_scratch": "--", "sup_imagenet": "--"}
_LEGEND_METHOD = {"sup_scratch": "SUPERVISED (scratch)", "sup_imagenet": "SUPERVISED (ImageNet)"}
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
    "sup_scratch": {"label": "supervised (scratch)"}, "sup_imagenet": {"label": "supervised (ImageNet)"},
}
METRIC_LABELS = {"auroc": "AUROC", "accuracy": "Accuracy", "f1": "F1", "ece": "ECE", "brier": "Brier",
                 # E6 (TS = after temperature scaling)
                 "temperature": "$T$", "id_ece_raw": "ECE", "id_ece_ts": "ECE (TS)",
                 "id_msp_raw": "ID MSP", "id_msp_ts": "ID MSP (TS)",
                 "ood_msp_raw": "OOD MSP", "ood_msp_ts": "OOD MSP (TS)",
                 "id_entropy_raw": "ID entropy", "id_entropy_ts": "ID entropy (TS)",
                 "ood_entropy_raw": "OOD entropy", "ood_entropy_ts": "OOD entropy (TS)",
                 "ood_conf90_raw": "OOD MSP$\\geq$0.9", "ood_conf90_ts": "OOD MSP$\\geq$0.9 (TS)",
                 "ood_auroc_raw": "OOD AUROC (MSP)", "ood_fpr95_raw": "FPR95 (MSP)",
                 "knn_ood_auroc": "OOD AUROC (kNN)", "knn_fpr95": "FPR95 (kNN)", "ood_pos_rate": "OOD pred. positive"}
DATASET_LABELS = {"pcam": "PCam", "panda": "PANDA"}

# Titles and axis labels as in the camera-ready
_TITLE_METRIC = {"auroc": "AUC", "accuracy": "ACCURACY", "f1": "F1", "ece": "ECE", "brier": "BRIER"}
_YLABEL = {"auroc": "AUC", "accuracy": "Accuracy", "f1": "F1", "ece": "ECE", "brier": "Brier score"}


# Up to this many series the legend stays inside the axes (camera-ready); with more it
# would cover the data, so it goes to the right of the axes.
MAX_LEGEND_INSIDE = 4
LEGEND_WIDTH = 3.2  # inches added to the figure for a legend outside the axes


def _legend(fig, ax, outside: bool, **kw) -> None:
    if outside:
        ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, fontsize="small", **kw)
    else:
        ax.legend(**kw)
    fig.tight_layout()


def _save(fig, out_pdf) -> None:
    fig.savefig(out_pdf, bbox_inches="tight")  # keeps a legend placed outside the axes
    plt.close(fig)


def series_label(method: str, backbone: str) -> str:
    from .federated import fed_label, parse_fed_method
    if parse_fed_method(method):
        return f"{fed_label(method)} + {backbone}"
    return f"{_LEGEND_METHOD.get(method, method.upper())} + {backbone}"


def method_label(method: str) -> str:
    """Method name in the LaTeX tables."""
    from .federated import parse_fed_method
    f = parse_fed_method(method)
    if f:
        part = "IID" if f["partition"] == "iid" else f"Dir $\\alpha$={f['alpha']:g}"
        return f"Fed-{METHOD_STYLES[f['ssl']]['label']} ({f['clients']} clients, {part})"
    return METHOD_STYLES.get(method, {}).get("label", method)


def series_color(method: str, backbone: str, used: dict) -> str:
    key = (method, backbone)
    if key in SERIES_COLORS:
        return SERIES_COLORS[key]
    if key not in used:
        used[key] = _EXTRA[len(used) % len(_EXTRA)]
    return used[key]


def plot_label_efficiency(summary, dataset: str, metric: str, out_pdf, figsize=(8, 4.5), title=None,
                          ylabel=None) -> None:
    """``summary`` needs columns dataset, backbone, method, label_frac, <metric>_mean, <metric>_std."""
    plt.style.use("default")
    df = summary[summary["dataset"] == dataset]
    fracs = sorted(df["label_frac"].unique())
    used = {}
    series = sorted(set(zip(df["method"], df["backbone"])))
    many = len(series) > MAX_LEGEND_INSIDE
    # with the legend outside, the figure is widened so the axes keep the same size
    fig, ax = plt.subplots(figsize=(figsize[0] + (LEGEND_WIDTH if many else 0), figsize[1]))
    for method, backbone in series:
        s = df[(df.method == method) & (df.backbone == backbone)].set_index("label_frac").reindex(fracs)
        mean, std = s[f"{metric}_mean"].to_numpy(), s[f"{metric}_std"].fillna(0).to_numpy()
        color = series_color(method, backbone, used)
        ax.plot(fracs, mean, marker="o", color=color, linestyle=METHOD_LINESTYLE.get(method, "-"),
                label=series_label(method, backbone))
        # lighter bands when many series overlap
        ax.fill_between(fracs, mean - std, mean + std, color=color, alpha=0.08 if many else 0.2, linewidth=0)
    ax.set_xscale("log")
    ax.set_xticks(fracs)
    ax.set_xticklabels([f"{f * 100:g}%" for f in fracs])
    ax.minorticks_off()
    ax.set_xlabel("Label fraction")
    ax.set_ylabel(ylabel or _YLABEL.get(metric, metric))
    ax.set_title(title or f"{dataset.upper()} label efficiency: {_TITLE_METRIC.get(metric, metric.upper())} (mean±std)")
    _legend(fig, ax, many)
    _save(fig, out_pdf)


# -----------------------
# E6 (OOD) figures
# -----------------------
def plot_confidence_bars(summary, dataset: str, ood_dataset: str, frac: float, out_pdf, figsize=(11, 4.5)) -> None:
    """Mean MSP on ID (evaluation half) and OOD images per series at one label fraction,
    before and after temperature scaling; error bars = std over splits x seeds."""
    import numpy as np
    plt.style.use("default")
    df = summary[(summary["dataset"] == dataset) & (summary["label_frac"] == frac)]
    series = sorted(set(zip(df["method"], df["backbone"])), key=lambda s: (s[0].startswith("sup"), s))
    bars = [("id_msp_raw", "ID", "tab:blue", None), ("id_msp_ts", "ID, TS", "tab:blue", "//"),
            ("ood_msp_raw", "OOD", "tab:red", None), ("ood_msp_ts", "OOD, TS", "tab:red", "//")]
    x, w = np.arange(len(series)), 0.2
    fig, ax = plt.subplots(figsize=figsize)
    for j, (col, label, color, hatch) in enumerate(bars):
        rows = [df[(df.method == m) & (df.backbone == b)].iloc[0] for m, b in series]
        ax.bar(x + (j - 1.5) * w, [r[f"{col}_mean"] for r in rows], w, yerr=[r[f"{col}_std"] for r in rows],
               color=color, alpha=0.45 if hatch else 0.85, hatch=hatch, edgecolor=color, capsize=2, label=label)
    ax.set_xticks(x)
    ax.set_xticklabels([series_label(m, b) for m, b in series], rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Mean MSP (confidence)")
    ax.set_title(f"{dataset.upper()} (ID) vs {ood_dataset.upper()} (OOD): confidence at {frac * 100:g}% labels "
                 "(mean±std)")
    _legend(fig, ax, True)
    _save(fig, out_pdf)


def plot_msp_histograms(panels, title: str, out_pdf, bins: int = 25) -> None:
    """``panels``: list of (series label, {"id_raw", "ood_raw", "id_ts", "ood_ts"} MSP arrays).
    One panel per series: raw MSP filled, after temperature scaling as a line."""
    import numpy as np
    plt.style.use("default")
    n = len(panels)
    cols = min(4, n)
    rows_ = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows_, cols, figsize=(3.2 * cols, 2.6 * rows_), squeeze=False, sharex=True)
    edges = np.linspace(0.5, 1.0, bins + 1)
    for ax, (label, d) in zip(axes.ravel(), panels):
        ax.hist(d["id_raw"], bins=edges, density=True, alpha=0.45, color="tab:blue", label="ID")
        ax.hist(d["ood_raw"], bins=edges, density=True, alpha=0.45, color="tab:red", label="OOD")
        ax.hist(d["id_ts"], bins=edges, density=True, histtype="step", color="tab:blue", label="ID, TS")
        ax.hist(d["ood_ts"], bins=edges, density=True, histtype="step", color="tab:red", label="OOD, TS")
        ax.set_title(label, fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    for ax in axes[-1]:
        ax.set_xlabel("MSP", fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    # one legend for all panels, below them
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=4, fontsize=8, frameon=False)
    _save(fig, out_pdf)
