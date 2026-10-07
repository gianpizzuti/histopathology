"""Figure style for the paper. All figures are saved as PDF (vector, fonts embedded).

TODO: align with the camera-ready figures (font, size, colours) once we have them.
Colour encodes the backbone and line style + marker encode the SSL method, so
a figure stays readable in greyscale print.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# Springer LNCS/LNBI text width is 12.2 cm (~4.8 in).
TEXT_WIDTH_IN = 4.8

# Colour follows the backbone, always in this fixed order (never by rank).
BACKBONE_COLORS = {
    "resnet18": "#2a78d6",
    "vit_b_16": "#eb6834",
    "resnet50": "#1baf7a",
    "vit_s_16": "#4a3aa7",
    "dinov2_vits14": "#e87ba4",
    "dinov2_vitb14": "#008300",
}
BACKBONE_LABELS = {
    "resnet18": "ResNet-18", "resnet50": "ResNet-50",
    "vit_b_16": "ViT-B/16", "vit_s_16": "ViT-S/16",
    "dinov2_vits14": "DINOv2 ViT-S/14", "dinov2_vitb14": "DINOv2 ViT-B/14",
}
METHOD_STYLES = {
    "simclr": {"linestyle": "-", "marker": "o", "label": "SimCLR"},
    "byol": {"linestyle": "--", "marker": "s", "label": "BYOL"},
    "barlow": {"linestyle": ":", "marker": "^", "label": "Barlow Twins"},
    "frozen": {"linestyle": "-.", "marker": "D", "label": "frozen"},
}
METRIC_LABELS = {"auroc": "AUROC", "accuracy": "Accuracy", "f1": "F1", "ece": "ECE", "brier": "Brier"}
DATASET_LABELS = {"pcam": "PCam", "panda": "PANDA"}

INK = "#0b0b0b"
INK_MUTED = "#52514e"
GRID = "#e4e3df"


def apply_style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"],
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.edgecolor": INK_MUTED, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK_MUTED, "ytick.color": INK_MUTED,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.6,
        "lines.linewidth": 1.4, "lines.markersize": 4.5,
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })


def plot_metrics_vs_fraction(summary, metrics, out_pdf, datasets=None) -> None:
    """Grid of panels: one row per dataset, one column per metric.
    ``summary`` needs columns dataset, backbone, method, label_frac, <m>_mean, <m>_std."""
    apply_style()
    datasets = datasets or [d for d in ["pcam", "panda"] if d in set(summary["dataset"])]
    b_order = {b: i for i, b in enumerate(BACKBONE_COLORS)}
    m_order = {m: i for i, m in enumerate(METHOD_STYLES)}
    series = sorted(set(zip(summary["backbone"], summary["method"])),
                    key=lambda t: (b_order.get(t[0], 99), m_order.get(t[1], 99), t))
    fracs = sorted(summary["label_frac"].unique())
    xs = range(len(fracs))

    fig, axes = plt.subplots(len(datasets), len(metrics), squeeze=False,
                             figsize=(TEXT_WIDTH_IN, 1.75 * len(datasets) + 0.45))
    for r, ds in enumerate(datasets):
        for c, m in enumerate(metrics):
            ax = axes[r][c]
            for i, (b, meth) in enumerate(series):
                dodge = (i - (len(series) - 1) / 2) * 0.06  # keep error bars from overlapping
                s = summary[(summary.dataset == ds) & (summary.backbone == b) & (summary.method == meth)]
                s = s.set_index("label_frac").reindex(fracs)
                st = METHOD_STYLES.get(meth, {"linestyle": "-", "marker": "o"})
                ax.errorbar([x + dodge for x in xs], s[f"{m}_mean"], yerr=s[f"{m}_std"],
                            color=BACKBONE_COLORS.get(b, INK_MUTED),
                            linestyle=st["linestyle"], marker=st["marker"],
                            capsize=2, elinewidth=0.8, markeredgecolor="white", markeredgewidth=0.6)
            ax.set_xticks(list(xs))
            ax.set_xticklabels([f"{f * 100:g}%" for f in fracs])
            if r == len(datasets) - 1:
                ax.set_xlabel("Labelled fraction")
            ax.set_ylabel(METRIC_LABELS.get(m, m))
            ax.set_title(DATASET_LABELS.get(ds, ds), loc="left", color=INK_MUTED)

    backbones = list(dict.fromkeys(b for b, _ in series))
    methods = list(dict.fromkeys(meth for _, meth in series))
    handles = [Line2D([], [], color=BACKBONE_COLORS.get(b, INK_MUTED), linewidth=2.5,
                      label=BACKBONE_LABELS.get(b, b)) for b in backbones]
    handles += [Line2D([], [], color=INK_MUTED, linestyle=METHOD_STYLES[m]["linestyle"],
                       marker=METHOD_STYLES[m]["marker"], label=METHOD_STYLES[m]["label"])
                for m in methods if m in METHOD_STYLES]
    fig.legend(handles=handles, loc="upper center", ncol=len(handles), frameon=False,
               bbox_to_anchor=(0.5, 1.0), handlelength=2.2, columnspacing=1.2)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_pdf)
    plt.close(fig)
