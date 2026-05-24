"""
plot_comparison.py — Comprehensive single-figure comparison of the
centralized vs. federated SpintronicEncoder experiments.

Run from project root:
    source code/eval_env/bin/activate
    python comparison/plot_comparison.py

Output:
    comparison/comparison_summary.png   — publication-ready summary figure
"""

import json
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
from pathlib import Path

ROOT    = Path(__file__).resolve().parent.parent
OUTDIR  = Path(__file__).resolve().parent
CODEDIR = ROOT / "code"

METRICS_JSON = CODEDIR / "analysis" / "metrics_comparison.json"
CEN_CSV      = CODEDIR / "central_checkpoints" / "loss_central.csv"
FED_LOG      = CODEDIR / "federation_run4.log"
UMAP_PNG     = CODEDIR / "analysis" / "latent_umap_comparison.png"


# ── Data loaders ───────────────────────────────────────────────────────────────

def load_metrics():
    return json.loads(METRICS_JSON.read_text())

def load_central_losses():
    df = pd.read_csv(CEN_CSV)
    return df["epoch"].astype(int).tolist(), df["loss"].astype(float).tolist()

def load_fed_losses():
    losses = []
    for line in FED_LOG.read_text().splitlines():
        m = re.search(r"Round metrics.*avg_loss=([0-9.]+)", line)
        if m:
            losses.append(float(m.group(1)))
    return list(range(1, len(losses)+1)), losses


# ── Color palette ──────────────────────────────────────────────────────────────

C_CEN = "#E07B39"   # warm orange — centralized
C_FED = "#3A7EBD"   # cool blue   — federated
C_BG  = "#F7F7F7"


# ── Panel builders ─────────────────────────────────────────────────────────────

def panel_metrics(ax, metrics):
    """Grouped bar chart — 5 metrics, two bars each."""
    cen = metrics["centralized"]
    fed = metrics["federated"]

    # Metrics where HIGHER is better → show as-is
    # Metrics where LOWER is better  → show inverted (1/x) so taller = better
    metric_defs = [
        ("Silhouette\n(↑ better)",        cen["silhouette"],            fed["silhouette"],           False),
        ("Davies-Bouldin\n(↓ better)",     cen["davies_bouldin"],        fed["davies_bouldin"],        True),
        ("Intra/Inter\nRatio (↓ better)",  cen["intra_inter_ratio"],     fed["intra_inter_ratio"],     True),
        ("Inter-cluster\nDist (↑ better)", cen["inter_cluster_dist"],    fed["inter_cluster_dist"],    False),
        ("Training\nLoss (↓ better)",      0.2111,                       0.2764,                       True),
    ]

    labels   = [d[0] for d in metric_defs]
    vals_cen = []
    vals_fed = []
    for _, vc, vf, invert in metric_defs:
        if invert:
            # normalize: centralized gets 1.0, federated gets vc/vf (>1 means worse)
            vals_cen.append(1.0)
            vals_fed.append(vc / vf)
        else:
            # normalize: centralized gets 1.0, federated gets vf/vc (<1 means worse)
            vals_cen.append(1.0)
            vals_fed.append(vf / vc)

    x     = np.arange(len(labels))
    width = 0.35

    bars_cen = ax.bar(x - width/2, vals_cen, width, label="Centralized",
                      color=C_CEN, alpha=0.9, zorder=3)
    bars_fed = ax.bar(x + width/2, vals_fed, width, label="Federated",
                      color=C_FED, alpha=0.9, zorder=3)

    # Annotate with actual values
    raw_cen = [d[1] for d in metric_defs]
    raw_fed = [d[2] for d in metric_defs]
    for i, (bar, rv) in enumerate(zip(bars_cen, raw_cen)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{rv:.3f}", ha="center", va="bottom", fontsize=7.5, color="#444")
    for i, (bar, rv) in enumerate(zip(bars_fed, raw_fed)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{rv:.3f}", ha="center", va="bottom", fontsize=7.5, color="#444")

    ax.axhline(1.0, color="black", lw=1.0, ls="--", alpha=0.4,
               label="Centralized baseline (1.0)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Score relative to centralized  (taller = better)", fontsize=9)
    ax.set_title("Latent Space Quality — All Metrics", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 1.35)
    ax.grid(axis="y", alpha=0.25, zorder=0)
    ax.legend(fontsize=9, loc="upper right")
    ax.set_facecolor(C_BG)


def panel_loss(ax, fed_rounds, fed_losses, cen_epochs, cen_losses):
    """Dual-axis loss curves: federated left, centralized right."""
    # Smooth centralized
    smooth = pd.Series(cen_losses).rolling(5, center=True, min_periods=1).mean()

    ax2 = ax.twinx()

    l1, = ax.plot(fed_rounds, fed_losses, "o-", color=C_FED, lw=2.5, ms=6,
                  label="Federated (rounds)", zorder=4)
    ax.set_xlabel("Federation Round", fontsize=9, color=C_FED)
    ax.set_ylabel("Federated NT-Xent Loss", fontsize=9, color=C_FED)
    ax.tick_params(axis="y", labelcolor=C_FED)
    ax.set_ylim(0.25, 0.52)

    l2, = ax2.plot(cen_epochs, cen_losses, "-", color=C_CEN, lw=1.2,
                   alpha=0.35, label="_nolegend_")
    l3, = ax2.plot(cen_epochs, smooth, "-", color=C_CEN, lw=2.5,
                   label="Centralized (epochs, smoothed)", zorder=4)
    ax2.set_ylabel("Centralized NT-Xent Loss", fontsize=9, color=C_CEN)
    ax2.tick_params(axis="y", labelcolor=C_CEN)
    ax2.set_ylim(0.19, 0.52)

    # Best-loss markers
    ax.axhline(min(fed_losses), color=C_FED, ls=":", lw=1.5, alpha=0.6)
    ax2.axhline(min(cen_losses), color=C_CEN, ls=":", lw=1.5, alpha=0.6)

    ax.text(20.3, min(fed_losses)+0.003, f"Fed best\n{min(fed_losses):.4f}",
            color=C_FED, fontsize=8, va="bottom")
    ax2.text(1, min(cen_losses)-0.005, f"Cen best {min(cen_losses):.4f}",
             color=C_CEN, fontsize=8, va="top")

    ax.grid(alpha=0.2, zorder=0)
    ax.set_title("Training Loss Convergence", fontsize=11, fontweight="bold")
    ax.set_facecolor(C_BG)

    lines  = [l1, l3]
    labels = [l.get_label() for l in lines]
    ax.legend(lines, labels, fontsize=9, loc="upper right")


def panel_summary_table(ax, metrics):
    """Clean results table at the bottom."""
    ax.axis("off")

    cen = metrics["centralized"]
    fed = metrics["federated"]
    dlt = metrics["delta"]

    col_labels = ["Metric", "Centralized", "Federated", "Δ (Cen−Fed)", "Better"]
    rows = [
        ["Silhouette ↑",
         f"{cen['silhouette']:.4f}",
         f"{fed['silhouette']:.4f}",
         f"{dlt['silhouette']:+.4f}",
         "Centralized (+0.58%)"],
        ["Davies-Bouldin ↓",
         f"{cen['davies_bouldin']:.4f}",
         f"{fed['davies_bouldin']:.4f}",
         f"{dlt['davies_bouldin']:+.4f}",
         "Centralized (−21.5%)"],
        ["Intra/Inter Ratio ↓",
         f"{cen['intra_inter_ratio']:.4f}",
         f"{fed['intra_inter_ratio']:.4f}",
         f"{dlt['intra_inter_ratio']:+.4f}",
         "Centralized (−18.1%)"],
        ["Inter-cluster Dist ↑",
         f"{cen['inter_cluster_dist']:.1f}",
         f"{fed['inter_cluster_dist']:.1f}",
         f"{dlt['inter_cluster_dist']:+.1f}",
         "Centralized (+101.8%)"],
        ["Training Best Loss ↓",
         "0.2111",
         "0.2764",
         "−0.0653",
         "Centralized (−23.6%)"],
    ]

    tbl = ax.table(
        cellText   = rows,
        colLabels  = col_labels,
        cellLoc    = "center",
        loc        = "center",
        bbox       = [0, 0, 1, 1],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)

    # Header styling
    for j in range(len(col_labels)):
        cell = tbl[0, j]
        cell.set_facecolor("#2C3E50")
        cell.set_text_props(color="white", fontweight="bold")

    # Row striping
    for i in range(1, len(rows)+1):
        bg = "#EAF2FB" if i % 2 == 0 else "white"
        for j in range(len(col_labels)):
            tbl[i, j].set_facecolor(bg)
        # Highlight "Better" column
        tbl[i, 4].set_text_props(color="#1A5276", fontweight="bold")

    ax.set_title("Numeric Summary — 40,832 windows × 32-dim embeddings × 11 clusters",
                 fontsize=10, fontweight="bold", pad=8)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    metrics                    = load_metrics()
    cen_epochs, cen_losses     = load_central_losses()
    fed_rounds, fed_losses     = load_fed_losses()
    umap_img                   = mpimg.imread(UMAP_PNG)

    fig = plt.figure(figsize=(20, 18), facecolor="white")
    fig.suptitle(
        "Federated vs. Centralized Contrastive Learning — SpintronicEncoder\n"
        "15 clients · 20 FL rounds · FedProx μ=0.1  vs.  100 epochs centralized · NT-Xent",
        fontsize=14, fontweight="bold", y=0.995
    )

    gs = gridspec.GridSpec(
        3, 2,
        figure     = fig,
        height_ratios = [2.2, 1.8, 1.0],
        hspace     = 0.42,
        wspace     = 0.30,
        top=0.96, bottom=0.04, left=0.06, right=0.97
    )

    # Row 0: UMAP images spanning both columns
    ax_umap = fig.add_subplot(gs[0, :])
    ax_umap.imshow(umap_img)
    ax_umap.axis("off")
    ax_umap.set_title(
        "UMAP Latent Space (32-dim → 2-dim)  —  same 40,832 windows encoded by each model",
        fontsize=11, fontweight="bold", pad=6
    )

    # Row 1 left: metrics bar chart
    ax_metrics = fig.add_subplot(gs[1, 0])
    panel_metrics(ax_metrics, metrics)

    # Row 1 right: loss curves
    ax_loss = fig.add_subplot(gs[1, 1])
    panel_loss(ax_loss, fed_rounds, fed_losses, cen_epochs, cen_losses)

    # Row 2: summary table spanning both columns
    ax_table = fig.add_subplot(gs[2, :])
    panel_summary_table(ax_table, metrics)

    out = OUTDIR / "comparison_summary.png"
    plt.savefig(out, dpi=160, bbox_inches="tight", facecolor="white")
    print(f"Saved → {out}")
    plt.close()


if __name__ == "__main__":
    main()
