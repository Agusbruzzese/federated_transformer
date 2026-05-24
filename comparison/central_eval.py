"""
central_eval.py — Compare centralized vs. federated latent spaces.

Encodes all 40 832 labeled windows with both models, computes separability
metrics on the 32-dim embeddings, generates UMAP plots and a loss-curve
comparison, then writes a JSON summary.

Usage
-----
    python central_eval.py \
        --features    ./central_cache/features_labeled.parquet \
        --labels_npy  ./central_cache/labels_all.npy \
        --fed_ckpt    ./server/checkpoints/global_model_latest.pt \
        --cen_ckpt    ./central_checkpoints/central_model_best.pt \
        --fed_log     ./federation_run4.log \
        --cen_csv     ./central_checkpoints/loss_central.csv \
        --out_dir     ./analysis

Outputs
-------
    analysis/latent_umap_centralized.png
    analysis/latent_umap_federated.png
    analysis/latent_umap_comparison.png
    analysis/loss_curve_comparison.png
    analysis/metrics_comparison.json
"""

import sys
import json
import logging
import argparse
import re
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.preprocessing import LabelEncoder

sys.path.insert(0, str(Path(__file__).parent.parent / "code" / "shared"))
from model import SpintronicEncoder, N_FEATURES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

FEATURE_COLS = [
    "dc_level", "dc_range", "rms", "dc_gradient",
    "energy_mean", "energy_std", "energy_max", "burst_ratio", "n_imfs",
]
for i in range(1, 9):
    FEATURE_COLS += [
        f"imf{i}_mean_ia", f"imf{i}_std_ia", f"imf{i}_max_ia",
        f"imf{i}_mean_if", f"imf{i}_std_if",
        f"imf{i}_energy",  f"imf{i}_energy_ratio",
    ]
FEATURE_COLS += ["dc_relative", "dc_range_rel", "rms_rel", "dc_baseline", "t_hours"]


# ── Model loading ──────────────────────────────────────────────────────────────

def load_trunk(path: Path) -> SpintronicEncoder:
    """Load trunk-only state dict (bare keys like '0.weight')."""
    model = SpintronicEncoder(n_features=N_FEATURES)
    state = torch.load(path, map_location="cpu")
    model.trunk.load_state_dict(state)
    model.eval()
    log.info("Loaded %s", path.name)
    return model


# ── Feature loading ────────────────────────────────────────────────────────────

def load_features(features_path: Path) -> tuple[torch.Tensor, np.ndarray]:
    df = pd.read_parquet(features_path)
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].fillna(0.0).values.astype(np.float32)
    if X.shape[1] < N_FEATURES:
        pad = np.zeros((X.shape[0], N_FEATURES - X.shape[1]), np.float32)
        X   = np.hstack([X, pad])
    cluster_labels = df["cluster_full"].values
    log.info("Features: %d × %d  |  %d clusters", X.shape[0], X.shape[1],
             len(np.unique(cluster_labels)))
    return torch.tensor(X[:, :N_FEATURES]), cluster_labels


# ── Encode ─────────────────────────────────────────────────────────────────────

def encode(model: SpintronicEncoder, X: torch.Tensor, batch: int = 512) -> np.ndarray:
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            z, _ = model(X[i:i+batch])
            embeddings.append(z.numpy())
    return np.vstack(embeddings)


# ── Metrics ────────────────────────────────────────────────────────────────────

def compute_metrics(embeddings: np.ndarray, labels: np.ndarray,
                    sample_size: int = 8000, seed: int = 42) -> dict:
    le      = LabelEncoder()
    y       = le.fit_transform(labels)
    n_cls   = len(le.classes_)

    # Silhouette: O(N²) — sample to keep memory reasonable
    rng  = np.random.default_rng(seed)
    idx  = rng.choice(len(embeddings), size=min(sample_size, len(embeddings)), replace=False)
    sil  = float(silhouette_score(embeddings[idx], y[idx], metric="euclidean"))

    # Davies-Bouldin: O(N·k), safe on full set
    db   = float(davies_bouldin_score(embeddings, y))

    # Intra-cluster distance: mean of per-cluster mean pairwise distance
    # approximated as mean distance to own centroid (cheaper, equivalent trend)
    centroids   = np.array([embeddings[y == c].mean(axis=0) for c in range(n_cls)])
    intra_dists = np.array([
        np.linalg.norm(embeddings[y == c] - centroids[c], axis=1).mean()
        for c in range(n_cls)
        if (y == c).sum() > 1
    ])
    intra = float(intra_dists.mean())

    # Inter-cluster distance: mean pairwise distance between centroids
    from scipy.spatial.distance import pdist
    inter = float(pdist(centroids).mean())

    return {
        "silhouette":        round(sil,   4),
        "davies_bouldin":    round(db,    4),
        "intra_cluster_dist":round(intra, 4),
        "inter_cluster_dist":round(inter, 4),
        "intra_inter_ratio": round(intra / (inter + 1e-9), 4),
        "n_clusters":        n_cls,
        "n_windows":         len(embeddings),
    }


# ── UMAP ───────────────────────────────────────────────────────────────────────

CLUSTER_COLORS = {
    "baseline": "#aaaaaa",
    "C1_s0": "#1f77b4", "C1_s1": "#aec7e8",
    "C2_s0": "#ff7f0e", "C2_s1": "#ffbb78", "C2_s2": "#d62728",
    "C2_s3": "#ff9896", "C2_s4": "#9467bd", "C2_s5": "#c5b0d5",
    "C3_s0": "#2ca02c", "C3_s1": "#98df8a",
}

def run_umap(embeddings: np.ndarray, n_neighbors: int = 15) -> np.ndarray:
    from umap import UMAP
    log.info("Running UMAP on %d × %d …", *embeddings.shape)
    return UMAP(n_components=2, random_state=42,
                n_neighbors=n_neighbors, min_dist=0.1).fit_transform(embeddings)


def _scatter_panel(ax, coords, labels, title, show_legend=True):
    unique = sorted(set(labels), key=lambda x: (x != "baseline", x))
    cmap   = plt.cm.get_cmap("tab20", len(unique))
    for i, lbl in enumerate(unique):
        mask  = labels == lbl
        color = CLUSTER_COLORS.get(lbl, cmap(i))
        alpha = 0.25 if lbl == "baseline" else 0.65
        size  = 4  if lbl == "baseline" else 8
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=[color], label=f"{lbl} (n={mask.sum()})",
                   alpha=alpha, s=size, linewidths=0)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.grid(alpha=0.1)
    if show_legend:
        ax.legend(fontsize=6, markerscale=3, loc="best",
                  ncol=max(1, len(unique) // 8))


def plot_umap_single(coords, labels, title, path):
    fig, ax = plt.subplots(figsize=(10, 7))
    _scatter_panel(ax, coords, labels, title)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved → %s", path.name)


def plot_umap_comparison(coords_cen, coords_fed, labels, metrics_cen, metrics_fed, path):
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    _scatter_panel(axes[0], coords_cen, labels,
                   f"Centralized  |  Silhouette={metrics_cen['silhouette']:.3f}"
                   f"  DB={metrics_cen['davies_bouldin']:.3f}", show_legend=True)
    _scatter_panel(axes[1], coords_fed, labels,
                   f"Federated (20 rounds)  |  Silhouette={metrics_fed['silhouette']:.3f}"
                   f"  DB={metrics_fed['davies_bouldin']:.3f}", show_legend=True)
    fig.suptitle("Latent Space Comparison — Centralized vs Federated SpintronicEncoder",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved → %s", path.name)


# ── Loss curves ────────────────────────────────────────────────────────────────

def load_fed_losses(log_path: Path) -> list[float]:
    losses = []
    for line in log_path.read_text().splitlines():
        m = re.search(r"Round metrics.*avg_loss=([0-9.]+)", line)
        if m:
            losses.append(float(m.group(1)))
    return losses


def plot_loss_comparison(fed_losses, cen_csv_path: Path, out_path: Path):
    cen_df = pd.read_csv(cen_csv_path)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: raw curves on their own scale
    ax = axes[0]
    ax.plot(range(1, len(fed_losses)+1), fed_losses,
            "o-", color="#1f77b4", lw=2, ms=5, label="Federated (rounds)")
    ax.set_xlabel("Federation Round"); ax.set_ylabel("NT-Xent Loss")
    ax.set_title("Federated Training Loss (20 rounds)", fontweight="bold")
    ax.grid(alpha=0.2); ax.legend()

    ax2 = axes[1]
    ax2.plot(cen_df["epoch"], cen_df["loss"].astype(float),
             "-", color="#ff7f0e", lw=1.5, alpha=0.6, label="Centralized (epochs)")
    # smooth
    smooth = pd.Series(cen_df["loss"].astype(float)).rolling(5, center=True).mean()
    ax2.plot(cen_df["epoch"], smooth,
             "-", color="#d62728", lw=2, label="Centralized (smoothed)")
    ax2.axhline(min(fed_losses), color="#1f77b4", lw=1.5, ls="--",
                label=f"Fed best={min(fed_losses):.4f}")
    ax2.axhline(float(cen_df["loss"].min()), color="#ff7f0e", lw=1.5, ls="--",
                label=f"Cen best={float(cen_df['loss'].min()):.4f}")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("NT-Xent Loss")
    ax2.set_title("Centralized Training Loss (100 epochs)", fontweight="bold")
    ax2.grid(alpha=0.2); ax2.legend(fontsize=8)

    fig.suptitle("Training Loss: Federated vs. Centralized", fontsize=13,
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Saved → %s", out_path.name)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features",   default="../code/central_cache/features_labeled.parquet")
    parser.add_argument("--fed_ckpt",   default="../code/server/checkpoints/global_model_latest.pt")
    parser.add_argument("--cen_ckpt",   default="../code/central_checkpoints/central_model_best.pt")
    parser.add_argument("--fed_log",    default="../code/federation_run4.log")
    parser.add_argument("--cen_csv",    default="../code/central_checkpoints/loss_central.csv")
    parser.add_argument("--out_dir",    default="../code/analysis")
    parser.add_argument("--umap_neighbors", type=int, default=15)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load features + labels
    X, cluster_labels = load_features(Path(args.features))

    # Load both models
    model_cen = load_trunk(Path(args.cen_ckpt))
    model_fed = load_trunk(Path(args.fed_ckpt))

    # Encode with both
    log.info("Encoding with centralized model …")
    emb_cen = encode(model_cen, X)
    log.info("Encoding with federated model …")
    emb_fed = encode(model_fed, X)

    # Compute metrics
    log.info("Computing metrics (centralized) …")
    metrics_cen = compute_metrics(emb_cen, cluster_labels)
    log.info("Computing metrics (federated) …")
    metrics_fed = compute_metrics(emb_fed, cluster_labels)

    log.info("=" * 55)
    log.info("%-25s  %10s  %10s", "Metric", "Central", "Federated")
    log.info("-" * 55)
    for key in ["silhouette", "davies_bouldin", "intra_cluster_dist",
                "inter_cluster_dist", "intra_inter_ratio"]:
        log.info("%-25s  %10.4f  %10.4f", key, metrics_cen[key], metrics_fed[key])
    log.info("=" * 55)

    # UMAP projections
    coords_cen = run_umap(emb_cen, args.umap_neighbors)
    coords_fed = run_umap(emb_fed, args.umap_neighbors)

    # Individual plots
    plot_umap_single(coords_cen, cluster_labels,
                     f"Centralized encoder — {len(np.unique(cluster_labels))} clusters"
                     f" | {len(X):,} windows | best loss=0.2111",
                     out_dir / "latent_umap_centralized.png")

    plot_umap_single(coords_fed, cluster_labels,
                     f"Federated encoder (20 rounds, 15 clients) —"
                     f" {len(np.unique(cluster_labels))} clusters | {len(X):,} windows"
                     f" | best loss=0.2764",
                     out_dir / "latent_umap_federated.png")

    # Side-by-side comparison
    plot_umap_comparison(coords_cen, coords_fed, cluster_labels,
                         metrics_cen, metrics_fed,
                         out_dir / "latent_umap_comparison.png")

    # Loss curves
    fed_losses = load_fed_losses(Path(args.fed_log))
    if fed_losses:
        plot_loss_comparison(fed_losses, Path(args.cen_csv),
                             out_dir / "loss_curve_comparison.png")

    # Save JSON summary
    summary = {
        "centralized": metrics_cen,
        "federated":   metrics_fed,
        "delta": {
            k: round(metrics_cen[k] - metrics_fed[k], 4)
            for k in ["silhouette", "davies_bouldin",
                      "intra_cluster_dist", "inter_cluster_dist", "intra_inter_ratio"]
        },
        "training": {
            "federated_best_loss":   min(fed_losses) if fed_losses else None,
            "centralized_best_loss": 0.2111,
            "federated_rounds":      len(fed_losses),
            "centralized_epochs":    100,
        }
    }
    json_path = out_dir / "metrics_comparison.json"
    json_path.write_text(json.dumps(summary, indent=2))
    log.info("Saved → %s", json_path.name)
    log.info("Done. All outputs in %s/", out_dir)


if __name__ == "__main__":
    main()
