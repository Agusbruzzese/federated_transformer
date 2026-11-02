"""
spintronic_fl/server/pretrain_server.py
=========================================
Pre-trains the SpintronicEncoder on the full Zenodo feature set
BEFORE federation begins. This gives every client a sensible initialization
rather than starting from random weights.

Usage
-----
    python pretrain_server.py \
        --features /data/features_20250206.parquet \
        --output   /app/global_model.pt \
        --epochs   50

The pre-training uses the same NT-Xent contrastive loss as the clients,
with KMeans pseudo-labels derived from the centralized clustering of the
full Zenodo dataset.
"""

import sys
import argparse
import logging
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, "/app/shared")
from model import SpintronicEncoder, NTXentLoss, N_FEATURES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [PRETRAIN] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── feature columns — same order as client.py ─────────────────────────────────

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


def load_features(path: Path) -> torch.Tensor:
    log.info("Loading features from %s", path)
    if path.suffix == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].fillna(0.0).values.astype(np.float32)

    if X.shape[1] < N_FEATURES:
        pad = np.zeros((X.shape[0], N_FEATURES - X.shape[1]), dtype=np.float32)
        X   = np.hstack([X, pad])
    elif X.shape[1] > N_FEATURES:
        X = X[:, :N_FEATURES]

    log.info("Loaded %d windows × %d features", X.shape[0], X.shape[1])
    return torch.tensor(X, device=DEVICE)


def pseudo_labels_kmeans(X: torch.Tensor, k: int = 8) -> torch.Tensor:
    """
    Run KMeans on the full feature set to generate pre-training pseudo-labels.
    k=8 gives enough resolution without over-segmenting on a single day.
    """
    log.info("Generating pseudo-labels with KMeans k=%d...", k)
    Xnp = StandardScaler().fit_transform(X.cpu().numpy())
    km   = KMeans(n_clusters=k, random_state=42, n_init=20)
    lbl  = km.fit_predict(Xnp)
    log.info("Pseudo-label distribution: %s",
             {c: int((lbl == c).sum()) for c in range(k)})
    return torch.tensor(lbl, dtype=torch.long, device=DEVICE)


def pretrain(
    features_path: Path,
    output_path: Path,
    epochs: int = 50,
    batch_size: int = 128,
    lr: float = 1e-3,
    k_pretrain: int = 8,
):
    X      = load_features(features_path)
    labels = pseudo_labels_kmeans(X, k=k_pretrain)

    model     = SpintronicEncoder(n_features=N_FEATURES).to(DEVICE)
    criterion = NTXentLoss(temperature=0.5)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    dataset = TensorDataset(X, labels)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    log.info(
        "Pre-training: %d windows  |  k=%d  |  epochs=%d  |  batch=%d  |  device=%s",
        len(X), k_pretrain, epochs, batch_size, DEVICE
    )

    best_loss = float("inf")
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        n_batches  = 0

        for X_batch, lbl_batch in loader:
            if lbl_batch.unique().numel() < 2:
                continue
            optimizer.zero_grad()
            _, proj = model(X_batch)
            loss    = criterion(proj, lbl_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            log.info("Epoch %3d/%d  loss=%.4f  lr=%.2e",
                     epoch + 1, epochs, avg_loss,
                     scheduler.get_last_lr()[0])

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.trunk.state_dict(), output_path)

    log.info("Pre-training complete. Best loss=%.4f. Saved to %s", best_loss, output_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True,
                        help="Path to features parquet or CSV")
    parser.add_argument("--output",   type=Path, default=Path("/app/global_model.pt"))
    parser.add_argument("--epochs",   type=int,  default=50)
    parser.add_argument("--batch",    type=int,  default=128)
    parser.add_argument("--lr",       type=float, default=1e-3)
    parser.add_argument("--k",        type=int,  default=8,
                        help="KMeans k for pre-training pseudo-labels")
    args = parser.parse_args()

    pretrain(
        features_path=args.features,
        output_path=args.output,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        k_pretrain=args.k,
    )
