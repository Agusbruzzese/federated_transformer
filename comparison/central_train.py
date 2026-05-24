"""
central_train.py — Centralized contrastive training (non-federated baseline).

Trains SpintronicEncoder with NT-Xent loss on all 40 832 labeled windows from
central_clustering.py. No FedProx, no partitioning — all data seen in every epoch.

This is the upper-bound benchmark: if federated training approaches this model's
latent-space quality, the privacy cost of federation is low.

Usage
-----
    python central_train.py \
        --features   ./central_cache/features_labeled.parquet \
        --labels     ./central_cache/labels_all.npy \
        --pretrained ./central_cache/central_pretrained.pt \
        --epochs     100 \
        --save_dir   ./central_checkpoints

Outputs
-------
    central_checkpoints/central_model_epoch_NNN.pt   trunk weights every 10 epochs
    central_checkpoints/central_model_final.pt        final trunk weights
    central_checkpoints/loss_central.csv              epoch, loss, lr
"""

import sys
import csv
import argparse
import logging
import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "code" / "shared"))
from model import SpintronicEncoder, NTXentLoss, N_FEATURES, set_trunk_weights

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Feature columns — must match the order expected by the model (same as pretrain_server.py)
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


def load_data(features_path: Path, labels_path: Path):
    log.info("Loading features from %s", features_path)
    df = pd.read_parquet(features_path)

    available = [c for c in FEATURE_COLS if c in df.columns]
    missing   = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        log.warning("Missing %d feature columns (will be zero-padded): %s", len(missing), missing[:5])

    X = df[available].fillna(0.0).values.astype(np.float32)
    if X.shape[1] < N_FEATURES:
        pad = np.zeros((X.shape[0], N_FEATURES - X.shape[1]), dtype=np.float32)
        X   = np.hstack([X, pad])
    elif X.shape[1] > N_FEATURES:
        X = X[:, :N_FEATURES]

    log.info("Loading labels from %s", labels_path)
    labels = np.load(labels_path).astype(np.int64)

    assert len(X) == len(labels), f"Feature/label length mismatch: {len(X)} vs {len(labels)}"
    log.info("Dataset: %d windows × %d features  |  %d unique labels",
             X.shape[0], X.shape[1], len(np.unique(labels)))

    X_t = torch.tensor(X, device=DEVICE)
    y_t = torch.tensor(labels, device=DEVICE)
    return X_t, y_t


def train(
    features_path: Path,
    labels_path: Path,
    pretrained_path: Path | None,
    save_dir: Path,
    epochs: int,
    batch_size: int,
    lr: float,
):
    save_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_data(features_path, labels_path)

    model     = SpintronicEncoder(n_features=N_FEATURES).to(DEVICE)
    criterion = NTXentLoss(temperature=0.5)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    if pretrained_path and pretrained_path.exists():
        trunk_state = torch.load(pretrained_path, map_location=DEVICE)
        model.trunk.load_state_dict(trunk_state)
        log.info("Loaded pre-trained trunk from %s", pretrained_path)
    else:
        log.info("No pre-trained weights — starting from random init")

    dataset = TensorDataset(X, y)
    loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    log.info(
        "Training: %d windows | %d batches/epoch | epochs=%d | device=%s",
        len(X), len(loader), epochs, DEVICE,
    )

    csv_path = save_dir / "loss_central.csv"
    with open(csv_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "loss", "lr"])

    best_loss = float("inf")
    best_path = save_dir / "central_model_best.pt"

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches  = 0

        for X_batch, y_batch in loader:
            if y_batch.unique().numel() < 2:
                continue
            optimizer.zero_grad()
            _, proj = model(X_batch)
            loss    = criterion(proj, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        current_lr = scheduler.get_last_lr()[0]

        with open(csv_path, "a", newline="") as f:
            csv.writer(f).writerow([epoch, f"{avg_loss:.6f}", f"{current_lr:.2e}"])

        if epoch % 10 == 0 or epoch == 1:
            log.info("Epoch %3d/%d  loss=%.4f  lr=%.2e", epoch, epochs, avg_loss, current_lr)
            ckpt = save_dir / f"central_model_epoch_{epoch:03d}.pt"
            torch.save(model.trunk.state_dict(), ckpt)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.trunk.state_dict(), best_path)

    final_path = save_dir / "central_model_final.pt"
    torch.save(model.trunk.state_dict(), final_path)
    log.info("=" * 60)
    log.info("Training complete. Best loss=%.4f", best_loss)
    log.info("Final model  → %s", final_path)
    log.info("Best model   → %s", best_path)
    log.info("Loss curve   → %s", csv_path)
    log.info("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Centralized SpintronicEncoder training")
    parser.add_argument("--features",   default="../code/central_cache/features_labeled.parquet")
    parser.add_argument("--labels",     default="../code/central_cache/labels_all.npy")
    parser.add_argument("--pretrained", default="../code/central_cache/central_pretrained.pt",
                        help="Pre-trained trunk weights from Step 4 (optional)")
    parser.add_argument("--epochs",     type=int,   default=100)
    parser.add_argument("--batch_size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--save_dir",   default="../code/central_checkpoints")
    args = parser.parse_args()

    train(
        features_path   = Path(args.features).resolve(),
        labels_path     = Path(args.labels).resolve(),
        pretrained_path = Path(args.pretrained).resolve(),
        save_dir        = Path(args.save_dir).resolve(),
        epochs          = args.epochs,
        batch_size      = args.batch_size,
        lr              = args.lr,
    )


if __name__ == "__main__":
    main()
