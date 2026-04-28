"""

Flower federated learning client for the spintronic sensor array.

What this client does each round
---------------------------------
1. Receive global encoder weights from server
2. Load local feature dataset (pipeline.py, runs once at startup)
3. Run DC-relative clustering to get pseudo-labels (clustering.py)
4. Fine-tune encoder with NT-Xent contrastive loss + FedProx regularization
5. Send updated trunk weights back to server
#https://discuss.pytorch.org/t/contrastive-learning/175636
Environment variables
---------------------
CLIENT_ID       : integer 1–15
SERVER_ADDRESS  : host:port of Flower server         [server:8080] #change this
DATA_DIR        : where BSD files are mounted         [/data]
TARGET_DATE     : YYYYMMDD                            [20250206]
SIMULATE_DATA   : 'true' for dry-run without BSD      [false]
LOCAL_EPOCHS    : fine-tuning epochs per FL round     [5]
BATCH_SIZE      : contrastive training batch size     [64]
LR              : learning rate                       [1e-3]
PROX_MU         : FedProx proximal strength           [0.1]
"""
#https://flower.ai/docs/baselines/fedprox.html

import os
import sys
import logging
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Add shared/ to path so model.py is importable in both containers
sys.path.insert(0, "/app/shared")

import flwr as fl

from pipeline   import run_pipeline
from clustering import run_clustering

from model import (
    SpintronicEncoder,
    NTXentLoss,
    proximal_loss,
    get_trunk_weights,
    set_trunk_weights,
    N_FEATURES,
)

# ── logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [CLIENT-%(CLIENT_ID)s] %(message)s"
        .replace("%(CLIENT_ID)s", os.environ.get("CLIENT_ID", "?")),
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── hyper-parameters from environment ─────────────────────────────────────────

CLIENT_ID      = int(os.environ.get("CLIENT_ID", "1"))
SERVER_ADDRESS = os.environ.get("SERVER_ADDRESS", "server:8080")
LOCAL_EPOCHS   = int(os.environ.get("LOCAL_EPOCHS", "5"))
BATCH_SIZE     = int(os.environ.get("BATCH_SIZE", "64"))
LR             = float(os.environ.get("LR", "1e-3"))
PROX_MU        = float(os.environ.get("PROX_MU", "0.1"))

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info("Device: %s  |  CLIENT_ID: %d", DEVICE, CLIENT_ID)


# ── feature column selection ───────────────────────────────────────────────────

# Must match pipeline.py output and be exactly N_FEATURES=70 columns.
# Order matters — same order used by server pre-training.
FEATURE_COLS = [
    "dc_level", "dc_range", "rms", "dc_gradient",
    "energy_mean", "energy_std", "energy_max", "burst_ratio", "n_imfs",
]
for i in range(1, 9):
    FEATURE_COLS += [
        f"imf{i}_mean_ia", f"imf{i}_std_ia", f"imf{i}_max_ia",
        f"imf{i}_mean_if", f"imf{i}_std_if",
        f"imf{i}_energy", f"imf{i}_energy_ratio",
    ]
# That gives 9 + 8*7 = 65 columns; pad to 70 with dc-relative computed features
FEATURE_COLS += ["dc_relative", "dc_range_rel", "rms_rel", "dc_baseline", "t_hours"]
assert len(FEATURE_COLS) == 70, f"Expected 70 features, got {len(FEATURE_COLS)}"


def df_to_tensor(df) -> torch.Tensor:
    """Select feature columns → fill NaN → float32 tensor."""
    available = [c for c in FEATURE_COLS if c in df.columns]
    X = df[available].fillna(0.0).values.astype(np.float32)

    # Pad to N_FEATURES if some columns are missing (e.g. simulate mode)
    if X.shape[1] < N_FEATURES:
        pad = np.zeros((X.shape[0], N_FEATURES - X.shape[1]), dtype=np.float32)
        X   = np.hstack([X, pad])
    elif X.shape[1] > N_FEATURES:
        X = X[:, :N_FEATURES]

    return torch.tensor(X, device=DEVICE)


# ── Flower client ──────────────────────────────────────────────────────────────

class SpintronicClient(fl.client.NumPyClient):

    def __init__(self):
        # Stagger EMD startup so clients don't all hammer RAM simultaneously
        # Client 1 waits 60s, client 2 waits 120s, ..., client 15 waits 900s
        stagger = int(os.environ.get("CLIENT_ID", 1)) * \
                int(os.environ.get("EMD_STAGGER_SECONDS", "60"))
        log.info("Waiting %ds before starting EMD (stagger)...", stagger)
        time.sleep(stagger)

        log.info("Running feature extraction pipeline...")
        self.df_raw = run_pipeline()
        # ── Run pipeline ONCE at startup ──────────────────────────────────────
        log.info("Running feature extraction pipeline...")
        self.df_raw = run_pipeline()

        # ── Run clustering ONCE at startup ────────────────────────────────────
        log.info("Running DC-relative clustering...")
        self.df, self.labels_np, self.label_map = run_clustering(self.df_raw)

        log.info(
            "Cluster distribution:\n%s",
            self.df["cluster_full"].value_counts().to_string()
        )

        # ── Build tensors ─────────────────────────────────────────────────────
        self.X      = df_to_tensor(self.df)                          # (N, 70)
        self.labels = torch.tensor(self.labels_np, device=DEVICE)   # (N,)
        self.N      = self.X.shape[0]

        # ── Encoder + loss ────────────────────────────────────────────────────
        self.model     = SpintronicEncoder(n_features=N_FEATURES).to(DEVICE)
        self.criterion = NTXentLoss(temperature=0.5)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)

        # Snapshot of global weights for FedProx proximal term
        self.global_params: list[torch.Tensor] = []

        log.info(
            "Client ready: %d windows | %d clusters | encoder params: %d",
            self.N,
            self.df["cluster_full"].nunique(),
            sum(p.numel() for p in self.model.parameters()),
        )

    # ── Flower interface ───────────────────────────────────────────────────────

    def get_parameters(self, config: dict) -> list[np.ndarray]:
        """Return current trunk weights to the server."""
        return get_trunk_weights(self.model)

    def set_parameters(self, parameters: list[np.ndarray]) -> None:
        """Load trunk weights received from server."""
        set_trunk_weights(self.model, parameters)
        # Snapshot global params for FedProx proximal term
        self.global_params = [p.clone().detach()
                              for p in self.model.trunk.parameters()]

    def fit(
        self,
        parameters: list[np.ndarray],
        config: dict,
    ) -> tuple[list[np.ndarray], int, dict]:
        """
        Receive global weights → fine-tune locally → return updated weights.

        The local loss is:
            L = L_contrastive(projections, pseudo_labels)
              + FedProx_proximal(local_params, global_params, mu)

        Pseudo-labels are the KMeans cluster assignments computed at startup.
        They do NOT change between rounds — only the encoder updates.
        """
        fl_round = int(config.get("fl_round", 0))
        log.info("=== FL Round %d — fit ===", fl_round)

        self.set_parameters(parameters)
        self.model.train()

        dataset = TensorDataset(self.X, self.labels)
        loader  = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

        total_loss = 0.0
        n_batches  = 0

        for epoch in range(LOCAL_EPOCHS):
            epoch_loss = 0.0
            for X_batch, lbl_batch in loader:

                # Skip batches where all windows have the same label
                # (contrastive loss needs at least 2 distinct classes)
                if lbl_batch.unique().numel() < 2:
                    continue

                self.optimizer.zero_grad()

                _, proj = self.model(X_batch)

                # Contrastive loss
                loss = self.criterion(proj, lbl_batch)

                # FedProx proximal term
                if self.global_params:
                    prox = proximal_loss(
                        list(self.model.trunk.parameters()),
                        self.global_params,
                        mu=PROX_MU,
                    )
                    loss = loss + prox

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                epoch_loss += loss.item()
                n_batches  += 1

            if n_batches > 0:
                log.info(
                    "  Epoch %d/%d  loss=%.4f",
                    epoch + 1, LOCAL_EPOCHS, epoch_loss / max(n_batches, 1)
                )
            total_loss += epoch_loss

        avg_loss = total_loss / max(n_batches, 1)
        log.info("Round %d done  avg_loss=%.4f", fl_round, avg_loss)

        return (
            get_trunk_weights(self.model),
            self.N,
            {
                "loss":        avg_loss,
                "n_clusters":  int(self.df["cluster_full"].nunique()),
                "n_baseline":  int((self.df["cluster_full"] == "baseline").sum()),
                "n_events":    int((self.df["cluster_full"] != "baseline").sum()),
                "client_id":   CLIENT_ID,
            },
        )

    def evaluate(
        self,
        parameters: list[np.ndarray],
        config: dict,
    ) -> tuple[float, int, dict]:
        """
        Evaluate current global model on local data.
        Metric: mean contrastive loss over all local windows (no gradient).
        """
        self.set_parameters(parameters)
        self.model.eval()

        loader = DataLoader(
            TensorDataset(self.X, self.labels),
            batch_size=BATCH_SIZE,
            shuffle=False,
            drop_last=False,
        )

        total_loss = 0.0
        n_batches  = 0

        with torch.no_grad():
            for X_batch, lbl_batch in loader:
                if lbl_batch.unique().numel() < 2:
                    continue
                _, proj = self.model(X_batch)
                loss = self.criterion(proj, lbl_batch)
                total_loss += loss.item()
                n_batches  += 1

        avg_loss = total_loss / max(n_batches, 1)
        log.info("Evaluate: loss=%.4f  n=%d", avg_loss, self.N)

        return avg_loss, self.N, {"loss": avg_loss, "client_id": CLIENT_ID}


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import time
    log.info("Starting Flower client (CLIENT_ID=%d, SERVER=%s)", CLIENT_ID, SERVER_ADDRESS)

    # Give server a moment after TCP healthcheck passes before gRPC is ready
    time.sleep(5 + CLIENT_ID * 0.5)  # stagger clients so they don't all hammer server at once

    spintronic_client = SpintronicClient()  # pipeline + clustering run here

    for attempt in range(10):
        try:
            log.info("Connection attempt %d/10...", attempt + 1)
            fl.client.start_numpy_client(
                server_address=SERVER_ADDRESS,
                client=spintronic_client,
            )
            break  # clean exit after all rounds complete
        except Exception as e:
            log.warning("Connection failed: %s — retrying in 10s", e)
            time.sleep(10)
    else:
        log.error("Could not connect to server after 10 attempts — exiting")
