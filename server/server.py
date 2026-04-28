"""
spintronic_fl/server/server.py
================================
Flower federated learning server for the spintronic sensor array.

Responsibilities
----------------
1. Load pre-trained encoder weights (trained on Zenodo Spintronic features).
   If no pre-trained weights exist, initializes from scratch and logs a warning.
2. Run FedProx aggregation across all 15 clients for N rounds.
3. Log per-round metrics: mean loss, cluster distribution summary.
4. Save the aggregated global model after each round.
5. Broadcast round number to clients via fit_config.

Environment variables
---------------------
FL_ROUNDS          : number of federation rounds       [10]
MIN_CLIENTS        : minimum clients before round start [15]
PROX_MU            : FedProx proximal strength          [0.1]
PRETRAINED_WEIGHTS : path to pre-trained .pt file       [/app/global_model.pt]
SAVE_DIR           : where to save round checkpoints    [/app/checkpoints]
PORT               : server listen port                 [8080]
"""

import os
import sys
import logging
import numpy as np
from pathlib import Path
from typing import Optional

sys.path.insert(0, "/app/shared")

import flwr as fl
from flwr.server.strategy import FedProx
from flwr.common import (
    Metrics, Parameters, FitIns, FitRes,
    EvaluateIns, EvaluateRes, Scalar,
    ndarrays_to_parameters, parameters_to_ndarrays,
)

import torch
from model import SpintronicEncoder, get_trunk_weights, N_FEATURES

# ── logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [SERVER] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ── configuration ──────────────────────────────────────────────────────────────

FL_ROUNDS          = int(os.environ.get("FL_ROUNDS", "10"))
MIN_CLIENTS        = int(os.environ.get("MIN_CLIENTS", "15"))
PROX_MU            = float(os.environ.get("PROX_MU", "0.1"))
PRETRAINED_WEIGHTS = Path(os.environ.get("PRETRAINED_WEIGHTS", "/app/global_model.pt"))
SAVE_DIR           = Path(os.environ.get("SAVE_DIR", "/app/checkpoints"))
PORT               = int(os.environ.get("PORT", "8080"))

SAVE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log.info("Server device: %s", DEVICE)


# ── pre-trained weight loading ─────────────────────────────────────────────────

def load_initial_parameters() -> Parameters:
    """
    Load pre-trained encoder trunk weights.
    If no checkpoint exists, returns random initialization with a warning.
    
    Pre-training is done separately (pretrain_server.py) on the full Zenodo
    feature set before federation begins.
    """
    model = SpintronicEncoder(n_features=N_FEATURES).to(DEVICE)

    if PRETRAINED_WEIGHTS.exists():
        try:
            state = torch.load(PRETRAINED_WEIGHTS, map_location=DEVICE)
            # Accept both full model state_dict and trunk-only
            if all(k.startswith("trunk.") for k in state.keys()):
                model.trunk.load_state_dict(state)
            else:
                model.load_state_dict(state, strict=False)
            log.info("Loaded pre-trained weights from %s", PRETRAINED_WEIGHTS)
        except Exception as e:
            log.warning("Failed to load pre-trained weights: %s — using random init", e)
    else:
        log.warning(
            "No pre-trained weights at %s — starting from random init. "
            "Run pretrain_server.py first for best results.",
            PRETRAINED_WEIGHTS
        )

    weights = get_trunk_weights(model)
    return ndarrays_to_parameters(weights)


# ── metric aggregation callbacks ───────────────────────────────────────────────

def fit_metrics_aggregation(metrics: list[tuple[int, Metrics]]) -> Metrics:
    """
    Aggregate fit metrics from all clients.
    Computes weighted mean loss and total cluster/event counts.
    """
    total_n    = sum(n for n, _ in metrics)
    avg_loss   = sum(n * m["loss"]      for n, m in metrics) / max(total_n, 1)
    total_ev   = sum(m["n_events"]      for _, m in metrics)
    total_bl   = sum(m["n_baseline"]    for _, m in metrics)
    n_clusters = [m["n_clusters"]       for _, m in metrics]

    log.info(
        "Round metrics — avg_loss=%.4f  events=%d  baseline=%d  "
        "clusters_per_client=%s",
        avg_loss, total_ev, total_bl,
        [f"C{int(m['client_id'])}:{int(m['n_clusters'])}" for _, m in metrics]
    )

    return {
        "loss":            avg_loss,
        "total_events":    total_ev,
        "total_baseline":  total_bl,
        "mean_n_clusters": float(np.mean(n_clusters)),
    }


def eval_metrics_aggregation(metrics: list[tuple[int, Metrics]]) -> Metrics:
    total_n  = sum(n for n, _ in metrics)
    avg_loss = sum(n * m["loss"] for n, m in metrics) / max(total_n, 1)
    log.info("Eval metrics — avg_loss=%.4f  total_windows=%d", avg_loss, total_n)
    return {"loss": avg_loss}


# ── custom FedProx strategy with checkpoint saving ─────────────────────────────

class SpintronicFedProx(FedProx):
    """
    Extends Flower FedProx to:
    - Inject fl_round into each client's fit config
    - Save global model checkpoint after each round
    """

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager,
    ) -> list[tuple]:
        config = {"fl_round": server_round}
        fit_ins = FitIns(parameters, config)
        clients = client_manager.sample(
            num_clients=self.min_fit_clients,
            min_num_clients=self.min_available_clients,
        )
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple],
        failures: list,
    ):
        aggregated = super().aggregate_fit(server_round, results, failures)

        if aggregated is not None:
            params, metrics = aggregated
            self._save_checkpoint(server_round, params)

        if failures:
            log.warning("Round %d: %d client(s) failed", server_round, len(failures))

        return aggregated

    def _save_checkpoint(self, round_n: int, params: Parameters):
        model = SpintronicEncoder(n_features=N_FEATURES).to(DEVICE)
        weights = parameters_to_ndarrays(params)

        import collections
        state = collections.OrderedDict(
            {k: torch.tensor(v)
             for k, v in zip(model.trunk.state_dict().keys(), weights)}
        )
        model.trunk.load_state_dict(state, strict=True)

        out = SAVE_DIR / f"global_model_round_{round_n:03d}.pt"
        torch.save(model.trunk.state_dict(), out)
        log.info("Checkpoint saved → %s", out)

        # Always overwrite latest
        latest = SAVE_DIR / "global_model_latest.pt"
        torch.save(model.trunk.state_dict(), latest)


# ── entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    initial_parameters = load_initial_parameters()

    strategy = SpintronicFedProx(
        # FedProx core
        proximal_mu=PROX_MU,

        # Participation
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=MIN_CLIENTS,
        min_evaluate_clients=MIN_CLIENTS,
        min_available_clients=MIN_CLIENTS,

        # Initial global model (pre-trained or random)
        initial_parameters=initial_parameters,

        # Metric aggregation
        fit_metrics_aggregation_fn=fit_metrics_aggregation,
        evaluate_metrics_aggregation_fn=eval_metrics_aggregation,
    )

    log.info(
        "Starting Flower server on port %d  |  rounds=%d  |  min_clients=%d  |  mu=%.2f",
        PORT, FL_ROUNDS, MIN_CLIENTS, PROX_MU
    )

    fl.server.start_server(
        server_address=f"0.0.0.0:{PORT}",
        config=fl.server.ServerConfig(num_rounds=FL_ROUNDS),
        strategy=strategy,
    )
