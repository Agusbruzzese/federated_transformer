"""
SpintronicEncoder(por decirle de algun modo) — shared model architecture
=============================================
Used identically by the server (pre-training) and every client (fine-tuning).
Both containers mount this file from the same source so architecture never drifts.

Architecture rationale
----------------------
Input  : 70 HHT features per 60s window --> after the extraction, a lot of features 
Hidden : 128 → 64  (BatchNorm + ReLU + Dropout at each layer)
Latent : 32 dims   (this is what gets federated and compared)
Projection head : 32 → 32 → 32  (used only during contrastive training,
                                  discarded for inference/clustering)

The projection head is standard practice from SimCLR — it prevents the
contrastive loss from collapsing the encoder representations.
"""
#quick reminder for me about the breakdown of features
"""
dc_level      # mean value of the raw signal in the window
dc_range      # peak-to-peak amplitude (max - min)
rms           # root mean square — overall signal power
dc_gradient   # (last_sample - first_sample) / 60s — is field growing or shrinking
energy_mean   # mean of per-second energy chunks
energy_std    # variability of energy within the window
energy_max    # peak energy in any single second
burst_ratio   # peak_energy / mean_energy — detects sudden EM bursts
n_imfs        # how many IMFs EMD actually found (up to 8)"""
#For each of the 8 IMFs, Hilbert gives you:
"""
imfN_mean_ia      # mean instantaneous amplitude — average oscillation strength
imfN_std_ia       # std of instantaneous amplitude — is the oscillation steady or pulsing
imfN_max_ia       # peak instantaneous amplitude
imfN_mean_if      # mean instantaneous frequency — dominant Hz at this time scale
imfN_std_if       # std of instantaneous frequency — is the frequency stable
imfN_energy       # total energy carried by this IMF
imfN_energy_ratio # fraction of total signal energy in this IMF"""
#all thanks to the EMD library <3
#https://emd.readthedocs.io/en/stable/emd_tutorials/02_spectrum_analysis/emd_tutorial_02_spectrum_01_hilberthuang.html
#then the clustering happens:
"""
dc_relative   # dc_level minus the 120min rolling median baseline
dc_range_rel  # dc_range minus the quiet floor (10th percentile rolling)
rms_rel       # rms minus the quiet floor
dc_baseline   # the rolling median itself (used internally)
t_hours       # time of window since midnight in hours"""
#So we have:
"""9 basic stats
+ 56 IMF features (7 × 8)
+ 5 DC-relative features
= 70"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── constants ──────────────────────────────────────────────────────────────────
N_FEATURES  = 70    # output of spintronic HHT feature extraction as described in the top 
LATENT_DIM  = 32    # latent space dimension that gets aggregated
HIDDEN_1    = 128
HIDDEN_2    = 64


# ── encoder ───────────────────────────────────────────────────────────────────

class SpintronicEncoder(nn.Module):
    """
    Maps a window feature vector → latent embedding.
    Only the encoder trunk weights are sent to the server each round.
    The projection head stays local and is re-initialized each round.
    """

    def __init__(
        self,
        n_features: int = N_FEATURES,
        latent_dim: int = LATENT_DIM,
    ):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(n_features, HIDDEN_1),
            nn.BatchNorm1d(HIDDEN_1),
            nn.ReLU(),
            nn.Dropout(0.20),

            nn.Linear(HIDDEN_1, HIDDEN_2),
            nn.BatchNorm1d(HIDDEN_2),
            nn.ReLU(),
            nn.Dropout(0.10),

            nn.Linear(HIDDEN_2, latent_dim),
        )

        # Projection head — contrastive training only, not federated
        self.projector = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor):
        """
        Returns
        -------
        z : trunk embedding  (latent_dim,)  — used for clustering / inference
        p : projected embedding             — used for contrastive loss only
        """
        z = self.trunk(x)
        p = self.projector(z)
        return z, p

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Inference-only: returns trunk embedding without projection."""
        with torch.no_grad():
            z, _ = self.forward(x)
        return z


# ── contrastive loss ──────────────────────────────────────────────────────────

class NTXentLoss(nn.Module):
    """
    NT-Xent (Normalized Temperature-scaled Cross-Entropy) loss.

    Positive pairs  : windows that share the same KMeans pseudo-label
    Negative pairs  : windows from different pseudo-label clusters

    This is the bridge between the unsupervised clustering output and the
    supervised-style contrastive signal the encoder needs to learn from.
    The pseudo-labels never leave the client — only the gradient signal does.

    Parameters
    ----------
    temperature : float
        Lower = sharper separation in latent space.
        0.5 is a safe default; tune down toward 0.2 if clusters collapse.
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        projections: torch.Tensor,   # (N, latent_dim)
        labels: torch.Tensor,        # (N,) integer pseudo-labels
    ) -> torch.Tensor:

        N = projections.shape[0]
        device = projections.device

        # L2-normalize so cosine similarity = dot product
        proj = F.normalize(projections, dim=1)

        # Cosine similarity matrix, temperature-scaled
        sim = torch.matmul(proj, proj.T) / self.temperature  # (N, N)

        # Positive mask: same cluster label, excluding self
        lbl = labels.unsqueeze(1)                            # (N, 1)
        pos_mask = (lbl == lbl.T).float()                   # (N, N)
        pos_mask -= torch.eye(N, device=device)             # remove diagonal

        # Negative mask: different cluster label
        neg_mask = 1.0 - (lbl == lbl.T).float()

        # Numerically stable softmax
        sim_max, _ = sim.max(dim=1, keepdim=True)
        exp_sim    = torch.exp(sim - sim_max)

        # Per-row: sum of positives / (sum of positives + sum of negatives)
        pos_sum = (exp_sim * pos_mask).sum(dim=1)
        neg_sum = (exp_sim * neg_mask).sum(dim=1)
        denom   = pos_sum + neg_sum + 1e-8

        loss_per_row = -torch.log(pos_sum / denom + 1e-8)

        # Only average over rows that actually have at least one positive pair
        # (rows where all other windows are in different clusters contribute
        #  nothing useful to the gradient)
        has_positive = pos_mask.sum(dim=1) > 0
        if has_positive.sum() == 0:
            return torch.zeros(1, device=device, requires_grad=True).squeeze()

        return loss_per_row[has_positive].mean()


# ── FedProx proximal term ─────────────────────────────────────────────────────

def proximal_loss(
    local_params: list[torch.Tensor],
    global_params: list[torch.Tensor],
    mu: float,
) -> torch.Tensor:
    """
    FedProx regularization term.
    Penalizes local model for drifting too far from the global model.
    Added to the contrastive loss during local fine-tuning.

    Parameters
    ----------
    mu : float
        Proximal strength. Higher = closer to pure FedAvg.
        Recommended range for spatially correlated clients: 0.05 – 0.2
    """
    prox = torch.tensor(0.0)
    for lp, gp in zip(local_params, global_params):
        prox = prox + ((lp - gp.detach()) ** 2).sum()
    return (mu / 2.0) * prox


# ── weight helpers (Flower interface) ─────────────────────────────────────────

def get_trunk_weights(model: SpintronicEncoder) -> list:
    """Extract trunk weights as numpy arrays for Flower transmission."""
    import numpy as np
    return [v.cpu().numpy() for v in model.trunk.state_dict().values()]


def set_trunk_weights(model: SpintronicEncoder, weights: list) -> None:
    """Load trunk weights received from Flower server."""
    import numpy as np
    import collections
    state = collections.OrderedDict(
        {k: torch.tensor(v)
         for k, v in zip(model.trunk.state_dict().keys(), weights)}
    )
    model.trunk.load_state_dict(state, strict=True)
