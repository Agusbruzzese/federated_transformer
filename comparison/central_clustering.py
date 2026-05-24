"""
central_clustering.py — Centralized DC-relative clustering over all BSD data.

Runs the same 7-step clustering pipeline used by the federated clients on the
full features_all.parquet produced by central_pipeline.py.

Two corrections vs. the per-day client behaviour:
  1. t_unix is rebased to an absolute timeline (t_hours × 3600) so that
     add_dc_relative sorts correctly across 18 days instead of by time-of-day.
  2. night_pct in main_clustering uses hour-of-day (t_unix % 86400 / 3600)
     instead of cumulative hours, so the baseline cluster heuristic still works.

Usage
-----
    python central_clustering.py \
        --features ./central_cache/features_all.parquet \
        --out_dir  ./central_cache

Outputs
-------
    central_cache/features_labeled.parquet   DataFrame + all cluster columns
    central_cache/labels_all.npy             Integer label array (aligned to rows)
    central_cache/label_map.json             {cluster_name: integer} mapping
"""

import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Add client/ to path so we can import the original modules unchanged
sys.path.insert(0, str(Path(__file__).parent.parent / "code" / "client"))

from clustering import (
    add_dc_relative,
    flag_temperature_drift,
    sub_clustering,
    encode_labels,
    K_MAIN,
    MAIN_FEATURES,
    _available,
    prepare_X,
)
from sklearn.cluster import KMeans


# ── Fixed main_clustering that uses hour-of-day for night_pct ─────────────────

def main_clustering_central(df: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """
    Identical to clustering.main_clustering except night_pct uses
    t_hour_of_day (0–24 h repeating) instead of cumulative t_hours.
    """
    log.info("=== MAIN CLUSTERING (k=%d) ===", K_MAIN)
    feat_cols = _available(df, MAIN_FEATURES)
    _, X_pca  = prepare_X(df, feat_cols)
    labels    = KMeans(n_clusters=K_MAIN, random_state=42, n_init=20).fit_predict(X_pca)
    df["cluster_main"] = labels

    # hour-of-day derived from absolute t_unix (seconds from epoch)
    t_hod = (df["t_unix"] % 86400) / 3600.0

    profiles = {}
    for c in range(K_MAIN):
        mask = labels == c
        profiles[c] = {
            "n":            int(mask.sum()),
            "imf1_er":      float(df["imf1_energy_ratio"][mask].mean()),
            "dc_range_rel": float(df["dc_range_rel"][mask].mean()),
            "burst_ratio":  float(df["burst_ratio"][mask].mean()),
            "night_pct":    float(((t_hod[mask] < 7) | (t_hod[mask] > 20)).mean()),
        }

    baseline_c = max(profiles, key=lambda c: profiles[c]["imf1_er"] + profiles[c]["night_pct"])
    df["is_baseline"] = labels == baseline_c

    for c, p in sorted(profiles.items()):
        tag = " ← BASELINE" if c == baseline_c else ""
        log.info(
            "  C%d  n=%5d (%4.1f%%)  imf1_er=%.3f  dc_range_rel=%6.0f"
            "  burst=%.2f  night=%.0f%%%s",
            c, p["n"], p["n"] / len(df) * 100,
            p["imf1_er"], p["dc_range_rel"], p["burst_ratio"],
            p["night_pct"] * 100, tag,
        )

    return df, baseline_c


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Centralized clustering pipeline")
    parser.add_argument("--features", default="../code/central_cache/features_all.parquet")
    parser.add_argument("--out_dir",  default="../code/central_cache")
    args = parser.parse_args()

    features_path = Path(args.features).resolve()
    out_dir       = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading %s …", features_path)
    df = pd.read_parquet(features_path)
    log.info("Loaded: %d windows × %d columns", len(df), len(df.columns))

    # ── Steps 1+2 applied per-date ─────────────────────────────────────────────
    # DC-relative normalization and drift detection MUST be done per date.
    # Reason: the sensor DC level can shift by thousands of ADC between recording
    # runs (power-cycle, temperature reset). If we compute a rolling baseline
    # across run boundaries the diff() sees those inter-run jumps as drift and
    # flags 99%+ of windows. Each federated client processed exactly one day, so
    # per-date normalization is the scientifically correct centralized equivalent.
    log.info("=== DC-RELATIVE BASELINE REMOVAL + DRIFT DETECTION (per date) ===")
    date_frames = []
    total_drift = 0

    for date, grp in df.groupby("date"):
        grp = grp.copy().sort_values("t_unix").reset_index(drop=True)
        grp = add_dc_relative(grp)
        grp = flag_temperature_drift(grp)
        n_d = int(grp["is_drift"].sum())
        total_drift += n_d
        log.info("  %s  %d windows / %d drift-flagged", date, len(grp), n_d)
        date_frames.append(grp)

    df = pd.concat(date_frames, ignore_index=True)

    # Rebase t_unix to absolute timeline so main_clustering sorts correctly
    df["t_unix"] = df["t_hours"] * 3600.0
    df = df.sort_values("t_unix").reset_index(drop=True)

    log.info("Windows: %d total / %d drift-flagged (informational only — all enter clustering)",
             len(df), total_drift)

    # ── Step 3+4: Main clustering (with corrected night_pct) ──────────────────
    # Drift flag is informational: drift windows still receive cluster labels and
    # participate in contrastive training, exactly as in the federated clients.
    df, baseline_c = main_clustering_central(df)

    # ── Step 5+6: Sub-clustering with physical validity guards ────────────────
    df = sub_clustering(df, baseline_c)

    # ── Step 7: Integer label encoding ────────────────────────────────────────
    labels, label_map = encode_labels(df)

    # ── Save outputs ──────────────────────────────────────────────────────────
    labeled_path  = out_dir / "features_labeled.parquet"
    labels_path   = out_dir / "labels_all.npy"
    labelmap_path = out_dir / "label_map.json"

    df.to_parquet(labeled_path, index=False)
    np.save(labels_path, labels)
    with open(labelmap_path, "w") as f:
        json.dump(label_map, f, indent=2)

    log.info("=" * 60)
    log.info("Cluster distribution:")
    for name, count in df["cluster_full"].value_counts().items():
        log.info("  %-20s  %5d  (%.1f%%)", name, count, 100 * count / len(df))
    log.info("Unique clusters : %d", df["cluster_full"].nunique())
    log.info("Total windows   : %d", len(df))
    log.info("Saved → %s", labeled_path.name)
    log.info("Saved → %s", labels_path.name)
    log.info("Saved → %s", labelmap_path.name)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
