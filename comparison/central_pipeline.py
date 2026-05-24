"""
central_pipeline.py — Centralized EMD/HHT feature extraction over all BSD data.

Scans all_bsd/ for every unique date, runs the same sliding-window EMD pipeline
used by the federated clients, saves a per-date checkpoint parquet, then merges
everything into features_all.parquet.

Usage
-----
    python central_pipeline.py \
        --data_dir  ../all_bsd \
        --cache_dir ./central_cache \
        --out       ./central_cache/features_all.parquet

Resume: already-processed date checkpoints are reused automatically.
"""

import re
import sys
import logging
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Signal constants (must match federated clients exactly) ────────────────────
FS_RAW   = 2400
DS       = 24
FS       = FS_RAW // DS   # 100 Hz
WIN_SEC  = 60
STEP_SEC = 30
WIN_N    = WIN_SEC  * FS  # 6 000 samples
STEP_N   = STEP_SEC * FS  # 3 000 samples
MAX_IMFS = 8

# Reference epoch: first recording day — t_hours computed from this point
EPOCH_DATE = "20250204"
EPOCH_T_START_H = 15 + 24 / 60 + 13 / 3600  # 15h24m13s

BSD_PAT = re.compile(
    r"S0C0 BIN Values - (\d+) Start@ (\d{8}) - (\d{2})h(\d{2})m(\d{2})s",
    re.IGNORECASE,
)

# Date → days since epoch (for t_hours computation)
from datetime import datetime

def date_offset_hours(date_str: str) -> float:
    """Hours elapsed from EPOCH_DATE 00:00 to the start of date_str 00:00."""
    epoch = datetime.strptime(EPOCH_DATE, "%Y%m%d")
    target = datetime.strptime(date_str, "%Y%m%d")
    return (target - epoch).total_seconds() / 3600.0


# ── File discovery ─────────────────────────────────────────────────────────────

def discover_dates(data_dir: Path) -> list[str]:
    dates = set()
    for f in data_dir.iterdir():
        m = BSD_PAT.search(f.name)
        if m:
            dates.add(m.group(2))
    return sorted(dates)


def discover_files_for_date(data_dir: Path, date: str) -> list[dict]:
    files = []
    for f in data_dir.iterdir():
        m = BSD_PAT.search(f.name)
        if m is None or m.group(2) != date:
            continue
        h, mn, s = int(m.group(3)), int(m.group(4)), int(m.group(5))
        files.append({
            "index":   int(m.group(1)),
            "path":    f,
            "t_start": h * 3600 + mn * 60 + s,
        })
    files.sort(key=lambda x: x["index"])
    return files


# ── Binary loader ──────────────────────────────────────────────────────────────

def load_values(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    return np.frombuffer(raw, dtype="<i2").astype(np.float64)[0::2]


def build_day_signal(files: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    all_vals, all_times = [], []
    for f in files:
        vals  = load_values(f["path"])
        n     = len(vals)
        times = f["t_start"] + np.arange(n) / FS_RAW
        all_vals.append(vals[::DS])
        all_times.append(times[::DS])
    return np.concatenate(all_vals), np.concatenate(all_times)


# ── EMD/HHT per window ─────────────────────────────────────────────────────────

import emd  # noqa: E402 — imported after install check at top

def extract_window_features(window: np.ndarray, t_start_s: float) -> dict | None:
    if np.std(window) < 1e-6:
        return None
    try:
        imfs = emd.sift.sift(window, max_imfs=MAX_IMFS)
    except Exception as e:
        log.debug("EMD failed at t=%.1fs: %s", t_start_s, e)
        return None

    if imfs.ndim == 1:
        imfs = imfs[:, np.newaxis]
    n_imfs = imfs.shape[1]

    energies  = np.array([np.sum(imfs[:, i] ** 2) for i in range(n_imfs)])
    total_e   = energies.sum() + 1e-12
    e_ratios  = energies / total_e

    chunk_e   = np.array([np.sum(window[i:i+FS]**2)
                          for i in range(0, len(window)-FS, FS//2)])
    burst_ratio = float(np.max(chunk_e) / (np.mean(chunk_e) + 1e-12))

    feat = {
        "t_unix":      t_start_s,
        "dc_level":    float(np.mean(window)),
        "dc_range":    float(np.ptp(window)),
        "rms":         float(np.sqrt(np.mean(window**2))),
        "dc_gradient": float((window[-1] - window[0]) / WIN_SEC),
        "energy_mean": float(np.mean(energies)),
        "energy_std":  float(np.std(energies)),
        "energy_max":  float(np.max(energies)),
        "burst_ratio": burst_ratio,
        "n_imfs":      n_imfs,
    }

    for i in range(MAX_IMFS):
        p = f"imf{i+1}"
        if i < n_imfs:
            try:
                ia, _ip, if_ = emd.spectra.frequency_transform(imfs[:, i], FS, "hilbert")
            except Exception:
                ia  = np.abs(imfs[:, i])
                if_ = np.full(len(ia), np.nan)
            feat[f"{p}_mean_ia"]      = float(np.nanmean(ia))
            feat[f"{p}_std_ia"]       = float(np.nanstd(ia))
            feat[f"{p}_max_ia"]       = float(np.nanmax(ia))
            feat[f"{p}_mean_if"]      = float(np.nanmean(if_))
            feat[f"{p}_std_if"]       = float(np.nanstd(if_))
            feat[f"{p}_energy"]       = float(energies[i])
            feat[f"{p}_energy_ratio"] = float(e_ratios[i])
        else:
            for sfx in ["mean_ia","std_ia","max_ia","mean_if","std_if","energy","energy_ratio"]:
                feat[f"{p}_{sfx}"] = 0.0

    return feat


# ── Per-date processing ────────────────────────────────────────────────────────

def process_date(date: str, data_dir: Path, cache_dir: Path) -> pd.DataFrame | None:
    checkpoint = cache_dir / f"features_{date}_central.parquet"
    if checkpoint.exists():
        log.info("[%s] Cache hit — loading checkpoint", date)
        return pd.read_parquet(checkpoint)

    files = discover_files_for_date(data_dir, date)
    if not files:
        log.warning("[%s] No BSD files found — skipping", date)
        return None

    log.info("[%s] Building signal from %d files …", date, len(files))
    signal, t_secs = build_day_signal(files)
    log.info("[%s] Signal: %d samples = %.2f hours", date, len(signal), len(signal)/FS/3600)

    if len(signal) < WIN_N:
        log.warning("[%s] Signal too short — skipping", date)
        return None

    starts = list(range(0, len(signal) - WIN_N, STEP_N))
    rows   = []
    for s in tqdm(starts, desc=date, unit="win", leave=False):
        feat = extract_window_features(signal[s:s+WIN_N], float(t_secs[s]))
        if feat is not None:
            rows.append(feat)

    if not rows:
        log.warning("[%s] EMD produced no windows — skipping", date)
        return None

    df = pd.DataFrame(rows)

    # t_hours: hours elapsed since the very first recording (epoch)
    day_offset = date_offset_hours(date)
    df["t_hours"] = day_offset + df["t_unix"] / 3600.0
    df["date"]    = date

    df.to_parquet(checkpoint, index=False)
    log.info("[%s] Saved %d windows → %s", date, len(df), checkpoint.name)
    return df


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Centralized BSD → parquet pipeline")
    parser.add_argument("--data_dir",  default="../../all_bsd",
                        help="Directory of BSD symlinks (default: ../all_bsd)")
    parser.add_argument("--cache_dir", default="../code/central_cache",
                        help="Directory for per-date checkpoints")
    parser.add_argument("--out",       default="../code/central_cache/features_all.parquet",
                        help="Final merged output parquet")
    args = parser.parse_args()

    data_dir  = Path(args.data_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    out_path  = Path(args.out).resolve()

    if not data_dir.exists():
        log.error("data_dir not found: %s", data_dir)
        sys.exit(1)

    cache_dir.mkdir(parents=True, exist_ok=True)

    dates = discover_dates(data_dir)
    log.info("Found %d unique dates: %s … %s", len(dates), dates[0], dates[-1])

    frames = []
    for date in dates:
        df = process_date(date, data_dir, cache_dir)
        if df is not None:
            frames.append(df)

    if not frames:
        log.error("No data extracted — check data_dir and BSD format")
        sys.exit(1)

    merged = pd.concat(frames, ignore_index=True)
    merged.sort_values("t_hours", inplace=True)
    merged.reset_index(drop=True, inplace=True)

    merged.to_parquet(out_path, index=False)
    log.info("=" * 60)
    log.info("features_all.parquet: %d windows × %d features", len(merged), len(merged.columns))
    log.info("Date range : %s → %s", merged["date"].min(), merged["date"].max())
    log.info("Time range : %.1f h → %.1f h", merged["t_hours"].min(), merged["t_hours"].max())
    log.info("Output     : %s", out_path)
    log.info("=" * 60)


if __name__ == "__main__":
    main()
