# VENDORED COPY -- provenance: research/extraction/D_features/features.py
# sha256 of the original at vendoring time (2026-09-30): 68e7db3b6fa100442872255abc42bdf79b9fe40859aec6487561752a304d8562
# Do not edit except where marked 'E-PATCH'. Workstream E (extraction pipeline).
"""
Reusable feature functions for single-channel TMR vehicle events (workstream D).

Conventions
-----------
* ``raw``  : event window at the native rate (FS = 2400 Hz nominal), counts, with the
             local (60-s running-median) baseline subtracted.  Used only by the legacy
             EMD/HHT set, which operated on un-filtered data.
* ``x``    : event window low-passed at 20 Hz (4th-order Butterworth, zero-phase) and
             decimated x24 -> FSD = 100 Hz, baseline subtracted.  8 s = 800 samples,
             centred on the envelope peak.  ALL recommended representations use ``x``.
             Rationale: vehicle energy is < ~4 Hz; everything above is mains (50 Hz +
             odd harmonics) and sensor noise, identical with or without a vehicle.

Representations
---------------
legacy66 + legacy_add_relative       faithful re-implementation of the old 70-feature set (from git
                                      branch origin/comparison-update: comparison/central_pipeline.py)
legacy_fixed_io(x)                    same 65 window features on correctly-read, anti-aliased data
hht_lp(x)                             same HHT statistics but on the 100 Hz low-passed signal
wavelet_feats(x)                      DWT (db4, L=6) band stats + CWT (Morlet) log band power
stft_feats(x)                         log-spaced ("mel-like") STFT band log-powers + spectral stats
morph_feats(x)                        time-domain morphology (amplitude, polarity, durations, lobes...)
waveform_feats(x)                     fixed-length (256) amplitude-normalised waveform (for a learned encoder)
"""
from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.stats import skew, kurtosis

FS = 2400.0
DEC = 24
FSD = FS / DEC
_SOS_LP20 = signal.butter(4, 20, fs=FS, output="sos")


def to_lp(raw: np.ndarray) -> np.ndarray:
    """Raw 2400 Hz window -> 20 Hz low-pass, decimated to 100 Hz."""
    return signal.sosfiltfilt(_SOS_LP20, raw - np.median(raw))[::DEC]


# ----------------------------------------------------------------------------------
# (a) legacy 70-feature EMD/HHT set
# ----------------------------------------------------------------------------------
LEGACY_NAMES = ["dc_level", "dc_range", "rms", "dc_gradient", "energy_mean", "energy_std",
                "energy_max", "burst_ratio", "n_imfs"]
for _i in range(1, 9):
    LEGACY_NAMES += [f"imf{_i}_{s}" for s in ("mean_ia", "std_ia", "max_ia", "mean_if", "std_if",
                                             "energy", "energy_ratio")]
LEGACY_NAMES += ["dc_relative", "dc_range_rel", "rms_rel", "dc_baseline", "t_hours"]
LEGACY_CONFOUNDS = ["t_hours", "dc_baseline", "dc_level"]   # absolute time / absolute (wrapped) level


def _imf_stats(imf: np.ndarray, fs: float):
    a = signal.hilbert(imf)
    ia = np.abs(a)
    ph = np.unwrap(np.angle(a))
    inst_f = np.diff(ph) * fs / (2 * np.pi)
    return [ia.mean(), ia.std(), ia.max(), inst_f.mean(), inst_f.std(), float(np.sum(imf ** 2))]


def emd_imfs(sig: np.ndarray, method: str = "emd", max_imf: int = -1, trials: int = 50, seed: int = 0):
    """Return IMFs (without residue) using PyEMD. method in {emd, eemd, ceemdan}."""
    from PyEMD import EMD, EEMD, CEEMDAN
    if method == "emd":
        e = EMD()
        e.emd(sig, max_imf=max_imf)
        imfs, _ = e.get_imfs_and_residue()
    elif method == "eemd":
        e = EEMD(trials=trials, parallel=False)
        e.noise_seed(seed)
        imfs = e.eemd(sig, max_imf=max_imf)[:-1]
    else:
        e = CEEMDAN(trials=trials, parallel=False)
        e.noise_seed(seed)
        imfs = e.ceemdan(sig, max_imf=max_imf)[:-1]
    return imfs


def legacy_read_int16(codes: np.ndarray) -> np.ndarray:
    """The legacy reader: np.frombuffer(bytes, '<i2')[0::2] == low 16 bits of each LE uint32 word,
    read as SIGNED int16.  The true signal is a 24-bit code (~11.2M-12.0M), so this keeps
    code mod 65536 shifted to [-32768, 32767]: every time the code crosses a multiple of 65536
    (+/-32768) the legacy signal jumps by 65536 counts."""
    c = np.round(np.asarray(codes)).astype(np.int64) & 0xFFFF
    return c.astype(np.uint16).view(np.int16).astype(np.float64)


def legacy_window_feats(w: np.ndarray, fs: float = FSD, max_imfs: int = 8):
    """Features computed by comparison/central_pipeline.py::extract_window_features (branch
    origin/comparison-update, commit 6d63a85) on a 100 Hz window w.  65 values:
    9 window stats + 8 IMFs x 7 HHT stats (zero-padded).  The original used the `emd` package
    (emd.sift.sift(max_imfs=8), emd.spectra.frequency_transform); that package is not installed, so
    PyEMD EMD with max_imf=max_imfs-1 (+ residue as last component, like emd.sift) and a
    Hilbert phase-derivative IF are used instead (documented deviation)."""
    win_sec = len(w) / fs
    from PyEMD import EMD
    e = EMD()
    comp = e.emd(w, max_imf=max_imfs - 1)          # rows: IMFs (+ residue)
    comp = comp[:max_imfs]
    n_imfs = comp.shape[0]
    energies = np.array([np.sum(c ** 2) for c in comp])
    ratios = energies / (energies.sum() + 1e-12)
    fs_i = int(fs)
    chunk_e = np.array([np.sum(w[i:i + fs_i] ** 2) for i in range(0, len(w) - fs_i, fs_i // 2)])
    f = [np.mean(w), np.ptp(w), np.sqrt(np.mean(w ** 2)), (w[-1] - w[0]) / win_sec,
         energies.mean(), energies.std(), energies.max(), chunk_e.max() / (chunk_e.mean() + 1e-12), n_imfs]
    for k in range(max_imfs):
        if k < n_imfs:
            a = signal.hilbert(comp[k]); ia = np.abs(a)
            inst_f = np.diff(np.unwrap(np.angle(a))) * fs / (2 * np.pi)
            f += [ia.mean(), ia.std(), ia.max(), inst_f.mean(), inst_f.std(), energies[k], ratios[k]]
        else:
            f += [0.0] * 7
    return np.asarray(f, float)


def legacy66(codes_2400: np.ndarray, t_hours_cum: float):
    """Faithful legacy front end: int16 low-word read -> naive [::24] decimation (no anti-alias
    filter) -> window features, + t_hours = hours since 2025-02-04 00:00 (cumulative, as in
    central_pipeline.py).  The 4 'relative' columns (dc_relative, dc_range_rel, rms_rel,
    dc_baseline) came from the lost client/clustering.py::add_dc_relative and are added by
    legacy_add_relative()."""
    w = legacy_read_int16(codes_2400)[::DEC]
    return np.r_[legacy_window_feats(w), t_hours_cum]


def legacy_add_relative(L66: np.ndarray, dc_base_codes: np.ndarray, group: np.ndarray, is_bg: np.ndarray):
    """Reconstruction (source lost) of add_dc_relative: per-date baselines.
    dc_baseline  = legacy int16 value of the local 60-s running-median baseline code
    dc_relative  = dc_level - dc_baseline
    dc_range_rel = dc_range - median(dc_range of background windows, same date)
    rms_rel      = rms      - median(rms of background windows, same date)
    Returns the 70 columns in client.py FEATURE_COLS order."""
    L66 = np.asarray(L66, float)
    dcb = legacy_read_int16(dc_base_codes)
    rng_b = np.zeros(len(L66)); rms_b = np.zeros(len(L66))
    for g in np.unique(group):
        m = group == g; mb = m & is_bg
        ref = mb if mb.sum() >= 3 else m
        rng_b[m] = np.median(L66[ref, 1]); rms_b[m] = np.median(L66[ref, 2])
    lead = L66[..., :65]
    return np.column_stack([lead, lead[..., 0] - dcb, lead[..., 1] - rng_b, lead[..., 2] - rms_b, dcb, L66[..., 65]])


def legacy_fixed_io(x: np.ndarray):
    """Same 65 legacy window features, but on the CORRECT signal (24-bit codes, 20 Hz anti-alias
    low-pass, 100 Hz, baseline removed) and without the time/absolute-level columns.
    Isolates 'bad I/O' from 'EMD/HHT feature design'."""
    return legacy_window_feats(np.asarray(x, float))


LEGACY_FIXED_NAMES = LEGACY_NAMES[:65]


def hht_stats_from_imfs(imfs, fs=FSD, n_keep: int = 5):
    tot = float(np.sum(imfs ** 2)) + 1e-12
    f = [len(imfs)]
    for k in range(n_keep):
        if k < len(imfs):
            st = _imf_stats(imfs[k], fs)
            f += [np.log1p(st[0]), np.log1p(st[1]), np.log1p(st[2]), st[3], st[4], np.log1p(st[5]), st[5] / tot]
        else:
            f += [0.0] * 7
    return np.asarray(f, float)


def hht_lp(x: np.ndarray, n_keep: int = 5, method: str = "emd"):
    """HHT stats on the 100 Hz low-passed window (vehicle band), log-compressed energies."""
    return hht_stats_from_imfs(emd_imfs(x, method=method), FSD, n_keep)


HHT_LP_NAMES = ["n_imfs"] + [f"imf{k+1}_{s}" for k in range(5) for s in
                             ("log_mean_ia", "log_std_ia", "log_max_ia", "mean_if", "std_if", "log_energy", "energy_ratio")]


# ----------------------------------------------------------------------------------
# (b) wavelet
# ----------------------------------------------------------------------------------
CWT_FREQS = np.geomspace(0.15, 10.0, 16)
WAVELET_NAMES = ([f"dwt_{b}_{s}" for b in ["d1", "d2", "d3", "d4", "d5", "d6", "a6"]
                  for s in ("logE", "relE", "logmax")]
                 + [f"cwt_logP_{f:.2f}Hz" for f in CWT_FREQS])


def wavelet_feats(x: np.ndarray):
    import pywt
    coeffs = pywt.wavedec(x, "db4", level=6)          # [a6, d6, d5, ..., d1]
    bands = coeffs[::-1][:-1] + [coeffs[0]]           # d1..d6, a6
    E = np.array([np.sum(c ** 2) for c in bands])
    tot = E.sum() + 1e-12
    f = []
    for c, e in zip(bands, E):
        f += [np.log10(e + 1.0), e / tot, np.log10(np.abs(c).max() + 1.0)]
    scales = pywt.central_frequency("morl") * FSD / CWT_FREQS
    cw, _ = pywt.cwt(x, scales, "morl", sampling_period=1 / FSD)
    f += list(np.log10(np.mean(cw ** 2, axis=1) + 1.0))
    return np.asarray(f, float)


# ----------------------------------------------------------------------------------
# (c) STFT / log-spaced ("mel-like") spectral features
# ----------------------------------------------------------------------------------
STFT_EDGES = np.r_[0.0, np.geomspace(0.5, 20.0, 12)]   # every band holds >=1 bin at 0.25 Hz resolution
STFT_NAMES = ([f"stft_logP_b{i}" for i in range(12)] + [f"stft_sdlogP_b{i}" for i in range(12)]
              + ["spec_centroid", "spec_bw", "spec_rolloff85", "spec_flatness", "spec_centroid_sd"])


def stft_feats(x: np.ndarray):
    f, t, Z = signal.stft(x, fs=FSD, nperseg=400, noverlap=400 - 25, boundary=None, padded=False)
    P = np.abs(Z) ** 2 + 1e-3
    bp = np.stack([P[(f >= lo) & (f < hi)].mean(0) for lo, hi in zip(STFT_EDGES[:-1], STFT_EDGES[1:])])
    lbp = np.log10(bp)
    pm = P.mean(1)
    cen = np.sum(f * pm) / pm.sum()
    bw = np.sqrt(np.sum((f - cen) ** 2 * pm) / pm.sum())
    roll = f[np.searchsorted(np.cumsum(pm) / pm.sum(), 0.85)]
    flat = np.exp(np.mean(np.log(pm))) / pm.mean()
    cen_t = np.sum(f[:, None] * P, 0) / P.sum(0)
    return np.r_[lbp.mean(1), lbp.std(1), cen, bw, roll, flat, cen_t.std()]


# ----------------------------------------------------------------------------------
# (d) time-domain morphology
# ----------------------------------------------------------------------------------
MORPH_NAMES = ["log_amp", "polarity", "pos_frac", "neg_frac", "dur10", "dur50_main", "rise_main",
               "fall_main", "asym", "n_lobes", "first_lobe_sign", "t_pp", "area_pos_n", "area_neg_n",
               "log_area_abs", "log_energy", "t_spread", "zc_rate", "kurt_env"]


def _cross(y, i0, thr, step):
    i = i0
    while 0 <= i + step < len(y) and y[i + step] >= thr:
        i += step
    return i


_SOS_LP5 = signal.butter(4, 5.0, fs=FSD, output="sos")


def morph_feats(x: np.ndarray, fs: float = FSD, edge_s: float = 1.5):
    """Morphology on a 5 Hz-smoothed copy. Thresholds are max(frac*A, k*noise) so that low-SNR
    events do not count noise spikes as lobes; noise sigma = MAD of the outer edge_s seconds."""
    x = signal.sosfiltfilt(_SOS_LP5, x)
    ne = int(edge_s * fs)
    edge = np.r_[x[:ne], x[-ne:]]
    sig = 1.4826 * np.median(np.abs(np.diff(edge))) / np.sqrt(2) + 1e-9
    ax = np.abs(x)
    ip = int(np.argmax(ax))
    A = ax[ip] + 1e-9
    pol = np.sign(x[ip])
    thr10 = max(0.1 * A, 3 * sig)
    above = np.where(ax > thr10)[0]
    i0, i1 = above[0], above[-1]
    dur10 = (i1 - i0 + 1) / fs
    y = x * pol                                   # main lobe (same sign as peak)
    l50, r50 = _cross(y, ip, 0.5 * A, -1), _cross(y, ip, 0.5 * A, +1)
    l10, r10 = _cross(y, ip, thr10, -1), _cross(y, ip, thr10, +1)
    l90, r90 = _cross(y, ip, 0.9 * A, -1), _cross(y, ip, 0.9 * A, +1)
    rise = (l90 - l10) / fs
    fall = (r10 - r90) / fs
    asym = (ip - i0) / max(i1 - i0, 1)
    thr20 = max(0.2 * A, 5 * sig)
    s = np.where(x > thr20, 1, np.where(x < -thr20, -1, 0))
    # lobes = runs of constant non-zero sign lasting >= 0.1 s
    lab = []
    run_s, run_n = 0, 0
    for v in np.r_[s, 0]:
        if v == run_s:
            run_n += 1
            continue
        if run_s != 0 and run_n >= int(0.1 * fs) and (not lab or lab[-1] != run_s):
            lab.append(run_s)
        run_s, run_n = v, 1
    lobes = len(lab)
    first = lab[0] if lab else 0
    seg = x[i0:i1 + 1]
    t_pp = (np.argmax(x) - np.argmin(x)) / fs
    ap = seg[seg > 0].sum() / fs
    an = -seg[seg < 0].sum() / fs
    norm = A * dur10 + 1e-9
    e = x ** 2
    tc = np.sum(np.arange(len(x)) * e) / e.sum()
    tsp = np.sqrt(np.sum((np.arange(len(x)) - tc) ** 2 * e) / e.sum()) / fs
    zc = np.sum(np.diff(np.sign(seg - 0.0)) != 0) / dur10
    return np.array([np.log10(A), pol, x.max() / A, -x.min() / A, dur10, (r50 - l50) / fs, rise, fall,
                     asym, lobes, first, t_pp, ap / norm, an / norm, np.log10(ap + an + 1),
                     np.log10(e.sum() + 1), tsp, zc, kurtosis(ax)], float)


# ----------------------------------------------------------------------------------
# (e) raw waveform for a learned encoder
# ----------------------------------------------------------------------------------
def waveform_feats(x: np.ndarray, n_out: int = 256):
    """Resample 800 -> 256 samples (32 Hz; content < 5 Hz) and divide by max|x|.
    The amplitude log10(max|x|) should be fed as a separate scalar (morph 'log_amp')."""
    y = signal.resample_poly(x, n_out, len(x))
    return y / (np.abs(y).max() + 1e-9)


REPS = {
    "wavelet": (wavelet_feats, WAVELET_NAMES),
    "stft": (stft_feats, STFT_NAMES),
    "morph": (morph_feats, MORPH_NAMES),
    "waveform": (waveform_feats, [f"w{i}" for i in range(256)]),
    "hht_lp": (hht_lp, HHT_LP_NAMES),
    "legacy_fixed_io": (legacy_fixed_io, LEGACY_FIXED_NAMES),
}
