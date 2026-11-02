"""Issue 6: LP cut-off test against the raw mains-regressed 2416 Hz signal, per amplitude class.
Real events: 8-sigma detections whose [a-6 s, e+6 s] lies in one dropout-free segment (v2_snippets.pkl).
For fc in FCS: y_fc = LP_fc(raw - baseline). Metrics on W = [a-1 s, e+1 s]:
  peak_raw  = max|y_fc| / max|raw|            (raw peak is noise-inflated for small events)
  peak_40   = max|y_fc| / max|LP40(raw)|      (reference just below the 50 Hz line)
  corr_n    = corr(y_fc, raw) / ceiling, ceiling = sqrt(max(var_W - var_noise, 0) / var_W)
  shape descriptors (rise, FWHM, lobes, d90) at each fc
Excess-energy spectrum: periodogram of raw over W minus the noise PSD of the flanks (same window),
integrated -> fraction of the event's excess variance above f (per event; class medians).
Output: data/out_cutoff2.txt, data/cutoff2_events.parquet, figures/fig10_cutoff_real.png"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import pickle, numpy as np, pandas as pd
from scipy import signal
import preproc as P, det2 as D
from style import plt, RUNC, INK2

FS = P.FS_ADC
FCS = [2, 3, 5, 8, 10, 15]
DEC = 5                                   # descriptors computed at 2416/5 = 483.2 Hz
FSD = FS / DEC
SOS = {fc: signal.butter(4, fc, fs=FS, output="sos") for fc in FCS + [40]}
EDGES = [0, 1500, 5000, 1e9]
LABELS = ["small <1.5k", "medium 1.5-5k", "large >5k"]
FGRID = np.array([1, 2, 3, 5, 8, 10, 15, 20, 30])


def excess_fraction(R, left, right):
    """Fraction of excess (event minus noise) variance above each f in FGRID, and 99 % excess bandwidth."""
    n = len(R)
    w = np.hanning(n)
    Pw = np.abs(np.fft.rfft((R - R.mean()) * w)) ** 2 / ((w ** 2).sum() * FS)
    f = np.fft.rfftfreq(n, 1 / FS)
    fl = np.r_[left, right]
    nper = min(n, len(left), len(right))
    fn, Pn = signal.welch(fl, fs=FS, nperseg=nper, window="hann")
    Pn = np.interp(f, fn, Pn)
    ex = (Pw - Pn)
    m = f < 45
    ex = ex[m]; f = f[m]
    tot = ex.sum()
    if tot <= 0:
        return np.full(len(FGRID), np.nan), np.nan
    cum = np.cumsum(ex) / tot
    above = np.array([1 - np.interp(fg, f, cum) for fg in FGRID])
    bw99 = f[np.argmax(cum >= 0.99)] if (cum >= 0.99).any() else np.nan
    return above, bw99


def analyse(sn):
    x = sn["x"].astype(float)
    n = len(x)
    # baseline: 50 Hz pipeline baseline interpolated to 2416 Hz
    tb = (sn["off50"] + np.arange(len(sn["base50"]))) / 50.0
    tx = sn["sa_s"] + np.arange(n) / FS
    base = np.interp(tx, tb, sn["base50"])
    R = x - base
    i0 = int(round((sn["a"] / 50.0 - 1.0 - sn["sa_s"]) * FS))
    i1 = int(round((sn["e"] / 50.0 + 1.0 - sn["sa_s"]) * FS))
    i0, i1 = max(i0, 0), min(i1, n)
    W = slice(i0, i1)
    left = R[: max(i0 - int(FS), 100)]
    right = R[min(i1 + int(FS), n - 100):]
    var_n = np.var(np.r_[left, right])
    var_W = np.var(R[W])
    ceil = np.sqrt(max(var_W - var_n, 0) / var_W)
    y40 = signal.sosfiltfilt(SOS[40], R, padtype="even")
    out = dict(span=sn["span"], a=sn["a"], e=sn["e"], sig50=sn["sig"], ceil=ceil, noise_raw=np.sqrt(var_n),
               amax40=np.abs(y40[W]).max(), amax_raw=np.abs(R[W]).max())
    ab, bw99 = excess_fraction(R[W], left, right)
    for fg, v in zip(FGRID, ab):
        out[f"above_{fg}"] = v
    out["bw99_excess"] = bw99
    for fc in FCS:
        y = signal.sosfiltfilt(SOS[fc], R, padtype="even")
        yw = y[W]
        out[f"peak_raw_{fc}"] = np.abs(yw).max() / out["amax_raw"]
        out[f"peak_40_{fc}"] = np.abs(yw).max() / out["amax40"]
        out[f"corr_{fc}"] = np.corrcoef(yw, R[W])[0, 1]
        out[f"corrn_{fc}"] = out[f"corr_{fc}"] / ceil if ceil > 0 else np.nan
        sh = D.shape(yw[::DEC], fs=FSD)
        for k in ("rise", "fwhm", "nlobes", "d90"):
            out[f"{k}_{fc}"] = sh[k]
    return out


if __name__ == "__main__":
    OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
    FIG = os.path.join(OUT, "..", "figures")
    sn = pickle.load(open(os.path.join(OUT, "v2_snippets.pkl"), "rb"))
    ev = pd.read_parquet(os.path.join(OUT, "v2_events.parquet"))
    ev = ev[ev.thr == 8.0].set_index(["span", "a"])
    from multiprocessing import Pool
    with Pool(5) as p:
        res = p.map(analyse, sn, chunksize=8)
    df = pd.DataFrame(res)
    df = df.join(ev[["amax", "dur", "run", "kind", "pattern"]], on=["span", "a"])
    df["cls"] = pd.cut(df.amax, EDGES, labels=LABELS)
    df["short"] = df.dur < 0.4
    df.to_parquet(os.path.join(OUT, "cutoff2_events.parquet"))
    lines = []
    pr = lambda s: lines.append(s) or print(s)
    v = df[~df.short]
    pr(f"events with raw snippets: {len(df)} (vehicle-like >=0.4 s: {len(v)}); by class: {v.cls.value_counts().to_dict()}")
    pr(f"by kind: {v.kind.value_counts().to_dict()}  by run: {v.run.value_counts().to_dict()}")
    pr("\n== per class, per cut-off: medians [p5 or p95] ==")
    for c in LABELS:
        s = v[v.cls == c]
        pr(f"\n{c}  n={len(s)}  raw-noise sigma (full band, mains-free) median {s.noise_raw.median():.0f}, corr ceiling median {s.ceil.median():.3f}")
        pr("  fc | peak/raw  | peak/LP40 (p5) | corr/raw | corr/ceiling (p5) | rise s (med) | FWHM s | nlobes | d90 s")
        for fc in FCS:
            pr(f"  {fc:>2} | {s[f'peak_raw_{fc}'].median():.3f}     | {s[f'peak_40_{fc}'].median():.3f} ({np.percentile(s[f'peak_40_{fc}'], 5):.3f})  | "
               f"{s[f'corr_{fc}'].median():.3f}    | {s[f'corrn_{fc}'].median():.3f} ({np.nanpercentile(s[f'corrn_{fc}'], 5):.3f})     | "
               f"{s[f'rise_{fc}'].median():.3f}        | {s[f'fwhm_{fc}'].median():.3f}  | {s[f'nlobes_{fc}'].mean():.2f}   | {s[f'd90_{fc}'].median():.2f}")
        # descriptor change vs 15 Hz
        for k in ("rise", "fwhm", "d90"):
            rel = [(np.median(np.abs(s[f'{k}_{fc}'] - s[f'{k}_15']) / s[f'{k}_15'].clip(lower=1e-3))) for fc in FCS]
            pr(f"  median |Δ{k}|/{k}(15 Hz): " + ", ".join(f"{fc}Hz {r:.2f}" for fc, r in zip(FCS, rel)))
        agree = [(s[f"nlobes_{fc}"] == s["nlobes_15"]).mean() for fc in FCS]
        pr("  lobe-count equal to 15 Hz: " + ", ".join(f"{fc}Hz {a_:.2f}" for fc, a_ in zip(FCS, agree)))
        pr("  excess-variance fraction above f (median over events, [IQR]): " +
           ", ".join(f">{fg}Hz {s[f'above_{fg}'].median():.3f} [{s[f'above_{fg}'].quantile(.25):.3f},{s[f'above_{fg}'].quantile(.75):.3f}]" for fg in FGRID))
        pr(f"  99 % excess-variance bandwidth: median {s.bw99_excess.median():.2f} Hz, p75 {s.bw99_excess.quantile(.75):.2f}, p95 {s.bw99_excess.quantile(.95):.2f}")
    s = df[df.short]
    if len(s):
        pr(f"\nshort (<0.4 s) detections with snippets n={len(s)}: excess above 5 Hz median {s.above_5.median():.3f}, bw99 median {s.bw99_excess.median():.1f} Hz")
    open(os.path.join(OUT, "out_cutoff2.txt"), "w").write("\n".join(lines))
    # figure
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for k, c in enumerate(LABELS):
        s = v[v.cls == c]
        ax[0].plot(FCS, [s[f"peak_40_{fc}"].median() for fc in FCS], "-o", color=RUNC[k], ms=4, label=f"{c} (n={len(s)})")
        ax[1].plot(FCS, [s[f"corrn_{fc}"].median() for fc in FCS], "-o", color=RUNC[k], ms=4, label=c)
        ax[2].plot(FGRID, [s[f"above_{fg}"].median() for fg in FGRID], "-o", color=RUNC[k], ms=4, label=c)
    ax[0].set_xlabel("LP cut-off (Hz)"); ax[0].set_ylabel("median peak / peak(LP 40 Hz)"); ax[0].legend(fontsize=7)
    ax[0].set_title("Peak retention (real events)")
    ax[1].set_xlabel("LP cut-off (Hz)"); ax[1].set_ylabel("median corr(y_fc, raw) / noise ceiling"); ax[1].set_title("Waveform fidelity vs raw 2416 Hz")
    ax[2].set_xscale("log"); ax[2].set_xlabel("f (Hz)"); ax[2].set_ylabel("fraction of excess variance above f"); ax[2].axhline(0, color=INK2, lw=.6)
    ax[2].set_title("Event energy above f (event − flank noise)")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig10_cutoff_real.png"), dpi=110)
