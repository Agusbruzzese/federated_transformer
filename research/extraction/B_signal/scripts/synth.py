"""Injection tests with known truth.
(1) Feature-path LP cut-off: real large-event templates (LP 40 Hz, high SNR), time-compressed by
    1x/2x/4x (faster events) and scaled to 700/1500/3000/20000 counts, injected into real quiet
    mains-free raw backgrounds (2416 Hz). For each fc: peak error, correlation with the truth,
    rise/FWHM relative error, lobe-count correctness.
(2) Baseline distortion (issue 9): templates stretched to 2-16 s support, injected at 20000 and 2000
    counts, run through LP 5 Hz -> 50 Hz -> running median (10/20/30/60 s, 1-pass and 2-pass with
    events bridged) -> compare residual with the truth passed through the same LP/resampling.
Outputs: data/out_synth.txt, data/synth_lp.parquet, data/synth_base.parquet, figures/fig11_synth.png"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
import pickle, numpy as np, pandas as pd
from scipy import signal
from multiprocessing import Pool
import preproc as P, det2 as D
from style import plt, RUNC, INK2

FS = P.FS_ADC
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
FIG = os.path.join(OUT, "..", "figures")
FCS = [2, 3, 5, 8, 10, 15, 40]
SOS = {fc: signal.butter(4, fc, fs=FS, output="sos") for fc in FCS}
DEC = 5
rng = np.random.default_rng(42)


def templates(n_max=40):
    sn = pickle.load(open(os.path.join(OUT, "v2_snippets.pkl"), "rb"))
    cut = pd.read_parquet(os.path.join(OUT, "cutoff2_events.parquet")).set_index(["span", "a"])
    T = []
    for s in sn:
        k = (s["span"], s["a"])
        if k not in cut.index:
            continue
        row = cut.loc[k]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        if row.amax40 < 10000 or row.dur < 0.4:
            continue
        x = s["x"].astype(float)
        tb = (s["off50"] + np.arange(len(s["base50"]))) / 50.0
        tx = s["sa_s"] + np.arange(len(x)) / FS
        R = x - np.interp(tx, tb, s["base50"])
        y = signal.sosfiltfilt(SOS[40], R, padtype="even")
        i0 = int((s["a"] / 50 - 1.5 - s["sa_s"]) * FS); i1 = int((s["e"] / 50 + 1.5 - s["sa_s"]) * FS)
        y = y[max(i0, 0):i1]
        y = y - np.linspace(y[:200].mean(), y[-200:].mean(), len(y))
        y *= signal.windows.tukey(len(y), 0.3)
        T.append(dict(y=y, pattern=row.pattern))
    idx = rng.permutation(len(T))[:n_max]
    return [T[i] for i in idx]


def resample_len(y, n):
    return np.interp(np.linspace(0, len(y) - 1, n), np.arange(len(y)), y)


def lp_job(args):
    ti, T, speed, amp, bg = args
    y = resample_len(T, max(int(len(T) / speed), 50))
    y = y / np.abs(y).max() * amp
    truth = D.shape(y[::DEC], fs=FS / DEC)
    x = bg.copy()
    c = len(x) // 2 - len(y) // 2
    x[c:c + len(y)] += y
    x -= np.median(np.r_[x[:c - 2000], x[c + len(y) + 2000:]])
    W = slice(c, c + len(y))
    rows = []
    for fc in FCS:
        z = signal.sosfiltfilt(SOS[fc], x, padtype="even")[W]
        sh = D.shape(z[::DEC], fs=FS / DEC)
        rows.append(dict(t=ti, speed=speed, amp=amp, fc=fc, peak=np.abs(z).max() / amp,
                         rho=np.corrcoef(z, y)[0, 1], rise_err=(sh["rise"] - truth["rise"]) / max(truth["rise"], 1e-3),
                         fwhm_err=(sh["fwhm"] - truth["fwhm"]) / truth["fwhm"], lobes_ok=sh["nlobes"] == truth["nlobes"],
                         true_rise=truth["rise"], true_fwhm=truth["fwhm"], true_d90=truth["d90"]))
    return rows


def base_job(args):
    ti, T, support_s, amp, bg = args
    y = resample_len(T, int(support_s * FS))
    y = y / np.abs(y).max() * amp
    x = bg.copy()
    c = len(x) // 2 - len(y) // 2
    x[c:c + len(y)] += y
    xs = P.to50(P.lowpass(x - np.median(x), P.DETECT_LP_HZ))
    tr = np.zeros(len(bg)); tr[c:c + len(y)] = y
    truth = P.to50(P.lowpass(tr, P.DETECT_LP_HZ))
    m = np.zeros(len(xs), bool); m[:2] = m[-2:] = True
    j0, j1 = int(c / FS * 50), int((c + len(y)) / FS * 50)
    rows = []
    for win in (10, 20, 30, 60):
        for npass in (1, 2):
            b = P.baseline(xs, m, win_s=win)
            r = xs - b
            if npass == 2:
                sig = P.robust_sigma(r, m)
                em = np.zeros(len(r), bool)
                for a, e in D.detect(r, sig, m, k_hi=5.0):
                    em[max(0, a - 100):e + 100] = True
                b = P.baseline(xs, m, win_s=win, event_mask=em)
                r = xs - b
            rw, tw = r[j0:j1], truth[j0:j1]
            rows.append(dict(t=ti, support=support_s, amp=amp, win=win, npass=npass,
                             peak=np.abs(rw).max() / np.abs(tw).max(),
                             area=np.abs(rw).sum() / np.abs(tw).sum(),
                             signed_area_err=(rw.sum() - tw.sum()) / np.abs(tw).sum(),
                             rho=np.corrcoef(rw, tw)[0, 1]))
    return rows


if __name__ == "__main__":
    T = templates()
    Q = [q["x"].astype(float) for q in pickle.load(open(os.path.join(OUT, "v2_quiet.pkl"), "rb"))]
    print(len(T), "templates", len(Q), "quiet backgrounds")
    jobs = [(i, t["y"], sp, amp, Q[rng.integers(len(Q))]) for i, t in enumerate(T) for sp in (1, 2, 4) for amp in (700, 1500, 3000, 20000)]
    with Pool(5) as p:
        L = pd.DataFrame([r for rr in p.map(lp_job, jobs, chunksize=4) for r in rr])
    L.to_parquet(os.path.join(OUT, "synth_lp.parquet"))
    lines = []
    pr = lambda s: lines.append(s) or print(s)
    pr(f"templates: {len(T)} real large events (LP 40 Hz, |peak|>10k); truth rise median by speed: " +
       str(L[L.fc == 40].groupby("speed").true_rise.median().round(3).to_dict()) + " s; FWHM: " +
       str(L[L.fc == 40].groupby("speed").true_fwhm.median().round(3).to_dict()) + " s")
    pr("\n== (1) LP cut-off vs truth: median peak ratio | median corr | median |rise err| | median |FWHM err| | lobe-count correct ==")
    for (amp, sp), s in L.groupby(["amp", "speed"]):
        pr(f"amp {amp:>5}  speed x{sp}: " + " ; ".join(
            f"{fc}Hz {g.peak.median():.2f}|{g.rho.median():.3f}|{g.rise_err.abs().median():.2f}|{g.fwhm_err.abs().median():.2f}|{g.lobes_ok.mean():.2f}"
            for fc, g in s.groupby("fc")))
    # best fc per (amp, speed) by corr
    best = L.groupby(["amp", "speed", "fc"]).rho.median().unstack("fc")
    pr("\nfc maximising median corr with truth:\n" + best.idxmax(axis=1).unstack("speed").to_string())
    pr("\nmedian corr table (rows amp,speed; cols fc):\n" + best.round(3).to_string())
    jobs = [(i, t["y"], sup, amp, Q[rng.integers(len(Q))]) for i, t in enumerate(T[:25]) for sup in (2, 4, 8, 12, 16) for amp in (2000, 20000)]
    with Pool(5) as p:
        B = pd.DataFrame([r for rr in p.map(base_job, jobs, chunksize=4) for r in rr])
    B.to_parquet(os.path.join(OUT, "synth_base.parquet"))
    pr("\n== (2) baseline distortion: median peak ratio | area ratio | corr  (amp 20000) ==")
    g = B[B.amp == 20000].groupby(["npass", "win", "support"])
    tab = g.agg(peak=("peak", "median"), area=("area", "median"), rho=("rho", "median"), peak_p5=("peak", lambda x: np.percentile(x, 5)))
    pr(tab.round(3).to_string())
    pr("\n(amp 2000)")
    pr(B[B.amp == 2000].groupby(["npass", "win", "support"]).agg(peak=("peak", "median"), area=("area", "median"), rho=("rho", "median")).round(3).to_string())
    open(os.path.join(OUT, "out_synth.txt"), "w").write("\n".join(lines))
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    for k, amp in enumerate([700, 1500, 3000, 20000]):
        s = L[(L.amp == amp) & (L.speed == 1)].groupby("fc").rho.median()
        ax[0].plot(s.index[:-1], s.values[:-1], "-o", ms=4, color=RUNC[k], label=f"{amp} counts")
        s = L[(L.amp == amp) & (L.speed == 4)].groupby("fc").rho.median()
        ax[1].plot(s.index[:-1], s.values[:-1], "-o", ms=4, color=RUNC[k], label=f"{amp} counts")
    for a_, ttl in zip(ax[:2], ["real-speed templates", "4x faster templates"]):
        a_.set_xlabel("LP cut-off (Hz)"); a_.set_ylabel("median corr with truth"); a_.set_title(f"Injected events: {ttl}"); a_.legend(fontsize=7)
    for k, win in enumerate([10, 20, 30, 60]):
        s = B[(B.amp == 20000) & (B.npass == 1) & (B.win == win)].groupby("support").peak.median()
        ax[2].plot(s.index, s.values, "-o", ms=4, color=RUNC[k], label=f"{win} s median, 1-pass")
    s = B[(B.amp == 20000) & (B.npass == 2) & (B.win == 20)].groupby("support").peak.median()
    ax[2].plot(s.index, s.values, "--o", ms=4, color="#0b0b0b", label="20 s median, 2-pass (events bridged)")
    ax[2].set_xlabel("injected event support (s)"); ax[2].set_ylabel("median peak retained"); ax[2].legend(fontsize=7)
    ax[2].set_title("Running-median baseline distortion")
    fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig11_synth.png"), dpi=110)
