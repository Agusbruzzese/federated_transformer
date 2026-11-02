"""Validation of the E pipeline outputs (data/events). Usage:
    python validate.py counts      # (a) counts per run/day/hour/tier + (b) diurnal figure
    python validate.py gallery     # (c) 30 random SNR>=20 windows stratified by run and hour
    python validate.py leakage     # (d) morph-scalar leakage (date, run, 6-h bin, has_dropout)
    python validate.py crosscheck  # (e) 2-h span vs standalone C detect_stream; 5 events' morph vs D features.py
Outputs go next to this script (*.csv, fig_*.png, *_stdout.txt via tee)."""
import os, sys, json, warnings
warnings.filterwarnings("ignore")
os.environ.setdefault("OMP_NUM_THREADS", "4")
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, REPO)
EV = os.path.join(REPO, "data", "events")
TIERS = [("all", 0.0), ("SNR>=5", 5.0), ("SNR>=20", 20.0), ("SNR>=30", 30.0)]
C_REF = {"all": 53.0, "SNR>=20": 16.3, "SNR>=30": 12.1}   # C REPORT §6.8, 64.6 sampled hours
COL = {"SNR>=5": "#2a78d6", "SNR>=20": "#eb6834", "SNR>=30": "#1baf7a"}


def load_meta():
    fs = sorted(f for f in os.listdir(EV) if f.endswith("_meta.parquet"))
    return pd.concat([pd.read_parquet(os.path.join(EV, f)).assign(file=f[:-13]) for f in fs], ignore_index=True)


def coverage():
    """Recorded hours per (date, hour) from the chunk index (each chunk spans (n_samples+n_missing)/fs_local s)."""
    from extraction.blockio import chunk_index
    ix = chunk_index()
    start = ix.wall_start.values.astype("datetime64[ns]").astype(np.int64) / 1e9
    dur = ((ix.n_samples + ix.n_missing) / ix.fs_local).values
    acc = {}
    for s, d in zip(start, dur):
        e = s + d
        while s < e:
            h0 = np.floor(s / 3600) * 3600
            seg = min(e, h0 + 3600) - s
            key = pd.Timestamp(h0, unit="s")
            acc[key] = acc.get(key, 0.0) + seg / 3600
            s += seg
    cv = pd.Series(acc).rename("hours").to_frame()
    cv["date"] = cv.index.strftime("%Y-%m-%d"); cv["hour"] = cv.index.hour
    return cv.reset_index(drop=True)


def counts():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    m = load_meta(); man = json.load(open(os.path.join(EV, "manifest.json")))
    cv = coverage()
    H = cv.hours.sum()
    print(f"events kept {len(m)}, recorded hours {H:.1f} (manifest core hours {man['runtime']['data_hours']:.1f})")
    print(f"excluded mask_frac>0.2: {man['counts']['total']['n_excluded_mask_frac']}; duplicates removed: "
          f"{man['counts']['total']['n_duplicates_removed']}")
    rows = []
    for name, thr in TIERS:
        s = m[m.snr >= thr]
        rows.append(dict(tier=name, n=len(s), per_h=len(s) / H, C_ref_per_h=C_REF.get(name, np.nan),
                         has_dropout=s.has_dropout.mean(), truncated=int(s.truncated.sum()),
                         mask_frac_gt0=(s.mask_frac > 0).mean(), share_07_09=s.hour.between(7, 8).mean(),
                         share_00_05=s.hour.between(0, 4).mean(), share_20_23=s.hour.between(20, 23).mean(),
                         **{f"run{r}": (s.run_id == r).mean() for r in range(4)}))
    T = pd.DataFrame(rows); T.to_csv(os.path.join(HERE, "tab_tiers.csv"), index=False)
    print(T.round(3).to_string(index=False))
    # per run
    hr = m.groupby("run_id").apply(lambda g: pd.Series({n: (g.snr >= t).sum() for n, t in TIERS}))
    cvr = cv.assign(run_id=pd.cut(pd.to_datetime(cv.date), pd.to_datetime(["2025-02-01", "2025-02-12", "2025-02-17", "2025-02-23", "2025-03-01"]),
                                  labels=[0, 1, 2, 3]).astype(int)).groupby("run_id").hours.sum()
    R = hr.assign(hours=cvr)
    for n, _ in TIERS:
        R[n + "/h"] = R[n] / R.hours
    R.to_csv(os.path.join(HERE, "tab_runs.csv")); print("\nper run\n" + R.round(2).to_string())
    # per day
    D = m.groupby("date").apply(lambda g: pd.Series({n: (g.snr >= t).sum() for n, t in TIERS}))
    D = D.join(cv.groupby("date").hours.sum())
    for n, _ in TIERS:
        D[n + "/h"] = D[n] / D.hours
    D.to_csv(os.path.join(HERE, "tab_days.csv")); print("\nper day\n" + D.round(2).to_string())
    # per hour of day (+ weekday/weekend)
    cv["wkend"] = pd.to_datetime(cv.date).dt.dayofweek >= 5
    m["wkend"] = pd.to_datetime(m.date).dt.dayofweek >= 5
    rows = []
    for (h, we), g in cv.groupby(["hour", "wkend"]):
        e = m[(m.hour == h) & (m.wkend == we)]
        for n, t in TIERS:
            k = int((e.snr >= t).sum())
            rows.append(dict(hour=h, weekend=we, tier=n, n=k, hours=g.hours.sum(), rate=k / g.hours.sum()))
    Hd = pd.DataFrame(rows)
    rows = []
    for h, g in cv.groupby("hour"):
        e = m[m.hour == h]
        for n, t in TIERS:
            k = int((e.snr >= t).sum())
            from scipy.stats import chi2
            lo = chi2.ppf(0.025, 2 * k) / 2 if k else 0.0; hi = chi2.ppf(0.975, 2 * k + 2) / 2
            rows.append(dict(hour=h, weekend="all", tier=n, n=k, hours=g.hours.sum(), rate=k / g.hours.sum(),
                             lo=lo / g.hours.sum(), hi=hi / g.hours.sum()))
    Ha = pd.DataFrame(rows)
    pd.concat([Ha, Hd]).to_csv(os.path.join(HERE, "tab_hourly.csv"), index=False)
    print("\nhourly rate (all days)\n" + Ha.pivot(index="hour", columns="tier", values="rate").round(1).to_string())
    for n in ("SNR>=5", "SNR>=20", "SNR>=30"):
        r = Ha[Ha.tier == n].set_index("hour").rate
        night = r.loc[[0, 1, 2, 3, 4]].mean(); peak = r.loc[[7, 8]].mean()
        print(f"{n}: night (00-05) {night:.1f}/h, morning peak (07-09) {peak:.1f}/h, ratio {night / peak:.3f}")
    # figure: diurnal rate per tier, all days (with 95% Poisson CI) + weekday/weekend panels
    fig, axs = plt.subplots(1, 3, figsize=(14, 4.2), sharey=True)
    for ax, (title, sub) in zip(axs, [("all days", Ha), ("weekdays", Hd[~Hd.weekend.astype(bool)]),
                                      ("weekend (Sat 8, Sun 9 Feb)", Hd[Hd.weekend.astype(bool)])]):
        for n in ("SNR>=5", "SNR>=20", "SNR>=30"):
            s = sub[sub.tier == n].sort_values("hour")
            ax.plot(s.hour + 0.5, s.rate.where(s.rate > 0), lw=2, color=COL[n], marker="o", ms=4, label=n)
            if "lo" in s and s.lo.notna().all():
                ax.fill_between(s.hour + 0.5, s.lo, s.hi, color=COL[n], alpha=0.15, lw=0)
            ax.text(24.3, s.rate.iloc[-1], n, color="#52514e", fontsize=8, va="center")
        ax.set_yscale("log"); ax.set_xlim(0, 27); ax.set_ylim(0.05, 300); ax.set_xticks(range(0, 25, 3))
        ax.set_title(title, fontsize=10, loc="left"); ax.set_xlabel("hour of day (PC clock)")
        ax.grid(alpha=0.25, lw=0.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axs[0].set_ylabel("events per recorded hour"); axs[0].legend(frameon=False, fontsize=8, loc="lower center")
    fig.suptitle("Extracted events per recorded hour by hour of day and SNR tier (shaded: 95 % Poisson CI; gaps: zero events)", fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_diurnal.png"), dpi=130)
    # figure: per-day rates
    fig, ax = plt.subplots(figsize=(11, 3.6))
    x = np.arange(len(D))
    for n in ("SNR>=5", "SNR>=20", "SNR>=30"):
        ax.plot(x, D[n + "/h"], marker="o", lw=2, color=COL[n], label=n)
    ax.set_xticks(x); ax.set_xticklabels([d[5:] for d in D.index], rotation=45, fontsize=8)
    ax.set_yscale("log"); ax.set_ylabel("events per recorded hour"); ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=0.25, lw=0.5); ax.set_title("Per-day rates (partial days at run edges)", fontsize=10, loc="left")
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_days.png"), dpi=130)


def arrays(file):
    return (np.load(os.path.join(EV, file + "_windows.npy"), mmap_mode="r"),
            np.load(os.path.join(EV, file + "_evmask.npy"), mmap_mode="r"),
            np.load(os.path.join(EV, file + "_repmask.npy"), mmap_mode="r"))


def gallery(seed=0, n=30):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    m = load_meta(); s = m[m.snr >= 20].copy()
    s["hbin"] = s.hour // 6
    rng = np.random.default_rng(seed)
    pick = []
    strata = [(r, h) for r in range(4) for h in range(4)]
    # round-robin over run x 6-h bin strata until n are drawn
    pools = {k: list(rng.permutation(s[(s.run_id == k[0]) & (s.hbin == k[1])].index)) for k in strata}
    while len(pick) < n and any(pools.values()):
        for k in strata:
            if pools[k] and len(pick) < n:
                pick.append(pools[k].pop())
    P = s.loc[pick].sort_values(["run_id", "wall_time"])
    P[["event_id", "run_id", "wall_time", "snr", "n_neighbours_in_window", "has_dropout", "mask_frac", "truncated",
       "m_log_amp", "m_dur10", "m_n_lobes"]].to_csv(os.path.join(HERE, "gallery_events.csv"), index=False)
    fig, axs = plt.subplots(6, 5, figsize=(18, 15))
    for ax, (_, e) in zip(axs.flat, P.iterrows()):
        W, EM, RM = arrays(e.file)
        w, em, rm = W[e.row], EM[e.row], RM[e.row]
        t = (np.arange(len(w)) - len(w) // 2) / e.fs_eff
        ax.axvspan(t[em.argmax()], t[len(em) - 1 - em[::-1].argmax()], color="#eb6834", alpha=0.12, lw=0)
        ax.plot(t, w, lw=0.9, color="#2a78d6")
        if rm.any():
            ax.plot(t[rm], w[rm], ".", ms=2.5, color="#e34948")
        ax.axvline(0, color="#52514e", lw=0.6, ls=":")
        ax.set_title(f"{e.event_id}  {pd.Timestamp(e.wall_time):%a %d %H:%M}\nSNR {e.snr:.0f}  nb {e.n_neighbours_in_window}"
                     f"{'  DROP' if e.has_dropout else ''}{'  TRUNC' if e.truncated else ''}", fontsize=7.5)
        ax.tick_params(labelsize=6); ax.set_xlim(-10.6, 10.6)
    fig.suptitle("30 random SNR>=20 windows (stratified run x 6-h bin, seed 0). Blue: feature window (counts); "
                 "shaded: C's [onset, offset]; red dots: dropout-affected samples; dotted: window centre (energy centroid)",
                 fontsize=10)
    fig.tight_layout(); fig.savefig(os.path.join(HERE, "fig_gallery.png"), dpi=110)
    print(P[["event_id", "wall_time", "snr", "n_neighbours_in_window", "has_dropout", "truncated"]].to_string())


def skill_cv(X, y, groups, n_splits=5, seed=0):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import balanced_accuracy_score
    y = np.asarray(y); classes = np.unique(y); K = len(classes)
    pred = np.empty_like(y)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=seed)
        clf.fit(X[tr], y[tr]); pred[te] = clf.predict(X[te])
    b = balanced_accuracy_score(y, pred)
    return (b - 1 / K) / (1 - 1 / K), K


def leakage():
    from extraction.morph import MORPH_NAMES
    m = load_meta()
    m["blk30"] = m.wall_time.dt.floor("30min").astype(str)
    m["hbin"] = m.hour // 6
    rng = np.random.default_rng(0)
    rows = []
    sets = {"all": m, "SNR>=20": m[m.snr >= 20], "SNR>=30": m[m.snr >= 30],
            "SNR>=20, no dropout": m[(m.snr >= 20) & ~m.has_dropout], "all, no dropout": m[~m.has_dropout],
            "SNR>=20, runs 1-3 only": m[(m.snr >= 20) & (m.run_id > 0)]}
    for fset, cols in (("morph21", [f"m_{k}" for k in MORPH_NAMES]), ("morph8", [f"m8_{k}" for k in MORPH_NAMES])):
        for sname, s in sets.items():
            if fset == "morph8" and sname not in ("all", "SNR>=20"):
                continue
            X = s[cols].to_numpy(float)
            for target, grp in (("date", "blk30"), ("run_id", "blk30"), ("hbin", "blk30"), ("has_dropout", "date")):
                if sname.endswith("no dropout") and target == "has_dropout":
                    continue
                if sname.endswith("runs 1-3 only") and target not in ("run_id", "date"):
                    continue
                y = s[target].astype(str).to_numpy(); g = s[grp].to_numpy()
                sk, K = skill_cv(X, y, g)
                # null: permute labels at the level of the CV group (labels constant within a 30-min block
                # stay constant within it); has_dropout (grouped by day): permute event labels within day
                if grp == "blk30":
                    ug = pd.unique(g); lab = s.groupby("blk30")[target].first().astype(str)
                    perm = dict(zip(ug, rng.permutation(lab.loc[ug].to_numpy())))
                    yn = np.array([perm[x] for x in g])
                else:
                    yn = y.copy()
                    for d in np.unique(g):
                        ii = np.flatnonzero(g == d); yn[ii] = rng.permutation(yn[ii])
                nk, _ = skill_cv(X, yn, g)
                rows.append(dict(features=fset, subset=sname, target=target, n=len(s), K=K, cv_groups=grp,
                                 n_groups=len(np.unique(g)), skill=sk, null_skill=nk))
                print(rows[-1], flush=True)
    L = pd.DataFrame(rows); L.to_csv(os.path.join(HERE, "tab_leakage.csv"), index=False)
    print(L.round(3).to_string(index=False))


def crosscheck():
    import importlib.util
    from extraction.blockio import chunk_index
    from extraction._vendor import bsd_io
    # ---- (e1) 2-h span: standalone C (C's own I/O on device ticks, C's own detector.py) vs pipeline
    sys.path.insert(0, os.path.join(REPO, "research", "extraction", "C_detection"))
    import cio
    spec = importlib.util.spec_from_file_location("C_det_orig", os.path.join(REPO, "research", "extraction", "C_detection", "detector.py"))
    Cd = importlib.util.module_from_spec(spec); spec.loader.exec_module(Cd)
    ix = chunk_index(); runs = sorted(ix.run.unique())
    m = load_meta()
    out = []
    for rid, t0 in ((2, "2025-02-19 07:00"), (0, "2025-02-06 16:00")):
        ixr = ix[ix.run == runs[rid]].sort_values("index").reset_index(drop=True)
        t0 = pd.Timestamp(t0); t1 = t0 + pd.Timedelta(hours=2)
        p0 = int(np.searchsorted(ixr.wall_start.values, np.datetime64(t0))) - 1
        p1 = int(np.searchsorted(ixr.wall_start.values, np.datetime64(t1)))
        ctx = 3
        x, t, c = cio.load_chunks(os.path.join(bsd_io.DEFAULT_ROOT, runs[rid]), int(ixr["index"][p0 - ctx]), p1 - p0 + 2 * ctx)
        ev, _ = Cd.detect_stream(x, t)
        _, slot_of_raw, _ = Cd.insert_dropouts(x, t)
        # map each C centroid to a raw sample -> (chunk, sample) -> A's run-global slot
        starts = {}
        for ci in np.unique(c):
            r = ixr[ixr["index"] == ci].iloc[0]
            tt = np.fromfile(r.time_path, "<u8"); sl, _ = bsd_io.slots_from_timestamps(tt)
            starts[ci] = int(r.slot_start) + sl
        first = {ci: int(np.flatnonzero(c == ci)[0]) for ci in np.unique(c)}
        cs, snr = [], []
        for e in ev:
            k = int(np.clip(np.searchsorted(slot_of_raw, round(e["centroid_t"] * Cd.FS_RAW)), 0, len(x) - 1))
            cs.append(starts[c[k]][k - first[c[k]]]); snr.append(e["snr"])
        cs = np.array(cs, float); snr = np.array(snr)
        s_lo = int(ixr.slot_start[p0 + 1]); s_hi = int(ixr.slot_start[p1 - 1])      # window strictly inside
        a = (cs >= s_lo) & (cs < s_hi)
        pm = m[(m.run_id == rid) & (m.slot_centroid >= s_lo) & (m.slot_centroid < s_hi)]
        ex = pd.read_parquet(os.path.join(EV, "excluded.parquet"))
        pe = ex[(ex.run_id == rid) & (ex.slot_centroid >= s_lo) & (ex.slot_centroid < s_hi)] if len(ex) else ex
        pc = np.r_[pm.slot_centroid.to_numpy(), pe.slot_centroid.to_numpy() if len(pe) else []]
        psnr = np.r_[pm.snr.to_numpy(), pe.snr.to_numpy() if len(pe) else []]
        ca, sa = cs[a], snr[a]
        dC = np.array([np.abs(pc - v).min() for v in ca]) / 2416 if len(pc) else np.full(len(ca), np.inf)
        dP = np.array([np.abs(ca - v).min() for v in pc]) / 2416 if len(ca) else np.full(len(pc), np.inf)
        j = np.array([np.abs(pc - v).argmin() for v in ca]) if len(pc) else np.zeros(0, int)
        mt = dC < 0.5
        rel = np.abs(psnr[j[mt]] / sa[mt] - 1) if mt.any() else np.array([np.nan])
        hours = (s_hi - s_lo) / 2416 / 3600
        # pipeline block boundaries inside the span
        nb = sum(1 for q in range(0, len(ixr), 60) if s_lo < ixr.slot_start[q] < s_hi)
        row = dict(run=rid, span_start=str(t0), hours=hours, pipeline_block_boundaries_inside=nb,
                   n_C_standalone=int(a.sum()), n_pipeline=len(pc), n_pipeline_excluded=len(pe),
                   C_matched=int(mt.sum()), pipe_unmatched=int((dP >= 0.5).sum()),
                   n20_C=int((sa >= 20).sum()), n20_pipe=int((psnr >= 20).sum()),
                   n20_C_matched=int((mt & (sa >= 20)).sum()),
                   median_abs_centroid_diff_s=float(np.median(dC[mt])) if mt.any() else np.nan,
                   median_rel_snr_diff=float(np.median(rel)), unmatched_C_snr=sorted(np.round(sa[~mt], 1).tolist()),
                   unmatched_pipe_snr=sorted(np.round(psnr[dP >= 0.5], 1).tolist()))
        out.append(row); print(row, flush=True)
    json.dump(out, open(os.path.join(HERE, "crosscheck_span.json"), "w"), indent=1, default=str)
    # ---- (e2) 5 events: E morph (port, true fs) vs D's features.morph_feats on the same windows.
    # D's function has a 5 Hz smoother fixed for fs = 100 Hz, so the E window (fs_eff ~50 Hz) is upsampled x2
    # (resample_poly 2/1, band-limited: content < 8 Hz) and D is called with fs = 2*fs_eff (the fs correction);
    # D's log_energy is a sample sum -> compared after subtracting log10(2*fs_eff).
    D = importlib.util.spec_from_file_location("D_feat", os.path.join(REPO, "research", "extraction", "D_features", "features.py"))
    DF = importlib.util.module_from_spec(D); D.loader.exec_module(DF)
    from scipy import signal
    from extraction.morph import MORPH_NAMES, morph_feats
    rng = np.random.default_rng(5)
    s = m[m.snr >= 20]
    pick = s.iloc[rng.choice(len(s), 5, replace=False)]
    rows = []
    for _, e in pick.iterrows():
        W, _, _ = arrays(e.file); w = np.asarray(W[e.row], float)
        mE = morph_feats(w, e.fs_eff)
        stored = e[[f"m_{k}" for k in MORPH_NAMES]].to_numpy(float)
        mD = DF.morph_feats(signal.resample_poly(w, 2, 1), fs=2 * e.fs_eff)
        mD[MORPH_NAMES.index("log_energy")] -= np.log10(2 * e.fs_eff)
        for k, a_, b_, c_ in zip(MORPH_NAMES, mE, mD, stored):
            rows.append(dict(event_id=e.event_id, feature=k, E=a_, E_stored=c_, D_fs_corrected=b_, diff=a_ - b_))
    X = pd.DataFrame(rows); X.to_csv(os.path.join(HERE, "crosscheck_morph.csv"), index=False)
    print("stored == recomputed:", np.allclose(X.E, X.E_stored, rtol=1e-6, atol=1e-6))
    piv = X.pivot(index="feature", columns="event_id", values="diff").loc[MORPH_NAMES]
    print("E - D (fs-corrected), per feature x event:\n" + piv.round(3).to_string())
    print(X.pivot(index="feature", columns="event_id", values="E").loc[MORPH_NAMES].round(3).to_string())


if __name__ == "__main__":
    {"counts": counts, "gallery": gallery, "leakage": leakage, "crosscheck": crosscheck}[sys.argv[1]]()
