"""Analyse injection/surrogate results.
usage: analyze_inj.py tune     -> picks operating points (FA<=0.2/h on surrogates), writes data/frozen_params.json,
                                  tables data/tab_<split>_*.csv, figures figs/inj_<split>_*.png
       analyze_inj.py holdout  -> same metrics at the frozen operating points (no selection)
Bootstrap: 200 resamples (seed 0) over clusters = (span, realisation)."""
import sys, os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE = os.path.join(os.path.dirname(__file__), '..')
split = sys.argv[1]
inj = pd.read_parquet(f'{HERE}/data/inj_{split}.parquet')
det = pd.read_parquet(f'{HERE}/data/det_{split}.parquet')
sur = pd.read_parquet(f'{HERE}/data/sur_{split}.parquet')
cost = pd.read_parquet(f'{HERE}/data/cost_{split}.parquet')
NAMES = ["FIXED", "FIXEDLOC", "HYST", "STALTA", "CUSUM", "WAVELET", "HYSTBP", "STALTABP", "CUSUMBP", "BDET"]
FA_MAX = 0.2
B, RNG = 200, np.random.default_rng(0)
COL = {"FIXED": "#7f7f7f", "HYST": "#1f77b4", "STALTA": "#ff7f0e", "CUSUM": "#2ca02c", "WAVELET": "#9467bd", "HYSTBP": "#d62728", "FIXEDLOC": "#bcbd22", "STALTABP": "#8c564b", "CUSUMBP": "#17becf", "BDET": "#e377c2", "BDET@legacy": "#f7b6d2",
       "FIXED@legacy": "#000000"}


def boot(cl_vals, fn):
    """cl_vals: DataFrame with 'cl' column; fn: DataFrame -> float. Returns (est, lo, hi)."""
    cls = cl_vals.cl.unique()
    g = {c: d for c, d in cl_vals.groupby('cl')}
    est = fn(cl_vals)
    bs = []
    for _ in range(B):
        pick = RNG.choice(cls, len(cls), replace=True)
        bs.append(fn(pd.concat([g[c] for c in pick])))
    return est, np.nanpercentile(bs, 2.5), np.nanpercentile(bs, 97.5)


# ------------------------------------------------------------------ surrogate FA rate
sa = sur.groupby(['det', 'thr']).agg(n_fa=('n_fa', 'sum'), hours=('hours', 'sum')).reset_index()
sa['fa_h'] = sa.n_fa / sa.hours
# ------------------------------------------------------------------ operating points
if split == 'tune':
    frozen = {}
    for d in NAMES:
        s = sa[sa.det == d].sort_values('thr')
        ok = s[s.fa_h <= FA_MAX]
        thr = float(ok.thr.min()) if len(ok) else float(s.thr.max())
        frozen[d] = dict(thr=thr, fa_h_surrogate=float(s[s.thr == thr].fa_h.iloc[0]))
    frozen['FIXED']['legacy_thr'] = 5.0   # legacy method's own threshold
    frozen['BDET']['legacy_thr'] = 8.0    # workstream B's own recommended threshold
    json.dump(frozen, open(f'{HERE}/data/frozen_params.json', 'w'), indent=1)
else:
    frozen = json.load(open(f'{HERE}/data/frozen_params.json'))
ops = [(d, frozen[d]['thr']) for d in NAMES] + [(f'{d}@legacy', frozen[d]['legacy_thr']) for d in NAMES if 'legacy_thr' in frozen[d]]


def sel(df, d, thr):
    base = d.split('@')[0]
    if split == 'tune':
        return df[(df.det == base) & (np.isclose(df.thr, thr))]
    return df[(df.det == d) & (np.isclose(df.thr, thr))]


rows, rec_curves, pair_rows = [], {}, []
for d, thr in ops:
    I = sel(inj, d, thr).copy(); I['cl'] = I.span + '_' + I.rep.astype(str)
    Dt = sel(det, d, thr).copy(); Dt['cl'] = Dt.span + '_' + Dt.rep.astype(str)
    S = sel(sur, d, thr)
    iso = I[I.kind == 'iso']; Diso = Dt[Dt.kind == 'iso']
    # recall per SNR
    rc = {}
    for snr, g in iso.groupby('snr'):
        rc[snr] = boot(g, lambda x: x.matched.mean())
    rec_curves[d] = rc
    def snr_at(level, rc=rc):
        s = np.array(sorted(rc)); r = np.array([rc[k][0] for k in s])
        above = np.where(r >= level)[0]
        if len(above) == 0: return np.inf
        j = above[0]
        if j == 0: return s[0]
        return float(np.interp(level, [r[j - 1], r[j]], [s[j - 1], s[j]]))
    # F1 pooled over all isolated injections (all SNR >= 2 = 6 dB) with detection-level FP = fragments + new unmatched
    # per-cluster aggregates so that the cluster bootstrap counts duplicated clusters' FPs correctly
    fpc = Diso.groupby('cl').apply(lambda x: (x.n_frag + x.n_fp_new).sum(), include_groups=False)
    agg = iso.groupby('cl').agg(tp=('matched', 'sum'), n=('matched', 'size'),
                                tp8=('matched', lambda m: m[iso.loc[m.index, 'snr'] >= 8].sum()),
                                n8=('matched', lambda m: (iso.loc[m.index, 'snr'] >= 8).sum())).reset_index()
    agg['fp'] = agg.cl.map(fpc).fillna(0)
    def f1(x, snr_min=2):
        tp, n = (x.tp.sum(), x.n.sum()) if snr_min <= 2 else (x.tp8.sum(), x.n8.sum())
        fp = x.fp.sum() * (n / max(x.n.sum(), 1))   # FP pro-rated to the SNR subset
        p = tp / max(tp + fp, 1); r = tp / max(n, 1)
        return 2 * p * r / max(p + r, 1e-9)
    def prec(x):
        return x.tp.sum() / max(x.tp.sum() + x.fp.sum(), 1)
    hi = iso[(iso.snr >= 10) & iso.matched]
    hi = hi.assign(cl=hi.cl)
    r = dict(det=d, thr=thr,
             fa_h_sur=S.n_fa.sum() / S.hours.sum(),
             recall_snr4=rc[4][0], recall_snr6=rc[6][0], recall_snr8=rc[8][0], recall_snr8_lo=rc[8][1], recall_snr8_hi=rc[8][2],
             snr50=snr_at(0.5), snr90=snr_at(0.9))
    r['F1_all'], r['F1_all_lo'], r['F1_all_hi'] = boot(agg, f1)
    r['F1_snr8'], r['F1_snr8_lo'], r['F1_snr8_hi'] = boot(agg, lambda x: f1(x, 8))
    r['prec_inj'], r['prec_inj_lo'], r['prec_inj_hi'] = boot(agg, prec)
    r['split_rate'] = (hi.n_frag > 0).mean()
    r['onset_mae'], r['onset_lo'], r['onset_hi'] = boot(hi, lambda x: x.onset_err.abs().median())
    r['offset_mae'], r['offset_lo'], r['offset_hi'] = boot(hi, lambda x: x.offset_err.abs().median())
    r['onset_bias'] = hi.onset_err.median(); r['offset_bias'] = hi.offset_err.median()
    r['capture_med'], r['capture_lo'], r['capture_hi'] = boot(hi, lambda x: x.capture.median())
    r['capture_p10'] = hi.capture.quantile(0.1)
    r['dur_med'] = hi.dur.median()
    r['fp_new_h'] = Diso.n_fp_new.sum() / Diso.hours.sum()
    # stretch effect at SNR 5-8
    for st, g in iso[(iso.snr >= 5) & (iso.snr <= 8)].groupby('stretch'):
        r[f'recall_snr5-8_x{st}'] = g.matched.mean()
    # pairs: resolved if both members matched
    P = I[I.kind == 'pair']
    for sep, g in P.groupby('sep'):
        res = g.groupby(['cl', 'pair']).matched.all().reset_index().rename(columns={'matched': 'res'})
        est = boot(res, lambda x: x.res.mean())
        r[f'pair_res_{sep:g}s'] = est[0]
        pair_rows.append(dict(det=d, sep=sep, res=est[0], lo=est[1], hi=est[2]))
    c = cost[(cost.det == d.split('@')[0])]
    r['cpu_s_per_h'] = c.cpu_s.sum() / c.hours.sum() if split == 'holdout' else c[np.isclose(c.thr, thr)].cpu_s.sum() / c[np.isclose(c.thr, thr)].hours.sum()
    rows.append(r)
tab = pd.DataFrame(rows)
tab.to_csv(f'{HERE}/data/tab_{split}_summary.csv', index=False)
pd.DataFrame(pair_rows).to_csv(f'{HERE}/data/tab_{split}_pairs.csv', index=False)
sa.to_csv(f'{HERE}/data/tab_{split}_surrogate_fa.csv', index=False)
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 60)
print(json.dumps(frozen, indent=0))
print(tab.round(3).T.to_string())

# ------------------------------------------------------------------ figures
fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
for d, rc in rec_curves.items():
    s = sorted(rc); e = np.array([rc[k] for k in s])
    ax[0].plot(s, e[:, 0], 'o-', color=COL[d], label=f'{d}', lw=1.6 if d == 'HYST' else 1.1)
    ax[0].fill_between(s, e[:, 1], e[:, 2], color=COL[d], alpha=.12)
ax[0].set_xscale('log'); ax[0].set_xticks([2, 3, 4, 6, 10, 20, 40]); ax[0].set_xticklabels([2, 3, 4, 6, 10, 20, 40])
ax[0].set_xlabel('injected peak SNR (|peak| / robust sigma)'); ax[0].set_ylabel('recall'); ax[0].grid(alpha=.3)
ax[0].set_title(f'{split}: recall vs SNR at operating point (95% CI)'); ax[0].legend(fontsize=8)
if split == 'tune':
    for d in NAMES:
        s = sa[sa.det == d].sort_values('thr')
        rec8 = [inj[(inj.det == d) & np.isclose(inj.thr, th) & (inj.kind == 'iso') & (inj.snr == 8)].matched.mean() for th in s.thr]
        ax[1].plot(s.fa_h + 0.01, rec8, 'o-', color=COL[d], label=d, ms=3)
        ax[1].plot(frozen[d]['fa_h_surrogate'] + 0.01, rec8[list(s.thr).index(frozen[d]['thr'])], 's', color=COL[d], ms=9, mfc='none')
    ax[1].axvline(FA_MAX, color='k', ls=':', lw=1); ax[1].set_xscale('log')
    ax[1].set_xlabel('false alarms / h on phase-randomised surrogates (+0.01)'); ax[1].set_ylabel('recall at SNR 8')
    ax[1].set_title('threshold sweep (square = chosen operating point)'); ax[1].grid(alpha=.3); ax[1].legend(fontsize=8)
else:
    pr = pd.DataFrame(pair_rows)
    for d in pr.det.unique():
        g = pr[pr.det == d]; ax[1].errorbar(g.sep, g.res, yerr=[g.res - g.lo, g.hi - g.res], fmt='o-', color=COL[d], label=d, capsize=2)
    ax[1].set_xlabel('peak separation (s)'); ax[1].set_ylabel('both vehicles resolved'); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(f'{HERE}/figs/inj_{split}_recall.png', dpi=90)

# ------------------------------------------------------------------ dropout-density strata (coordinator request)
# high: Run 1 (0204) spans on/after Feb 6 (one ~105-sample dropout every 2.4-5 s); low: everything else.
def stratum(span):
    return 'drop_high' if span.startswith('r1_') and span[3:8] not in ('wed05',) else 'drop_low'
srows = []
for d, thr in ops:
    I = sel(inj, d, thr); Dt = sel(det, d, thr); S = sel(sur, d, thr)
    for st in ('drop_high', 'drop_low'):
        Ii = I[(I.kind == 'iso') & (I.span.map(stratum) == st)]
        Di = Dt[(Dt.kind == 'iso') & (Dt.span.map(stratum) == st)]
        Si = S[S.span.map(stratum) == st]
        if len(Ii) == 0:
            continue
        tp = Ii.matched.sum(); fn = (~Ii.matched).sum(); fp = (Di.n_frag + Di.n_fp_new).sum()
        p = tp / max(tp + fp, 1); rr = tp / max(tp + fn, 1)
        srows.append(dict(det=d, stratum=st, n_inj=len(Ii), recall_snr8=Ii[Ii.snr == 8].matched.mean(),
                          F1_all=2 * p * rr / max(p + rr, 1e-9), fa_h_sur=Si.n_fa.sum() / max(Si.hours.sum(), 1e-9),
                          fp_new_h=Di.n_fp_new.sum() / max(Di.hours.sum(), 1e-9)))
st = pd.DataFrame(srows); st.to_csv(f'{HERE}/data/tab_{split}_strata.csv', index=False)
print(st.round(3).to_string())
