"""Large-sample summaries + extra injection statistics.
 - diurnal event-rate plausibility (pre-registered: night(00-05) rate < 20 % of weekday peak-hour rate; weekend 07-10 rate <
   weekday 07-10 rate), per detector, tuning runs and Run-3 holdout separately; figs/diurnal.png
 - chunk-boundary straddling, dropout coincidence vs chance, compute cost per hour of data
 - two-tier injection metrics for HYST-BP (all / SNR>=30 clean tier), paired bootstrap F1 differences
All bootstraps: 200 resamples, seed 0. Outputs data/tab_*.csv."""
import sys, os, json
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE = os.path.join(os.path.dirname(__file__), '..')
ev = pd.read_parquet(f'{HERE}/data/diurnal_events.parquet')
bl = pd.read_parquet(f'{HERE}/data/diurnal_blocks.parquet')
RULE = json.load(open(f'{HERE}/data/postfilter_rule.json'))
RNG = np.random.default_rng(0)
pd.set_option('display.width', 250)
ev['split'] = np.where(ev.run == 2, 'holdout', 'tune'); bl['split'] = np.where(bl.run == 2, 'holdout', 'tune')
ev['weekend'] = ev.dow >= 5; bl['weekend'] = bl.dow >= 5
clean = ev[(ev.det == 'HYSTBP') & (ev.snr >= RULE['snr_min'])].copy(); clean['det'] = 'HYSTBP+SNR>=30'
ev = pd.concat([ev, clean])
DETS = list(ev.det.unique())

# ------------------------------------------------------------------ diurnal
rows = []; curves = {}
for split in ('tune', 'holdout'):
    B = bl[bl.split == split]
    for d in DETS:
        E = ev[(ev.det == d) & (ev.split == split)]
        for we in (False, True):
            h = B[B.weekend == we].groupby('hour').hours.sum()
            n = E[E.weekend == we].groupby('hour').size().reindex(h.index, fill_value=0)
            curves[(split, d, we)] = (n / h)
        wk = curves[(split, d, False)]
        # bootstrap over blocks for the ratio night / peak
        def stat(Bs):
            Es = E[E.block.isin(Bs.block) & (~E.weekend)]
            hh = Bs[~Bs.weekend].groupby('hour').hours.sum(); nn = Es.groupby('hour').size().reindex(hh.index, fill_value=0)
            r = nn / hh
            night = nn[(nn.index < 5)].sum() / max(hh[hh.index < 5].sum(), 1e-9)
            return night / max(r.max(), 1e-9)
        est = stat(B)
        bs = []
        for _ in range(200):
            Bs = B.iloc[RNG.integers(0, len(B), len(B))]
            bs.append(stat(Bs))
        nightw = E[~E.weekend & (E.hour < 5)].shape[0] / max(B[~B.weekend & (B.hour < 5)].hours.sum(), 1e-9)
        amw = E[~E.weekend & E.hour.between(7, 9)].shape[0] / max(B[~B.weekend & B.hour.between(7, 9)].hours.sum(), 1e-9)
        amwe = E[E.weekend & E.hour.between(7, 9)].shape[0] / max(B[B.weekend & B.hour.between(7, 9)].hours.sum(), 1e-9)
        rows.append(dict(split=split, det=d, rate_per_h=len(E) / B.hours.sum(), night_rate=nightw, peak_hour=int(wk.idxmax()),
                         peak_rate=wk.max(), night_over_peak=est, ci_lo=np.nanpercentile(bs, 2.5), ci_hi=np.nanpercentile(bs, 97.5),
                         wkday_07_10=amw, wkend_07_10=amwe,
                         pass_night=bool(est < 0.2), pass_weekend=(bool(amwe < amw) if split == 'tune' else None)))
dt = pd.DataFrame(rows); dt.to_csv(f'{HERE}/data/tab_diurnal.csv', index=False)
print(dt.round(3).to_string())
fig, ax = plt.subplots(1, 3, figsize=(17, 4.2), sharey=False)
show = ['FIXED@legacy', 'FIXEDLOC', 'BDET@legacy', 'CUSUMBP', 'HYSTBP', 'HYSTBP+SNR>=30']
for a, (split, we, title) in zip(ax, [('tune', False, 'tuning runs, weekdays'), ('tune', True, 'tuning runs, weekend (Feb 8-9)'),
                                      ('holdout', False, 'Run 3 holdout, weekdays')]):
    for d in show:
        c = curves[(split, d, we)]
        a.plot(c.index, c.values, 'o-', ms=3, lw=2 if d.startswith('HYSTBP') else 1, label=d)
    a.set_title(title); a.set_xlabel('hour of day (wall clock)'); a.set_yscale('log'); a.grid(alpha=.3); a.set_xticks(range(0, 24, 3))
ax[0].set_ylabel('events / hour'); ax[0].legend(fontsize=7)
plt.tight_layout(); plt.savefig(f'{HERE}/figs/diurnal.png', dpi=90)

# ------------------------------------------------------------------ boundaries, dropouts, cost
r2 = []
for d in DETS:
    E = ev[ev.det == d]
    for strat, g in (('all', E), ('r1_after_Feb6', E[(E.run == 0) & (E.date >= '02-06')]), ('other', E[~((E.run == 0) & (E.date >= '02-06'))])):
        Bm = bl if strat == 'all' else (bl[(bl.run == 0) & (bl.date >= '02-06')] if strat == 'r1_after_Feb6' else bl[~((bl.run == 0) & (bl.date >= '02-06'))])
        r2.append(dict(det=d, stratum=strat, n=len(g), straddle_segment=g.straddle_seg.mean(), straddle_energy=g.straddle_e.mean(),
                       has_dropout=g.has_dropout.mean(), dropout_cover=Bm.drop_frac.mean(),
                       rate_per_h=len(g) / max(Bm.hours.sum(), 1e-9)))
bt = pd.DataFrame(r2); bt.to_csv(f'{HERE}/data/tab_boundaries_dropouts.csv', index=False)
print(bt.round(3).to_string())
cost = dict(io_s_per_h=bl.t_io.sum() / bl.data_hours.sum(), frontend_prep_s_per_h=bl.t_prep.sum() / bl.data_hours.sum(),
            **{c.replace('cpu_', 'det_s_per_h_'): bl[c].sum() / bl.data_hours.sum() for c in bl.columns if c.startswith('cpu_')})
pd.Series(cost).to_csv(f'{HERE}/data/tab_cost.csv'); print(pd.Series(cost).round(4))

# ------------------------------------------------------------------ two-tier injection metrics + paired bootstrap
out = []
for split in ('tune', 'holdout'):
    inj = pd.read_parquet(f'{HERE}/data/inj_{split}.parquet'); det = pd.read_parquet(f'{HERE}/data/det_{split}.parquet')
    fz = json.load(open(f'{HERE}/data/frozen_params.json'))
    def pick(df, d):
        base = d.split('@')[0]; thr = fz[base]['legacy_thr'] if '@' in d else fz[base]['thr']
        key = d if (split == 'holdout') else base
        return df[(df.det == key) & np.isclose(df.thr, thr) & (df.kind == 'iso')]
    agg = {}
    for d in ['HYSTBP', 'CUSUMBP', 'FIXEDLOC', 'STALTABP', 'WAVELET', 'FIXED', 'FIXED@legacy', 'BDET', 'BDET@legacy']:
        I = pick(inj, d).copy(); Dt = pick(det, d).copy()
        I['cl'] = I.span + '_' + I.rep.astype(str); Dt['cl'] = Dt.span + '_' + Dt.rep.astype(str)
        fp = Dt.groupby('cl').apply(lambda x: (x.n_frag + x.n_fp_new).sum(), include_groups=False)
        a = I.groupby('cl').agg(tp=('matched', 'sum'), n=('matched', 'size')).join(fp.rename('fp')).fillna(0)
        agg[d] = a
    def f1(a):
        tp, n, fp = a.tp.sum(), a.n.sum(), a.fp.sum(); p = tp / max(tp + fp, 1); r = tp / max(n, 1)
        return 2 * p * r / max(p + r, 1e-9)
    cls = agg['HYSTBP'].index.values
    for other in ['CUSUMBP', 'FIXEDLOC', 'STALTABP', 'WAVELET', 'FIXED', 'FIXED@legacy', 'BDET', 'BDET@legacy']:
        diffs = []
        for _ in range(200):
            c = RNG.choice(cls, len(cls), replace=True)
            diffs.append(f1(agg['HYSTBP'].loc[c]) - f1(agg[other].reindex(c).fillna(0)))
        out.append(dict(split=split, comparison=f'HYSTBP - {other}', dF1=f1(agg['HYSTBP']) - f1(agg[other]),
                        lo=np.percentile(diffs, 2.5), hi=np.percentile(diffs, 97.5)))
    # clean tier on injections: recall by injected SNR when the detection's own SNR must be >= 30 is approximated by
    # injected SNR >= 30 (only the SNR-40 level qualifies); report recall of SNR-40 injections for HYSTBP
    I = pick(inj, 'HYSTBP'); out.append(dict(split=split, comparison='HYSTBP recall @ injected SNR 40', dF1=I[I.snr == 40].matched.mean(), lo=np.nan, hi=np.nan))
pt = pd.DataFrame(out); pt.to_csv(f'{HERE}/data/tab_paired.csv', index=False); print(pt.round(3).to_string())
