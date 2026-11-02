"""Audit issues 6 and 8: exact (Garwood) Poisson 95 % CIs for count-based rates, replacing degenerate bootstraps.
 - surrogate false-alarm rate per detector at the frozen operating points (tune and holdout)
 - weekday night (00-05 h) event rate and night/peak ratio per tier, tune and holdout (diurnal pass);
   the ratio CI uses the exact conditional binomial method for a ratio of two Poisson rates.
-> data/tab_poisson_fa.csv, data/tab_poisson_night.csv"""
import os, json
import numpy as np, pandas as pd
from scipy.stats import chi2, beta
HERE = os.path.join(os.path.dirname(__file__), '..')
FZ = json.load(open(f'{HERE}/data/frozen_params.json'))
RULE = json.load(open(f'{HERE}/data/postfilter_rule.json'))


def garwood(k, T):
    lo = 0.0 if k == 0 else chi2.ppf(0.025, 2 * k) / 2
    hi = chi2.ppf(0.975, 2 * k + 2) / 2
    return k / T, lo / T, hi / T


def ratio_ci(k1, T1, k2, T2):
    """exact CI for (k1/T1)/(k2/T2) via the binomial p = k1/(k1+k2) (Clopper-Pearson)."""
    n = k1 + k2
    lo = 0.0 if k1 == 0 else beta.ppf(0.025, k1, n - k1 + 1)
    hi = 1.0 if k1 == n else beta.ppf(0.975, k1 + 1, n - k1)
    f = lambda p: p / (1 - p) * T2 / T1 if p < 1 else np.inf
    return (k1 / T1) / (k2 / T2) if k2 else np.inf, f(lo), f(hi)


rows = []
for split in ('tune', 'holdout'):
    s = pd.read_parquet(f'{HERE}/data/sur_{split}.parquet')
    for d in FZ:
        for lab, thr in [(d, FZ[d]['thr'])] + ([(f'{d}@legacy', FZ[d]['legacy_thr'])] if 'legacy_thr' in FZ[d] else []):
            key = d if split == 'tune' else lab
            g = s[(s.det == key) & np.isclose(s.thr, thr)]
            if not len(g):
                continue
            k, T = int(g.n_fa.sum()), float(g.hours.sum())
            r, lo, hi = garwood(k, T)
            rows.append(dict(split=split, det=lab, thr=thr, n_fa=k, surrogate_hours=T, fa_per_h=r, lo=lo, hi=hi))
fa = pd.DataFrame(rows); fa.to_csv(f'{HERE}/data/tab_poisson_fa.csv', index=False)

ev = pd.read_parquet(f'{HERE}/data/diurnal_events.parquet'); bl = pd.read_parquet(f'{HERE}/data/diurnal_blocks.parquet')
tiers = {'HYSTBP (all)': ev[ev.det == 'HYSTBP'], 'HYSTBP SNR>=20': ev[(ev.det == 'HYSTBP') & (ev.snr >= 20)],
         'HYSTBP SNR>=30 (clean)': ev[(ev.det == 'HYSTBP') & (ev.snr >= RULE['snr_min'])],
         'BDET@8': ev[ev.det == 'BDET@legacy'], 'FIXED legacy': ev[ev.det == 'FIXED@legacy'], 'FIXEDLOC': ev[ev.det == 'FIXEDLOC']}
rows = []
for split, runs in (('tune', [0, 1, 3]), ('holdout', [2])):
    B = bl[bl.run.isin(runs) & (bl.dow < 5)]
    for name, E in tiers.items():
        E = E[E.run.isin(runs) & (E.dow < 5)]
        hn = B[B.hour < 5].hours.sum(); kn = int((E.hour < 5).sum())
        hh = B.groupby('hour').hours.sum(); kh = E.groupby('hour').size().reindex(hh.index, fill_value=0)
        ph = int((kh / hh).idxmax()); kp, Tp = int(kh[ph]), float(hh[ph])
        r, lo, hi = garwood(kn, hn)
        q, qlo, qhi = ratio_ci(kn, hn, kp, Tp)
        rows.append(dict(split=split, tier=name, night_events=kn, night_hours=hn, night_rate=r, night_lo=lo, night_hi=hi,
                         peak_hour=ph, peak_events=kp, peak_hours=Tp, night_over_peak=q, ratio_lo=qlo, ratio_hi=qhi,
                         pass_night_lt_0_2=bool(qhi < 0.2), note='weekend untestable on holdout' if split == 'holdout' else ''))
ni = pd.DataFrame(rows); ni.to_csv(f'{HERE}/data/tab_poisson_night.csv', index=False)
pd.set_option('display.width', 250)
print(fa.round(3).to_string()); print(ni.round(4).to_string())
