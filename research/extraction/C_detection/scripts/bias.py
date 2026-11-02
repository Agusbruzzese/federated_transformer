"""Audit issue 5: selection bias of the clean tier (HYST-BP, SNR>=30) vs all HYST-BP events and an SNR>=20 tier.
Large-sample pass (data/diurnal_events.parquet, ~1/4 of all chunks). Compared: |peak| amplitude (counts), local noise
sigma (= |peak|/snr), segment duration, hour of day, dominant polarity (sign of the largest-|z| sample), run, dropout
stratum. Two-sample tests of each tier vs all events (KS for continuous, chi-square for categorical).
Also: how many events SNR>=20 adds over SNR>=30, per hour and extrapolated to the full 14-day dataset.
-> data/tab_bias_summary.csv, data/tab_bias_categorical.csv, figs/clean_tier_bias.png"""
import os, json
import numpy as np, pandas as pd
from scipy.stats import ks_2samp, chi2_contingency
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE = os.path.join(os.path.dirname(__file__), '..')
RULE = json.load(open(f'{HERE}/data/postfilter_rule.json'))
ev = pd.read_parquet(f'{HERE}/data/diurnal_events.parquet'); bl = pd.read_parquet(f'{HERE}/data/diurnal_blocks.parquet')
ev = ev[ev.det == 'HYSTBP'].copy()
ev['amp'] = ev.peak_z.abs(); ev['sigma'] = ev.amp / ev.snr; ev['pol'] = np.where(ev.peak_z > 0, '+', '-')
ev['hour_cls'] = pd.cut(ev.hour, [-1, 5, 6, 9, 15, 19, 23], labels=['00-05', '06', '07-09', '10-15', '16-19', '20-23'])
ev['stratum'] = np.where((ev.run == 0) & (ev.date >= '02-06'), 'run1>=Feb6 (dense dropouts)', 'other')
ev['run'] = ev.run.map({0: 'Run1', 1: 'Run2', 2: 'Run3', 3: 'Run4'})
tiers = {'all (SNR>=5)': ev, 'SNR>=20': ev[ev.snr >= 20], 'clean SNR>=30': ev[ev.snr >= RULE['snr_min']]}
hours_sampled = bl.hours.sum()
# full-dataset hours: all chunks of all runs (41.4 s each)
import sys; sys.path.insert(0, os.path.join(HERE)); import cio
total_hours = sum(len(cio.index(r)) for r in cio.runs()) * 41.4 / 3600
rows = []
for name, g in tiers.items():
    r = dict(tier=name, n_sampled=len(g), per_hour=len(g) / hours_sampled, est_total_14d=len(g) / hours_sampled * total_hours)
    for col in ('amp', 'sigma', 'dur', 'snr'):
        q = g[col].quantile([0.1, 0.5, 0.9])
        r.update({f'{col}_p10': q[0.1], f'{col}_p50': q[0.5], f'{col}_p90': q[0.9]})
        if name != 'all (SNR>=5)':
            r[f'{col}_KS_vs_all'] = ks_2samp(g[col], ev[col]).statistic
    r['frac_positive_pol'] = (g.pol == '+').mean()
    r['frac_07_09'] = g.hour.between(7, 9).mean(); r['frac_00_05'] = (g.hour <= 5).mean()
    rows.append(r)
summ = pd.DataFrame(rows)
summ['added_by_SNR20_vs_30'] = np.nan
summ.loc[summ.tier == 'SNR>=20', 'added_by_SNR20_vs_30'] = len(tiers['SNR>=20']) - len(tiers['clean SNR>=30'])
summ.to_csv(f'{HERE}/data/tab_bias_summary.csv', index=False)
cat = []
for col in ('hour_cls', 'pol', 'run', 'stratum'):
    for name, g in tiers.items():
        d = g[col].value_counts(normalize=True)
        for k, v in d.items():
            cat.append(dict(variable=col, level=str(k), tier=name, share=v))
    for name in ('SNR>=20', 'clean SNR>=30'):
        sel = ev.snr >= (20 if name == 'SNR>=20' else RULE['snr_min'])
        ct = pd.crosstab(ev[col], sel)
        p = chi2_contingency(ct)[1]
        cat.append(dict(variable=col, level='chi2 p (in-tier vs rest)', tier=name, share=p))
cat = pd.DataFrame(cat); cat.to_csv(f'{HERE}/data/tab_bias_categorical.csv', index=False)
pd.set_option('display.width', 250)
print(f'total data hours {total_hours:.0f}, sampled {hours_sampled:.1f}')
print(summ.round(3).T.to_string())
print(cat.pivot_table(index=['variable', 'level'], columns='tier', values='share').round(3).to_string())
fig, ax = plt.subplots(1, 4, figsize=(18, 3.8))
for name, g in tiers.items():
    ax[0].hist(np.log10(g.amp), bins=40, histtype='step', density=True, label=name)
    ax[1].hist(g.dur.clip(0, 20), bins=40, histtype='step', density=True, label=name)
    h = g.hour.value_counts(normalize=True).sort_index(); ax[2].plot(h.index, h.values, 'o-', ms=3, label=name)
    ax[3].hist(np.log10(g.sigma), bins=40, histtype='step', density=True, label=name)
ax[0].set_xlabel('log10 |peak| (counts)'); ax[1].set_xlabel('segment duration (s)'); ax[2].set_xlabel('hour of day'); ax[3].set_xlabel('log10 local sigma (counts)')
ax[2].set_ylabel('share of events'); ax[0].legend(fontsize=8)
plt.tight_layout(); plt.savefig(f'{HERE}/figs/clean_tier_bias.png', dpi=90)
