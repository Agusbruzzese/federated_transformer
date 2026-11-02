"""Segment representation: fixed-length centroid-centred window vs variable length.
Uses the large-sample pass (data/diurnal_events.parquet) for the recommended detector (HYSTBP).
For each candidate length L: truncation = fraction of events whose 1-99 % energy extent is not contained in the
centroid-centred window; contamination = fraction of windows that overlap another detected event's energy extent;
also the energy-extent (e_dur) and segment-duration distributions.
A second, extended-extent variant uses the energy extent computed on a +-15 s context (ramps included) for a
subsample of events (data/segrepr_context.parquet produced here)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE = os.path.join(os.path.dirname(__file__), '..')
import json
RULE = json.load(open(f'{HERE}/data/postfilter_rule.json'))
allev = pd.read_parquet(f'{HERE}/data/diurnal_events.parquet')
allev = allev[allev.det == 'HYSTBP'].copy()
LS = [4, 6, 8, 10, 12, 14, 16, 20, 22, 24]
rows = []
for tier, ev in (('HYSTBP (all)', allev), ('HYSTBP+SNR>=30 (clean)', allev[allev.snr >= RULE['snr_min']])):
    for L in LS:
        # truncation: context energy extent (1-99 %, ramps included, +-10 s) not inside the centroid-centred window
        trunc = ((ev.x_on < ev.centroid_t - L / 2) | (ev.x_off > ev.centroid_t + L / 2)).mean()
        trunc_core = ((ev.e_on < ev.centroid_t - L / 2) | (ev.e_off > ev.centroid_t + L / 2)).mean()
        # hysteresis segment (|env| > 2 sigma, i.e. ramps down to 2 sigma) not inside the window -- the recommended measure
        trunc_seg = ((ev.onset_t < ev.centroid_t - L / 2) | (ev.offset_t > ev.centroid_t + L / 2)).mean()
        cont, cont_big = [], []
        for (r, b), g in ev.groupby(['run', 'block']):
            o_all = allev[(allev.run == r) & (allev.block == b)]
            for c, x0 in zip(g.centroid_t.values, g.x_on.values):
                w0, w1 = c - L / 2, c + L / 2
                o = (o_all.e_on < w1) & (o_all.e_off > w0) & (np.abs(o_all.centroid_t - c) > 1e-6)
                cont.append(o.any()); cont_big.append((o & (o_all.snr >= RULE['snr_min'])).any())
        rows.append(dict(tier=tier, L_s=L, truncation_seg=trunc_seg, truncation_ctx=trunc, truncation_core=trunc_core, contamination_any=np.mean(cont),
                         contamination_clean=np.mean(cont_big), n=len(ev)))
tab = pd.DataFrame(rows)
ev = allev
q = pd.concat([allev[['dur', 'e_dur', 'x_dur']].quantile([0.5, 0.75, 0.9, 0.95, 0.99]).T.assign(tier='all'),
               allev[allev.snr >= RULE['snr_min']][['dur', 'e_dur', 'x_dur']].quantile([0.5, 0.75, 0.9, 0.95, 0.99]).T.assign(tier='clean')])
tab.to_csv(f'{HERE}/data/tab_segrepr.csv', index=False)
# PRE-REGISTERED RULE AS WRITTEN (audit issue 10): L = smallest length covering the 95th percentile of the 1-99 %
# energy extent INCLUDING ramps (x_dur: +-10 s context, clipped halfway to neighbours). Note: x_dur is capped by that
# context (segment + 20 s), so the rule's L is a lower bound.
rule = []
for tier, ev_ in (('HYSTBP (all)', allev), ('HYSTBP+SNR>=30 (clean)', allev[allev.snr >= RULE['snr_min']]),
                  ('HYSTBP+SNR>=20', allev[allev.snr >= 20])):
    p95 = ev_.x_dur.quantile(0.95)
    rule.append(dict(tier=tier, n=len(ev_), x_dur_p50=ev_.x_dur.median(), x_dur_p95=p95, L_rule_s=float(np.ceil(p95)),
                     x_dur_at_context_cap=float(((ev_.x_dur >= (ev_.dur + 19.5))).mean())))
rule = pd.DataFrame(rule); rule.to_csv(f'{HERE}/data/tab_segrepr_rule.csv', index=False); print(rule.round(3).to_string())
q.to_csv(f'{HERE}/data/tab_durations.csv')
print(tab.round(3).to_string()); print(q.round(2))
# by SNR class (large events have longer ramps)
ev['snr_cls'] = pd.cut(ev.snr, [0, 10, 30, 100, 1e9], labels=['5-10', '10-30', '30-100', '>100'])
print(ev.groupby('snr_cls', observed=True)[['dur', 'e_dur', 'x_dur']].quantile(0.95).round(2))
ev.groupby('snr_cls', observed=True)[['dur', 'e_dur', 'x_dur']].quantile([0.5, 0.95]).to_csv(f'{HERE}/data/tab_durations_by_snr.csv')
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
cl = ev[ev.snr >= RULE['snr_min']]
ax[0].hist(cl.dur.clip(0, 30), bins=60, alpha=.6, label='segment (hysteresis) duration')
ax[0].hist(cl.x_dur.clip(0, 30), bins=60, alpha=.6, label='1-99 % energy extent incl. ramps')
ax[0].set_xlabel('s'); ax[0].legend(); ax[0].set_title(f'clean tier (SNR>=30), n={len(cl)}')
for tier, g in tab.groupby('tier'):
    ax[1].plot(g.L_s, g.truncation_seg, 'o-', label=f'segment truncated: {tier}')
    ax[1].plot(g.L_s, g.contamination_clean, 's--', label=f'another clean event in window: {tier}')
ax[1].set_xlabel('fixed window length L (s), centred on energy centroid'); ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
plt.tight_layout(); plt.savefig(f'{HERE}/figs/segment_repr.png', dpi=90)
