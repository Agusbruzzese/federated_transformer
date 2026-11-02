"""Parameter-perturbation stability of the recommended detector (HYST-BP) on the real TUNING spans.
Each parameter is perturbed by -25 % and +25 % one at a time (booleans toggled). Criteria (pre-registered):
event count change < 15 % and >= 90 % of reference events keep an overlapping counterpart (and vice versa).
Also run for the other detectors' main threshold (+-25 %) for comparison.
usage: stability.py -> data/tab_stability.csv"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, os.path.dirname(__file__))
from common import *
import pandas as pd
from inject_eval import in_eval, run_setting

FZ = json.load(open(os.path.join(os.path.dirname(__file__), '..', 'data', 'frozen_params.json')))
REF = dict(D.DEFAULTS)
PERTURB = ['k_on', 'k_off', 'env_s', 'merge_gap_s', 'min_dur_s', 'split_ratio', 'split_min_sep_s', 'hp_s',
           'pulse_med_s', 'base_win_s', 'sigma_win_s', 'mask_k', 'mask_dil_s']
PRE = {'pulse_med_s', 'base_win_s', 'sigma_win_s', 'mask_k', 'mask_dil_s'}   # affect preprocessing


def segs_for(sp, m, p):
    P = prep(sp['y'], p)
    zh, sh = D.highpass_channel(P['zd'], p['hp_s'])
    return in_eval(D.detect_hyst(sp['t'], zh, sh, p), m['e0'], m['e1'])


def overlap_frac(a, b):
    """fraction of segments in a that overlap any segment in b"""
    if len(a) == 0:
        return np.nan
    if len(b) == 0:
        return 0.0
    return float(np.mean([np.any((b[:, 0] < s[1]) & (s[0] < b[:, 1])) for s in a]))


def compare(ref, alt):
    n0 = sum(len(r) for r in ref); n1 = sum(len(a) for a in alt)
    kept = np.nanmean(np.concatenate([[overlap_frac(r, a)] * len(r) for r, a in zip(ref, alt) if len(r)]))
    new = np.nanmean(np.concatenate([[overlap_frac(a, r)] * len(a) for r, a in zip(ref, alt) if len(a)]))
    return n0, n1, (n1 - n0) / max(n0, 1), kept, new


spans = [(m, load_span(m['name'])) for m in tune_spans()]
ref = [segs_for(sp, m, REF) for m, sp in spans]
rows = []
for k in PERTURB + ['split']:
    for f in ([0.75, 1.25] if k != 'split' else [None]):
        p = dict(REF)
        if k == 'split':
            p['split'] = not REF['split']; lab = f'split={p["split"]}'
        else:
            p[k] = REF[k] * f; lab = f'{k} x{f}'
            if k == 'k_off':
                p[k] = min(p[k], p['k_on'] - 0.5)
        alt = [segs_for(sp, m, p) for m, sp in spans]
        n0, n1, dn, kept, new = compare(ref, alt)
        rows.append(dict(det='HYSTBP', param=lab, n_ref=n0, n_alt=n1, count_change=dn, ref_kept=kept, alt_matched=new,
                         stable=bool(abs(dn) < 0.15 and kept >= 0.9 and new >= 0.9)))
# other detectors: main threshold +-25 %
for d in NAMES:
    thr = FZ[d]['thr']
    r0, r1, r2 = [], [], []
    for m, sp in spans:
        P = prep(sp['y'])
        r0.append(in_eval(run_setting(d, thr, sp['t'], P), m['e0'], m['e1']))
        r1.append(in_eval(run_setting(d, thr * 0.75, sp['t'], P), m['e0'], m['e1']))
        r2.append(in_eval(run_setting(d, thr * 1.25, sp['t'], P), m['e0'], m['e1']))
    for lab, alt in (('thr x0.75', r1), ('thr x1.25', r2)):
        n0, n1, dn, kept, new = compare(r0, alt)
        rows.append(dict(det=d, param=lab, n_ref=n0, n_alt=n1, count_change=dn, ref_kept=kept, alt_matched=new,
                         stable=bool(abs(dn) < 0.15 and kept >= 0.9 and new >= 0.9)))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'tab_stability.csv'), index=False)
pd.set_option('display.width', 200)
print(df.round(3).to_string())

# ---- clean tier (post-filter SNR >= 30, data/postfilter_rule.json): stability to k_on, hp_s, pulse_med_s and the SNR cut
RULE = json.load(open(os.path.join(os.path.dirname(__file__), '..', 'data', 'postfilter_rule.json')))


def clean_for(sp, m, p, smin):
    P = prep(sp['y'], p)
    zh, sh = D.highpass_channel(P['zd'], p['hp_s'])
    segs = in_eval(D.detect_hyst(sp['t'], zh, sh, p), m['e0'], m['e1'])
    keep = [s for s in segs if np.abs(P['z'][s[0]:s[1]]).max() / np.median(P['sigma'][s[0]:s[1]]) >= smin]
    return np.array(keep, int).reshape(-1, 2)


s0 = RULE['snr_min']
refc = [clean_for(sp, m, REF, s0) for m, sp in spans]
crow = []
for lab, p, smin in ([(f'k_on x{f}', dict(REF, k_on=REF['k_on'] * f), s0) for f in (0.75, 1.25)] +
                     [(f'hp_s x{f}', dict(REF, hp_s=REF['hp_s'] * f), s0) for f in (0.75, 1.25)] +
                     [(f'pulse_med_s x{f}', dict(REF, pulse_med_s=REF['pulse_med_s'] * f), s0) for f in (0.75, 1.25)] +
                     [(f'snr_min x{f}', dict(REF), s0 * f) for f in (0.75, 1.25)]):
    alt = [clean_for(sp, m, p, smin) for m, sp in spans]
    n0, n1, dn, kept, new = compare(refc, alt)
    crow.append(dict(det='HYSTBP+SNR>=30', param=lab, n_ref=n0, n_alt=n1, count_change=dn, ref_kept=kept, alt_matched=new,
                     stable=bool(abs(dn) < 0.15 and kept >= 0.9 and new >= 0.9)))
cdf = pd.DataFrame(crow)
pd.concat([df, cdf]).to_csv(os.path.join(os.path.dirname(__file__), '..', 'data', 'tab_stability.csv'), index=False)
print(cdf.round(3).to_string())
