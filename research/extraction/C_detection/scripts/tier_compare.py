"""Clean-tier comparison: HYST-BP + SNR>=30 post-filter vs workstream B's detector (BDET, native 8 sigma and matched 6 sigma).
  tier_compare.py make   -> 30 random BDET@8 holdout detections (seed 3), blind sheets figs/hand_pf/bdet_XX.png,
                            candidates data/bdet_cand_hold.csv (label into data/bdet_labels_hold.csv)
  tier_compare.py score  -> precision (Wilson) of BDET@8; relative recall of every tier on the V-labelled holdout candidates
                            of the main blind hand-check; precision with and without detections < 0.4 s (B's request).
Outputs data/tab_tier_compare.csv, data/tab_hand_precision_minlen.csv."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, os.path.dirname(__file__))
from common import *
import pandas as pd
from inject_eval import in_eval, run_setting
from handcheck import wilson, SETS
from postfilter import sheets
HERE = os.path.join(os.path.dirname(__file__), '..')
FS = D.FS
RULE = json.load(open(f'{HERE}/data/postfilter_rule.json'))


def tiers():
    out = {}
    for m in META:
        if m['split'] != 'holdout': continue
        sp = load_span(m['name']); P = prep(sp['y'])
        hb = in_eval(run_setting('HYSTBP', 5.0, sp['t'], P), m['e0'], m['e1'])
        snr = np.array([np.abs(P['z'][a:b]).max() / np.median(P['sigma'][a:b]) for a, b in hb]) if len(hb) else np.zeros(0)
        out[m['name']] = {'HYSTBP': hb, 'HYSTBP+SNR>=30': hb[snr >= RULE['snr_min']] if len(hb) else hb,
                          'BDET@8': in_eval(run_setting('BDET', 8.0, sp['t'], P), m['e0'], m['e1']),
                          'BDET@6': in_eval(run_setting('BDET', 6.0, sp['t'], P), m['e0'], m['e1'])}
    return out


if __name__ == '__main__':
    if sys.argv[1] == 'make':
        T = tiers()
        pool = [(nm, int(a), int(b)) for nm, d in T.items() for a, b in d['BDET@8']]
        rng = np.random.default_rng(3)
        pick = [pool[i] for i in rng.choice(len(pool), min(30, len(pool)), replace=False)]
        df = pd.DataFrame(pick, columns=['span', 'a', 'b']); df['dur'] = (df.b - df.a) / FS; df['id'] = np.arange(len(df))
        df.to_csv(f'{HERE}/data/bdet_cand_hold.csv', index=False); sheets(df, 'bdet'); print(len(pool), 'BDET@8 holdout detections')
    else:
        T = tiers()
        rows = []
        d = pd.read_csv(f'{HERE}/data/bdet_cand_hold.csv').merge(pd.read_csv(f'{HERE}/data/bdet_labels_hold.csv'), on='id')
        n = len(d); p, lo, hi = wilson((d.label == 'V').sum(), n); pc, clo, chi = wilson(d.label.isin(['V', 'M', 'F']).sum(), n)
        rows.append(dict(tier='BDET@8', n_labelled=n, prec_strict=p, lo=lo, hi=hi, prec_vehicle=pc, pv_lo=clo, pv_hi=chi))
        pf = pd.read_csv(f'{HERE}/data/tab_postfilter.csv')
        r = pf[(pf.split == 'hold') & (pf.group == 'pass')].iloc[0]
        rows.append(dict(tier='HYSTBP+SNR>=30', n_labelled=r.n, prec_strict=r.prec_strict, lo=r.lo, hi=r.hi,
                         prec_vehicle=r.prec_vehicle, pv_lo=r.pv_lo, pv_hi=r.pv_hi))
        # relative recall on V-labelled holdout candidates (main hand-check)
        c = pd.read_csv(f'{HERE}/data/hand_candidates.csv').merge(pd.read_csv(f'{HERE}/data/hand_labels.csv'), on='id')
        v = c[(c.split == 'holdout') & (c.label == 'V')]
        rr = {}
        for tier in ['HYSTBP', 'HYSTBP+SNR>=30', 'BDET@8', 'BDET@6']:
            hit = [bool(len(T[x.span][tier]) and np.any((T[x.span][tier][:, 0] < x.b) & (x.a < T[x.span][tier][:, 1]))) for x in v.itertuples()]
            rr[tier] = wilson(sum(hit), len(hit))
        n_h = sum(1 for m in META if m['split'] == 'holdout') * 20 / 60
        for row in rows:
            row.update(rel_recall=rr[row['tier']][0], rr_lo=rr[row['tier']][1], rr_hi=rr[row['tier']][2])
        for tier in ['HYSTBP', 'BDET@6']:
            rows.append(dict(tier=tier, rel_recall=rr[tier][0], rr_lo=rr[tier][1], rr_hi=rr[tier][2]))
        for row in rows:
            row['events_per_h_holdout'] = sum(len(T[s][row['tier']]) for s in T) / n_h
        out = pd.DataFrame(rows); out.to_csv(f'{HERE}/data/tab_tier_compare.csv', index=False)
        pd.set_option('display.width', 250); print(out.round(3).to_string())
        # precision with / without < 0.4 s detections (main hand-check)
        c['dur'] = (c.b - c.a) / FS
        rows = []
        for split in ('holdout', 'all'):
            g0 = c if split == 'all' else c[c.split == split]
            for dname, _ in SETS:
                for lab, g in (('all', g0), ('dur>=0.4s', g0[g0.dur >= 0.4])):
                    h = g[g.sampled_by.str.split('|').apply(lambda x: dname in x)]
                    p, lo, hi = wilson((h.label == 'V').sum(), len(h))
                    rows.append(dict(split=split, det=dname, subset=lab, n=len(h), prec_strict=p, lo=lo, hi=hi,
                                     n_short=int((g0[g0.sampled_by.str.split('|').apply(lambda x: dname in x)].dur < 0.4).sum())))
        out2 = pd.DataFrame(rows); out2.to_csv(f'{HERE}/data/tab_hand_precision_minlen.csv', index=False)
        print(out2[out2.split == 'holdout'].round(3).to_string())
