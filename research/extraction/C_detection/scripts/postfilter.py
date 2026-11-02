"""Post-filter for HYST-BP false positives found by the blind hand-check (narrow sub-second dips / wander at low SNR).
Stage 1 (tuning only):  postfilter.py make_tune   -> 100 random HYST-BP detections from TUNING spans (seed 1) +
                                                    features; sheets figs/hand_pf/tune_XX.png; label data/pf_labels_tune.csv
                         postfilter.py fit         -> choose rule on tuning labels only; writes data/postfilter_rule.json
Stage 2 (holdout):       postfilter.py make_hold   -> 60 random HYST-BP detections from HOLDOUT spans (seed 2) that pass the
                                                    frozen rule + 20 that fail it; sheets figs/hand_pf/hold_XX.png;
                                                    label data/pf_labels_hold.csv (labelled blind: pass/fail not shown)
                         postfilter.py score       -> precision with Wilson CI; data/tab_postfilter.csv
Features (all from the detection segment on the waveform/detection channels): snr (peak |z|/sigma), dur (segment s),
width (s, full width of |zd| above half its max, i.e. the width of the dominant lobe)."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, os.path.dirname(__file__))
from common import *
import pandas as pd
from inject_eval import in_eval, run_setting
from handcheck import wilson
HERE = os.path.join(os.path.dirname(__file__), '..')
FS = D.FS


def features(P, a, b):
    z, zd, s = P['z'][a:b], P['zd'][a:b], P['sigma'][a:b]
    j = int(np.argmax(np.abs(zd)))
    half = np.abs(zd) >= 0.5 * np.abs(zd[j])
    # contiguous run around the peak
    l = j
    while l > 0 and half[l - 1]: l -= 1
    r = j
    while r < len(zd) - 1 and half[r + 1]: r += 1
    return dict(snr=float(np.abs(z).max() / np.median(s)), dur=(b - a) / FS, width=(r - l + 1) / FS)


def detections(split):
    rows = []
    for m in META:
        if m['split'] != split: continue
        sp = load_span(m['name']); P = prep(sp['y'])
        for a, b in in_eval(run_setting('HYSTBP', 5.0, sp['t'], P), m['e0'], m['e1']):
            rows.append(dict(span=m['name'], a=int(a), b=int(b), **features(P, a, b)))
    return pd.DataFrame(rows)


def sheets(df, prefix):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    os.makedirs(f'{HERE}/figs/hand_pf', exist_ok=True)
    cache = {}
    for sh in range(int(np.ceil(len(df) / 20))):
        fig, ax = plt.subplots(5, 4, figsize=(20, 15)); ax = ax.ravel()
        for k, r in enumerate(df.iloc[sh * 20:(sh + 1) * 20].itertuples()):
            if r.span not in cache:
                sp = load_span(r.span); cache[r.span] = (sp['t'], prep(sp['y']))
            t, P = cache[r.span]
            c0, c1 = max(r.a - int(15 * FS), 0), min(r.b + int(15 * FS), len(t))
            tt = t[c0:c1] - t[r.a]
            ax[k].plot(tt, P['z'][c0:c1], color='0.65', lw=.5); ax[k].plot(tt, P['zd'][c0:c1], color='k', lw=.7)
            ax[k].axvspan(0, t[r.b - 1] - t[r.a], color='tab:orange', alpha=.25)
            s = np.median(P['sigma'][c0:c1]); lo, hi = np.percentile(P['zd'][c0:c1], [0.5, 99.5]); pad = max(hi - lo, 12 * s) * .15
            ax[k].set_ylim(min(lo, -6 * s) - pad, max(hi, 6 * s) + pad)
            ax[k].set_title(f'#{r.id}  dur {r.dur:.1f}s', fontsize=9); ax[k].tick_params(labelsize=7)
        for a in ax[k + 1:]: a.axis('off')
        plt.tight_layout(); plt.savefig(f'{HERE}/figs/hand_pf/{prefix}_{sh:02d}.png', dpi=55); plt.close()


def rule_pass(df, rule):
    return (df.snr >= rule['snr_min']) & (df.width >= rule['width_min'])


if __name__ == '__main__':
    cmd = sys.argv[1]
    if cmd == 'make_tune':
        d = detections('tune').sample(n=100, random_state=1).reset_index(drop=True); d['id'] = np.arange(len(d))
        d.to_csv(f'{HERE}/data/pf_cand_tune.csv', index=False); sheets(d, 'tune'); print(len(d))
    elif cmd == 'fit':
        d = pd.read_csv(f'{HERE}/data/pf_cand_tune.csv').merge(pd.read_csv(f'{HERE}/data/pf_labels_tune.csv'), on='id')
        d['good'] = d.label == 'V'; d['veh'] = d.label.isin(['V', 'M', 'F'])
        print(d.groupby('label')[['snr', 'dur', 'width']].describe().T.round(2).to_string())
        best = None; res = []
        for smin in [5, 6, 8, 10, 12, 15, 20, 30]:
            for wmin in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8]:
                k = (d.snr >= smin) & (d.width >= wmin)
                if k.sum() < 10: continue
                prec = d[k].good.mean(); precv = d[k].veh.mean(); keep_v = d[k].good.sum() / max(d.good.sum(), 1)
                res.append(dict(snr_min=smin, width_min=wmin, n_pass=int(k.sum()), prec_strict=prec, prec_vehicle=precv, recall_of_V=keep_v))
        res = pd.DataFrame(res); res.to_csv(f'{HERE}/data/tab_postfilter_grid_tune.csv', index=False)
        # pre-registered constraint: strict precision >= 0.90 (lower Wilson bound not required on n~100); maximise retained V
        ok = res[res.prec_strict >= 0.90].sort_values(['recall_of_V', 'snr_min'], ascending=[False, True])
        pick = (ok.iloc[0] if len(ok) else res.sort_values('prec_strict', ascending=False).iloc[0]).to_dict()
        print(res.round(3).to_string()); print('PICK', pick)
        json.dump(dict(snr_min=float(pick['snr_min']), width_min=float(pick['width_min']), tune_stats=pick),
                  open(f'{HERE}/data/postfilter_rule.json', 'w'), indent=1)
    elif cmd == 'make_hold':
        rule = json.load(open(f'{HERE}/data/postfilter_rule.json'))
        d = detections('holdout'); ps = rule_pass(d, rule)
        a = d[ps].sample(n=min(60, ps.sum()), random_state=2); b = d[~ps].sample(n=min(20, (~ps).sum()), random_state=2)
        d = pd.concat([a.assign(passed=True), b.assign(passed=False)]).sample(frac=1, random_state=2).reset_index(drop=True)
        d['id'] = np.arange(len(d)); d.to_csv(f'{HERE}/data/pf_cand_hold.csv', index=False); sheets(d, 'hold')
        print(len(d), 'holdout detections total', len(ps), 'pass frac', ps.mean())
    elif cmd == 'score':
        rule = json.load(open(f'{HERE}/data/postfilter_rule.json'))
        rows = []
        for split in ('tune', 'hold'):
            d = pd.read_csv(f'{HERE}/data/pf_cand_{split}.csv').merge(pd.read_csv(f'{HERE}/data/pf_labels_{split}.csv'), on='id')
            if 'passed' not in d: d['passed'] = rule_pass(d, rule)
            for grp, g in (('all', d), ('pass', d[d.passed]), ('fail', d[~d.passed])):
                n = len(g); kv = (g.label == 'V').sum(); ku = (g.label == 'U').sum(); kc = g.label.isin(['V', 'M', 'F']).sum()
                p, lo, hi = wilson(kv, n); pc, clo, chi = wilson(kc, n)
                rows.append(dict(split=split, group=grp, n=n, V=kv, M=(g.label == 'M').sum(), F=(g.label == 'F').sum(),
                                 N=(g.label == 'N').sum(), U=ku, prec_strict=p, lo=lo, hi=hi, prec_vehicle=pc, pv_lo=clo, pv_hi=chi))
        out = pd.DataFrame(rows); out.to_csv(f'{HERE}/data/tab_postfilter.csv', index=False)
        pd.set_option('display.width', 200); print(out.round(3).to_string())
