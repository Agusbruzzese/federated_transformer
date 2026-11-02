"""Blind hand-verification set (per-detection design; the first union-based design was abandoned because the
FIXED/CUSUM segments are minutes long and the union merged many vehicles into single candidates).
  handcheck.py make   -> data/hand_candidates.csv + figs/hand/sheet_XX.png (20 candidates / sheet, detector identity hidden)
  handcheck.py score  -> reads data/hand_labels.csv (id,label), writes data/tab_hand_precision.csv
Candidates: per detector, a random sample (seed 0) of its detections at the frozen operating point -- 25 per detector
from the holdout spans (Run 3) and 8 per detector from the tuning spans. Segments from different detectors with
IoU > 0.8 are merged into one candidate (labelled once, credited to all). Order shuffled (seed 0).
Labels: V = one vehicle-like transient, sensibly segmented; M = merged/over-long (>=2 distinct transients or segment
> ~3x the transient); F = fragment (cuts the transient core, < ~half inside); N = no vehicle-like transient
(wander, pulse train, step, noise); U = uncertain."""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, os.path.dirname(__file__))
from common import *
import pandas as pd
from inject_eval import in_eval, run_setting
HERE = os.path.join(os.path.dirname(__file__), '..')
FZ = json.load(open(f'{HERE}/data/frozen_params.json'))
# pinned to the detector list at labelling time (BDET was added later and has its own sample in postfilter.py)
HC_NAMES = ['FIXED', 'FIXEDLOC', 'HYST', 'STALTA', 'CUSUM', 'WAVELET', 'HYSTBP', 'STALTABP', 'CUSUMBP']
SETS = [(d, FZ[d]['thr']) for d in HC_NAMES] + [('FIXED@legacy', FZ['FIXED']['legacy_thr'])]
FS = D.FS


def union(segs_by_det, n):
    cov = np.zeros(n, bool)
    for s in segs_by_det.values():
        for a, b in s:
            cov[a:b] = True
    return D.runs_of(cov)


def make():
    rng = np.random.default_rng(0)
    cands = []   # dict(span, split, a, b, dets=set)
    alldet = {}
    for m in META:
        sp = load_span(m['name']); P = prep(sp['y'])
        sd = {d: in_eval(run_setting(d, thr, sp['t'], P), m['e0'], m['e1']) for d, thr in SETS}
        alldet[m['name']] = sd
    for split, K in (('holdout', 25), ('tune', 8)):
        for d, _ in SETS:
            pool = [(m['name'], a, b) for m in META if m['split'] == split for a, b in alldet[m['name']][d]]
            pick = rng.choice(len(pool), min(K, len(pool)), replace=False) if pool else []
            for i in pick:
                nm, a, b = pool[i]
                hit = None
                for c in cands:
                    if c['span'] == nm:
                        inter = max(0, min(b, c['b']) - max(a, c['a'])); uni = max(b, c['b']) - min(a, c['a'])
                        if inter / uni > 0.8:
                            hit = c; break
                if hit is None:
                    cands.append(dict(span=nm, split=split, a=int(a), b=int(b), sampled_by={d}))
                else:
                    hit['sampled_by'].add(d)
    rows = []
    for c in cands:
        sd = alldet[c['span']]
        # which detectors produce a near-identical segment (IoU > 0.8): the candidate's precision is credited to them
        same = {}
        for d, _ in SETS:
            s_ = sd[d]
            ok = False
            for a, b in s_:
                inter = max(0, min(b, c['b']) - max(a, c['a'])); uni = max(b, c['b']) - min(a, c['a'])
                if inter / uni > 0.8:
                    ok = True; break
            same[d] = ok
        rows.append(dict(span=c['span'], split=c['split'], a=c['a'], b=c['b'],
                         sampled_by='|'.join(sorted(c['sampled_by'])), **same))
    df = pd.DataFrame(rows).sample(frac=1, random_state=0).reset_index(drop=True)
    df['id'] = np.arange(len(df))
    df.to_csv(f'{HERE}/data/hand_candidates.csv', index=False)
    import pickle
    pickle.dump(alldet, open(f'{HERE}/data/hand_alldet.pkl', 'wb'))
    print('candidates', len(df), df.split.value_counts().to_dict())
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    os.makedirs(f'{HERE}/figs/hand', exist_ok=True)
    cache = {}
    for sh in range(int(np.ceil(len(df) / 20))):
        fig, ax = plt.subplots(5, 4, figsize=(20, 15)); ax = ax.ravel()
        for k, r in enumerate(df.iloc[sh * 20:(sh + 1) * 20].itertuples()):
            if r.span not in cache:
                sp = load_span(r.span); cache[r.span] = (sp['t'], prep(sp['y']))
            t, P = cache[r.span]
            c0, c1 = max(r.a - int(15 * FS), 0), min(r.b + int(15 * FS), len(t))
            tt = t[c0:c1] - t[r.a]
            ax[k].plot(tt, P['z'][c0:c1], color='0.65', lw=.5)
            ax[k].plot(tt, P['zd'][c0:c1], color='k', lw=.7)
            ax[k].axvspan(0, t[r.b - 1] - t[r.a], color='tab:orange', alpha=.25)
            s = np.median(P['sigma'][c0:c1])
            lo, hi = np.percentile(P['zd'][c0:c1], [0.5, 99.5])
            pad = max(hi - lo, 12 * s) * 0.15
            ax[k].set_ylim(min(lo, -6 * s) - pad, max(hi, 6 * s) + pad)
            for q in (-5, 5):
                ax[k].axhline(q * s, color='tab:blue', lw=.5, ls=':')
            ax[k].set_title(f'#{r.id}  dur {(r.b - r.a) / FS:.1f}s  peak {np.abs(P["z"][r.a:r.b]).max() / s:.0f} sigma', fontsize=9)
            ax[k].tick_params(labelsize=7)
        for a in ax[k + 1:]:
            a.axis('off')
        plt.tight_layout(); plt.savefig(f'{HERE}/figs/hand/sheet_{sh:02d}.png', dpi=55); plt.close()


def wilson(k, n, z=1.96):
    if n == 0:
        return np.nan, np.nan, np.nan
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return p, c - h, c + h


def score():
    df = pd.read_csv(f'{HERE}/data/hand_candidates.csv')
    lab = pd.read_csv(f'{HERE}/data/hand_labels.csv')
    df = df.merge(lab, on='id', how='inner')
    rows = []
    for split in ('holdout', 'tune', 'all'):
        g = df if split == 'all' else df[df.split == split]
        for d, _ in SETS:
            h = g[g.sampled_by.str.split('|').apply(lambda x: d in x)]
            n = len(h)
            kv = (h.label == 'V').sum(); ku = (h.label == 'U').sum()
            kc = h.label.isin(['V', 'M', 'F']).sum()
            p, lo, hi = wilson(kv, n); po, loo, hio = wilson(kv + ku, n); pc, loc, hic = wilson(kc + ku, n)
            rows.append(dict(split=split, det=d, n=n, V=kv, M=(h.label == 'M').sum(), F=(h.label == 'F').sum(),
                             N=(h.label == 'N').sum(), U=ku, prec_strict=p, prec_strict_lo=lo, prec_strict_hi=hi,
                             prec_strict_optimistic=po, prec_contains_vehicle=pc, prec_contains_lo=loc, prec_contains_hi=hic))
    out = pd.DataFrame(rows)
    out.to_csv(f'{HERE}/data/tab_hand_precision.csv', index=False)
    # relative recall: of V-labelled candidates (any detector's sample), fraction overlapped by each detector
    import pickle
    alldet = pickle.load(open(f'{HERE}/data/hand_alldet.pkl', 'rb'))
    rr = []
    for split in ('holdout', 'tune'):
        v = df[(df.split == split) & (df.label == 'V')]
        for d, _ in SETS:
            hit = [bool(len(alldet[r.span][d]) and np.any((alldet[r.span][d][:, 0] < r.b) & (r.a < alldet[r.span][d][:, 1])))
                   for r in v.itertuples()]
            p, lo, hi = wilson(sum(hit), len(hit))
            rr.append(dict(split=split, det=d, n_V=len(hit), rel_recall=p, lo=lo, hi=hi))
    rr = pd.DataFrame(rr); rr.to_csv(f'{HERE}/data/tab_hand_relrecall.csv', index=False)
    pd.set_option('display.width', 250)
    print(out.round(3).to_string()); print(rr.round(3).to_string())
    print(df.groupby(['split', 'label']).size())


if __name__ == '__main__':
    make() if sys.argv[1] == 'make' else score()
