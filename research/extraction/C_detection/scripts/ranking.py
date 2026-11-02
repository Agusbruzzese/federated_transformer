"""Audit issue 2: does the detector ranking survive injection templates NOT selected by HYST?
Template sets: hyst (data/templates.npz), bdet (B's detector @8 sigma), wavelet (WAVELET @ frozen threshold), and for
bdet/wavelet also the subset of templates NOT shared with the HYST set (events >1 s from every HYST template peak).
All detectors at their frozen operating points (data/frozen_params.json). F1 over isolated injections (all SNR>=2, i.e.
>= 6 dB), FP = fragments + new unmatched detections. Cluster bootstrap (span x realisation), 200 resamples, seed 0.
-> data/tab_ranking_templates.csv"""
import os, json
import numpy as np, pandas as pd
HERE = os.path.join(os.path.dirname(__file__), '..')
FZ = json.load(open(f'{HERE}/data/frozen_params.json'))
DETS = ['HYSTBP', 'CUSUMBP', 'WAVELET', 'BDET', 'BDET@legacy', 'FIXEDLOC', 'STALTABP', 'FIXED@legacy']
L = {k: np.load(f'{HERE}/data/{f}') for k, f in [('hyst', 'templates.npz'), ('bdet', 'templates_bdet.npz'), ('wavelet', 'templates_wavelet.npz')]}


def not_shared(tset):
    A, B = L[tset], L['hyst']
    return {i for i, (s, p) in enumerate(zip(A['span'], A['peak_t'])) if not any((B['span'] == s) & (np.abs(B['peak_t'] - p) < 1))}


def load(split, tset):
    suf = '' if tset == 'hyst' else f'_{tset}'
    return pd.read_parquet(f'{HERE}/data/inj_{split}{suf}.parquet'), pd.read_parquet(f'{HERE}/data/det_{split}{suf}.parquet')


def pick(df, d, split, tset):
    base = d.split('@')[0]
    thr = FZ[base]['legacy_thr'] if '@' in d else FZ[base]['thr']
    key = base if (split == 'tune' and tset == 'hyst') else d
    return df[(df.det == key) & np.isclose(df.thr, thr) & (df.kind == 'iso')]


rng = np.random.default_rng(0)
rows = []
for split in ('tune', 'holdout'):
    for tset in ('hyst', 'bdet', 'wavelet'):
        inj, det = load(split, tset)
        subsets = [('all', None)] + ([('not_in_hyst_set', not_shared(tset))] if tset != 'hyst' else [])
        for sub, keep in subsets:
            aggs = {}
            for d in DETS:
                I = pick(inj, d, split, tset).copy(); Dt = pick(det, d, split, tset).copy()
                I['cl'] = I.span + '_' + I.rep.astype(str); Dt['cl'] = Dt.span + '_' + Dt.rep.astype(str)
                fp = Dt.groupby('cl').apply(lambda x: (x.n_frag + x.n_fp_new).sum(), include_groups=False)
                frac = 1.0
                if keep is not None:
                    n_all = I.groupby('cl').size()
                    I = I[I.tid.isin(keep)]
                    frac = (I.groupby('cl').size() / n_all).reindex(n_all.index).fillna(0)   # FP pro-rated to subset
                a = I.groupby('cl').agg(tp=('matched', 'sum'), n=('matched', 'size'),
                                        tp8=('matched', lambda m: m[I.loc[m.index, 'snr'] == 8].sum()),
                                        n8=('matched', lambda m: (I.loc[m.index, 'snr'] == 8).sum()))
                a['fp'] = (fp * frac).reindex(a.index).fillna(0)
                aggs[d] = (a, I)
            cls = aggs['HYSTBP'][0].index.values
            def f1(a):
                tp, n, fp = a.tp.sum(), a.n.sum(), a.fp.sum(); p = tp / max(tp + fp, 1e-9); r = tp / max(n, 1)
                return 2 * p * r / max(p + r, 1e-9)
            boots = [rng.choice(cls, len(cls), replace=True) for _ in range(200)]
            for d, (a, I) in aggs.items():
                bs = [f1(a.reindex(c).fillna(0)) for c in boots]
                dl = [f1(aggs['HYSTBP'][0].reindex(c).fillna(0)) - f1(a.reindex(c).fillna(0)) for c in boots]
                hi = I[I.matched & (I.snr >= 10)]
                rows.append(dict(split=split, templates=tset, subset=sub, n_templates=len(L[tset]['T']) if keep is None else len(keep),
                                 det=d, n_inj=int(a.n.sum()), F1=f1(a), F1_lo=np.percentile(bs, 2.5), F1_hi=np.percentile(bs, 97.5),
                                 dF1_HYSTBP_minus=f1(aggs['HYSTBP'][0]) - f1(a), d_lo=np.percentile(dl, 2.5), d_hi=np.percentile(dl, 97.5),
                                 recall_snr8=a.tp8.sum() / max(a.n8.sum(), 1), split_rate=(hi.n_frag > 0).mean() if len(hi) else np.nan))
out = pd.DataFrame(rows)
out['rank'] = out.groupby(['split', 'templates', 'subset']).F1.rank(ascending=False).astype(int)
out.to_csv(f'{HERE}/data/tab_ranking_templates.csv', index=False)
pd.set_option('display.width', 250)
for (sp, t, sub), g in out.groupby(['split', 'templates', 'subset'], sort=False):
    g = g.sort_values('rank')
    print(f'{sp:8s} {t:8s} {sub:16s} n_tpl={g.n_templates.iloc[0]:3d} | ' + ' > '.join(f'{r.det} {r.F1:.3f}' for r in g.itertuples()))
print(out.round(3).to_string())
