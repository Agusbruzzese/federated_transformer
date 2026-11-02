"""Synthetic-injection + surrogate false-alarm evaluation.

usage: inject_eval.py tune            -> threshold sweep on tuning spans (Runs 1,2,4)
       inject_eval.py holdout FROZEN  -> frozen operating points (data/frozen_params.json) on Run-3 spans
Outputs data/inj_<split>.parquet (per injection), data/det_<split>.parquet (per realisation counts),
        data/sur_<split>.parquet (surrogate false alarms), data/cost_<split>.parquet.
Seeds: realisation seed = 1000*span_index + rep (numpy default_rng); fully deterministic.
"""
import sys, os, json, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from common import *
import pandas as pd
from multiprocessing import Pool

FS = D.FS
SNRS = [2, 3, 4, 5, 6, 8, 10, 15, 20, 40]
STRETCH = [1.0, 1.5, 2.5]
PAIR_SEP = [1.0, 2.0, 3.0, 5.0, 8.0]
R_ISO, R_PAIR, R_SUR = 8, 4, 6
GRID = {"FIXED": [0.5, 0.75, 1, 1.5, 2, 3, 5, 8, 15],
        "FIXEDLOC": [3, 4, 5, 6, 7, 8, 10, 12, 15],
        "STALTABP": [4, 6, 8, 10, 15, 20, 30, 50, 80, 120],
        "BDET": [5, 6, 7, 8, 10, 12, 15, 20, 30],
        "CUSUMBP": [25, 50, 100, 200, 400, 800, 1600, 3200, 6400],
        "HYST": [3, 3.5, 4, 4.5, 5, 6, 7, 8, 10],
        "STALTA": [4, 6, 8, 10, 15, 20, 30, 50, 80, 120],
        "CUSUM": [100, 200, 400, 800, 1200, 2000, 3000, 5000, 8000],
        "WAVELET": [3, 3.5, 4, 4.5, 5, 6, 7, 8, 10],
        "HYSTBP": [3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8, 10]}
TSET = os.environ.get('TSET', '')   # '' = HYST-selected templates; 'bdet' / 'wavelet' = independent sets (audit issue 2)
_t = np.load(os.path.join(os.path.dirname(__file__), '..', 'data', 'templates.npz' if not TSET else f'templates_{TSET}.npz')); TPL = {k: _t[k] for k in _t.files}; _t.close()
H = (TPL['T'].shape[1] - 1) // 2


def stretched(tid, s):
    w = TPL['T'][tid]
    x = np.arange(-H, H + 1)
    n = int(np.ceil(H * s))
    xs = np.arange(-n, n + 1)
    return np.interp(xs / s, x, w, left=0, right=0), n, np.array([TPL['gt5'][tid], TPL['gt95'][tid], TPL['gt1'][tid], TPL['gt99'][tid]]) * s


def settings(split):
    if split == 'tune' and not TSET:
        return [(d, thr) for d in NAMES for thr in GRID[d]]
    fz = json.load(open(os.path.join(os.path.dirname(__file__), '..', 'data', 'frozen_params.json')))
    return [(d, fz[d]['thr']) for d in NAMES] + [(d + '@legacy', fz[d]['legacy_thr']) for d in NAMES if 'legacy_thr' in fz[d]]


def run_setting(name, thr, t, P):
    d = name.split('@')[0]
    p = {A.THRESH[d]: thr}
    if d in ('HYST', 'WAVELET', 'HYSTBP'):  # keep k_off < k_on
        p['k_off'] = min(2.0, 0.5 * thr)
    return A.run(d, t, P['y'], P['zd'], P['sigma'], p)


def in_eval(segs, e0, e1):
    if len(segs) == 0:
        return segs
    mid = (segs[:, 0] + segs[:, 1]) // 2
    return segs[(mid >= e0) & (mid < e1)]


def overlaps(seg, iv):
    return seg[0] < iv[1] and iv[0] < seg[1]


def match(segs, inj):
    """inj: list of dicts with pk, ext(1-99%) sample indices. Returns per-injection matched seg index (or -1),
    and per-seg label: 'tp' / 'frag' / 'other'."""
    used = set()
    m = [-1] * len(inj)
    order = np.argsort([-j['snr'] for j in inj])
    for k in order:
        j = inj[k]
        cand = [i for i, (a, b) in enumerate(segs) if a <= j['pk'] < b and i not in used]
        if cand:
            i = min(cand, key=lambda i: abs((segs[i][0] + segs[i][1]) / 2 - j['pk']))
            m[k] = i
            used.add(i)
    lab = []
    for i, s in enumerate(segs):
        if i in used:
            lab.append('tp')
        elif any(overlaps(s, j['ext']) for j in inj):
            lab.append('frag')
        else:
            lab.append('other')
    return m, lab


def free_slots(t, P, e0, e1):
    segs = D.detect_hyst(t, P['zd'], P['sigma'], dict(D.DEFAULTS, k_on=4.0))
    bad = np.zeros(len(t), bool)
    for a, b in segs:
        bad[max(a - int(10 * FS), 0):b + int(10 * FS)] = True
    ok = np.zeros(len(t), bool)
    ok[e0 + int(20 * FS):e1 - int(20 * FS)] = True
    return np.where(ok & ~bad)[0]


def place(rng, slots, n, spacing):
    rng.shuffle(slots)
    out = []
    for s in slots:
        if all(abs(s - o) >= spacing for o in out):
            out.append(int(s))
        if len(out) >= n:
            break
    return sorted(out)


def inject(y, sigma, specs):
    yi = y.copy()
    inj = []
    for sp in specs:
        w, n, gt = stretched(sp['tid'], sp['stretch'])
        amp = sp['snr'] * sigma[sp['pk']]
        a, b = sp['pk'] - n, sp['pk'] + n + 1
        if a < 0 or b > len(y):
            continue
        yi[a:b] += amp * w
        e = np.zeros(len(y)); e[a:b] = (amp * w) ** 2
        inj.append(dict(sp, gt=sp['pk'] + gt * FS, ext=(sp['pk'] + gt[2] * FS, sp['pk'] + gt[3] * FS), energy=e))
    return yi, inj


def surrogate(P, rng):
    """Null model: stationary Gaussian noise with the span's ROBUST noise PSD (median-averaged Welch over 81 s
    segments -- insensitive to sparse vehicle transients, which otherwise leak their low-frequency power into
    an FFT-phase-randomised surrogate), level-matched to the robust sigma of the un-masked residual, added to
    the 300 s-smoothed baseline (slow drift only)."""
    from scipy.signal import welch
    from scipy.ndimage import uniform_filter1d
    n = len(P['z'])
    f, S = welch(P['z'], fs=FS, nperseg=4096, noverlap=2048, average='median')
    fr = np.fft.rfftfreq(n, 1 / FS)
    amp = np.sqrt(np.interp(fr, f, S))
    Zs = amp * (rng.standard_normal(len(fr)) + 1j * rng.standard_normal(len(fr)))
    Zs[0] = 0
    zs = np.fft.irfft(Zs, n)
    q = ~P['mask']
    sc = (1.4826 * np.median(np.abs(P['z'][q] - np.median(P['z'][q])))) / (1.4826 * np.median(np.abs(zs - np.median(zs))))
    return uniform_filter1d(P['base'], int(300 * FS), mode='nearest') + zs * sc


def work(args):
    si, name, split = args
    m = [k for k in META if k['name'] == name][0]
    s = load_span(name)
    t, y = s['t'], s['y']
    e0, e1 = m['e0'], m['e1']
    hours = (e1 - e0) / FS / 3600
    P0 = prep(y)
    sets = settings(split)
    inj_rows, det_rows, sur_rows, cost_rows = [], [], [], []
    clean = {}
    for (d, thr) in sets:
        T0 = time.process_time()
        clean[(d, thr)] = in_eval(run_setting(d, thr, t, P0), e0, e1)
        cost_rows.append(dict(span=name, det=d, thr=thr, cpu_s=time.process_time() - T0, hours=(len(t) / FS) / 3600))
        det_rows.append(dict(span=name, rep=-1, kind='clean', det=d, thr=thr, n_det=len(clean[(d, thr)]), hours=hours))
    slots = free_slots(t, P0, e0, e1)
    # ---------------- injections
    for kind, R in (('iso', R_ISO), ('pair', R_PAIR)):
        for rep in range(R):
            rng = np.random.default_rng(1000 * si + rep + (500 if kind == 'pair' else 0))
            pk = place(rng, slots.copy(), 16 if kind == 'iso' else 10, int(35 * FS))
            specs = []
            for k, p in enumerate(pk):
                if kind == 'iso':
                    specs.append(dict(pk=p, snr=SNRS[(k + rep) % len(SNRS)], tid=int(rng.integers(len(TPL['T']))),
                                      stretch=float(rng.choice(STRETCH)), pair=-1, sep=np.nan))
                else:
                    sep = PAIR_SEP[(k + rep) % len(PAIR_SEP)]
                    t1, t2 = rng.choice(len(TPL['T']), 2, replace=False)
                    specs.append(dict(pk=p, snr=20, tid=int(t1), stretch=1.0, pair=k, sep=sep))
                    specs.append(dict(pk=p + int(round(sep * FS)), snr=20, tid=int(t2), stretch=1.0, pair=k, sep=sep))
            yi, inj = inject(y, P0['sigma'], specs)
            P = prep(yi)
            for (d, thr) in sets:
                segs = in_eval(run_setting(d, thr, t, P), e0, e1)
                mt, lab = match(segs, inj)
                cl = clean[(d, thr)]
                n_fp = sum(1 for i, l in enumerate(lab) if l == 'other' and not any(overlaps(segs[i], c) for c in cl))
                det_rows.append(dict(span=name, rep=rep, kind=kind, det=d, thr=thr, n_det=len(segs), n_tp=lab.count('tp'),
                                     n_frag=lab.count('frag'), n_fp_new=n_fp, n_inj=len(inj), hours=hours))
                for j, i in zip(inj, mt):
                    r = dict(span=name, rep=rep, kind=kind, det=d, thr=thr, snr=j['snr'], stretch=j['stretch'], tid=j['tid'],
                             pair=j['pair'], sep=j['sep'], pk=j['pk'], matched=i >= 0)
                    if i >= 0:
                        a, b = segs[i]
                        r.update(onset_err=(a - j['gt'][0]) / FS, offset_err=(b - j['gt'][1]) / FS, dur=(b - a) / FS,
                                 capture=j['energy'][a:b].sum() / j['energy'].sum(),
                                 n_frag=sum(1 for q, l in zip(segs, lab) if l == 'frag' and overlaps(q, j['ext'])))
                    inj_rows.append(r)
    # ---------------- surrogates (pure stationary noise with the span's quiet PSD)
    for rep in range(R_SUR):
        rng = np.random.default_rng(10 ** 6 + 1000 * si + rep)
        P = prep(surrogate(P0, rng))
        for (d, thr) in sets:
            sur_rows.append(dict(span=name, rep=rep, det=d, thr=thr, n_fa=len(in_eval(run_setting(d, thr, t, P), e0, e1)), hours=hours))
    return inj_rows, det_rows, sur_rows, cost_rows


if __name__ == '__main__':
    split = sys.argv[1]
    if TSET:
        R_SUR = 0   # surrogates do not depend on templates
    spans = [m['name'] for m in (tune_spans() if split == 'tune' else holdout_spans())]
    idx = {m['name']: i for i, m in enumerate(META)}
    with Pool(5) as pool:
        res = pool.map(work, [(idx[n], n, split) for n in spans])
    out = os.path.join(os.path.dirname(__file__), '..', 'data')
    for k, nm in enumerate(['inj', 'det', 'sur', 'cost']):
        pd.DataFrame([r for x in res for r in x[k]]).to_parquet(f'{out}/{nm}_{split}{"_" + TSET if TSET else ""}.parquet')
    print('done', split, len(spans))
