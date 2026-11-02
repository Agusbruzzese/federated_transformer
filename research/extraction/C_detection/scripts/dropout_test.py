"""Do dropouts create artefacts / false events? Compare front-end treatments on dropout-dense tuning spans."""
import sys, os; sys.path.insert(0, '.'); sys.path.insert(0, 'scripts')
from common import *
import cio
from build_spans import load as load_raw
from scipy.signal import upfirdn
rows = []
for nm in ['r1_fri07_0800', 'r1_sun09_0300', 'r1_sat08_1000', 'r1_sun09_1600']:
    m = [k for k in META if k['name'] == nm][0]
    x, t, c, wall = load_raw(cio.runs()[m['run']], m['i0'], m['n'])
    variants = {}
    # (i) concat: treat raw as contiguous
    t50, y50, idx, _ = D.front_end(x, None); variants['concat'] = (y50, None)
    # (ii) raw interpolation over re-inserted slots
    t50, y50, idx, drop = D.front_end(x, t); variants['mask50'] = (y50, drop)   # (iii) = detector.front_end
    xs, _, _ = D.insert_dropouts(x, t); h = D._fir(); half = (len(h) - 1) // 2
    yr = upfirdn(h, xs - xs.mean(), down=D.DEC) + xs.mean(); ii = np.arange(len(yr)) * D.DEC - half
    yr = yr[(ii >= half) & (ii + half < len(xs))]; variants['raw_interp'] = (yr, drop)       # (ii) no masking
    for k, (y, dr) in variants.items():
        P = prep(y); u = P['zd'] / P['sigma']
        if dr is None:  # locate dropouts on the concat grid: raw index of each dropout -> 50 Hz sample
            dd = np.where(np.diff(t) > 1.5 * np.median(np.diff(t)))[0]
            dr = np.zeros(len(y), bool); j = np.searchsorted(idx, dd); dr[np.clip(j, 0, len(y) - 1)] = True
            dr = np.convolve(dr, np.ones(11), 'same') > 0
        # artefact: step in waveform channel across dropout = |z(after)-z(before)| over +-0.1 s, relative to sigma
        zd = np.abs(np.diff(P['z'])) / P['sigma'][1:]
        segs = D.detect_hyst(t50[:len(y)], P['zd'], P['sigma'], D.DEFAULTS)
        on = np.zeros(len(y), bool)
        for a, b in segs: on[a:b] = True
        rows.append(dict(span=nm, variant=k, drop_frac=dr.mean(), dz_at_drop=np.median(zd[dr[1:]]), dz_else=np.median(zd[~dr[1:]]),
                         p99_dz_at_drop=np.percentile(zd[dr[1:]], 99), p99_dz_else=np.percentile(zd[~dr[1:]], 99),
                         n_events=len(segs),
                         frac_starts_at_drop=np.mean([dr[max(a-5,0):a+5].any() for a, b in segs]) if len(segs) else np.nan))
import pandas as pd
df = pd.DataFrame(rows); pd.set_option('display.width', 200); print(df.round(3).to_string())
df.to_csv('data/dropout_test.csv', index=False)
