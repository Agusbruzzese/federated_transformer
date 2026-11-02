import sys, json, numpy as np
sys.path.insert(0, '.')
import detector as D, detectors_all as A
META = json.load(open('data/spans.json'))
NAMES = ["FIXED", "FIXEDLOC", "HYST", "STALTA", "CUSUM", "WAVELET", "HYSTBP", "STALTABP", "CUSUMBP", "BDET"]

def load_span(name):
    d = np.load(f'data/span_{name}.npz')
    return {k: d[k] for k in d.files}

def prep(y, p=None):
    return D.preprocess(y, dict(D.DEFAULTS, **(p or {})))

def detect_all(t, P, names=NAMES, params=None):
    """P = output of prep(). FIXED sees the gap-filled waveform (legacy DC-relative); others the detection channel."""
    out = {}
    for n in names:
        p = (params or {}).get(n)
        out[n] = A.run(n, t, P["y"], P["zd"], P["sigma"], p)
    return out

def tune_spans():
    return [m for m in META if m["split"] == "tune"]

def holdout_spans():
    return [m for m in META if m["split"] == "holdout"]
