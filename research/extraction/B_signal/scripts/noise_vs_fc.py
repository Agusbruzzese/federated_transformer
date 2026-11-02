"""Noise sigma of the recommended chain vs LP cut-off, on the 29 quiet 60-s mains-free raw stretches
(v2_quiet.pkl): LP fc -> 50 Hz -> 20 s running median -> MAD sigma. Output: data/out_noise_fc.txt"""
import os, pickle, numpy as np, preproc as P
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
Q = pickle.load(open(os.path.join(OUT, "v2_quiet.pkl"), "rb"))
L = []
for fc in (2, 3, 5, 8, 10, 15, 20):
    s = []
    for q in Q:
        y = P.to50(P.lowpass(q["x"].astype(float), fc)); m = np.zeros(len(y), bool)
        r = (y - P.baseline(y, m))[100:-100]
        s.append(1.4826 * np.median(np.abs(r - np.median(r))))
    L.append(f"LP {fc:>2} Hz: noise sigma median {np.median(s):.1f}, p10/p90 {np.percentile(s, [10, 90]).round(1)} counts (n={len(s)} quiet 60 s stretches)")
print("\n".join(L)); open(os.path.join(OUT, "out_noise_fc.txt"), "w").write("\n".join(L))
