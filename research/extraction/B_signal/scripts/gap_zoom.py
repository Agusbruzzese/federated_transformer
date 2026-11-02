"""Zoom at raw samples around dropouts: subtract a local 50 Hz+harmonics fit, look at the residual."""
import numpy as np, io_utils as io
from style import plt, RUNC, INK2
FS = 2416.0
C = io.list_chunks()
c = C[1009]  # run0 night chunk
v, t = io.load(c); d = np.diff(t); G = np.where(d > io.GAP_US)[0]
print(c["start"], "gaps at", G, d[G])
def harm_fit(n, x, f0=50.0 * 2416 / FS):
    M = [np.ones_like(n, float)] + [fn(2*np.pi*k*f0*n/FS) for k in range(1, 8) for fn in (np.cos, np.sin)]
    M = np.array(M).T; co = np.linalg.lstsq(M, x, rcond=None)[0]; return M @ co
fig, ax = plt.subplots(min(4, len(G)), 1, figsize=(14, 10))
for k, g in enumerate(G[:4]):
    a, b = g - 1200, g + 1200
    n = np.arange(a, b); x = v[a:b]
    pre = slice(0, 1100)
    # fit on pre-gap only, extrapolate (index-continuous hypothesis) and residual
    nn = n - a
    Mfit = harm_fit(nn[pre], x[pre])
    # local mains period from data
    ax[k].plot(n / FS, x - np.median(x), lw=.5, color=RUNC[0])
    ax[k].axvline((g + .5) / FS, color=RUNC[1]); ax[k].set_xlim((g - 400) / FS, (g + 400) / FS)
    ax[k].set_title(f"gap after sample {g}: Δt = {d[g]} µs", fontsize=8)
    # jump in first difference at gap
    dv = np.diff(v[a:b]); print(g, "dv at gap", v[g + 1] - v[g], " typical |dv| p99", np.percentile(np.abs(dv), 99))
fig.tight_layout(); fig.savefig("../figures/_gapzoom.png", dpi=80)
