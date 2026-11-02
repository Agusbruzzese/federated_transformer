"""IMF-level EMD audit: mode mixing, IMF-index consistency under tiny shifts/noise,
where the vehicle energy lives, EMD vs EEMD vs CEEMDAN.  (single process)"""
import numpy as np, pandas as pd, time, json
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import signal
from features import emd_imfs, hht_stats_from_imfs, FS, FSD

SCR = "/tmp/claude-1000/-home-augustin-Documents-federated-transformer/8ccb016b-de96-4655-ab49-d3318afeb915/scratchpad/"
LP = np.load(SCR + "LP.npy", mmap_mode="r"); RAW = np.load(SCR + "RAW.npy", mmap_mode="r")
meta = pd.read_parquet(SCR + "meta.parquet")
rng = np.random.default_rng(7)
ev = np.where(meta.kind == "event")[0]; bg = np.where(meta.kind == "bg")[0]
N = 100
S = rng.choice(ev, N, replace=False)


def imf_if(imf, fs):
    a = signal.hilbert(imf); ph = np.unwrap(np.angle(a))
    f = np.diff(ph) * fs / 2 / np.pi; w = np.abs(a[1:]) ** 2
    return np.sum(f * w) / w.sum()          # amplitude-weighted mean IF


def lowband_frac(imf, fs, fc=4.0):
    f, p = signal.periodogram(imf, fs=fs)
    return p[f < fc].sum() / (p.sum() + 1e-12)


def orth_index(imfs):
    s = imfs.sum(0); num = 0.0
    for a in range(len(imfs)):
        for b in range(len(imfs)):
            if a != b: num += np.sum(imfs[a] * imfs[b])
    return abs(num) / np.sum(s ** 2)


def match_corr(A, B, K):
    """|corr| of IMF k (same index) between two decompositions of shifted/noisy copies, aligned."""
    out = []
    for k in range(K):
        if k < len(A) and k < len(B):
            out.append(abs(np.corrcoef(A[k], B[k])[0, 1]))
        else:
            out.append(np.nan)
    return out


rows = []; ifs_raw = []; ifs_lp = []; ifs_ce = []
corr = {m: [] for m in ["raw_emd_shift", "raw_emd_noise", "lp_emd_shift", "lp_emd_noise",
                        "lp_ceemdan_shift", "lp_ceemdan_noise", "lp_eemd_shift", "lp_eemd_noise"]}
featv = {m: {'orig': [], 'shift': [], 'noise': []} for m in ['emd', 'eemd', 'ceemdan']}
cost = {"raw_emd": [], "lp_emd": [], "lp_eemd": [], "lp_ceemdan": []}
sh_r, sh_l = 240, 10  # 0.1 s shift
for n, i in enumerate(S):
    r = np.array(RAW[i], float); x = np.array(LP[i], float)
    j = rng.choice(bg); rb = np.array(RAW[j], float); rb -= np.median(rb); xb = np.array(LP[j], float)
    r0 = r[2400:2400 + 19200]; r1 = r[2400 + sh_r:2400 + sh_r + 19200]; rn = r0 + rb[2400:2400 + 19200]
    x0 = x[200:1000]; x1 = x[200 + sh_l:1000 + sh_l]; xn = x0 + xb[200:1000]
    t = time.perf_counter(); I0 = emd_imfs(r0 - r0.mean()); cost["raw_emd"].append(time.perf_counter() - t)
    I1 = emd_imfs(r1 - r1.mean()); In = emd_imfs(rn - rn.mean())
    # align shifted decomposition to original time base for the overlapping part
    corr["raw_emd_shift"].append(match_corr(I0[:, sh_r:], I1[:, :-sh_r], 10))
    corr["raw_emd_noise"].append(match_corr(I0, In, 10))
    lf = [lowband_frac(im, FS) for im in I0]; en = np.array([np.sum(im ** 2) for im in I0])
    ifs_raw.append([imf_if(im, FS) for im in I0])
    for meth in ["emd", "eemd", "ceemdan"]:
        t = time.perf_counter(); J0 = emd_imfs(x0, method=meth); cost["lp_" + meth].append(time.perf_counter() - t)
        J1 = emd_imfs(x1, method=meth); Jn = emd_imfs(xn, method=meth)
        corr[f"lp_{meth}_shift"].append(match_corr(J0[:, sh_l:], J1[:, :-sh_l], 8))
        corr[f"lp_{meth}_noise"].append(match_corr(J0, Jn, 8))
        for nm, JJ in (("orig", J0), ("shift", J1), ("noise", Jn)):
            featv[meth][nm].append(hht_stats_from_imfs(JJ, FSD))
        if meth == "emd":
            ifs_lp.append([imf_if(im, FSD) for im in J0]); LJ = J0
        if meth == "ceemdan":
            ifs_ce.append([imf_if(im, FSD) for im in J0])
    enl = np.array([np.sum(im ** 2) for im in LJ])
    rows.append(dict(i=i, n_raw=len(I0), n_raw_shift=len(I1), n_raw_noise=len(In),
                     veh_imf_raw=int(np.argmax(en * np.array(lf))) + 1,
                     veh_share_first8=float((en[:8] * np.array(lf[:8])).sum() / (en * np.array(lf)).sum()),
                     frac_E_imf1to4_lowband=float((en[:4] * np.array(lf[:4])).sum() / en[:4].sum()),
                     frac_E_first8_highband=float((en[:8] * (1 - np.array(lf[:8]))).sum() / en[:8].sum()),
                     n_lp=len(LJ), veh_imf_lp=int(np.argmax(enl)) + 1,
                     io_raw=orth_index(I0), io_lp=orth_index(LJ)))
    if n % 20 == 0: print(n, flush=True)

df = pd.DataFrame(rows); df.to_csv("emd_audit_per_event.csv", index=False)
summ = {
    "n_imfs_raw_distribution": df.n_raw.value_counts().sort_index().to_dict(),
    "n_imfs_raw_changes_under_0.1s_shift": float((df.n_raw != df.n_raw_shift).mean()),
    "n_imfs_raw_changes_under_noise": float((df.n_raw != df.n_raw_noise).mean()),
    "vehicle_dominant_imf_index_raw_distribution": df.veh_imf_raw.value_counts().sort_index().to_dict(),
    "vehicle_lowband_energy_share_captured_by_imf1-8": float(df.veh_share_first8.median()),
    "median_frac_of_imf1-8_energy_above_4Hz(noise/mains)": float(df.frac_E_first8_highband.median()),
    "n_imfs_lp_distribution": df.n_lp.value_counts().sort_index().to_dict(),
    "vehicle_dominant_imf_index_lp_distribution": df.veh_imf_lp.value_counts().sort_index().to_dict(),
    "orthogonality_index_raw_median": float(df.io_raw.median()), "orthogonality_index_lp_median": float(df.io_lp.median()),
    "cost_s_median": {k: float(np.median(v)) for k, v in cost.items()},
    "same_index_IMF_abs_corr_median_per_k": {k: [float(np.nanmedian(np.array(v)[:, q])) for q in range(np.array(v).shape[1])] for k, v in corr.items()},
}
from scipy.stats import spearmanr
fs_ = {}
for meth, d in featv.items():
    O = np.array(d["orig"])
    for nm in ("shift", "noise"):
        A = np.array(d[nm])
        rho = [spearmanr(O[:, j], A[:, j])[0] for j in range(O.shape[1]) if O[:, j].std() > 0 and A[:, j].std() > 0]
        fs_[f"{meth}_{nm}"] = dict(median_rho=float(np.nanmedian(rho)), frac_rho_lt_0_8=float(np.nanmean(np.array(rho) < 0.8)),
                                   n_imf_changed=float(np.mean(O[:, 0] != A[:, 0])))
summ["hht_lp_feature_stability(0.1s shift / real noise)"] = fs_
json.dump(summ, open("emd_audit_summary.json", "w"), indent=1)
print(json.dumps(summ, indent=1))

# figure: mean IF per IMF index across events (mode-mixing spread)
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
for a, L, fs, tt in [(ax[0], ifs_raw, FS, "EMD on raw 2400 Hz (legacy)"), (ax[1], ifs_lp, FSD, "EMD on 20 Hz-LP / 100 Hz"),
                     (ax[2], ifs_ce, FSD, "CEEMDAN (50 trials) on LP")]:
    K = 10 if fs == FS else 7
    data = [[v[k] for v in L if len(v) > k] for k in range(K)]
    a.boxplot(data, whis=(5, 95), showfliers=False); a.set_yscale("log"); a.set_title(tt)
    a.set_xlabel("IMF index"); a.set_ylabel("energy-weighted mean inst. freq (Hz)")
    a.axhspan(0.05, 4, color="tab:green", alpha=.1, label="vehicle band (<4 Hz)"); a.axhline(50, color="r", ls=":", lw=1, label="50 Hz mains")
    a.legend(fontsize=7)
plt.tight_layout(); plt.savefig("fig_emd_modemixing.png", dpi=90)

fig, ax = plt.subplots(figsize=(8, 4))
for k, v in corr.items():
    ax.plot(range(1, np.array(v).shape[1] + 1), np.nanmedian(np.array(v), 0), marker="o", label=k,
            ls="-" if "shift" in k else "--")
ax.set_xlabel("IMF index k"); ax.set_ylabel("median |corr(IMF_k, IMF_k')|"); ax.set_ylim(0, 1)
ax.set_title("Same-index IMF consistency after 0.1 s shift / added real background noise")
ax.legend(fontsize=7, ncol=2); plt.tight_layout(); plt.savefig("fig_emd_imf_consistency.png", dpi=90)
