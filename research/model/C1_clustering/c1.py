"""C1 clustering by magnetic signature (PREREG.md v2, frozen cd10a449669338cc).

    python research/model/C1_clustering/c1.py dryrun
    python research/model/C1_clustering/c1.py run
"""
import os, sys, json
import numpy as np, pandas as pd, torch
ROOT = os.getcwd(); sys.path[:0] = [ROOT, f"{ROOT}/research/model/M4_methods", f"{ROOT}/research/model/M5_long"]
import m4, m5
from ssl_central import augment as A, config as C
from extraction.morph import morph_feats
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score as ARI

HERE = os.path.dirname(os.path.abspath(__file__)); DEV = "cuda"; m5.OUT = os.path.join(ROOT, "data", "c1")
SIG0, SHIFT, MCS = 100.0, 50, 15
DECISION = ["R1_awd_s100", "R3_morph8"] + [f"R2_simclr_s{s}" for s in range(5)]
rng0 = np.random.default_rng(0)


# ------------------------------------------------------------------ signals
def raw_crops():
    m, W = m4.load()
    c = torch.full((len(W),), C.CENTRE, device=DEV)
    x = A.render(torch.tensor(W, device=DEV), torch.tensor(m.fs_eff.to_numpy(np.float32), device=DEV), c, torch.ones(len(W), device=DEV))
    return x.cpu().numpy().astype(np.float32)                          # counts, NOT max-normalised (D8)


def logc(x, s0=SIG0): return np.sign(x) * np.log1p(np.abs(x) / s0)


@torch.no_grad()
def aligned(X, mode="dist", shift=SHIFT, chunk=48):
    """mode='dist': min over |s|<=shift of Euclidean distance; mode='corr': max over shifts of Pearson correlation."""
    X = torch.tensor(X, device=DEV, dtype=torch.float32)
    if mode == "corr":
        X = X - X.mean(1, keepdim=True); X = X / (X.norm(dim=1, keepdim=True) + 1e-9)
    n, L = X.shape; F = torch.fft.rfft(X, n=2 * L); sq = (X ** 2).sum(1)
    out = np.empty((n, n), dtype=np.float32)
    for i in range(0, n, chunk):
        c = torch.fft.irfft(torch.conj(F[i:i + chunk])[:, None, :] * F[None, :, :], n=2 * L)
        best = torch.cat([c[..., :shift + 1], c[..., -shift:]], -1).amax(-1)       # lags 0..shift and -shift..-1
        if mode == "corr":
            out[i:i + chunk] = best.cpu().numpy()
        else:
            out[i:i + chunk] = torch.sqrt(torch.clamp(sq[i:i + chunk, None] + sq[None, :] - 2 * best, min=0)).cpu().numpy()
    if mode == "dist":
        out = np.minimum(out, out.T); np.fill_diagonal(out, 0)
    return out


def hdb(Z, precomputed=False, mcs=MCS):
    return HDBSCAN(min_cluster_size=mcs, metric="precomputed" if precomputed else "euclidean").fit_predict(Z.astype(np.float64))


# ------------------------------------------------------------------ dry run (D10; D9 failed, see PREREG)
def phase_rand(X, rng):
    F = np.fft.rfft(X, axis=1); ph = rng.uniform(0, 2 * np.pi, F.shape); ph[:, 0] = 0
    return np.fft.irfft(np.abs(F) * np.exp(1j * ph), n=X.shape[1], axis=1).astype(np.float32)


def dryrun():
    m, W = m4.load(); x = raw_crops(); rng = np.random.default_rng(1)
    big = np.flatnonzero(np.abs(x).max(1) > 15000); tmpl = x[rng.choice(big, 5, replace=False)]
    ev, lab = [], []
    for k, t in enumerate(tmpl):
        for _ in range(60):
            a = rng.uniform(0.5, 2.0); s = rng.integers(-25, 26); snr = rng.uniform(20, 300)
            y = np.roll(t * a, s); ev.append(y + rng.standard_normal(600) * np.abs(y).max() / snr); lab.append(k)
    for _ in range(20):
        i, j = rng.choice(5, 2, replace=False); ev.append(tmpl[i] + np.roll(tmpl[j], rng.integers(-150, 150)) + rng.standard_normal(600) * 300); lab.append(-2)
    for e in phase_rand(x[rng.choice(len(x), 300, replace=False)], rng):
        ev.append(e); lab.append(-1)                                   # singletons: real spectrum, unique shape
    ev = np.array(ev, np.float32); lab = np.array(lab)
    res = {}
    D = aligned(logc(ev)); r1 = hdb(D, True)
    feats = np.array([morph_feats(e.astype(np.float64), 50.0) for e in ev]); r3 = hdb(StandardScaler().fit_transform(np.nan_to_num(feats)))
    for nm, r in (("R1_awd", r1), ("R3_morph", r3)):
        t = lab >= 0
        res[nm] = dict(ari_templates=float(ARI(lab[t], r[t])), noise_as_noise=float((r[lab == -1] == -1).mean()), n_clusters=int(r.max() + 1),
                       template_recall=float((r[t] >= 0).mean()), overlaps_as_noise=float((r[lab == -2] == -1).mean()))
    pure = (rng.standard_normal((400, 600)) * 100).astype(np.float32)
    res["null_R1"] = int(hdb(aligned(logc(pure)), True).max() + 1)
    res["null_R1_phase"] = int(hdb(aligned(logc(phase_rand(x[rng.choice(len(x), 400, replace=False)], rng))), True).max() + 1)
    res["null_vec"] = int(hdb(rng.standard_normal((400, 19))).max() + 1)
    res["PASS"] = bool(all(res[k]["ari_templates"] >= 0.70 and res[k]["noise_as_noise"] >= 0.60 for k in ("R1_awd", "R3_morph"))
                       and res["null_R1"] <= 1 and res["null_R1_phase"] <= 1 and res["null_vec"] <= 1)
    json.dump(res, open(os.path.join(HERE, "dryrun.json"), "w"), indent=1); print(json.dumps(res, indent=1))


# ------------------------------------------------------------------ validation helpers
def cramers_v(a, b):
    t = pd.crosstab(a, b).to_numpy().astype(float); n = t.sum()
    if t.shape[0] < 2 or t.shape[1] < 2: return 0.0
    e = t.sum(1, keepdims=True) * t.sum(0, keepdims=True) / n; chi2 = ((t - e) ** 2 / e).sum()
    return float(np.sqrt(chi2 / (n * (min(t.shape) - 1))))


def tightness(lab, amp, dur, P, rng, max_pairs=1500):
    def stats(L):
        cv_a, cv_d, cr, w = [], [], [], []
        for c in np.unique(L[L >= 0]):
            i = np.flatnonzero(L == c)
            cv_a.append(amp[i].std() / amp[i].mean()); cv_d.append(dur[i].std() / dur[i].mean())
            a, b = rng.integers(0, len(i), max_pairs), rng.integers(0, len(i), max_pairs); ok = a != b
            cr.append(np.median(P[i[a[ok]], i[b[ok]]])); w.append(len(i))
        w = np.array(w) / np.sum(w)
        return float(np.dot(w, cv_a)), float(np.dot(w, cv_d)), float(np.dot(w, cr))
    real = stats(lab); null = []
    cl = np.flatnonzero(lab >= 0)
    for _ in range(100):
        L = lab.copy(); L[cl] = rng.permutation(lab[cl]); null.append(stats(L))
    null = np.array(null)
    return dict(cv_amp=real[0], cv_dur=real[1], corr=real[2], null_cv_amp_p5=float(np.quantile(null[:, 0], .05)),
                null_cv_dur_p5=float(np.quantile(null[:, 1], .05)), null_corr_p95=float(np.quantile(null[:, 2], .95)))


def ampmatched(lab, amp, P, rng, draws=100, max_pairs=1500):
    """D11: within-cluster aligned corr vs equal-size groups drawn outside the cluster, matched on amplitude 20-quantile bin."""
    q = np.asarray(pd.qcut(np.log10(amp), 20, labels=False, duplicates="drop")); bins = {b: np.flatnonzero(q == b) for b in np.unique(q)}
    def med(i):
        a, b = rng.integers(0, len(i), max_pairs), rng.integers(0, len(i), max_pairs); ok = a != b
        return np.median(P[i[a[ok]], i[b[ok]]])
    cl = np.unique(lab[lab >= 0]); w = np.array([(lab == c).sum() for c in cl], float); w /= w.sum()
    real = float(np.dot(w, [med(np.flatnonzero(lab == c)) for c in cl])); null = []
    for _ in range(draws):
        v = []
        for c in cl:
            mem = np.flatnonzero(lab == c); inc = set(mem.tolist()); g = []
            for j in mem:
                pool = [k for k in bins[q[j]] if k not in inc] if len(bins[q[j]]) < 400 else bins[q[j]]
                g.append(rng.choice(pool) if len(pool) else rng.integers(len(lab)))
            v.append(med(np.array(g)))
        null.append(np.dot(w, v))
    return dict(corr=real, null_ampmatched_p95=float(np.quantile(null, .95)), null_ampmatched_median=float(np.median(null)))


def transfer(labA, distBA):
    """Each B item takes the label of its nearest A item (noise stays noise)."""
    return labA[np.argmin(distBA, axis=1)]


def agree(own, tr):
    j = (own >= 0) & (tr >= 0)
    return dict(ari_joint=float(ARI(own[j], tr[j])) if j.sum() > 1 else np.nan, ari_with_noise=float(ARI(own, tr)), frac_joint=float(j.mean()))


def split_repro(get_dist, cluster, groups):
    """groups: array of group ids; returns the mean ARI over 'cluster A, transfer to B, compare with B's own'."""
    out = []
    for gA in np.unique(groups):
        A_, B_ = np.flatnonzero(groups != gA), np.flatnonzero(groups == gA)
        la, lb = cluster(A_), cluster(B_)
        out.append(agree(lb, transfer(la, get_dist(B_, A_))))
    return {k: float(np.nanmean([o[k] for o in out])) for k in out[0]}


# ------------------------------------------------------------------ main run
def run():
    m, W = m4.load(); x = raw_crops(); n = len(m); rng = np.random.default_rng(0)
    amp = 10 ** m.m8_log_amp.to_numpy(); dur = m.m8_dur10.to_numpy()
    day = pd.to_datetime(m.date.astype(str)).dt.day.to_numpy(); half = day % 2
    runid = m.run_id.to_numpy(); hbin = (m.hour.to_numpy() // 6); snrt = pd.qcut(m.snr, 3, labels=False).to_numpy()
    P = aligned(x, "corr"); np.save(os.path.join(ROOT, "data", "c1", "aligned_corr.npy"), P)
    reps = {}
    # R1 AWD
    for s0 in (SIG0, 50.0, 200.0):
        reps[f"R1_awd_s{int(s0)}"] = ("pre", aligned(logc(x, s0)))
    # R2 SimCLR-TF + scale, 5 seeds
    scale = np.c_[np.log10(amp), np.log10(np.clip(dur, 0.05, None))]
    pca_var = []
    for s in range(5):
        e = m5.emb("simclr_tf", s, "final"); p = PCA(8, random_state=0).fit(StandardScaler().fit_transform(e))
        pca_var.append(float(p.explained_variance_ratio_.sum()))
        reps[f"R2_simclr_s{s}"] = ("vec", np.c_[StandardScaler().fit_transform(p.transform(StandardScaler().fit_transform(e))), StandardScaler().fit_transform(scale)])
    reps["R3_morph8"] = ("vec", StandardScaler().fit_transform(m[[c for c in m.columns if c.startswith("m8_")]].fillna(0).to_numpy(float)))
    results, labels = {}, {}
    for name, (kind, Z) in reps.items():
        pre = kind == "pre"
        dist = (lambda B, A_: Z[np.ix_(B, A_)]) if pre else (lambda B, A_: np.linalg.norm(Z[B][:, None, :] - Z[A_][None, :, :], axis=2))
        clus = (lambda idx: hdb(Z[np.ix_(idx, idx)], True)) if pre else (lambda idx: hdb(Z[idx]))
        lab = clus(np.arange(n)); labels[name] = lab
        r = dict(n_clusters=int(lab.max() + 1), coverage=float((lab >= 0).mean()),
                 largest_share=float(np.bincount(lab[lab >= 0]).max() / max(1, (lab >= 0).sum())) if (lab >= 0).any() else 1.0)
        if name in DECISION:
            r["V1"] = tightness(lab, amp, dur, P, rng); r["V1_ampmatched"] = ampmatched(lab, amp, P, rng)
            r["V2_oddeven"] = split_repro(dist, clus, half); r["V2b_loro"] = split_repro(dist, clus, runid)
            r["confounds"] = dict(run=cramers_v(lab[lab >= 0], runid[lab >= 0]), hour=cramers_v(lab[lab >= 0], hbin[lab >= 0]), snr=cramers_v(lab[lab >= 0], snrt[lab >= 0]))
            r["sens_mcs"] = {mc: int((hdb(Z, pre, mc)).max() + 1) for mc in (10, 25)}
        results[name] = r; print(name, {k: v for k, v in r.items() if k in ("n_clusters", "coverage", "largest_share")}, flush=True)
    # V3 seed stability (R2)
    seedari = [agree(labels[f"R2_simclr_s{a}"], labels[f"R2_simclr_s{b}"]) for a in range(5) for b in range(a + 1, 5)]
    results["V3_R2_seed"] = {k: float(np.nanmean([o[k] for o in seedari])) for k in seedari[0]}; results["R2_pca8_explained"] = pca_var
    # V4 nulls (D6, D11: 10 draws each)
    results["V4_null"] = {}
    for name in DECISION:
        k = []
        for _ in range(10):
            if name.startswith("R1"):
                k.append(int(hdb(aligned(logc(phase_rand(x, rng))), True).max() + 1))
            else:
                Z = reps[name][1]; k.append(int(hdb(rng.multivariate_normal(Z.mean(0), np.cov(Z.T), size=n)).max() + 1))
        results["V4_null"][name] = dict(draws=k, max=max(k), pass_=sum(c <= 1 for c in k) >= 9)
        print("null", name, k, flush=True)
    # validity (decision rules + D2, D5, D11)
    def valid(name):
        r = results[name]; v1 = r["V1"]
        checks = dict(corr=v1["corr"] >= 0.80 and v1["corr"] > v1["null_corr_p95"], cv_amp=v1["cv_amp"] < v1["null_cv_amp_p5"],
                      cv_dur=v1["cv_dur"] < v1["null_cv_dur_p5"], repro_oddeven=r["V2_oddeven"]["ari_joint"] >= 0.50,
                      repro_loro=r["V2b_loro"]["ari_joint"] >= 0.50, null=results["V4_null"][name]["pass_"],
                      coverage=r["coverage"] >= 0.30, largest=r["largest_share"] <= 0.50)
        return dict(checks=checks, valid=all(checks.values()))
    results["validity"] = {k: valid(k) for k in DECISION}
    nv = sum(results["validity"][f"R2_simclr_s{s}"]["valid"] for s in range(5))
    results["validity"]["R2_simclr"] = dict(seeds_valid=nv, seed_ari=results["V3_R2_seed"]["ari_joint"],
                                            valid=bool(nv >= 3 and results["V3_R2_seed"]["ari_joint"] >= 0.50))
    top = {"R1_awd_s100": "R1_awd_s100", "R3_morph8": "R3_morph8", "R2_simclr": "R2_simclr_s0"}
    ok = [k for k in top if results["validity"][k]["valid"]]
    key = lambda k: (min(results[top[k]]["V2_oddeven"]["ari_joint"], results[top[k]]["V2b_loro"]["ari_joint"]), results[top[k]]["V1"]["corr"])
    results["recommended"] = max(ok, key=key) if ok else None
    json.dump(results, open(os.path.join(HERE, "results.json"), "w"), indent=1, default=float)
    L = pd.DataFrame({"event_id": m.event_id, **{k: v for k, v in labels.items()}}); L.to_csv(os.path.join(HERE, "cluster_labels.csv"), index=False)
    profiles(m, x, labels, amp, dur)
    print(json.dumps({k: results[k] for k in ("validity", "recommended", "V3_R2_seed")}, indent=1, default=float))


def profiles(m, x, labels, amp, dur):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    rows = []
    for name in ("R1_awd_s100", "R2_simclr_s0", "R3_morph8"):
        lab = labels[name]
        for c in np.unique(lab[lab >= 0]):
            i = lab == c
            rows.append(dict(rep=name, cluster=int(c), size=int(i.sum()), amp_median=float(np.median(amp[i])), dur_median=float(np.median(dur[i])),
                             pos_polarity=float((m.m8_polarity.to_numpy()[i] > 0).mean()), hour_median=float(np.median(m.hour.to_numpy()[i]))))
        cl = np.unique(lab[lab >= 0]); top = sorted(cl, key=lambda c: -(lab == c).sum())[:12]
        if not len(top): continue
        fig, axs = plt.subplots(3, 4, figsize=(16, 9)); t = np.arange(600) / 50
        for ax, c in zip(axs.flat, top):
            i = np.flatnonzero(lab == c)
            for j in i[:40]: ax.plot(t, x[j], lw=0.5, alpha=0.4, color="0.4")
            ax.plot(t, np.median(x[i], 0), color="C0", lw=1.6)
            ax.set_title(f"cluster {c}: n={len(i)}, median peak {np.median(amp[i]):.0f}", fontsize=9)
        for ax in axs.flat[len(top):]: ax.axis("off")
        fig.suptitle(f"{name}: 12 largest clusters (grey: up to 40 members, blue: median), counts vs s"); fig.tight_layout()
        fig.savefig(os.path.join(HERE, f"gallery_{name}.png"), dpi=90); plt.close(fig)
    pd.DataFrame(rows).to_csv(os.path.join(HERE, "cluster_profiles.csv"), index=False)


if __name__ == "__main__":
    {"dryrun": dryrun, "run": run}[sys.argv[1]]()
