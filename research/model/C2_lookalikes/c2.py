"""C2 look-alike groups (PREREG.md v2, exploratory).

    python research/model/C2_lookalikes/c2.py dryrun
    python research/model/C2_lookalikes/c2.py run
"""
import os, sys, json
import numpy as np, pandas as pd
ROOT = os.getcwd(); sys.path[:0] = [ROOT, f"{ROOT}/research/model/C1_clustering"]
import c1
from c1 import m4, m5, aligned, phase_rand, hdb
from extraction.morph import morph_feats
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score as ARI, homogeneity_score

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(ROOT, "data", "c2"); os.makedirs(OUT, exist_ok=True)
TOL = {"main": (0.90, 1.5, 1.5), "strict": (0.95, 1.25, 1.25), "loose": (0.85, 2.0, 2.0)}
MINSIZE = 10; KEEP_SEEDS = (0, 1, 3); ALL_SEEDS = (0, 1, 2, 3, 4)


# ------------------------------------------------------------------ core
def tol_dist(P, la, ld, tol=TOL["main"]):
    c, a, d = tol
    D = np.maximum((1 - P) / (1 - c), np.abs(la[:, None] - la[None, :]) / np.log10(a))
    D = np.maximum(D, np.abs(ld[:, None] - ld[None, :]) / np.log10(d)).astype(np.float64)
    np.fill_diagonal(D, 0); return np.maximum(D, D.T)


def small_to_noise(lab):
    lab = lab.copy(); u, cnt = np.unique(lab[lab >= 0], return_counts=True)
    for g, k in zip(u, cnt):
        if k < MINSIZE: lab[lab == g] = -1
    keep = np.unique(lab[lab >= 0]); remap = {g: i for i, g in enumerate(keep)}
    return np.array([remap.get(g, -1) for g in lab])


def c2a(D):
    """Complete linkage cut at 1: every within-group pair satisfies the tolerance."""
    if len(D) < MINSIZE: return -np.ones(len(D), int)
    lab = fcluster(linkage(squareform(D, checks=False), "complete"), t=1.0, criterion="distance") - 1
    return small_to_noise(lab)


def collapsed(lab):
    g = lab[lab >= 0]
    return (lab.max() + 1) < 5 or (len(g) and np.bincount(g).max() / len(g) > 0.50)


def consensus(labs):
    n = len(labs[0]); C = np.zeros((n, n), np.float32)
    for l in labs:
        C += ((l[:, None] == l[None, :]) & (l[:, None] >= 0)).astype(np.float32)
    C /= len(labs); Dc = 1 - C.astype(np.float64); np.fill_diagonal(Dc, 0)
    lab = fcluster(linkage(squareform(Dc, checks=False), "average"), t=0.5, criterion="distance") - 1
    noise_majority = np.mean([l < 0 for l in labs], 0) > 0.5
    lab[noise_majority] = -1
    return small_to_noise(lab)


def refine(D, pre):
    """C2a inside each pre-partition group."""
    out = -np.ones(len(D), int); nxt = 0
    for g in np.unique(pre[pre >= 0]):
        i = np.flatnonzero(pre == g); l = c2a(D[np.ix_(i, i)])
        for k in np.unique(l[l >= 0]): out[i[l == k]] = nxt; nxt += 1
    return out


def r2_rep(e, scale):
    p = PCA(8, random_state=0).fit(StandardScaler().fit_transform(e))
    return np.c_[StandardScaler().fit_transform(p.transform(StandardScaler().fit_transform(e))), StandardScaler().fit_transform(scale)]


def c2b(D, embs, scale, idx, seeds):
    """embs: dict seed -> embedding of ALL events; HDBSCAN per seed on idx subset, collapse rule, consensus, refine."""
    labs = {s: hdb(r2_rep(embs[s][idx], scale[idx])) for s in seeds}
    kept = [s for s in seeds if not collapsed(labs[s])]
    if len(kept) < 2: return -np.ones(len(idx), int), kept, None
    pre = consensus([labs[s] for s in kept])
    return refine(D[np.ix_(idx, idx)], pre), kept, pre


# ------------------------------------------------------------------ validation
def medoid_transfer(D, labA, iA, iB):
    gs = np.unique(labA[labA >= 0])
    if not len(gs): return -np.ones(len(iB), int)
    med = []
    for g in gs:
        m_ = iA[labA == g]; med.append(m_[np.argmin(D[np.ix_(m_, m_)].sum(1))])
    d = D[np.ix_(iB, np.array(med))]; j = d.argmin(1)
    return np.where(d[np.arange(len(iB)), j] <= 1.0, gs[j], -1)


def compare(own, tr, rng):
    j = (own >= 0) & (tr >= 0)
    r = dict(ari_joint=float(ARI(own[j], tr[j])) if j.sum() > 1 else 0.0, ari_with_noise=float(ARI(own, tr)),
             frac_joint=float(j.mean()), n_joint=int(j.sum()), n_groups_B=int(len(np.unique(own[own >= 0]))))
    g = np.flatnonzero(own >= 0); null = []
    for _ in range(100):
        perm = own.copy(); perm[g] = rng.permutation(own[g]); null.append(ARI(perm[j], tr[j]) if j.sum() > 1 else 0.0)
    r["rand_p95"] = float(np.quantile(null, .95)); return r


def repro(D, method, groups, rng):
    """method(idx, fold) -> labels for events idx."""
    out = {}
    for g in np.unique(groups):
        iA, iB = np.flatnonzero(groups != g), np.flatnonzero(groups == g)
        la, lb = method(iA, g, "A"), method(iB, g, "B")
        out[int(g)] = compare(lb, medoid_transfer(D, la, iA, iB), rng)
    return out


def summarise(folds):
    v = list(folds.values())
    return dict(per_fold=folds, mean_ari_joint=float(np.mean([f["ari_joint"] for f in v])), min_ari_joint=float(min(f["ari_joint"] for f in v)),
                mean_frac_joint=float(np.mean([f["frac_joint"] for f in v])),
                beats_random_all=bool(all(f["ari_joint"] > f["rand_p95"] for f in v)), min_n_joint=int(min(f["n_joint"] for f in v)))


def tightness(lab, amp, dur, P):
    rows = []
    for g in np.unique(lab[lab >= 0]):
        i = np.flatnonzero(lab == g); sub = P[np.ix_(i, i)][np.triu_indices(len(i), 1)]
        rows.append((len(i), np.quantile(amp[i], .9) / np.quantile(amp[i], .1), np.quantile(dur[i], .9) / np.quantile(dur[i], .1), np.median(sub), sub.min()))
    if not rows: return {}
    r = np.array(rows); w = r[:, 0] / r[:, 0].sum()
    return dict(amp_p90_p10=float(w @ r[:, 1]), dur_p90_p10=float(w @ r[:, 2]), median_corr=float(w @ r[:, 3]), min_pair_corr=float(r[:, 4].min()))


def basic(lab):
    g = lab[lab >= 0]
    return dict(n_groups=int(len(np.unique(g))), coverage=float((lab >= 0).mean()),
                largest=int(np.bincount(g).max()) if len(g) else 0, median_size=float(np.median(np.bincount(g))) if len(g) else 0)


def feats(x):
    """log10 peak amplitude and log10 dur10 for arbitrary crops (dry run / nulls), via morph_feats on the central 8 s."""
    f = np.array([morph_feats(e[99:501].astype(np.float64), 50.0) for e in x])
    return np.nan_to_num(f[:, 0], nan=0.0), np.log10(np.clip(np.nan_to_num(f[:, 4], nan=0.05), 0.05, None))


# ------------------------------------------------------------------ dry run
def synth(x, rng, amp_rng, snr_rng):
    big = np.flatnonzero(np.abs(x).max(1) > 15000); tmpl = x[rng.choice(big, 5, replace=False)]
    ev, lab = [], []
    for k, t in enumerate(tmpl):
        for _ in range(60):
            a = rng.uniform(*amp_rng); s = rng.integers(-25, 26); snr = rng.uniform(*snr_rng)
            y = np.roll(t * a, s); ev.append(y + rng.standard_normal(600) * np.abs(y).max() / snr); lab.append(k)
    for _ in range(20):
        i, j = rng.choice(5, 2, replace=False); ev.append(tmpl[i] + np.roll(tmpl[j], rng.integers(-150, 150)) + rng.standard_normal(600) * 300); lab.append(-2)
    for e in phase_rand(x[rng.choice(len(x), 300, replace=False)], rng): ev.append(e); lab.append(-1)
    return np.array(ev, np.float32), np.array(lab)


def run_c2a_on(ev, tol=TOL["main"]):
    la, ld = feats(ev); return c2a(tol_dist(aligned(ev, "corr"), la, ld, tol))


def dryrun():
    m, W = m4.load(); x = c1.raw_crops(); rng = np.random.default_rng(2); res = {}
    for name, ar, sr in (("main", (0.9, 1.11), (20, 300)), ("snr20_40", (0.9, 1.11), (20, 40)), ("amp0.8_1.25", (0.8, 1.25), (20, 300)), ("amp0.5_2", (0.5, 2.0), (20, 300))):
        ev, lab = synth(x, np.random.default_rng(1), ar, sr); r = run_c2a_on(ev); t = lab >= 0
        res[name] = dict(ari_templates=float(ARI(lab[t], r[t])), homogeneity=float(homogeneity_score(lab[t], r[t])), singletons_as_noise=float((r[lab == -1] == -1).mean()),
                         recall_per_template=[float((r[lab == k] >= 0).mean()) for k in range(5)], n_groups=int(r.max() + 1),
                         overlaps_as_noise=float((r[lab == -2] == -1).mean()))
    res["null_phase_groups"] = int(run_c2a_on(phase_rand(x[rng.choice(len(x), 400, replace=False)], rng)).max() + 1)
    res["PASS"] = bool(res["main"]["ari_templates"] >= 0.70 and res["main"]["singletons_as_noise"] >= 0.80 and res["null_phase_groups"] <= 1)
    json.dump(res, open(os.path.join(HERE, "dryrun.json"), "w"), indent=1); print(json.dumps(res, indent=1))


# ------------------------------------------------------------------ main run
def load_emb(out_dir, seeds):
    m5.OUT = out_dir; return {s: m5.emb("simclr_tf", s, "final") for s in seeds}


def run():
    m, W = m4.load(); x = c1.raw_crops(); n = len(m); rng = np.random.default_rng(2)
    P = np.load(os.path.join(ROOT, "data", "c1", "aligned_corr.npy"))
    la = m.m8_log_amp.to_numpy(); ld = np.log10(np.clip(m.m8_dur10.to_numpy(), 0.05, None)); amp, dur = 10 ** la, 10 ** ld
    runid = m.run_id.to_numpy(); half = pd.to_datetime(m.date.astype(str)).dt.day.to_numpy() % 2; hbin = m.hour.to_numpy() // 6
    scale = np.c_[la, ld]; D = tol_dist(P, la, ld); R = {}; labels = {}

    # C2a
    labels["C2a"] = c2a(D); R["C2a"] = dict(**basic(labels["C2a"]), tight=tightness(labels["C2a"], amp, dur, P))
    meth_a = lambda idx, g, side: c2a(D[np.ix_(idx, idx)])
    R["C2a"]["loro"] = summarise(repro(D, meth_a, runid, rng)); R["C2a"]["oddeven"] = summarise(repro(D, meth_a, half, rng))
    print("C2a", R["C2a"]["n_groups"], R["C2a"]["coverage"], R["C2a"]["loro"]["mean_ari_joint"], flush=True)
    # tolerance + crop sensitivity
    R["C2a_sens"] = {}
    for k in ("strict", "loose"):
        l = c2a(tol_dist(P, la, ld, TOL[k])); R["C2a_sens"][k] = basic(l); labels[f"C2a_{k}"] = l
    P402 = aligned(x[:, 99:501], "corr"); l = c2a(tol_dist(P402, la, ld)); R["C2a_sens"]["crop402"] = dict(**basic(l), ari_vs_main=float(ARI(labels["C2a"], l)))
    hi = m.snr.to_numpy() >= 30; l = c2a(D[np.ix_(hi, hi)]); R["C2a_sens"]["snr30"] = basic(l)

    # C2b (full data, C1 models)
    embs = load_emb(os.path.join(ROOT, "data", "c1"), ALL_SEEDS); allidx = np.arange(n)
    lb, kept, pre = c2b(D, embs, scale, allidx, KEEP_SEEDS); labels["C2b"] = lb
    R["C2b"] = dict(**basic(lb), kept_seeds=kept, consensus=basic(pre), tight=tightness(lb, amp, dur, P))
    lb5, kept5, _ = c2b(D, embs, scale, allidx, ALL_SEEDS); labels["C2b_all5"] = lb5; R["C2b_all5"] = dict(**basic(lb5), kept_seeds=kept5)
    # E5a matched-coverage random pre-partition control
    sizes = np.bincount(pre[pre >= 0]); ctrl = []
    for d_ in range(5):
        perm = rng.permutation(n); rp = -np.ones(n, int); o = 0
        for g, s in enumerate(sizes): rp[perm[o:o + s]] = g; o += s
        ctrl.append(refine(D, rp))
    R["C2b_ctrl_random_pre"] = dict(coverage_mean=float(np.mean([(c >= 0).mean() for c in ctrl])), n_groups_mean=float(np.mean([c.max() + 1 for c in ctrl])))
    ctrl_meth = lambda idx, g, side: refine(D[np.ix_(idx, idx)], rand_pre(idx))
    def rand_pre(idx):
        k = len(idx) / n; rp = -np.ones(len(idx), int); perm = rng.permutation(len(idx)); o = 0
        for g, s in enumerate(np.maximum(1, (sizes * k).astype(int))): rp[perm[o:o + s]] = g; o += s
        return rp
    R["C2b_ctrl_random_pre"]["loro"] = summarise(repro(D, ctrl_meth, runid, rng))
    # C2b LORO with fold-retrained SSL
    fold_emb = {r: load_emb(os.path.join(OUT, f"loro_r{r}"), KEEP_SEEDS) for r in range(4)}
    R["C2b"]["fold_kept"] = {}
    def meth_b(idx, g, side):
        lab, kept_, _ = c2b(D, fold_emb[int(g)], scale, idx, KEEP_SEEDS); R["C2b"]["fold_kept"][f"{g}{side}"] = kept_; return lab
    R["C2b"]["loro"] = summarise(repro(D, meth_b, runid, rng))
    R["C2b"]["oddeven_optimistic"] = summarise(repro(D, lambda idx, g, side: c2b(D, embs, scale, idx, KEEP_SEEDS)[0], half, rng))
    print("C2b", R["C2b"]["n_groups"], R["C2b"]["coverage"], R["C2b"]["loro"]["mean_ari_joint"], flush=True)

    # nulls
    nulls = []
    for _ in range(10):
        ph = phase_rand(x, rng); Pn = aligned(ph, "corr"); nulls.append(basic(c2a(tol_dist(Pn, la, ld))))
    R["null_phase"] = dict(draws=nulls, mean_groups=float(np.mean([d["n_groups"] for d in nulls])))
    bg = c1.np.load(os.path.join(c1.C.OUT, "bgpool.npy"))
    bgx = bg[:, 20:620].astype(np.float32)                            # bgpool windows are already 640-sample 50 Hz crops: centre 600
    bla, bld = feats(bgx); R["null_bgpool"] = basic(c2a(tol_dist(aligned(bgx, "corr"), bla, bld)))
    rev = x[:, ::-1].copy(); Xr = np.r_[x, rev]; Pr = aligned(Xr, "corr", chunk=8)
    lr = c2a(tol_dist(Pr, np.r_[la, la], np.r_[ld, ld])); orig = np.r_[np.ones(n, bool), np.zeros(n, bool)]
    mixed = [g for g in np.unique(lr[lr >= 0]) if 0 < orig[lr == g].mean() < 1]
    R["null_timereversed_descriptive"] = dict(groups=int(lr.max() + 1), mixed_groups=len(mixed))

    # decision
    def reliable(k):
        L = R[k]["loro"]
        checks = dict(ari=L["mean_ari_joint"] >= 0.50 and L["min_ari_joint"] >= 0.30, frac=L["mean_frac_joint"] >= 0.30,
                      beats_random=L["beats_random_all"], n_joint=L["min_n_joint"] >= 50,
                      null=R["null_phase"]["mean_groups"] <= 0.10 * max(1, R[k]["n_groups"]), groups=R[k]["n_groups"] >= 5)
        return dict(checks=checks, reliable=all(checks.values()))
    R["decision"] = {k: reliable(k) for k in ("C2a", "C2b")}
    R["ssl_helps"] = bool(R["C2b"]["loro"]["mean_ari_joint"] > R["C2b_ctrl_random_pre"]["loro"]["mean_ari_joint"]
                          and R["C2b"]["coverage"] > R["C2b_ctrl_random_pre"]["coverage_mean"])
    # composition
    for k in ("C2a", "C2b"):
        l = labels[k]; comp = []
        for g in np.unique(l[l >= 0]):
            i = l == g; rs = np.bincount(runid[i], minlength=4) / i.sum()
            comp.append(dict(group=int(g), size=int(i.sum()), top_run_share=float(rs.max()), amp_median=float(np.median(amp[i])),
                             dur_median=float(np.median(dur[i])), polarity_pos=float((m.m8_polarity.to_numpy()[i] > 0).mean()),
                             hour_median=float(np.median(m.hour.to_numpy()[i]))))
        pd.DataFrame(comp).to_csv(os.path.join(HERE, f"profiles_{k}.csv"), index=False)
        R[k]["groups_run_dominated"] = int(sum(c["top_run_share"] > 0.80 for c in comp))
    json.dump(R, open(os.path.join(HERE, "results.json"), "w"), indent=1, default=float)
    pd.DataFrame({"event_id": m.event_id, **labels}).to_csv(os.path.join(HERE, "c2_labels.csv"), index=False)
    gallery(x, labels, amp, dur)
    print(json.dumps({k: R[k] for k in ("decision", "ssl_helps")}, indent=1, default=float))


def gallery(x, labels, amp, dur):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    t = np.arange(600) / 50
    for k in ("C2a", "C2b"):
        lab = labels[k]; gs = np.unique(lab[lab >= 0])
        if not len(gs): continue
        top = sorted(gs, key=lambda g: -(lab == g).sum())[:12]; rnd = list(np.random.default_rng(3).choice(gs, min(12, len(gs)), replace=False))
        for tag, sel in (("largest", top), ("random", rnd)):
            fig, axs = plt.subplots(3, 4, figsize=(16, 9))
            for ax, g in zip(axs.flat, sel):
                i = np.flatnonzero(lab == g)
                for j in i[:40]: ax.plot(t, x[j], lw=0.5, alpha=0.4, color="0.4")
                ax.plot(t, np.median(x[i], 0), color="C0", lw=1.6)
                ax.set_title(f"group {g}: n={len(i)}, peak {np.median(amp[i]):.0f}, dur {np.median(dur[i]):.1f}s", fontsize=9)
            for ax in axs.flat[len(sel):]: ax.axis("off")
            fig.suptitle(f"{k}: {tag} look-alike groups (grey: up to 40 members, blue: median), counts vs s"); fig.tight_layout()
            fig.savefig(os.path.join(HERE, f"gallery_{k}_{tag}.png"), dpi=90); plt.close(fig)


if __name__ == "__main__":
    {"dryrun": dryrun, "run": run}[sys.argv[1]]()
