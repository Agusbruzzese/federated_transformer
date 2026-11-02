"""Label-free comparison of representations (PROTOCOL.md v1).

Usage:  python evaluate.py [feats.npz] [meta.parquet] [tag]
Splits: dev = runs 0,1,3 (folders Run 20250204/20250212/20250224); holdout = run 2 (Run 20250218).
Every metric: value, 95% bootstrap CI (200 resamples, seed 0), and its null baseline.
"""
import sys, os, json, time, warnings
import numpy as np, pandas as pd
from multiprocessing import Pool
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, balanced_accuracy_score, r2_score
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier, NearestNeighbors
from features import LEGACY_NAMES, LEGACY_CONFOUNDS, MORPH_NAMES, legacy_read_int16

warnings.filterwarnings("ignore")
SCR = "/tmp/claude-1000/-home-augustin-Documents-federated-transformer/8ccb016b-de96-4655-ab49-d3318afeb915/scratchpad/"
B = int(os.environ.get("EVAL_B", 200))   # protocol: >=200; override only for smoke tests
SEED = 0
K_PRIMARY = 5                       # pre-registered k for CI-level cluster metrics
AUGS = ["orig", "shift+0.1", "shift-0.25", "shift+0.5", "noise", "scale0.8", "stretch1.15"]
NUIS = ["shift+0.1", "shift-0.25", "shift+0.5", "noise"]      # nuisance augs -> invariance score
HOLDOUT_RUN = 2

# ranking criteria: (name, higher_is_better)
CRITERIA = [("invariance_R@1", True), ("leak_date_skill", False), ("leak_hour_skill", False),
            ("sil_excess", True), ("ari_excess", True), ("proxy", True)]


def legacy70_from66(L66, meta_rows, ref):
    """Add the 4 reconstructed 'relative' columns (see features.legacy_add_relative) using per-date
    background baselines `ref` (dict date -> (median dc_range, median rms))."""
    date = meta_rows.wall.dt.strftime("%m%d").values
    dcb = legacy_read_int16(meta_rows.dc_baseline.values)
    rb = np.array([ref[d][0] for d in date]); sb = np.array([ref[d][1] for d in date])
    if L66.ndim == 3:
        dcb, rb, sb = dcb[:, None], rb[:, None], sb[:, None]
    lead = L66[..., :65]
    return np.concatenate([lead, (lead[..., 0] - dcb)[..., None], (lead[..., 1] - rb)[..., None],
                           (lead[..., 2] - sb)[..., None], np.broadcast_to(dcb, lead.shape[:-1])[..., None],
                           L66[..., 65:66]], -1)


def build_reps(F, meta, prefix="orig_", rows=None):
    """Dict rep -> matrix (n, d) [or (n, A, d) for aug_] from a feats npz-like mapping."""
    L66o = F["orig_legacy66"]; date = meta.wall.dt.strftime("%m%d").values; bg = (meta.kind == "bg").values
    ref = {}
    for d in np.unique(date):
        m = (date == d) & bg
        m = m if m.sum() >= 3 else (date == d)
        ref[d] = (np.median(L66o[m, 1]), np.median(L66o[m, 2]))
    mrows = meta if rows is None else meta.iloc[rows].reset_index(drop=True)
    L = legacy70_from66(F[prefix + "legacy66"], mrows, ref)
    keep = [i for i, n in enumerate(LEGACY_NAMES) if n not in LEGACY_CONFOUNDS]
    reps = {
        "legacy70": L,
        "legacy_noconf": L[..., keep],
        "legacy_fixed_io": F[prefix + "legacy_fixed_io"],
        "hht_lp": F[prefix + "hht_lp"],
        "wavelet": F[prefix + "wavelet"],
        "stft": F[prefix + "stft"],
        "morph": F[prefix + "morph"],
        "waveform": F[prefix + "waveform"],
        "morph+wavelet": np.concatenate([F[prefix + "morph"], F[prefix + "wavelet"]], -1),
    }
    return reps


class Scaler:
    """Robust z-score (median/IQR) with clipping at +-8; NaN/inf -> column median."""
    def fit(self, X):
        X = np.where(np.isfinite(X), X, np.nan)
        self.med = np.nanmedian(X, 0)
        q = np.nanpercentile(X, [25, 75], 0)
        iqr = (q[1] - q[0]) / 1.349
        sd = np.nanstd(X, 0)
        self.s = np.where(iqr > 1e-12, iqr, np.where(sd > 1e-12, sd, 1.0))
        return self

    def transform(self, X):
        X = np.where(np.isfinite(X), X, np.nan)
        X = np.where(np.isnan(X), self.med, X)
        return np.clip((X - self.med) / self.s, -8, 8)


def boot_ci(per_item, rng=None, stat=np.mean):
    rng = np.random.default_rng(SEED)
    per_item = np.asarray(per_item)
    n = len(per_item)
    bs = [stat(per_item[rng.integers(0, n, n)]) for _ in range(B)]
    return float(stat(per_item)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def fmt(v):
    return f"{v[0]:.3f} [{v[1]:.3f},{v[2]:.3f}]"


# ------------------------------------------------------------------ invariance
def invariance(Z0, Zaug):
    """Z0: (n,d) standardized originals; Zaug: (n,A,d). Retrieval@1 (own source is nearest original),
    normalized displacement, median per-feature Spearman."""
    nn = NearestNeighbors(n_neighbors=1).fit(Z0)
    typ = np.median(np.linalg.norm(Z0[np.random.default_rng(0).integers(0, len(Z0), 2000)] -
                                   Z0[np.random.default_rng(1).integers(0, len(Z0), 2000)], axis=1))
    out = {}
    hits_nuis = []
    for a, name in enumerate(AUGS):
        if name == "orig":
            continue
        Za = Zaug[:, a]
        idx = nn.kneighbors(Za, return_distance=False)[:, 0]
        hit = (idx == np.arange(len(Z0))).astype(float)
        disp = np.linalg.norm(Za - Z0, axis=1) / typ
        rho = [spearmanr(Z0[:, j], Za[:, j])[0] for j in range(Z0.shape[1]) if np.std(Z0[:, j]) > 0]
        out[name] = dict(R1=boot_ci(hit), disp=boot_ci(disp, stat=np.median), rho_med=float(np.nanmedian(rho)),
                         frac_feat_rho_lt_0_8=float(np.nanmean(np.array(rho) < 0.8)))
        if name in NUIS:
            hits_nuis.append(hit)
    out["invariance_R@1"] = boot_ci(np.mean(hits_nuis, 0))
    out["null_R@1"] = 1.0 / len(Z0)
    return out


# ------------------------------------------------------------------ redundancy
def redundancy(Z):
    C = np.corrcoef(Z[:, Z.std(0) > 0].T)
    ev = np.clip(np.linalg.eigvalsh(C), 0, None)[::-1]
    pr = ev.sum() ** 2 / (ev ** 2).sum()
    n95 = int(np.searchsorted(np.cumsum(ev) / ev.sum(), 0.95) + 1)
    iu = np.triu_indices_from(C, 1)
    rng = np.random.default_rng(SEED)
    bs = []
    for _ in range(B):
        s = rng.integers(0, len(Z), len(Z)); Zs = Z[s][:, Z.std(0) > 0]
        Cs = np.corrcoef(Zs.T); Cs = np.nan_to_num(Cs); e = np.clip(np.linalg.eigvalsh(Cs), 0, None)
        bs.append(e.sum() ** 2 / (e ** 2).sum())
    return dict(d=Z.shape[1], d_nonconst=int((Z.std(0) > 0).sum()), PR=(float(pr), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))),
                PR_over_d=float(pr / Z.shape[1]), n95=n95, frac_pairs_absr_gt_0_9=float(np.mean(np.abs(C[iu]) > 0.9)),
                null_PR_colperm=float((Z.std(0) > 0).sum()))


# ------------------------------------------------------------------ leakage
def _cv_pred(X, y, seed=SEED):
    clf = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, random_state=seed)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    return cross_val_predict(clf, X, y, cv=cv)


def skill_ci(y, p):
    """balanced-accuracy skill = (bacc - 1/K)/(1 - 1/K), bootstrap over items."""
    K = len(np.unique(y)); rng = np.random.default_rng(SEED)
    f = lambda yy, pp: (balanced_accuracy_score(yy, pp) - 1 / K) / (1 - 1 / K)
    v = f(y, p); n = len(y); bs = []
    for _ in range(B):
        s = rng.integers(0, n, n); bs.append(f(y[s], p[s]))
    return float(v), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def leakage(X, y):
    y = np.asarray(y)
    real = skill_ci(y, _cv_pred(X, y))
    ysh = np.random.default_rng(SEED).permutation(y)
    shuf = skill_ci(ysh, _cv_pred(X, ysh))
    maj = np.full_like(y, pd.Series(y).mode()[0])
    return dict(skill=real, null_shuffled=shuf, null_majority=float((balanced_accuracy_score(y, maj) - 1 / len(np.unique(y))) / (1 - 1 / len(np.unique(y)))),
                n_classes=int(len(np.unique(y))))


# ------------------------------------------------------------------ clusters
def pca_space(Z, cap=20):
    p = PCA(n_components=min(cap, Z.shape[1]), random_state=SEED).fit(Z)
    k = int(np.searchsorted(np.cumsum(p.explained_variance_ratio_), 0.95) + 1)
    k = max(2, min(k, cap))
    return p.transform(Z)[:, :k]


def colperm(Z, seed=SEED):
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.permutation(Z[:, j]) for j in range(Z.shape[1])])


def cluster_metrics(Z, k=K_PRIMARY):
    """silhouette, DB, bootstrap-ARI stability at k, real vs column-permuted null, with bootstrap CI."""
    out = {}
    for tag, ZZ in [("real", Z), ("null", colperm(Z))]:
        P = pca_space(ZZ)
        ref = KMeans(k, n_init=4, random_state=SEED).fit(P)
        rng = np.random.default_rng(SEED)
        sil, db, ari = [], [], []
        for b in range(B):
            s = np.unique(rng.integers(0, len(P), len(P)))
            km = KMeans(k, n_init=1, random_state=b).fit(P[s])
            ari.append(adjusted_rand_score(ref.labels_, km.predict(P)))
            if len(np.unique(km.labels_)) > 1:
                ss = s if len(s) <= 1500 else rng.choice(s, 1500, replace=False)
                lab = km.predict(P[ss])
                if len(np.unique(lab)) > 1:
                    sil.append(silhouette_score(P[ss], lab)); db.append(davies_bouldin_score(P[ss], lab))
        f = lambda a: (float(np.mean(a)), float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
        out[tag] = dict(sil=f(sil), db=f(db), ari=f(ari), dims=P.shape[1],
                        sizes=np.bincount(ref.labels_).tolist())
    out["sil_excess"] = out["real"]["sil"][0] - out["null"]["sil"][0]
    out["db_excess"] = out["null"]["db"][0] - out["real"]["db"][0]
    out["ari_excess"] = out["real"]["ari"][0] - out["null"]["ari"][0]
    return out


def k_curves(Z, ks=range(2, 11), n_seeds=8, n_ref=5):
    P = pca_space(Z); rows = []
    rng = np.random.default_rng(SEED)
    lo, hi = P.min(0), P.max(0)
    for k in ks:
        labs = [KMeans(k, n_init=1, random_state=s).fit(P).labels_ for s in range(n_seeds)]
        aris = [adjusted_rand_score(labs[a], labs[b]) for a in range(n_seeds) for b in range(a + 1, n_seeds)]
        km = KMeans(k, n_init=4, random_state=SEED).fit(P)
        ss = rng.choice(len(P), min(2000, len(P)), replace=False)
        refs = [np.log(KMeans(k, n_init=1, random_state=r).fit(rng.uniform(lo, hi, P.shape)).inertia_) for r in range(n_ref)]
        gap = np.mean(refs) - np.log(km.inertia_)
        rows.append(dict(k=k, sil=silhouette_score(P[ss], km.labels_[ss]), db=davies_bouldin_score(P, km.labels_),
                         seed_ari=np.mean(aris), gap=gap, gap_sd=np.std(refs) * np.sqrt(1 + 1 / n_ref)))
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ proxies
def proxies(Z, m, Zbg=None):
    out = {}
    cv = KFold(5, shuffle=True, random_state=SEED)
    y = np.log10(m.dur.values)
    p = cross_val_predict(KNeighborsRegressor(15), Z, y, cv=cv)
    out["logdur_R2"] = boot_ci(np.c_[y, p], stat=lambda a: r2_score(a[:, 0], a[:, 1]))
    y = np.log10(m.peak.values)
    p = cross_val_predict(KNeighborsRegressor(15), Z, y, cv=cv)
    out["logamp_R2"] = boot_ci(np.c_[y, p], stat=lambda a: r2_score(a[:, 0], a[:, 1]))
    y = (m.pmax.values > -m.pmin.values).astype(int)
    p = cross_val_predict(KNeighborsClassifier(15), Z, y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED))
    out["polarity_skill"] = skill_ci(y, p)
    # null: shuffled targets
    ysh = np.random.default_rng(SEED).permutation(np.log10(m.dur.values))
    out["null_logdur_R2"] = float(r2_score(ysh, cross_val_predict(KNeighborsRegressor(15), Z, ysh, cv=cv)))
    out["proxy"] = (out["logdur_R2"][0] + out["polarity_skill"][0]) / 2
    if Zbg is not None:
        X = np.r_[Z, Zbg]; yy = np.r_[np.ones(len(Z), int), np.zeros(len(Zbg), int)]
        out["event_vs_bg_skill"] = skill_ci(yy, _cv_pred(X, yy))
    return out


# ------------------------------------------------------------------ driver
def eval_rep(args):
    name, X_ev, X_bg, Xaug, m, split = args
    t0 = time.time()
    sc = Scaler().fit(X_ev)
    Z = sc.transform(X_ev)
    Zbg = sc.transform(X_bg) if X_bg is not None and len(X_bg) else None
    res = dict(rep=name, split=split)
    if Xaug is not None:
        Z0 = sc.transform(Xaug[:, 0]); Za = np.stack([sc.transform(Xaug[:, a]) for a in range(Xaug.shape[1])], 1)
        res["inv"] = invariance(Z0, Za)
    res["red"] = redundancy(Z)
    date = m.wall.dt.strftime("%m%d").values
    hbin = (m.wall.dt.hour.values // 6)
    res["leak_date"] = leakage(Z, date)
    res["leak_hour"] = leakage(Z, hbin)
    if "snr" in m and m.snr.notna().all():     # C's SNR tier (<20, 20-30, >=30) as a confound target
        tier = np.digitize(m.snr.values, [20, 30])
        if min(np.bincount(tier, minlength=3)[np.unique(tier)]) >= 20 and len(np.unique(tier)) > 1:
            res["leak_tier"] = leakage(Z, tier)
    if m.run.nunique() > 1:
        res["leak_run"] = leakage(Z, m.run.values)
    if "n_gaps" in m and min((m.n_gaps > 0).sum(), (m.n_gaps == 0).sum()) >= 20:
        res["leak_dropout"] = leakage(Z, (m.n_gaps.values > 0).astype(int))   # can the rep 'see' a dropout?
    res["clu"] = cluster_metrics(Z)
    if "n_gaps" in m:   # does the dropout artefact explain cluster membership?
        from sklearn.metrics import normalized_mutual_info_score as nmi
        lab = KMeans(K_PRIMARY, n_init=4, random_state=SEED).fit_predict(pca_space(Z))
        hd = (m.n_gaps.values > 0).astype(int)
        ct = pd.crosstab(lab, hd).values
        from scipy.stats import chi2_contingency
        chi2, pval = chi2_contingency(ct)[:2] if ct.shape[1] > 1 else (0.0, 1.0)
        cv = float(np.sqrt(chi2 / (ct.sum() * max(1, min(ct.shape) - 1)))) if ct.shape[1] > 1 else 0.0
        rngp = np.random.default_rng(SEED)
        null = [nmi(lab, rngp.permutation(hd)) for _ in range(B)]
        res["clu_vs_dropout"] = dict(nmi=float(nmi(lab, hd)), nmi_null_95=float(np.percentile(null, 95)),
                                     cramers_v=cv, chi2_p=float(pval), frac_dropout=float(hd.mean()))
    res["prox"] = proxies(Z, m, Zbg)
    res["sec"] = time.time() - t0
    return res


def summarize(results):
    rows = []
    for r in results:
        rows.append({"rep": r["rep"], "split": r["split"],
                     "invariance_R@1": r["inv"]["invariance_R@1"][0] if "inv" in r else np.nan,
                     "leak_date_skill": r["leak_date"]["skill"][0], "leak_hour_skill": r["leak_hour"]["skill"][0],
                     "sil_excess": r["clu"]["sil_excess"], "ari_excess": r["clu"]["ari_excess"],
                     "proxy": r["prox"]["proxy"],
                     "leak_tier_skill": r["leak_tier"]["skill"][0] if "leak_tier" in r else np.nan,
                     "leak_run_skill": r["leak_run"]["skill"][0] if "leak_run" in r else np.nan,
                     "leak_dropout_skill": r["leak_dropout"]["skill"][0] if "leak_dropout" in r else np.nan,
                     "clu_dropout_nmi": r["clu_vs_dropout"]["nmi"] if "clu_vs_dropout" in r else np.nan,
                     "clu_dropout_cramersV": r["clu_vs_dropout"]["cramers_v"] if "clu_vs_dropout" in r else np.nan, "PR_over_d": r["red"]["PR_over_d"], "PR": r["red"]["PR"][0],
                     "d": r["red"]["d"]})
    return pd.DataFrame(rows)


def rank_table(S):
    out = []
    for split, g in S.groupby("split"):
        g = g.set_index("rep").copy()
        rk = pd.DataFrame(index=g.index)
        for c, hib in CRITERIA:
            rk[c] = g[c].rank(ascending=not hib)
        g["mean_rank"] = rk.mean(1)
        for c in rk:
            g["rank_" + c] = rk[c]
        g["split"] = split
        out.append(g.reset_index())
    return pd.concat(out)


def stability(R, a="dev", b="holdout"):
    A = R[R.split == a].set_index("rep"); Bb = R[R.split == b].set_index("rep")
    common = A.index.intersection(Bb.index)
    A, Bb = A.loc[common], Bb.loc[common]
    per = {c: float(spearmanr(A[c], Bb[c])[0]) for c, _ in CRITERIA}
    comp = float(spearmanr(A["mean_rank"], Bb["mean_rank"])[0])
    top_a = list(A.sort_values("mean_rank").index[:2]); top_b = list(Bb.sort_values("mean_rank").index[:2])
    return dict(per_criterion_rho=per, composite_rho=comp, median_criterion_rho=float(np.median(list(per.values()))),
                top2_a=top_a, top2_b=top_b, same_top2=set(top_a) == set(top_b),
                stable=(set(top_a) == set(top_b)) and comp >= 0.8 and float(np.median(list(per.values()))) >= 0.8)


def main(feats=SCR + "feats.npz", metap=SCR + "meta2.parquet", tag="main", exclude_dropout=False, min_peak=0.0, min_snr=0.0):
    F = dict(np.load(feats))
    meta = pd.read_parquet(metap)
    if "orig_legacy66" not in F:     # LP-cut-off runs: legacy (raw path) identical to main run
        F0 = np.load(SCR + "feats.npz")
        assert (F0["EVAL"] == F["EVAL"]).all()
        F["orig_legacy66"] = F0["orig_legacy66"]; F["aug_legacy66"] = F0["aug_legacy66"]
    fp = metap.replace("meta2", "meta_flags")
    fp = fp if os.path.exists(fp) else SCR + "meta_flags.parquet"
    if "n_gaps" not in meta and os.path.exists(fp):
        fl = pd.read_parquet(fp)
        assert (fl.wall.values == meta.wall.values).all()
        meta["n_gaps"] = fl.n_gaps.values; meta["n_missing"] = fl.n_missing.values
    reps = build_reps(F, meta)
    augreps = build_reps(F, meta, "aug_", rows=F["EVAL"])
    EVAL = F["EVAL"]
    ev = (meta.kind == "event").values.copy(); bg = (meta.kind == "bg").values.copy()
    if exclude_dropout:      # keep only windows without any dropout (A: dropouts identify the run)
        clean = (meta.n_gaps.values == 0)
        keep_e = clean[EVAL]
        augreps = {k: v[keep_e] for k, v in augreps.items()}
        EVAL = EVAL[keep_e]
        ev &= clean; bg &= clean
    if min_snr > 0:          # C's SNR tiers (clean tier = SNR >= 30)
        hi = (meta.snr.values >= min_snr) | bg
        keep_e = hi[EVAL]; augreps = {k: v[keep_e] for k, v in augreps.items()}; EVAL = EVAL[keep_e]
        ev &= hi
    if min_peak > 0:         # large-event subset (B: small events are noise-dominated)
        big = meta.peak.values >= min_peak
        keep_e = big[EVAL]; augreps = {k: v[keep_e] for k, v in augreps.items()}; EVAL = EVAL[keep_e]
        ev &= big
    tasks = []
    for split, runmask in [("dev", meta.run.values != HOLDOUT_RUN), ("holdout", meta.run.values == HOLDOUT_RUN)]:
        me = ev & runmask; mb = bg & runmask
        eva = runmask[EVAL]
        for name, X in reps.items():
            tasks.append((name, X[me], X[mb], augreps[name][eva], meta[me].reset_index(drop=True), split))
    with Pool(int(os.environ.get('NPROC', 4))) as p:
        results = p.map(eval_rep, tasks)
    json.dump(results, open(f"eval_{tag}_full.json", "w"), indent=1, default=float)
    S = summarize(results); R = rank_table(S)
    R.to_csv(f"eval_{tag}_summary.csv", index=False)
    st = stability(R)
    json.dump(st, open(f"eval_{tag}_stability.json", "w"), indent=1)
    pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
    print(R[["split", "rep", "invariance_R@1", "leak_date_skill", "leak_hour_skill", "sil_excess", "ari_excess", "proxy", "leak_run_skill", "leak_dropout_skill", "PR", "d", "mean_rank"]].round(3).to_string())
    print(json.dumps(st, indent=1))
    return results, R, st


if __name__ == "__main__":
    if "--no-dropout" in sys.argv:
        main(tag="nodrop", exclude_dropout=True)
    elif "--C-tier" in sys.argv:
        t = float(sys.argv[sys.argv.index("--C-tier") + 1])
        main(SCR + "C_feats.npz", SCR + "C_meta2.parquet", f"C_snr{int(t)}", min_snr=t)
    elif "--large" in sys.argv:
        main(tag="large", min_peak=5000.0)
    elif "--lp" in sys.argv:
        c = sys.argv[sys.argv.index("--lp") + 1]
        main(SCR + f"lp{c}_feats.npz", SCR + f"lp{c}_meta2.parquet", f"lp{c}")
    else:
        a = sys.argv[1:]
        main(*a) if a else main()
