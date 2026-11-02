"""Committee C: client partitions for the federated arms + heterogeneity tables.

    .venv/bin/python research/model/K_committee/C_federated/partition.py

Split = M4/M5 (test 02-07/13/19/25, val 02-10, SNR>=20). Two designs:
  P13  all 13 training days -> 10 clients. Rule: one client per calendar day; a day with < 60 SNR>=20 events is
       merged into its adjacent same-run TRAINING day (02-04 -> 02-05; Sat 02-08 + Sun 02-09 together; 02-24 -> 02-26).
  P11  server owns run 0218 (02-18, 02-20; excluded from every client and from the central arm's data);
       the remaining 11 days -> 10 clients. Rule: one client per calendar day; the weekend (02-08 + 02-09) is one client.
For each, an IID control: same client sizes, events assigned by a seeded random permutation (seed 0).
Writes partitions.json and heterogeneity CSVs to this folder.
"""
import os, sys, json
import numpy as np, pandas as pd
ROOT = os.getcwd(); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "research/model/M4_methods"))
import m4

HERE = os.path.dirname(os.path.abspath(__file__))
GROUPS = {
    "P13": [["2025-02-04", "2025-02-05"], ["2025-02-06"], ["2025-02-08", "2025-02-09"], ["2025-02-11"], ["2025-02-12"],
            ["2025-02-14"], ["2025-02-18"], ["2025-02-20"], ["2025-02-24", "2025-02-26"], ["2025-02-27"]],
    "P11": [["2025-02-04"], ["2025-02-05"], ["2025-02-06"], ["2025-02-08", "2025-02-09"], ["2025-02-11"], ["2025-02-12"],
            ["2025-02-14"], ["2025-02-24"], ["2025-02-26"], ["2025-02-27"]],
}
SERVER_DAYS = {"P13": [], "P11": ["2025-02-18", "2025-02-20"]}


def load():
    m, W = m4.load()
    fam = pd.read_csv(os.path.join(ROOT, "research/model/C2_lookalikes/families.csv"))
    m = m.merge(fam, on="event_id", how="left"); m["family"] = m.family.fillna(-1).astype(int)
    m["day"] = m.date.astype(str)
    return m, W


def build(m):
    tr = (m.role == "train").to_numpy(); out = {}
    for name, groups in GROUPS.items():
        used = sorted(d for g in groups for d in g)
        assert len(set(used)) == len(used)
        assert set(used) | set(SERVER_DAYS[name]) == set(m.day[tr]), name
        nat = [np.flatnonzero(tr & m.day.isin(g).to_numpy()).tolist() for g in groups]
        pool = np.concatenate([np.array(c) for c in nat]); rng = np.random.default_rng(0); perm = rng.permutation(pool)
        sizes = np.cumsum([0] + [len(c) for c in nat]); iid = [np.sort(perm[a:b]).tolist() for a, b in zip(sizes[:-1], sizes[1:])]
        srv = np.flatnonzero(tr & m.day.isin(SERVER_DAYS[name]).to_numpy()).tolist()
        out[name] = dict(groups=groups, server_days=SERVER_DAYS[name], natural=nat, iid=iid, server=srv)
    return out


def tvd(p, q): return 0.5 * np.abs(p - q).sum()


def table(m, clients, groups=None):
    fams = sorted(m.family.unique()); allidx = np.concatenate([np.array(c) for c in clients])
    pool = m.family.iloc[allidx].value_counts(normalize=True).reindex(fams, fill_value=0).to_numpy()
    rows = []
    for k, c in enumerate(clients):
        s = m.iloc[c]; p = s.family.value_counts(normalize=True).reindex(fams, fill_value=0).to_numpy()
        rows.append(dict(client=k, days="+".join(d[5:] for d in groups[k]) if groups else "iid",
                         runs="+".join(str(r) for r in sorted(s.run_id.unique())), n=len(s),
                         **{f"fam{f}": round(float(v), 3) for f, v in zip(fams, p)},
                         tvd_vs_pool=round(tvd(p, pool), 3), pos_polarity=round(float((s.m8_polarity > 0).mean()), 3),
                         sigma_med=round(float(s.sigma_ev.median()), 1), sigma_iqr=f"{s.sigma_ev.quantile(.25):.0f}-{s.sigma_ev.quantile(.75):.0f}",
                         log_amp_med=round(float(s.m8_log_amp.median()), 2), night_frac=round(float(((s.hour < 6) | (s.hour >= 22)).mean()), 3)))
    return pd.DataFrame(rows)


def main():
    m, W = load()
    tr = m.role == "train"
    per_day = m[tr].groupby(["run_id", "day"]).agg(n=("event_id", "size"), sigma_med=("sigma_ev", "median")).reset_index()
    per_day["weekday"] = pd.to_datetime(per_day.day).dt.day_name().str[:3]
    per_day.to_csv(os.path.join(HERE, "per_day_counts.csv"), index=False)
    print(per_day.to_string(index=False)); print("per run:", m[tr].groupby("run_id").size().to_dict(), "total", int(tr.sum()))
    P = build(m)
    json.dump(P, open(os.path.join(HERE, "partitions.json"), "w"))
    summ = {}
    for name, p in P.items():
        for kind in ("natural", "iid"):
            t = table(m, p[kind], p["groups"] if kind == "natural" else None)
            t.to_csv(os.path.join(HERE, f"hetero_{name}_{kind}.csv"), index=False)
            print(f"\n== {name} {kind} (server events: {len(p['server'])})"); print(t.to_string(index=False))
            n = t.n.to_numpy()
            # Kruskal-Wallis on sigma_ev across clients: noise-level heterogeneity
            from scipy.stats import kruskal, chi2_contingency
            H = kruskal(*[m.sigma_ev.iloc[c].to_numpy() for c in p[kind]])
            ct = pd.crosstab(np.concatenate([[k] * len(c) for k, c in enumerate(p[kind])]),
                             np.concatenate([m.family.iloc[c].to_numpy() for c in p[kind]]))
            chi = chi2_contingency(ct)
            summ[f"{name}_{kind}"] = dict(n_total=int(n.sum()), n_min=int(n.min()), n_max=int(n.max()), cv_n=float(n.std() / n.mean()),
                                         mean_tvd_family=float(t.tvd_vs_pool.mean()), max_tvd_family=float(t.tvd_vs_pool.max()),
                                         family_chi2=float(chi[0]), family_p=float(chi[1]), family_cramers_v=float(np.sqrt(chi[0] / (n.sum() * (min(ct.shape) - 1)))),
                                         sigma_kruskal_H=float(H.statistic), sigma_p=float(H.pvalue),
                                         clients_single_run=int(sum(len(set(m.run_id.iloc[c])) == 1 for c in p[kind])))
    json.dump(summ, open(os.path.join(HERE, "hetero_summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
