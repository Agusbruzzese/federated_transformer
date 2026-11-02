"""Data intake kit (report 50). One command takes a NEW recording folder through the validated pipeline and the pre-registered
external test of the frozen study models:
    .venv/bin/python -m intake.run --raw "<folder containing 'Run YYYYMMDD HHhMMmSSs - YYYYMMDD HHhMMmSSs' run folders>" --name <name>
Stages (each writes a done-marker; rerunning resumes; nothing is overwritten):
  1 integrity   chunk index + full scan of every Time/Values file (workstream A code), checks: pairing, 24-bit values,
                sampling rate, dropouts
  2 extract     the validated extraction (extraction/ package, unchanged) -> events
  3 quality     events per day, SNR tiers, dropout share, morphology drift vs the reference dataset (SNR >= 20)
  4 external    the frozen study models (central C256, federated F-nat, server-pretrained S+F; 10 seeds; trained ONLY on the
                reference data) scored on the new SNR >= 20 events with day-level cross-fitted probes; study margins and
                decision rules; validity controls C1 (C256rep vs C256 EQUIVALENT) and C3 (untrained vs C256 INFERIOR)
  5 report      INTAKE_REPORT.txt + intake.json
Output: data/intake/<name>/. --selftest: runs the kit on ONE reference run and checks the extraction is identical to data/events.
"""
from __future__ import annotations

import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):   # as extraction/run.py, but BEFORE numpy is imported:
    os.environ.setdefault(_k, "1")                                          # multi-threaded BLAS changed the last float bit (report 50)
import argparse, datetime as dt, glob, hashlib, json, sys, time
import numpy as np, pandas as pd

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROT = os.path.join(REPO, "research/model/K_committee/protocol")
STUDY_RUNS = os.path.join(REPO, "data/k_committee/protocol/runs3")
MIN_DAY_EVENTS = 30                    # a day enters the external test only with >= 30 events of SNR >= 20
KS_WARN = 0.15                         # morphology drift flag (two-sample KS statistic vs the reference)


def _bsd(out):
    """workstream A's bsd_io, redirected to this intake's own index/full-scan folder (module globals; inherited by forked workers)."""
    from extraction._vendor import bsd_io as b
    d = os.path.join(out, "A_index"); os.makedirs(d, exist_ok=True)
    b.HERE = d; b.INDEX_PATH = os.path.join(d, "chunk_index.parquet"); return b


def _done(out, stage): return os.path.exists(os.path.join(out, f".done_{stage}"))
def _mark(out, stage, info): json.dump(info, open(os.path.join(out, f".done_{stage}"), "x"), indent=1, default=str)


# ------------------------------------------------------------------ 1 integrity
def integrity(raw, out, procs):
    b = _bsd(out); sys.path.insert(0, os.path.join(REPO, "research/extraction/A_integrity")); import full_scan as FS
    FS.b = b                                                         # full_scan's own bsd_io -> the redirected one
    ix0 = b.list_chunks(root=raw, cache=False)
    assert len(ix0), f"no chunk files found under {raw} (expected 'Run ...' folders with 'S0C0 BIN Values/Time' .bsd files)"
    pairs = ix0[ix0.paired]
    from multiprocessing import Pool
    with Pool(min(procs, 8)) as p:
        res = p.map(FS.one, [(r.run, r.index, r.time_path, r.values_path) for r in pairs.itertuples()], chunksize=4)
    C = pd.DataFrame([r[0] for r in res]); G = pd.concat([r[1] for r in res], ignore_index=True) if res else pd.DataFrame()
    C.to_parquet(os.path.join(b.HERE, "full_scan_chunks.parquet")); G.to_parquet(os.path.join(b.HERE, "full_scan_gaps.parquet"))
    ix = b.list_chunks(root=raw, cache=False); ix.to_parquet(b.INDEX_PATH)      # with exact dropout counts; cached for stage 2
    info = dict(runs=sorted(ix.run.unique()), n_chunks=int(len(ix)), paired_frac=float(ix.paired.mean()),
                fs_local=[float(ix.fs_local.min()), float(ix.fs_local.max())], top_byte_max=int(C.top_byte.max()),
                missing_frac=float(C.n_missing.sum() / max(1, C.n.sum() + C.n_missing.sum())), n_nonpos_dt=int(C.n_nonpos.sum()),
                first=str(ix.wall_start.min()), last=str(ix.wall_start.max()), bytes=int(ix.values_size.sum() + ix.time_size.sum()))
    chk = dict(paired=info["paired_frac"] >= 0.99, values_24bit=info["top_byte_max"] == 0, fs_in_range=2400 <= info["fs_local"][0] and info["fs_local"][1] <= 2430,
               timestamps_monotonic=info["n_nonpos_dt"] == 0)
    info["checks"] = chk; info["PASS"] = all(chk.values()); return info


# ------------------------------------------------------------------ 2 extract
def extract(out, procs):
    _bsd(out); from extraction import run as X
    ix = pd.read_parquet(os.path.join(out, "A_index", "chunk_index.parquet")); runs = sorted(ix.run.unique())
    X.main(["--out", os.path.join(out, "events"), "--procs", str(procs), "--runs", ",".join(str(i) for i in range(len(runs)))])
    return json.load(open(os.path.join(out, "events", "manifest.json")))


def load_events(events_dir):
    """Like ssl_central.data.load_all, without the reference dataset's run split (new data has its own runs)."""
    M, W = [], []
    for f in sorted(glob.glob(os.path.join(events_dir, "run*_meta.parquet"))):
        m = pd.read_parquet(f).sort_values("row"); w = np.load(f[:-len("_meta.parquet")] + "_windows.npy")
        assert len(w) == len(m) and (m.row.to_numpy() == np.arange(len(m))).all(); M.append(m); W.append(w)
    m = pd.concat(M, ignore_index=True); m["day"] = m.date.astype(str); m["hour"] = m.wall_time.dt.hour
    return m, np.concatenate(W)


# ------------------------------------------------------------------ 3 quality
def quality(out):
    from scipy.stats import ks_2samp
    m, _ = load_events(os.path.join(out, "events")); ref, _ = load_events(os.path.join(REPO, "data/events"))
    hi, rhi = m[m.snr >= 20], ref[ref.snr >= 20]
    per_day = m.groupby("day").apply(lambda g: dict(all=int(len(g)), snr10=int((g.snr >= 10).sum()), snr20=int((g.snr >= 20).sum()),
                                                    dropout_frac=round(float(g.has_dropout.mean()), 3))).to_dict()
    feats = ["m8_log_amp", "m8_dur10", "m8_n_lobes", "m8_polarity", "m8_asym", "snr", "dur_s"]
    drift = {f: float(ks_2samp(hi[f].dropna(), rhi[f].dropna()).statistic) for f in feats if f in hi and len(hi) > 20}
    shape = dict(new=dict(polarity_pos=float((hi.m8_polarity > 0).mean()), lobes=hi.m8_n_lobes.clip(upper=4).value_counts(normalize=True).sort_index().round(3).to_dict()),
                 ref=dict(polarity_pos=float((rhi.m8_polarity > 0).mean()), lobes=rhi.m8_n_lobes.clip(upper=4).value_counts(normalize=True).sort_index().round(3).to_dict()))
    flags = [f"morphology drift in {f} (KS {v:.2f} > {KS_WARN})" for f, v in drift.items() if v > KS_WARN]
    days_ok = [d for d, v in per_day.items() if v["snr20"] >= MIN_DAY_EVENTS]
    if len(days_ok) < 2: flags.append(f"only {len(days_ok)} day(s) with >= {MIN_DAY_EVENTS} SNR>=20 events: external test impossible")
    elif len(days_ok) < 4: flags.append(f"only {len(days_ok)} usable days: external test will be imprecise")
    return dict(n_events=int(len(m)), snr_tiers=dict(lt10=int((m.snr < 10).sum()), s10_20=int(((m.snr >= 10) & (m.snr < 20)).sum()), ge20=int(len(hi))),
                per_day=per_day, ks_vs_reference_snr20=drift, shape_mix=shape, usable_days=days_ok, flags=flags)


# ------------------------------------------------------------------ 4 external
ARMS = {"C256": "v21_C256_f{f}_s{s}", "C256rep": "v21_C256rep_f{f}_s{s10}", "F-nat": "v21_F-nat_f{f}_s{s}", "S+F": "v21_S+F_f{f}_s{s}"}


def external(out, usable_days):
    """Frozen study models, never trained on the new data. Seed s uses the model of fold s % 3. Probes are cross-fitted by day:
    the usable days are split into two halves (alternating in date order); probes are fitted on one half and scored per day on
    the other, both ways. Per-day composites = mean of kNN and linear probe (the study's evaluator functions)."""
    import torch
    sys.path[:0] = [PROT, REPO, os.path.join(REPO, "research/model/M4_methods")]
    import evaluator as E, evaluator2 as E2, m4
    from ssl_central import augment as A
    m, W = load_events(os.path.join(out, "events")); k = (m.snr >= 20).to_numpy() & m.day.isin(usable_days).to_numpy()
    m, W = m[k].reset_index(drop=True), W[k]; day = m.day.to_numpy(); tg = m4.targets(m)
    with torch.no_grad():
        x = A.eval_view(torch.tensor(W, device=E.DEV), torch.tensor(m.fs_eff.to_numpy(np.float32), device=E.DEV))
    def emb(enc):
        with torch.no_grad(): return torch.cat([enc(x[i:i + 2048]) for i in range(0, len(x), 2048)]).double().cpu().numpy()
    days = sorted(usable_days); halves = [days[0::2], days[1::2]]
    def score(Z):
        out_ = {}
        for h in (0, 1):
            tr = np.isin(day, halves[1 - h]);
            for target, name in (("lobes", "LOBES"), ("polarity", "POLARITY"), ("duration", "DURATION")):
                y = tg[target]; P = {kind: E._fit_predict(kind, target, Z[tr], y[tr], Z[~tr])[0] for kind in ("knn", "lin")}
                ev_days = day[~tr]
                for d in halves[h]:
                    kk = ev_days == d; out_[d, name] = float(np.mean([m4.metric(target, y[~tr][kk], P[kind][kk]) for kind in P]))
        return out_
    S = []
    for s in range(10):
        f = s % 3
        for arm, pat in ARMS.items():
            sd = s + 10 if arm == "C256rep" else s
            sc = score(emb(E.load_encoder(os.path.join(STUDY_RUNS, pat.format(f=f, s=s, s10=s + 10), "final.pt"))))
            S += [dict(arm=arm, seed=sd, day=d, metric=n, value=v) for (d, n), v in sc.items()]
        sc = score(emb(E2.init_encoder(s))); S += [dict(arm="INIT", seed=s, day=d, metric=n, value=v) for (d, n), v in sc.items()]
    S = pd.DataFrame(S); S.to_parquet(os.path.join(out, "external_scores.parquet"))
    def Y(arm, metric, seeds): return S[(S.arm == arm) & (S.metric == metric)].pivot(index="seed", columns="day", values="value").loc[seeds, days].to_numpy()
    MAIN, REP = list(range(10)), list(range(10, 20))
    def contrast(a, b, sa=MAIN, sb=MAIN):
        r = {}
        for k_ in ("LOBES", "POLARITY", "DURATION"):
            t = E.two_stratum(Y(a, k_, sa), Y(b, k_, sb)); t["outcome"] = E.outcome(t["L"], t["U"], E.MARGIN[k_]); r[k_] = t
        return dict(metrics=r, label=E.contrast_label({k_: r[k_]["outcome"] for k_ in E.PRIMARY}))
    c1, c3 = contrast("C256rep", "C256", REP, MAIN), contrast("INIT", "C256")
    qf, sf = contrast("F-nat", "C256"), contrast("S+F", "F-nat")
    rej = E.holm([sf["metrics"][k_]["p_sup"] for k_ in E.PRIMARY])
    helps = any(r_ and sf["metrics"][k_]["delta"] >= E.MARGIN[k_] / 2 for r_, k_ in zip(rej, E.PRIMARY))
    valid = dict(C1_EQUIVALENT=c1["label"] == "EQUIVALENT", C3_INFERIOR=c3["label"] == "INFERIOR")
    return dict(days=days, halves=halves, n_events=int(len(m)), validity=valid, PROCEDURE_VALID=all(valid.values()),
                federated_vs_central=qf, pretraining_vs_federated=dict(**sf, holm_reject=dict(zip(E.PRIMARY, rej)), HELPS=helps),
                controls=dict(C1=c1, C3=c3),
                means={a: {k_: float(S[(S.arm == a) & (S.metric == k_)].value.mean()) for k_ in ("LOBES", "POLARITY", "DURATION")} for a in S.arm.unique()},
                study_reference=dict(federated_vs_central="EQUIVALENT (lobes -0.023)", pretraining="HELPS (lobes +0.049)"))


# ------------------------------------------------------------------ 5 report
def report(out, name, res):
    L = [f"INTAKE REPORT: {name}   ({dt.datetime.now():%Y-%m-%d %H:%M})", "=" * 78]
    i = res.get("integrity")
    if i: L += ["1 INTEGRITY  " + ("PASS" if i["PASS"] else "FAIL") + f"   runs {len(i['runs'])}, chunks {i['n_chunks']}, {i['first'][:16]} -> {i['last'][:16]}",
                f"   paired {i['paired_frac']:.3f}; fs {i['fs_local'][0]:.1f}-{i['fs_local'][1]:.1f} Hz; missing samples {100 * i['missing_frac']:.3f} %; checks {i['checks']}"]
    q = res.get("quality")
    if q:
        L += [f"3 QUALITY    events {q['n_events']} (SNR>=20: {q['snr_tiers']['ge20']}, 10-20: {q['snr_tiers']['s10_20']}, <10: {q['snr_tiers']['lt10']}); usable days {len(q['usable_days'])}",
              "   KS vs reference (SNR>=20): " + ", ".join(f"{k} {v:.2f}" for k, v in q["ks_vs_reference_snr20"].items())]
        L += [f"   FLAG: {f}" for f in q["flags"]] or ["   no flags"]
    e = res.get("external")
    if e:
        g = lambda c: f"{c['label']} (lobes {c['metrics']['LOBES']['delta']:+.3f} [{c['metrics']['LOBES']['L']:+.3f}, {c['metrics']['LOBES']['U']:+.3f}]," \
                      f" polarity {c['metrics']['POLARITY']['delta']:+.3f})"
        L += [f"4 EXTERNAL   frozen study models on {e['n_events']} new SNR>=20 events, {len(e['days'])} days; procedure "
              + ("VALID" if e["PROCEDURE_VALID"] else f"NOT VALID {e['validity']} -> results not interpretable"),
              f"   federated vs central:     {g(e['federated_vs_central'])}   [study: EQUIVALENT, -0.023]",
              f"   pretraining vs federated: {g(e['pretraining_vs_federated'])}, HELPS={e['pretraining_vs_federated']['HELPS']}   [study: HELPS, +0.049]"]
    if res.get("selftest"): L += [f"SELFTEST: extraction identical to data/events: {res['selftest']}"]
    p = os.path.join(out, "INTAKE_REPORT.txt"); open(p, "w").write("\n".join(L) + "\n"); print("\n".join(L)); return p


def selftest_compare(out, ref_run_label):
    m, W = load_events(os.path.join(out, "events")); f = glob.glob(os.path.join(REPO, "data/events", f"run*_{ref_run_label}_meta.parquet"))[0]
    rm = pd.read_parquet(f).sort_values("row"); rW = np.load(f[:-len("_meta.parquet")] + "_windows.npy")
    cols = [c for c in rm.columns if c not in ("run_id", "event_id", "row")]
    same_meta = all(np.array_equal(m[c].to_numpy(), rm[c].to_numpy()) or m[c].equals(rm[c]) for c in cols if c in m)
    return dict(n_new=int(len(m)), n_ref=int(len(rm)), windows_identical=bool(len(W) == len(rW) and np.array_equal(W, rW)), meta_identical=bool(same_meta))


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--raw"); ap.add_argument("--name", required=True); ap.add_argument("--procs", type=int, default=16)
    ap.add_argument("--selftest", default="", help="reference run label (e.g. 20250224): link that run only and compare the extraction")
    ap.add_argument("--skip-external", action="store_true")
    a = ap.parse_args(argv); out = os.path.join(REPO, "data/intake", a.name); os.makedirs(out, exist_ok=True)
    raw = a.raw
    if a.selftest:
        raw = os.path.join(out, "raw_link"); os.makedirs(raw, exist_ok=True)
        src = [d for d in glob.glob(os.path.join(REPO, "Spintronic*", f"Run {a.selftest}*"))][0]
        if not os.path.exists(os.path.join(raw, os.path.basename(src))): os.symlink(src, os.path.join(raw, os.path.basename(src)))
    assert raw and os.path.isdir(raw), "--raw folder required"
    res = {}
    for stage, fn in (("integrity", lambda: integrity(raw, out, a.procs)), ("extract", lambda: extract(out, a.procs)), ("quality", lambda: quality(out))):
        if not _done(out, stage):
            t0 = time.time(); r = fn(); r = r if isinstance(r, dict) else {}; r["secs"] = round(time.time() - t0); _mark(out, stage, r)
            print(f"[intake] {stage} done in {r['secs']} s", flush=True)
        res[stage] = json.load(open(os.path.join(out, f".done_{stage}")))
    if a.selftest: res["selftest"] = selftest_compare(out, a.selftest)
    if not a.skip_external and len(res["quality"]["usable_days"]) >= 2:
        if not _done(out, "external"):
            t0 = time.time(); r = external(out, res["quality"]["usable_days"]); r["secs"] = round(time.time() - t0); _mark(out, "external", r)
        res["external"] = json.load(open(os.path.join(out, ".done_external")))
    if a.selftest and res.get("external"): res["external"]["NOTE"] = "SELFTEST on reference data: in-sample, NOT evidence"
    json.dump(res, open(os.path.join(out, f"intake_{time.strftime('%Y%m%d_%H%M%S')}.json"), "x"), indent=1, default=str)
    report(out, a.name, res)


if __name__ == "__main__":
    main()
