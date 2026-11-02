#!/usr/bin/env bash
# M3c (report 30): full deployment recipe S+F with SecAgg+ (200 rounds, seeds 0-2), then comparison with the study's S+F models.
cd "$(dirname "$0")/.."
for s in 0 1 2; do
  .venv-fl/bin/python fl_system2/run_deploy.py --app fl_system3 --fed data/fl_system/fold0_v2/federation.json --run m3c_SpF_s$s --seed $s \
     --rounds 200 --examples 2560000 --batch 64 --lr 1e-3 --init data/fl_system/fold0_v2/server/server_init_s$s.pt | tail -1
done
.venv/bin/python - <<'PY'
import json, os, sys, time, torch
R = os.getcwd(); sys.path[:0] = [R + "/research/model/K_committee/protocol", R, R + "/research/model/M4_methods"]
import evaluator as E, evaluator2 as E2
agg = lambda rows, k: sum(r["value"] for r in rows if r["metric"] == k) / 3
res = {}
for s in (0, 1, 2):
    run = f"{R}/data/fl_system/fold0_v2/server/runs3/m3c_SpF_s{s}"; study = f"{R}/data/k_committee/protocol/runs3/v21_S+F_f0_s{s}/final.pt"
    A, B = torch.load(run + "/final.pt", map_location="cpu"), torch.load(study, map_location="cpu")
    wd = max(float((A[m][k] - B[m][k]).abs().max()) for m in ("enc", "head") for k in A[m])
    ra, _ = E2.score(E.load_encoder(run + "/final.pt"), 0, do_gate=False); rb, _ = E2.score(E.load_encoder(study), 0, do_gate=False)
    recs = [json.loads(l) for l in open(run + "/rounds.jsonl")]
    res[s] = dict(weight_maxabs_vs_study=wd, score_diff={k: agg(ra, k) - agg(rb, k) for k in ("LOBES", "POLARITY", "DURATION")},
                  secagg_scores={k: agg(ra, k) for k in ("LOBES", "POLARITY", "DURATION")}, rounds=len(recs),
                  round_secs_median=sorted(r["secs"] for r in recs)[len(recs) // 2], over_clip_total=sum(r["n_over_clip"] for r in recs),
                  max_abs_update=max(r["max_abs_update"] for r in recs))
out = dict(per_seed=res, M3c_PASS=all(abs(v["score_diff"][k]) <= 0.01 for v in res.values() for k in ("LOBES", "POLARITY")))
p = time.strftime(R + "/fl_system3/tests_M3c_%Y%m%d_%H%M%S.json"); json.dump(out, open(p, "x"), indent=1); print(json.dumps(out), "->", p)
PY
echo CHAIN_EXIT 0
