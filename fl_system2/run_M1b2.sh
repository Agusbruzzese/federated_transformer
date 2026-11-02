#!/usr/bin/env bash
# M1b (report 30): full deployment recipe S+F (200 rounds, seed 0) on Flower's current runtime, then comparison with the study model.
cd "$(dirname "$0")/.."
.venv-fl/bin/python fl_system2/run_deploy.py --fed data/fl_system/fold0_v2/federation.json --run m1b2_SpF_s0 --seed 0 --rounds 200 \
   --examples 2560000 --batch 64 --lr 1e-3 --init data/fl_system/fold0_v2/server/server_init_s0.pt | tail -1
.venv/bin/python - <<'PY'
import json, os, sys, time, torch
R = os.getcwd(); sys.path[:0] = [R + "/research/model/K_committee/protocol", R, R + "/research/model/M4_methods"]
import evaluator as E, evaluator2 as E2
run = R + "/data/fl_system/fold0_v2/server/runs2/m1b2_SpF_s0"; study = R + "/data/k_committee/protocol/runs3/v21_S+F_f0_s0/final.pt"
A, B = torch.load(run + "/final.pt", map_location="cpu"), torch.load(study, map_location="cpu")
wd = max(float((A[m][k] - B[m][k]).abs().max()) for m in ("enc", "head") for k in A[m])
ra, _ = E2.score(E.load_encoder(run + "/final.pt"), 0, do_gate=False); rb, _ = E2.score(E.load_encoder(study), 0, do_gate=False)
agg = lambda rows, k: sum(r["value"] for r in rows if r["metric"] == k) / 3
diff = {k: agg(ra, k) - agg(rb, k) for k in ("LOBES", "POLARITY", "DURATION")}
recs = [json.loads(l) for l in open(run + "/rounds.jsonl")]; secs = [r["secs"] for r in recs]
res = dict(weight_maxabs=wd, bit_identical=wd == 0.0, score_diff=diff, M1b_PASS=all(abs(diff[k]) <= 0.01 for k in ("LOBES", "POLARITY")),
           M1e=dict(rounds=len(recs), retries=sum(r.get("retries", 0) for r in recs), round_secs_median=sorted(secs)[len(secs) // 2], round_secs_max=max(secs), total_min=round(sum(secs) / 60, 1),
                    bytes_total=recs[-1]["bytes_total"]))
out = time.strftime(R + "/fl_system2/tests_M1b2_%Y%m%d_%H%M%S.json"); json.dump(res, open(out, "x"), indent=1); print(json.dumps(res), "->", out)
PY
echo CHAIN_EXIT 0
