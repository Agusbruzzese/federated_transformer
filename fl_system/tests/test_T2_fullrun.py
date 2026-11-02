"""T2 (report 25): full-run equivalence of the deployment recipe S+F (200 rounds, X = 2.56 M, lr 1e-3, server init) between the
real system and the study's own S+F runs (fold 0, seeds 0-2). Criterion: per seed |LOBES and POLARITY difference| <= 0.01
(frozen evaluator, fold-0 evaluation days); also reports weight-level max |diff| (bit-identity expected from T1).
    .venv/bin/python fl_system/tests/test_T2_fullrun.py"""
import json, os, sys, time, torch
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.chdir(ROOT)
sys.path[:0] = [os.path.join(ROOT, "research/model/K_committee/protocol"), ROOT, os.path.join(ROOT, "research/model/M4_methods")]
import evaluator as E, evaluator2 as E2
res = {}
for s in (0, 1, 2):
    real = os.path.join(ROOT, f"data/fl_system/fold0_v2/server/runs/t2_SpF_s{s}/final.pt"); study = os.path.join(ROOT, f"data/k_committee/protocol/runs3/v21_S+F_f0_s{s}/final.pt")
    A, B = torch.load(real, map_location="cpu"), torch.load(study, map_location="cpu")
    wd = max(float((A["enc"][k] - B["enc"][k]).abs().max()) for k in A["enc"])
    ra, _ = E2.score(E.load_encoder(real), 0, do_gate=False); rb, _ = E2.score(E.load_encoder(study), 0, do_gate=False)
    agg = lambda rows, k: sum(r["value"] for r in rows if r["metric"] == k) / 3
    d = {k: agg(ra, k) - agg(rb, k) for k in ("LOBES", "POLARITY", "DURATION")}
    summ = json.load(open(real.replace("final.pt", "summary.json")))
    res[s] = dict(weight_maxabs=wd, bit_identical=wd == 0.0, score_diff=d, real_scores={k: agg(ra, k) for k in d}, wall_s=summ["wall_s"],
                  bytes_total=summ["bytes_total"])
ok = all(abs(v["score_diff"][k]) <= 0.01 for v in res.values() for k in ("LOBES", "POLARITY"))
out = dict(per_seed=res, PASS=ok, all_bit_identical=all(v["bit_identical"] for v in res.values()))
p = os.path.join(ROOT, "fl_system/tests", time.strftime("T2_%Y%m%d_%H%M%S.json")); json.dump(out, open(p, "x"), indent=1, default=str); print(json.dumps(out, default=str), "->", p)
