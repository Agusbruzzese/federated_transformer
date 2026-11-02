"""Step 6 freeze check. Run at the start of every later step; exits non-zero if any frozen file differs.
    .venv/bin/python research/model/K_committee/protocol/verify_freeze.py [--write]   (--write: create the record once)"""
import hashlib, json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "../../../.."))
REC = os.path.join(HERE, "freeze_v2_1.json")
FILES = [f"research/model/K_committee/protocol/{f}" for f in ("fedsim3.py", "evaluator.py", "evaluator2.py", "analyze.py", "score_all.py",
         "tune.py", "launch.py", "manifest.py", "manifest_v2_1.json", "folds.py", "folds.json", "noisepool_v2.py")] + \
        ["data/k_committee/protocol/bgpool_day_v2.npy", "data/k_committee/protocol/bgpool_day_v2_meta.parquet",
         "reports/09_2026-10-10_protocol_v2_central_vs_federated.txt", "reports/14_2026-10-11_protocol_v2_1_amendment.txt",
         "reports/16_2026-10-11_step5c_confirmation_fixes.txt", "reports/17_2026-10-11_step6_freeze.txt"]
sha = lambda p: hashlib.sha256(open(os.path.join(ROOT, p), "rb").read()).hexdigest()
if "--write" in sys.argv:
    json.dump(dict(frozen_at="2026-10-11", protocol="v2.1 (reports 09 + 14 + 17 sec. 3)", files={p: sha(p) for p in FILES}), open(REC, "x"), indent=1)
    print("written", REC); sys.exit(0)
rec = json.load(open(REC))["files"]; bad = [p for p, h in rec.items() if sha(p) != h]
print("FREEZE OK" if not bad else f"FREEZE BROKEN: {bad}"); sys.exit(1 if bad else 0)
