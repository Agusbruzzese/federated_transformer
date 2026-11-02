#!/usr/bin/env bash
# Pretraining diagnostic chain (report 43, pre-registered). Resumable.
cd "$(dirname "$0")/../../.."
.venv/bin/python research/model/K_committee/protocol/verify_freeze.py 2>&1 | grep -v Warn | tail -1
sha256sum research/model/R_replication/rep.py research/model/R_replication/rep2.py research/model/R_replication/run_rep2.sh
R=research/model/R_replication/rep2.py
.venv/bin/python $R tune 2>&1 | grep -v Warn; [ -f research/model/R_replication/rep2_tuned.json ] || { echo "TUNING FAILED"; exit 1; }
[ -f research/model/R_replication/rep2_manifest.json ] || .venv/bin/python $R plan 2>&1 | grep -v Warn
.venv/bin/python $R launch 2>&1 | grep -v Warn
n=$(.venv/bin/python -c "import json,os;m=json.load(open('research/model/R_replication/rep2_manifest.json'))['runs'];print(sum(not os.path.exists('data/k_committee/replication/runs/'+e['id']+'/final.pt') for e in m))")
[ "$n" = "0" ] || { echo "TRAINING INCOMPLETE: $n runs missing"; exit 1; }
.venv/bin/python $R score 12 2>&1 | grep -v Warn | tail -1
.venv/bin/python $R analyze 2>&1 | grep -v Warn | tail -1
echo CHAIN_EXIT 0
