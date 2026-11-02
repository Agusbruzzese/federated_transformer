#!/usr/bin/env bash
# Training-tier study chain (report 47, pre-registered). Resumable.
cd "$(dirname "$0")/../../.."
.venv/bin/python research/model/K_committee/protocol/verify_freeze.py 2>&1 | grep -v Warn | tail -1
sha256sum research/model/R_replication/rep.py research/model/R_replication/rep4.py research/model/R_replication/rep4_manifest.json research/model/R_replication/run_rep4.sh
R=research/model/R_replication/rep4.py
.venv/bin/python $R launch 2>&1 | grep -v Warn
n=$(.venv/bin/python -c "import json,os;m=json.load(open('research/model/R_replication/rep4_manifest.json'));print(sum(not os.path.exists('data/k_committee/tier10/runs/'+e['id']+'/final.pt') for e in m))")
[ "$n" = "0" ] || { echo "TRAINING INCOMPLETE: $n runs missing"; exit 1; }
.venv/bin/python $R score 12 2>&1 | grep -v Warn | tail -1
.venv/bin/python $R analyze 2>&1 | grep -v Warn | tail -1
echo CHAIN_EXIT 0
