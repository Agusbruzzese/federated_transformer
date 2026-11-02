#!/usr/bin/env bash
# Low-SNR replication chain (report 41, pre-registered). Resumable: rerunning skips finished runs.
cd "$(dirname "$0")/../../.."
.venv/bin/python research/model/K_committee/protocol/verify_freeze.py 2>&1 | grep -v Warn | tail -3
sha256sum research/model/R_replication/rep.py research/model/R_replication/rep_manifest.json research/model/R_replication/run_rep.sh
.venv/bin/python research/model/R_replication/rep.py launch 2>&1 | grep -v Warn
n=$(.venv/bin/python -c "import json,os;m=json.load(open('research/model/R_replication/rep_manifest.json'));print(sum(not os.path.exists(f'data/k_committee/replication/runs/{e[\"id\"]}/final.pt') for e in m))")
[ "$n" = "0" ] || { echo "TRAINING INCOMPLETE: $n runs missing"; exit 1; }
[ -f data/fl_system/fold0_rep10/federation.json ] || .venv/bin/python research/model/R_replication/rep.py provision 2>&1 | grep -v Warn
for s in 0 1 2 3 4 5 6 7 8 9; do
  [ -f data/fl_system/fold0_rep10/server/dpsim/rep10_secagg_s$s/final.pt ] && continue
  rm -rf data/fl_system/fold0_rep10/server/dpsim/rep10_secagg_s$s.partial 2>/dev/null || true
  [ -d data/fl_system/fold0_rep10/server/dpsim/rep10_secagg_s$s ] && mv data/fl_system/fold0_rep10/server/dpsim/rep10_secagg_s$s data/fl_system/fold0_rep10/server/dpsim/rep10_secagg_s$s.interrupted_$(date +%H%M%S)
  .venv-fl/bin/python fl_system4/dpsim.py --fed data/fl_system/fold0_rep10/federation.json --tag rep10_secagg_s$s --seed $s --rounds 200 \
     --examples 2560000 --batch 64 --lr 1e-3 --init data/fl_system/fold0_rep10/server/server_init_s$s.pt --clipping-range 0.5 2>&1 | grep -v Warn | tail -1
done
.venv-fl/bin/python - <<'PY'
import json
for s in range(10):
    r = [json.loads(l) for l in open(f"data/fl_system/fold0_rep10/server/dpsim/rep10_secagg_s{s}/rounds.jsonl")]
    assert len(r) == 200 and sum(x["n_over_clip"] for x in r) == 0, (s, len(r))
print("SecAgg+ runs valid: 200 rounds, 0 clipped values each")
PY
[ $? = 0 ] || { echo "SECAGG RUNS INVALID"; exit 1; }
.venv/bin/python research/model/R_replication/rep.py score 12 2>&1 | grep -v Warn | tail -2
.venv/bin/python research/model/R_replication/rep.py analyze 2>&1 | grep -v Warn | tail -2
echo CHAIN_EXIT 0
