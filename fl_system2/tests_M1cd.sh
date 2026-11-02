#!/usr/bin/env bash
# M1c (trace) and M1d (SuperNode killed in round 2, then resume) on fl_system2 (report 30)
cd "$(dirname "$0")/.."
C=(--fed data/fl_system/fold0_v2/federation.json --seed 0 --rounds 3 --examples 16384 --batch 64 --lr 1e-3 --init data/fl_system/fold0_v2/server/server_init_s0.pt)
[ -n "$SKIP_C" ] || { echo "== M1c traced"; timeout 900 .venv-fl/bin/python fl_system2/run_deploy.py "${C[@]}" --run m1c --trace | tail -1; }
echo "== M1d kill";    timeout 900 .venv-fl/bin/python fl_system2/run_deploy.py "${C[@]}" --run m1d2 --kill-site 3 --kill-at-round 2 | tail -2
cat data/fl_system/fold0_v2/server/runs2/m1d2/last_round.json; echo
grep -h "RuntimeError: round" data/fl_system/fold0_v2/server/runs2/m1d2/deploy/flwr_run.log | tail -1 | cut -c1-250
echo "== M1d resume";  timeout 900 .venv-fl/bin/python fl_system2/run_deploy.py "${C[@]}" --run m1d2 --resume | tail -1
