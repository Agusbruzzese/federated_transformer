#!/usr/bin/env bash
# C3 (report 28): clean container run; container run with `docker kill client_03` during round 2; resume; compare.
cd "$(dirname "$0")/../.."
C=(--fed data/fl_system/fold0_v2/federation.json --seed 0 --rounds 3 --examples 163840 --batch 64 --lr 1e-3 --init server_init_s0.pt)
echo "== clean";  .venv-fl/bin/python fl_system/containers/run_compose.py "${C[@]}" --run "${P:-c3}_ref" | tail -1
echo "== kill";   .venv-fl/bin/python fl_system/containers/run_compose.py "${C[@]}" --run "${P:-c3}_kill" --kill-client 3 --kill-at-round 2 | tail -2
cat data/fl_system/fold0_v2/server/runs/${P:-c3}_kill/last_round.json; echo
grep -h "RuntimeError" data/fl_system/fold0_v2/server/runs/${P:-c3}_kill/containers/flserver.log | tail -1
echo "== resume"; .venv-fl/bin/python fl_system/containers/run_compose.py "${C[@]}" --run "${P:-c3}_kill" --resume | tail -1
