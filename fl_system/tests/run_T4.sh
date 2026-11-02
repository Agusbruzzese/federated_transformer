#!/usr/bin/env bash
# T4 sequence (report 25): clean run, fault-injected run (client 3 dies in round 2), resume of the faulty run.
cd "$(dirname "$0")/../.."
C=(--fed data/fl_system/fold0_v2/federation.json --seed 0 --rounds 3 --examples 16384 --batch 64 --lr 1e-3 --init data/fl_system/fold0_v2/server/server_init_s0.pt)
[ -n "$SKIP_A" ] || { echo "== A clean"; timeout 900 .venv-fl/bin/python fl_system/run_local.py "${C[@]}" --run t4_ref --port 8095 | tail -1; }
echo "== B fault"; timeout 900 .venv-fl/bin/python fl_system/run_local.py "${C[@]}" --run "${TAG:-t4_fault}" --port 8096 --die-client 3 --die-at-round 2 | tail -2; echo "B exit ${PIPESTATUS[0]}"
cat data/fl_system/fold0_v2/server/runs/${TAG:-t4_fault}/last_round.json; echo
grep -hE "Error|failure|stopping|dying" data/fl_system/fold0_v2/server/runs/${TAG:-t4_fault}/server.log data/fl_system/fold0_v2/server/runs/${TAG:-t4_fault}/client_03.log | tail -4
echo "== C resume"; timeout 900 .venv-fl/bin/python fl_system/run_local.py "${C[@]}" --run "${TAG:-t4_fault}" --port 8097 --resume | tail -1
grep -h "restored" data/fl_system/fold0_v2/server/runs/${TAG:-t4_fault}/client_0*_resume.log | head -3
