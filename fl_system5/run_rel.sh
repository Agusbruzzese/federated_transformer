#!/usr/bin/env bash
# Release test chain (report 38, pre-registered): R3 intruders -> R4 fault+resume (5-round schedule) -> R1/R2/R5 full 200-round run -> verdicts.
cd "$(dirname "$0")/.."
sha256sum fl_system5/run_stack.py fl_system5/check_rel.py fl_system5/app/fl_system5/*.py fl_system5/app/pyproject.toml
RS="timeout 21600 sg docker -c"
$RS ".venv-fl/bin/python fl_system5/run_stack.py --mode intruders --run rel_intr" | tail -1
.venv-fl/bin/python fl_system4/dpsim.py --fed data/fl_system/fold0_v2/federation.json --tag rel_ref5b_s0 --seed 0 --rounds 5 --examples 64000 --batch 64 --lr 1e-3 \
   --init data/fl_system/fold0_v2/server/server_init_s0.pt --clipping-range 0.5 2>&1 | grep -v Warn | tail -1
$RS ".venv-fl/bin/python fl_system5/run_stack.py --run rel_fault --rounds 5 --examples 64000 --kill-site 3 --kill-at-round 3 --stall-min 10" | tail -1
sleep 20
$RS ".venv-fl/bin/python fl_system5/run_stack.py --run rel_fault --rounds 5 --examples 64000 --resume" | tail -1
sleep 20
$RS ".venv-fl/bin/python fl_system5/run_stack.py --run rel_SpF_s0 --seed 0 --rounds 200 --examples 2560000" | tail -1
.venv-fl/bin/python fl_system5/check_rel.py 2>&1 | grep -v Warn
echo CHAIN_EXIT $?
