#!/usr/bin/env bash
# M4 (report 35, pre-registered). Waits for the M3c' chain (report 34) so the GPU is not shared, then:
# V1 sim-vs-real (seeds 0-2, 200 rounds) -> calib C -> V2 real fl_system4 vs sim (3 rounds, z=1) -> curve (8 z x 3 seeds) -> analysis.
cd "$(dirname "$0")/.."
M3CP_LOG=$(ls fl_system3/M3cp_chain_*.log | head -1)
until grep -q "CHAIN_EXIT\|FAILED" "$M3CP_LOG"; do sleep 300; done
echo "M3c' chain finished: $(grep 'CHAIN_EXIT\|FAILED' $M3CP_LOG) at $(date)"
sha256sum fl_system4/dp.py fl_system4/dpsim.py fl_system4/check_M4v.py fl_system4/analyze_M4.py fl_system4/fl_system4/*.py
SIM=".venv-fl/bin/python fl_system4/dpsim.py --fed data/fl_system/fold0_v2/federation.json --batch 64 --lr 1e-3"
INIT=data/fl_system/fold0_v2/server/server_init_s
for s in 0 1 2; do $SIM --tag m4v1_s$s --seed $s --rounds 200 --examples 2560000 --init $INIT$s.pt --clipping-range 0.5 2>&1 | grep -v Warn | tail -1; done
.venv-fl/bin/python fl_system4/check_M4v.py v1 2>&1 | grep -v Warn
.venv-fl/bin/python fl_system4/check_M4v.py calib 2>&1 | grep -v Warn
C=$(.venv-fl/bin/python -c "import json,glob;print(repr(json.load(open(sorted(glob.glob('fl_system4/calib_M4_*.json'))[0]))['C']))")
CR=$(.venv-fl/bin/python -c "import sys;sys.path.insert(0,'fl_system4');import dp;print(repr(float(dp.secagg_clipping_range($C,1.0,10))))")
echo "C=$C CR(z=1)=$CR"
$SIM --tag m4v2_sim_s0 --seed 0 --rounds 3 --examples 38400 --init ${INIT}0.pt --dp-clip $C --dp-z 1.0 2>&1 | grep -v Warn | tail -1
.venv-fl/bin/python fl_system2/run_deploy.py --app fl_system4 --fed data/fl_system/fold0_v2/federation.json --run m4v2_real_s0 --seed 0 --rounds 3 \
   --examples 38400 --batch 64 --lr 1e-3 --init ${INIT}0.pt --extra-run-config "dp-clip=$C dp-z=1.0 clipping-range=$CR" | tail -1
.venv-fl/bin/python fl_system4/check_M4v.py v2 2>&1 | grep -v Warn
for z in 0 0.01 0.03 0.1 0.3 1 3 10; do for s in 0 1 2; do
  $SIM --tag m4c_z${z}_s$s --seed $s --rounds 200 --examples 2560000 --init $INIT$s.pt --dp-clip $C --dp-z $z 2>&1 | grep -v Warn | tail -1
done; done
.venv/bin/python fl_system4/analyze_M4.py 2>&1 | grep -v Warn
echo CHAIN_EXIT $?
