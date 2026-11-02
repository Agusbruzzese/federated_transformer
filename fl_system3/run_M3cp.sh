#!/usr/bin/env bash
# M3c' (report 34, pre-registered): SecAgg+ S+F full runs, seeds 3-9, then the frozen analysis analyze_M3cp.py.
cd "$(dirname "$0")/.."
sha256sum fl_system3/analyze_M3cp.py
for s in 3 4 5 6 7 8 9; do
  run=m3cp_SpF_s$s; fin=data/fl_system/fold0_v2/server/runs3/$run/final.pt
  .venv-fl/bin/python fl_system2/run_deploy.py --app fl_system3 --fed data/fl_system/fold0_v2/federation.json --run $run --seed $s \
     --rounds 200 --examples 2560000 --batch 64 --lr 1e-3 --init data/fl_system/fold0_v2/server/server_init_s$s.pt | tail -1
  for t in 1 2; do   # a stopped run (strict server) is resumed exactly from its last committed round (proven M1d/C3)
    [ -f $fin ] && break; echo "resume $run attempt $t"; sleep 20
    .venv-fl/bin/python fl_system2/run_deploy.py --app fl_system3 --fed data/fl_system/fold0_v2/federation.json --run $run --seed $s \
       --rounds 200 --examples 2560000 --batch 64 --lr 1e-3 --init data/fl_system/fold0_v2/server/server_init_s$s.pt --resume | tail -1
  done
  [ -f $fin ] || { echo "RUN $run FAILED"; exit 1; }
done
.venv/bin/python fl_system3/analyze_M3cp.py 2>&1 | grep -v Warning
echo CHAIN_EXIT $?
