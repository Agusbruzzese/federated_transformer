#!/usr/bin/env bash
cd "$(dirname "$0")/../.."
.venv-fl/bin/python fl_system/containers/run_compose.py --fed data/fl_system/fold0_v2/federation.json --run c1b_SpF_s0 --seed 0 --rounds 200 \
   --examples 2560000 --batch 64 --lr 1e-3 --init server_init_s0.pt | tail -1
.venv/bin/python fl_system/tests/test_C1b.py
echo CHAIN_EXIT 0
