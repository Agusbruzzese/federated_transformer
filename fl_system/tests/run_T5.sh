#!/usr/bin/env bash
# T5 (report 25): measured wire bytes (loopback interface counters) for a 3-round run vs the theoretical payload.
cd "$(dirname "$0")/../.."
rx0=$(cat /sys/class/net/lo/statistics/rx_bytes)
.venv-fl/bin/python fl_system/run_local.py --fed data/fl_system/fold0_v2/federation.json --run t5_comms --seed 0 --rounds 3 --examples 16384 \
   --batch 64 --lr 1e-3 --init data/fl_system/fold0_v2/server/server_init_s0.pt --port 8099 | tail -1
rx1=$(cat /sys/class/net/lo/statistics/rx_bytes)
payload=$(python3 -c "print(3*2*10*215744*4 + 10*215744*4)")   # 3 rounds x (down+up) x 10 clients + initial parameter download
echo "{\"loopback_bytes\": $((rx1-rx0)), \"theoretical_payload_bytes\": $payload, \"overhead_ratio\": $(python3 -c "print(round(($rx1-$rx0)/$payload,4))")}"
