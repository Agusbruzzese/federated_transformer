#!/usr/bin/env bash
# M2 (report 30): TLS + SuperNode authentication.
#  M2a: 3-round TLS+auth run (bit-identity vs plaintext m1a4 checked afterwards) ; M2c: loopback bytes for TLS vs a plaintext rerun
cd "$(dirname "$0")/.."; P=fl_system2/pki
C=(--fed data/fl_system/fold0_v2/federation.json --seed 0 --rounds 3 --examples 16384 --batch 64 --lr 1e-3 --init data/fl_system/fold0_v2/server/server_init_s0.pt)
TLS_SL="--ssl-ca-certfile $P/ca.crt --ssl-certfile $P/server.pem --ssl-keyfile $P/server.key --enable-supernode-auth"
TLS_SN="--root-certificates $P/ca.crt --auth-supernode-private-key $P/keys/site_{k} --auth-supernode-public-key $P/keys/site_{k}.pub"
lo() { cat /sys/class/net/lo/statistics/rx_bytes; }
echo "== M2a TLS+auth"; b0=$(lo)
timeout 1200 .venv-fl/bin/python fl_system2/run_deploy.py "${C[@]}" --run m2a_tls --conn fl2tls --superlink-args "$TLS_SL" --supernode-args "$TLS_SN" --register-keys "$P/keys/site_{k}.pub" | tail -1
b1=$(lo); echo "lo_bytes_tls $((b1-b0))"
echo "== M2c plaintext rerun"; b0=$(lo)
timeout 1200 .venv-fl/bin/python fl_system2/run_deploy.py "${C[@]}" --run m2c_plain | tail -1
b1=$(lo); echo "lo_bytes_plain $((b1-b0))"
