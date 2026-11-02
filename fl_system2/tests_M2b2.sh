#!/usr/bin/env bash
# M2b2: a plaintext SuperNode WITHOUT keys against the TLS+auth SuperLink (server-side refusal of plaintext).
cd "$(dirname "$0")/.."; P=fl_system2/pki; export PATH=$PWD/.venv-fl/bin:$PATH
D=data/fl_system/m2b2_$(date +%Y%m%d_%H%M%S); mkdir -p $D; : > $D/pids.txt
setsid flower-superlink --database $D/state.db --port 8190 --ssl-ca-certfile $P/ca.crt --ssl-certfile $P/server.pem --ssl-keyfile $P/server.key \
   --enable-supernode-auth > $D/superlink.log 2>&1 < /dev/null & echo $! >> $D/pids.txt
sleep 6
setsid flower-supernode --superlink 127.0.0.1:9092 --port 9454 --insecure > $D/B2_plaintext_nokeys.log 2>&1 < /dev/null & echo $! >> $D/pids.txt
sleep 40
echo "activated on superlink: $(grep -c 'Activated node' $D/superlink.log)"
grep -iE "error|fail|exit code|handshake" $D/B2_plaintext_nokeys.log | grep -v "retrying in" | head -3 | cut -c1-200
echo "superlink-side: $(grep -ciE 'handshake|ssl|tls' $D/superlink.log) TLS-related lines"; grep -iE "handshake|ssl" $D/superlink.log | head -2 | cut -c1-200
for p in $(cat $D/pids.txt); do kill -- -$p 2>/dev/null || kill $p 2>/dev/null; done; sleep 3; for p in $(cat $D/pids.txt); do kill -9 -- -$p 2>/dev/null; done; echo "logs: $D"
