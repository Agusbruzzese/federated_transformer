#!/usr/bin/env bash
# M2b (report 30): refusal tests on a TLS + authentication SuperLink. Registered keys: site_00, site_01 only.
#  A good (site_00, right CA) -> must be activated ; B no TLS ; C wrong CA ; D unregistered key -> must be refused.
cd "$(dirname "$0")/.."; P=fl_system2/pki; export PATH=$PWD/.venv-fl/bin:$PATH
D=data/fl_system/m2b_$(date +%Y%m%d_%H%M%S); mkdir -p $D; : > $D/pids.txt
setsid flower-superlink --database $D/state.db --port 8190 --ssl-ca-certfile $P/ca.crt --ssl-certfile $P/server.pem --ssl-keyfile $P/server.key \
   --enable-supernode-auth > $D/superlink.log 2>&1 < /dev/null & echo $! >> $D/pids.txt
sleep 6
for s in 00 01; do flwr supernode register $P/keys/site_$s.pub fl2tls > $D/register_$s.log 2>&1 < /dev/null; echo "register site_$s rc $?"; done
NC="--superlink 127.0.0.1:9092"
setsid flower-supernode $NC --port 9450 --root-certificates $P/ca.crt --auth-supernode-private-key $P/keys/site_00 --auth-supernode-public-key $P/keys/site_00.pub > $D/A_good.log 2>&1 < /dev/null & echo $! >> $D/pids.txt
setsid flower-supernode $NC --port 9451 --insecure --auth-supernode-private-key $P/keys/site_01 --auth-supernode-public-key $P/keys/site_01.pub > $D/B_no_tls.log 2>&1 < /dev/null & echo $! >> $D/pids.txt
setsid flower-supernode $NC --port 9452 --root-certificates $P/other_ca.crt --auth-supernode-private-key $P/keys/site_01 --auth-supernode-public-key $P/keys/site_01.pub > $D/C_wrong_ca.log 2>&1 < /dev/null & echo $! >> $D/pids.txt
setsid flower-supernode $NC --port 9453 --root-certificates $P/ca.crt --auth-supernode-private-key $P/keys/site_unlisted --auth-supernode-public-key $P/keys/site_unlisted.pub > $D/D_unregistered.log 2>&1 < /dev/null & echo $! >> $D/pids.txt
sleep 45
echo "activated on superlink: $(grep -c 'Activated node' $D/superlink.log)"
for f in A_good B_no_tls C_wrong_ca D_unregistered; do
  printf "%-15s node_id=%-22s errors=%s | %s\n" $f "$(grep -oE 'SuperNode ID: [0-9]+' $D/$f.log | head -1 | cut -d' ' -f3)" "$(grep -ciE 'error|fail|denied|unauth|refus' $D/$f.log)" "$(grep -iE 'error|fail|denied|unauth|refus|retry' $D/$f.log | tail -1 | cut -c1-110)"
done
flwr supernode ls fl2tls 2>&1 | tail -6 | cut -c1-120
for p in $(cat $D/pids.txt); do kill -- -$p 2>/dev/null || kill $p 2>/dev/null; done; sleep 3; for p in $(cat $D/pids.txt); do kill -9 -- -$p 2>/dev/null; done
echo "logs: $D"
