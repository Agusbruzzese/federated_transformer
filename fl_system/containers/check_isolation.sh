#!/usr/bin/env bash
# C2 (report 28): from INSIDE a client container and the server container (same mounts as run_compose.py), list what data is
# visible and try to open other sites' / study files. Expected: only the own folder exists.
R=/home/augustin/Documents/federated_transformer; F=$R/data/fl_system/fold0_v2; IMG=fl-base:2026-10-13
RO="-v $R/.venv-fl:$R/.venv-fl:ro -v $R/.venv:$R/.venv:ro -v $R/fl_system:$R/fl_system:ro -v $R/ssl_central:$R/ssl_central:ro"
PROBE='import os,json,sys
R="/home/augustin/Documents/federated_transformer"; F=R+"/data/fl_system/fold0_v2"
vis=[os.path.join(dp,f)[len(R)+1:] for dp,_,fs in os.walk(R+"/data") for f in fs][:50]
targets=[F+"/client_00/events.npz",F+"/client_07/events.npz",F+"/server/server_init_s0.pt",F+"/federation.json",
 R+"/data/k_committee/protocol/bgpool_day_v2.npy",R+"/data/events/run0_20250204_windows.npy",R+"/data/ssl_central/bgpool.npy",R+"/research/RECAP.md",R+"/reports/24_2026-10-12_final_report_central_vs_federated.txt"]
res={}
for t in targets:
    try: open(t,"rb").read(16); res[t[len(R)+1:]]="READABLE"
    except Exception as e: res[t[len(R)+1:]]=type(e).__name__
print(json.dumps(dict(role=sys.argv[1], visible_data_files=sorted(vis), probes=res)))'
echo "== client_03 container"
docker run --rm --user 1000:1000 --network none $RO -v $F/client_03:$F/client_03:rw $IMG $R/.venv-fl/bin/python -c "$PROBE" client_03
echo "== server container"
docker run --rm --user 1000:1000 --network none $RO -v $F/federation.json:$F/federation.json:ro -v $F/server:$F/server:rw $IMG $R/.venv-fl/bin/python -c "$PROBE" server
