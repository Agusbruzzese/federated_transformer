#!/bin/sh
# Phase G reproduction (from the repo root). Training shards run 3-way parallel on one GPU.
PY=.venv/bin/python; L=research/model/G_loro/logs
$PY research/model/G_loro/hashes.py $L/hashes_at_freeze.txt
for i in 0 1 2; do $PY -m ssl_central_v3.train --jobs $L/jobs_grid.json --shard $i --nshard 3 & done; wait
$PY -m ssl_central_v3.select
PYTHONPATH=. $PY research/model/G_loro/make_jobs_final.py
for i in 0 1 2; do $PY -m ssl_central_v3.train --jobs $L/jobs_final.json --shard $i --nshard 3 & done; wait
$PY -m ssl_central_v3.embed
$PY -m ssl_central_v3.evaluate sanity p1 p2 verdict p2ws tiers p3p4 hgb
$PY research/model/G_loro/hashes.py $L/hashes_at_end.txt
