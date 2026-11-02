"""Reviewer E: does fedsim.py implement FedAvg (every client starts the round from the SAME global weights)?
fedsim.load_vec uses torch.nn.utils.vector_to_parameters, which makes each parameter a VIEW of the global vector;
client optimiser steps then mutate the global vector in place. This test runs fedsim.run unchanged (1 round,
10 P11 clients, 1 local step each), records the global vector each client is loaded from, and repeats with a
copying load_vec. Outputs go to E_review/_runs only.
    .venv/bin/python research/model/K_committee/E_review/test_fedavg_alias.py
"""
import os, sys, json, hashlib, torch
ROOT = os.getcwd(); sys.path.insert(0, os.path.join(ROOT, "research/model/K_committee/C_federated"))
import fedsim as fs
HERE = os.path.dirname(os.path.abspath(__file__)); fs.OUT = os.path.join(HERE, "_runs")   # symlinked pools, outputs here
orig = fs.load_vec; seen = []
def rec_alias(v, enc, head):
    seen.append(float(v.double().norm())); orig(v, enc, head)
def rec_copy(v, enc, head):
    seen.append(float(v.double().norm())); torch.nn.utils.vector_to_parameters(v.clone(), fs.params(enc, head))
res = {}
for tag, fn in (("alias_as_shipped", rec_alias), ("copy_fixed", rec_copy)):
    seen.clear(); fs.load_vec = fn
    a = fs.parser().parse_args(["run", "--arm", "fed", "--part", "P11", "--kind", "natural", "--seed", "0", "--rounds", "1",
                                "--examples", "320", "--batch", "64", "--eval-every", "0", "--ref", "", "--tag", tag])
    summ, enc, head = fs.run(a)
    res[tag] = dict(start_norms_per_client=seen[:10], distinct_start_points=len(set(round(x, 9) for x in seen[:10])),
                    final_vec=fs.vec(enc, head))
d = float((res["alias_as_shipped"]["final_vec"] - res["copy_fixed"]["final_vec"]).abs().max())
out = {k: {kk: vv for kk, vv in v.items() if kk != "final_vec"} for k, v in res.items()}
out["max_abs_diff_final_global_alias_vs_copy"] = d
json.dump(out, open(os.path.join(HERE, "test_fedavg_alias.json"), "w"), indent=1); print(json.dumps(out, indent=1))
