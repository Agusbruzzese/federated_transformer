"""Release test verdicts (report 38, pre-registered). Reads the chain's outputs; writes fl_system5/results_rel_<time>.json."""
import glob, json, os, statistics as st, time
import torch
R = os.getcwd(); S = R + "/data/fl_system/fold0_v2/server"
def g(a, b): return float((torch.load(a) - torch.load(b)).abs().max())
def sd(a, b):
    A, B = torch.load(a, map_location="cpu"), torch.load(b, map_location="cpu"); return max(float((A[m][k] - B[m][k]).abs().max()) for m in A for k in A[m])
res = {}
# R1 exactness: full stack vs fl_system3 local SecAgg+ run m3c_SpF_s0
run, ref = f"{S}/runs3/rel_SpF_s0", f"{S}/runs3/m3c_SpF_s0"
d = {f"r{r:03d}": g(f"{run}/global_r{r:03d}.pt", f"{ref}/global_r{r:03d}.pt") for r in (1, 2, 3, 50, 100, 150, 200)}; d["final"] = sd(run + "/final.pt", ref + "/final.pt")
res["R1"] = dict(diffs=d, PASS=all(v == 0.0 for v in d.values()))
# R2 isolation (listing taken inside each container during R1)
J = json.load(open(f"{run}/stack/stack.json")); iso = J["isolation"]
ok = {k: iso[str(k)]["fold"] == [f"client_{k:02d}"] and sorted(iso[str(k)]["keys"]) == [f"site_{k:02d}", f"site_{k:02d}.pub"] and sorted(iso[str(k)]["pki"]) == ["ca.crt", "keys"] for k in range(10)}
s = iso["superlink"]; ok["superlink"] = (s["fold"] == ["federation.json", "server"] and all(x.endswith(".pub") for x in s["keys"])
                                          and sorted(s["pki"]) == ["ca.crt", "keys", "server.key", "server.pem"])
res["R2"] = dict(per_container=ok, PASS=all(ok.values()))
# R3 intruders
I = json.load(open(f"{S}/runs3/rel_intr/stack_intruders/stack.json"))
res["R3"] = dict(containers={n: (c["status"], c["exit_code"]) for n, c in I["containers"].items()}, online_count=I["supernode_list"].count("online"), PASS=bool(I["PASS"]))
# R4 fault + resume vs the exact simulator (5-round schedule)
F0 = json.load(open(f"{S}/runs3/rel_fault/stack/stack.json")); F1 = json.load(open(f"{S}/runs3/rel_fault/stack_resume/stack.json"))
recs = [json.loads(l) for l in open(f"{S}/runs3/rel_fault/rounds.jsonl")]
d4 = {f"r{r:03d}": g(f"{S}/runs3/rel_fault/global_r{r:03d}.pt", f"{S}/dpsim/rel_ref5b_s0/global_r{r:03d}.pt") for r in (1, 2, 3, 5)}
d4["final"] = sd(f"{S}/runs3/rel_fault/final.pt", f"{S}/dpsim/rel_ref5b_s0/final.pt")
log0 = open(f"{S}/runs3/rel_fault/stack/flwr_run.log").read()
res["R4"] = dict(killed=F0["killed"], first_attempt_finished=F0["finished"], stopped_by_guard="no skipped round" in log0,
                 rounds_committed_in_order=[r["round"] for r in recs] == sorted(set(r["round"] for r in recs)), resumed_finished=F1["finished"], diffs=d4)
res["R4"]["PASS"] = (F0["killed"] is not None and not F0["finished"] and res["R4"]["stopped_by_guard"] and F1["finished"]
                     and res["R4"]["rounds_committed_in_order"] and all(v == 0.0 for v in d4.values()))
# R5 traffic and time (reported)
smp = [x for x in J["traffic_samples"] if x["round"] > 0]; per = {str(k): dict(down=[], up=[]) for k in range(10)}
for a, b in zip(smp, smp[1:]):
    if b["round"] == a["round"] + 1:
        for k in per:
            per[k]["down"].append((b["c"][k][0] - a["c"][k][0]) / 1e6); per[k]["up"].append((b["c"][k][1] - a["c"][k][1]) / 1e6)
rr = [json.loads(l) for l in open(f"{run}/rounds.jsonl")]
res["R5"] = dict(MB_per_round_median={k: dict(down=round(st.median(v["down"]), 3), up=round(st.median(v["up"]), 3)) for k, v in per.items()},
                 MB_per_run_site_total={k: round(sum(v["down"]) + sum(v["up"]), 1) for k, v in per.items()}, rounds_measured=len(per["0"]["down"]),
                 round_secs_median=st.median(r["secs"] for r in rr), round_secs_m3c_local=st.median(json.loads(l)["secs"] for l in open(f"{ref}/rounds.jsonl")),
                 wall_s=J["wall_s"], model_payload_MB_float32=215744 * 4 / 1e6, netem=J["netem_check"])
res["RELEASE_PASS"] = all(res[k]["PASS"] for k in ("R1", "R2", "R3", "R4"))
p = time.strftime(R + "/fl_system5/results_rel_%Y%m%d_%H%M%S.json"); json.dump(res, open(p, "x"), indent=1)
print(json.dumps({k: (v.get("PASS") if isinstance(v, dict) else v) for k, v in res.items()}), "->", p)
