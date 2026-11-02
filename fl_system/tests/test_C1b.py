"""C1b (report 28): full 200-round S+F seed 0 in containers vs the study's v21_S+F_f0_s0; C4 network from the server samples."""
import json, os, sys, time, torch
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); os.chdir(ROOT)
sys.path[:0] = [os.path.join(ROOT, "research/model/K_committee/protocol"), ROOT, os.path.join(ROOT, "research/model/M4_methods")]
import evaluator as E, evaluator2 as E2
run = os.path.join(ROOT, "data/fl_system/fold0_v2/server/runs/c1b_SpF_s0"); study = os.path.join(ROOT, "data/k_committee/protocol/runs3/v21_S+F_f0_s0/final.pt")
A, B = torch.load(os.path.join(run, "final.pt"), map_location="cpu"), torch.load(study, map_location="cpu")
wd = max(float((A[m][k] - B[m][k]).abs().max()) for m in ("enc", "head") for k in A[m])
ra, _ = E2.score(E.load_encoder(os.path.join(run, "final.pt")), 0, do_gate=False); rb, _ = E2.score(E.load_encoder(study), 0, do_gate=False)
agg = lambda rows, k: sum(r["value"] for r in rows if r["metric"] == k) / 3
diff = {k: agg(ra, k) - agg(rb, k) for k in ("LOBES", "POLARITY", "DURATION")}
cj = json.load(open(os.path.join(run, "containers", "run.json"))); s = [x for x in cj["net_samples"] if x["last_round"] > 0]; last = s[-1]
per_round = (last["rx"] + last["tx"]) / last["last_round"]; theory = 2 * 10 * 215744 * 4
res = dict(weight_maxabs=wd, bit_identical=wd == 0.0, score_diff=diff, C1b_PASS=all(abs(diff[k]) <= 0.01 for k in ("LOBES", "POLARITY")),
           C4=dict(distinct_ips=len(set(cj["ips"].values())) == len(cj["ips"]), ips=cj["ips"], sampled_round=last["last_round"],
                   server_bytes_rx_tx=[last["rx"], last["tx"]], bytes_per_round_measured=round(per_round), bytes_per_round_payload=theory,
                   overhead=round(per_round / theory - 1, 4)), wall_s=cj["wall_s"])
out = os.path.join(ROOT, "fl_system/tests", time.strftime("C1b_C4_%Y%m%d_%H%M%S.json")); json.dump(res, open(out, "x"), indent=1); print(json.dumps(res), "->", out)
