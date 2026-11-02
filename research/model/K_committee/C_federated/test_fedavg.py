"""Control C0 (protocol): proofs that fedsim implements FedAvg for K > 1. Run after any change to fedsim.py.
  C0a  every client starts each round from bit-identical weights (hash of the loaded parameters)
  C0b  order invariance: visiting clients in reverse order gives bit-identical per-client results and global
       weights; this holds only if no state leaks from one client to the next
  C0c  the global vector equals the example-weighted mean of the client vectors
    .venv/bin/python research/model/K_committee/C_federated/test_fedavg.py
"""
import hashlib, json, os, sys, torch
ROOT = os.getcwd(); HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import fedsim as fs
fs.OUT = os.path.join(ROOT, "research/model/K_committee/E_review/_runs")      # pools symlinked there; outputs scratch
H = lambda t: hashlib.sha256(t.detach().double().cpu().numpy().tobytes()).hexdigest()[:16]
orig_load, orig_vec = fs.load_vec, fs.vec
def go(reverse, rounds=2):
    starts, ends, svecs = [], [], []
    def load(v, enc, head):
        orig_load(v, enc, head); x = orig_vec(enc, head); starts.append(H(x)); svecs.append(x)
    def vec(enc, head):
        out = orig_vec(enc, head); ends.append(out); return out
    fs.load_vec, fs.vec = load, vec
    args = ["run", "--arm", "fed", "--part", "P11", "--kind", "natural", "--seed", "0", "--rounds", str(rounds),
            "--examples", str(640 * rounds), "--batch", "64", "--eval-every", "0", "--ref", "", "--tag", f"c0_rev{int(reverse)}"]
    if reverse: args.append("--reverse-clients")
    summ, enc, head = fs.run(fs.parser().parse_args(args))
    fs.load_vec, fs.vec = orig_load, orig_vec
    return starts, ends, svecs, orig_vec(enc, head), summ["plan"]
s0, e0, v0, g0, plan = go(False); s1, e1, _, g1, _ = go(True)
K = plan["K"]; w = torch.tensor(plan["b"], dtype=torch.float64) * torch.tensor(plan["s"], dtype=torch.float64); w = w / w.sum()
res = {}
# C0a: loads per round = K (clients) ; within each round all start hashes equal
res["C0a_identical_starts_per_round"] = [len(set(s0[r * K:(r + 1) * K])) == 1 for r in range(2)]
# client end vectors: e0 = [gvec init] + per round K client vecs (visiting order) + final
cl0 = e0[1:1 + K]; cl1 = e1[1:1 + K][::-1]                          # reverse run visited K-1..0
res["C0b_client_vectors_bit_identical_under_reordering"] = all(torch.equal(a, b) for a, b in zip(cl0, cl1))
res["C0b_global_bit_identical_under_reordering"] = bool(torch.equal(g0, g1))
mean = sum(w[k] * cl0[k].double() for k in range(K))
d = float((mean - v0[K].double()).abs().max())                      # v0[K] = first load of round 2 = round-1 global
res["C0c_round1_global_vs_weighted_mean_maxabs"] = d
res["C0c_match_float32_tolerance"] = d < 1e-5
res["PASS"] = all(res["C0a_identical_starts_per_round"]) and res["C0b_client_vectors_bit_identical_under_reordering"] \
    and res["C0b_global_bit_identical_under_reordering"] and res["C0c_match_float32_tolerance"]
json.dump(res, open(os.path.join(HERE, "test_fedavg.json"), "w"), indent=1); print(json.dumps(res, indent=1))
