"""In-process simulator of the REAL federated system with SecAgg+ (and optional site-level DP), report 35 (M4).
Same per-site operations as fl_system3/fl_system4's ClientApp (fl_system/core.py, the site's own provisioned folder, per-site AdamW,
generators and batch stream), then dp.privatize, then the exact SecAgg+ arithmetic (dp.secagg_site / dp.secagg_decode), then the
server update glob = float32(float64(glob) + decoded). Validity is not assumed: V1 (report 35) checks it is bit-identical to the
real SecAgg+ runs of M3c, V2 to a real fl_system4 DP run.
    .venv-fl/bin/python fl_system4/dpsim.py --fed data/fl_system/fold0_v2/federation.json --tag <name> --seed 0 --rounds 200 \
        --examples 2560000 --batch 64 --lr 1e-3 --init data/fl_system/fold0_v2/server/server_init_s0.pt [--dp-clip C --dp-z z]
Output: <fold>/server/dpsim/<tag>/{rounds.jsonl, global_rNNN.pt (r<=3, every 50, last), final.pt, summary.json}; never overwrites."""
import argparse, hashlib, json, os, sys, time
import numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "fl_system"), os.path.join(ROOT, "fl_system4")):
    if p not in sys.path: sys.path.insert(0, p)
import core, dp
from ssl_central import models as Mo
DEV = "cuda"; sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()


def load_site(folder):
    """As the ClientApp: hash-checked own files only."""
    info = json.load(open(os.path.join(folder, "info.json")))
    for f in ("events", "noise"):
        assert sha(os.path.join(folder, f"{f}.npz")) == info[f"{f}_sha"], f"{f}.npz altered"
    ev = np.load(os.path.join(folder, "events.npz"), allow_pickle=False); nz = np.load(os.path.join(folder, "noise.npz"))
    st = {d: int(s) for d, s in zip(nz["days"], nz["start"])}; ct = {d: int(c) for d, c in zip(nz["days"], nz["count"])}
    T = dict(W=torch.tensor(ev["W"], device=DEV), fs=torch.tensor(ev["fs_eff"], device=DEV), ev0=torch.tensor(ev["ev_i0"], device=DEV),
             ev1=torch.tensor(ev["ev_i1"], device=DEV), sig=torch.tensor(ev["sigma_ev"], device=DEV), t=torch.tensor(ev["t_s"], device=DEV),
             st=torch.tensor([st[d] for d in ev["day"]], device=DEV), ct=torch.tensor([ct[d] for d in ev["day"]], device=DEV))
    return info["k"], T, torch.tensor(nz["pool"], device=DEV), int(len(ev["W"]))


def main():
    ap = argparse.ArgumentParser()
    for k, t in (("--fed", str), ("--tag", str), ("--seed", int), ("--rounds", int), ("--examples", int), ("--batch", int), ("--lr", float)):
        ap.add_argument(k, type=t, required=True)
    ap.add_argument("--init", default=""); ap.add_argument("--dp-clip", type=float, default=0.0); ap.add_argument("--dp-z", type=float, default=0.0)
    ap.add_argument("--clipping-range", type=float, default=None, help="SecAgg+ range; default dp.secagg_clipping_range(C, z, K)")
    ap.add_argument("--quantization-range", type=int, default=4194304); ap.add_argument("--modulus-range", type=int, default=4294967296)
    ap.add_argument("--max-rounds", type=int, default=0, help="stop early (tests); rounds still sets the schedule")
    a = ap.parse_args(); torch.backends.cudnn.benchmark = False
    fed = json.load(open(a.fed)); base = os.path.dirname(os.path.abspath(a.fed)); K, N = fed["K"], fed["N_total"]; R, X, B = a.rounds, a.examples, a.batch
    cr = a.clipping_range if a.clipping_range is not None else float(dp.secagg_clipping_range(a.dp_clip, a.dp_z, K))
    out = os.path.join(base, "server", "dpsim", a.tag); assert not os.path.exists(out), f"refusing to overwrite {out}"; os.makedirs(out)
    enc, head = core.build(a.seed, "cpu")
    if a.init:
        sd = torch.load(a.init, map_location="cpu"); enc.load_state_dict(sd["enc"]); head.load_state_dict(sd["head"])
    glob = core.get_weights(enc, head)
    sites = []
    for c in fed["clients"]:
        k, T, bg, n = load_site(os.path.join(base, f"client_{c['k']:02d}")); assert k == c["k"] and n == c["n"]
        e_, h_ = Mo.make("tf").to(DEV), Mo.Head().to(DEV); ck = a.seed * 1000 + k + 1; g = torch.Generator(device=DEV); g.manual_seed(ck)
        sk, bk = core.plan(n, N, X, R, B)
        sites.append(dict(k=k, T=T, bg=bg, enc=e_, head=h_, g=g, stream=core.Stream(n, int(min(B, n)), ck, DEV), sk=sk, bk=bk,
                          opt=torch.optim.AdamW(core.params(e_, h_), lr=a.lr, weight_decay=core.WD)))
    max_weight = max(s["sk"] * s["bk"] for s in sites); steps_ref = X // core.REF_BATCH
    acct = dp.accountant(R, a.dp_z) if a.dp_z > 0 else None
    log = open(os.path.join(out, "rounds.jsonl"), "x"); t_all = time.time()
    for rnd in range(1, (a.max_rounds or R) + 1):
        t0 = time.time(); enc_sites, rows = [], []
        for s in sites:
            core.set_weights(s["enc"], s["head"], glob)
            m = core.local_round(s["enc"], s["head"], s["opt"], s["T"], s["bg"], s["g"], s["stream"], rnd - 1, R, s["sk"], steps_ref, a.lr)
            new = core.get_weights(s["enc"], s["head"]); delta = [(x - y).astype(np.float32) for x, y in zip(new, glob)]
            mx = float(max(np.abs(d).max() for d in delta)); n_k = s["sk"] * s["bk"]
            delta, info = dp.privatize(delta, s["k"], rnd, a.seed, n_k, max_weight, K, a.dp_clip, a.dp_z)
            q = dp.secagg_site(delta, n_k, max_weight, cr, a.quantization_range, a.seed, s["k"], rnd); enc_sites.append(q)
            rows.append(dict(k=s["k"], n=n_k, loss_mean=m["loss_mean"], norm=info["norm"], scale=info["scale"], noise_sd=info["noise_sd"],
                             max_abs_update=mx, over_clip=q[2]))
        decoded = dp.secagg_decode(enc_sites, cr, a.quantization_range, a.modulus_range)
        glob = [(x.astype(np.float64) + d).astype(np.float32) for x, d in zip(glob, decoded)]
        rec = dict(round=rnd, secs=round(time.time() - t0, 2), loss=float(np.average([r["loss_mean"] for r in rows], weights=[r["n"] for r in rows])),
                   n_over_clip=sum(r["over_clip"] for r in rows), sites=rows)
        log.write(json.dumps(rec) + "\n"); log.flush()
        vec = torch.cat([torch.as_tensor(x).reshape(-1) for x in glob])
        if rnd <= 3 or rnd % 50 == 0 or rnd == R: torch.save(vec, os.path.join(out, f"global_r{rnd:03d}.pt"))
        if rnd % 25 == 0 or rnd <= 2: print(f"[dpsim {a.tag}] round {rnd} {rec['secs']}s loss {rec['loss']:.4f} over-clip {rec['n_over_clip']}", flush=True)
    core.set_weights(enc, head, glob); torch.save(dict(enc=enc.state_dict(), head=head.state_dict()), os.path.join(out, "final.pt"))
    json.dump(dict(args=vars(a), K=K, N=N, max_weight=max_weight, secagg_clipping_range=cr, accountant=acct, wall_s=round(time.time() - t_all, 1),
                   rounds_run=rnd, provenance=dict(init=sha(a.init) if a.init else None, final=sha(os.path.join(out, "final.pt")),
                                                   core=sha(os.path.join(ROOT, "fl_system/core.py")), dp=sha(os.path.join(ROOT, "fl_system4/dp.py")),
                                                   dpsim=sha(os.path.abspath(__file__)))),
              open(os.path.join(out, "summary.json"), "x"), indent=1)
    print(f"[dpsim {a.tag}] done {rnd} rounds in {time.time() - t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
