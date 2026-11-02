"""SecAgg+ and site-level DP ClientApp (report 35, M4; fl_system3 + dp.privatize).  Same local training as fl_system2 (fl_system/core.py, state restored from Context or the site's
own disk copy); returns the UPDATE (new - global weights) and its example count. secaggplus_mod then weights it by
num_examples / max_weight, quantises it (stochastic rounding, seeded here per site and round for reproducibility), adds the
pairwise and private masks and sends only the masked vector: the server can decode the weighted SUM of all sites, never one site.
run-config dump=true: the site also saves its true update in its OWN folder (M3a/M3b tests only)."""
import hashlib, io, json, os, sys

import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, Context
from flwr.client import NumPyClient
from flwr.client.mod import secaggplus_mod
from flwr.clientapp import ClientApp


class Site(NumPyClient):
    def __init__(self, context: Context):
        self.context = context

    def fit(self, parameters, config):
        context = self.context; root = context.node_config["code-root"]; folder = context.node_config["data-path"]
        for p in (root, os.path.join(root, "fl_system"), os.path.join(root, "fl_system4")):
            if p not in sys.path: sys.path.insert(0, p)
        import torch, core, dp
        from ssl_central import models as Mo
        DEV = "cuda" if torch.cuda.is_available() else "cpu"; torch.backends.cudnn.benchmark = False
        info = json.load(open(os.path.join(folder, "info.json"))); k = info["k"]
        for f in ("events", "noise"):
            assert hashlib.sha256(open(os.path.join(folder, f"{f}.npz"), "rb").read()).hexdigest() == info[f"{f}_sha"], f"{f}.npz altered"
        ev = np.load(os.path.join(folder, "events.npz"), allow_pickle=False); nz = np.load(os.path.join(folder, "noise.npz"))
        st = {d: int(s) for d, s in zip(nz["days"], nz["start"])}; ct = {d: int(c) for d, c in zip(nz["days"], nz["count"])}
        T = dict(W=torch.tensor(ev["W"], device=DEV), fs=torch.tensor(ev["fs_eff"], device=DEV), ev0=torch.tensor(ev["ev_i0"], device=DEV),
                 ev1=torch.tensor(ev["ev_i1"], device=DEV), sig=torch.tensor(ev["sigma_ev"], device=DEV), t=torch.tensor(ev["t_s"], device=DEV),
                 st=torch.tensor([st[d] for d in ev["day"]], device=DEV), ct=torch.tensor([ct[d] for d in ev["day"]], device=DEV))
        bg = torch.tensor(nz["pool"], device=DEV); n = int(len(ev["W"]))
        rnd, R, X, N, lr, batch, seed = (int(config["round"]), int(config["R"]), int(config["X"]), int(config["N"]), float(config["lr"]),
                                         int(config["batch"]), int(config["seed"]))
        enc, head = Mo.make("tf").to(DEV), Mo.Head().to(DEV)
        ck_seed = seed * 1000 + k + 1; g = torch.Generator(device=DEV); g.manual_seed(ck_seed)
        sdir = os.path.join(folder, "runs4", context.run_config["run-name"]); os.makedirs(sdir, exist_ok=True)
        blob = None
        if "state" in context.state and int(context.state["meta"]["round"]) == rnd - 1:
            blob = context.state["state"].to_numpy_ndarrays()[0].tobytes()
        elif rnd > 1:
            p = os.path.join(sdir, f"state_r{rnd - 1:03d}.pt"); assert os.path.exists(p), f"site {k}: no state for round {rnd - 1}"
            blob = open(p, "rb").read()
        opt = torch.optim.AdamW(core.params(enc, head), lr=lr, weight_decay=core.WD)
        if blob is None:
            stream = core.Stream(n, int(min(batch, n)), ck_seed, DEV)
        else:
            s = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=False); opt.load_state_dict(s["opt"]); g.set_state(s["g"])
            stream = core.Stream(n, s["bs"], ck_seed, DEV); stream.gp.set_state(s["gp"])
            stream.perm = s["perm"].to(DEV) if s["perm"] is not None else None; stream.b = s["b"]
        sk, bk = core.plan(n, N, X, R, batch); steps_ref = X // core.REF_BATCH
        core.set_weights(enc, head, parameters)
        m = core.local_round(enc, head, opt, T, bg, g, stream, rnd - 1, R, sk, steps_ref, lr)
        buf = io.BytesIO(); torch.save(dict(round=rnd, opt=opt.state_dict(), g=g.get_state(), gp=stream.gp.get_state(), perm=stream.perm,
                                           b=stream.b, bs=stream.bs), buf); raw = buf.getvalue()
        context.state["state"] = ArrayRecord([np.frombuffer(raw, dtype=np.uint8).copy()]); context.state["meta"] = ConfigRecord({"round": rnd})
        tmp = os.path.join(sdir, f"state_r{rnd:03d}.pt.tmp"); open(tmp, "wb").write(raw); os.replace(tmp, tmp[:-4])
        old = os.path.join(sdir, f"state_r{rnd - 3:03d}.pt")
        if os.path.exists(old): os.remove(old)
        new = core.get_weights(enc, head); delta = [(a - b).astype(np.float32) for a, b in zip(new, parameters)]
        c = float(context.run_config["clipping-range"]); mx = float(max(np.abs(d).max() for d in delta))
        n_max, K = float(config["nmax"]), int(config["K"]); Q = int(context.run_config["quantization-range"])
        delta, info = dp.privatize(delta, k, rnd, seed, sk * bk, n_max, K, float(context.run_config["dp-clip"]), float(context.run_config["dp-z"]))
        dq = round(sk * bk / n_max * Q) / Q                                 # secaggplus_mod's weight factor: values it clips are |dq * x| > c
        if context.run_config["dump"]:
            np.savez(os.path.join(sdir, f"private_update_r{rnd:03d}_node{context.node_id}.npz"), *delta, n_examples=np.array(sk * bk))
        np.random.seed(dp.qrand_seed(seed, k, rnd))                         # deterministic stochastic rounding inside secaggplus_mod
        return delta, int(sk * bk), dict(k=k, sk=sk, bk=bk, max_abs_update=mx, n_over_clip=int(sum((np.abs(d * dq) > c).sum() for d in delta)),
                                         norm=info["norm"], scale=info["scale"], noise_sd=info["noise_sd"], **{kk: float(v) for kk, v in m.items()})


def client_fn(context: Context):
    return Site(context).to_client()


app = ClientApp(client_fn=client_fn, mods=[secaggplus_mod])
