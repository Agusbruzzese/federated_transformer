"""ClientApp (report 30, M1). Flower's current runtime starts a NEW process for every message, so ALL site state (AdamW moments,
CUDA generator, batch-stream generator / permutation / position) is restored from Context.state each round (and from the site's
own folder after a SuperNode restart). The local step is fl_system/core.py, unchanged (validated bit-identical to fedsim3).
Node config: data-path = the site's own folder; code-root = repository root (for fl_system/core.py and ssl_central/)."""
import hashlib, io, json, os, sys

import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

app = ClientApp()


def _setup(context):
    root = context.node_config["code-root"]
    for p in (root, os.path.join(root, "fl_system")):
        if p not in sys.path: sys.path.insert(0, p)
    return context.node_config["data-path"]


@app.query()
def query(msg: Message, context: Context):
    info = json.load(open(os.path.join(_setup(context), "info.json")))
    return Message(content=RecordDict({"site": ConfigRecord({"k": info["k"], "n": info["n"]})}), reply_to=msg)


@app.train()
def train(msg: Message, context: Context):
    trace = context.node_config.get("trace-file", "")
    if trace:                                                            # M1c: record every file this ClientApp opens
        opened = []
        sys.addaudithook(lambda ev, args: opened.append(os.path.abspath(os.fsdecode(args[0])))
                         if ev == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)) else None)
    try:
        return _train(msg, context)
    finally:
        if trace:
            with open(trace, "a") as f: f.write("\n".join(sorted(set(opened))) + "\n")


def _train(msg: Message, context: Context):
    folder = _setup(context)
    import torch, core
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
    c = msg.content["cfg"]; rnd, R, X, N, lr, batch, seed = (int(c["round"]), int(c["R"]), int(c["X"]), int(c["N"]), float(c["lr"]),
                                                             int(c["batch"]), int(c["seed"]))
    enc, head = Mo.make("tf").to(DEV), Mo.Head().to(DEV)                # values overwritten below (as fedsim3)
    ck_seed = seed * 1000 + k + 1; g = torch.Generator(device=DEV); g.manual_seed(ck_seed)
    sdir = os.path.join(folder, "runs2", context.run_config["run-name"]); os.makedirs(sdir, exist_ok=True)
    blob = None
    if "state" in context.state and int(context.state["meta"]["round"]) == rnd - 1:
        blob = context.state["state"].to_numpy_ndarrays()[0].tobytes()
    elif rnd > 1:                                                        # new run (resume) or SuperNode restart: own disk copy
        p = os.path.join(sdir, f"state_r{rnd - 1:03d}.pt"); assert os.path.exists(p), f"site {k}: no state for round {rnd - 1}"
        blob = open(p, "rb").read()
    if blob is None:
        opt = torch.optim.AdamW(core.params(enc, head), lr=lr, weight_decay=core.WD); stream = core.Stream(n, int(min(batch, n)), ck_seed, DEV)
    else:
        s = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=False)
        opt = torch.optim.AdamW(core.params(enc, head), lr=lr, weight_decay=core.WD); opt.load_state_dict(s["opt"])
        g.set_state(s["g"]); stream = core.Stream(n, s["bs"], ck_seed, DEV); stream.gp.set_state(s["gp"])
        stream.perm = s["perm"].to(DEV) if s["perm"] is not None else None; stream.b = s["b"]
    sk, bk = core.plan(n, N, X, R, batch); steps_ref = X // core.REF_BATCH
    core.set_weights(enc, head, msg.content["params"].to_numpy_ndarrays())
    m = core.local_round(enc, head, opt, T, bg, g, stream, rnd - 1, R, sk, steps_ref, lr)
    buf = io.BytesIO(); torch.save(dict(round=rnd, opt=opt.state_dict(), g=g.get_state(), gp=stream.gp.get_state(), perm=stream.perm,
                                       b=stream.b, bs=stream.bs), buf)
    raw = buf.getvalue()
    context.state["state"] = ArrayRecord([np.frombuffer(raw, dtype=np.uint8).copy()]); context.state["meta"] = ConfigRecord({"round": rnd})
    tmp = os.path.join(sdir, f"state_r{rnd:03d}.pt.tmp"); open(tmp, "wb").write(raw); os.replace(tmp, tmp[:-4])
    old = os.path.join(sdir, f"state_r{rnd - 3:03d}.pt")
    if os.path.exists(old): os.remove(old)
    met = MetricRecord({"k": k, "sk": sk, "bk": bk, "num_examples": int(sk * bk), **{kk: float(v) for kk, v in m.items()}})
    return Message(content=RecordDict({"params": ArrayRecord(core.get_weights(enc, head)), "metrics": met}), reply_to=msg)
