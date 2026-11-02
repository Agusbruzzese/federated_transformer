"""Federated client (one site). Holds ONLY its own folder: events, own-day noise pool, persistent optimiser and RNG state.
Each round: receive global weights -> fedsim3-identical local training -> return weights + example count.
    .venv-fl/bin/python fl_system/client.py --folder data/fl_system/fold0/client_03 --run <tag> --seed 0 [--server 127.0.0.1:8080]
        [--trace-files out.txt] [--die-at-round r]
State is saved after every round in <folder>/runs/<run>/state.pt (runtime state, rotated), so a restarted client resumes
exactly. --die-at-round is for the fault-handling test only.
"""
import argparse, json, os, sys

TRACE = []
if "--trace-files" in sys.argv:                                      # T3: record every file this process opens (from the start)
    def _hook(event, args):
        if event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
            TRACE.append(os.path.abspath(os.fsdecode(args[0])))
    sys.addaudithook(_hook)

import hashlib
import numpy as np
import torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path[:0] = [ROOT, os.path.join(ROOT, "fl_system")]
import core
from ssl_central import models as Mo
from flwr.compat.client.app import start_client
from flwr.client import NumPyClient

DEV = "cuda" if torch.cuda.is_available() else "cpu"


class SiteClient(NumPyClient):
    def __init__(self, folder, run, seed, die_at=None):
        self.folder, self.die_at = folder, die_at
        info = json.load(open(os.path.join(folder, "info.json"))); self.k = info["k"]
        for f in ("events", "noise"):                                # data integrity vs provisioning
            assert hashlib.sha256(open(os.path.join(folder, f"{f}.npz"), "rb").read()).hexdigest() == info[f"{f}_sha"], f"{f}.npz altered"
        ev = np.load(os.path.join(folder, "events.npz"), allow_pickle=False); nz = np.load(os.path.join(folder, "noise.npz"))
        st = {d: int(s) for d, s in zip(nz["days"], nz["start"])}; ct = {d: int(c) for d, c in zip(nz["days"], nz["count"])}
        self.T = dict(W=torch.tensor(ev["W"], device=DEV), fs=torch.tensor(ev["fs_eff"], device=DEV), ev0=torch.tensor(ev["ev_i0"], device=DEV),
                      ev1=torch.tensor(ev["ev_i1"], device=DEV), sig=torch.tensor(ev["sigma_ev"], device=DEV), t=torch.tensor(ev["t_s"], device=DEV),
                      st=torch.tensor([st[d] for d in ev["day"]], device=DEV), ct=torch.tensor([ct[d] for d in ev["day"]], device=DEV))
        self.bg = torch.tensor(nz["pool"], device=DEV); self.n = int(len(ev["W"]))
        torch.backends.cudnn.benchmark = False
        self.enc, self.head = Mo.make("tf").to(DEV), Mo.Head().to(DEV)          # values overwritten every round (as fedsim3)
        self.seed = seed; self.ck_seed = seed * 1000 + self.k + 1                # fedsim3: K > 1 client generator seed
        self.g = torch.Generator(device=DEV); self.g.manual_seed(self.ck_seed)
        self.opt = None; self.stream = None; self.last_round = 0
        self.sdir = os.path.join(folder, "runs", run); os.makedirs(self.sdir, exist_ok=True)

    # ---------------------------------------------------------- persistent state (exact resume)
    def _ensure(self, lr, batch):
        if self.opt is None:
            self.opt = torch.optim.AdamW(core.params(self.enc, self.head), lr=lr, weight_decay=core.WD)
            self.stream = core.Stream(self.n, int(min(batch, self.n)), self.ck_seed, DEV)

    def _spath(self, r): return os.path.join(self.sdir, f"state_r{r:03d}.pt")

    def _save_state(self):
        """State after round r, kept per round (last 3), so the client can roll back to the last round the server committed."""
        tmp = self._spath(self.last_round) + ".tmp"
        torch.save(dict(round=self.last_round, opt=self.opt.state_dict(), g=self.g.get_state(), gp=self.stream.gp.get_state(),
                        perm=self.stream.perm, b=self.stream.b, bs=self.stream.bs), tmp)
        os.replace(tmp, self._spath(self.last_round))                   # atomic
        old = self._spath(self.last_round - 3)
        if os.path.exists(old): os.remove(old)

    def _load_state(self, r):
        s = torch.load(self._spath(r), map_location="cpu", weights_only=False); self.last_round = s["round"]   # RNG states must stay on CPU
        self.opt = torch.optim.AdamW(core.params(self.enc, self.head), lr=1e-3, weight_decay=core.WD); self.opt.load_state_dict(s["opt"])
        self.g.set_state(s["g"]); self.stream = core.Stream(self.n, s["bs"], self.ck_seed, DEV)
        self.stream.gp.set_state(s["gp"]); self.stream.perm = s["perm"].to(DEV) if s["perm"] is not None else None; self.stream.b = s["b"]
        print(f"[client {self.k}] state restored to the end of round {self.last_round}", flush=True)

    # ---------------------------------------------------------- Flower API
    def get_parameters(self, config):
        return core.get_weights(self.enc, self.head)

    def fit(self, parameters, config):
        rnd = int(config["round"])
        if rnd != self.last_round + 1:                                   # restart / server rolled back: restore the committed state
            assert rnd >= 2 and os.path.exists(self._spath(rnd - 1)), f"client {self.k}: no saved state for round {rnd - 1}"
            self._load_state(rnd - 1)
        if self.die_at is not None and rnd == self.die_at:
            print(f"[client {self.k}] fault injection: dying in round {rnd}", flush=True); os._exit(3)
        R, X, N, lr, batch = int(config["R"]), int(config["X"]), int(config["N"]), float(config["lr"]), int(config["batch"])
        self._ensure(lr, batch); sk, bk = core.plan(self.n, N, X, R, batch); steps_ref = X // core.REF_BATCH
        core.set_weights(self.enc, self.head, parameters)
        m = core.local_round(self.enc, self.head, self.opt, self.T, self.bg, self.g, self.stream, rnd - 1, R, sk, steps_ref, lr)
        self.last_round = rnd; self._save_state()
        return core.get_weights(self.enc, self.head), int(sk * bk), dict(k=self.k, sk=sk, bk=bk, **m)

    def evaluate(self, parameters, config):
        return 0.0, self.n, {}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--folder", required=True); ap.add_argument("--run", required=True)
    ap.add_argument("--seed", type=int, required=True); ap.add_argument("--server", default="127.0.0.1:8080")
    ap.add_argument("--trace-files", default=""); ap.add_argument("--die-at-round", type=int, default=None)
    a = ap.parse_args(); c = SiteClient(a.folder, a.run, a.seed, a.die_at_round)
    try:
        start_client(server_address=a.server, client=c.to_client())
    finally:
        if a.trace_files:
            open(a.trace_files, "w").write("\n".join(sorted(set(TRACE))))


if __name__ == "__main__":
    main()
