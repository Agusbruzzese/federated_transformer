"""SecAgg+ and site-level DP ServerApp (report 35, M4; fl_system3 + DP config, accountant in summary).  Flower's SecAggPlusWorkflow runs the protocol (key setup, share exchange, masked-vector
collection, unmasking) and decodes ONLY the weighted average of the sites' updates; StrictSecAggFedAvg adds it to the global
model. Strict: any failure or missing site in a round stops the run (no partial aggregation, no dropout recovery used).
run-config dump=true: the grid is wrapped to save every masked vector the server receives and the decoded average (tests)."""
import hashlib, json, os, sys, time

import numpy as np
from flwr.app import Context
from flwr.common import bytes_to_ndarray, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server import LegacyContext, ServerConfig
from flwr.server.strategy import FedAvg
from flwr.server.workflow import DefaultWorkflow, SecAggPlusWorkflow
from flwr.serverapp import Grid, ServerApp

app = ServerApp()
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()


class RecordingGrid:
    """Delegates to the real Grid; saves the masked vectors in every reply (what the server actually receives)."""
    def __init__(self, grid, out, strat):
        self._g, self._out, self._s = grid, out, strat
    def __getattr__(self, name):
        return getattr(self._g, name)
    def send_and_receive(self, messages, *a, **kw):
        rep = list(self._g.send_and_receive(messages, *a, **kw))
        for r in rep:
            if r.has_content() and "secaggplus_configs" in r.content.config_records and "masked_params" in r.content.config_records["secaggplus_configs"]:
                arrs = [bytes_to_ndarray(b) for b in r.content.config_records["secaggplus_configs"]["masked_params"]]
                np.savez(os.path.join(self._out, f"masked_r{self._s.current_round:03d}_node{r.metadata.src_node_id}.npz"), *arrs)
        return rep


class StrictSecAggFedAvg(FedAvg):
    def __init__(self, K, global_arrays, out, offset, R, cfg_fn, save_every, dump, **kw):
        super().__init__(initial_parameters=ndarrays_to_parameters(global_arrays), fraction_fit=1.0, fraction_evaluate=0.0, min_fit_clients=K,
                         min_evaluate_clients=0, min_available_clients=K, accept_failures=False, on_fit_config_fn=cfg_fn, **kw)
        self.K, self.glob, self.out, self.offset, self.R, self.save_every, self.dump = K, global_arrays, out, offset, R, save_every, dump
        self.log = open(os.path.join(out, "rounds.jsonl"), "a"); self.t0 = time.time(); self.current_round = offset

    def configure_fit(self, server_round, parameters, client_manager):
        self.current_round = server_round + self.offset; self.t_round = time.time()
        return super().configure_fit(server_round, parameters, client_manager)

    def aggregate_fit(self, server_round, results, failures):
        import torch
        rnd = server_round + self.offset
        if failures or len(results) != self.K:
            raise RuntimeError(f"round {rnd}: {len(results)}/{self.K} results, {len(failures)} failure(s) -> stopping (no partial aggregation)")
        decoded = parameters_to_ndarrays(results[0][1].parameters)       # SecAgg+ puts the decoded weighted mean in every result
        self.glob = [(g.astype(np.float64) + d).astype(np.float32) for g, d in zip(self.glob, decoded)]
        mets = [dict(r.metrics) for _, r in results]
        rec = dict(round=rnd, secs=round(time.time() - self.t_round, 2), examples=sorted(int(r.num_examples) for _, r in results),
                   loss=float(np.average([m["loss_mean"] for m in mets], weights=[r.num_examples for _, r in results])),
                   max_abs_update=max(m["max_abs_update"] for m in mets), n_over_clip=sum(m["n_over_clip"] for m in mets),
                   sites=sorted([dict(k=int(m["k"]), norm=m["norm"], scale=m["scale"], noise_sd=m["noise_sd"]) for m in mets], key=lambda x: x["k"]))
        self.log.write(json.dumps(rec) + "\n"); self.log.flush()
        if self.dump: np.savez(os.path.join(self.out, f"decoded_r{rnd:03d}.npz"), *decoded)
        vec = torch.cat([torch.as_tensor(a).reshape(-1) for a in self.glob])
        if rnd % self.save_every == 0 or rnd <= 3 or rnd == self.R: torch.save(vec, os.path.join(self.out, f"global_r{rnd:03d}.pt"))
        torch.save(vec, os.path.join(self.out, "global_last.pt")); json.dump(dict(round=rnd), open(os.path.join(self.out, "last_round.json"), "w"))
        print(f"[server4] round {rnd} loss {rec['loss']:.4f} {rec['secs']}s max|upd| {rec['max_abs_update']:.4g} over-clip {rec['n_over_clip']}", flush=True)
        return ndarrays_to_parameters(self.glob), {}


@app.main()
def main(grid: Grid, context: Context):
    cfg = context.run_config; root = cfg["code-root"]
    for p in (root, os.path.join(root, "fl_system"), os.path.join(root, "fl_system4")):
        if p not in sys.path: sys.path.insert(0, p)
    import torch, core, dp
    fed = json.load(open(cfg["fed"])); K, N = fed["K"], fed["N_total"]; R, X, batch = int(cfg["rounds"]), int(cfg["examples"]), int(cfg["batch"])
    out = os.path.join(os.path.dirname(cfg["fed"]), "server", "runs4", cfg["run-name"]); os.makedirs(out, exist_ok=True)
    enc, head = core.build(int(cfg["seed"]), "cpu")
    if cfg["init"]:
        sd = torch.load(cfg["init"], map_location="cpu"); enc.load_state_dict(sd["enc"]); head.load_state_dict(sd["head"])
    shapes = [tuple(p.shape) for p in core.params(enc, head)]; offset = 0
    if cfg["resume"]:
        offset = json.load(open(os.path.join(out, "last_round.json")))["round"]; glob = core.unflat(torch.load(os.path.join(out, "global_last.pt")), shapes)
    else:
        assert not os.path.exists(os.path.join(out, "rounds.jsonl")), f"refusing to overwrite {out}"; glob = core.get_weights(enc, head)
    max_weight = max(sk * bk for sk, bk in (core.plan(c["n"], N, X, R, batch) for c in fed["clients"]))   # tight: every ratio <= 1
    cfg_fn = lambda r: dict(round=r + offset, R=R, X=X, N=N, lr=float(cfg["lr"]), batch=batch, seed=int(cfg["seed"]), nmax=int(max_weight), K=K)
    strat = StrictSecAggFedAvg(K, glob, out, offset, R, cfg_fn, int(cfg["save-every"]), bool(cfg["dump"]))
    lctx = LegacyContext(context=context, config=ServerConfig(num_rounds=R - offset), strategy=strat)
    wf = DefaultWorkflow(fit_workflow=SecAggPlusWorkflow(num_shares=int(cfg["num-shares"]), reconstruction_threshold=int(cfg["reconstruction-threshold"]),
                                                         max_weight=float(max_weight), clipping_range=float(cfg["clipping-range"]),
                                                         quantization_range=int(cfg["quantization-range"]), modulus_range=int(cfg["modulus-range"])))
    t0 = time.time(); wf(RecordingGrid(grid, out, strat) if cfg["dump"] else grid, lctx)
    last = json.load(open(os.path.join(out, "last_round.json")))["round"]; assert last == R, f"stopped after round {last} of {R}"
    core.set_weights(enc, head, strat.glob); torch.save(dict(enc=enc.state_dict(), head=head.state_dict()), os.path.join(out, "final.pt"))
    json.dump(dict(run_config={k: v for k, v in cfg.items()}, K=K, N=N, max_weight=max_weight, wall_s=round(time.time() - t0, 1),
                   accountant=dp.accountant(R, float(cfg["dp-z"])) if float(cfg["dp-z"]) > 0 else None,
                   provenance=dict(init=sha(cfg["init"]) if cfg["init"] else None, final=sha(os.path.join(out, "final.pt")),
                                   core=sha(os.path.join(root, "fl_system/core.py")), dp=sha(os.path.join(root, "fl_system4/dp.py")))),
              open(os.path.join(out, "summary.json"), "w" if cfg["resume"] else "x"), indent=1)
    print("[server4] done", flush=True)
