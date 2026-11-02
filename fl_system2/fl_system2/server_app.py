"""ServerApp (report 30, M1): strict synchronous FedAvg with the Grid API. All K sites every round; aggregation in site order
with the simulator's formula (fl_system/core.py, on GPU); NO partial round (any missing or error reply stops the run);
checkpoint every round; resume (run-config resume=true). Output: <fold>/server/runs2/<run-name>/."""
import hashlib, json, os, sys, time

import numpy as np
from flwr.app import ArrayRecord, ConfigRecord, Context, Message, RecordDict
from flwr.serverapp import Grid, ServerApp

app = ServerApp()
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()


@app.main()
def main(grid: Grid, context: Context):
    cfg = context.run_config; root = cfg["code-root"]
    for p in (root, os.path.join(root, "fl_system")):
        if p not in sys.path: sys.path.insert(0, p)
    import torch, core
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    fed = json.load(open(cfg["fed"])); K, N = fed["K"], fed["N_total"]; R = int(cfg["rounds"])
    out = os.path.join(os.path.dirname(cfg["fed"]), "server", "runs2", cfg["run-name"]); os.makedirs(out, exist_ok=True)
    enc, head = core.build(int(cfg["seed"]), "cpu")
    if cfg["init"]:
        sd = torch.load(cfg["init"], map_location="cpu"); enc.load_state_dict(sd["enc"]); head.load_state_dict(sd["head"])
    shapes = [tuple(p.shape) for p in core.params(enc, head)]; offset = 0
    if cfg["resume"]:
        offset = json.load(open(os.path.join(out, "last_round.json")))["round"]; arrays = core.unflat(torch.load(os.path.join(out, "global_last.pt")), shapes)
    else:
        assert not os.path.exists(os.path.join(out, "rounds.jsonl")), f"refusing to overwrite {out}"; arrays = core.get_weights(enc, head)
    t0 = time.time()
    while True:                                                          # query EVERY registered node; keep those that answer
        nodes = list(grid.get_node_ids())
        if len(nodes) >= K:
            q = list(grid.send_and_receive([Message(content=RecordDict({"q": ConfigRecord({"x": 1})}), dst_node_id=n, message_type="query") for n in nodes]))
            ok = [r for r in q if not r.has_error()]; ks = [int(r.content["site"]["k"]) for r in ok]
            if sorted(ks) == list(range(K)): break                       # exactly one live node per site
            assert len(ks) == len(set(ks)), f"two live nodes claim the same site: {sorted(ks)}"
        assert time.time() - t0 < cfg["node-wait-s"], f"not all sites live: {len(nodes)} registered"; time.sleep(2)
    node_of = {int(r.content["site"]["k"]): r.metadata.src_node_id for r in ok}
    print(f"[server2] {len(nodes)} registered nodes, {K} live sites mapped", flush=True)
    log = open(os.path.join(out, "rounds.jsonl"), "a"); bytes_total = 0
    for rnd in range(offset + 1, R + 1):
        tr = time.time(); c = ConfigRecord({"round": rnd, "R": R, "X": int(cfg["examples"]), "N": N, "lr": float(cfg["lr"]), "batch": int(cfg["batch"]),
                                             "seed": int(cfg["seed"])})
        retries = 0
        while True:                                                       # a failed round is REPLAYED in full (sites restore the
            msgs = [Message(content=RecordDict({"params": ArrayRecord(arrays), "cfg": c}), dst_node_id=node_of[k], message_type="train")
                    for k in range(K)]                                    # state of round rnd-1 from disk) -> identical result
            rep = list(grid.send_and_receive(msgs))
            bad = [r for r in rep if r.has_error()]
            if len(rep) == K and not bad: break
            msg_ = f"round {rnd}: {len(rep)}/{K} replies, {len(bad)} error(s): {[r.error.reason[:120] for r in bad][:2]}"
            if retries >= int(cfg["max-retries"]):
                raise RuntimeError(msg_ + " -> stopping (no partial aggregation)")
            retries += 1; print(f"[server2] {msg_} -> full replay of round {rnd} (retry {retries})", flush=True); time.sleep(5)
        by_k = {int(r.content["metrics"]["k"]): r for r in rep}; assert sorted(by_k) == list(range(K))
        assert all(by_k[k].metadata.src_node_id == node_of[k] for k in range(K)), "site identity mismatch"
        vecs = [core.flat(by_k[k].content["params"].to_numpy_ndarrays(), dev) for k in range(K)]
        ex = [int(by_k[k].content["metrics"]["num_examples"]) for k in range(K)]
        new = core.aggregate(vecs, ex); arrays = core.unflat(new, shapes)
        nb = sum(a.nbytes for a in arrays); bytes_total += 2 * K * nb
        rec = dict(round=rnd, secs=round(time.time() - tr, 2), retries=retries, examples=ex, bytes_round=2 * K * nb, bytes_total=bytes_total,
                   loss=float(np.average([float(by_k[k].content["metrics"]["loss_mean"]) for k in range(K)], weights=ex)))
        log.write(json.dumps(rec) + "\n"); log.flush()
        if rnd % int(cfg["save-every"]) == 0 or rnd <= 3 or rnd == R: torch.save(new.cpu(), os.path.join(out, f"global_r{rnd:03d}.pt"))
        torch.save(new.cpu(), os.path.join(out, "global_last.pt")); json.dump(dict(round=rnd), open(os.path.join(out, "last_round.json"), "w"))
        print(f"[server2] round {rnd} loss {rec['loss']:.4f} {rec['secs']}s", flush=True)
    core.set_weights(enc, head, arrays); torch.save(dict(enc=enc.state_dict(), head=head.state_dict()), os.path.join(out, "final.pt"))
    json.dump(dict(run_config={k: v for k, v in cfg.items()}, K=K, N=N, wall_s=round(time.time() - t0, 1), bytes_total=bytes_total,
                   provenance=dict(init=sha(cfg["init"]) if cfg["init"] else None, final=sha(os.path.join(out, "final.pt")),
                                   core=sha(os.path.join(root, "fl_system/core.py")))),
              open(os.path.join(out, "summary.json"), "w" if cfg["resume"] else "x"), indent=1)
    print("[server2] done", flush=True)
