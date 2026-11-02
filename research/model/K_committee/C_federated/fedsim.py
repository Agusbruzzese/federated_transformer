"""Committee C: in-process, deterministic FedAvg SimCLR simulation that reuses the M5 recipe unchanged.

Only the training TOPOLOGY differs between arms. Everything else is M5's simclr_tf (research/model/M5_long/m5.py):
encoder ssl_central.models.PatchTransformer + Head, init under torch.manual_seed(seed), augment.train_view
(two views, same order), models.nt_xent (T = 0.1, 30 s false-negative guard), AdamW(lr 3e-4, wd 1e-4),
5 % linear warm-up then cosine, epoch-permutation batches with drop_last.

Arms (--arm):
  central  one "client" that holds the union of the partition's client events (FedAvg with 1 client is the identity,
           so this IS M5's loop; `selftest` proves it reproduces m5.train bit-for-bit).
  fed      10 clients (partitions.json, --kind natural|iid); every client participates every round.
  server   central topology on the server-owned events (P11: run 0218) = pretraining for the fedpre arm.
  Any arm accepts --init <ckpt> (encoder+head state) -> fedpre = `fed --init <server ckpt>`.

Budget accounting: --examples X (single-event examples, each giving 2 views; M5 = 10_000 x 256 = 2_560_000).
Per round, client k runs s_k = max(1, round(X / R * n_k / N / b_k)) local steps of batch b_k = min(b, n_k)
(equal local EPOCHS for all clients, so each event is seen as often as in central). The LR at local step j of round r
is M5's schedule evaluated at the central-equivalent step i = (r * s_k + j) * X / 256 / (R * s_k).

    .venv/bin/python research/model/K_committee/C_federated/fedsim.py selftest
    .venv/bin/python research/model/K_committee/C_federated/fedsim.py run --arm fed --part P13 --kind natural --seed 0 \
        --rounds 200 --batch 64 [--max-rounds 3] [--opt persist|reset] [--server fedavg|fedadam] [--tag NAME]
"""
import os, sys, json, math, time, argparse
import numpy as np, pandas as pd, torch

ROOT = os.getcwd(); sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, "research/model/M4_methods"))
import m4
from ssl_central import augment as A, models as Mo

HERE = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(ROOT, "data", "k_committee", "C_federated")
DEV = "cuda"; LR, WD, WARM, T_NCE, REF_BATCH = 3e-4, 1e-4, 0.05, 0.1, 256


# ---------------------------------------------------------------- data
def day_pool(days):
    """Per-DAY real background pools (noisepool.py). Every event draws noise ONLY from stretches recorded on its own
    calendar day, in EVERY arm (central, FL natural, FL IID, server), so the augmentation distribution of each event is
    identical across arms and no client ever touches another client's noise. Returns (pool [M,640], start, count) where
    start/count index the rows of each day in `days` order."""
    bg = np.load(os.path.join(OUT, "bgpool_day.npy")); day = pd.read_parquet(os.path.join(OUT, "bgpool_day_meta.parquet")).day.to_numpy()
    rows, start, cnt = [], {}, {}
    for d in sorted(set(days)):
        r = np.flatnonzero(day == d); assert len(r) > 0, f"no noise segments for {d}"
        start[d] = sum(len(x) for x in rows); cnt[d] = len(r); rows.append(r)
    return torch.tensor(bg[np.concatenate(rows)], device=DEV), start, cnt


def train_view_day(W, fs_eff, ev0, ev1, sigma_ev, bg, g, st, ct):
    """augment.train_view with ONE change: the noise row k is drawn uniformly within the event's own day block
    [st, st + ct) of the per-day pool instead of uniformly over a shared pool. Order of all other ops unchanged."""
    from ssl_central import config as C
    dev = W.device; B = W.shape[0]
    s = A._u(B, *C.STRETCH, g, dev); c = A.crop_centres(ev0, ev1, fs_eff, s, g); x = A.render(W, fs_eff, c, s)
    x = x * A._u(B, *C.AMP, g, dev)[:, None]
    k = st + (torch.rand(B, generator=g, device=dev) * ct).long().clamp(max=ct - 1)
    o = torch.randint(0, bg.shape[1] - C.CROP_N + 1, (B,), generator=g, device=dev)
    n = torch.gather(bg[k], 1, o[:, None] + torch.arange(C.CROP_N, device=dev)[None, :])
    x = x + n * (sigma_ev * A._u(B, *C.NOISE_ALPHA, g, dev))[:, None]
    Am = x.abs().amax(1, keepdim=True)
    off = A._u(B, -C.OFFSET, C.OFFSET, g, dev)[:, None]; dr = A._u(B, -C.DRIFT, C.DRIFT, g, dev)[:, None]
    ramp = torch.linspace(-0.5, 0.5, C.CROP_N, device=dev)[None, :]
    return A.maxnorm(x + Am * (off + dr * ramp))


def setup(arm, part, kind):
    m, W = m4.load(); P = json.load(open(os.path.join(HERE, "partitions.json")))[part]
    if arm == "server":
        clients = [P["server"]]; cdays = [P["server_days"]]
    elif arm == "central":
        clients = [sorted(i for c in P["natural"] for i in c)]; cdays = [sorted({d for g in P["groups"] for d in g})]
    else:
        clients = P[kind]; cdays = P["groups"] if kind == "natural" else [sorted({d for g in P["groups"] for d in g})] * len(clients)
    U = np.array(sorted({i for c in clients for i in c}))           # union, in m4 row order (== m5's train idx for P13)
    assert set(m.role.iloc[U]) == {"train"}, "firewall: only training-role events"
    pos = {g: k for k, g in enumerate(U)}; s = m.iloc[U]
    T = dict(W=torch.tensor(W[U], device=DEV), fs=torch.tensor(s.fs_eff.to_numpy(np.float32), device=DEV),
             ev0=torch.tensor(s.ev_i0.to_numpy(), device=DEV), ev1=torch.tensor(s.ev_i1.to_numpy(), device=DEV),
             sig=torch.tensor(s.sigma_ev.to_numpy(np.float32), device=DEV), t=torch.tensor(s.t_s.to_numpy(np.float64), device=DEV))
    T["day"] = s.date.astype(str).to_numpy()
    cidx = [torch.tensor([pos[i] for i in c], device=DEV) for c in clients]
    return m, W, T, cidx, cdays


class Stream:
    """m5's batch order: a fresh randperm per epoch (CPU generator), N // bs batches, drop_last."""
    def __init__(self, n, bs, seed):
        self.n, self.bs = n, bs; self.spe = n // bs; self.b = self.spe
        self.gp = torch.Generator(); self.gp.manual_seed(seed)
    def next(self):
        if self.b >= self.spe:
            self.perm = torch.randperm(self.n, generator=self.gp).to(DEV); self.b = 0
        ii = self.perm[self.b * self.bs:(self.b + 1) * self.bs]; self.b += 1; return ii


def lam(i, steps):                                                   # m5's LambdaLR, float-valued step index
    wu = int(WARM * steps)
    return (i + 1) / wu if i < wu else 0.5 * (1 + math.cos(math.pi * (i - wu) / (steps - wu)))


def build(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    return Mo.make("tf").to(DEV), Mo.Head().to(DEV)


def params(enc, head): return list(enc.parameters()) + list(head.parameters())
def vec(enc, head): return torch.nn.utils.parameters_to_vector(params(enc, head)).detach().clone()
@torch.no_grad()
def load_vec(v, enc, head):
    """COPY v into the parameters. (v0 used vector_to_parameters, which makes every parameter a VIEW of v: client
    optimiser steps then wrote into the global vector, so clients trained sequentially. Found by reviewer E, 2026-10-10.)"""
    i = 0
    for p in params(enc, head):
        k = p.numel(); p.copy_(v[i:i + k].view_as(p)); i += k
    assert i == v.numel()


# ---------------------------------------------------------------- monitoring
_VAL = {}
@torch.no_grad()
def val_metrics(enc, head, ref_emb):
    """Held-out yardstick on the validation day (02-10, never trained on): NT-Xent on fixed augmented views
    (fixed generator, M5 noise pool), effective rank of the 64-d embedding, linear CKA to a central reference."""
    if "x" not in _VAL:
        m, W = m4.load(); va = np.flatnonzero((m.role == "val").to_numpy()); s = m.iloc[va]; bg, _ = m4.bgpool()
        a = (torch.tensor(W[va], device=DEV), torch.tensor(s.fs_eff.to_numpy(np.float32), device=DEV),
             torch.tensor(s.ev_i0.to_numpy(), device=DEV), torch.tensor(s.ev_i1.to_numpy(), device=DEV),
             torch.tensor(s.sigma_ev.to_numpy(np.float32), device=DEV), torch.tensor(bg, device=DEV))
        g = torch.Generator(device=DEV); g.manual_seed(12345)
        _VAL.update(v1=A.train_view(*a, g), v2=A.train_view(*a, g), t=torch.tensor(s.t_s.to_numpy(np.float64), device=DEV),
                    x=A.eval_view(a[0], a[1]))
    enc.eval(); head.eval()
    loss = Mo.nt_xent(head(enc(_VAL["v1"])), head(enc(_VAL["v2"])), _VAL["t"], T_NCE)[0].item()
    e = enc(_VAL["x"]); enc.train(); head.train()
    sv = torch.linalg.svdvals(e - e.mean(0)); p = sv / sv.sum(); erank = float(torch.exp(-(p * torch.log(p + 1e-12)).sum()))
    out = dict(val_loss=loss, erank=erank)
    if ref_emb is not None: out["cka_ref"] = cka(e, ref_emb)
    return out


def cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    return float((X.T @ Y).norm() ** 2 / ((X.T @ X).norm() * (Y.T @ Y).norm()))


@torch.no_grad()
def ref_embedding(path):
    if not path or not os.path.exists(path): return None
    enc = Mo.make("tf").to(DEV); sd = torch.load(path, map_location=DEV); enc.load_state_dict(sd.get("enc", sd)); enc.eval()
    val_metrics(enc, Mo.Head().to(DEV), None); return enc(_VAL["x"])


# ---------------------------------------------------------------- simulation
def run(a):
    torch.backends.cudnn.benchmark = False
    m, W, T, cidx, cdays = setup(a.arm, a.part, a.kind)
    if a.noise == "m4":
        bg = torch.tensor(m4.bgpool()[0], device=DEV); view = lambda *z: A.train_view(*z)          # M5 exactly (selftest)
        npool = dict(m4_pool=int(bg.shape[0]))
    else:
        bg, st, ct = day_pool(T["day"])
        T["st"] = torch.tensor([st[d] for d in T["day"]], device=DEV); T["ct"] = torch.tensor([ct[d] for d in T["day"]], device=DEV)
        view = None; npool = {d: ct[d] for d in sorted(ct)}
    K = len(cidx); n = np.array([len(c) for c in cidx]); N = int(n.sum()); R = a.rounds; X = a.examples
    bk = np.minimum(a.batch, n); sk = np.maximum(1, np.round(X / R * n / N / bk)).astype(int)
    steps_ref = X // REF_BATCH                                       # central-equivalent steps (10_000 for M5)
    plan = dict(K=K, n=n.tolist(), b=bk.tolist(), s=sk.tolist(), examples_per_round=int((sk * bk).sum()),
                examples_total=int(R * (sk * bk).sum()), budget=X, grad_steps_total=int(R * sk.sum()),
                seq_depth=int(R * sk.max()), local_epochs_per_round=(sk * bk / n).round(2).tolist(),
                noise_segments_per_day=npool)
    print(json.dumps(plan), flush=True)
    enc, head = build(a.seed)
    if a.init:
        sd = torch.load(a.init, map_location=DEV); enc.load_state_dict(sd["enc"]); head.load_state_dict(sd["head"])
    gvec = vec(enc, head)
    # clients: own module copy, own AdamW state, own generators (central: client 0 uses seed exactly like m5)
    cl = []
    for k in range(K):
        ck_seed = a.seed if K == 1 else a.seed * 1000 + k + 1
        if K == 1: e_, h_ = enc, head
        else:
            e_, h_ = Mo.make("tf").to(DEV), Mo.Head().to(DEV)
        g = torch.Generator(device=DEV); g.manual_seed(ck_seed)
        cl.append(dict(enc=e_, head=h_, g=g, stream=Stream(int(n[k]), int(bk[k]), ck_seed),
                       opt=torch.optim.AdamW(params(e_, h_), lr=LR, weight_decay=WD)))
    srv = dict(m=torch.zeros_like(gvec), v=torch.zeros_like(gvec))      # FedAdam (Reddi et al. 2021) state
    ref = ref_embedding(a.ref); tag = a.tag or f"{a.arm}_{a.part}_{a.kind}_s{a.seed}"
    d = os.path.join(OUT, "runs", tag); os.makedirs(d, exist_ok=True); log = open(os.path.join(d, "rounds.jsonl"), "w")
    torch.cuda.reset_peak_memory_stats(); t_all = time.time(); losses_all = []
    for r in range(min(R, a.max_rounds or R)):
        torch.cuda.synchronize(); t0 = time.time(); new = torch.zeros_like(gvec); cvecs, rows = [], []
        order = list(range(K))[::-1] if a.reverse_clients else list(range(K))
        cv_by_k = {}
        for k in order:
            c = cl[k]; e_, h_ = c["enc"], c["head"]; load_vec(gvec, e_, h_)
            if a.opt == "reset" and r > 0: c["opt"] = torch.optim.AdamW(params(e_, h_), lr=LR, weight_decay=WD)
            L, gf = [], []; e_.train(); h_.train()
            for j in range(sk[k]):
                i_eq = (r * sk[k] + j) * steps_ref / (R * sk[k])
                for pg in c["opt"].param_groups: pg["lr"] = LR * lam(i_eq, steps_ref)
                ii = cidx[k][c["stream"].next()]
                args = (T["W"][ii], T["fs"][ii], T["ev0"][ii], T["ev1"][ii], T["sig"][ii], bg, c["g"])
                if view is not None: v1, v2 = view(*args), view(*args)
                else: v1, v2 = train_view_day(*args, T["st"][ii], T["ct"][ii]), train_view_day(*args, T["st"][ii], T["ct"][ii])
                loss, frac = Mo.nt_xent(h_(e_(v1)), h_(e_(v2)), T["t"][ii], T_NCE)
                c["opt"].zero_grad(set_to_none=True); loss.backward(); c["opt"].step(); L.append(loss.item()); gf.append(frac)
            cv = vec(e_, h_); cv_by_k[k] = cv
            rows.append(dict(k=k, loss_first=L[0], loss_last=L[-1], loss_mean=float(np.mean(L)), guard=float(np.mean(gf)),
                             upd=float((cv - gvec).norm())))
            losses_all += L
        for k in range(K):
            cvecs.append(cv_by_k[k]); new += (sk[k] * bk[k] / (sk * bk).sum()) * cv_by_k[k]
        rows.sort(key=lambda r_: r_["k"])
        if a.server == "fedadam":                                        # server optimiser on the pseudo-gradient
            dlt = new - gvec; srv["m"] = 0.9 * srv["m"] + 0.1 * dlt; srv["v"] = 0.99 * srv["v"] + 0.01 * dlt ** 2
            new = gvec + a.server_lr * srv["m"] / (srv["v"].sqrt() + 1e-3)
        step = float((new - gvec).norm()); drift = float(np.mean([(cv - new).norm().item() for cv in cvecs])) if K > 1 else 0.0
        gvec = new; torch.cuda.synchronize(); dt = time.time() - t0
        rec = dict(round=r, secs=dt, global_step=step, drift=drift, drift_ratio=drift / max(step, 1e-12),
                   loss=float(np.average([x["loss_mean"] for x in rows], weights=sk * bk)), clients=rows,
                   peak_mem_mb=torch.cuda.max_memory_allocated() / 2 ** 20)
        if a.eval_every and (r % a.eval_every == 0 or r == R - 1):
            load_vec(gvec, enc, head); rec.update(val_metrics(enc, head, ref))
        log.write(json.dumps(rec) + "\n"); log.flush()
        print(f"[{tag}] r{r} {dt:.2f}s loss {rec['loss']:.4f} drift/step {rec['drift_ratio']:.2f} "
              + " ".join(f"{k}={rec[k]:.3f}" for k in ("val_loss", "erank", "cka_ref") if k in rec), flush=True)
    load_vec(gvec, enc, head)
    torch.save(dict(enc=enc.state_dict(), head=head.state_dict()), os.path.join(d, "final.pt"))
    summ = dict(args=vars(a), plan=plan, wall_s=time.time() - t_all, peak_mem_mb=torch.cuda.max_memory_allocated() / 2 ** 20,
                rounds_run=r + 1, step_losses=losses_all if K == 1 else None)
    json.dump(summ, open(os.path.join(d, "summary.json"), "w"), indent=1)
    return summ, enc, head


def parser():
    p = argparse.ArgumentParser(); p.add_argument("cmd")
    p.add_argument("--arm", default="fed", choices=["central", "fed", "server"]); p.add_argument("--part", default="P13")
    p.add_argument("--kind", default="natural", choices=["natural", "iid"]); p.add_argument("--seed", type=int, default=0)
    p.add_argument("--rounds", type=int, default=200); p.add_argument("--max-rounds", type=int, default=0)
    p.add_argument("--batch", type=int, default=64); p.add_argument("--examples", type=int, default=10_000 * 256)
    p.add_argument("--opt", default="persist", choices=["persist", "reset"]); p.add_argument("--server", default="fedavg", choices=["fedavg", "fedadam"])
    p.add_argument("--server-lr", type=float, default=1e-3); p.add_argument("--noise", default="day", choices=["day", "m4"])
    p.add_argument("--init", default=""); p.add_argument("--ref", default=os.path.join(ROOT, "data/m5/ckpt/simclr_tf_s0/final.pt"))
    p.add_argument("--eval-every", type=int, default=1); p.add_argument("--tag", default="")
    p.add_argument("--reverse-clients", action="store_true", help="test only: visit clients in reverse order")
    return p


def selftest():
    """Proof that `central` == M5: run m5.train for 40 steps (into a scratch dir, nothing of M5 is overwritten) and
    fedsim central (P13 union == M5 train set, M5 noise pool, 4 rounds x 10 steps, batch 256); compare."""
    sys.path.insert(0, os.path.join(ROOT, "research/model/M5_long")); import m5
    scratch = os.path.join(OUT, "selftest_m5"); m5.OUT = scratch; m5.STEPS = 40
    m5.train("simclr_tf", 0)
    ref_log = json.load(open(m5.ck("simclr_tf", 0, "log").replace(".pt", ".json")))
    a = parser().parse_args(["run", "--arm", "central", "--part", "P13", "--seed", "0", "--rounds", "4", "--batch", "256",
                             "--examples", str(40 * 256), "--noise", "m4", "--eval-every", "0", "--tag", "selftest_central"])
    summ, enc, head = run(a)
    L = np.array(summ["step_losses"]); curve = [float(np.mean(L[i:i + 10])) for i in range(0, 40, 10)]
    sd_ref = torch.load(m5.ck("simclr_tf", 0, "final"), map_location=DEV)
    maxdiff = max(float((sd_ref[k] - v).abs().max()) for k, v in enc.state_dict().items())
    res = dict(m5_curve=ref_log["curve"], fedsim_curve=curve, max_abs_weight_diff=maxdiff,
               identical=bool(maxdiff == 0.0 and np.allclose(curve, ref_log["curve"], rtol=0, atol=0)))
    json.dump(res, open(os.path.join(HERE, "selftest.json"), "w"), indent=1); print(json.dumps(res, indent=1))


if __name__ == "__main__":
    a = parser().parse_args()
    if a.cmd == "selftest": selftest()
    else: run(a)
