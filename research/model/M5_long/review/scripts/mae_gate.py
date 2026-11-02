# Reviewer: re-evaluate corrected G1 with fairer comparators (zero predictor; interpolation shrunk by the optimal scalar).
import sys, json, numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "research/model/M5_long"); sys.path.insert(0, ".")
import m5, m4
m, _ = m4.load(); va = (m.role == "val").to_numpy(); x = m5.crops()[torch.tensor(va, device="cuda")]; out = {}
with torch.no_grad():
    for s in range(5):
        r = {}
        for which in ("final", "init"):
            enc, _ = m5.build("mae_tf"); enc.load_state_dict(torch.load(m5.ck("mae_tf", s, which), map_location="cuda")); enc = enc.cuda().eval()
            g = torch.Generator(device="cuda"); g.manual_seed(12345); n = x.shape[1] // enc.enc.patch
            mk = enc.make_mask(x.shape[0], n, 0.75, g, "cuda"); pred, P = enc.forward_masked(x, mk); tn = enc.norm_t(P)
            ok = mk & (P.std(-1) >= 1e-3); r[f"model_{which}"] = F.mse_loss(pred[ok], tn[ok]).item()
        xs = x.cpu().numpy(); smk = mk.repeat_interleave(enc.enc.patch, 1).cpu().numpy(); it = xs.copy()
        for i in range(len(xs)):
            kn = np.flatnonzero(~smk[i]); it[i, smk[i]] = np.interp(np.flatnonzero(smk[i]), kn, xs[i, kn])
        Pi = torch.tensor(it, device="cuda").view_as(P); sd = Pi.std(-1, keepdim=True)
        Pn = torch.where(sd > 1e-8, (Pi - Pi.mean(-1, keepdim=True)) / (sd + 1e-12), torch.zeros_like(Pi))
        a, b = Pn[ok].flatten(), tn[ok].flatten(); k = (a @ b / (a @ a)).item()
        r.update(interp=F.mse_loss(Pn[ok], tn[ok]).item(), interp_shrunk=F.mse_loss(k * Pn[ok], tn[ok]).item(), shrink=k,
                 zero=(tn[ok] ** 2).mean().item(), frac_masked_kept=float(ok.sum() / mk.sum()))
        # how much does the model explain? R^2 vs zero predictor
        out[s] = r; print(s, {q: round(v, 3) for q, v in r.items()}, flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
