import sys, os, numpy as np, torch
sys.path.insert(0, "research/model/M4_methods"); import m4
from ssl_central import augment as A
m, W = m4.load(); sel = {"simclr_tf":0,"simclr_cnn":0,"vicreg_tf":1,"mae_tf":0}
out = {}
for meth, c in sel.items():
    for s in range(5):
        for w in ("final","init"):
            out[f"{meth}|{w}|{s}"] = m4.embed(meth, c, s, w)
# MAE flattened tokens (no pooling)
@torch.no_grad()
def mae_tokens(s, w):
    enc,_ = m4.build("mae_tf"); enc.load_state_dict(torch.load(f"data/m4/ckpt/mae_tf_c0_s{s}/{w}.pt")); enc=enc.cuda().eval()
    res=[]; rec=[]
    for i in range(0,len(W),2048):
        x = A.eval_view(torch.tensor(W[i:i+2048],device="cuda"), torch.tensor(m.fs_eff.to_numpy(np.float32)[i:i+2048],device="cuda"))
        _, t = enc.tokens(x); h = enc.enc.norm(enc.enc.enc(t+enc.enc.pos))
        res.append(h.flatten(1).cpu().numpy())
    return np.concatenate(res)
for s in range(5):
    for w in ("final","init"):
        out[f"maeflat|{w}|{s}"] = mae_tokens(s,w)
raw = A.eval_view(torch.tensor(W,device="cuda"), torch.tensor(m.fs_eff.to_numpy(np.float32),device="cuda")).cpu().numpy()
out["raw"] = raw
np.savez(sys.argv[1], **{k.replace("|","__"):v for k,v in out.items()})
print(len(out), raw.shape)
