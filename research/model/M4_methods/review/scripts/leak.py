import sys, numpy as np, torch, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,"research/model/M4_methods"); import m4
from sklearn.decomposition import PCA
m,_ = m4.load(); E = np.load(sys.argv[1]); tr=(m.role=="train").to_numpy()
raw = E["raw"]; morph = m[[c for c in m.columns if c.startswith("m8_")]].fillna(0).to_numpy(float)
for nm, X in [("raw",raw),("pca64",PCA(64,random_state=0).fit(raw[tr]).transform(raw)),("morph8",morph),
              ("maeflat_init0",E["maeflat__init__0"]),("maeflat_final0",E["maeflat__final__0"])]:
    print("leak", nm, {k: round(v,3) for k,v in m4.leak_scores(X, m).items()}, flush=True)
# MAE trivial reconstruction baselines on eval crops (24 patches x 25), mask 50%
x = torch.tensor(raw[tr]).view(-1,24,25); g=torch.Generator().manual_seed(0)
r = torch.rand(x.shape[0],24,generator=g); mask = torch.zeros_like(r,dtype=torch.bool); mask.scatter_(1, r.argsort(1)[:,:12], True)
print("MSE zero-pred on masked:", float((x[mask]**2).mean()))
# linear interpolation across masked runs from nearest visible samples
xf = x.reshape(x.shape[0],-1).numpy(); mf = mask.repeat_interleave(25,1).numpy()
rec = xf.copy()
for i in range(len(xf)):
    vis = np.flatnonzero(~mf[i]); rec[i] = np.interp(np.arange(600), vis, xf[i,vis])
print("MSE linear-interp on masked:", float(((rec-xf)[mf]**2).mean()))
