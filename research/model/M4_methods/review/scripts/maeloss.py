import sys, numpy as np, torch
sys.path.insert(0,"research/model/M4_methods"); import m4
from ssl_central import augment as A
m,W = m4.load(); tr=np.flatnonzero((m.role=="train").to_numpy())
s=m.iloc[tr]; bg,_=m4.bgpool()
Wt=torch.tensor(W[tr],device="cuda"); fs=torch.tensor(s.fs_eff.to_numpy(np.float32),device="cuda")
g=torch.Generator(device="cuda"); g.manual_seed(7)
xtr = A.train_view(Wt,fs,torch.tensor(s.ev_i0.to_numpy(),device="cuda"),torch.tensor(s.ev_i1.to_numpy(),device="cuda"),
     torch.tensor(s.sigma_ev.to_numpy(np.float32),device="cuda"),torch.tensor(bg,device="cuda"),g,parts=("crop",))
xev = A.eval_view(Wt,fs)
with torch.no_grad():
  for nm,x in [("train-crop",xtr),("eval-crop",xev)]:
    print(nm,"zero-pred MSE", float((x**2).mean()))
    for s_ in (0,1):
        enc,_=m4.build("mae_tf"); enc.load_state_dict(torch.load(f"data/m4/ckpt/mae_tf_c0_s{s_}/final.pt")); enc=enc.cuda().eval()
        g2=torch.Generator(device="cuda"); g2.manual_seed(1)
        L=[float(enc.loss(x[i:i+512],0.5,g2)) for i in range(0,len(x),512)]
        # also: is decoder output ~0? measure pred energy
        P,t=enc.tokens(x[:512]); n=t.shape[1]
        print(nm,"seed",s_,"MAE masked MSE",np.mean(L))
