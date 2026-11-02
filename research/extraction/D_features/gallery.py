import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
SCR="/tmp/claude-1000/-home-augustin-Documents-federated-transformer/8ccb016b-de96-4655-ab49-d3318afeb915/scratchpad/"
LP=np.load(SCR+"LP.npy"); m=pd.read_parquet(SCR+"meta.parquet")
ev=np.where(m.kind=="event")[0]; rng=np.random.default_rng(0); pick=rng.choice(ev,42,replace=False)
bg=rng.choice(np.where(m.kind=="bg")[0],6,replace=False)
fig,ax=plt.subplots(8,6,figsize=(20,18)); t=np.arange(1200)/100-6
for a,i in zip(ax.flat,list(pick)+list(bg)):
    a.plot(t,LP[i],lw=.8); a.axvspan(m.on[i]-6,m.off[i]-6,alpha=.15,color='orange') if m.kind[i]=="event" else None
    a.set_title(f"{m.kind[i]} {m.wall[i]:%m-%d %H:%M} d={m.dur[i]:.1f} pk={m.peak[i]/1e3:.0f}k",fontsize=8)
plt.tight_layout(); plt.savefig("fig_event_gallery.png",dpi=60)
e=m[m.kind=="event"]
print(e[["dur","peak","sigma","dc_baseline"]].describe(percentiles=[.05,.25,.5,.75,.95]).T)
print("polarity (|pmax|>|pmin|) frac", (e.pmax>-e.pmin).mean())
