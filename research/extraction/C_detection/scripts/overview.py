import sys; sys.path.insert(0,'.')
import cio, prep, numpy as np, datetime as dt, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R=cio.runs()
cases=[(0,dt.datetime(2025,2,6,3,0)),(0,dt.datetime(2025,2,5,7,30)),(0,dt.datetime(2025,2,5,12,0)),(0,dt.datetime(2025,2,5,17,30)),(0,dt.datetime(2025,2,8,11,0)),(0,dt.datetime(2025,2,9,15,0)),(2,dt.datetime(2025,2,19,8,0)),(3,dt.datetime(2025,2,25,17,0))]
fig,ax=plt.subplots(len(cases),1,figsize=(20,3*len(cases)))
for a,(ri,w) in zip(ax,cases):
    r=R[ri]; i0=cio.find_chunk(r,w); tb,y,g,cb=prep.load_binned(r,i0,86)
    a.plot((tb-tb[0])/60,y,lw=.4); a.set_title(f'run{ri} {w:%a %d %H:%M} 1h (50Hz binned)',fontsize=9)
    d=np.abs(np.diff(y)); print(w, 'range',np.ptp(y),'MAD diff',np.median(d))
plt.tight_layout(); plt.savefig('figs/overview_1h.png',dpi=60)
