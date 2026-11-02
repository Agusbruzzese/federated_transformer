import sys; sys.path.insert(0,'.')
import cio, prep, numpy as np, datetime as dt, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy.ndimage import median_filter
r=cio.runs()[2]; i0=cio.find_chunk(r,dt.datetime(2025,2,19,8,25)); tb,y,g,cb=prep.load_binned(r,i0,22)
base=median_filter(y,size=int(30*50),mode='nearest'); z=y-base
fig,ax=plt.subplots(8,1,figsize=(20,24))
T=(tb-tb[0])
for k in range(4):
    m=(T>=k*150)&(T<(k+1)*150); ax[k].plot(T[m],z[m],lw=.5); ax[k].set_ylim(-3000,3000); ax[k].grid()
# individual zooms of big events
from scipy.signal import find_peaks
pk,_=find_peaks(np.abs(z),height=5000,distance=250)
for a,p in zip(ax[4:],pk[:4]):
    s=slice(max(p-400,0),p+400); a.plot(T[s],z[s]); a.grid()
plt.tight_layout(); plt.savefig('figs/zoom_busy.png',dpi=55)
print(len(pk), T[pk])
