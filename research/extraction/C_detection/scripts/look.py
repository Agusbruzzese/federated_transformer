import sys; sys.path.insert(0,'.')
import cio, numpy as np, datetime as dt, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
r=cio.runs()[0]
i0=cio.find_chunk(r, dt.datetime(2025,2,5,8,0,0))
x,t,c=cio.load_chunks(r,i0,15)   # ~10 min
print(len(x), t[-1]-t[0], 'time diffs', np.percentile(np.diff(t),[0,1,50,99,100]))
bd=np.where(np.diff(c))[0]; print('boundary dt', np.diff(t)[bd])
print('fs est', 1/np.median(np.diff(t)))
fig,ax=plt.subplots(4,1,figsize=(18,14))
ax[0].plot(t-t[0],x,lw=.3); ax[0].set_title('raw 10 min 2025-02-05 08:00 run1')
for k,(a,b) in enumerate([(60,120),(200,215),(300,303)]):
    m=(t-t[0]>=a)&(t-t[0]<b); ax[k+1].plot(t[m]-t[0],x[m],lw=.5)
for b in bd: ax[0].axvline(t[b]-t[0],color='r',lw=.3)
plt.tight_layout(); plt.savefig('figs/look_raw.png',dpi=70)
# spectrum
from scipy.signal import welch
f,p=welch(x-x.mean(),fs=2386,nperseg=2**16); 
plt.figure(figsize=(10,4)); plt.loglog(f[1:],p[1:]); plt.savefig('figs/look_psd.png',dpi=70)
for fr in [50,100,150]: 
    j=np.argmin(abs(f-fr)); print(fr, p[j]/np.median(p[j-50:j+50]))
