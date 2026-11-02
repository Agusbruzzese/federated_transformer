import sys; sys.path.insert(0,'.')
from common import *
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt, time
s=load_span('r1_wed05_0745'); t=s['t']; T=time.time(); P=prep(s['y']); z=P['zd']; sig=P['sigma']; print('prep s',time.time()-T)
T=time.time(); segs=detect_all(t,P); print('det s',time.time()-T)
for k,v in segs.items(): print(k,len(v), 'median dur', np.median(np.diff(v,axis=1))/D.FS if len(v) else 0)
fig,ax=plt.subplots(3,1,figsize=(20,12))
for a,(t0,t1) in zip(ax,[(150,450),(450,750),(750,1050)]):
    m=(t>=t0)&(t<t1); a.plot(t[m],z[m],lw=.5,c='k'); a.plot(t[m],5*sig[m],c='gray',lw=.5); a.plot(t[m],-5*sig[m],c='gray',lw=.5)
    a.set_ylim(-4000,4000)
    for j,(k,v) in enumerate(segs.items()):
        for (i0,i1) in v:
            if t[i0]<t1 and t[min(i1,len(t)-1)]>t0: a.axvspan(t[i0],t[min(i1,len(t)-1)],ymin=0.02+j*0.06,ymax=0.07+j*0.06,color=f'C{j}')
ax[0].set_title(' / '.join(f'C{j}={k}' for j,k in enumerate(segs)))
plt.tight_layout(); plt.savefig('figs/sanity_r1.png',dpi=55)
