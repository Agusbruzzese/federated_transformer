"""Template bank for synthetic injection: clear isolated real events from TUNING spans only.
usage: templates.py [hyst|bdet|wavelet]  -- which detector SELECTS the template events (audit issue 2):
  hyst    -> data/templates.npz          (original: HYST k_on=4 on the detection channel)
  bdet    -> data/templates_bdet.npz     (workstream B's detector at its native 8 sigma)
  wavelet -> data/templates_wavelet.npz  (WAVELET at its frozen threshold)
Same isolation (no other same-detector event within 20 s), SNR>40 and taper rules for every set."""
import sys, os; sys.path.insert(0,'.'); sys.path.insert(0, os.path.dirname(__file__))
from common import *
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
FS=D.FS; H=int(15*FS)
import cio, datetime as dt
from build_spans import load as load_raw
EXTRA=[(0,dt.datetime(2025,2,6,7,45)),(0,dt.datetime(2025,2,10,7,45)),(0,dt.datetime(2025,2,11,7,45)),
       (1,dt.datetime(2025,2,14,7,45)),(1,dt.datetime(2025,2,12,17,0)),(3,dt.datetime(2025,2,26,7,45)),
       (3,dt.datetime(2025,2,27,7,45)),(3,dt.datetime(2025,2,25,7,45))]   # template harvesting only (tuning runs)
def sources():
    for m in tune_spans():
        s=load_span(m['name']); yield m['name'], s['t'], s['y']
    for ri,w in EXTRA:
        r=cio.runs()[ri]; i0=cio.find_chunk(r,w); x,tt,_,_=load_raw(r,i0,44)
        t50,y50,_,_=D.front_end(x,tt); yield f'x_r{ri+1}_{w:%d%H%M}', t50, y50
SEL=sys.argv[1] if len(sys.argv)>1 else 'hyst'
OUT='data/templates.npz' if SEL=='hyst' else f'data/templates_{SEL}.npz'
T=[]; info=[]
for name,t,y in sources():
    P=prep(y); m={'name':name}
    if SEL=='hyst': segs=D.detect_hyst(t,P['zd'],P['sigma'],dict(D.DEFAULTS,k_on=4.0))   # liberal, for isolation test
    elif SEL=='bdet': segs=A.run('BDET',t,P['y'],P['zd'],P['sigma'],{'k_hi':8.0})
    else: segs=A.run('WAVELET',t,P['y'],P['zd'],P['sigma'],{'k_on':10.0,'k_off':2.0})
    ev=D.describe(segs,t,P['z'],P['sigma'])
    pk=np.array([e['peak_t'] for e in ev])
    for e in ev:
        if e['snr']<40: continue
        others=np.abs(pk-e['peak_t']); others=others[others>0]
        # isolation: no other detection's peak within 20 s, and the event's own extent within +-12 s of its peak
        if len(others) and others.min()<20: continue
        if e['e_on']<e['peak_t']-12 or e['e_off']>e['peak_t']+12: continue
        j=int(round((e['peak_t']-t[0])*FS))
        if j-H<0 or j+H+1>len(t): continue
        w=P['z'][j-H:j+H+1].copy()
        c=np.cumsum(w**2); c/=c[-1]; a=np.searchsorted(c,0.005); b=np.searchsorted(c,0.995)
        tap=np.zeros_like(w); L=int(1*FS)
        tap[a:b]=1; tap[max(a-L,0):a]=np.hanning(2*L)[:L][-(a-max(a-L,0)):] if a>0 else 0
        tap[b:min(b+L,len(w))]=np.hanning(2*L)[L:][:min(b+L,len(w))-b]
        w=w*tap; w/=np.abs(w).max()
        c=np.cumsum(w**2); c/=c[-1]
        T.append(w); info.append((m['name'],e['peak_t'],e['snr'],np.searchsorted(c,0.05),np.searchsorted(c,0.95),np.searchsorted(c,0.01),np.searchsorted(c,0.99)))
T=np.array(T); info=np.array(info,dtype=object)
print('templates',len(T),'from spans',sorted(set(info[:,0])))
gt=np.array(info[:,3:],float)/FS-15
print('core 5-95% dur median',np.median(gt[:,1]-gt[:,0]),'1-99% dur median',np.median(gt[:,3]-gt[:,2]), 'max', (gt[:,3]-gt[:,2]).max())
np.savez(OUT,T=T,span=info[:,0].astype(str),peak_t=info[:,1].astype(float),snr=info[:,2].astype(float),
         gt5=gt[:,0],gt95=gt[:,1],gt1=gt[:,2],gt99=gt[:,3])
n=len(T); cols=6; rows=int(np.ceil(n/cols))
fig,ax=plt.subplots(rows,cols,figsize=(18,1.8*rows)); ax=ax.ravel()
tt=np.arange(-H,H+1)/FS
for i in range(n):
    ax[i].plot(tt,T[i],lw=.7); ax[i].axvspan(gt[i,0],gt[i,1],alpha=.2); ax[i].set_title(f'{i} {info[i,0][:8]} snr{info[i,2]:.0f}',fontsize=7); ax[i].tick_params(labelsize=6)
for a in ax[n:]: a.axis('off')
plt.tight_layout(); plt.savefig('figs/templates.png' if SEL=='hyst' else f'figs/templates_{SEL}.png',dpi=60)
