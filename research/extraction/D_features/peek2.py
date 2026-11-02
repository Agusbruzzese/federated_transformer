import numpy as np, glob, os, re
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R="/home/augustin/Documents/federated_transformer/Spintronic Sensor Dataset for Vehicle Detection and Car Model Recognition (14 Days, Unlabeled with Timestamps)"
r=sorted(glob.glob(R+"/Run*"))[0]
vs=sorted(glob.glob(r+"/S0C0 BIN Values*"))
sel=[f for f in vs if re.search(r"Start@ 20250205 - (08|12|17|03)h(00|01)m",f)][:4]
fig,ax=plt.subplots(len(sel),1,figsize=(16,3*len(sel)))
for a,f in zip(ax,sel):
    v=np.fromfile(f,dtype='<u4').astype(float); t=np.fromfile(f.replace("BIN Values","BIN Time"),dtype='<u8')/1e6
    a.plot(t-t[0],v-np.median(v),lw=.4); a.set_title(os.path.basename(f)[17:60])
plt.tight_layout(); plt.savefig('peek_chunks.png',dpi=70)
# zoom
f=sel[0]; v=np.fromfile(f,dtype='<u4').astype(float)
fig,ax=plt.subplots(2,1,figsize=(16,6)); ax[0].plot(v[:5000]); 
from scipy.signal import welch
fr,p=welch(v-v.mean(),fs=2390,nperseg=8192); ax[1].loglog(fr,p); plt.savefig('peek_zoom.png',dpi=70)
print(sel)
