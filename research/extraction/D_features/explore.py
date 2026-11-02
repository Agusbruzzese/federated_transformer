import numpy as np, glob, os, re
from scipy import signal
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R="/home/augustin/Documents/federated_transformer/Spintronic Sensor Dataset for Vehicle Detection and Car Model Recognition (14 Days, Unlabeled with Timestamps)"
r=sorted(glob.glob(R+"/Run*"))[0]
vs=sorted(glob.glob(r+"/S0C0 BIN Values*"))
i0=[i for i,f in enumerate(vs) if "Start@ 20250205 - 08h00" in f][0]
V=[];T=[]
for f in vs[i0:i0+30]:
    V.append(np.fromfile(f,dtype='<u4').astype(np.int64)); T.append(np.fromfile(f.replace("BIN Values","BIN Time"),dtype='<u8'))
T=np.concatenate(T); V=np.concatenate(V).astype(float)
d=np.diff(T); print("dt stats",np.median(d),d.min(),d.max(), (d>1000).sum())
fs=1e6/np.median(d); print("fs",fs)
sos=signal.butter(4,20,fs=fs,output='sos'); x=signal.sosfiltfilt(sos,V-np.median(V))
xd=x[::24]; fsd=fs/24
fig,ax=plt.subplots(3,1,figsize=(18,10))
tt=np.arange(len(xd))/fsd
ax[0].plot(tt,xd,lw=.5); ax[0].set_title('LP20 decimated, 30 chunks from 08:00')
ax[1].plot(tt[:int(300*fsd)],xd[:int(300*fsd)],lw=.7)
# spectrum of an event vs quiet
k=np.argmax(np.abs(xd)); seg=V[k*24-5*2390:k*24+5*2390]; q=V[:10*2390]
for s,l in [(seg,'event'),(q,'quiet')]:
    f,p=signal.welch(s-s.mean(),fs=fs,nperseg=8192); ax[2].loglog(f,p,label=l)
ax[2].legend(); plt.tight_layout(); plt.savefig('explore_0800.png',dpi=70)
np.save('/tmp/claude-1000/-home-augustin-Documents-federated-transformer/8ccb016b-de96-4655-ab49-d3318afeb915/scratchpad/x0800.npy',xd)
mad=np.median(np.abs(xd-np.median(xd)))*1.4826; print("MAD",mad, "max",np.abs(xd).max())
