import numpy as np, glob, os
R="/home/augustin/Documents/federated_transformer/Spintronic Sensor Dataset for Vehicle Detection and Car Model Recognition (14 Days, Unlabeled with Timestamps)"
runs=sorted(glob.glob(R+"/Run*"))
for r in runs:
    vs=sorted(glob.glob(r+"/S0C0 BIN Values*"))
    for f in [vs[0], vs[len(vs)//2]]:
        v=np.fromfile(f,dtype='<u4'); t=np.fromfile(f.replace("BIN Values","BIN Time"),dtype='<u8')
        print(os.path.basename(f)[:60], v.shape, v.min(), v.max(), np.median(v), v.std(), t[:2], np.diff(t)[:5], np.median(np.diff(t)), (t[-1]-t[0])/1e6)
