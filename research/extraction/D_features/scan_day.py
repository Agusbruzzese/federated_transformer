import numpy as np, time, sys
from io_detect import *
ch = list_chunks(); print(len(ch))
day = [c for c in ch if c[2].strftime("%Y%m%d") == "20250205"]
print(len(day))
t0 = time.time(); ev = []
B = 30
for i in range(0, len(day), B):
    blk = day[i:i+B]
    v, t = load_block([c[3] for c in blk])
    x = lowpass_dec(v); r, sig, segs = detect(x)
    for s, e in segs:
        ev.append((blk[0][2].hour + blk[0][2].minute/60 + s/FSD/3600, (e-s)/FSD, np.abs(r[s:e]).max()/sig, sig, r[s:e].max(), r[s:e].min()))
print("time", time.time()-t0)
ev = np.array(ev); np.save("/tmp/claude-1000/-home-augustin-Documents-federated-transformer/8ccb016b-de96-4655-ab49-d3318afeb915/scratchpad/day0205.npy", ev)
print("n events", len(ev)); print("dur pct", np.percentile(ev[:,1],[5,25,50,75,95,99]))
print("snr pct", np.percentile(ev[:,2],[5,25,50,75,95])); print("sigma pct", np.percentile(ev[:,3],[5,50,95]))
print(np.histogram(ev[:,0], bins=24, range=(0,24))[0])
