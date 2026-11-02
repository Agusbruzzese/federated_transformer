"""Full-dataset metadata scan: file names, sizes, first/last Time & Values samples (tiny reads only)."""
import os, re, sys, numpy as np, pandas as pd
from datetime import datetime
ROOT = "/home/augustin/Documents/federated_transformer/Spintronic Sensor Dataset for Vehicle Detection and Car Model Recognition (14 Days, Unlabeled with Timestamps)"
PAT = re.compile(r"S0C0 BIN (Values|Time) - (\d+) Start@ (\d{8}) - (\d\d)h(\d\d)m(\d\d)s Stop@ (\d{8}) - (\d\d)h(\d\d)m(\d\d)s\.bsd$")
def dt(d, h, m, s): return datetime.strptime(f"{d}{h}{m}{s}", "%Y%m%d%H%M%S")
rows = []
for run in sorted(os.listdir(ROOT)):
    rd = os.path.join(ROOT, run)
    if not os.path.isdir(rd): continue
    for fn in os.listdir(rd):
        m = PAT.match(fn)
        if not m:
            continue
        kind, idx = m.group(1), int(m.group(2))
        p = os.path.join(rd, fn); size = os.path.getsize(p)
        item = 8 if kind == "Time" else 4
        dtp = "<u8" if kind == "Time" else "<u4"
        with open(p, "rb") as f:
            first = np.frombuffer(f.read(item), dtp)[0]
            f.seek(size - item); last = np.frombuffer(f.read(item), dtp)[0]
        rows.append(dict(run=run, kind=kind, index=idx, fn_start=dt(*m.group(3, 4, 5, 6)), fn_stop=dt(*m.group(7, 8, 9, 10)),
                         path=p, size=size, first=int(first), last=int(last)))
df = pd.DataFrame(rows)
df.to_parquet(os.path.join(os.path.dirname(__file__), "raw_header_scan.parquet"))
print(df.groupby(["run", "kind"]).size())
