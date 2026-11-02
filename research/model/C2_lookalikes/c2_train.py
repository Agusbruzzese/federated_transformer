"""C2 LORO: SimCLR-TF (M5 recipe, 10k steps) trained on all SNR>=20 events EXCEPT run <r>.
    python research/model/C2_lookalikes/c2_train.py <held_run> <seed>"""
import os, sys, numpy as np, pandas as pd
ROOT = os.getcwd(); sys.path[:0] = [ROOT, f"{ROOT}/research/model/M4_methods", f"{ROOT}/research/model/M5_long"]
import m4, m5
from ssl_central import config as C
HELD = int(sys.argv[1]); SEED = int(sys.argv[2])
_load = m4.load
m, _ = _load(); HELD_DAYS = set(m.loc[m.run_id == HELD, "date"].astype(str))
def load_fold():
    m, W = _load(); m = m.copy(); m["role"] = np.where(m.run_id.to_numpy() == HELD, "test", "train"); return m, W
def bg_fold():
    bg = np.load(os.path.join(C.OUT, "bgpool.npy")); meta = pd.read_parquet(os.path.join(C.OUT, "bgpool_meta.parquet"))
    d = pd.to_datetime(meta.wall_time).dt.date.astype(str).to_numpy(); return bg[~np.isin(d, list(HELD_DAYS))], 0
m4.load = load_fold; m4.bgpool = bg_fold
m5.OUT = os.path.join(ROOT, "data", "c2", f"loro_r{HELD}")
if __name__ == "__main__":
    m5.train("simclr_tf", SEED)
