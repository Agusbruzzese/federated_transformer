"""C1: SimCLR transformer (M5 recipe, 10k steps) trained on ALL SNR>=20 events of the whole timeline.
    python research/model/C1_clustering/c1_train.py <seed>"""
import os, sys, numpy as np
ROOT = os.getcwd(); sys.path[:0] = [ROOT, f"{ROOT}/research/model/M4_methods", f"{ROOT}/research/model/M5_long"]
import m4, m5
from ssl_central import config as C
_load = m4.load
def load_all_train():
    m, W = _load(); m = m.copy(); m["role"] = "train"; return m, W          # whole timeline, no held-out split
m4.load = load_all_train
m4.bgpool = lambda: (np.load(os.path.join(C.OUT, "bgpool.npy")), 0)           # full background pool (all days)
m5.OUT = os.path.join(ROOT, "data", "c1")
if __name__ == "__main__":
    m5.train("simclr_tf", int(sys.argv[1]))
