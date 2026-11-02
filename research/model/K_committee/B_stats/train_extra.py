"""Committee B: extra checkpoints for variance estimation and planted controls.
Exact M5 SimCLR-TF recipe (m5.train), only the output dir and (for the degraded control) the step count change.

    python research/model/K_committee/B_stats/train_extra.py full 5 6 7      # extra central seeds, 10k steps
    python research/model/K_committee/B_stats/train_extra.py short 0 1 2 3 4 # degraded control: 1k steps (same schedule shape)
Outputs: data/k_committee/B_stats/{full,short}/ckpt/simclr_tf_s{seed}/{init,final}.pt, log.json
"""
import os, sys
ROOT = os.getcwd(); sys.path.insert(0, os.path.join(ROOT, "research/model/M5_long"))
import m5

mode = sys.argv[1]; seeds = [int(s) for s in sys.argv[2:]]
m5.OUT = os.path.join(ROOT, "data", "k_committee", "B_stats", mode)
m5.STEPS = 10_000 if mode == "full" else 1_000
for s in seeds:
    m5.train("simclr_tf", s)
