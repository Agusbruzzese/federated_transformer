# Reviewer: dump frozen-checkpoint embeddings (no training, no probing) to scratch for independent analysis.
import sys, os, numpy as np
sys.path.insert(0, "research/model/M5_long"); sys.path.insert(0, ".")
import m5
out = sys.argv[1]; D = {}
for meth in ["simclr_tf", "vicreg_tf", "simclr_cnn", "mae_tf"]:
    for s in range(5):
        D[f"{meth}_s{s}"] = m5.emb(meth, s, "final")
for s in range(5):
    D[f"B1_s{s}"] = m5.emb("simclr_cnn", s, "init")
D["crop"] = m5.crops().cpu().numpy()
np.savez(out, **D); print({k: v.shape for k, v in D.items()})
