"""Figures for report 49 (consolidation addendum). Read-only: uses saved result files and saved models; writes NEW png files only.
  fig1  forest plot: the key comparisons (lobes, 90 % CI) across the study, the replication and the follow-ups, against the margin
  fig2  embedding maps: unseen evaluation events of fold 0 in the embedding of central, federated and server-pretrained federated
        (study models, seed 0), t-SNE, coloured by lobe count; the "clustering" view of the trained models
    .venv/bin/python research/model/R_replication/figs49.py
"""
import json, os, sys, glob
import numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../..")); os.chdir(ROOT)
sys.path[:0] = [os.path.join(ROOT, "research/model/K_committee/protocol"), ROOT, os.path.join(ROOT, "research/model/M4_methods")]
OUT = os.path.join(ROOT, "reports/figures"); os.makedirs(OUT, exist_ok=True)
INK, INK2, MUTED, GRID, SURF = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
R = lambda p: json.load(open(sorted(glob.glob(p))[-1]))
plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
                     "axes.facecolor": SURF, "figure.facecolor": SURF, "savefig.facecolor": SURF})


def lob(c): m = c["metrics"]["LOBES"]; return m["delta"], m["L"], m["U"]


def fig1():
    st = R("research/model/K_committee/protocol/results_main_*.json")["contrasts"]; rp = R("research/model/R_replication/results_rep_*.json")
    r3 = R("research/model/R_replication/results_rep3_*.json"); r4 = R("research/model/R_replication/results_rep4_*.json")
    sa = R("fl_system3/results_M3cp_*.json")["primary_seeds_3_9"]["LOBES"]
    rows = [("A  Federated vs central", None),
            ("study, clean events (rep. 24)", lob(st["F-nat vs C256"])),
            ("new events, noisy labels (rep. 42)", lob(rp["contrasts"]["F-nat vs C256"])),
            ("new events, clean labels (rep. 46)", lob(r3["others"]["Fnat_vs_C256"])),
            ("B  Server pretraining (S+F vs federated)", None),
            ("study, clean events (rep. 24)", lob(st["S+F vs F-nat"])),
            ("new events, noisy labels (rep. 42)", lob(rp["contrasts"]["S+F vs F-nat"])),
            ("new events, clean labels (rep. 46)", lob(r3["primary_contrasts"]["S+F vs F-nat"])),
            ("C  Secure aggregation vs plain", None),
            ("unseen seeds 3-9 (rep. 36)", (sa["delta"], sa["CI90"][0], sa["CI90"][1])),
            ("new events (rep. 42)", (rp["secagg_fold0"]["LOBES"]["delta"], rp["secagg_fold0"]["LOBES"]["L"], rp["secagg_fold0"]["LOBES"]["U"])),
            ("D  Training data for S+F (vs clean only)", None),
            ("all events SNR >= 10 (rep. 48)", lob(r4["contrasts"]["S+F: T10 vs T20"])),
            ("noisy events only (rep. 48)", lob(r4["contrasts"]["S+F: TLO vs T20"]))]
    fig, ax = plt.subplots(figsize=(9.5, 7.2)); m = 0.04; y = len(rows)
    ax.axvspan(-m, m, color=GRID, zorder=0); ax.axvline(0, color=MUTED, lw=1, zorder=1)
    ax.text(m, len(rows) + 0.35, "  equivalence margin ±0.04", color=INK2, fontsize=8.5, va="center", ha="right")
    yt, yl = [], []
    for label, v in rows:
        y -= 1
        if v is None:
            ax.text(-0.135, y, label, fontsize=10.5, fontweight="bold", color=INK, va="center"); continue
        d, L, U = v; ax.plot([L, U], [y, y], color=BLUE, lw=2, solid_capstyle="round", zorder=3)
        ax.plot(d, y, "o", ms=8, color=BLUE, mec=SURF, mew=2, zorder=4)
        ax.text(0.128, y, f"{d:+.3f} [{L:+.3f}, {U:+.3f}]", fontsize=8.5, color=INK2, va="center", ha="right")
        yt.append(y); yl.append(label)
    ax.set_yticks(yt); ax.set_yticklabels(yl, color=INK2); ax.set_xlim(-0.135, 0.13); ax.set_ylim(-0.7, len(rows) + 0.7)
    ax.set_xlabel("difference in lobe-count skill (first minus second), 90 % CI")
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0); ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_title("Key comparisons on lobe structure, 10 seeds x 9 unseen days", color=INK, loc="left", fontsize=12)
    fig.text(0.01, 0.01, "Grey band: within ±0.04 counts as equivalent. Inside the band = no meaningful difference; "
             "right of 0 = first model better.", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1)); p = os.path.join(OUT, "49_fig1_key_comparisons.png"); assert not os.path.exists(p); fig.savefig(p, dpi=150); return p


def fig2():
    import torch, evaluator as E
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    C = E.data(); tr, ev, days = E.fold_sets(0); lob = np.minimum(C["tg"]["lobes"][ev], 3)
    models = [("Central (C256)", "v21_C256_f0_s0"), ("Federated, 10 sites (F-nat)", "v21_F-nat_f0_s0"),
              ("Server-pretrained federated (S+F)", "v21_S+F_f0_s0")]
    fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.8))
    groups = [(1, "1 lobe", BLUE, "o"), (2, "2 lobes", ORANGE, "s"), (3, "3+ lobes", AQUA, "^")]
    for ax, (title, tag) in zip(axs, models):
        enc = E.load_encoder(os.path.join(ROOT, "data/k_committee/protocol/runs3", tag, "final.pt"))
        Z = E.embed(enc, ev); Z = Z.cpu().numpy() if hasattr(Z, "cpu") else np.asarray(Z)
        Y = TSNE(2, perplexity=30, random_state=0, init=PCA(2, random_state=0).fit_transform(Z)).fit_transform(Z)
        for g, name, col, mk in groups:
            k = lob == g; ax.scatter(Y[k, 0], Y[k, 1], s=9, c=col, marker=mk, alpha=0.75, linewidths=0, label=f"{name} (n={int(k.sum())})")
        ax.set_title(title, color=INK, fontsize=11, loc="left"); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values(): s.set_visible(False)
    axs[0].legend(loc="lower left", fontsize=8.5, frameon=False, markerscale=1.8, labelcolor=INK2)
    fig.suptitle(f"Unseen events (fold 0, days {', '.join(days)}, n={len(ev)}) in each model's embedding (t-SNE, seed 0)",
                 color=INK, fontsize=12, x=0.01, ha="left")
    fig.text(0.01, 0.01, "Same events in all three panels. Colour = lobe count from the waveform (no vehicle labels exist). "
             "Shapes form a continuum with soft regions, not discrete clusters (report 03).", fontsize=8, color=MUTED)
    fig.tight_layout(rect=(0, 0.04, 1, 0.94)); p = os.path.join(OUT, "49_fig2_embedding_maps.png"); assert not os.path.exists(p); fig.savefig(p, dpi=150); return p


if __name__ == "__main__":
    print(fig1()); print(fig2())
