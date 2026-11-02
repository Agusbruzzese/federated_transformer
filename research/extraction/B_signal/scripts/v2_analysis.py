"""Audit fixes on the v2 end-to-end run (data/v2_events.parquet, data/v2_spans.parquet):
 issue 4  bimodality of amplitude at 5/6/8 sigma; blip/vehicle separation by features other than
          the class-defining duration (log-duration bimodality itself, dropout proximity, run, polarity)
 issue 1  §3.4 morphology tables excluding events < 5 s from a span edge / dropout mask
          (contiguous windows preferred; sampled chunks shown for comparison)
 issue 3  hourly rates with bootstrap-by-day 95 % CIs and number of independent days per hour
 issue 5  morphology per run; run 0204 with 'repaired' events (mask_frac <= 0.2) vs clean only
 headline confirmation vs v1.  Output: data/out_v2_analysis.txt, figures/fig12_bimodality.png, fig13_rates.png"""
import os, numpy as np, pandas as pd
from style import plt, RUNC, INK2
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
FIG = os.path.join(OUT, "..", "figures")
ev = pd.read_parquet(os.path.join(OUT, "v2_events.parquet"))
sp = pd.read_parquet(os.path.join(OUT, "v2_spans.parquet"))
ev["hour"] = pd.to_datetime(ev.start).dt.hour
ev["day"] = pd.to_datetime(ev.start).dt.date
L = []
pr = lambda s: L.append(str(s)) or print(s)
pct = lambda x, q=(5, 50, 95): np.round(np.percentile(np.asarray(x, float)[~np.isnan(np.asarray(x, float))], q), 2)

pr(f"spans: {sp.groupby('kind').size().to_dict()}, hours {sp.groupby('kind').hours.sum().round(1).to_dict()}")
pr(f"detection noise sigma (5 Hz LP, 20 s 2-pass median, 60 s MAD) per span: median {sp.sig_med.median():.1f}, p10/p90 {pct(sp.sig_med,(10,90))}")

# ---------------- issue 4: bimodality vs threshold
pr("\n== issue 4: amplitude histogram (log10 amax, clean = mask_frac 0, dur >= 0.4 s) per threshold, sampled chunks ==")
bins = np.arange(2.2, 5.6, 0.2)
fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
for k, thr in enumerate([5.0, 6.0, 8.0]):
    s = ev[(ev.thr == thr) & (ev.mask_frac == 0)]
    for kind in ("chunk", "window"):
        ss = s[(s.kind == kind) & (s.dur >= 0.4)]
        h, _ = np.histogram(np.log10(ss.amax), bins)
        pr(f"thr {thr} {kind:6s} n={len(ss):5d}  hist " + " ".join(f"{v}" for v in h))
        # valley test: density in 3-5k vs 1.5-3k and 5-10k
        a = ((ss.amax > 1500) & (ss.amax <= 3000)).sum(); b = ((ss.amax > 3000) & (ss.amax <= 5000)).sum(); c = ((ss.amax > 5000) & (ss.amax <= 10000)).sum()
        pr(f"      per-dex density: 1.5-3k {a/np.log10(2):.0f}, 3-5k {b/np.log10(5/3):.0f}, 5-10k {c/np.log10(2):.0f}")
    ss = s[(s.kind == "chunk") & (s.dur >= 0.4)]
    ax[0].hist(np.log10(ss.amax), bins=bins, histtype="step", lw=1.5, color=RUNC[k], label=f"{thr:.0f}σ (n={len(ss)})")
    ax[1].hist(np.log10(s[s.kind == "chunk"].dur), bins=np.linspace(-1, 1.3, 47), histtype="step", lw=1.5, color=RUNC[k], label=f"{thr:.0f}σ")
ax[0].set_xlabel("log10 peak |deviation| (counts)"); ax[0].set_title("Amplitude, clean events ≥0.4 s, by trigger"); ax[0].legend(fontsize=7)
ax[1].set_xlabel("log10 duration above 2σ (s)"); ax[1].set_title("Duration of ALL clean detections (no class cut)"); ax[1].legend(fontsize=7)

pr("\n== issue 4: blip/vehicle separation with features that do not define the class ==")
s = ev[(ev.thr == 8.0) & (ev.kind == "chunk")].copy()
s["short"] = s.dur < 0.4
sp_ch = sp[sp.kind == "chunk"].set_index("span")
s["n_lost"] = s.span.map(sp_ch.n_lost)
s["lossy"] = s.n_lost > 500
for nm, g in s.groupby("short"):
    pr(f"short={nm}: n={len(g)}, frac in run0 {np.mean(g.run == 0):.2f}, frac in span with >500 lost samples {g.lossy.mean():.2f}, "
       f"frac touching mask {np.mean(g.mask_frac > 0):.2f}, frac within 0.5 s of mask {np.mean(g.edge_dist < 0.5):.2f}, "
       f"positive dominant polarity {np.mean(g.dom_pol > 0):.2f}, amax median {g.amax.median():.0f}")
tot_h = sp_ch.groupby("run").hours.sum()
for nm, g in s[s.mask_frac == 0].groupby("short"):
    pr(f"  clean short={nm} rate per hour by run: {(g.groupby('run').size() / tot_h).round(1).to_dict()}")
lossbin = pd.cut(sp_ch.n_lost, [-1, 0, 300, 1000, 3000, 1e9])
for nm, g in s[s.mask_frac == 0].groupby("short"):
    cnt = g.groupby("span").size().reindex(sp_ch.index, fill_value=0)
    pr(f"  clean short={nm} per chunk vs lost samples in chunk: {cnt.groupby(lossbin).mean().round(2).to_dict()}")
# log-duration density valley (no class cut)
h, e = np.histogram(np.log10(s[s.mask_frac == 0].dur), np.linspace(-1, 1.2, 23))
pr("  log10(dur) histogram of ALL clean 8σ detections (edges -1..1.2 step .1): " + " ".join(map(str, h)))
for kind_run, g in [("runs 1-3", s[(s.mask_frac == 0) & (s.run > 0)]), ("run 0", s[(s.mask_frac == 0) & (s.run == 0)])]:
    h, _ = np.histogram(np.log10(g.dur), np.linspace(-1, 1.2, 23)); pr(f"  {kind_run}: " + " ".join(map(str, h)))
ax[2].hist(np.log10(s[(s.mask_frac == 0) & (s.run == 0)].dur), bins=np.linspace(-1, 1.2, 23), histtype="step", lw=1.5, color=RUNC[0], label="run 0204 (lossy)")
ax[2].hist(np.log10(s[(s.mask_frac == 0) & (s.run > 0)].dur), bins=np.linspace(-1, 1.2, 23), histtype="step", lw=1.5, color=RUNC[1], label="runs 0212-0224")
ax[2].set_xlabel("log10 duration (s)"); ax[2].set_title("8σ clean detections: duration by run"); ax[2].legend(fontsize=7)
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig12_bimodality.png"), dpi=110)

# ---------------- issue 1: tables excluding edge-near events
def table(g, title):
    pr(f"\n{title}  n={len(g)}")
    for col in ["dur", "d90", "amax", "p2p", "snr", "rise", "fwhm"]:
        pr(f"   {col:5s} p5/p50/p95 = {pct(g[col])}")
    pr(f"   pattern: {g.pattern.value_counts(normalize=True).round(3).head(7).to_dict()}")
    pr(f"   dominant polarity negative: {np.mean(g.dom_pol < 0):.2f}")
base = ev[(ev.thr == 8.0) & (ev.dur >= 0.4) & (ev.mask_frac == 0)]
far = base[base.edge_dist >= 5]
pr("\n== issue 1: vehicle-like (8σ, ≥0.4 s, clean) — events ≥5 s from span edge and dropout mask ==")
pr(f"exclusion effect: sampled chunks keep {np.mean(base[base.kind=='chunk'].edge_dist >= 5):.2f}, windows keep {np.mean(base[base.kind=='window'].edge_dist >= 5):.2f}")
for kind in ("window", "chunk"):
    g = far[far.kind == kind]
    table(g[g.amax > 5000], f"[{kind}] LARGE (>5k)")
    table(g[g.amax <= 5000], f"[{kind}] SMALL+MEDIUM (<=5k)")
    table(g, f"[{kind}] ALL vehicle-like")
g = far[far.kind == "window"]
pr(f"\nwindows, all vehicle-like: dur>8 s: {np.mean(g.dur > 8):.3f}, dur>12 s: {np.mean(g.dur > 12):.3f}; unipolar among dur>8 s: {np.mean(g[g.dur > 8].nlobes == 1):.2f}")
day = g.hour.between(7, 18)
pr(f"windows day(07-18h) vs night: amax median {g[day].amax.median():.0f} vs {g[~day].amax.median():.0f}; frac>5k {np.mean(g[day].amax > 5000):.2f} vs {np.mean(g[~day].amax > 5000):.2f}; "
   f"dur median {g[day].dur.median():.2f} vs {g[~day].dur.median():.2f}; n {day.sum()} vs {(~day).sum()}")

# ---------------- issue 3: rates with bootstrap by day
pr("\n== issue 3: hourly rates from sampled chunks (8σ, ≥0.4 s, mask_frac<=0.2), bootstrap over days (2000 reps) ==")
ch = sp[sp.kind == "chunk"].copy()
ch["hour"] = pd.to_datetime(ch.start).dt.hour; ch["day"] = pd.to_datetime(ch.start).dt.date
e = ev[(ev.thr == 8.0) & (ev.kind == "chunk") & (ev.dur >= 0.4) & (ev.mask_frac <= 0.2)]
rng = np.random.default_rng(0)
rows = []
for h in range(24):
    c = ch[ch.hour == h]
    days = sorted(c.day.unique())
    hrs = c.groupby("day").hours.sum()
    eh = e[e.hour == h]
    nall = eh.groupby("day").size().reindex(days, fill_value=0)
    nbig = eh[eh.amax > 5000].groupby("day").size().reindex(days, fill_value=0)
    full = [d for d in days if hrs[d] > 0.03]  # >= ~3 chunks of that hour on that day
    bs_a, bs_b = [], []
    for _ in range(2000):
        dd = rng.choice(days, len(days))
        bs_a.append(nall[dd].sum() / hrs[dd].sum()); bs_b.append(nbig[dd].sum() / hrs[dd].sum())
    rows.append(dict(hour=h, days=len(days), days_ge3chunks=len(full), hours=hrs.sum(), rate_all=nall.sum() / hrs.sum(),
                     all_lo=np.percentile(bs_a, 2.5), all_hi=np.percentile(bs_a, 97.5), rate_big=nbig.sum() / hrs.sum(),
                     big_lo=np.percentile(bs_b, 2.5), big_hi=np.percentile(bs_b, 97.5)))
R = pd.DataFrame(rows)
pr(R.round(2).to_string(index=False))
fig, ax = plt.subplots(figsize=(11, 4))
ax.errorbar(R.hour - 0.12, R.rate_all, yerr=[R.rate_all - R.all_lo, R.all_hi - R.rate_all], fmt="o", ms=4, color=RUNC[0], label="all vehicle-like")
ax.errorbar(R.hour + 0.12, R.rate_big, yerr=[R.rate_big - R.big_lo, R.big_hi - R.rate_big], fmt="s", ms=4, color=RUNC[1], label="amax > 5000")
ax.set_xlabel("hour of day"); ax.set_ylabel("events per hour (95 % CI, bootstrap by day)"); ax.legend(fontsize=8)
ax.set_title("Hourly event rate, every-8th-chunk sample (8σ trigger)")
fig.tight_layout(); fig.savefig(os.path.join(FIG, "fig13_rates.png"), dpi=110)
w = ev[(ev.thr == 8.0) & (ev.kind == "window") & (ev.dur >= 0.4) & (ev.mask_frac <= 0.2)]
pr("\ncontiguous windows: large (>5k) count per 2-h window: " + str(w[w.amax > 5000].groupby("span").size().reindex(sp[sp.kind == 'window'].span, fill_value=0).to_dict()))
# overlap lower bound
ov = []
for spn, g in w.sort_values("a").groupby("span"):
    gb = g[g.amax > 5000]
    if len(gb) < 2: continue
    st = g.a.values / 50; en = g.e.values / 50
    for i, (a_, e_) in enumerate(zip(gb.a.values / 50, gb.e.values / 50)):
        others = [(x, y) for x, y in zip(st, en) if not (x == a_)]
        near = any((x < e_ + 5) and (y > a_ - 5) for x, y in others)
        ov.append(dict(span=spn, near=near, long=(e_ - a_) > 8))
ov = pd.DataFrame(ov)
pr(f"large events with another detection within 5 s (lower bound on overlap): {ov.near.mean():.3f} (n={len(ov)}); extent>8 s (possible merged pairs): {ov.long.mean():.3f}")

# ---------------- issue 5: per-run morphology
pr("\n== issue 5: large events (>5k, 8σ, ≥5 s from edge/mask) per run; run 0 also with repaired events (mask_frac<=0.2) ==")
lg = ev[(ev.thr == 8.0) & (ev.dur >= 0.4) & (ev.amax > 5000) & (ev.edge_dist >= 5)]
for r_ in range(4):
    g = lg[(lg.run == r_) & (lg.mask_frac == 0)]
    pr(f"run {r_}: n={len(g)}  amax p50 {g.amax.median():.0f}  dur p50 {g.dur.median():.2f}  d90 p50 {g.d90.median():.2f}  rise p50 {g.rise.median():.3f}  fwhm p50 {g.fwhm.median():.2f}  "
       f"+- {np.mean(g.pattern == '+-'):.2f}  nlobes mean {g.nlobes.mean():.2f}")
lg2 = ev[(ev.thr == 8.0) & (ev.dur >= 0.4) & (ev.amax > 5000) & (ev.run == 0) & (ev.mask_frac > 0) & (ev.mask_frac <= 0.2)]
pr(f"run 0 repaired (0<mask_frac<=0.2): n={len(lg2)}  amax p50 {lg2.amax.median():.0f}  dur p50 {lg2.dur.median():.2f}  rise p50 {lg2.rise.median():.3f}  +- {np.mean(lg2.pattern == '+-'):.2f}")
lg3 = ev[(ev.thr == 8.0) & (ev.dur >= 0.4) & (ev.amax > 5000) & (ev.run == 0)]
pr(f"run 0 all large incl. masked: n={len(lg3)}; clean {np.mean(lg3.mask_frac == 0):.2f}, repairable (<=0.2) {np.mean(lg3.mask_frac <= 0.2):.2f}")
from scipy.stats import mannwhitneyu
g0 = lg[(lg.run == 0) & (lg.mask_frac == 0)]; g1 = lg[(lg.run > 0) & (lg.mask_frac == 0)]
for col in ["amax", "dur", "d90", "rise", "fwhm", "nlobes"]:
    pr(f"  run0 vs runs1-3 {col}: medians {g0[col].median():.3f} vs {g1[col].median():.3f}, Mann-Whitney p={mannwhitneyu(g0[col], g1[col]).pvalue:.3g}")
open(os.path.join(OUT, "out_v2_analysis.txt"), "w").write("\n".join(L))
