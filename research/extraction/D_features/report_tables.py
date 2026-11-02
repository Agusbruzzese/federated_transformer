"""Regenerate the markdown tables used in REPORT.md from eval_*_full.json and friends -> report_tables.md"""
import json, os, numpy as np, pandas as pd
from features import LEGACY_NAMES

f3 = lambda v: f"{v[0]:.2f} [{v[1]:.2f},{v[2]:.2f}]"
out = []


def md(df, index=True):
    df = df.reset_index() if index else df
    cols = [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, float) else str(v) for v in r.values) + " |")
    return "\n".join(lines)


def main_table(tag, title):
    p = f"eval_{tag}_full.json"
    if not os.path.exists(p):
        return
    R = json.load(open(p))
    out.append(f"\n### {title}  (`{p}`)\n")
    out.append("| split | rep | d | invariance R@1 (null 1/n) | leak date skill (shuffled null) | leak hour skill (null) | leak run skill | sees dropout (skill) | silhouette k=5 real / col-perm null | boot-ARI k=5 real / null | logdur R² | polarity skill | event-vs-bg skill | PR (eff. dim) |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in R:
        inv = r.get("inv", {}).get("invariance_R@1")
        lr = r.get("leak_run", {}).get("skill"); ld = r.get("leak_dropout", {}).get("skill")
        c = r["clu"]; p_ = r["prox"]
        out.append("| " + " | ".join([
            r["split"], r["rep"], str(r["red"]["d"]),
            f3(inv) if inv else "–",
            f"{f3(r['leak_date']['skill'])} ({r['leak_date']['null_shuffled'][0]:.2f})",
            f"{f3(r['leak_hour']['skill'])} ({r['leak_hour']['null_shuffled'][0]:.2f})",
            f3(lr) if lr else "–", f3(ld) if ld else "–",
            f"{f3(c['real']['sil'])} / {c['null']['sil'][0]:.2f}",
            f"{f3(c['real']['ari'])} / {c['null']['ari'][0]:.2f}",
            f3(p_["logdur_R2"]), f3(p_["polarity_skill"]),
            f3(p_["event_vs_bg_skill"]) if "event_vs_bg_skill" in p_ else "–",
            f"{r['red']['PR'][0]:.1f}"]) + " |")
    S = pd.read_csv(f"eval_{tag}_summary.csv")
    out.append("\nMean rank over the 6 pre-registered criteria (1 = best):\n")
    piv = S.pivot(index="rep", columns="split", values="mean_rank").sort_values(S.split.iloc[0])
    out.append(md(piv.round(2)))
    st = json.load(open(f"eval_{tag}_stability.json"))
    out.append(f"\nStability dev→holdout: top-2 dev {st['top2_a']}, holdout {st['top2_b']}, composite ρ = {st['composite_rho']:.2f}, "
               f"median per-criterion ρ = {st['median_criterion_rho']:.2f}, per-criterion ρ = "
               + ", ".join(f"{k} {v:.2f}" for k, v in st["per_criterion_rho"].items()) + f" → **stable = {st['stable']}**\n")


def inv_detail(tag="main"):
    R = json.load(open(f"eval_{tag}_full.json"))
    out.append("\n### Per-augmentation retrieval@1 (holdout, 500 events; bootstrap 95% CI)\n")
    augs = ["shift+0.1", "shift-0.25", "shift+0.5", "noise", "scale0.8", "stretch1.15"]
    out.append("| rep | " + " | ".join(augs) + " | median per-feature ρ (noise) |")
    out.append("|---|" + "---|" * (len(augs) + 1))
    for r in R:
        if r["split"] != "holdout":
            continue
        out.append(f"| {r['rep']} | " + " | ".join(f3(r['inv'][a]['R1']) for a in augs) + f" | {r['inv']['noise']['rho_med']:.2f} |")


main_table("main", "Main comparison — D provisional events, all windows")
inv_detail()
main_table("nodrop", "Windows without any dropout only")
main_table("large", "Large events only (peak ≥ 5000 counts)")
for c in (5, 8, 10):
    main_table(f"lp{c}", f"Feature-path extra low-pass at {c} Hz")
main_table("C", "Events from workstream C's final detector.py (all SNR)")
main_table("C_snr20", "C events, SNR >= 20 tier")
main_table("C_snr30", "C events, SNR >= 30 (clean) tier")
if os.path.exists("eval_Ctier_vs_D_stability.json"):
    out.append("\n### D-events vs C SNR-tier events\n```\n" + open("eval_Ctier_vs_D_stability.json").read() + "\n```\n")
if os.path.exists("eval_C_vs_D_stability.json"):
    out.append("\n### D-events vs C-events ranking stability\n```\n" + open("eval_C_vs_D_stability.json").read() + "\n```\n")
if os.path.exists("confound_leakage.csv"):
    d = pd.read_csv("confound_leakage.csv")
    d["skill [CI]"] = d.apply(lambda r: f"{r.skill:.2f} [{r.skill_lo:.2f},{r.skill_hi:.2f}]", axis=1)
    out.append("\n### Confound test (legacy columns)\n")
    out.append((d[["windows", "split", "features", "target", "K", "n", "skill [CI]", "null_shuffled", "null_majority"]]).pipe(lambda z: md(z, False)))
if os.path.exists("norm_test.csv"):
    out.append("\n### Per-client vs global normalisation (dev)\n")
    out.append((pd.read_csv("norm_test.csv")).pipe(lambda z: md(z, False)))
# LP-cutoff sensitivity summary
rows = []
for tag in ["main", "lp10", "lp8", "lp5"]:
    if os.path.exists(f"eval_{tag}_summary.csv"):
        S = pd.read_csv(f"eval_{tag}_summary.csv"); S["cut"] = {"main": "20 Hz"}.get(tag, tag[2:] + " Hz"); rows.append(S)
if rows:
    S = pd.concat(rows)
    S = S[~S.rep.str.startswith("legacy")]
    out.append("\n### LP cut-off sensitivity (LP-path representations; holdout)\n")
    H = S[S.split == "holdout"]
    for m in ["invariance_R@1", "leak_date_skill", "leak_hour_skill", "sil_excess", "ari_excess", "proxy", "mean_rank"]:
        out.append(f"\n**{m}**\n\n" + md(H.pivot(index="rep", columns="cut", values=m).round(3)))
open("report_tables.md", "w").write("\n".join(out))
print("\n".join(out))
