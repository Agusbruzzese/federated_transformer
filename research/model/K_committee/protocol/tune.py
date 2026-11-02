"""Step 7 scorer and selection rule (report 09 section 4; report 14 A1). Scores the 28 tuning runs on Feb 10 (probes fitted
on fold-0 client events), applies the rule, writes tuned_lr.json (refuses to overwrite).
Rule: keep 3e-4 unless another value exceeds the 3e-4 score by more than DELTA_SEL; if several do, take the highest-scoring."""
import json, os, sys, time
import evaluator as E, evaluator2 as E2
HERE = os.path.dirname(os.path.abspath(__file__)); DELTA_SEL, DEFAULT = 0.02, 3e-4
INHERIT = {"F-IID": "F-nat", "C256rep": "C256", "F-nat-reset": "F-nat", "F-nat-R50": "F-nat"}


def select(scores):
    """scores: {lr: score}. Returns the chosen lr."""
    assert DEFAULT in scores; base = scores[DEFAULT]
    better = {lr: s for lr, s in scores.items() if lr != DEFAULT and s > base + DELTA_SEL}
    return max(better, key=better.get) if better else DEFAULT


def main():
    man = json.load(open(os.path.join(HERE, "manifest_v2_1.json"))); out = dict(rule=f"keep {DEFAULT} unless > +{DELTA_SEL}", arms={})
    for e in [x for x in man if x["group"] == "tuning"]:
        lr = float(e["cli"][e["cli"].index("--lr") + 1]); r = E2.score_tuning(E.load_encoder(os.path.join(E2.RUNS, e["id"], "final.pt")))
        out["arms"].setdefault(e["arm"], {})[str(lr)] = r
    out["chosen"] = {a: select({float(k): v["score"] for k, v in d.items()}) for a, d in out["arms"].items()}
    for a, src in INHERIT.items(): out["chosen"][a] = out["chosen"][src]
    p = os.path.join(HERE, "tuned_lr.json"); json.dump(out, open(p, "x"), indent=1); print(json.dumps(out["chosen"], indent=1), "->", p)


if __name__ == "__main__":
    main()
