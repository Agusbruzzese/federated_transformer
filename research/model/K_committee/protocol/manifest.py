"""Generate manifest_v2_1.json (report 14, A1): every run of the study, with its full fedsim3 CLI. Deterministic.
Learning rates of tuned arms are placeholders "TUNED:<arm>" resolved by launch.py from tuned_lr.json (step 7 output);
server initialisation is "SERVER:<seed>", resolved from server_hashes.json (pinned after the server group).
    python manifest.py        -> manifest_v2_1.json (refuses to overwrite)
"""
import json, os
HERE = os.path.dirname(os.path.abspath(__file__))
X, P = 10_000 * 256, 1_656 * 256
LR_GRID = [1.5e-4, 3e-4, 6e-4, 1e-3]; SERVER_LR = 3e-4
ARMS = {  # protocol name: (fedsim3 arm, kind, batch, rounds, examples, lr source, server init?)
    "C256": ("central", "natural", 256, 200, X, "C256", False), "C64": ("central", "natural", 64, 200, X, "C64", False),
    "F-nat": ("fed", "natural", 64, 200, X, "F-nat", False), "F-IID": ("fed", "iid", 64, 200, X, "F-nat", False),
    "F-X+P": ("fed", "natural", 64, 233, X + P, "F-X+P", False), "S+F": ("fed", "natural", 64, 200, X, "S+F", True),
    "S+C": ("central", "natural", 256, 200, X, "S+C", True), "CJ": ("cj", "natural", 256, 188, X + P, "CJ", False),
    "C256rep": ("central", "natural", 256, 200, X, "C256", False),
    "F-nat-reset": ("fed", "natural", 64, 200, X, "F-nat", False), "F-nat-R50": ("fed", "natural", 64, 50, X, "F-nat", False),
}
TUNED = ["C256", "C64", "F-nat", "S+F", "S+C", "F-X+P", "CJ"]


def entry(group, name, fold, seed, lr, tag):
    fa, kind, b, R, ex, _, srv = ARMS[name] if name != "server" else ("server", "natural", 256, 184, P, None, False)
    cli = ["--arm", fa, "--fold", str(fold), "--kind", kind, "--seed", str(seed), "--rounds", str(R), "--batch", str(b),
           "--examples", str(ex), "--lr", str(lr), "--tag", tag]
    if name == "F-nat-reset": cli += ["--opt", "reset"]
    return dict(id=tag, group=group, arm=name, fold=int(fold), seed=int(seed), cli=cli, init=f"SERVER:{seed}" if srv else None,
                expect=dict(K=10 if fa == "fed" else 1, examples=ex, rounds=R, batch=b, tolerance=0.01))


def build():
    E = []
    for s in range(10): E.append(entry("server", "server", 0, s, SERVER_LR, f"v21_server_s{s}"))
    for name in TUNED:
        for lr in LR_GRID:
            e = entry("tuning", name, 0, 0, lr, f"v21_tune_{name}_lr{lr:g}")
            if ARMS[name][6]: e["init"] = "SERVER:0"
            E.append(e)
    for f in range(3):
        for s in range(20):
            E.append(entry("dryrun", "C256" if s < 10 else "C256rep", f, s, "TUNED:C256", f"v21_{'C256' if s < 10 else 'C256rep'}_f{f}_s{s}"))
        for name in ("C64", "F-nat", "F-IID", "F-X+P", "S+F", "S+C", "CJ"):
            for s in range(10): E.append(entry("main", name, f, s, f"TUNED:{ARMS[name][5]}", f"v21_{name}_f{f}_s{s}"))
        for name in ("F-nat-reset", "F-nat-R50"):
            for s in range(5): E.append(entry("ablation", name, f, s, f"TUNED:{ARMS[name][5]}", f"v21_{name}_f{f}_s{s}"))
    assert len({e["id"] for e in E}) == len(E), "duplicate tags"
    return E


if __name__ == "__main__":
    E = build(); p = os.path.join(HERE, "manifest_v2_1.json"); json.dump(E, open(p, "x"), indent=1)
    from collections import Counter; print(p, len(E), dict(Counter(e["group"] for e in E)))
