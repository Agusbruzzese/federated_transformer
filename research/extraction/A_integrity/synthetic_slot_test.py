"""Synthetic validation of bsd_io.slots_from_timestamps (v2) vs the superseded v1.

Generator (parameters from measure_postgap.py / real data):
  * uniform ADC grid, period P ticks in {414, 419, 435} with a linear in-chunk drift of
    +-0.3 ticks (real: <=0.17), white stamp jitter 2 ticks rms, stamps rounded to integers;
  * dropouts of size 1, 2, 5, 105 samples, plus 'late-only' events (size 0);
  * the first stamp after a dropout is late by U(0, 0.9) slot (real: 0..0.87, mean 0.24);
  * bursts: 2-4 dropouts separated by s in {1,2,3,4,8,12,16,24,28,40} delivered samples.
Metric: per-dropout size error (est - true), per-chunk total-missing error, and fraction of
samples whose reconstructed slot is wrong."""
import numpy as np, pandas as pd, sys
import bsd_io as b
rng = np.random.default_rng(2026)
N = 100_000

def make(P, events, drift=0.3, jit=2.0):
    """events: list of (position_in_delivered_samples, size, lateness_slots)."""
    ev = sorted(events)
    keep_slots = []; s = 0; late = np.zeros(N)
    pos_prev = 0
    for pos, size, lt in ev:
        keep_slots.append(np.arange(s, s + (pos - pos_prev))); s += pos - pos_prev
        s += size
        late[pos] = lt
        pos_prev = pos
    keep_slots.append(np.arange(s, s + (N - pos_prev)))
    slots = np.concatenate(keep_slots)[:N]
    Pk = P + drift * (slots / slots[-1] - 0.5) * 2
    t = np.cumsum(np.r_[0, Pk[:-1]])  # placeholder not used
    tt = slots * P + drift * slots ** 2 / slots[-1] + (late * P) + rng.normal(0, jit, N) + 5e6
    return np.rint(tt).astype(np.int64), slots

def evaluate(t, true, fn):
    est, _ = fn(t)
    est = est - est[0]; tr = true - true[0]
    di = np.diff(tr); de = np.diff(est)
    ev = np.flatnonzero((di > 1) | (de > 1))
    return dict(total_err=int((est[-1] - tr[-1])), frac_wrong=float(np.mean(est != tr)),
                gap_err=(de[ev] - di[ev]).tolist(), gap_true=(di[ev] - 1).tolist())

def scenario(kind, P, n_chunks):
    out = []
    for c in range(n_chunks):
        events = []
        if kind.startswith("isolated"):
            size = int(kind.split("_")[1])
            for pos in rng.choice(np.arange(200, N - 200, 400), 20, replace=False):
                lt = rng.uniform(0.5, 0.95) if size == 0 else rng.uniform(0, 0.9)
                events.append((int(pos), size, lt))
        elif kind.startswith("burst"):
            sp = int(kind.split("_")[1])
            for pos in rng.choice(np.arange(500, N - 500, 1000), 15, replace=False):
                k = rng.integers(2, 5)
                for i in range(k):
                    events.append((int(pos + i * sp), 105 if rng.random() < .8 else int(rng.choice([1, 2, 5])), rng.uniform(0, 0.9)))
        elif kind == "realistic_0204":
            pos = np.sort(rng.choice(np.arange(300, N - 300), 16, replace=False))
            for p in pos:
                events.append((int(p), 105 if rng.random() < .75 else int(rng.choice([104, 100, 1, 2, 3, 0])), rng.uniform(0, 0.9)))
                if rng.random() < 0.23:
                    events.append((int(p + rng.choice([12, 16, 24, 28, 36, 40])), 105, rng.uniform(0, 0.9)))
        # dedupe positions
        seen = {}; 
        for e in events: seen.setdefault(e[0], e)
        t, tr = make(P, list(seen.values()))
        for name, fn in [("v2", b.slots_from_timestamps), ("v1", b._slots_v1)]:
            r = evaluate(t, tr, fn); r.update(kind=kind, P=P, est=name); out.append(r)
    return out

rows = []
kinds = ["isolated_0", "isolated_1", "isolated_2", "isolated_5", "isolated_105"] + \
        [f"burst_{s}" for s in [1, 2, 3, 4, 8, 12, 16, 24, 28, 40]] + ["realistic_0204"]
for P in [414, 419, 435]:
    for k in kinds:
        rows += scenario(k, P, 20)
df = pd.DataFrame(rows)
df["n_events"] = df.gap_err.map(len)
df["events_wrong"] = df.gap_err.map(lambda x: int(np.sum(np.array(x) != 0)))
summ = df.groupby(["kind", "est"]).agg(chunks=("total_err", "size"), events=("n_events", "sum"), events_wrong=("events_wrong", "sum"),
                                       total_err_mean=("total_err", "mean"), total_err_absmax=("total_err", lambda x: np.abs(x).max()),
                                       frac_samples_wrong=("frac_wrong", "mean")).reset_index()
pd.set_option("display.width", 200)
print(summ.to_string(index=False))
allerr = np.concatenate([np.array(x) for x in df[df.est == "v2"].gap_err])
print("\nv2 per-event size-error distribution:", dict(zip(*np.unique(allerr, return_counts=True))))
allerr1 = np.concatenate([np.array(x) for x in df[df.est == "v1"].gap_err])
print("v1 per-event size-error distribution:", dict(zip(*np.unique(allerr1, return_counts=True))))
summ.to_csv("synthetic_slot_test.csv", index=False)
