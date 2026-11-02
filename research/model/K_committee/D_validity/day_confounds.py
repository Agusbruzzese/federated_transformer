"""Committee member D: per-day (candidate client) confounds and noise-pool provenance.

    .venv/bin/python research/model/K_committee/D_validity/day_confounds.py

Writes day_confounds.csv and day_confounds.json next to this file. Read-only on data/.
"""
import os, sys, json
import numpy as np, pandas as pd

ROOT = os.getcwd(); sys.path.insert(0, ROOT)
from ssl_central import data as D

HERE = os.path.dirname(os.path.abspath(__file__))
TEST = ["2025-02-07", "2025-02-13", "2025-02-19", "2025-02-25"]; VAL = "2025-02-10"

m, W, _, _ = D.load_all()
m["day"] = m.date.astype(str)
m["role"] = np.where(m.day.isin(TEST), "test", np.where(m.day == VAL, "val", "train"))
fam = pd.read_csv(os.path.join(ROOT, "research/model/C2_lookalikes/families.csv"))
s = m[m.snr >= 20].merge(fam, on="event_id", how="left")
pool = pd.read_parquet(os.path.join(ROOT, "data/ssl_central/bgpool_meta.parquet"))
pool["day"] = pd.to_datetime(pool.wall_time).dt.date.astype(str)

# hours of recording per day: union of event-time coverage is not enough; use the 30-min blocks that have
# at least one event of ANY snr as a lower bound on covered time
m["b30"] = pd.to_datetime(m.wall_time).dt.floor("30min")
glob_fam = s.family.value_counts(normalize=True).sort_index()
glob_hour = (s.hour // 6).value_counts(normalize=True).sort_index()
rows = []
for d, g in s.groupby("day"):
    a = m[m.day == d]
    fm = g.family.value_counts(normalize=True).reindex(glob_fam.index, fill_value=0)
    hm = (g.hour // 6).value_counts(normalize=True).reindex(glob_hour.index, fill_value=0)
    rows.append(dict(
        day=d, weekday=pd.Timestamp(d).day_name()[:3], run=a.run.iloc[0][4:12], role=g.role.iloc[0],
        n_all=len(a), n_snr20=len(g), blocks30_with_events=a.b30.nunique(),
        first_h=round(pd.to_datetime(a.wall_time).dt.hour.min() + 0.0, 0), last_h=pd.to_datetime(a.wall_time).dt.hour.max(),
        frac_dropout_all=round(a.has_dropout.mean(), 3), frac_dropout_snr20=round(g.has_dropout.mean(), 3),
        sigma_ev_median=round(g.sigma_ev.median(), 1), snr_median=round(g.snr.median(), 1),
        log_amp_median=round(g.m8_log_amp.median(), 3),
        polarity_pos=round((g.m8_polarity > 0).mean(), 3), lobes_mean=round(g.m8_n_lobes.clip(upper=4).mean(), 2),
        dur_median=round(g.m8_dur10.median(), 2),
        night_frac_0_6h=round((g.hour < 6).mean(), 3),
        fam_tvd_vs_global=round(0.5 * (fm - glob_fam).abs().sum(), 3),
        hour6_tvd_vs_global=round(0.5 * (hm - glob_hour).abs().sum(), 3),
        unclustered_frac=round((g.family < 0).mean(), 3),
        pool_segments=int((pool.day == d).sum()),
        pool_sigma_median=round(pool[pool.day == d].sigma.median(), 1) if (pool.day == d).any() else None,
        full_batches_256=len(g) // 256,
    ))
df = pd.DataFrame(rows)
df.to_csv(os.path.join(HERE, "day_confounds.csv"), index=False)
tr = df[df.role == "train"]
summ = dict(
    n_train_days=len(tr), train_days=tr.day.tolist(),
    snr20_per_train_day=dict(min=int(tr.n_snr20.min()), max=int(tr.n_snr20.max()), ratio=round(tr.n_snr20.max() / tr.n_snr20.min(), 1)),
    train_days_with_fewer_than_256_snr20=tr.day[tr.n_snr20 < 256].tolist(),
    train_days_without_pool_noise=tr.day[tr.pool_segments == 0].tolist(),
    pool_segments_by_day_all=pool.day.value_counts().sort_index().to_dict(),
    pool_segments_on_test_days=int(pool.day.isin(TEST).sum()), pool_segments_on_val=int((pool.day == VAL).sum()),
    pool_after_m4_filter=int((~pool.day.isin(TEST + [VAL])).sum()),
    pool_share_largest_day_after_filter=round(pool[~pool.day.isin(TEST + [VAL])].day.value_counts(normalize=True).max(), 3),
    families_csv_covers_test_events=int(fam.event_id.isin(s.event_id[s.role == "test"]).sum()),
)
json.dump(summ, open(os.path.join(HERE, "day_confounds.json"), "w"), indent=1)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
print(df.to_string(index=False)); print(json.dumps(summ, indent=1))
