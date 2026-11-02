"""Audit issue 3: BLIND human-labelling pack (not labelled by the agent).
80 holdout (Run 3) candidates, fresh random draw (seed 7): 40 clean-tier (HYST-BP, SNR>=30), 20 HYST-BP failing the SNR
cut, 20 BDET@8 (workstream B's detector at its native threshold). A BDET@8 segment overlapping (IoU>0.5) an already
drawn HYST-BP candidate is replaced by another draw, so no trace appears twice. Order shuffled.
Panels show ONLY an ID, the waveform (+-15 s context) and the shaded segment: no detector, tier, SNR, sigma, duration,
no y-axis values and no sigma reference lines.
-> figs/human_pack/sheet_XX.png, figs/human_pack/README.md, data/human_pack_key.csv, data/human_pack_labels.csv (empty)"""
import sys, os, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..')); sys.path.insert(0, os.path.dirname(__file__))
from common import *
import pandas as pd
from inject_eval import in_eval, run_setting
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
HERE = os.path.join(os.path.dirname(__file__), '..')
FS = D.FS
RULE = json.load(open(f'{HERE}/data/postfilter_rule.json'))
rows, cache = [], {}
for m in META:
    if m['split'] != 'holdout':
        continue
    sp = load_span(m['name']); P = prep(sp['y']); cache[m['name']] = (sp['t'], P)
    for a, b in in_eval(run_setting('HYSTBP', 5.0, sp['t'], P), m['e0'], m['e1']):
        snr = np.abs(P['z'][a:b]).max() / np.median(P['sigma'][a:b])
        rows.append(dict(span=m['name'], a=int(a), b=int(b), detector='HYSTBP', snr=snr,
                         tier='clean (HYST-BP SNR>=30)' if snr >= RULE['snr_min'] else 'HYST-BP failing SNR cut'))
    for a, b in in_eval(run_setting('BDET', 8.0, sp['t'], P), m['e0'], m['e1']):
        rows.append(dict(span=m['name'], a=int(a), b=int(b), detector='BDET@8',
                         snr=np.abs(P['z'][a:b]).max() / np.median(P['sigma'][a:b]), tier='BDET@8 (workstream B)'))
pool = pd.DataFrame(rows)
rng = np.random.default_rng(7)
pick = []
for tier, n in (('clean (HYST-BP SNR>=30)', 40), ('HYST-BP failing SNR cut', 20)):
    g = pool[pool.tier == tier]
    pick.append(g.iloc[rng.choice(len(g), n, replace=False)])
pick = pd.concat(pick)
g = pool[pool.tier == 'BDET@8 (workstream B)']
order = rng.permutation(len(g)); chosen = []
for i in order:
    r = g.iloc[i]
    same = pick[pick.span == r.span]
    iou = [max(0, min(r.b, q.b) - max(r.a, q.a)) / (max(r.b, q.b) - min(r.a, q.a)) for q in same.itertuples()]
    if not iou or max(iou) <= 0.5:
        chosen.append(i)
    if len(chosen) == 20:
        break
pick = pd.concat([pick, g.iloc[chosen]]).sample(frac=1, random_state=7).reset_index(drop=True)
pick['id'] = [f'H{i:03d}' for i in range(len(pick))]
pick['dur_s'] = (pick.b - pick.a) / FS
pick[['id', 'span', 'a', 'b', 'detector', 'tier', 'snr', 'dur_s']].to_csv(f'{HERE}/data/human_pack_key.csv', index=False)
pd.DataFrame({'id': pick.id, 'label': ''}).to_csv(f'{HERE}/data/human_pack_labels.csv', index=False)
out = f'{HERE}/figs/human_pack'; os.makedirs(out, exist_ok=True)
for sh in range(int(np.ceil(len(pick) / 20))):
    fig, ax = plt.subplots(5, 4, figsize=(20, 15)); ax = ax.ravel()
    for k, r in enumerate(pick.iloc[sh * 20:(sh + 1) * 20].itertuples()):
        t, P = cache[r.span]
        c0, c1 = max(r.a - int(15 * FS), 0), min(r.b + int(15 * FS), len(t))
        tt = t[c0:c1] - t[r.a]
        ax[k].plot(tt, P['z'][c0:c1], color='0.65', lw=.5); ax[k].plot(tt, P['zd'][c0:c1], color='k', lw=.7)
        ax[k].axvspan(0, t[r.b - 1] - t[r.a], color='tab:orange', alpha=.25)
        lo, hi = np.percentile(P['zd'][c0:c1], [0.5, 99.5]); pad = (hi - lo) * .15 + 1e-9
        ax[k].set_ylim(lo - pad, hi + pad); ax[k].set_yticklabels([])
        ax[k].set_title(r.id, fontsize=11); ax[k].tick_params(labelsize=7); ax[k].set_xlabel('s', fontsize=7)
    for a in ax[k + 1:]:
        a.axis('off')
    plt.tight_layout(); plt.savefig(f'{out}/sheet_{sh:02d}.png', dpi=60); plt.close()
open(f'{out}/README.md', 'w').write('''# Blind labelling pack (workstream C): please label by hand

Label codes (write one per ID in `data/human_pack_labels.csv`, column `label`):
- **V**: one vehicle-like transient (smooth, typically bi- or multi-polar, >= ~0.5 s core), sensibly covered by the
  shaded segment.
- **M**: merged or over-long: the shaded segment contains >= 2 distinct vehicle transients, or is > ~3x the transient.
- **F**: fragment: a vehicle transient is present but the segment cuts its core (< ~half of it inside).
- **N**: no vehicle-like transient in the segment (noise, slow wander, step, narrow spikes or pulse trains).
- **U**: uncertain.

How to read a panel: black = detection channel (0.5 s median-filtered), grey = waveform; orange = the candidate segment;
x axis in seconds relative to the segment start, +-15 s context; the y axis is auto-scaled per panel and has no values
on purpose (amplitude, SNR and the source detector are hidden). Sheets: `sheet_00.png` ... `sheet_03.png`, IDs H000-H079.

Do not open `data/human_pack_key.csv` (it reveals detector, tier and SNR) until labelling is finished.
Scoring after labelling: merge the labels with the key on `id` and compute precision per tier (Wilson CI), e.g. with
`scripts/handcheck.py`'s `wilson()`.
''')
print(pick.tier.value_counts())
