# Blind labelling pack (workstream C): please label by hand

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
