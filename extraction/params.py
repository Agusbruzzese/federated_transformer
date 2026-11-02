"""All pipeline parameters in one place (recorded verbatim in manifest.json)."""

# --- block processing (stitched chunks) ---
CORE_CHUNKS = 60          # chunks whose events a block owns (~41 min)
CONTEXT_CHUNKS = 3        # extra chunks each side (3 x 41.4 s = 124 s >= 60 s required by C's detector)

# --- feature path (B's chain, preproc.py) ---
FEATURE_LP_HZ = 8.0       # 4th-order Butterworth, zero-phase, even padding, per segment
UP, DOWN = 25, 1208       # resample_poly ratio; effective output rate = fs_local * 25 / 1208
FILL_MAX = 4              # dropouts of 1-4 samples are linearly filled; >= 5 split the stream
MIN_SEG_S = 0.5           # segments shorter than this are masked
GAP_GUARD_S = 0.15        # B: flag +-0.15 s around each split (>= 5-sample dropout); only the hole itself is bridged
RESAMPLE_PADTYPE = "line" # E fix: B's default ('constant' = zero padding) pulls segment edges toward 0 (see REPORT)
BASE_WIN_S = 20.0         # two-pass running median
PASS1_K = 5.0             # pass-1 event threshold (sigma) for the baseline event mask (B: 5 sigma)
PASS1_BRIDGE_S = 2.0      # events bridged +-2 s in pass 2 (B)

# --- windows ---
WIN_N = 1050              # 21 s at ~50 Hz
CROP8_N = 402             # central ~8 s crop (D's validated morph window length) for the secondary morph set

# --- selection ---
MASK_FRAC_MAX = 0.2       # events with mask_frac > 0.2 are excluded (counted in the manifest)
TIERS = (5.0, 20.0, 30.0)


def as_dict():
    return {k: v for k, v in globals().items() if k.isupper()}
