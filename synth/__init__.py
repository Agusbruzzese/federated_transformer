"""Family-conditioned synthetic event generator (report 06).

Imitates SNR >= 20 events as the encoder sees them (12 s, 50 Hz, 600 samples, baseline-subtracted counts,
before max-normalisation). One sub-model per (C2 shape family x polarity) "stratum":

  shape  = registered template + PCA modes            (learned from low-passed, peak-normalised real crops)
  params = (PCA coefficients, stretch, lag, log amp, log noise sigma), sampled jointly by smoothed bootstrap
  event  = amp * warp(shape, stretch, lag) + sigma * real background noise (ssl_central bgpool)

    python -m synth fit                      # fit on training days, write data/synth/model.npz
    python -m synth sample --n 5000          # write data/synth/sample_<n>.npz
    python -m synth.validate                 # congruence tests vs held-out real days
    python -m synth.compare                  # whole dataset vs a simulated 14-day run (report 07)
"""
