"""Vehicle-event extraction pipeline (workstream E).

Implements research/extraction/SYNTHESIS.md §3:
  stitched chunks (A: bsd_io) -> detection (C: detector.detect_stream) and feature path (B: preproc chain)
  -> per event: 21 s window (1050 samples) centred on C's energy centroid, [onset, offset] mask,
     19 morphology scalars on the true time base (D: features.morph_feats, fs-corrected), metadata.

Entry point:  python -m extraction.run --out data/events
Self-tests:   python -m extraction.tests
The validated modules of workstreams A-D are vendored unchanged (except marked E-PATCH lines) in _vendor/.
"""
