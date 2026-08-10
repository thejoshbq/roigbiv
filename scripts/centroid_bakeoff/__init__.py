"""Centroid-detection bake-off — OpenCV vs. Cellpose vs. Suite2p.

Point-first counterpart to ``scripts/cv_bakeoff/`` (which compares full
segmentation boundaries). This package benchmarks bare centroid localization
against ground truth, for the cross-session cell-registration work: pyramidal
neurons' apical dendrites make boundary fitting unreliable, so centroid
localization is evaluated as its own contract rather than derived from masks.

See scripts/centroid_bakeoff/run_centroid_bakeoff.py for usage.
"""
