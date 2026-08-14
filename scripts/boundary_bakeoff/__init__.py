"""Measure seeded cell boundaries against hand-drawn ImageJ ground truth.

Three arms, all scored the same way:

``free_cellpose``  Cellpose's own instance segmentation — what Stage 1 detects
                   today, with no knowledge of where the cells are.
``disk_stamps``    fixed-radius disks at each centroid — what the tracking
                   workflow writes to ``merged_masks.tif`` (ADR-0003).
``seeded``         :mod:`roigbiv.pipeline.seeded_masks` — Cellpose's flow field
                   for extent, confirmed centroids for identity.

The point of the ``disk_stamps`` arm is that it is the *current* answer for the
/cells page, so "did this help" is measured against what it replaces rather than
against free Cellpose alone.
"""
