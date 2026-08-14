"""Draw cell boundaries conditioned on confirmed centroids.

Cellpose forms masks in two steps: a network predicts a flow field ``dP`` plus a
per-pixel ``cellprob``, then pixel dynamics follow that field and the *converged*
pixels are clustered by histogram peaks. Step one is learned; step two is a
heuristic that has no idea which cells are real. It invents basins in empty
prism background, drops somata the histogram never peaks on, and merges cells
whose flows share an attractor.

This module keeps step one and replaces step two with the centroids a human has
already confirmed on the ``/cells`` page:

    1. extent   — a pixel is cell material if its flow trajectory converges
                  within ``capture_px`` of *some* seed. Basins that attract no
                  seed are not cells, whatever the histogram thought.
    2. identity — within that extent, watershed on ``-cellprob`` with the seeds
                  as markers decides which cell each pixel belongs to.
    3. cleanup  — per label, keep the connected component holding its own seed,
                  fill holes, and drop anything outside the area bounds.
    4. fallback — a seed that still owns no pixels gets the canonical disk, so a
                  confirmed cell can never vanish from the output.

Why step 2 is not just "nearest seed in converged space": measured on two
synthetic somata 36 px apart, Cellpose emitted a single 3198-px label whose flow
field has one attractor sitting ~18 px from *both* true centroids. Every pixel
converges to the same point, so a nearest-seed rule assigns nothing at any sane
``capture_px``. The network merged the cells in flow space and no partition of
that space can undo it. The watershed does: 1334 + 1864 px, centroids recovered
within ~2 px of truth. Extent and identity are genuinely different questions and
the flow field only answers the first.

Labels are the caller's own — see :func:`seeded_labels`. Nothing here is
positional.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from roigbiv.pipeline.roi_stamp import disk_mask

# Where a label's pixels came from, recorded per label in boundaries.json.
ORIGIN_FLOW = "flow"                    # basin + watershed, the intended path
ORIGIN_DISK_FALLBACK = "disk_fallback"  # seed captured nothing; canonical disk
ORIGIN_MANUAL = "manual"                # hand-drawn; see pipeline/boundary_edits.py


@dataclass
class SeededMasks:
    """Outcome of one :func:`seeded_labels` call.

    Mirrors :class:`~roigbiv.pipeline.centroid_masks.StampedMasks` so the two
    geometry tracks report themselves the same way to ``run_tracking``.
    """

    labels: np.ndarray                  # uint16, 0 = background
    n_seeds: int
    origins: dict[int, str]             # label -> ORIGIN_*
    areas: dict[int, int]               # label -> pixel count
    n_orphan_basin_px: int              # cell-prob pixels attracted to no seed
    capture_px: float
    fallback_radius: int
    min_area: int = 0
    warnings: list[str] = field(default_factory=list)
    # False when the caller computed but did not write (see write_boundaries).
    written: bool = True

    @property
    def n_disk_fallback(self) -> int:
        return sum(1 for o in self.origins.values() if o == ORIGIN_DISK_FALLBACK)

    @property
    def present_labels(self) -> tuple[int, ...]:
        """Non-zero labels actually present, ascending.

        Cast to plain ``int``: ``np.unique`` on a uint16 array yields
        ``numpy.uint16`` scalars, which SQLAlchemy stores as raw bytes rather
        than integers if one reaches an ``ObservationRecord``.
        """
        return tuple(int(v) for v in sorted(np.unique(self.labels)[1:]))


def converge_pixels(
    dP: np.ndarray,
    cellprob: np.ndarray,
    *,
    cellprob_threshold: float,
    niter: int,
    dp_scale: float = 5.0,
    device: Optional[object] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Follow Cellpose's flow field to where each cell pixel converges.

    Returns ``(inds, converged)``: ``inds`` is the ``(2, N)`` original pixel
    coordinates, ``converged`` the ``(2, N)`` positions after ``niter`` steps.

    The scaling and masking here are not ours to choose — they replicate
    ``cellpose.dynamics.compute_masks`` exactly::

        follow_flows(dP * (cellprob > cellprob_threshold) / 5., inds=inds, ...)

    Diverging from it would converge pixels differently than the inference that
    produced the field, and every threshold downstream is calibrated against
    Cellpose's own behavior.
    """
    import torch
    from cellpose import dynamics

    inds_t = np.nonzero(cellprob > cellprob_threshold)
    if len(inds_t[0]) == 0:
        empty = np.zeros((2, 0), dtype=np.float32)
        return empty, empty

    if device is None:
        device = (torch.device("cuda") if torch.cuda.is_available()
                  else torch.device("cpu"))

    flow = dP * (cellprob > cellprob_threshold) / dp_scale
    converged = dynamics.follow_flows(
        flow.astype(np.float32), inds_t, niter=int(niter), interp=True,
        device=device,
    )
    # follow_flows returns a CUDA tensor whenever interp=True.
    if hasattr(converged, "cpu"):
        converged = converged.cpu().numpy()

    inds = np.asarray(np.vstack(inds_t), dtype=np.int32)
    return inds, np.asarray(converged, dtype=np.float32)


def seeded_labels(
    seeds: dict[int, tuple[float, float]],
    shape: tuple[int, int],
    *,
    inds: np.ndarray,
    converged: np.ndarray,
    cellprob: np.ndarray,
    capture_px: float,
    fallback_radius: int,
    min_area: int = 0,
    max_area: Optional[int] = None,
) -> SeededMasks:
    """Boundary per seed, using the flow field for extent and seeds for identity.

    *seeds* maps an **explicit label id** to a ``(y, x)`` centroid — the same
    convention :func:`~roigbiv.pipeline.centroid_masks.stamp_labeled_centroids`
    uses. Label ids are never renumbered: ``CellObservation.local_label_id``
    references them, so a positional relabel would silently repoint the registry
    at the wrong cell.

    Every seed gets exactly one label in the output. That is the contract the
    ``/cells`` page depends on — a confirmed cell that produced no boundary
    would otherwise disappear from the timeline rather than look wrong.

    *inds* / *converged* come from :func:`converge_pixels`; they are passed in
    rather than computed here so tests can exercise the partition logic without
    a GPU or a Cellpose install.
    """
    from scipy import ndimage as ndi
    from skimage.segmentation import watershed

    height, width = int(shape[0]), int(shape[1])
    labels = np.zeros((height, width), dtype=np.uint16)
    origins: dict[int, str] = {}
    areas: dict[int, int] = {}
    warnings: list[str] = []

    if not seeds:
        return SeededMasks(
            labels=labels, n_seeds=0, origins={}, areas={},
            n_orphan_basin_px=0, capture_px=float(capture_px),
            fallback_radius=int(fallback_radius), min_area=int(min_area),
            warnings=["no seeds — nothing to draw"],
        )

    ordered = sorted(seeds)
    seed_yx = np.asarray([seeds[k] for k in ordered], dtype=np.float64)

    # --- 1. extent: which pixels are cell material at all -------------------
    basin = np.zeros((height, width), dtype=bool)
    n_orphan = 0
    if converged.size:
        d = np.hypot(
            converged[0][:, None] - seed_yx[None, :, 0],
            converged[1][:, None] - seed_yx[None, :, 1],
        )
        captured = d.min(axis=1) <= capture_px
        n_orphan = int((~captured).sum())
        basin[inds[0][captured], inds[1][captured]] = True

    # --- 2. identity: which cell each of those pixels belongs to ------------
    markers = np.zeros((height, width), dtype=np.int32)
    for i, key in enumerate(ordered, start=1):
        cy, cx = seeds[key]
        y = min(max(int(round(cy)), 0), height - 1)
        x = min(max(int(round(cx)), 0), width - 1)
        markers[y, x] = i

    if basin.any():
        # Watershed on -cellprob: the descent runs downhill from each seed, so
        # a boundary lands where probability is lowest between two somata.
        ws = watershed(-np.asarray(cellprob, dtype=np.float32),
                       markers=markers, mask=basin)
    else:
        ws = np.zeros((height, width), dtype=np.int32)

    # --- 3. cleanup: decide each seed's region, painting nothing yet --------
    #      Painting happens in the two passes below so a disk fallback can
    #      never steal pixels a flow-derived neighbour already legitimately
    #      owns — see the module docstring's "why step 2 is not just nearest
    #      seed" for why flow-derived identity is the one that has real data
    #      behind it and a disk is only ever a placeholder guess.
    needs_fallback: list[int] = []
    for i, key in enumerate(ordered, start=1):
        region = ws == i
        if region.any():
            region = _seed_component(region, seeds[key], height, width)
            region = ndi.binary_fill_holes(region)

        npix = int(region.sum())
        too_small = npix < min_area
        too_big = max_area is not None and npix > max_area
        if npix and (too_small or too_big):
            warnings.append(
                f"label {key}: {npix}px outside area bounds "
                f"[{min_area}, {max_area}] — using disk fallback"
            )
            region = np.zeros((height, width), dtype=bool)
            npix = 0

        if npix == 0:
            needs_fallback.append(key)
        else:
            origins[key] = ORIGIN_FLOW
            labels[region] = key

    # --- 4. disk fallback, in two steps so no seed can ever end up with zero
    #      pixels just because a neighbour's disk happened to paint later.
    #
    #      4a. reserve each fallback seed's own centre pixel first — this is
    #      the floor no other seed's disk may ever take, painted below the
    #      flow snapshot so a real flow boundary still wins the pixel if one
    #      already claimed it (a disk is only ever a guess; a flow region is
    #      measured).
    flow_owned = labels.copy()
    for key in needs_fallback:
        y = min(max(int(round(seeds[key][0])), 0), height - 1)
        x = min(max(int(round(seeds[key][1])), 0), width - 1)
        if labels[y, x] == 0:
            labels[y, x] = key
    center_owner = labels.copy()

    #      4b. paint the full disks in ascending label order — later disks
    #      may still overwrite an *earlier* disk's ordinary pixels, matching
    #      the "later label wins" convention centroid_masks.py already
    #      documents for canonical stamps — but never a flow pixel, and never
    #      another seed's reserved centre, regardless of paint order.
    for key in needs_fallback:
        disk = disk_mask(*seeds[key], fallback_radius, height, width)
        blocked = (flow_owned != 0) | ((center_owner != 0) & (center_owner != key))
        labels[disk & ~blocked] = key
        origins[key] = ORIGIN_DISK_FALLBACK

    # A neighbour can still claim part of a fallback disk's non-floor area, so
    # re-read areas from the image rather than trusting the per-region counts.
    # Only reachable now when a seed's own centre pixel is itself inside
    # another seed's flow boundary — the one case the floor above cannot
    # cover, since a measured boundary is deliberately never displaced by a
    # guess.
    for key in ordered:
        areas[key] = int(np.count_nonzero(labels == key))
        if areas[key] == 0:
            warnings.append(
                f"label {key}: fully overwritten by a neighbouring label — "
                f"it will not reach the registry")

    return SeededMasks(
        labels=labels,
        n_seeds=len(seeds),
        origins=origins,
        areas=areas,
        n_orphan_basin_px=n_orphan,
        capture_px=float(capture_px),
        fallback_radius=int(fallback_radius),
        min_area=int(min_area),
        warnings=warnings,
    )


def _seed_component(region: np.ndarray, seed, height: int, width: int) -> np.ndarray:
    """The connected component of *region* containing *seed*.

    Watershed can hand back a region split across the mask; only the piece the
    human actually pointed at is this cell. Falls back to the largest component
    when the seed itself sits just outside the basin (a centroid moved by an
    edit onto a dim pixel), which is still a better answer than dropping it.
    """
    from scipy import ndimage as ndi

    cc, n = ndi.label(region)
    if n <= 1:
        return region

    cy, cx = seed
    y = min(max(int(round(cy)), 0), height - 1)
    x = min(max(int(round(cx)), 0), width - 1)
    at_seed = int(cc[y, x])
    if at_seed:
        return cc == at_seed

    sizes = ndi.sum_labels(region, cc, index=np.arange(1, n + 1))
    return cc == (int(np.argmax(sizes)) + 1)
