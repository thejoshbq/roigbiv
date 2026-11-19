"""Gate 3 rejects Stage 3 candidates that look like noise rather than calcium activity.

Two checks sit beside the waveform checks:

- Sign: calcium transients only go up, so large trace excursions of a real cell are positive. Noise bursts swing
  both ways, so a candidate whose large excursions are often negative is rejected.
- Activity contrast: an active cell is brighter in the activity image (``std_S``) than the ring around it. A
  candidate no brighter than its surroundings is rejected.

A candidate here passes the waveform checks by construction (its event waveforms are the pipeline's own template),
so any rejection comes from the two new checks.
"""

from __future__ import annotations

import numpy as np

from roigbiv.pipeline.gate3 import compute_activity_contrast, compute_negative_excursion_share, evaluate_gate3
from roigbiv.pipeline.stage3_templates import build_template_bank
from roigbiv.pipeline.types import ROI, PipelineConfig

FRAME_RATE_HZ = 7.5
DECAY_TAU_S = 1.0
N_FRAMES = 2000
IMAGE_SIZE = 64
CELL_CENTER = (32, 32)
CELL_RADIUS_PX = 5
EVENT_FRAMES = (200, 700, 1200, 1700)
EVENT_AMPLITUDE = 20.0
NOISE_SD = 1.0


def build_config() -> PipelineConfig:
    """A config at the lab's frame rate with the default Gate 3 thresholds."""
    return PipelineConfig(fs=FRAME_RATE_HZ, tau=DECAY_TAU_S)


def build_template() -> np.ndarray:
    """The pipeline's own single-event template, scaled to a peak of 1."""
    template = build_template_bank(FRAME_RATE_HZ, DECAY_TAU_S)[0][1]
    return template / template.max()


def build_disk_mask() -> np.ndarray:
    """A disk the size of a small soma at the centre of the image."""
    rows, columns = np.mgrid[0:IMAGE_SIZE, 0:IMAGE_SIZE]
    distance = np.hypot(rows - CELL_CENTER[0], columns - CELL_CENTER[1])
    return distance <= CELL_RADIUS_PX


def build_transient_trace(rng: np.random.Generator) -> np.ndarray:
    """Gaussian noise with upward calcium transients at EVENT_FRAMES."""
    trace = rng.normal(0.0, NOISE_SD, N_FRAMES).astype(np.float32)
    template = build_template()
    for frame in EVENT_FRAMES:
        trace[frame : frame + len(template)] += EVENT_AMPLITUDE * template
    return trace


def build_symmetric_burst_trace(rng: np.random.Generator) -> np.ndarray:
    """Gaussian noise with bursts that jump up and down by the same size."""
    trace = rng.normal(0.0, NOISE_SD, N_FRAMES).astype(np.float32)
    for frame in EVENT_FRAMES:
        burst = rng.choice([-1.0, 1.0], size=20) * EVENT_AMPLITUDE
        trace[frame : frame + len(burst)] += burst
    return trace


def build_candidate(trace: np.ndarray) -> ROI:
    """A Stage 3 candidate whose event waveforms are the clean template, so the waveform checks pass."""
    template = build_template()
    waveform = np.concatenate([np.zeros(5, np.float32), template.astype(np.float32)])
    return ROI(
        mask=build_disk_mask(),
        label_id=1,
        source_stage=3,
        confidence="low",
        gate_outcome="reject",
        event_count=len(EVENT_FRAMES),
        trace=trace,
        event_waveforms=[waveform.copy() for _ in EVENT_FRAMES],
        features={"events": [{"frame": frame} for frame in EVENT_FRAMES]},
    )


def build_activity_image(cell_brightness: float, rng: np.random.Generator) -> np.ndarray:
    """A noisy std_S image with the candidate's disk raised by cell_brightness."""
    image = rng.normal(1.0, 0.1, (IMAGE_SIZE, IMAGE_SIZE)).astype(np.float32)
    image[build_disk_mask()] += cell_brightness
    return image


def test_negative_share_is_zero_for_upward_transients():
    trace = build_transient_trace(np.random.default_rng(0))
    assert compute_negative_excursion_share(trace, excursion_z=5.0) == 0.0


def test_negative_share_is_near_half_for_symmetric_bursts():
    trace = build_symmetric_burst_trace(np.random.default_rng(1))
    assert 0.3 <= compute_negative_excursion_share(trace, excursion_z=5.0) <= 0.7


def test_negative_share_is_zero_without_large_excursions():
    trace = np.random.default_rng(2).normal(0.0, NOISE_SD, N_FRAMES)
    assert compute_negative_excursion_share(trace, excursion_z=10.0) == 0.0


def test_activity_contrast_is_high_for_a_bright_disk():
    rng = np.random.default_rng(3)
    image = build_activity_image(cell_brightness=1.0, rng=rng)
    assert compute_activity_contrast(image, build_disk_mask(), build_config()) > 2.0


def test_activity_contrast_is_near_zero_for_a_flat_image():
    rng = np.random.default_rng(4)
    image = build_activity_image(cell_brightness=0.0, rng=rng)
    assert abs(compute_activity_contrast(image, build_disk_mask(), build_config())) < 0.5


def test_gate3_keeps_a_bright_candidate_with_upward_transients():
    rng = np.random.default_rng(5)
    candidate = build_candidate(build_transient_trace(rng))
    image = build_activity_image(cell_brightness=1.0, rng=rng)
    evaluate_gate3([candidate], [], build_template_bank(FRAME_RATE_HZ, DECAY_TAU_S), build_config(),
                   activity_image=image)
    assert candidate.gate_outcome != "reject", candidate.gate_reasons


def test_gate3_keeps_upward_transients_on_a_dropping_baseline():
    """The baseline drops by 15 noise SDs two thirds of the way in (bleaching or a slow focus shift), so the last
    third sits far below the whole-trace median. The sign check must read excursions from the sliding baseline,
    so the drop does not count as negative excursions."""
    rng = np.random.default_rng(10)
    frames = np.arange(N_FRAMES)
    drop = (-15.0 * NOISE_SD / (1.0 + np.exp(-(frames - 1350) / 20.0))).astype(np.float32)
    candidate = build_candidate(build_transient_trace(rng) + drop)
    image = build_activity_image(cell_brightness=1.0, rng=rng)
    evaluate_gate3([candidate], [], build_template_bank(FRAME_RATE_HZ, DECAY_TAU_S), build_config(),
                   activity_image=image)
    assert candidate.gate_outcome != "reject", candidate.gate_reasons


def test_gate3_rejects_symmetric_bursts():
    rng = np.random.default_rng(6)
    candidate = build_candidate(build_symmetric_burst_trace(rng))
    image = build_activity_image(cell_brightness=1.0, rng=rng)
    evaluate_gate3([candidate], [], build_template_bank(FRAME_RATE_HZ, DECAY_TAU_S), build_config(),
                   activity_image=image)
    assert candidate.gate_outcome == "reject"
    assert any(reason.startswith("negative_excursions") for reason in candidate.gate_reasons)


def test_gate3_rejects_a_candidate_no_brighter_than_its_surround():
    rng = np.random.default_rng(7)
    candidate = build_candidate(build_transient_trace(rng))
    image = build_activity_image(cell_brightness=0.0, rng=rng)
    evaluate_gate3([candidate], [], build_template_bank(FRAME_RATE_HZ, DECAY_TAU_S), build_config(),
                   activity_image=image)
    assert candidate.gate_outcome == "reject"
    assert any(reason.startswith("activity_contrast") for reason in candidate.gate_reasons)


def test_gate3_does_not_count_a_bright_neighbouring_cell_as_surround():
    """A Stage 1 cell sits right beside the candidate and is far brighter in std_S. Its pixels are left out of the
    annulus, so the candidate is compared with the tissue around it, not with its neighbour."""
    rng = np.random.default_rng(11)
    candidate = build_candidate(build_transient_trace(rng))
    rows, columns = np.mgrid[0:IMAGE_SIZE, 0:IMAGE_SIZE]
    neighbour_mask = np.hypot(rows - CELL_CENTER[0], columns - (CELL_CENTER[1] + 12)) <= CELL_RADIUS_PX
    neighbour = ROI(mask=neighbour_mask, label_id=2, source_stage=1, confidence="high", gate_outcome="accept")
    image = build_activity_image(cell_brightness=1.0, rng=rng)
    image[neighbour_mask] += 20.0
    evaluate_gate3([candidate], [neighbour], build_template_bank(FRAME_RATE_HZ, DECAY_TAU_S), build_config(),
                   activity_image=image)
    assert candidate.gate_outcome != "reject", candidate.gate_reasons


def test_gate3_handles_no_candidates_beside_prior_rois():
    """Stage 3 can find nothing on a FOV where Stages 1 and 2 kept ROIs; Gate 3 then returns an empty list."""
    prior = ROI(mask=build_disk_mask(), label_id=1, source_stage=1, confidence="high", gate_outcome="accept")
    image = build_activity_image(cell_brightness=1.0, rng=np.random.default_rng(12))
    result = evaluate_gate3([], [prior], build_template_bank(FRAME_RATE_HZ, DECAY_TAU_S), build_config(),
                            activity_image=image)
    assert result == []


def test_gate3_rejects_a_trace_with_missing_samples():
    """A non-finite sample makes the sign measure undefined; the check fails rather than passes."""
    rng = np.random.default_rng(13)
    trace = build_transient_trace(rng)
    trace[100] = np.nan
    candidate = build_candidate(trace)
    image = build_activity_image(cell_brightness=1.0, rng=rng)
    evaluate_gate3([candidate], [], build_template_bank(FRAME_RATE_HZ, DECAY_TAU_S), build_config(),
                   activity_image=image)
    assert any(reason.startswith("negative_excursions") for reason in candidate.gate_reasons)


def test_gate3_rejects_an_activity_image_with_missing_pixels():
    """A non-finite pixel under the candidate makes the contrast undefined; the check fails rather than passes."""
    rng = np.random.default_rng(14)
    candidate = build_candidate(build_transient_trace(rng))
    image = build_activity_image(cell_brightness=1.0, rng=rng)
    image[CELL_CENTER] = np.nan
    evaluate_gate3([candidate], [], build_template_bank(FRAME_RATE_HZ, DECAY_TAU_S), build_config(),
                   activity_image=image)
    assert any(reason.startswith("activity_contrast") for reason in candidate.gate_reasons)


def test_gate3_records_both_measures():
    rng = np.random.default_rng(8)
    candidate = build_candidate(build_transient_trace(rng))
    image = build_activity_image(cell_brightness=1.0, rng=rng)
    evaluate_gate3([candidate], [], build_template_bank(FRAME_RATE_HZ, DECAY_TAU_S), build_config(),
                   activity_image=image)
    assert candidate.features["gate3_negative_excursion_share"] == 0.0
    assert candidate.features["gate3_activity_contrast"] > 2.0


def test_gate3_skips_the_contrast_check_without_an_activity_image():
    rng = np.random.default_rng(9)
    candidate = build_candidate(build_transient_trace(rng))
    evaluate_gate3([candidate], [], build_template_bank(FRAME_RATE_HZ, DECAY_TAU_S), build_config())
    assert candidate.gate_outcome != "reject", candidate.gate_reasons
    assert candidate.features["gate3_activity_contrast"] is None
