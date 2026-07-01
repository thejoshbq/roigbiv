"""Synthetic soma injection for benchmark ground-truth generation (issue #31).

Injects controlled synthetic somas into an existing background movie — not a
full generative simulator. Five soma types are supported: dim, overlapping,
sparse_transient, slow_modulation, elevated_baseline.

SNR definition: target SNR is injected center-pixel peak amplitude divided by
the background's per-pixel temporal std (median over the soma's spatial
footprint, computed once from the pristine input before any injection).
``amplitude = snr_target * noise_floor``, with the temporal profile
peak-normalized to 1.0 and center spatial weight 1.0. For ``dim``/``overlapping``
specs with ``n_events > 1``, temporally close events superpose before
peak-normalization, so an individual event's realized SNR can undershoot
``snr_target``; this does not affect the default ``n_events=1``.

Ground truth: ``InjectionResult.soma_masks`` (N, H, W) is the authoritative
per-soma mask stack and always preserves every soma exactly, including
overlaps. ``InjectionResult.label_mask`` (H, W) is a convenience single-label
image for visualization/pipeline-comparison only — where somas' spatial
footprints overlap, the later soma's ``label_id`` overwrites the earlier one
in the shared pixels. Prefer ``soma_masks`` for anything that must be exact.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import tifffile

from roigbiv.pipeline.stage3_templates import build_template_bank

_DEFAULT_SNR_BANDS: dict[str, tuple[float, float]] = {
    "dim": (1.5, 2.5),
    "overlapping": (3.0, 4.0),
    "sparse_transient": (5.0, 6.0),
    "slow_modulation": (4.0, 5.0),
    "elevated_baseline": (3.0, 4.0),
}


@dataclass
class SomaSpec:
    soma_type: str
    center: tuple[int, int]
    radius: float = 4.0
    snr_target: float | None = None
    label_id: int = 0
    n_events: int = 1
    event_rate_hz: float = 0.1
    mod_period_s: float = 30.0
    falloff_sigma2: float = 8.0


@dataclass
class InjectionResult:
    movie: np.ndarray
    label_mask: np.ndarray
    soma_masks: np.ndarray
    specs: list[SomaSpec] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


def _disk_footprint(
    center: tuple[int, int], radius: float, falloff_sigma2: float, shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    H, W = shape
    cy, cx = center
    r = int(np.ceil(radius))
    ys_list: list[int] = []
    xs_list: list[int] = []
    weights_list: list[float] = []
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            dist2 = dy * dy + dx * dx
            if dist2 > radius * radius:
                continue
            y, x = cy + dy, cx + dx
            if 0 <= y < H and 0 <= x < W:
                ys_list.append(y)
                xs_list.append(x)
                weights_list.append(float(np.exp(-dist2 / falloff_sigma2)))
    return (
        np.asarray(ys_list, dtype=np.intp),
        np.asarray(xs_list, dtype=np.intp),
        np.asarray(weights_list, dtype=np.float32),
    )


def _noise_floor(bg_std: np.ndarray, ys: np.ndarray, xs: np.ndarray) -> float:
    return float(np.median(bg_std[ys, xs]))


def _phasic_profile(T: int, template: np.ndarray, event_times: list[int]) -> np.ndarray:
    profile = np.zeros(T, dtype=np.float32)
    for t0 in event_times:
        end = min(t0 + len(template), T)
        if end <= t0:
            continue
        profile[t0:end] += template[: end - t0]
    peak = float(profile.max())
    if peak > 0:
        profile = profile / peak
    return profile


def _temporal_profile(
    spec: SomaSpec, T: int, template: np.ndarray, fs: float,
) -> tuple[np.ndarray, list[int]]:
    if spec.soma_type in ("dim", "overlapping"):
        last_start = max(T - len(template), 0)
        event_times = [
            int(t) for t in np.linspace(0, last_start, spec.n_events, dtype=int)
        ]
        return _phasic_profile(T, template, event_times), event_times

    if spec.soma_type == "slow_modulation":
        t = np.arange(T, dtype=np.float32)
        period_frames = max(spec.mod_period_s * fs, 1e-6)
        profile = 0.5 * (1.0 + np.sin(2.0 * np.pi * t / period_frames))
        return profile.astype(np.float32), []

    if spec.soma_type == "elevated_baseline":
        return np.ones(T, dtype=np.float32), []

    raise ValueError(f"unknown soma_type: {spec.soma_type!r}")


def _sparse_transient_profile(
    T: int, template: np.ndarray, event_rate_hz: float, fs: float, rng: np.random.Generator,
) -> tuple[np.ndarray, list[int]]:
    event_times: list[int] = []
    if event_rate_hz > 0:
        mean_interval_frames = fs / event_rate_hz
        t = float(rng.exponential(mean_interval_frames))
        while t < T:
            event_times.append(int(t))
            t += float(rng.exponential(mean_interval_frames))
    return _phasic_profile(T, template, event_times), event_times


def inject_somas(
    movie: np.ndarray,
    specs: list[SomaSpec],
    *,
    fs: float = 7.5,
    tau: float = 1.0,
    seed: int = 0,
    in_place: bool = False,
) -> InjectionResult:
    if in_place and np.issubdtype(movie.dtype, np.integer):
        raise ValueError("in_place=True requires a float movie array, got integer dtype")

    T, H, W = movie.shape
    bg_std = movie.astype(np.float32, copy=False).std(axis=0)

    if in_place:
        working = movie
    else:
        working = movie.astype(np.float32, copy=True)

    rng = np.random.default_rng(seed)

    if not specs:
        return InjectionResult(
            movie=working,
            label_mask=np.zeros((H, W), dtype=np.uint16),
            soma_masks=np.zeros((0, H, W), dtype=bool),
            specs=[],
            metadata={"seed": seed, "shape": [T, H, W], "fs": fs, "tau": tau, "somas": []},
        )

    bank = build_template_bank(fs=fs, tau=tau)
    template = bank[0][1]

    label_mask = np.zeros((H, W), dtype=np.uint16)
    soma_masks: list[np.ndarray] = []
    finalized_specs: list[SomaSpec] = []
    somas_meta: list[dict] = []
    next_label_id = 1
    used_label_ids: set[int] = set()

    for spec in specs:
        label_id = spec.label_id if spec.label_id != 0 else next_label_id
        if label_id in used_label_ids:
            raise ValueError(
                f"label_id {label_id} is already assigned to another soma "
                f"(soma_type={spec.soma_type!r}, center={spec.center!r})"
            )
        used_label_ids.add(label_id)
        next_label_id = max(next_label_id, label_id + 1)

        if spec.snr_target is None:
            lo, hi = _DEFAULT_SNR_BANDS[spec.soma_type]
            snr_target = float(rng.uniform(lo, hi))
        else:
            snr_target = spec.snr_target

        resolved = replace(spec, label_id=label_id, snr_target=snr_target)
        finalized_specs.append(resolved)

        if spec.soma_type == "sparse_transient":
            profile, event_times = _sparse_transient_profile(
                T, template, spec.event_rate_hz, fs, rng,
            )
        else:
            profile, event_times = _temporal_profile(resolved, T, template, fs)

        ys, xs, weights = _disk_footprint(spec.center, spec.radius, spec.falloff_sigma2, (H, W))

        soma_mask = np.zeros((H, W), dtype=bool)
        if len(ys) == 0:
            soma_masks.append(soma_mask)
            somas_meta.append({
                "soma_type": spec.soma_type,
                "center": [int(spec.center[0]), int(spec.center[1])],
                "radius": float(spec.radius),
                "label_id": int(label_id),
                "snr_target": float(snr_target),
                "amplitude": 0.0,
                "noise_floor": 0.0,
                "event_times": event_times,
            })
            continue

        noise_floor = _noise_floor(bg_std, ys, xs)
        amplitude = snr_target * noise_floor

        working[:, ys, xs] += amplitude * weights[np.newaxis, :] * profile[:, np.newaxis]

        label_mask[ys, xs] = label_id
        soma_mask[ys, xs] = True
        soma_masks.append(soma_mask)

        somas_meta.append({
            "soma_type": spec.soma_type,
            "center": [int(spec.center[0]), int(spec.center[1])],
            "radius": float(spec.radius),
            "label_id": int(label_id),
            "snr_target": float(snr_target),
            "amplitude": float(amplitude),
            "noise_floor": float(noise_floor),
            "event_times": [int(t) for t in event_times],
        })

    metadata = {
        "seed": int(seed),
        "shape": [int(T), int(H), int(W)],
        "fs": float(fs),
        "tau": float(tau),
        "somas": somas_meta,
    }

    return InjectionResult(
        movie=working,
        label_mask=label_mask,
        soma_masks=np.stack(soma_masks, axis=0),
        specs=finalized_specs,
        metadata=metadata,
    )


def save_injection(
    result: InjectionResult, output_dir: Path, *, save_movie: bool = False,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}

    masks_tif = output_dir / "ground_truth_masks.tif"
    tifffile.imwrite(str(masks_tif), result.label_mask)
    written["ground_truth_masks.tif"] = masks_tif

    masks_npy = output_dir / "ground_truth_masks.npy"
    np.save(str(masks_npy), result.soma_masks)
    written["ground_truth_masks.npy"] = masks_npy

    metadata_json = output_dir / "injection_metadata.json"
    with open(metadata_json, "w") as f:
        json.dump(result.metadata, f, indent=2)
    written["injection_metadata.json"] = metadata_json

    if save_movie:
        movie_npy = output_dir / "injected_movie.npy"
        np.save(str(movie_npy), result.movie)
        written["injected_movie.npy"] = movie_npy

    return written


def default_spec(soma_type: str, center: tuple[int, int], **overrides) -> SomaSpec:
    return SomaSpec(soma_type=soma_type, center=center, **overrides)


def overlapping_pair(
    center: tuple[int, int], offset: tuple[int, int] = (3, 3), **kw,
) -> list[SomaSpec]:
    cy, cx = center
    dy, dx = offset
    return [
        SomaSpec(soma_type="overlapping", center=(cy, cx), **kw),
        SomaSpec(soma_type="overlapping", center=(cy + dy, cx + dx), **kw),
    ]


def inject_from_tif(
    input_path: Path, specs: list[SomaSpec], output_dir: Path, **kw,
) -> InjectionResult:
    movie = tifffile.imread(str(input_path)).astype(np.float32)
    result = inject_somas(movie, specs, **kw)
    save_injection(result, output_dir)
    return result
