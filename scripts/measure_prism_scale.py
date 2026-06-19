"""Empirical cell-scale measurement on mean_M summary images.

One-off diagnostic used to ground the prism preset values for the Logan
remediation. Outputs median equivalent diameter + 5th/95th percentile
area across detected somata in three FOV mean_M.tif files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile
from skimage.feature import peak_local_max
from skimage.filters import difference_of_gaussians, threshold_otsu
from skimage.measure import regionprops, label


def measure_fov(mean_m_path: Path, n_peaks: int = 20, box_radius: int = 40):
    img = tifffile.imread(str(mean_m_path)).astype(np.float32)
    dog = difference_of_gaussians(img, low_sigma=3.0, high_sigma=15.0)
    # peak detection
    min_distance = 25
    peaks = peak_local_max(
        dog,
        min_distance=min_distance,
        threshold_rel=0.15,
        num_peaks=n_peaks,
        exclude_border=box_radius,
    )
    diameters = []
    areas = []
    H, W = img.shape
    for (y, x) in peaks:
        y0, y1 = max(0, y - box_radius), min(H, y + box_radius)
        x0, x1 = max(0, x - box_radius), min(W, x + box_radius)
        crop = img[y0:y1, x0:x1]
        if crop.size < 100:
            continue
        # Otsu on local crop
        try:
            t = threshold_otsu(crop)
        except Exception:
            continue
        binmask = crop > t
        # connected components, take component containing center
        lab = label(binmask)
        cy, cx = y - y0, x - x0
        if lab[cy, cx] == 0:
            continue
        target = lab[cy, cx]
        for r in regionprops((lab == target).astype(np.uint8)):
            if r.area < 30 or r.area > 8000:
                continue
            diameters.append(r.equivalent_diameter)
            areas.append(r.area)
            break
    return np.array(diameters), np.array(areas), peaks


def summarize(name: str, diameters: np.ndarray, areas: np.ndarray) -> dict:
    if diameters.size == 0:
        return {"name": name, "n": 0}
    return {
        "name": name,
        "n": int(diameters.size),
        "diameter_med": float(np.median(diameters)),
        "diameter_p5": float(np.percentile(diameters, 5)),
        "diameter_p95": float(np.percentile(diameters, 95)),
        "area_med": float(np.median(areas)),
        "area_p5": float(np.percentile(areas, 5)),
        "area_p95": float(np.percentile(areas, 95)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base",
        default="/home/thejoshbq/Otis-Lab/Projects/roigbiv/data/logan_cousa_trial/output",
    )
    parser.add_argument(
        "--fovs",
        nargs="+",
        default=[
            "052126_DS-Prism-3_VI15_D2_FOV2_pre-005",
            "052126_DS-Prism-3_VI15_D2_FOV2_beh-006",
            "052126_DS-Prism-3_VI15_D2_FOV2_post-007",
        ],
    )
    args = parser.parse_args()

    base = Path(args.base)
    all_d, all_a = [], []
    for fov in args.fovs:
        mean_m = base / fov / "summary" / "mean_M.tif"
        if not mean_m.exists():
            print(f"SKIP {fov}: {mean_m} missing", file=sys.stderr)
            continue
        d, a, peaks = measure_fov(mean_m)
        s = summarize(fov, d, a)
        print(f"\n[{fov}] n_peaks_found={len(peaks)} n_measured={s.get('n', 0)}")
        if s.get("n", 0):
            print(
                f"  diameter (px): med={s['diameter_med']:.1f}"
                f"  p5={s['diameter_p5']:.1f}"
                f"  p95={s['diameter_p95']:.1f}"
            )
            print(
                f"  area (px^2):   med={s['area_med']:.0f}"
                f"  p5={s['area_p5']:.0f}"
                f"  p95={s['area_p95']:.0f}"
            )
            all_d.extend(d.tolist())
            all_a.extend(a.tolist())

    if all_d:
        all_d_arr = np.array(all_d)
        all_a_arr = np.array(all_a)
        print("\n=== AGGREGATE (across FOVs) ===")
        print(f"n = {all_d_arr.size}")
        print(
            f"diameter (px): med={np.median(all_d_arr):.1f}"
            f"  p5={np.percentile(all_d_arr, 5):.1f}"
            f"  p95={np.percentile(all_d_arr, 95):.1f}"
        )
        print(
            f"area (px^2):   med={np.median(all_a_arr):.0f}"
            f"  p5={np.percentile(all_a_arr, 5):.0f}"
            f"  p95={np.percentile(all_a_arr, 95):.0f}"
        )
        print("\n=== SUGGESTED PRESET ===")
        print(f"  --diameter {int(round(np.median(all_d_arr)))}")
        print(f"  --min-area {int(np.percentile(all_a_arr, 5))}")
        print(f"  --max-area {int(np.percentile(all_a_arr, 95) * 1.5)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
