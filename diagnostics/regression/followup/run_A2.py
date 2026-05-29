"""Phase A.2 — IoU diff between A.0 (current pipeline) and A.1 (reference).

Writes:
  A2_summary.md     — match counts, unmatched cell lists, decision
  A2_missing.png    — reference cells with no A.0 match (potential recall gaps)
  A2_extra.png      — current-pipeline cells with no reference match (potential FP)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import tifffile

PROJECT = Path("/home/thejoshbq/Otis-Lab/Projects/roigbiv")
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "scripts"))

from diagnostic_compare import iou_match, render_overlay  # noqa: E402


def main() -> None:
    fov_dir = PROJECT / "test_output/cli_verify/T1_221209_PrL-NAc-G6-5M_HI-D1_FOV1_BEH_PT2-002"
    summary = fov_dir / "summary"
    out_dir = PROJECT / "diagnostics/regression/followup"

    mean_M = tifffile.imread(str(summary / "mean_M.tif")).astype(np.float32)
    a0_labels = tifffile.imread(str(out_dir / "A0_accept_labels.tif")).astype(np.uint32)
    a1_labels = tifffile.imread(str(out_dir / "A1_reference_labels.tif")).astype(np.uint32)

    n_a0 = int(np.unique(a0_labels[a0_labels > 0]).size)
    n_a1 = int(np.unique(a1_labels[a1_labels > 0]).size)
    print(f"A.0 (current pipeline) accepts: {n_a0}")
    print(f"A.1 (reference)        cells  : {n_a1}")

    # Match A.1 (reference) against A.0 (pipeline). unmatched_a = ref cells that the
    # pipeline missed → "recall gap" (what the hypothesis predicted).
    matches, missing_ref, extra_pipe = iou_match(a1_labels, a0_labels, min_iou=0.3)
    print(f"IoU-matched pairs (min_iou=0.3): {len(matches)}")
    print(f"Reference cells missing from pipeline (recall gap): {len(missing_ref)}")
    print(f"Pipeline cells with no reference match (potential FP/over-detection): {len(extra_pipe)}")

    # Build label images for just the missing / extra subsets
    missing_img = np.where(np.isin(a1_labels, np.asarray(missing_ref, dtype=a1_labels.dtype)),
                           a1_labels, 0).astype(np.uint32)
    extra_img = np.where(np.isin(a0_labels, np.asarray(extra_pipe, dtype=a0_labels.dtype)),
                         a0_labels, 0).astype(np.uint32)

    # Missing = red outlines on mean_M. We show all A.0 accepts as a dim green
    # context layer so the reader sees where the pipeline DID fire.
    render_overlay(
        mean_M,
        {
            f"A.0 accepts ({n_a0})": (a0_labels, (30, 140, 60)),
            f"A.1 missing from A.0 ({len(missing_ref)})": (missing_img, (255, 40, 40)),
        },
        out_path=out_dir / "A2_missing.png",
        title=f"A.2 — reference cells missing from current pipeline (recall gap)",
    )

    # Extra = yellow outlines on mean_M; A.1 reference shown as dim green context.
    render_overlay(
        mean_M,
        {
            f"A.1 reference ({n_a1})": (a1_labels, (30, 140, 60)),
            f"A.0 extras vs A.1 ({len(extra_pipe)})": (extra_img, (255, 210, 0)),
        },
        out_path=out_dir / "A2_extra.png",
        title=f"A.2 — current-pipeline cells with no reference match (over-detection)",
    )

    # Per-cell centroids for missing/extra (helps locate them on the FOV)
    from skimage.measure import regionprops

    def centroid_list(labels: np.ndarray, ids: list[int]) -> list[dict]:
        id_set = set(int(i) for i in ids)
        rows = []
        for p in regionprops(labels):
            if int(p.label) in id_set:
                rows.append({
                    "label": int(p.label),
                    "area": int(p.area),
                    "centroid_yx": [round(float(p.centroid[0]), 1),
                                    round(float(p.centroid[1]), 1)],
                })
        return rows

    missing_rows = centroid_list(a1_labels, missing_ref)
    extra_rows = centroid_list(a0_labels, extra_pipe)

    # Write markdown summary
    md_lines = [
        "# A.2 — IoU diff summary\n",
        "## Configuration",
        "- A.0 (current pipeline): `mean_M` + `vcorr_S`, `use_denoise=True`, `channels=(1,2)`, `max_area=600`",
        "- A.1 (reference):        `mean_M` only,       `use_denoise=False`, `channels=(0,0)`, no gate",
        "- IoU matching: min_iou=0.3, greedy",
        "",
        "## Counts",
        f"- A.0 accepted ROIs: **{n_a0}**",
        f"- A.1 reference ROIs: **{n_a1}**",
        f"- Matched pairs: **{len(matches)}**",
        f"- Reference cells missed by pipeline (recall gap): **{len(missing_ref)}**",
        f"- Pipeline cells with no reference match (over-detection): **{len(extra_pipe)}**",
        "",
        "## Decision",
    ]
    if len(missing_ref) == 0:
        md_lines.append("**H-stale confirmed.** The current pipeline covers every reference cell. "
                        "The absorption / recall-gap hypothesis does not apply to the post-fix code. "
                        "No further remediation needed.")
    elif len(missing_ref) <= 3:
        md_lines.append(f"**Borderline.** Only {len(missing_ref)} reference cells are unmatched — "
                        "likely boundary-disagreement cases or split/merge artifacts, not a recall regression. "
                        "Review missing centroids visually before proceeding.")
    else:
        md_lines.append(f"**Recall gap present.** {len(missing_ref)} reference cells are not covered by the "
                        "current pipeline. Proceed to Phase A.3 ablations (denoise-off / single-channel).")
    md_lines += ["", "## Missing reference cells (ref_id, area, centroid)",]
    if missing_rows:
        for r in missing_rows:
            md_lines.append(f"- id={r['label']}  area={r['area']}  yx={r['centroid_yx']}")
    else:
        md_lines.append("- (none)")
    md_lines += ["", "## Over-detected pipeline cells (no reference match)",]
    if extra_rows:
        for r in extra_rows:
            md_lines.append(f"- id={r['label']}  area={r['area']}  yx={r['centroid_yx']}")
    else:
        md_lines.append("- (none)")

    (out_dir / "A2_summary.md").write_text("\n".join(md_lines) + "\n")
    print(f"Saved {out_dir}/A2_summary.md")


if __name__ == "__main__":
    main()
