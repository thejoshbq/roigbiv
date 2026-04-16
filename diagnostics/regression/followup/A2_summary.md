# A.2 — IoU diff summary

## Configuration
- A.0 (current pipeline): `mean_M` + `vcorr_S`, `use_denoise=True`, `channels=(1,2)`, `max_area=600`
- A.1 (reference):        `mean_M` only,       `use_denoise=False`, `channels=(0,0)`, no gate
- IoU matching: min_iou=0.3, greedy

## Counts
- A.0 accepted ROIs: **101**
- A.1 reference ROIs: **94**
- Matched pairs: **93**
- Reference cells missed by pipeline (recall gap): **1**
- Pipeline cells with no reference match (over-detection): **8**

## Decision
**Borderline.** Only 1 reference cells are unmatched — likely boundary-disagreement cases or split/merge artifacts, not a recall regression. Review missing centroids visually before proceeding.

## Missing reference cells (ref_id, area, centroid)
- id=49  area=200  yx=[219.9, 303.3]

## Over-detected pipeline cells (no reference match)
- id=12  area=273  yx=[100.7, 359.9]
- id=27  area=260  yx=[151.0, 181.2]
- id=67  area=170  yx=[285.3, 370.9]
- id=72  area=199  yx=[308.2, 426.4]
- id=73  area=156  yx=[310.9, 372.5]
- id=77  area=242  yx=[333.3, 374.5]
- id=92  area=173  yx=[395.2, 374.2]
- id=95  area=415  yx=[411.7, 352.3]
