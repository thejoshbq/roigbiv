# External ROI editing (Fiji / ImageJ handoff)

The ROIGBIV UI deliberately doesn't ship a polygon / freehand / eraser
editor. ImageJ and Fiji are the de facto standard for 2-photon ROI work
and have far more capable editing tools — so when corrections are needed
we hand off to them and round-trip the result through a single CLI.

## Workflow

1. In the **Review** page, pick the FOV and the session you want to edit.
   The "Edit ROIs in Fiji / ImageJ" card shows the active session's
   output directory and a **Open output folder** button.

2. Open the output dir in Fiji. The two relevant files are:

   - `merged_masks.tif`  — the frozen pipeline ROI labels (uint16; one
     integer per ROI).
   - `summary/mean_M.tif` — the mean projection (greyscale; layer
     underneath as a background).

   If you've already round-tripped corrections through the CLI before,
   prefer `corrections/corrected_masks.tif` over `merged_masks.tif` —
   that's the live state.

   Recommended Fiji setup:

   - `Image → Color → Image LUT → Glasbey` on the label image so each
     label gets a distinct color.
   - `Image → Overlay → Add Image…` to layer the labels over `mean_M`.

3. Edit. Add new ROIs, delete spurious ones, fix boundaries — whatever
   the data calls for. ImageJ doesn't care about preserving the original
   label numbering; the reingest step recovers per-ROI continuity by
   IoU.

4. Save the edited label image as a TIFF (e.g. `edited_masks.tif`).

5. Back on the terminal:

   ```bash
   roigbiv-reingest \
     --output-dir /path/to/inference/pipeline/<stem> \
     --new-mask /path/to/edited_masks.tif \
     --notes "Fiji touch-up YYYY-MM-DD"
   ```

   The CLI diffs the new mask against the current state (frozen pipeline
   output + any previous corrections) and emits one entry per change to
   `corrections/corrections.jsonl`. It also re-materialises
   `corrections/corrected_masks.tif` and `corrections/corrected_metadata.json`.

   Pass `--dry-run` first if you want to preview the diff:

   ```bash
   roigbiv-reingest \
     --output-dir … --new-mask … --dry-run
   ```

## What the diff does

For each ROI in the current state, the reingest CLI looks for the
edited ROI with the highest IoU. The four buckets:

| Outcome  | Threshold       | Op emitted                         |
| -------- | --------------- | ---------------------------------- |
| Preserve | `IoU >= 0.95`   | (none — treated as unchanged)      |
| Edit     | `0.50 <= IoU`   | `edit` (preserves original label)  |
| Delete   | unmatched       | `delete`                           |
| Add      | unmatched (new) | `add`                              |

Both thresholds are tunable: `--preserve-iou 0.95` and `--edit-iou 0.50`.

## Why round-trip rather than open-and-replace?

`corrections.jsonl` is the audit log of record. Every external edit is
appended as a discrete `CorrectionOp`, timestamped, with the supplied
note attached. The pipeline outputs (`merged_masks.tif`,
`roi_metadata.json`, the per-stage TIFFs) are never overwritten — the
only thing that ever changes is the corrections directory.

That means you can:

- Replay the whole correction history at any time
  (`apply_corrections(rois, load_corrections(...), shape)`).
- Revert by deleting the JSONL line(s) for an erroneous edit and
  re-materialising.
- Trace any ROI in the corrected output back to either a pipeline stage
  or a specific external edit.
