"""
ROI G. Biv — Sequential subtractive detection pipeline.

Replaces the parallel three-branch (Cellpose + Suite2p + tonic) consensus
pipeline with a strictly sequential, subtractive chain:

    Foundation (motion correction + SVD + L+S + summary images)
      -> Stage 1 Cellpose -> Gate 1 morphology -> Source Subtraction -> S1
      -> Stage 2 Suite2p temporal -> Gate 2 cross-validation -> Subtract -> S2
      -> Stage 3 Template sweep -> Gate 3 waveform -> Subtract -> S3
      -> Stage 4 Tonic search -> Gate 4 correlation contrast validation
      -> Unified QC + HITL

The full four-stage detection chain is implemented. Unified QC and HITL
review slot in afterward (Phase 1F).

See docs/roi-pipeline-specification.md for the authoritative algorithm reference.
"""

from roigbiv.pipeline.types import ROI, FOVData, PipelineConfig
from roigbiv.pipeline.run import run_pipeline, main
from roigbiv.pipeline.napari_viewer import display_pipeline_results
from roigbiv.pipeline.stage2 import run_stage2, extract_traces_from_residual
from roigbiv.pipeline.gate2 import evaluate_gate2
from roigbiv.pipeline.stage3 import run_stage3
from roigbiv.pipeline.stage3_templates import build_template_bank
from roigbiv.pipeline.gate3 import evaluate_gate3
from roigbiv.pipeline.stage4 import run_stage4
from roigbiv.pipeline.gate4 import evaluate_gate4

__all__ = [
    "ROI",
    "FOVData",
    "PipelineConfig",
    "run_pipeline",
    "main",
    "display_pipeline_results",
    "run_stage2",
    "extract_traces_from_residual",
    "evaluate_gate2",
    "run_stage3",
    "build_template_bank",
    "evaluate_gate3",
    "run_stage4",
    "evaluate_gate4",
]
