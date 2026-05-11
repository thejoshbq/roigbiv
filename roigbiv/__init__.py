"""ROI G. Biv — sequential subtractive ROI detection for two-photon calcium imaging.

Stages (see ``docs/roi-pipeline-specification.md``):
    Foundation → Stage 1 (Cellpose spatial)
              → Stage 2 (Suite2p temporal)
              → Stage 3 (template sweep on residual)
              → Stage 4 (tonic neuron search)

Entry points (installed by ``pyproject.toml``):
    roigbiv-pipeline  — run the pipeline on one FOV or a directory
                        (with optional email-on-done + overlay PNG)
    roigbiv-ui        — launch the Dash + Plotly web UI
    roigbiv-registry  — cross-session FOV + cell registry CLI
    roigbiv-reingest  — ingest externally-edited ROI masks
"""

__version__ = "0.3.0"

__all__ = ["__version__"]
