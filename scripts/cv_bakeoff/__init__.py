"""CV bake-off — compare segmentation methods on FOV summary images.

Research tooling (not shipped with the package). Runs several interchangeable
detectors on the same summary images and emits side-by-side overlay grids for
visual comparison. No ground truth required.

Outputs go to ``experiments/runs/cv_bakeoff/``; nothing is written to
``inference/``. See ``run_bakeoff.py`` for the CLI.
"""
