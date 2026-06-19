"""Cellpose-SAM (CP4) detector — sidecar, ``cellpose>=4``.

Conflicts with the repo's ``cellpose<4.0.0`` pin, so it runs in the ``cp-sam``
conda env (build with ``bash envs/build_cellpose_sam.sh``). Strong zero-shot
generalist segmentation — no fine-tuning needed.
"""
from __future__ import annotations

from pathlib import Path

from cv_bakeoff.detector import DetectorInputs, DetectorResult
from cv_bakeoff.detectors._sidecar import run_sidecar

_BUILD_HINT = (
    "create it with `bash envs/build_cellpose_sam.sh` and verify with "
    "`conda run -n cp-sam python -c \"import cellpose\"`. Or point "
    "ROIGBIV_CPSAM_PYTHON at a Python interpreter that has cellpose>=4."
)


def _worker_path() -> Path:
    # scripts/cv_bakeoff/detectors/cp_sam.py -> scripts/cv_bakeoff/workers/...
    return Path(__file__).resolve().parents[1] / "workers" / "cp_sam_worker.py"


class CPSAMDetector:
    name = "cp-sam"

    def __init__(
        self,
        env: str = "cp-sam",
        channel: str = "mean_M",
        diameter: float | None = None,
        flow_threshold: float = 0.4,
    ):
        self.env = env
        self.channel = channel
        self.diameter = diameter
        self.flow_threshold = flow_threshold

    def detect(self, inputs: DetectorInputs) -> DetectorResult:
        extra = ["--channel", self.channel,
                 "--flow-threshold", str(self.flow_threshold)]
        if self.diameter is not None:
            extra += ["--diameter", str(self.diameter)]
        return run_sidecar(
            env=self.env,
            override_var="ROIGBIV_CPSAM_PYTHON",
            import_stmt="import cellpose",
            build_hint=_BUILD_HINT,
            worker_path=_worker_path(),
            method=self.name,
            channels={self.channel: inputs.summary[self.channel]},
            stem=inputs.fov_stem,
            extra_args=extra,
        )
