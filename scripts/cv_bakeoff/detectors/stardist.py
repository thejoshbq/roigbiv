"""StarDist 2D detector — sidecar, TensorFlow.

Runs the pretrained ``2D_versatile_fluo`` model in the ``stardist`` conda env
(build with ``bash envs/build_stardist.sh``). Star-convex blob prior fits round
somata / nuclei in fluorescence well.
"""
from __future__ import annotations

from pathlib import Path

from cv_bakeoff.detector import DetectorInputs, DetectorResult
from cv_bakeoff.detectors._sidecar import run_sidecar

_BUILD_HINT = (
    "create it with `bash envs/build_stardist.sh` and verify with "
    "`conda run -n stardist python -c \"import stardist\"`. Or point "
    "ROIGBIV_STARDIST_PYTHON at a Python interpreter that has stardist."
)


def _worker_path() -> Path:
    return Path(__file__).resolve().parents[1] / "workers" / "stardist_worker.py"


class StarDistDetector:
    name = "stardist"

    def __init__(
        self,
        env: str = "stardist",
        channel: str = "mean_M",
        model: str = "2D_versatile_fluo",
        prob_thresh: float | None = None,
        nms_thresh: float | None = None,
    ):
        self.env = env
        self.channel = channel
        self.model = model
        self.prob_thresh = prob_thresh
        self.nms_thresh = nms_thresh

    def detect(self, inputs: DetectorInputs) -> DetectorResult:
        extra = ["--channel", self.channel, "--model", self.model]
        if self.prob_thresh is not None:
            extra += ["--prob-thresh", str(self.prob_thresh)]
        if self.nms_thresh is not None:
            extra += ["--nms-thresh", str(self.nms_thresh)]
        return run_sidecar(
            env=self.env,
            override_var="ROIGBIV_STARDIST_PYTHON",
            import_stmt="import stardist",
            build_hint=_BUILD_HINT,
            worker_path=_worker_path(),
            method=self.name,
            channels={self.channel: inputs.summary[self.channel]},
            stem=inputs.fov_stem,
            extra_args=extra,
        )
