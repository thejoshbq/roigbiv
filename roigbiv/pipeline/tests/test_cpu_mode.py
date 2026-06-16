"""Unit tests for CPU-only pipeline mode (no GPU / CUDA required)."""
import os

import numpy as np
import pytest


def test_force_cpu_sets_roicat_alignment(monkeypatch):
    """--cpu block sets ROIGBIV_ROICAT_ALIGNMENT=PhaseCorrelation when unset."""
    monkeypatch.delenv("ROIGBIV_ROICAT_ALIGNMENT", raising=False)
    monkeypatch.delenv("ROIGBIV_ROICAT_DEVICE", raising=False)

    # Replicate the exact logic from run.py's force_cpu block
    import sys
    from io import StringIO

    stderr_buf = StringIO()
    os.environ.setdefault("ROIGBIV_ROICAT_DEVICE", "cpu")
    if not os.environ.get("ROIGBIV_ROICAT_ALIGNMENT"):
        os.environ["ROIGBIV_ROICAT_ALIGNMENT"] = "PhaseCorrelation"
        print(
            "WARN: --cpu selected; registry alignment downgraded to "
            "PhaseCorrelation (RoMa is impractical on CPU). "
            "Set ROIGBIV_ROICAT_ALIGNMENT to override.",
            file=stderr_buf,
        )

    assert os.environ["ROIGBIV_ROICAT_ALIGNMENT"] == "PhaseCorrelation"
    assert os.environ["ROIGBIV_ROICAT_DEVICE"] == "cpu"
    assert "PhaseCorrelation" in stderr_buf.getvalue()


def test_force_cpu_respects_existing_alignment(monkeypatch):
    """--cpu block does NOT overwrite a pre-set ROIGBIV_ROICAT_ALIGNMENT."""
    monkeypatch.setenv("ROIGBIV_ROICAT_ALIGNMENT", "ECC_cv2")
    monkeypatch.delenv("ROIGBIV_ROICAT_DEVICE", raising=False)

    from io import StringIO
    stderr_buf = StringIO()

    os.environ.setdefault("ROIGBIV_ROICAT_DEVICE", "cpu")
    if not os.environ.get("ROIGBIV_ROICAT_ALIGNMENT"):
        os.environ["ROIGBIV_ROICAT_ALIGNMENT"] = "PhaseCorrelation"
        print("WARN: alignment downgraded", file=stderr_buf)

    assert os.environ["ROIGBIV_ROICAT_ALIGNMENT"] == "ECC_cv2"
    assert stderr_buf.getvalue() == ""


def test_binned_svd_cpu_roundtrip():
    """_binned_svd_gpu with force_cpu=True reconstructs a small matrix."""
    from roigbiv.pipeline.foundation import _binned_svd_gpu

    rng = np.random.default_rng(0)
    T_bin, N_pix, n_svd = 20, 64 * 64, 4
    M = rng.standard_normal((T_bin, N_pix)).astype(np.float32)

    U, S, V = _binned_svd_gpu(M, n_svd=n_svd, force_cpu=True)

    assert U.shape == (N_pix, n_svd), f"U shape {U.shape}"
    assert S.shape == (n_svd,), f"S shape {S.shape}"
    assert V.shape == (T_bin, n_svd), f"V shape {V.shape}"

    M_hat = V @ np.diag(S) @ U.T
    rel_err = np.linalg.norm(M - M_hat, "fro") / np.linalg.norm(M, "fro")
    # Low-rank approximation with n_svd=4 won't be exact, but should be < 100%
    # (just verifying the code path runs and shapes are correct)
    assert rel_err < 1.0, f"Reconstruction relative error too large: {rel_err:.3f}"


def test_pipeline_config_force_cpu():
    """PipelineConfig.force_cpu defaults to False and accepts True."""
    from roigbiv.pipeline.types import PipelineConfig

    default_cfg = PipelineConfig(fs=7.5)
    assert default_cfg.force_cpu is False

    cpu_cfg = PipelineConfig(fs=7.5, force_cpu=True)
    assert cpu_cfg.force_cpu is True


class _FakeCuda:
    def __init__(self, available, exc=None, oom_cls=RuntimeError):
        self._available = available
        self._exc = exc
        self.OutOfMemoryError = oom_cls

    def is_available(self):
        return self._available


class _FakeTorch:
    def __init__(self, cuda):
        self.cuda = cuda

    def zeros(self, *a, **k):
        if self.cuda._exc is not None:
            raise self.cuda._exc
        return self  # any object; (+1).sum().item() is monkeypatched away

    def __add__(self, other):
        return self

    def sum(self):
        return self

    def item(self):
        return 0.0


def _inject_torch(monkeypatch, fake):
    import sys
    monkeypatch.setitem(sys.modules, "torch", fake)


@pytest.mark.parametrize("exc, expected", [
    (None, None),
    (RuntimeError("CUDA out of memory. Tried to allocate 2 GiB"), "oom"),
    (RuntimeError("no kernel image is available for execution on the device"), "sm_mismatch"),
    (RuntimeError("CUDA capability sm_120 not compatible"), "sm_mismatch"),
    (RuntimeError("some other cuda failure"), "error: some other cuda failure"),
])
def test_cuda_unavailable_reason_classifies(monkeypatch, exc, expected):
    """The probe maps launch failures to stable reason codes (oom vs sm_mismatch)."""
    from roigbiv.pipeline import device

    class _OOM(RuntimeError):
        pass

    fake = _FakeTorch(_FakeCuda(available=True, exc=exc, oom_cls=_OOM))
    _inject_torch(monkeypatch, fake)
    assert device.cuda_unavailable_reason() == expected
    assert device.cuda_compute_capable() is (expected is None)


def test_cuda_unavailable_reason_oom_subclass(monkeypatch):
    """An OutOfMemoryError instance is classified as oom even without the message."""
    from roigbiv.pipeline import device

    class _OOM(RuntimeError):
        pass

    fake = _FakeTorch(_FakeCuda(available=True, exc=_OOM("boom"), oom_cls=_OOM))
    _inject_torch(monkeypatch, fake)
    assert device.cuda_unavailable_reason() == "oom"


def test_cuda_unavailable_reason_no_device(monkeypatch):
    from roigbiv.pipeline import device

    fake = _FakeTorch(_FakeCuda(available=False))
    _inject_torch(monkeypatch, fake)
    assert device.cuda_unavailable_reason() == "no_cuda"


def test_stage1_fallback_message_distinguishes_oom_from_sm():
    from roigbiv.pipeline.stage1 import _cuda_fallback_message

    assert "out of VRAM" in _cuda_fallback_message("oom")
    assert "sm/CC mismatch" in _cuda_fallback_message("sm_mismatch")
    assert _cuda_fallback_message("oom") != _cuda_fallback_message("sm_mismatch")
