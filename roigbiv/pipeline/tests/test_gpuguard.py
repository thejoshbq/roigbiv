"""Unit tests for the GPU VRAM preflight (no GPU / no ollama required).

The orchestration in ``free_gpu_for_run`` is exercised by monkeypatching the
three side-effecting helpers (``_free_gpu_bytes``, ``_ollama_loaded_models``,
``_ollama_unload``) so the logic is tested without CUDA or a live ollama.
"""
from roigbiv.pipeline import gpuguard

_GB = gpuguard._GB


def _capture(monkeypatch, *, free_sequence, loaded, unload_ok=True):
    """Wire fakes; return (logs, calls) where calls tracks unloads."""
    logs: list[str] = []
    calls: list[str] = []
    seq = iter(free_sequence)
    last = {"v": free_sequence[-1]}

    def fake_free():
        try:
            last["v"] = next(seq)
        except StopIteration:
            pass
        return last["v"]

    monkeypatch.setattr(gpuguard, "_cuda_device_present", lambda: True)
    monkeypatch.setattr(gpuguard, "_free_gpu_bytes", fake_free)
    monkeypatch.setattr(gpuguard, "_ollama_loaded_models", lambda timeout_s: list(loaded))

    def fake_unload(model, timeout_s):
        calls.append(model)
        return unload_ok

    monkeypatch.setattr(gpuguard, "_ollama_unload", fake_unload)
    monkeypatch.setattr(gpuguard.time, "sleep", lambda s: None)
    return logs, calls


def test_disabled_is_noop(monkeypatch):
    called = {"present": False}
    monkeypatch.setattr(gpuguard, "_cuda_device_present",
                        lambda: called.__setitem__("present", True) or True)
    gpuguard.free_gpu_for_run(8.0, enabled=False)
    assert called["present"] is False  # short-circuits before any probe


def test_no_cuda_device_is_noop(monkeypatch):
    monkeypatch.setattr(gpuguard, "_cuda_device_present", lambda: False)
    # Should not even ask ollama what's loaded.
    monkeypatch.setattr(gpuguard, "_ollama_loaded_models",
                        lambda timeout_s: (_ for _ in ()).throw(AssertionError("queried")))
    gpuguard.free_gpu_for_run(8.0)  # must not raise


def test_unprobeable_vram_still_evicts(monkeypatch):
    """Regression: free=None (GPU too full to probe) must trigger eviction.

    Caught live — when the 30B model leaves ~170 MB free, mem_get_info can't
    even init a CUDA context and returns None; that must mean "contended,"
    not "no GPU, skip." After unload the probe starts reporting real bytes.
    """
    logs, calls = _capture(
        monkeypatch,
        free_sequence=[None, 15 * _GB],  # unprobeable, then freed
        loaded=["qwen3-coder:30b"],
    )
    gpuguard.free_gpu_for_run(8.0, log_cb=logs.append)
    assert calls == ["qwen3-coder:30b"]
    assert any("unprobeable" in m for m in logs)
    assert any("GPU freed" in m for m in logs)


def test_ample_headroom_skips_unload(monkeypatch):
    logs, calls = _capture(monkeypatch, free_sequence=[12 * _GB], loaded=["qwen3-coder:30b"])
    gpuguard.free_gpu_for_run(8.0, log_cb=logs.append)
    assert calls == []  # already enough free → no eviction
    assert logs == []


def test_evicts_when_below_threshold(monkeypatch):
    # 1 GB free at first probe, 15 GB after the unload+poll.
    logs, calls = _capture(
        monkeypatch,
        free_sequence=[1 * _GB, 15 * _GB],
        loaded=["qwen3-coder:30b"],
    )
    gpuguard.free_gpu_for_run(8.0, log_cb=logs.append)
    assert calls == ["qwen3-coder:30b"]
    assert any("GPU freed" in m for m in logs)


def test_no_models_loaded_warns_without_unload(monkeypatch):
    logs, calls = _capture(monkeypatch, free_sequence=[1 * _GB], loaded=[])
    gpuguard.free_gpu_for_run(8.0, log_cb=logs.append)
    assert calls == []
    assert any("no ollama model is loaded" in m for m in logs)


def test_still_tight_after_unload_warns(monkeypatch):
    # Stays at 1 GB even after unload (some other process holds VRAM).
    logs, calls = _capture(
        monkeypatch,
        free_sequence=[1 * _GB] * 50,
        loaded=["qwen3-coder:30b"],
    )
    gpuguard.free_gpu_for_run(8.0, timeout_s=1.0, log_cb=logs.append)
    assert calls == ["qwen3-coder:30b"]
    assert any("still tight" in m for m in logs)


def test_never_raises_on_unload_failure(monkeypatch):
    logs, calls = _capture(
        monkeypatch,
        free_sequence=[1 * _GB, 1 * _GB, 1 * _GB],
        loaded=["m1", "m2"],
        unload_ok=False,
    )
    # Both models attempted; failure is swallowed.
    gpuguard.free_gpu_for_run(8.0, timeout_s=1.0, log_cb=logs.append)
    assert calls == ["m1", "m2"]


def test_config_defaults():
    from roigbiv.pipeline.types import PipelineConfig
    cfg = PipelineConfig(fs=7.5)
    assert cfg.free_gpu is True
    assert cfg.gpu_min_free_gb == 8.0
