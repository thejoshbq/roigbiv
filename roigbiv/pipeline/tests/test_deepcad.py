"""Tests for DeepCAD-RT denoiser backend (roigbiv/pipeline/deepcad.py).

Tests the sidecar subprocess driver via stub scripts, verifying:
(1) happy path with correct output validation,
(2) subprocess error handling with setup hints,
(3) output shape/dtype/NaN validation and cleanup,
(4) conda env name validation (no injection attacks).
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import tifffile

from roigbiv.pipeline.deepcad import run_deepcad_denoise
from roigbiv.pipeline.types import PipelineConfig


def _write_stub_sidecar(tmp_path, body: str) -> Path:
    """Write a Python script stub to tmp_path / 'stub.py'.

    The body string is the full script content; it reads sys.argv[1] as the
    manifest JSON path and performs whatever the test specifies.

    Parameters
    ----------
    tmp_path : Path
        Temporary directory.
    body : str
        Full script content (will be written as-is to stub.py).

    Returns
    -------
    Path
        Path to the written stub.py file.
    """
    stub_path = tmp_path / "stub.py"
    with open(stub_path, "w") as f:
        f.write(body)
    return stub_path


def test_happy_path_copies_input_to_output():
    """Happy path: stub copies input to output, metadata written, returns path."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_dir = temp_path / "input"
        output_dir = temp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        # Create synthetic uint16 input data.
        input_data = np.random.randint(0, 65535, size=(10, 32, 32), dtype=np.uint16)
        tif_path = input_dir / "fov1.tif"
        tifffile.imwrite(str(tif_path), input_data)

        # Stub script: read manifest, copy input to output, exit 0.
        stub_script = """\
import json
import sys
from pathlib import Path

import numpy as np
import tifffile

manifest_path = sys.argv[1]
with open(manifest_path, 'r') as f:
    manifest = json.load(f)

input_path = Path(manifest['input'])
output_path = Path(manifest['output'])

data = tifffile.imread(str(input_path))
tifffile.imwrite(str(output_path), data)
sys.exit(0)
"""
        stub_path = _write_stub_sidecar(temp_path, stub_script)

        # Config: both python and script set, so _probe_env is skipped.
        cfg = PipelineConfig(
            deepcad_denoise=True,
            deepcad_python=sys.executable,
            deepcad_script=str(stub_path),
        )

        # Run and validate.
        output_tif_path = run_deepcad_denoise(tif_path, output_dir, cfg)

        assert output_tif_path.exists()
        output_data = tifffile.imread(str(output_tif_path))
        assert output_data.shape == (10, 32, 32)
        assert output_data.dtype == np.uint16

        # Metadata JSON must exist with exactly the expected keys.
        metadata_path = output_dir / "fov1_deepcad_metadata.json"
        assert metadata_path.exists()
        with open(metadata_path) as f:
            metadata = json.load(f)
        assert set(metadata.keys()) == {
            "input_path",
            "output_path",
            "exit_code",
            "cmd_line",
        }
        assert metadata["exit_code"] == 0


def test_nonzero_exit_raises_with_setup_hint():
    """Nonzero worker exit: RuntimeError with setup instructions."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_dir = temp_path / "input"
        output_dir = temp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        input_data = np.random.randint(0, 65535, size=(10, 32, 32), dtype=np.uint16)
        tif_path = input_dir / "fov1.tif"
        tifffile.imwrite(str(tif_path), input_data)

        # Stub: write error to stderr and exit with code 1.
        stub_script = """\
import sys

print("ImportError: no module named 'deepcad'", file=sys.stderr)
sys.exit(1)
"""
        stub_path = _write_stub_sidecar(temp_path, stub_script)

        cfg = PipelineConfig(
            deepcad_denoise=True,
            deepcad_python=sys.executable,
            deepcad_script=str(stub_path),
        )

        with pytest.raises(RuntimeError) as exc_info:
            run_deepcad_denoise(tif_path, output_dir, cfg)

        exc_str = str(exc_info.value)
        # All three hint substrings must be present.
        assert "deepcad sidecar" in exc_str
        assert "conda env" in exc_str
        assert "ROIGBIV_DEEPCAD_PYTHON" in exc_str

        # Cleanup: no TIFF or metadata left behind.
        output_tif_path = output_dir / "fov1_deepcad.tif"
        metadata_path = output_dir / "fov1_deepcad_metadata.json"
        assert not output_tif_path.exists()
        assert not metadata_path.exists()


def test_wrong_frame_count_raises_and_cleans_up():
    """Shape mismatch (frame count): RuntimeError, invalid TIFF deleted."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_dir = temp_path / "input"
        output_dir = temp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        input_data = np.random.randint(0, 65535, size=(10, 32, 32), dtype=np.uint16)
        tif_path = input_dir / "fov1.tif"
        tifffile.imwrite(str(tif_path), input_data)

        # Stub: write wrong number of frames (5 instead of 10).
        stub_script = """\
import json
import sys
from pathlib import Path

import numpy as np
import tifffile

manifest_path = sys.argv[1]
with open(manifest_path, 'r') as f:
    manifest = json.load(f)

output_path = Path(manifest['output'])
wrong_data = np.random.randint(0, 65535, size=(5, 32, 32), dtype=np.uint16)
tifffile.imwrite(str(output_path), wrong_data)
sys.exit(0)
"""
        stub_path = _write_stub_sidecar(temp_path, stub_script)

        cfg = PipelineConfig(
            deepcad_denoise=True,
            deepcad_python=sys.executable,
            deepcad_script=str(stub_path),
        )

        with pytest.raises(RuntimeError) as exc_info:
            run_deepcad_denoise(tif_path, output_dir, cfg)

        assert "shape mismatch" in str(exc_info.value)

        # Cleanup: invalid TIFF and metadata must be deleted.
        output_tif_path = output_dir / "fov1_deepcad.tif"
        metadata_path = output_dir / "fov1_deepcad_metadata.json"
        assert not output_tif_path.exists()
        assert not metadata_path.exists()


def test_wrong_dtype_raises():
    """Dtype mismatch: RuntimeError, invalid TIFF deleted."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_dir = temp_path / "input"
        output_dir = temp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        input_data = np.random.randint(0, 65535, size=(10, 32, 32), dtype=np.uint16)
        tif_path = input_dir / "fov1.tif"
        tifffile.imwrite(str(tif_path), input_data)

        # Stub: write correct shape but wrong dtype (float32 instead of uint16).
        stub_script = """\
import json
import sys
from pathlib import Path

import numpy as np
import tifffile

manifest_path = sys.argv[1]
with open(manifest_path, 'r') as f:
    manifest = json.load(f)

output_path = Path(manifest['output'])
wrong_data = np.random.rand(10, 32, 32).astype(np.float32)
tifffile.imwrite(str(output_path), wrong_data)
sys.exit(0)
"""
        stub_path = _write_stub_sidecar(temp_path, stub_script)

        cfg = PipelineConfig(
            deepcad_denoise=True,
            deepcad_python=sys.executable,
            deepcad_script=str(stub_path),
        )

        with pytest.raises(RuntimeError) as exc_info:
            run_deepcad_denoise(tif_path, output_dir, cfg)

        assert "dtype mismatch" in str(exc_info.value)

        # Cleanup check.
        output_tif_path = output_dir / "fov1_deepcad.tif"
        assert not output_tif_path.exists()


def test_nan_output_raises():
    """NaN/Inf in output: RuntimeError, invalid TIFF deleted.

    Note: uint16 cannot represent NaN, so we use float32 input for this test.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_dir = temp_path / "input"
        output_dir = temp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        # Use float32 input so output can also be float32 (matching dtype).
        input_data = np.random.rand(10, 32, 32).astype(np.float32)
        tif_path = input_dir / "fov1.tif"
        tifffile.imwrite(str(tif_path), input_data)

        # Stub: write all NaN values.
        stub_script = """\
import json
import sys
from pathlib import Path

import numpy as np
import tifffile

manifest_path = sys.argv[1]
with open(manifest_path, 'r') as f:
    manifest = json.load(f)

output_path = Path(manifest['output'])
nan_data = np.full((10, 32, 32), np.nan, dtype=np.float32)
tifffile.imwrite(str(output_path), nan_data)
sys.exit(0)
"""
        stub_path = _write_stub_sidecar(temp_path, stub_script)

        cfg = PipelineConfig(
            deepcad_denoise=True,
            deepcad_python=sys.executable,
            deepcad_script=str(stub_path),
        )

        with pytest.raises(RuntimeError) as exc_info:
            run_deepcad_denoise(tif_path, output_dir, cfg)

        assert "non-finite" in str(exc_info.value)

        # Cleanup check.
        output_tif_path = output_dir / "fov1_deepcad.tif"
        assert not output_tif_path.exists()


def test_missing_output_raises():
    """Missing output file after clean exit: RuntimeError."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_dir = temp_path / "input"
        output_dir = temp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        input_data = np.random.randint(0, 65535, size=(10, 32, 32), dtype=np.uint16)
        tif_path = input_dir / "fov1.tif"
        tifffile.imwrite(str(tif_path), input_data)

        # Stub: read manifest, ensure output doesn't exist, exit cleanly.
        stub_script = """\
import json
import sys
import os
from pathlib import Path

manifest_path = sys.argv[1]
with open(manifest_path, 'r') as f:
    manifest = json.load(f)

output_path = Path(manifest['output'])
# Remove the output file if it exists (e.g., from a prior failed run)
if output_path.exists():
    output_path.unlink()

# Exit without writing output
sys.exit(0)
"""
        stub_path = _write_stub_sidecar(temp_path, stub_script)

        cfg = PipelineConfig(
            deepcad_denoise=True,
            deepcad_python=sys.executable,
            deepcad_script=str(stub_path),
        )

        with pytest.raises(RuntimeError) as exc_info:
            run_deepcad_denoise(tif_path, output_dir, cfg)

        assert "was not created" in str(exc_info.value)


@pytest.mark.parametrize(
    "bad_env",
    [
        "bad env",  # space — not in the allowed character set
        "-n",  # leading dash — the actual argument-injection vector this
        "--help",  # guard exists to close (would be parsed as a conda flag,
        "-malicious",  # not a value, if it ever reached `conda run -n <env>`)
    ],
)
def test_invalid_env_name_rejected(monkeypatch, bad_env):
    """Invalid conda env name: RuntimeError without subprocess launch.

    This test does NOT override deepcad_python/deepcad_script, so the
    env-name validation in _interp_cmd/_probe_env fires before subprocess.
    ROIGBIV_DEEPCAD_PYTHON is explicitly unset so a developer's/CI's shell
    env can't accidentally take the interpreter-override path and skip
    env-name validation entirely.
    """
    monkeypatch.delenv("ROIGBIV_DEEPCAD_PYTHON", raising=False)
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_dir = temp_path / "input"
        output_dir = temp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        input_data = np.random.randint(0, 65535, size=(10, 32, 32), dtype=np.uint16)
        tif_path = input_dir / "fov1.tif"
        tifffile.imwrite(str(tif_path), input_data)

        cfg = PipelineConfig(deepcad_denoise=True, deepcad_env=bad_env)

        with pytest.raises(RuntimeError) as exc_info:
            run_deepcad_denoise(tif_path, output_dir, cfg)

        assert "invalid conda env name" in str(exc_info.value)


def test_paths_with_spaces_succeed():
    """Paths with spaces: list-form subprocess.run handles them safely."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_dir = temp_path / "input"
        output_dir = temp_path / "dir with spaces" / "output"
        input_dir.mkdir()
        output_dir.mkdir(parents=True)

        input_data = np.random.randint(0, 65535, size=(10, 32, 32), dtype=np.uint16)
        tif_path = input_dir / "fov1.tif"
        tifffile.imwrite(str(tif_path), input_data)

        # Stub: happy path copy.
        stub_script = """\
import json
import sys
from pathlib import Path

import numpy as np
import tifffile

manifest_path = sys.argv[1]
with open(manifest_path, 'r') as f:
    manifest = json.load(f)

input_path = Path(manifest['input'])
output_path = Path(manifest['output'])

data = tifffile.imread(str(input_path))
tifffile.imwrite(str(output_path), data)
sys.exit(0)
"""
        stub_path = _write_stub_sidecar(temp_path, stub_script)

        cfg = PipelineConfig(
            deepcad_denoise=True,
            deepcad_python=sys.executable,
            deepcad_script=str(stub_path),
        )

        output_tif_path = run_deepcad_denoise(tif_path, output_dir, cfg)

        assert output_tif_path.exists()
        output_data = tifffile.imread(str(output_tif_path))
        assert output_data.shape == (10, 32, 32)
        assert output_data.dtype == np.uint16

        metadata_path = output_dir / "fov1_deepcad_metadata.json"
        assert metadata_path.exists()
        with open(metadata_path) as f:
            metadata = json.load(f)
        assert metadata["exit_code"] == 0


def test_stale_output_never_trusted_when_worker_is_a_noop():
    """A stale valid-looking output at the target path must not be mistaken
    for this run's result if the worker exits 0 without actually writing.

    Regression test: ensure_free_space (diskguard) pre-creates/fallocates the
    output path before the subprocess runs, so "does the file exist after a
    clean exit" cannot be the freshness signal. run_deepcad_denoise must
    unlink any pre-existing file at that path before staging the run.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_dir = temp_path / "input"
        output_dir = temp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        input_data = np.random.randint(0, 65535, size=(10, 32, 32), dtype=np.uint16)
        tif_path = input_dir / "fov1.tif"
        tifffile.imwrite(str(tif_path), input_data)

        # Pre-seed a stale, VALID, matching-shape/dtype TIFF at the exact
        # output path a prior run would have used — simulates leftover state
        # from a previous crashed/partial run.
        stale_output = output_dir / "fov1_deepcad.tif"
        stale_data = np.random.randint(0, 65535, size=(10, 32, 32), dtype=np.uint16)
        tifffile.imwrite(str(stale_output), stale_data)

        # No-op worker: exits 0 without writing anything (simulates a broken
        # worker that silently does nothing rather than a crash).
        stub_script = """\
import sys
sys.exit(0)
"""
        stub_path = _write_stub_sidecar(temp_path, stub_script)

        cfg = PipelineConfig(
            deepcad_denoise=True,
            deepcad_python=sys.executable,
            deepcad_script=str(stub_path),
        )

        # Must fail rather than silently returning the stale file as if it
        # were this run's fresh output.
        with pytest.raises(RuntimeError):
            run_deepcad_denoise(tif_path, output_dir, cfg)


def test_stem_only_strips_trailing_mc_suffix():
    """A stem containing "_mc" as a substring (not just a trailing motion-
    correction marker) must not have every occurrence stripped."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_dir = temp_path / "input"
        output_dir = temp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        input_data = np.random.randint(0, 65535, size=(10, 32, 32), dtype=np.uint16)
        # "_mc" appears both mid-stem and as the trailing marker.
        tif_path = input_dir / "fov_mc_test_mc.tif"
        tifffile.imwrite(str(tif_path), input_data)

        stub_script = """\
import json
import sys
from pathlib import Path

import tifffile

manifest_path = sys.argv[1]
with open(manifest_path, 'r') as f:
    manifest = json.load(f)

data = tifffile.imread(manifest['input'])
tifffile.imwrite(manifest['output'], data)
sys.exit(0)
"""
        stub_path = _write_stub_sidecar(temp_path, stub_script)

        cfg = PipelineConfig(
            deepcad_denoise=True,
            deepcad_python=sys.executable,
            deepcad_script=str(stub_path),
        )

        output_tif_path = run_deepcad_denoise(tif_path, output_dir, cfg)

        # Only the trailing "_mc" is stripped; the mid-stem occurrence stays.
        assert output_tif_path.name == "fov_mc_test_deepcad.tif"


def test_bad_python_override_raises_actionable_error():
    """A nonexistent interpreter override (typo'd deepcad_python) must fail
    gracefully with a setup hint, not a bare FileNotFoundError traceback.

    This exercises the main subprocess launch (not the _probe_env pre-flight,
    which is skipped on this exact override path).
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        input_dir = temp_path / "input"
        output_dir = temp_path / "output"
        input_dir.mkdir()
        output_dir.mkdir()

        input_data = np.random.randint(0, 65535, size=(10, 32, 32), dtype=np.uint16)
        tif_path = input_dir / "fov1.tif"
        tifffile.imwrite(str(tif_path), input_data)

        # A real file must exist at deepcad_script for the worker.exists()
        # check to pass; its content is never executed because the
        # interpreter override itself is bogus.
        stub_path = _write_stub_sidecar(temp_path, "sys.exit(0)\n")

        cfg = PipelineConfig(
            deepcad_denoise=True,
            deepcad_python="/nonexistent/interpreter/xyz123",
            deepcad_script=str(stub_path),
        )

        with pytest.raises(RuntimeError) as exc_info:
            run_deepcad_denoise(tif_path, output_dir, cfg)

        exc_str = str(exc_info.value)
        assert "could not launch the sidecar interpreter" in exc_str
        assert "ROIGBIV_DEEPCAD_PYTHON" in exc_str
