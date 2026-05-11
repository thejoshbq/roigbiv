"""Background service for the Cellpose fine-tuning workflow.

Covers the full HITL fine-tuning loop:

  1. launch_gui     — open hitl_staging images in Cellpose GUI (detached subprocess)
  2. start_ingest   — convert *_seg.npy corrections → *_masks.tif (daemon thread)
  3. start_training — run ``scripts/train.py`` in background (daemon thread)
  4. deploy         — copy best checkpoint to ``models/deployed/current_model``

All long-running work runs in daemon threads so Dash callbacks return immediately.
The UI polls the :class:`TrainerSnapshot` via ``dcc.Interval``.
"""
from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
import threading
import time
import traceback
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import tifffile

from roigbiv.ui.services.external_editor import resolve_mask_target

_MAX_LOG_LINES = 2_000

# roigbiv/ui/services/ → roigbiv/ui/ → roigbiv/ (pkg) → roigbiv/ (project root)
_BASE_DIR: Path = Path(__file__).resolve().parents[3]
_SCRIPTS_DIR: Path = _BASE_DIR / "scripts"
_DEPLOYED_MODEL: Path = _BASE_DIR / "models" / "deployed" / "current_model"
_CHECKPOINTS_DIR: Path = _BASE_DIR / "models" / "checkpoints" / "models"


class CellposeNotFoundError(RuntimeError):
    """``cellpose`` package not importable in the active environment."""


@dataclass
class TrainerSnapshot:
    """Serializable snapshot for the Review page's interval callback."""

    state: str                         # idle | ingesting | training | done | error
    logs: list[str] = field(default_factory=list)
    run_id: Optional[str] = None
    error: Optional[str] = None


class CellposeTrainer:
    """Single-slot trainer covering the full Cellpose HITL fine-tuning loop."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state = "idle"
        self._logs: deque[str] = deque(maxlen=_MAX_LOG_LINES)
        self._thread: Optional[threading.Thread] = None
        self._run_id: Optional[str] = None
        self._error: Optional[str] = None

    # ── GUI launch ────────────────────────────────────────────────────────────

    def launch_gui(self, output_dir: Path) -> Path:
        """Open the FOV mean projection in the Cellpose GUI (detached subprocess).

        Uses ``cellpose.gui.gui.run(image=...)`` directly rather than the CLI
        binary, because the Cellpose CLI dispatches to headless inference whenever
        ``--dir`` or ``--image_path`` is provided (cellpose/__main__.py:45) —
        there is no CLI flag to preload an image without leaving GUI mode.

        Returns the staging images directory path for display in the UI.
        """
        output_dir = Path(output_dir)
        staging_images = output_dir / "hitl_staging" / "images"
        if not staging_images.exists():
            raise FileNotFoundError(
                f"No hitl_staging/images found under {output_dir}. "
                "Run the pipeline to generate staging materials first."
            )

        if importlib.util.find_spec("cellpose.gui.gui") is None:
            raise CellposeNotFoundError(
                "cellpose not importable in the current environment. "
                "Activate the roigbiv conda environment and try again."
            )

        image_tifs = sorted(staging_images.glob("*.tif"))
        if not image_tifs:
            raise FileNotFoundError(
                f"No staging images found in {staging_images}. "
                "Run the pipeline to generate staging materials first."
            )
        image_path = image_tifs[0]

        # Cellpose's gui.run(image=...) loads the image into the interactive GUI and
        # auto-detects {stem}_seg.npy beside the TIF to overlay existing masks.
        _prepare_staging_npy(staging_images, output_dir)
        cmd = [
            sys.executable, "-c",
            f"from cellpose.gui.gui import run; run(image={str(image_path)!r})",
        ]

        kwargs: dict = {
            "stdin": subprocess.DEVNULL,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "close_fds": True,
        }
        if sys.platform.startswith("win"):
            kwargs["creationflags"] = (
                getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                | getattr(subprocess, "DETACHED_PROCESS", 0)
            )
        else:
            kwargs["start_new_session"] = True

        subprocess.Popen(cmd, **kwargs)
        return staging_images

    # ── Ingest ────────────────────────────────────────────────────────────────

    def start_ingest(self, annotated_dir: Path, masks_dir: Path) -> bool:
        """Convert *_seg.npy files to *_masks.tif in a daemon thread.

        Returns ``False`` if a job is already running.
        """
        with self._lock:
            if self._state in ("ingesting", "training"):
                return False
            self._state = "ingesting"
            self._error = None
        t = threading.Thread(
            target=self._run_ingest,
            args=(Path(annotated_dir), Path(masks_dir)),
            name="roigbiv-cellpose-ingest",
            daemon=True,
        )
        self._thread = t
        t.start()
        return True

    def _run_ingest(self, annotated_dir: Path, masks_dir: Path) -> None:
        self._log(f"Ingest: scanning {annotated_dir}")
        try:
            masks_dir.mkdir(parents=True, exist_ok=True)
            seg_files = sorted(annotated_dir.glob("*_seg.npy"))
            if not seg_files:
                raise FileNotFoundError(
                    f"No *_seg.npy files found in {annotated_dir}. "
                    "Save corrections in Cellpose GUI first."
                )

            n_done = 0
            for seg_file in seg_files:
                stem = seg_file.stem
                for suffix in ("_mean_seg", "_max_seg", "_vcorr_seg", "_seg"):
                    if stem.endswith(suffix):
                        stem = stem[: -len(suffix)]
                        break

                data = np.load(str(seg_file), allow_pickle=True).item()
                masks = data.get("masks")
                if masks is None or masks.max() == 0:
                    self._log(f"  {seg_file.name}: empty masks, skipping")
                    continue

                n_rois = int(masks.max())
                out_path = masks_dir / f"{stem}_masks.tif"
                if out_path.exists():
                    backup = masks_dir / f"{stem}_masks.tif.bak"
                    shutil.copy2(str(out_path), str(backup))
                    self._log(f"  {stem}: backed up existing mask → .bak")

                tifffile.imwrite(str(out_path), masks.astype(np.uint16))
                self._log(f"  {stem}: {n_rois} ROIs → {out_path.name}")
                n_done += 1

            self._log(f"Ingest complete: {n_done} mask(s) written to {masks_dir}")
        except Exception as exc:
            tb = traceback.format_exc()
            with self._lock:
                self._error = f"{type(exc).__name__}: {exc}"
                self._state = "error"
            self._log(f"ERROR: {self._error}")
            for line in tb.strip().splitlines():
                self._log(line)
            return

        with self._lock:
            self._state = "idle"

    # ── Training ──────────────────────────────────────────────────────────────

    def start_training(
        self,
        run_id: str,
        data_dir: Path,
        masks_dir: Path,
        *,
        epochs: int = 200,
        lr: float = 0.05,
    ) -> bool:
        """Run ``scripts/train.py`` in a daemon thread.

        Returns ``False`` if a job is already running.
        """
        with self._lock:
            if self._state in ("ingesting", "training"):
                return False
            self._state = "training"
            self._run_id = run_id
            self._error = None
        t = threading.Thread(
            target=self._run_training,
            args=(run_id, Path(data_dir), Path(masks_dir), epochs, lr),
            name="roigbiv-cellpose-train",
            daemon=True,
        )
        self._thread = t
        t.start()
        return True

    def _run_training(
        self,
        run_id: str,
        data_dir: Path,
        masks_dir: Path,
        epochs: int,
        lr: float,
    ) -> None:
        train_script = _SCRIPTS_DIR / "train.py"
        cmd = [
            sys.executable, str(train_script),
            "--run_id", run_id,
            "--data_dir", str(data_dir),
            "--masks_dir", str(masks_dir),
            "--epochs", str(epochs),
            "--lr", str(lr),
        ]
        self._log(f"run_id={run_id}  epochs={epochs}  lr={lr}")
        self._log(f"$ {' '.join(str(c) for c in cmd)}")
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=str(_SCRIPTS_DIR),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert proc.stdout is not None
            for line in proc.stdout:
                self._log(line.rstrip())
            proc.wait()
            if proc.returncode != 0:
                raise RuntimeError(f"train.py exited with returncode {proc.returncode}")
            self._log("Training finished successfully.")
        except Exception as exc:
            tb = traceback.format_exc()
            with self._lock:
                self._error = f"{type(exc).__name__}: {exc}"
                self._state = "error"
            self._log(f"ERROR: {self._error}")
            for line in tb.strip().splitlines():
                self._log(line)
            return

        with self._lock:
            self._state = "done"

    # ── Deploy ────────────────────────────────────────────────────────────────

    def deploy(self, run_id: str) -> Optional[Path]:
        """Copy ``models/checkpoints/models/{run_id}`` to the deployed model slot.

        The existing ``current_model`` is backed up with a timestamp suffix
        before being overwritten. Returns the backup path, or ``None`` if
        there was nothing to back up.
        """
        src = _CHECKPOINTS_DIR / run_id
        if not src.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {src}\n"
                "Confirm training completed successfully."
            )

        _DEPLOYED_MODEL.parent.mkdir(parents=True, exist_ok=True)

        backup: Optional[Path] = None
        if _DEPLOYED_MODEL.exists():
            ts = time.strftime("%Y%m%d_%H%M%S")
            backup = _DEPLOYED_MODEL.parent / f"current_model_{ts}"
            shutil.copy2(str(_DEPLOYED_MODEL), str(backup))

        shutil.copy2(str(src), str(_DEPLOYED_MODEL))
        return backup

    # ── Query / reset ─────────────────────────────────────────────────────────

    def snapshot(self) -> TrainerSnapshot:
        with self._lock:
            return TrainerSnapshot(
                state=self._state,
                logs=list(self._logs)[-100:],
                run_id=self._run_id,
                error=self._error,
            )

    def reset(self) -> None:
        """Reset to idle state (no-op if a job is in flight)."""
        with self._lock:
            if self._state in ("ingesting", "training"):
                return
            self._state = "idle"
            self._logs.clear()
            self._run_id = None
            self._error = None

    # ── Internal ──────────────────────────────────────────────────────────────

    def _log(self, line: str) -> None:
        with self._lock:
            self._logs.append(line)


_trainer: Optional[CellposeTrainer] = None
_trainer_lock = threading.Lock()


def get_trainer() -> CellposeTrainer:
    """Return the process-local singleton :class:`CellposeTrainer`."""
    global _trainer
    if _trainer is None:
        with _trainer_lock:
            if _trainer is None:
                _trainer = CellposeTrainer()
    return _trainer


def _prepare_staging_npy(staging_images: Path, output_dir: Path) -> None:
    """Seed each staging image with a Cellpose-loadable ``{stem}_seg.npy``.

    Cellpose's ``_load_seg`` (``cellpose/gui/io.py:270``) requires the
    ``"outlines"`` key — without it, the loader hits a bare ``except``, prints
    "ERROR: not NPY", and shows no overlay. We compute outlines from the current
    pipeline mask (corrections if present, else ``merged_masks.tif``) and write
    the minimum keys Cellpose needs to display the overlay.

    Any prior ``_seg.npy`` is moved to ``{stem}_seg.npy.bak`` before refresh.
    Durable corrections live in ``hitl_staging/masks/{stem}_masks.tif`` after
    "Ingest corrections"; the seg.npy is just a working view of the pipeline mask.
    """
    from cellpose.utils import masks_to_outlines

    try:
        mask_tif = resolve_mask_target(output_dir)
        masks = tifffile.imread(str(mask_tif))
    except FileNotFoundError:
        # No masks available yet (pipeline hasn't run); Cellpose opens blank.
        return

    outlines = masks_to_outlines(masks).astype(np.uint16)

    for img_tif in sorted(staging_images.glob("*.tif")):
        stem = img_tif.stem
        npy_path = staging_images / f"{stem}_seg.npy"
        if npy_path.exists():
            backup = staging_images / f"{stem}_seg.npy.bak"
            shutil.move(str(npy_path), str(backup))
        img = tifffile.imread(str(img_tif))
        np.save(str(npy_path), {
            "masks": masks,
            "img": img,
            "outlines": outlines,
        })
