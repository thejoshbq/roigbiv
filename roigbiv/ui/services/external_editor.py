"""Discover and launch a TIFF-capable external editor (Fiji / ImageJ / GIMP).

ROI editing is intentionally not in the Dash UI — see
``docs/external-editing.md``. The Review page surfaces a button that calls
:func:`launch_editor` to open the active session's mask in whichever editor
is installed locally; the researcher edits there, saves, then runs
``roigbiv-reingest`` to fold the diff into the corrections log.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from shutil import which


class EditorNotFoundError(RuntimeError):
    """No TIFF-capable editor was discovered on this machine."""


_PATH_NAMES_FIJI: tuple[str, ...] = (
    "fiji",
    "ImageJ-linux64",
    "ImageJ-macosx",
    "ImageJ-win64.exe",
    "imagej",
    "ImageJ",
)

_PATH_NAMES_GIMP: tuple[str, ...] = (
    "gimp",
    "gimp-2.10",
)


def _candidate_install_paths() -> list[Path]:
    """Platform-aware list of likely Fiji install locations."""
    home = Path.home()
    if sys.platform.startswith("linux"):
        return [
            home / "Fiji.app" / "ImageJ-linux64",
            home / "fiji" / "ImageJ-linux64",
            Path("/opt/Fiji.app/ImageJ-linux64"),
            Path("/opt/fiji/ImageJ-linux64"),
            Path("/usr/local/Fiji.app/ImageJ-linux64"),
        ]
    if sys.platform == "darwin":
        return [
            Path("/Applications/Fiji.app/Contents/MacOS/ImageJ-macosx"),
            home / "Applications" / "Fiji.app" / "Contents" / "MacOS"
            / "ImageJ-macosx",
        ]
    if sys.platform.startswith("win"):
        return [
            Path(r"C:\Program Files\Fiji.app\ImageJ-win64.exe"),
            Path(r"C:\Program Files (x86)\Fiji.app\ImageJ-win64.exe"),
        ]
    return []


def find_tiff_editor() -> Path:
    """Return the resolved path to a TIFF-capable editor.

    Resolution order:

      1. ``$ROIGBIV_TIFF_EDITOR`` — explicit override. May be a bare name
         resolved through PATH or an absolute path.
      2. Fiji / ImageJ binaries on PATH.
      3. Fiji install dirs (platform-aware).
      4. GIMP fallback (``gimp`` / ``gimp-2.10``).

    Raises :class:`EditorNotFoundError` if nothing resolves.
    """
    override = os.environ.get("ROIGBIV_TIFF_EDITOR", "").strip()
    if override:
        resolved = which(override)
        if resolved:
            return Path(resolved)
        as_path = Path(override).expanduser()
        if as_path.is_file() and os.access(as_path, os.X_OK):
            return as_path
        raise EditorNotFoundError(
            f"ROIGBIV_TIFF_EDITOR={override!r} is set but does not point to "
            "an executable."
        )

    for name in _PATH_NAMES_FIJI:
        resolved = which(name)
        if resolved:
            return Path(resolved)

    for candidate in _candidate_install_paths():
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate

    for name in _PATH_NAMES_GIMP:
        resolved = which(name)
        if resolved:
            return Path(resolved)

    raise EditorNotFoundError(
        "No TIFF editor found. Install Fiji (https://fiji.sc/) or set "
        "ROIGBIV_TIFF_EDITOR to the path of an editor binary that accepts "
        "a TIFF file as its first argument."
    )


def resolve_mask_target(output_dir: Path) -> Path:
    """Return the mask TIF the editor should open.

    Prefers ``corrections/corrected_masks.tif`` (the live edit state) when
    it exists, else ``merged_masks.tif`` (the frozen pipeline output).
    """
    output_dir = Path(output_dir)
    corrected = output_dir / "corrections" / "corrected_masks.tif"
    if corrected.is_file():
        return corrected
    merged = output_dir / "merged_masks.tif"
    if merged.is_file():
        return merged
    raise FileNotFoundError(
        f"No mask file found under {output_dir} "
        "(looked for corrections/corrected_masks.tif and merged_masks.tif)."
    )


def launch_editor(target: Path) -> Path:
    """Spawn an editor on ``target`` in the background. Return the editor path.

    The editor is detached from the Dash worker via ``start_new_session``
    (POSIX) / ``CREATE_NEW_PROCESS_GROUP`` (Windows) and inherits no stdio,
    so closing the Dash server does not kill the editor.
    """
    editor = find_tiff_editor()
    target = Path(target)
    popen_kwargs: dict = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
        "close_fds": True,
    }
    if sys.platform.startswith("win"):
        popen_kwargs["creationflags"] = (
            getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
            | getattr(subprocess, "DETACHED_PROCESS", 0)
        )
    else:
        popen_kwargs["start_new_session"] = True
    subprocess.Popen([str(editor), str(target)], **popen_kwargs)
    return editor
