"""Plain-ASCII terminal formatting helpers for pipeline output.

All functions return str with no ANSI codes — safe for the Dash UI log console
which renders text verbatim through the _QueuedStdout batch shim.

Width convention:
  _W  = 72  — banner width (fov_banner, pipeline_complete). These are emitted
              from the main process only and never get the "[FOV X/N] " prefix.
  _SW = 60  — stage-header width (stage_header). In batch mode the worker shim
              prepends up to 13 chars, keeping total ≤ 80 cols.
"""

_W = 72
_SW = 60


def fov_banner(name: str, idx: int, total: int, width: int = _W) -> str:
    """Heavy-rule FOV header. Emit from main process only (not inside worker)."""
    rule = "=" * width
    label = f" FOV {idx}/{total} | {name}"
    return f"\n{rule}\n{label}\n{rule}"


def fov_separator(width: int = _W) -> str:
    """Light rule between FOVs in a sequential batch run. Main process only."""
    return "\n" + ("-" * width)


def stage_header(n: int | str, label: str, width: int = _SW) -> str:
    """Rule-style stage section header. Safe inside worker subprocesses."""
    prefix = f"--- Stage {n}: {label} "
    return "\n" + prefix + "-" * max(0, width - len(prefix))


def gate_outcome(n: int | str, det: int, acc: int, flg: int, rej: int) -> str:
    """Compact gate result line, 2-space indented."""
    return f"  Gate {n}: {det} detect | {acc} accept | {flg} flag | {rej} reject"


def sub_phase(label: str, elapsed_s: float | None = None) -> str:
    """Sub-process annotation, 2-space indented, with optional timing."""
    if elapsed_s is not None:
        return f"  {label} [{elapsed_s:.1f}s]"
    return f"  {label}"


def stage_done(elapsed_s: float) -> str:
    """Short stage-completion footer."""
    return f"  done  [{elapsed_s:.1f}s]"


def pipeline_complete(name: str, total_s: float | None = None, width: int = _W) -> str:
    """Heavy-rule pipeline completion banner. Emit from main process only."""
    rule = "=" * width
    body = f"  Pipeline complete: {name}"
    if total_s is not None:
        mins, secs = divmod(int(total_s), 60)
        body += f"  [{mins}m {secs:02d}s]" if mins else f"  [{secs}s]"
    return f"\n{rule}\n{body}\n{rule}"
