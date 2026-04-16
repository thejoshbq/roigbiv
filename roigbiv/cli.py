"""Terminal entry point for the ROIGBIV detection pipeline.

Wraps ``roigbiv.pipeline.run.run_pipeline`` so lab members can kick off a
detection run from a terminal, walk away, and receive a flat PNG overlay
by email when it finishes. Single-FOV and directory batch inputs are both
supported. Never reimplements pipeline logic.

The Streamlit app (``app.py``) and the pre-existing ``roigbiv-pipeline``
console script remain untouched — this is a parallel entry point.

Example
-------
    export ROIGBIV_SMTP_PASSWORD='xxxx xxxx xxxx xxxx'   # Gmail App Password
    roigbiv-cli \\
        --input test_raw/ \\
        --fs 30 \\
        --email-to scientist@example.com \\
        --smtp-user scientist@gmail.com

Gmail App Password setup (for Gmail SMTP with 2FA enabled):

    1. Go to https://myaccount.google.com/apppasswords
    2. Generate an app password for "Mail"
    3. export ROIGBIV_SMTP_PASSWORD="xxxx xxxx xxxx xxxx"
    4. (Optional) append that export to ~/.bashrc for persistence
"""
from __future__ import annotations

import argparse
import os
import smtplib
import sys
import time
import traceback
from dataclasses import dataclass, field
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path
from typing import Optional

from roigbiv.pipeline.types import FOVData, PipelineConfig

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_MODEL = "models/deployed/current_model"
_CHECKPOINTS_GLOB = "models/checkpoints/models/run*_epoch_*"
_MAX_ATTACH_BYTES = 20 * 1024 * 1024   # Gmail caps at 25 MiB — keep headroom


@dataclass
class _FOVResult:
    """Per-FOV outcome — populated whether the run succeeded or failed."""
    tif: Path
    output_dir: Path
    fov: Optional[FOVData] = None
    png_path: Optional[Path] = None
    duration_s: float = 0.0
    error: Optional[str] = None
    roi_counts: dict = field(default_factory=dict)


# ── Argument parsing ─────────────────────────────────────────────────────────


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="roigbiv-cli",
        description=(
            "Run the ROIGBIV detection pipeline from a terminal and email "
            "the resulting ROI overlay. Wraps roigbiv.pipeline.run.run_pipeline "
            "— Suite2p (Stage 2) is always included."
        ),
        epilog=(
            "Gmail App Password setup:\n"
            "  1. https://myaccount.google.com/apppasswords\n"
            "  2. Generate an app password for \"Mail\"\n"
            "  3. export ROIGBIV_SMTP_PASSWORD=\"xxxx xxxx xxxx xxxx\"\n"
            "  4. (Optional) add the export to ~/.bashrc for persistence\n\n"
            "Example:\n"
            "  roigbiv-cli --input stack_mc.tif --fs 30 \\\n"
            "      --email-to me@example.com --smtp-user me@gmail.com\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--input", required=True, type=Path,
                        help="Path to a .tif stack OR a directory of stacks.")
    parser.add_argument("--fs", required=True, type=float,
                        help="Acquisition frame rate (Hz).")
    parser.add_argument("--tau", type=float, default=1.0,
                        help="Indicator decay (default 1.0, GCaMP6s).")
    parser.add_argument("--model", type=str, default=_DEFAULT_MODEL,
                        help=("Cellpose model path or 'latest' to pick the "
                              "newest mtime in models/checkpoints/models/. "
                              f"Default: {_DEFAULT_MODEL}"))
    parser.add_argument("--diameter", type=int, default=12,
                        help="Cellpose diameter (default 12).")
    parser.add_argument("--cellprob-threshold", type=float, default=-2.0,
                        help="Cellpose cell-probability threshold (default -2.0).")
    parser.add_argument("--flow-threshold", type=float, default=0.6,
                        help="Cellpose flow threshold (default 0.6).")
    parser.add_argument("--channels", type=_parse_channels, default=(1, 2),
                        help="Cellpose channels as 'cyto,nucleus' (default 1,2).")
    parser.add_argument("--k", type=int, default=30,
                        help="k_background for L+S separation (default 30).")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help=("Output directory. Default: per-FOV auto path "
                              "(inference/pipeline/{stem}/)."))
    parser.add_argument("--n-workers", type=int, default=1,
                        help="Parallel FOV workers for batch mode (capped at 2).")

    parser.add_argument("--email-to", type=str, default=None,
                        help="Recipient email. Omit to skip email entirely.")
    parser.add_argument("--smtp-host", type=str, default="smtp.gmail.com",
                        help="SMTP server hostname.")
    parser.add_argument("--smtp-port", type=int, default=587,
                        help="SMTP server port (STARTTLS).")
    parser.add_argument("--smtp-user", type=str, default=None,
                        help="SMTP username. Required when --email-to is set.")
    parser.add_argument("--smtp-password-env", type=str, default="ROIGBIV_SMTP_PASSWORD",
                        help=("Env-var name holding the SMTP password. Never "
                              "pass the password on the command line."))
    parser.add_argument("--no-email", action="store_true",
                        help="Skip sending email even when --email-to is set.")

    args = parser.parse_args(argv)

    if args.email_to and not args.no_email and not args.smtp_user:
        parser.error("--smtp-user is required when --email-to is set")

    return args


def _parse_channels(spec: str) -> tuple:
    parts = [p.strip() for p in spec.split(",")]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"--channels must be 'int,int' (got {spec!r})"
        )
    try:
        return (int(parts[0]), int(parts[1]))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid --channels value: {exc}")


# ── Model / input resolution ─────────────────────────────────────────────────


def _resolve_model(spec: str) -> str:
    """Resolve the --model argument to a path or Cellpose builtin name."""
    if spec == "latest":
        candidates = sorted(
            _PROJECT_ROOT.glob(_CHECKPOINTS_GLOB),
            key=lambda p: p.stat().st_mtime,
        )
        if candidates:
            chosen = candidates[-1]
            print(f"Resolved --model latest → {chosen}", flush=True)
            return str(chosen)
        fallback = _PROJECT_ROOT / _DEFAULT_MODEL
        print(
            f"WARN: no checkpoints under {_CHECKPOINTS_GLOB}; "
            f"falling back to {fallback}",
            file=sys.stderr,
        )
        return str(fallback)
    return spec


def _iter_inputs(input_path: Path) -> list[Path]:
    path = input_path.resolve()
    if not path.exists():
        raise FileNotFoundError(f"--input path does not exist: {path}")
    if path.is_file():
        return [path]
    tifs = sorted(p for p in path.glob("*.tif") if p.is_file())
    if not tifs:
        raise FileNotFoundError(f"no .tif files found in {path}")
    return tifs


def _build_cfg(args: argparse.Namespace, tif: Path) -> PipelineConfig:
    if args.output_dir is not None:
        out_dir = args.output_dir.resolve() / tif.stem.replace("_mc", "")
    else:
        out_dir = None   # let run_pipeline compute the default per-FOV path

    return PipelineConfig(
        fs=args.fs,
        tau=args.tau,
        k_background=args.k,
        cellpose_model=_resolve_model(args.model),
        diameter=args.diameter,
        cellprob_threshold=args.cellprob_threshold,
        flow_threshold=args.flow_threshold,
        channels=args.channels,
        output_dir=out_dir,
        no_viewer=True,   # CLI is non-interactive
    )


# ── Execution ────────────────────────────────────────────────────────────────


def _roi_counts(fov: Optional[FOVData]) -> dict:
    if fov is None:
        return {}
    counts = {"accept": 0, "flag": 0, "reject": 0}
    for roi in fov.rois:
        counts[roi.gate_outcome] = counts.get(roi.gate_outcome, 0) + 1
    return counts


def _run_single(tif: Path, cfg: PipelineConfig, model_name: str) -> _FOVResult:
    from roigbiv.pipeline.run import run_pipeline
    from roigbiv import overlay as _overlay

    fov_stem = tif.stem.replace("_mc", "")
    t0 = time.perf_counter()
    print(f"=== Running pipeline on {tif.name} ===", flush=True)
    try:
        fov = run_pipeline(tif, cfg)
    except BaseException as exc:   # noqa: BLE001
        traceback.print_exc()
        out_dir = cfg.output_dir if cfg.output_dir else tif.parent
        return _FOVResult(
            tif=tif,
            output_dir=Path(out_dir),
            duration_s=time.perf_counter() - t0,
            error=f"{type(exc).__name__}: {exc}",
        )

    duration = time.perf_counter() - t0

    png_path: Optional[Path] = None
    try:
        png_path = _overlay.render_overlay(
            fov=fov,
            output_dir=fov.output_dir,
            model_name=Path(cfg.cellpose_model).name,
            fov_stem=fov_stem,
        )
        print(f"Overlay written: {png_path}", flush=True)
    except BaseException as exc:   # noqa: BLE001
        print(f"WARN: overlay render failed for {fov_stem}: {exc}", file=sys.stderr)

    return _FOVResult(
        tif=tif,
        output_dir=fov.output_dir,
        fov=fov,
        png_path=png_path,
        duration_s=duration,
        roi_counts=_roi_counts(fov),
    )


def _run_batch(
    jobs: list[tuple[Path, PipelineConfig]],
    n_workers: int,
    model_name: str,
) -> list[_FOVResult]:
    from roigbiv.pipeline import batch as _batch
    from roigbiv import overlay as _overlay

    results: list[Optional[_FOVResult]] = [None] * len(jobs)
    start_times: dict[int, float] = {}

    def _log_cb(idx: int, line: str) -> None:
        print(f"[FOV {idx}] {line}", flush=True)

    def _complete_cb(idx: int, fov: Optional[FOVData], exc: Optional[BaseException]) -> None:
        tif, cfg = jobs[idx]
        duration = time.perf_counter() - start_times.get(idx, time.perf_counter())
        if exc is not None:
            results[idx] = _FOVResult(
                tif=tif,
                output_dir=cfg.output_dir or tif.parent,
                duration_s=duration,
                error=f"{type(exc).__name__}: {exc}",
            )
            return
        results[idx] = _FOVResult(
            tif=tif,
            output_dir=fov.output_dir if fov else (cfg.output_dir or tif.parent),
            fov=fov,
            duration_s=duration,
            roi_counts=_roi_counts(fov),
        )

    for i in range(len(jobs)):
        start_times[i] = time.perf_counter()

    print(f"=== Batch: {len(jobs)} FOVs, n_workers={n_workers} ===", flush=True)
    _batch.run_batch(
        jobs=jobs,
        n_workers=n_workers,
        log_callback=_log_cb,
        on_complete=_complete_cb,
    )

    for idx, res in enumerate(results):
        if res is None or res.error is not None:
            continue
        tif, cfg = jobs[idx]
        fov_stem = tif.stem.replace("_mc", "")
        try:
            if res.fov is None:
                print(
                    f"[FOV {idx}] live FOVData missing — reloading from disk",
                    flush=True,
                )
                res.png_path = _overlay.render_overlay_from_disk(
                    output_dir=res.output_dir,
                    model_name=model_name,
                    fov_stem=fov_stem,
                )
            else:
                res.png_path = _overlay.render_overlay(
                    fov=res.fov,
                    output_dir=res.output_dir,
                    model_name=model_name,
                    fov_stem=fov_stem,
                )
            print(f"[FOV {idx}] Overlay written: {res.png_path}", flush=True)
        except BaseException as exc:   # noqa: BLE001
            print(f"WARN: overlay render failed for {fov_stem}: {exc}", file=sys.stderr)

    return [r for r in results if r is not None]


# ── Email ────────────────────────────────────────────────────────────────────


def _send_email(results: list[_FOVResult], args: argparse.Namespace) -> bool:
    """Send a single email with all overlay attachments.

    Returns True on success, False otherwise. On failure, PNGs remain on
    disk — the caller prints a warning and continues.
    """
    password = os.environ.get(args.smtp_password_env)
    if not password:
        print(
            f"WARN: env var {args.smtp_password_env!r} is unset; "
            f"skipping email. Overlay(s) remain at:",
            file=sys.stderr,
        )
        for r in results:
            if r.png_path:
                print(f"  {r.png_path}", file=sys.stderr)
        return False

    subject, body, attachments = _compose_message(results, args)
    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = args.smtp_user
    msg["To"] = args.email_to
    msg.attach(MIMEText(body, "plain"))
    for name, payload in attachments:
        part = MIMEImage(payload, _subtype="png")
        part.add_header("Content-Disposition", "attachment", filename=name)
        msg.attach(part)

    try:
        with smtplib.SMTP(args.smtp_host, args.smtp_port, timeout=30) as server:
            server.starttls()
            server.login(args.smtp_user, password)
            server.sendmail(args.smtp_user, [args.email_to], msg.as_string())
    except (smtplib.SMTPException, OSError) as exc:
        print(f"WARN: SMTP send failed ({exc}); overlays preserved on disk.",
              file=sys.stderr)
        for r in results:
            if r.png_path:
                print(f"  {r.png_path}", file=sys.stderr)
        return False

    print(f"Email sent to {args.email_to}", flush=True)
    return True


def _compose_message(
    results: list[_FOVResult],
    args: argparse.Namespace,
) -> tuple[str, str, list[tuple[str, bytes]]]:
    successes = [r for r in results if r.error is None]
    total_rois = sum(
        r.roi_counts.get("accept", 0) + r.roi_counts.get("flag", 0)
        for r in successes
    )
    if len(results) == 1:
        subject_name = results[0].tif.name
    else:
        subject_name = f"batch ({len(results)} FOVs)"
    subject = f"ROIGBIV: {subject_name} — {total_rois} ROIs detected"

    model_name = Path(_resolve_model(args.model)).name
    body_lines: list[str] = []
    for r in results:
        body_lines.append(f"FOV: {r.tif.name}")
        if r.error is not None:
            body_lines.append(f"  FAILED: {r.error}")
            body_lines.append(f"  Duration: {_fmt_duration(r.duration_s)}")
            body_lines.append("")
            continue
        c = r.roi_counts
        body_lines.append(
            f"  ROIs: accept={c.get('accept', 0)}  "
            f"flag={c.get('flag', 0)}  "
            f"reject={c.get('reject', 0)}"
        )
        body_lines.append(f"  Model: {model_name}")
        body_lines.append(f"  Duration: {_fmt_duration(r.duration_s)}")
        body_lines.append(
            f"  Params: fs={args.fs} tau={args.tau} "
            f"diameter={args.diameter} "
            f"cellprob_threshold={args.cellprob_threshold} "
            f"flow_threshold={args.flow_threshold}"
        )
        body_lines.append("  Stage 2 (Suite2p): included")
        if r.png_path is None:
            body_lines.append("  Overlay: MISSING (render failed)")
        else:
            body_lines.append(f"  Overlay file: {r.png_path.name}")
        body_lines.append("")

    body_lines.append("Overlays attached.")
    body = "\n".join(body_lines)

    attachments: list[tuple[str, bytes]] = []
    total_bytes = 0
    for r in results:
        if r.png_path is None or not r.png_path.exists():
            continue
        payload = _read_possibly_downsampled(r.png_path)
        if total_bytes + len(payload) > _MAX_ATTACH_BYTES:
            body += (
                f"\nNote: some overlays were downsampled to fit SMTP size limits."
            )
            payload = _downsample_png_bytes(payload)
        total_bytes += len(payload)
        attachments.append((r.png_path.name, payload))

    return subject, body, attachments


def _read_possibly_downsampled(path: Path) -> bytes:
    data = path.read_bytes()
    if len(data) > _MAX_ATTACH_BYTES // 2:
        return _downsample_png_bytes(data)
    return data


def _downsample_png_bytes(data: bytes) -> bytes:
    from io import BytesIO

    from PIL import Image

    img = Image.open(BytesIO(data))
    img.thumbnail((2048, 2048))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _fmt_duration(seconds: float) -> str:
    s = int(round(seconds))
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


# ── Entry point ──────────────────────────────────────────────────────────────


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)

    try:
        tifs = _iter_inputs(args.input)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    model_name = Path(_resolve_model(args.model)).name

    results: list[_FOVResult]
    if len(tifs) == 1:
        cfg = _build_cfg(args, tifs[0])
        results = [_run_single(tifs[0], cfg, model_name)]
    else:
        jobs = [(tif, _build_cfg(args, tif)) for tif in tifs]
        results = _run_batch(jobs, args.n_workers, model_name)

    successes = [r for r in results if r.error is None]
    failures = [r for r in results if r.error is not None]

    print("\n=== Summary ===", flush=True)
    for r in results:
        if r.error is not None:
            print(f"  {r.tif.name}: FAILED — {r.error}")
        else:
            c = r.roi_counts
            png = r.png_path.name if r.png_path else "<no overlay>"
            print(
                f"  {r.tif.name}: accept={c.get('accept', 0)} "
                f"flag={c.get('flag', 0)} reject={c.get('reject', 0)}  "
                f"[{_fmt_duration(r.duration_s)}]  → {png}"
            )
    print(f"{len(successes)} succeeded, {len(failures)} failed.", flush=True)

    if args.email_to and not args.no_email and successes:
        _send_email(results, args)
    elif args.no_email:
        print("--no-email set; skipping email dispatch.", flush=True)
    elif not args.email_to:
        print("No --email-to provided; skipping email dispatch.", flush=True)

    if not successes:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
