"""SMTP + overlay-PNG email helpers for ``roigbiv-pipeline``.

The pipeline CLI (``roigbiv/pipeline/run.py``) calls into this module to
ship a flat overlay PNG per FOV to a researcher when a run finishes.
SMTP delivery goes through Proton Mail Bridge by default (loopback
STARTTLS at ``127.0.0.1:1025``); see ``docs/email-notifications.md`` for
the one-time Bridge setup. On send failure the overlays are preserved on
disk and the caller exits 3.

Public helpers:

- :func:`send_email` — compose + send one email with all attachments.
- :func:`compose_message` — build subject / body / attachments tuple.
- :func:`build_ssl_context` — SSL context that additively trusts the
  system CA bundle (Conda's Python doesn't read it by default).
- :func:`fmt_duration` — pretty-print a duration in s/m/h.
- :data:`MAX_ATTACH_BYTES` — soft cap on total attachment size.
"""
from __future__ import annotations

import os
import smtplib
import ssl
import sys
from dataclasses import dataclass
from email.mime.image import MIMEImage
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from io import BytesIO
from pathlib import Path
from typing import Optional

MAX_ATTACH_BYTES = 20 * 1024 * 1024   # leave headroom under typical SMTP per-message limits (e.g. Proton's 25 MiB inbound cap)
_SYSTEM_CA_BUNDLE = Path("/etc/ssl/certs/ca-certificates.crt")


@dataclass
class EmailParams:
    """SMTP config + recipient — all strings the user supplies on the CLI."""

    email_to: str
    smtp_host: str
    smtp_port: int
    smtp_user: str
    smtp_password_env: str


@dataclass
class EmailFOVResult:
    """Per-FOV outcome for email rendering. Mirrors the subset of
    :class:`roigbiv.pipeline.workspace.FOVRunResult` that
    :func:`compose_message` actually reads."""

    tif: Path
    output_dir: Path
    duration_s: float
    error: Optional[str] = None
    png_path: Optional[Path] = None
    roi_counts: dict = None  # type: ignore[assignment]


def build_ssl_context() -> ssl.SSLContext:
    """SSL context that trusts certifi *and* a locally-administered CA bundle.

    Conda's Python (which ships its own ``cert.pem``) doesn't see the
    system trust store, so a Proton Bridge cert added via
    ``update-ca-certificates`` is invisible to ``smtplib`` unless we
    additively load the system bundle here. ``ROIGBIV_SMTP_CA_FILE``
    overrides the path for non-Debian distros.
    """
    ctx = ssl.create_default_context()
    override = os.environ.get("ROIGBIV_SMTP_CA_FILE")
    extra = Path(override) if override else _SYSTEM_CA_BUNDLE
    if extra.exists():
        ctx.load_verify_locations(cafile=str(extra))
    return ctx


def send_email(
    results: list[EmailFOVResult],
    params: EmailParams,
    pipeline_summary: dict,
) -> bool:
    """Send a single email with all overlay attachments.

    Returns True on success, False otherwise. On failure, PNGs remain on
    disk — the caller prints a warning and continues.

    ``pipeline_summary`` carries the run-level params (fs, tau, model,
    Cellpose knobs) that get echoed in the email body.
    """
    password = os.environ.get(params.smtp_password_env)
    if not password:
        print(
            f"WARN: env var {params.smtp_password_env!r} is unset; "
            f"skipping email. Overlay(s) remain at:",
            file=sys.stderr,
        )
        for r in results:
            if r.png_path:
                print(f"  {r.png_path}", file=sys.stderr)
        return False

    subject, body, attachments = compose_message(results, params, pipeline_summary)
    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = params.smtp_user
    msg["To"] = params.email_to
    msg.attach(MIMEText(body, "plain"))
    for name, payload in attachments:
        part = MIMEImage(payload, _subtype="png")
        part.add_header("Content-Disposition", "attachment", filename=name)
        msg.attach(part)

    try:
        with smtplib.SMTP(params.smtp_host, params.smtp_port, timeout=30) as server:
            server.starttls(context=build_ssl_context())
            server.login(params.smtp_user, password)
            server.sendmail(params.smtp_user, [params.email_to], msg.as_string())
    except (smtplib.SMTPException, OSError, ssl.SSLError) as exc:
        print(f"WARN: SMTP send failed ({exc}); overlays preserved on disk.",
              file=sys.stderr)
        for r in results:
            if r.png_path:
                print(f"  {r.png_path}", file=sys.stderr)
        return False

    print(f"Email sent to {params.email_to}", flush=True)
    return True


def compose_message(
    results: list[EmailFOVResult],
    params: EmailParams,
    pipeline_summary: dict,
) -> tuple[str, str, list[tuple[str, bytes]]]:
    successes = [r for r in results if r.error is None]
    total_rois = sum(
        (r.roi_counts or {}).get("accept", 0) + (r.roi_counts or {}).get("flag", 0)
        for r in successes
    )
    if len(results) == 1:
        subject_name = results[0].tif.name
    else:
        subject_name = f"batch ({len(results)} FOVs)"
    subject = f"ROIGBIV: {subject_name} — {total_rois} ROIs detected"

    model_name = pipeline_summary.get("model_name", "?")
    fs = pipeline_summary.get("fs", "?")
    tau = pipeline_summary.get("tau", "?")
    diameter = pipeline_summary.get("diameter", "?")
    cellprob_threshold = pipeline_summary.get("cellprob_threshold", "?")
    flow_threshold = pipeline_summary.get("flow_threshold", "?")
    stage_flags = pipeline_summary.get("stage_flags", {})

    body_lines: list[str] = []
    for r in results:
        body_lines.append(f"FOV: {r.tif.name}")
        if r.error is not None:
            body_lines.append(f"  FAILED: {r.error}")
            body_lines.append(f"  Duration: {fmt_duration(r.duration_s)}")
            body_lines.append("")
            continue
        c = r.roi_counts or {}
        body_lines.append(
            f"  ROIs: accept={c.get('accept', 0)}  "
            f"flag={c.get('flag', 0)}  "
            f"reject={c.get('reject', 0)}"
        )
        body_lines.append(f"  Model: {model_name}")
        body_lines.append(f"  Duration: {fmt_duration(r.duration_s)}")
        body_lines.append(
            f"  Params: fs={fs} tau={tau} "
            f"diameter={diameter} "
            f"cellprob_threshold={cellprob_threshold} "
            f"flow_threshold={flow_threshold}"
        )
        if stage_flags:
            on = ", ".join(f"S{n}" for n in (2, 3, 4) if stage_flags.get(n))
            off = ", ".join(f"S{n}" for n in (2, 3, 4) if not stage_flags.get(n))
            body_lines.append(f"  Stages: on=[{on or 'none'}]  off=[{off or 'none'}]")
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
        if total_bytes + len(payload) > MAX_ATTACH_BYTES:
            body += (
                "\nNote: some overlays were downsampled to fit SMTP size limits."
            )
            payload = _downsample_png_bytes(payload)
        total_bytes += len(payload)
        attachments.append((r.png_path.name, payload))

    return subject, body, attachments


def _read_possibly_downsampled(path: Path) -> bytes:
    data = path.read_bytes()
    if len(data) > MAX_ATTACH_BYTES // 2:
        return _downsample_png_bytes(data)
    return data


def _downsample_png_bytes(data: bytes) -> bytes:
    from PIL import Image

    img = Image.open(BytesIO(data))
    img.thumbnail((2048, 2048))
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def fmt_duration(seconds: float) -> str:
    s = int(round(seconds))
    m, s = divmod(s, 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"
