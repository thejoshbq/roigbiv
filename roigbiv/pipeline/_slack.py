"""Slack notification helpers for ``roigbiv-pipeline``.

A sibling of :mod:`roigbiv.pipeline._email`. When a pipeline run finishes,
:func:`send_slack` posts a batch-summary message to a Slack channel and
uploads each FOV's overlay PNG into that message's thread, so researchers
who leave a run going unattended see results directly in Slack — no
personal email account or SMTP bridge required.

Transport is hand-rolled on stdlib :mod:`urllib` (the repo's HTTP norm; see
``roigbiv/io.py``) — no ``slack_sdk`` dependency. File upload uses Slack's
modern three-call flow (``files.getUploadURLExternal`` → binary POST to the
returned single-use URL → ``files.completeUploadExternal``); the legacy
``files.upload`` is deprecated.

The bot token is read from an environment variable at send time (never the
CLI, never yaml, never logged). On any failure the overlay PNGs are left on
disk and :func:`send_slack` returns ``False`` — it never raises, mirroring
:func:`roigbiv.pipeline._email.send_email`.

Required Slack bot scopes: ``chat:write`` and ``files:write``. See
``docs/slack-notifications.md`` for the one-time bot setup.

Public helpers:

- :func:`send_slack` — post the summary + upload all overlays in a thread.
- :func:`compose_slack_message` — build the message text.
- :data:`SLACK_MAX_TEXT` — defensive cap on message length.
"""
from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass

from roigbiv.pipeline._email import (
    EmailFOVResult,
    _read_possibly_downsampled,
    fmt_duration,
)

_SLACK_API = "https://slack.com/api"
SLACK_MAX_TEXT = 38000   # Slack caps message text near 40k chars; leave headroom
_HTTP_TIMEOUT = 30
_MAX_RETRY_SLEEP = 30     # bound on honored Retry-After (s) for a single 429 retry


class SlackError(Exception):
    """Slack call failed (HTTP error, transport, or an ``{ok: false}`` envelope).

    The message is sanitized to a method name + Slack error code — it never
    contains the bot token or the ``Authorization`` header.
    """


@dataclass
class SlackParams:
    """Slack target — what the user supplies on the CLI / UI.

    The token itself is *not* stored here; only the name of the env var that
    holds it, read at send time (mirrors ``EmailParams.smtp_password_env``).
    """

    channel: str       # Slack channel ID (e.g. "C0123ABCD"), not a "#name"
    token_env: str     # env-var NAME holding the xoxb- bot token


def send_slack(
    results: list[EmailFOVResult],
    params: SlackParams,
    pipeline_summary: dict,
) -> bool:
    """Post one summary message + thread the overlay PNG uploads under it.

    Returns True on success, False otherwise. On failure (missing token,
    HTTP/transport error, Slack ``{ok: false}``) the PNGs remain on disk and
    the caller prints a warning and continues. Never raises.

    ``pipeline_summary`` carries the run-level params (fs, tau, model, Cellpose
    knobs) echoed in the message body — the same dict
    :func:`roigbiv.pipeline._email.send_email` consumes.
    """
    token = os.environ.get(params.token_env)
    if not token:
        print(
            f"WARN: env var {params.token_env!r} is unset; "
            f"skipping Slack. Overlay(s) remain at:",
            file=sys.stderr,
        )
        for r in results:
            if r.png_path:
                print(f"  {r.png_path}", file=sys.stderr)
        return False

    text = compose_slack_message(results, params, pipeline_summary)
    try:
        thread_ts = _post_message(token, params.channel, text)
        for r in results:
            if r.png_path is None or not r.png_path.exists():
                continue
            payload = _read_possibly_downsampled(r.png_path)
            _upload_file(token, params.channel, r.png_path.name, payload,
                         thread_ts)
    except (SlackError, urllib.error.URLError, OSError) as exc:
        print(f"WARN: Slack send failed ({exc}); overlays preserved on disk.",
              file=sys.stderr)
        for r in results:
            if r.png_path:
                print(f"  {r.png_path}", file=sys.stderr)
        return False

    print(f"Slack message posted to {params.channel}", flush=True)
    return True


def compose_slack_message(
    results: list[EmailFOVResult],
    params: SlackParams,
    pipeline_summary: dict,
) -> str:
    """Build the channel message text. Mirrors
    :func:`roigbiv.pipeline._email.compose_message`'s body (header ROI total
    counts accept+flag, excludes reject), then notes the thread attachments.
    """
    successes = [r for r in results if r.error is None]
    total_rois = sum(
        (r.roi_counts or {}).get("accept", 0) + (r.roi_counts or {}).get("flag", 0)
        for r in successes
    )
    if len(results) == 1:
        subject_name = results[0].tif.name
    else:
        subject_name = f"batch ({len(results)} FOVs)"
    header = f"ROIGBIV: {subject_name} — {total_rois} ROIs detected"

    model_name = pipeline_summary.get("model_name", "?")
    fs = pipeline_summary.get("fs", "?")
    tau = pipeline_summary.get("tau", "?")
    diameter = pipeline_summary.get("diameter", "?")
    cellprob_threshold = pipeline_summary.get("cellprob_threshold", "?")
    flow_threshold = pipeline_summary.get("flow_threshold", "?")
    stage_flags = pipeline_summary.get("stage_flags", {})

    lines: list[str] = [header, ""]
    for r in results:
        lines.append(f"FOV: {r.tif.name}")
        if r.error is not None:
            lines.append(f"  FAILED: {r.error}")
            lines.append(f"  Duration: {fmt_duration(r.duration_s)}")
            lines.append("")
            continue
        c = r.roi_counts or {}
        lines.append(
            f"  ROIs: accept={c.get('accept', 0)}  "
            f"flag={c.get('flag', 0)}  "
            f"reject={c.get('reject', 0)}"
        )
        lines.append(f"  Model: {model_name}")
        lines.append(f"  Duration: {fmt_duration(r.duration_s)}")
        lines.append(
            f"  Params: fs={fs} tau={tau} "
            f"diameter={diameter} "
            f"cellprob_threshold={cellprob_threshold} "
            f"flow_threshold={flow_threshold}"
        )
        if stage_flags:
            on = ", ".join(f"S{n}" for n in (2, 3, 4) if stage_flags.get(n))
            off = ", ".join(f"S{n}" for n in (2, 3, 4) if not stage_flags.get(n))
            lines.append(f"  Stages: on=[{on or 'none'}]  off=[{off or 'none'}]")
        if r.png_path is None:
            lines.append("  Overlay: MISSING (render failed)")
        lines.append("")

    n_overlays = sum(1 for r in results if r.png_path is not None)
    if n_overlays:
        lines.append(f"{n_overlays} overlay(s) attached in thread.")

    text = "\n".join(lines)
    if len(text) > SLACK_MAX_TEXT:
        text = text[:SLACK_MAX_TEXT - 16] + "\n… (truncated)"
    return text


# ── Slack Web API transport (stdlib urllib) ──────────────────────────────────


def _post_message(token: str, channel: str, text: str) -> str:
    """``chat.postMessage`` → return the new message's ``ts`` (thread anchor)."""
    env = _api_call("chat.postMessage", token,
                    {"channel": channel, "text": text})
    return env["ts"]


def _upload_file(
    token: str,
    channel: str,
    filename: str,
    payload: bytes,
    thread_ts: str,
) -> None:
    """Upload ``payload`` and share it into ``channel`` under ``thread_ts``.

    Three calls: reserve an upload URL, POST the raw bytes to it, then
    finalize + share. ``payload`` is the already-(possibly-)downsampled PNG
    bytes from :func:`roigbiv.pipeline._email._read_possibly_downsampled`.
    """
    reserved = _api_call("files.getUploadURLExternal", token,
                         {"filename": filename, "length": str(len(payload))})
    _post_bytes(reserved["upload_url"], payload)
    _api_call("files.completeUploadExternal", token, {
        "files": json.dumps([{"id": reserved["file_id"], "title": filename}]),
        "channel_id": channel,
        "thread_ts": thread_ts,
    })


def _api_call(method: str, token: str, params: dict) -> dict:
    """POST form-encoded ``params`` to a Slack Web API ``method``.

    Token rides in the ``Authorization`` header (never the URL/body). Raises
    :class:`SlackError` on a non-JSON response or an ``{ok: false}`` envelope.
    """
    data = urllib.parse.urlencode(params).encode("utf-8")
    req = urllib.request.Request(
        f"{_SLACK_API}/{method}",
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
    )
    body = _urlopen_with_retry(req)
    try:
        envelope = json.loads(body)
    except json.JSONDecodeError as exc:
        raise SlackError(f"{method}: non-JSON response from Slack") from exc
    if not envelope.get("ok"):
        raise SlackError(f"{method}: {envelope.get('error', 'unknown_error')}")
    return envelope


def _post_bytes(upload_url: str, payload: bytes) -> None:
    """POST raw bytes to a Slack-provided single-use upload URL (no auth).

    Transport errors are re-raised as :class:`SlackError` without the URL,
    which embeds a single-use upload token in its query string — keeping all
    token-shaped material out of logs.
    """
    req = urllib.request.Request(
        upload_url,
        data=payload,
        method="POST",
        headers={"Content-Type": "application/octet-stream"},
    )
    try:
        _urlopen_with_retry(req)   # Slack returns 200 "OK"; body is ignored
    except (urllib.error.URLError, OSError) as exc:
        reason = getattr(exc, "reason", exc.__class__.__name__)
        raise SlackError(f"file upload POST failed: {reason}") from None


def _urlopen_with_retry(req: urllib.request.Request, *, _retried: bool = False) -> bytes:
    """Open ``req``; on a single HTTP 429 honor ``Retry-After`` once, bounded."""
    try:
        with urllib.request.urlopen(req, timeout=_HTTP_TIMEOUT) as resp:
            return resp.read()
    except urllib.error.HTTPError as exc:
        if exc.code == 429 and not _retried:
            try:
                retry_after = int(exc.headers.get("Retry-After", "1") or "1")
            except (TypeError, ValueError):
                retry_after = 1
            time.sleep(min(max(retry_after, 0), _MAX_RETRY_SLEEP))
            return _urlopen_with_retry(req, _retried=True)
        raise
