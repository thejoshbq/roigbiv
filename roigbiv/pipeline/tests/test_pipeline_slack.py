"""Tests for the Slack notification path in :mod:`roigbiv.pipeline`.

Covers :func:`roigbiv.pipeline._slack.send_slack` (the three-call upload wire
sequence, missing-token / transport-failure / ``{ok:false}`` branches, the
text-only and multi-FOV cases, and a single 429 retry) and
:func:`roigbiv.pipeline._slack.compose_slack_message`. Also pins the
``roigbiv-pipeline`` ``main()`` exit-code contract for Slack (4) including
precedence against email (3).

No sockets are bound: ``urllib.request.urlopen`` is replaced with a fake that
records call order + decoded form fields and returns canned Slack envelopes.
A redaction test asserts the bot token never reaches stderr.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.parse
from pathlib import Path
from typing import Optional

import pytest

from roigbiv.pipeline import _slack
from roigbiv.pipeline._email import EmailFOVResult
from roigbiv.pipeline._slack import (
    SlackParams,
    compose_slack_message,
    send_slack,
)
# Reuse the email suite's main()-stub + fake SMTP to drive exit-code tests.
from roigbiv.pipeline.tests.test_pipeline_email import (
    _FakeSMTP,
    _stub_pipeline_for_main,
)

_TOKEN = "xoxb-super-secret-bot-token"


# ── Helpers ──────────────────────────────────────────────────────────────────


def _params(*, channel: str = "C0123ABCD",
            token_env: str = "ROIGBIV_SLACK_TOKEN") -> SlackParams:
    return SlackParams(channel=channel, token_env=token_env)


def _summary() -> dict:
    return {
        "model_name": "current_model",
        "fs": 7.5,
        "tau": 1.0,
        "diameter": 12,
        "cellprob_threshold": 0.0,
        "flow_threshold": 0.6,
        "stage_flags": {2: True, 3: True, 4: True},
    }


def _result(
    *,
    tif: Path,
    png_path: Optional[Path] = None,
    error: Optional[str] = None,
    accept: int = 0,
    flag: int = 0,
    reject: int = 0,
    duration_s: float = 1.0,
) -> EmailFOVResult:
    return EmailFOVResult(
        tif=tif,
        output_dir=tif.parent,
        png_path=png_path,
        duration_s=duration_s,
        error=error,
        roi_counts={"accept": accept, "flag": flag, "reject": reject},
    )


def _write_tiny_png(path: Path, n_bytes: int = 256) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * (n_bytes - 8))


def _endpoint(url: str) -> str:
    for name in ("chat.postMessage", "files.getUploadURLExternal",
                 "files.completeUploadExternal"):
        if name in url:
            return name
    return "upload"   # the Slack-provided single-use upload URL


class _Resp:
    """Minimal context-manager response wrapping canned bytes."""

    def __init__(self, body: bytes) -> None:
        self._body = body

    def __enter__(self) -> "_Resp":
        return self

    def __exit__(self, *exc_info) -> None:
        return None

    def read(self) -> bytes:
        return self._body


class _FakeTransport:
    """Stand-in for ``urllib.request.urlopen``.

    Records ``(endpoint, fields, auth)`` per call and returns the canned
    success envelope for each Slack method. Configurable failure injection:

    - ``fail_on``: endpoint name → return ``{"ok": false, "error": ...}``.
    - ``raise_on``: endpoint name → raise :class:`urllib.error.URLError`.
    - ``http_429_on``: endpoint name → raise one HTTP 429 (then succeed).
    """

    def __init__(self, *, fail_on: Optional[str] = None,
                 raise_on: Optional[str] = None,
                 http_429_on: Optional[str] = None) -> None:
        self.calls: list[tuple] = []
        self._fail_on = fail_on
        self._raise_on = raise_on
        self._http_429_on = http_429_on
        self._429_fired: set[str] = set()

    def __call__(self, req, timeout=None):  # type: ignore[no-untyped-def]
        url = req.full_url
        ep = _endpoint(url)
        raw = req.data or b""
        if ep == "upload":
            fields = {"_len": len(raw)}
        else:
            fields = {k: v[0] for k, v in
                      urllib.parse.parse_qs(raw.decode()).items()}
        auth = req.get_header("Authorization")
        self.calls.append((ep, fields, auth))

        if self._http_429_on == ep and ep not in self._429_fired:
            self._429_fired.add(ep)
            raise urllib.error.HTTPError(
                url, 429, "Too Many Requests", {"Retry-After": "0"}, None)
        if self._raise_on == ep:
            raise urllib.error.URLError("simulated transport failure")
        if self._fail_on == ep:
            return _Resp(json.dumps({"ok": False, "error": "boom"}).encode())

        if ep == "chat.postMessage":
            return _Resp(json.dumps({"ok": True, "ts": "111.222"}).encode())
        if ep == "files.getUploadURLExternal":
            return _Resp(json.dumps(
                {"ok": True,
                 "upload_url": "https://files.slack.com/u/F1",
                 "file_id": "F1"}).encode())
        if ep == "files.completeUploadExternal":
            return _Resp(json.dumps(
                {"ok": True, "files": [{"id": "F1"}]}).encode())
        return _Resp(b"OK")   # binary PUT/POST target; body ignored

    def endpoints(self) -> list[str]:
        return [c[0] for c in self.calls]


def _install(monkeypatch, transport: _FakeTransport) -> _FakeTransport:
    monkeypatch.setattr(_slack.urllib.request, "urlopen", transport)
    return transport


# ── send_slack ───────────────────────────────────────────────────────────────


def test_send_slack_happy_path_call_sequence(tmp_path, monkeypatch) -> None:
    """postMessage → getUploadURLExternal → PUT bytes → completeUploadExternal,
    threaded under the post's ts, bearer token present, returns True."""
    png = tmp_path / "fov_overlay.png"
    _write_tiny_png(png)
    monkeypatch.setenv("ROIGBIV_SLACK_TOKEN", _TOKEN)
    t = _install(monkeypatch, _FakeTransport())

    results = [_result(tif=tmp_path / "fov_mc.tif", png_path=png, accept=42)]
    ok = send_slack(results, _params(), _summary())

    assert ok is True
    assert t.endpoints() == [
        "chat.postMessage",
        "files.getUploadURLExternal",
        "upload",
        "files.completeUploadExternal",
    ]
    # Bearer token rides in the header on every authenticated Slack call.
    for ep, _fields, auth in t.calls:
        if ep != "upload":
            assert auth == f"Bearer {_TOKEN}"
    post_fields = t.calls[0][1]
    assert post_fields["channel"] == "C0123ABCD"
    # Upload finalized + shared into the same channel, threaded under the ts.
    complete_fields = t.calls[3][1]
    assert complete_fields["channel_id"] == "C0123ABCD"
    assert complete_fields["thread_ts"] == "111.222"
    assert json.loads(complete_fields["files"]) == [{"id": "F1",
                                                     "title": "fov_overlay.png"}]


def test_send_slack_missing_token_skips(tmp_path, monkeypatch, capsys) -> None:
    """Unset env var → zero HTTP calls, returns False, stderr names var + PNG."""
    monkeypatch.delenv("ROIGBIV_SLACK_TOKEN", raising=False)

    def _explode(*_a, **_kw):  # pragma: no cover - must not be reached
        raise AssertionError("urlopen must not be called without a token")

    monkeypatch.setattr(_slack.urllib.request, "urlopen", _explode)

    png = tmp_path / "fov_overlay.png"
    _write_tiny_png(png)
    results = [_result(tif=tmp_path / "fov_mc.tif", png_path=png, accept=1)]

    ok = send_slack(results, _params(), _summary())

    assert ok is False
    err = capsys.readouterr().err
    assert "ROIGBIV_SLACK_TOKEN" in err
    assert str(png) in err


def test_send_slack_upload_failure_preserves_png(tmp_path, monkeypatch, capsys) -> None:
    """An {ok:false} envelope on completion → False, PNG preserved on disk."""
    png = tmp_path / "fov_overlay.png"
    _write_tiny_png(png)
    monkeypatch.setenv("ROIGBIV_SLACK_TOKEN", _TOKEN)
    _install(monkeypatch, _FakeTransport(fail_on="files.completeUploadExternal"))

    results = [_result(tif=tmp_path / "fov_mc.tif", png_path=png, accept=1)]
    ok = send_slack(results, _params(), _summary())

    assert ok is False
    assert png.exists()
    err = capsys.readouterr().err
    assert "Slack send failed" in err
    assert str(png) in err


def test_send_slack_transport_error_returns_false(tmp_path, monkeypatch, capsys) -> None:
    """A URLError (DNS/socket) is caught and surfaced, returns False."""
    png = tmp_path / "fov_overlay.png"
    _write_tiny_png(png)
    monkeypatch.setenv("ROIGBIV_SLACK_TOKEN", _TOKEN)
    _install(monkeypatch, _FakeTransport(raise_on="chat.postMessage"))

    results = [_result(tif=tmp_path / "fov_mc.tif", png_path=png, accept=1)]
    ok = send_slack(results, _params(), _summary())

    assert ok is False
    err = capsys.readouterr().err
    assert "Slack send failed" in err
    assert "simulated transport failure" in err


def test_send_slack_png_none_posts_text_only(tmp_path, monkeypatch) -> None:
    """No overlay → postMessage only, zero upload calls, returns True."""
    monkeypatch.setenv("ROIGBIV_SLACK_TOKEN", _TOKEN)
    t = _install(monkeypatch, _FakeTransport())

    results = [_result(tif=tmp_path / "fov_mc.tif", png_path=None, accept=3)]
    ok = send_slack(results, _params(), _summary())

    assert ok is True
    assert t.endpoints() == ["chat.postMessage"]


def test_send_slack_multi_fov_batch(tmp_path, monkeypatch) -> None:
    """N FOVs → one postMessage + one upload per FOV-with-overlay, threaded.
    A failed FOV (no PNG) contributes no upload."""
    png_a = tmp_path / "a_overlay.png"
    png_b = tmp_path / "b_overlay.png"
    _write_tiny_png(png_a)
    _write_tiny_png(png_b)
    monkeypatch.setenv("ROIGBIV_SLACK_TOKEN", _TOKEN)
    t = _install(monkeypatch, _FakeTransport())

    results = [
        _result(tif=tmp_path / "a.tif", png_path=png_a, accept=1),
        _result(tif=tmp_path / "b.tif", png_path=png_b, accept=2),
        _result(tif=tmp_path / "bad.tif", error="RuntimeError: boom"),
    ]
    ok = send_slack(results, _params(), _summary())

    assert ok is True
    eps = t.endpoints()
    assert eps.count("chat.postMessage") == 1
    assert eps.count("files.completeUploadExternal") == 2   # only the two with PNGs
    # All uploads share the single thread ts from the one message.
    for ep, fields, _auth in t.calls:
        if ep == "files.completeUploadExternal":
            assert fields["thread_ts"] == "111.222"


def test_send_slack_429_retry(tmp_path, monkeypatch) -> None:
    """A single 429 on getUploadURLExternal is retried once, then succeeds."""
    png = tmp_path / "fov_overlay.png"
    _write_tiny_png(png)
    monkeypatch.setenv("ROIGBIV_SLACK_TOKEN", _TOKEN)
    t = _install(monkeypatch,
                 _FakeTransport(http_429_on="files.getUploadURLExternal"))

    results = [_result(tif=tmp_path / "fov_mc.tif", png_path=png, accept=1)]
    ok = send_slack(results, _params(), _summary())

    assert ok is True
    # getUploadURLExternal appears twice: the 429 attempt + the retry.
    assert t.endpoints().count("files.getUploadURLExternal") == 2


def test_send_slack_token_never_logged(tmp_path, monkeypatch, capsys) -> None:
    """On failure the stderr warning must never echo the bot token."""
    png = tmp_path / "fov_overlay.png"
    _write_tiny_png(png)
    monkeypatch.setenv("ROIGBIV_SLACK_TOKEN", _TOKEN)
    _install(monkeypatch, _FakeTransport(fail_on="chat.postMessage"))

    results = [_result(tif=tmp_path / "fov_mc.tif", png_path=png, accept=1)]
    send_slack(results, _params(), _summary())

    captured = capsys.readouterr()
    assert _TOKEN not in captured.err
    assert _TOKEN not in captured.out


# ── compose_slack_message ────────────────────────────────────────────────────


def test_compose_header_excludes_rejects(tmp_path) -> None:
    """Header ROI total sums accept + flag only — rejects don't contribute."""
    results = [_result(tif=tmp_path / "a_mc.tif", accept=5, flag=1, reject=99)]
    text = compose_slack_message(results, _params(), _summary())
    assert text.splitlines()[0] == "ROIGBIV: a_mc.tif — 6 ROIs detected"


def test_compose_header_batch(tmp_path) -> None:
    results = [
        _result(tif=tmp_path / "a.tif", accept=3, flag=1),
        _result(tif=tmp_path / "b.tif", accept=4, flag=0),
    ]
    text = compose_slack_message(results, _params(), _summary())
    assert text.splitlines()[0] == "ROIGBIV: batch (2 FOVs) — 8 ROIs detected"


def test_compose_partial_failure(tmp_path) -> None:
    results = [
        _result(tif=tmp_path / "ok.tif", accept=7, flag=2, reject=1),
        _result(tif=tmp_path / "bad.tif", error="RuntimeError: boom",
                duration_s=42.0),
    ]
    text = compose_slack_message(results, _params(), _summary())
    assert "FOV: ok.tif" in text
    assert "accept=7" in text
    assert "FAILED: RuntimeError: boom" in text


def test_compose_records_active_stage_flags(tmp_path) -> None:
    summary = _summary()
    summary["stage_flags"] = {2: True, 3: False, 4: True}
    results = [_result(tif=tmp_path / "a.tif", accept=1, png_path=None)]
    text = compose_slack_message(results, _params(), summary)
    assert "Stages: on=[S2, S4]" in text
    assert "off=[S3]" in text


# ── main() exit-code propagation ─────────────────────────────────────────────


def test_main_returns_0_when_slack_succeeds(tmp_path, monkeypatch) -> None:
    _stub_pipeline_for_main(monkeypatch, tmp_path)
    monkeypatch.setenv("ROIGBIV_SLACK_TOKEN", _TOKEN)
    _install(monkeypatch, _FakeTransport())

    from roigbiv.pipeline.run import main
    rc = main([
        "--input", str(tmp_path / "ws"),
        "--fs", "7.5",
        "--slack-channel", "C0123ABCD",
    ])
    assert rc == 0


def test_main_returns_4_when_slack_fails(tmp_path, monkeypatch, capsys) -> None:
    _stub_pipeline_for_main(monkeypatch, tmp_path)
    monkeypatch.setenv("ROIGBIV_SLACK_TOKEN", _TOKEN)
    _install(monkeypatch, _FakeTransport(raise_on="chat.postMessage"))

    from roigbiv.pipeline.run import main
    rc = main([
        "--input", str(tmp_path / "ws"),
        "--fs", "7.5",
        "--slack-channel", "C0123ABCD",
    ])
    assert rc == 4
    assert "Slack FAILED" in capsys.readouterr().err


def test_main_email_ok_slack_fails_returns_4(tmp_path, monkeypatch) -> None:
    """Email succeeds, Slack fails → exit 4."""
    _stub_pipeline_for_main(monkeypatch, tmp_path)
    monkeypatch.setenv("ROIGBIV_SMTP_PASSWORD", "x")
    monkeypatch.setenv("ROIGBIV_SLACK_TOKEN", _TOKEN)
    import roigbiv.pipeline._email as _email
    monkeypatch.setattr(
        _email.smtplib, "SMTP",
        lambda host, port, timeout=None: _FakeSMTP(host, port, timeout=timeout),
    )
    _install(monkeypatch, _FakeTransport(raise_on="chat.postMessage"))

    from roigbiv.pipeline.run import main
    rc = main([
        "--input", str(tmp_path / "ws"),
        "--fs", "7.5",
        "--email-to", "x@example.com", "--smtp-user", "y@proton.me",
        "--slack-channel", "C0123ABCD",
    ])
    assert rc == 4


def test_main_both_fail_email_precedence_returns_3(tmp_path, monkeypatch) -> None:
    """Email fails AND Slack fails → exit 3 (email takes precedence)."""
    _stub_pipeline_for_main(monkeypatch, tmp_path)
    monkeypatch.setenv("ROIGBIV_SMTP_PASSWORD", "x")
    monkeypatch.setenv("ROIGBIV_SLACK_TOKEN", _TOKEN)
    import roigbiv.pipeline._email as _email
    monkeypatch.setattr(
        _email.smtplib, "SMTP",
        lambda host, port, timeout=None: _FakeSMTP(host, port, timeout=timeout,
                                                   raise_on="login"),
    )
    _install(monkeypatch, _FakeTransport(raise_on="chat.postMessage"))

    from roigbiv.pipeline.run import main
    rc = main([
        "--input", str(tmp_path / "ws"),
        "--fs", "7.5",
        "--email-to", "x@example.com", "--smtp-user", "y@proton.me",
        "--slack-channel", "C0123ABCD",
    ])
    assert rc == 3
