"""Tests for the SMTP email-notification path in :mod:`roigbiv.pipeline`.

Covers :func:`roigbiv.pipeline._email.send_email` (auth/STARTTLS wire
sequence, missing-password and SMTP-failure branches) and
:func:`roigbiv.pipeline._email.compose_message` (subject formatting,
partial failures, attachment-cap downsampling). Also pins the
``roigbiv-pipeline`` ``main()`` exit-code contract (0 / 1 / 3).

No sockets are bound: ``smtplib.SMTP`` is replaced with a fake context
manager that records call order and arguments. The pipeline never holds
an idle SMTP connection during compute — the socket is opened fresh
after ``run_pipeline`` returns — so an "idle timeout during long runs"
cannot manifest in the current call sites.
"""
from __future__ import annotations

import email
import smtplib
from pathlib import Path
from typing import Optional

import pytest

from roigbiv.pipeline import _email
from roigbiv.pipeline._email import (
    EmailFOVResult,
    EmailParams,
    compose_message,
    send_email,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _params(
    *,
    email_to: str = "scientist@example.com",
    smtp_user: str = "lab@proton.me",
    smtp_host: str = "127.0.0.1",
    smtp_port: int = 1025,
    smtp_password_env: str = "ROIGBIV_SMTP_PASSWORD",
) -> EmailParams:
    return EmailParams(
        email_to=email_to,
        smtp_user=smtp_user,
        smtp_host=smtp_host,
        smtp_port=smtp_port,
        smtp_password_env=smtp_password_env,
    )


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
    """Write a stub byte payload — ``send_email`` reads bytes verbatim."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * (n_bytes - 8))


class _FakeSMTP:
    """Context-manager stand-in for :class:`smtplib.SMTP`.

    Records every call so tests can assert the wire sequence
    (``starttls`` → ``login`` → ``sendmail``).
    """

    instances: list["_FakeSMTP"] = []

    def __init__(self, host: str, port: int, *, timeout: Optional[float] = None,
                 raise_on: Optional[str] = None) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.calls: list[tuple] = []
        self._raise_on = raise_on
        _FakeSMTP.instances.append(self)

    def __enter__(self) -> "_FakeSMTP":
        return self

    def __exit__(self, *exc_info) -> None:
        return None

    def starttls(self, *, context=None) -> None:
        self.calls.append(("starttls", context))
        if self._raise_on == "starttls":
            raise smtplib.SMTPException("starttls boom")

    def login(self, user: str, password: str) -> None:
        self.calls.append(("login", user, password))
        if self._raise_on == "login":
            raise smtplib.SMTPAuthenticationError(535, b"bad password")

    def sendmail(self, sender: str, recipients: list[str], body: str) -> None:
        self.calls.append(("sendmail", sender, tuple(recipients), body))
        if self._raise_on == "sendmail":
            raise smtplib.SMTPException("sendmail boom")


@pytest.fixture(autouse=True)
def _reset_fake_smtp_registry() -> None:
    _FakeSMTP.instances.clear()


# ── send_email ───────────────────────────────────────────────────────────────


def test_send_email_happy_path_wire_sequence(tmp_path: Path, monkeypatch) -> None:
    """STARTTLS → login → sendmail in that order, with the right credentials."""
    png = tmp_path / "fov_overlay.png"
    _write_tiny_png(png)

    captured: dict = {}

    def _factory(host, port, timeout=None):
        captured["host"] = host
        captured["port"] = port
        captured["timeout"] = timeout
        return _FakeSMTP(host, port, timeout=timeout)

    monkeypatch.setenv("ROIGBIV_SMTP_PASSWORD", "appp asss wddd")
    monkeypatch.setattr(_email.smtplib, "SMTP", _factory)

    params = _params()
    results = [_result(tif=tmp_path / "fov_mc.tif", png_path=png, accept=42)]

    ok = send_email(results, params, _summary())

    assert ok is True
    assert captured == {"host": "127.0.0.1", "port": 1025, "timeout": 30}
    assert len(_FakeSMTP.instances) == 1
    server = _FakeSMTP.instances[0]
    op_names = [c[0] for c in server.calls]
    assert op_names == ["starttls", "login", "sendmail"]
    assert server.calls[1] == ("login", "lab@proton.me", "appp asss wddd")

    _, sender, recipients, raw = server.calls[2]
    assert sender == "lab@proton.me"
    assert recipients == ("scientist@example.com",)

    msg = email.message_from_string(raw)
    assert msg["From"] == "lab@proton.me"
    assert msg["To"] == "scientist@example.com"
    decoded_subject = str(email.header.make_header(email.header.decode_header(msg["Subject"])))
    assert decoded_subject.startswith("ROIGBIV: fov_mc.tif")
    image_parts = [p for p in msg.walk() if p.get_content_type() == "image/png"]
    assert len(image_parts) == 1
    assert image_parts[0].get_filename() == "fov_overlay.png"


def test_send_email_missing_password_skips_connect(tmp_path: Path, monkeypatch, capsys) -> None:
    """Unset env var → no socket attempt, returns False, stderr names env var + PNG path."""
    monkeypatch.delenv("ROIGBIV_SMTP_PASSWORD", raising=False)

    def _explode(*_a, **_kw):  # pragma: no cover - must not be reached
        raise AssertionError("smtplib.SMTP must not be called when password is missing")

    monkeypatch.setattr(_email.smtplib, "SMTP", _explode)

    png = tmp_path / "fov_overlay.png"
    _write_tiny_png(png)
    params = _params()
    results = [_result(tif=tmp_path / "fov_mc.tif", png_path=png, accept=1)]

    ok = send_email(results, params, _summary())

    assert ok is False
    err = capsys.readouterr().err
    assert "ROIGBIV_SMTP_PASSWORD" in err
    assert str(png) in err


def test_send_email_auth_failure_returns_false(tmp_path: Path, monkeypatch, capsys) -> None:
    monkeypatch.setenv("ROIGBIV_SMTP_PASSWORD", "wrong")
    monkeypatch.setattr(
        _email.smtplib, "SMTP",
        lambda host, port, timeout=None: _FakeSMTP(host, port, timeout=timeout, raise_on="login"),
    )

    png = tmp_path / "fov_overlay.png"
    _write_tiny_png(png)
    params = _params()
    results = [_result(tif=tmp_path / "fov_mc.tif", png_path=png, accept=1)]

    ok = send_email(results, params, _summary())

    assert ok is False
    err = capsys.readouterr().err
    assert "SMTP send failed" in err
    assert str(png) in err


def test_send_email_oserror_returns_false(tmp_path: Path, monkeypatch, capsys) -> None:
    """OSError (e.g. socket timeout, DNS failure) is also caught and surfaced."""
    monkeypatch.setenv("ROIGBIV_SMTP_PASSWORD", "x")

    def _factory(host, port, timeout=None):
        raise OSError("simulated network/timeout failure")

    monkeypatch.setattr(_email.smtplib, "SMTP", _factory)

    png = tmp_path / "fov_overlay.png"
    _write_tiny_png(png)
    params = _params()
    results = [_result(tif=tmp_path / "fov_mc.tif", png_path=png, accept=1)]

    ok = send_email(results, params, _summary())

    assert ok is False
    err = capsys.readouterr().err
    assert "SMTP send failed" in err
    assert "simulated network/timeout failure" in err


# ── compose_message ──────────────────────────────────────────────────────────


def test_compose_subject_single_fov(tmp_path: Path) -> None:
    results = [_result(tif=tmp_path / "fovA_mc.tif", accept=10, flag=2)]
    subject, _body, _atts = compose_message(results, _params(), _summary())
    assert subject == "ROIGBIV: fovA_mc.tif — 12 ROIs detected"


def test_compose_subject_batch(tmp_path: Path) -> None:
    results = [
        _result(tif=tmp_path / "a.tif", accept=3, flag=1),
        _result(tif=tmp_path / "b.tif", accept=4, flag=0),
    ]
    subject, _body, _atts = compose_message(results, _params(), _summary())
    assert subject == "ROIGBIV: batch (2 FOVs) — 8 ROIs detected"


def test_compose_subject_excludes_rejects(tmp_path: Path) -> None:
    """Subject ROI count sums accept + flag only — rejected ROIs do not contribute."""
    results = [_result(tif=tmp_path / "a.tif", accept=5, flag=1, reject=99)]
    subject, _body, _atts = compose_message(results, _params(), _summary())
    assert subject == "ROIGBIV: a.tif — 6 ROIs detected"


def test_compose_body_partial_failure(tmp_path: Path) -> None:
    results = [
        _result(tif=tmp_path / "ok.tif", accept=7, flag=2, reject=1),
        _result(tif=tmp_path / "bad.tif", error="RuntimeError: boom", duration_s=42.0),
    ]
    _subject, body, _atts = compose_message(results, _params(), _summary())
    assert "FOV: ok.tif" in body
    assert "accept=7" in body
    assert "FOV: bad.tif" in body
    assert "FAILED: RuntimeError: boom" in body


def test_compose_body_records_active_stage_flags(tmp_path: Path) -> None:
    """Body echoes which stages were on/off so the email is self-describing."""
    summary = _summary()
    summary["stage_flags"] = {2: True, 3: False, 4: True}
    results = [_result(tif=tmp_path / "a.tif", accept=1)]
    _subject, body, _atts = compose_message(results, _params(), summary)
    assert "Stages: on=[S2, S4]" in body
    assert "off=[S3]" in body


def test_compose_attachment_downsampled_when_over_cap(tmp_path: Path, monkeypatch) -> None:
    """When the running attachment total exceeds the cap the next PNG is downsampled.

    We shrink the cap to 1 byte and stub ``_downsample_png_bytes`` to a sentinel
    so the assertion does not depend on Pillow output bytes.
    """
    monkeypatch.setattr(_email, "MAX_ATTACH_BYTES", 1)
    monkeypatch.setattr(_email, "_downsample_png_bytes", lambda _data: b"DOWNSAMPLED")

    png_a = tmp_path / "a_overlay.png"
    png_b = tmp_path / "b_overlay.png"
    _write_tiny_png(png_a, n_bytes=512)
    _write_tiny_png(png_b, n_bytes=512)

    results = [
        _result(tif=tmp_path / "a.tif", png_path=png_a, accept=1),
        _result(tif=tmp_path / "b.tif", png_path=png_b, accept=1),
    ]

    _subject, body, attachments = compose_message(results, _params(), _summary())

    assert [name for name, _payload in attachments] == ["a_overlay.png", "b_overlay.png"]
    assert all(payload == b"DOWNSAMPLED" for _name, payload in attachments)
    assert "downsampled to fit SMTP size limits" in body


# ── main() exit-code propagation ─────────────────────────────────────────────


def _stub_pipeline_for_main(monkeypatch, tmp_path: Path) -> Path:
    """Replace heavy pipeline + workspace machinery so ``main()`` returns
    quickly with one successful FOV. Returns the synthesized PNG path so
    callers can assert it survives email failure."""
    from roigbiv.pipeline import run as _run
    from roigbiv.pipeline.workspace import FOVRunResult

    png = tmp_path / "stub_overlay.png"
    _write_tiny_png(png)

    def _fake_run_with_workspace(workspace, overrides, **kwargs):  # type: ignore[no-untyped-def]
        # Synthesize one successful FOVRunResult.
        return [FOVRunResult(
            tif=workspace.tifs[0],
            output_dir=workspace.output_root / "stub",
            duration_s=0.0,
            fov=object(),  # truthy sentinel; overlay render is monkeypatched
            roi_counts={"accept": 1, "flag": 0, "reject": 0},
        )]

    def _fake_render_overlay(*, fov, output_dir, model_name, fov_stem, outcomes=None):  # type: ignore[no-untyped-def]
        return png

    monkeypatch.setattr(_run, "run_pipeline",
                        lambda *_a, **_kw: object())
    # The workspace path is the simpler one to stub — make any --input path
    # resolve to a directory.
    fake_dir = tmp_path / "ws"
    (fake_dir).mkdir(exist_ok=True)
    (fake_dir / "stub_mc.tif").write_bytes(b"\x49\x49")  # not validated in stub

    import roigbiv.pipeline.workspace as ws_mod
    monkeypatch.setattr(ws_mod, "run_with_workspace", _fake_run_with_workspace)
    # Also stub the import-time alias inside run.py so the call there hits
    # our fake regardless of how it was bound.
    monkeypatch.setattr(
        "roigbiv.pipeline.workspace.run_with_workspace",
        _fake_run_with_workspace,
        raising=False,
    )

    from roigbiv import overlay as _overlay
    monkeypatch.setattr(_overlay, "render_overlay", _fake_render_overlay)

    return png


def test_main_returns_3_when_email_fails(tmp_path: Path, monkeypatch, capsys) -> None:
    _stub_pipeline_for_main(monkeypatch, tmp_path)
    monkeypatch.setenv("ROIGBIV_SMTP_PASSWORD", "x")
    monkeypatch.setattr(
        _email.smtplib, "SMTP",
        lambda host, port, timeout=None: _FakeSMTP(host, port, timeout=timeout, raise_on="login"),
    )

    from roigbiv.pipeline.run import main
    rc = main([
        "--input", str(tmp_path / "ws"),
        "--fs", "7.5",
        "--email-to", "x@example.com",
        "--smtp-user", "y@proton.me",
    ])

    assert rc == 3
    err = capsys.readouterr().err
    assert "Email FAILED" in err


def test_main_returns_0_when_email_succeeds(tmp_path: Path, monkeypatch) -> None:
    _stub_pipeline_for_main(monkeypatch, tmp_path)
    monkeypatch.setenv("ROIGBIV_SMTP_PASSWORD", "x")
    monkeypatch.setattr(
        _email.smtplib, "SMTP",
        lambda host, port, timeout=None: _FakeSMTP(host, port, timeout=timeout),
    )

    from roigbiv.pipeline.run import main
    rc = main([
        "--input", str(tmp_path / "ws"),
        "--fs", "7.5",
        "--email-to", "x@example.com",
        "--smtp-user", "y@proton.me",
    ])

    assert rc == 0


def test_main_returns_0_when_email_not_requested(tmp_path: Path, monkeypatch) -> None:
    _stub_pipeline_for_main(monkeypatch, tmp_path)

    def _explode(*_a, **_kw):  # pragma: no cover - must not be reached
        raise AssertionError("smtplib.SMTP must not be called without --email-to")

    monkeypatch.setattr(_email.smtplib, "SMTP", _explode)

    from roigbiv.pipeline.run import main
    rc = main(["--input", str(tmp_path / "ws"), "--fs", "7.5"])
    assert rc == 0


def test_main_rejects_email_to_without_smtp_user(tmp_path: Path) -> None:
    """argparse error path: --email-to requires --smtp-user (parser.error → exit 2)."""
    from roigbiv.pipeline.run import main
    with pytest.raises(SystemExit):
        main([
            "--input", str(tmp_path),
            "--fs", "7.5",
            "--email-to", "x@example.com",
        ])
