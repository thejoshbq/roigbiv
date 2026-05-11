"""Manual smoke test for the ROIGBIV email-notification path.

Exercises the same ``roigbiv.pipeline._email.send_email`` machinery a real
pipeline run would use, but with a synthesized 1×1 PNG instead of an
overlay render. Use it to confirm Proton Mail Bridge auth + STARTTLS work
on a new machine before relying on the full pipeline to deliver results.

Not part of CI; not imported anywhere else.

Example
-------
::

    export ROIGBIV_SMTP_PASSWORD='<bridge mailbox-details password>'
    python scripts/verify_email_smoke.py \\
        --email-to me@example.com \\
        --smtp-user me@proton.me

The script prints ``True`` on success (and a real email lands in the
recipient inbox) or ``False`` on failure (with the same stderr surfacing
the full pipeline produces).
"""
from __future__ import annotations

import argparse
import struct
import sys
import zlib
from pathlib import Path
from tempfile import TemporaryDirectory

from roigbiv.pipeline._email import EmailFOVResult, EmailParams, send_email


def _make_1px_png(path: Path) -> None:
    """Write a valid 1×1 black PNG (~70 bytes) without depending on Pillow."""
    sig = b"\x89PNG\r\n\x1a\n"

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", 1, 1, 8, 0, 0, 0, 0)
    raw = b"\x00\x00"   # filter byte + 1 pixel (grayscale, value 0)
    idat = zlib.compress(raw)
    path.write_bytes(sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b""))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="verify_email_smoke",
        description=(
            "Send a single, tiny test email through "
            "roigbiv.pipeline._email.send_email to verify SMTP auth/TLS "
            "without running the pipeline."
        ),
    )
    parser.add_argument("--email-to", required=True, type=str)
    parser.add_argument("--smtp-host", default="127.0.0.1")
    parser.add_argument("--smtp-port", type=int, default=1025)
    parser.add_argument("--smtp-user", required=True, type=str)
    parser.add_argument("--smtp-password-env", default="ROIGBIV_SMTP_PASSWORD")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    summary = {
        "model_name": "models/deployed/current_model",
        "fs": 7.5,
        "tau": 1.0,
        "diameter": 12,
        "cellprob_threshold": 0.0,
        "flow_threshold": 0.6,
        "stage_flags": {2: True, 3: True, 4: True},
    }
    params = EmailParams(
        email_to=args.email_to,
        smtp_host=args.smtp_host,
        smtp_port=args.smtp_port,
        smtp_user=args.smtp_user,
        smtp_password_env=args.smtp_password_env,
    )

    with TemporaryDirectory(prefix="roigbiv_smoke_") as tmp:
        tmp_path = Path(tmp)
        png = tmp_path / "smoke_overlay.png"
        _make_1px_png(png)
        result = EmailFOVResult(
            tif=tmp_path / "smoke_mc.tif",
            output_dir=tmp_path,
            png_path=png,
            duration_s=0.0,
            roi_counts={"accept": 0, "flag": 0, "reject": 0},
        )
        ok = send_email([result], params, summary)
        print(ok)
        return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
