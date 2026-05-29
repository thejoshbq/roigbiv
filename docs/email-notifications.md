# Email notifications on pipeline completion

`roigbiv-pipeline` sends a single email with the per-FOV PNG overlay
attached when the run finishes — pass `--email-to <addr> --smtp-user
<bridge-mailbox>` and the email goes out automatically. Use it for
remote runs where you'd rather not poll the host. (`roigbiv-ui` does
not send email.)

Email goes through **Proton Mail Bridge** running on the same host —
Bridge exposes a loopback STARTTLS SMTP relay on `127.0.0.1:1025`, and
that's what the CLI's `--smtp-host` / `--smtp-port` defaults point at.
No other SMTP providers are supported; pick the GUI or headless setup
below and run the one-time configuration once per lab box.

## CLI flags

| Flag | Default | Notes |
| --- | --- | --- |
| `--email-to` | — | Omit to skip email entirely. |
| `--smtp-host` | `127.0.0.1` | Local Proton Mail Bridge SMTP relay. |
| `--smtp-port` | `1025` | Bridge's STARTTLS port. |
| `--smtp-user` | — | Required when `--email-to` is set; your full Proton address (the Bridge mailbox login). |
| `--smtp-password-env` | `ROIGBIV_SMTP_PASSWORD` | Env-var name holding the Bridge mailbox password. **Never** pass the password on the command line. |
| `--no-email` | off | Skip dispatch even if `--email-to` is set. |
| `--overlay-outcomes` | `accept,flag,reject` | Comma-separated subset of `accept,flag,reject`. Restricts which gate outcomes are drawn on the overlay PNG. Default is all three — every detected ROI is shown so gate-discard issues surface in the email instead of requiring napari. Pass e.g. `--overlay-outcomes accept,flag` to hide rejects. |

> **Default change (May 2026).** Previous releases silently skipped `reject` ROIs in the overlay PNG; now they're drawn in red so mis-rejections are visible at a glance. The annotation block always reports all three totals regardless of the filter.

## Desktop / GUI Proton Bridge setup

Use this path when the lab box has a desktop session (the Pop!_OS
workstation case). Bridge runs as a tray app and exposes the loopback
STARTTLS SMTP relay on `127.0.0.1:1025` that the CLI's defaults
already target — only `--smtp-user` needs to be passed at run time.

If you only have SSH (no display), skip to the headless section below.

### Prerequisites

- A paid Proton Mail plan (Mail Plus or higher; Bridge isn't on the
  free tier).
- The `protonmail-bridge` `.deb` (or equivalent) installed from
  <https://proton.me/mail/bridge>. The package ships
  `/usr/bin/protonmail-bridge`, an Activities launcher, and a desktop
  autostart entry.
- A working system tray. GNOME 40+ needs an AppIndicator extension to
  show the Bridge tray icon (`gnome-shell-extension-appindicator` on
  stock GNOME; built into Pop!_OS Cosmic). Without it, Bridge still
  runs and SMTP still works — the tray UI is just hidden.

### Step 1 — Launch Bridge

Open *Proton Mail Bridge* from the Activities overview (or run
`protonmail-bridge` from a terminal). The main window opens with an
empty account list and a tray icon appears.

### Step 2 — Sign in

Click **Add account** in the main window. Enter your Proton email,
password, and 2FA code at the prompts. On success, the account row
shows a green **Connected** indicator.

### Step 3 — Read the SMTP credentials

Click the account row and open **Mailbox details** (gear / kebab menu
— exact label varies by Bridge version). The panel lists:

- **SMTP server**: `127.0.0.1`
- **SMTP port**: `1025` (STARTTLS). Bridge also serves IMAP on `1143`;
  we don't use IMAP.
- **Username**: your full Proton address (e.g.
  `thejoshbq.ext@proton.me`).
- **Password**: a 16-character Bridge-generated password, auto-set on
  first login. Click the eye icon to reveal, then the copy icon.

> This is a **Bridge-generated password**, not your Proton account
> password. It changes whenever you click *Change password* in the
> GUI — re-copy it into `ROIGBIV_SMTP_PASSWORD` if you ever rotate it.

### Step 4 — Trust Bridge's TLS certificate

Bridge serves a self-signed cert on the loopback. Export it from the
GUI and add it to the system trust store:

1. **Settings → Advanced settings → Export TLS Certificates** (label
   varies slightly by version; on some 3.x builds it's under
   **Settings → Connection mode**). Save into a directory you control,
   e.g. `~/proton-bridge-certs/`.
2. Bridge writes `cert.pem` (server cert) and `key.pem` (private key —
   leave it alone). We only need `cert.pem`.
3. Install and refresh the system trust store:

   ```bash
   sudo cp ~/proton-bridge-certs/cert.pem \
       /usr/local/share/ca-certificates/proton-bridge.crt
   sudo update-ca-certificates
   # expect: "1 added, 0 removed; done."
   ```

   Note the rename to `.crt` — `update-ca-certificates` only picks up
   `.crt` files in `/usr/local/share/ca-certificates/`.

4. Verify trust from the loopback:

   ```bash
   echo | openssl s_client -connect 127.0.0.1:1025 -starttls smtp 2>&1 \
       | grep "Verify return code"
   # expect: Verify return code: 0 (ok)
   ```

> **Conda Python doesn't read the system trust store.** Same caveat as
> the headless section below: `roigbiv-pipeline` runs in the `roigbiv`
> conda env, whose Python ships its own `cert.pem` (public CAs only).
> `roigbiv.pipeline._email.send_email` compensates by additively
> loading `/etc/ssl/certs/ca-certificates.crt` on top of certifi, so
> the step above is sufficient on Debian/Ubuntu/Pop_OS hosts. On
> Fedora/RHEL hosts, point the CLI at the right bundle via:
>
> ```bash
> export ROIGBIV_SMTP_CA_FILE=/etc/pki/tls/certs/ca-bundle.crt
> ```

### Step 5 — Persist Bridge across reboots

In the Bridge window, open **Settings** and toggle **Open on startup**
(sometimes labelled **Launch Bridge at system startup**). This drops a
`~/.config/autostart/protonmail-bridge.desktop` entry that the desktop
session honors at graphical login.

Verify after the next reboot:

```bash
ss -tlnp | grep 1025
# expect: a LISTEN row on 127.0.0.1:1025 owned by protonmail-bridge.
```

The tray icon should reappear within a few seconds of unlocking the
session.

> **Autostart fires only on graphical login.** If you also need Bridge
> running for cron jobs that fire while no one is logged in, use the
> user-level systemd unit from the headless section below (with
> `loginctl enable-linger`).

### Step 6 — Export the password and run the smoke test

```bash
export ROIGBIV_SMTP_PASSWORD='<the password from Step 3>'
# Optional: append to ~/.bashrc

conda activate roigbiv
python scripts/verify_email_smoke.py \
    --email-to joshuaboquiren@pm.me \
    --smtp-user thejoshbq.ext@proton.me
# expect: "Email sent to joshuaboquiren@pm.me" then "True"; email
#         lands in the recipient inbox within seconds (Proton → Proton
#         is near-instant).
```

`--smtp-host` / `--smtp-port` are omitted because the CLI defaults
already point at Bridge (`127.0.0.1:1025`).

### Step 7 — Optional convenience alias

If you don't want to type your Proton address every run, alias it.
Append to `~/.bashrc`:

```bash
alias roigbiv-mail='roigbiv-pipeline --smtp-user thejoshbq.ext@proton.me'
```

Real runs become:

```bash
roigbiv-mail --input <path> --fs 7.5 \
    --email-to joshuaboquiren@pm.me
```

(Workspace mode is implicit when `<path>` is a directory; no
`--workspace` flag.)

> Aliases don't expand inside cron jobs, systemd units, or
> `bash -c '…'` invocations. For non-interactive contexts, pass
> `--smtp-user thejoshbq.ext@proton.me` explicitly or use a wrapper
> shell script.

### Failure modes specific to the GUI path

- **Tray icon missing.** GNOME without an AppIndicator extension hides
  the tray. Bridge is still running and SMTP still works; only the UI
  is unreachable. Install the extension and restart the shell, or
  relaunch `protonmail-bridge` to re-show the main window.
- **`[X]` closes the window but the app keeps running.** Expected —
  the close button minimizes to tray. Use **Quit** from the tray menu
  to actually stop Bridge.
- **`Connection refused` on `127.0.0.1:1025` after reboot.** Bridge
  didn't autostart. Confirm the **Open on startup** toggle is on; if
  yes, check that `~/.config/autostart/protonmail-bridge.desktop`
  exists and is readable.
- **`SSLCertVerificationError` after a Bridge upgrade.** Bridge
  rotated its self-signed cert. Re-export via *Settings → Advanced
  settings → Export TLS Certificates* and re-run the `cp` +
  `update-ca-certificates` step.
- **`5.7.8 BadCredentials` from Bridge.** The password from *Mailbox
  details* is stale — copy the current value into
  `ROIGBIV_SMTP_PASSWORD` (and `source ~/.bashrc`).

## Headless / SSH-only Proton Bridge setup

Use this path when the lab box is reached only over SSH (no display).
Bridge runs locally and exposes the loopback STARTTLS SMTP relay on
`127.0.0.1:1025` that the CLI's defaults already target — only
`--smtp-user` needs to be passed at run time.

### Prerequisites

- A paid Proton Mail plan (Mail Plus or higher; Bridge isn't on the free
  tier).
- The `protonmail-bridge` `.deb` (or equivalent) installed. The package
  ships a binary at `/usr/bin/protonmail-bridge`. No systemd unit ships
  with the package.

### Step 1 — One-time login (interactive, over SSH)

Bridge has a real CLI — no Xvfb tricks needed. Run it inside `tmux` or
`screen` so the session survives SSH disconnect during initial setup:

```bash
tmux new -s bridge
protonmail-bridge --cli
```

Inside the Bridge sub-shell:

```text
>>> login
# enter your Proton email, password, and 2FA code at the prompts

>>> list
# confirms the account is registered

>>> info 0
# prints the account's IMAP and SMTP settings — copy the SMTP password.
# This is a Bridge-generated 16-character password, NOT your Proton
# account password.

>>> quit
```

Detach the tmux session with `Ctrl-b d`; you can leave it running for
the moment, but the persistence step below replaces it with systemd.

> **Cert handling on Bridge 3.24.2.** The obvious-looking sub-commands
> don't work on this version: there is no `cert install`, and `cert
> export` rejects every path you give its interactive prompt with
> "Please fill enter a path…" (regardless of whitespace, quoting, or
> typed-vs-pasted entry — three tries and it gives up). The reliable
> route is to extract Bridge's served cert directly off the wire — see
> Step 2.

### Step 2 — Trust Bridge's cert via wire extraction

Bridge must be running for this. From any shell on the lab box:

```bash
echo | openssl s_client -connect 127.0.0.1:1025 -starttls smtp -showcerts 2>/dev/null \
    | sed -n '/-----BEGIN CERTIFICATE-----/,/-----END CERTIFICATE-----/p' \
    > /tmp/proton-bridge-served.pem

# sanity check:
head -1 /tmp/proton-bridge-served.pem        # expect: -----BEGIN CERTIFICATE-----
openssl x509 -noout -subject -issuer -in /tmp/proton-bridge-served.pem
# expect: subject=CN = 127.0.0.1; issuer=CN = 127.0.0.1 (self-signed)

sudo cp /tmp/proton-bridge-served.pem /usr/local/share/ca-certificates/proton-bridge.crt
sudo update-ca-certificates
# expect: "1 added, 0 removed; done."

# verify trust:
echo | openssl s_client -connect 127.0.0.1:1025 -starttls smtp 2>&1 \
    | grep "Verify return code"
# expect: Verify return code: 0 (ok)
```

`update-ca-certificates` only picks up files ending in `.crt` (not
`.pem`) under `/usr/local/share/ca-certificates/` — the `cp` above
already renames, but if you copy by hand, mind the extension.

If `Verify return code` becomes non-zero again after a Bridge upgrade,
Bridge probably regenerated its cert; re-run this Step 2 in full to
re-extract and re-trust.

> **Conda Python doesn't read the system trust store.** `roigbiv-pipeline`
> runs from the `roigbiv` conda env, whose Python ships its own
> `cert.pem` (public CAs only). The CLI compensates by additively
> loading `/etc/ssl/certs/ca-certificates.crt` on top of certifi inside
> `roigbiv.pipeline._email.send_email`, so the system-trust step above
> is sufficient on Debian/Ubuntu/Pop_OS hosts. On Fedora/RHEL the
> system bundle lives elsewhere — point the CLI at it via:
>
> ```bash
> export ROIGBIV_SMTP_CA_FILE=/etc/pki/tls/certs/ca-bundle.crt
> ```

### Step 3 — Persist Bridge across reboots / SSH sessions

The `.deb` doesn't ship a systemd unit. Drop a user-level one in:

```ini
# ~/.config/systemd/user/protonmail-bridge.service
[Unit]
Description=Proton Mail Bridge (headless)
After=network-online.target
Wants=network-online.target

[Service]
ExecStart=/usr/bin/protonmail-bridge --noninteractive
Restart=on-failure
RestartSec=5

[Install]
WantedBy=default.target
```

Then enable it:

```bash
loginctl enable-linger $USER          # required: keeps the user manager
                                      # alive across logouts so the
                                      # service doesn't die when you
                                      # disconnect SSH
systemctl --user daemon-reload
systemctl --user enable --now protonmail-bridge.service
systemctl --user status protonmail-bridge.service
# expect: active (running)
```

`enable-linger` is critical. Without it, the user manager exits on
logout and the Bridge service goes with it — the next remote run
`Connection refused`s on `127.0.0.1:1025`.

You can now kill the tmux session from Step 1; systemd owns Bridge.

### Step 4 — Export the SMTP password and run the smoke test

```bash
export ROIGBIV_SMTP_PASSWORD='<the password from `info 0` in Step 1>'
# Optional: append to ~/.bashrc

conda activate roigbiv
python scripts/verify_email_smoke.py \
    --email-to joshuaboquiren@pm.me \
    --smtp-user thejoshbq.ext@proton.me
# expect: "Email sent to joshuaboquiren@pm.me" then "True"; email lands
#         in the recipient inbox within seconds (Proton → Proton is
#         near-instant).
```

`--smtp-host` / `--smtp-port` are omitted because the CLI defaults
already point at Bridge (`127.0.0.1:1025`).

### Step 5 — Optional convenience alias

If you don't want to type your Proton address every run, alias it.
Append to `~/.bashrc`:

```bash
alias roigbiv-mail='roigbiv-pipeline --smtp-user thejoshbq.ext@proton.me'
```

Real runs become:

```bash
roigbiv-mail --input <path> --fs 7.5 \
    --email-to joshuaboquiren@pm.me
```

(Workspace mode is implicit when `<path>` is a directory; no
`--workspace` flag.)

> Aliases don't expand inside cron jobs or systemd-invoked scripts.
> For non-interactive contexts, pass `--smtp-user
> thejoshbq.ext@proton.me` explicitly or use a wrapper shell script.

### Failure modes specific to Bridge

- **`Connection refused` on `127.0.0.1:1025`** — Bridge isn't running.
  Check `systemctl --user status protonmail-bridge.service` and restart
  if needed. After a reboot, confirm `loginctl enable-linger $USER` is
  still set (`loginctl show-user $USER | grep Linger`).
- **`SSLCertVerificationError`** — Bridge's cert isn't (or no longer
  is) trusted. First re-run Step 2's openssl extraction +
  `update-ca-certificates` and confirm `Verify return code: 0 (ok)`.
  If that's fine but the CLI still errors, the conda env's Python is
  reading a different bundle; verify with
  `conda run -n roigbiv python -c "import ssl; print(ssl.get_default_verify_paths())"`.
  Either set `ROIGBIV_SMTP_CA_FILE` to the path that holds the Bridge
  CA, or extend Step 2's `cp` target to that bundle as well.
- **`5.7.8 BadCredentials` from Bridge** — the password from `info 0`
  is wrong or stale. Bridge regenerates the password on `change`; if
  you ran that, copy the new value into `ROIGBIV_SMTP_PASSWORD`.

## Smoke test (run once on a fresh machine)

`scripts/verify_email_smoke.py` exercises the same
`roigbiv.pipeline._email.send_email` code path as the full pipeline,
but with a 1×1 PNG instead of an overlay render. It confirms Bridge
auth/TLS work without waiting on a multi-minute pipeline run.

```bash
conda activate roigbiv
export ROIGBIV_SMTP_PASSWORD='<bridge mailbox-details password>'
python scripts/verify_email_smoke.py \
    --email-to you@proton.me \
    --smtp-user you@proton.me
# expect: prints "True" and an email lands in your inbox.
```

The script's `--smtp-host` / `--smtp-port` defaults already match
Bridge (`127.0.0.1:1025`); pass them explicitly only if you've
re-bound Bridge to a non-default port.

If it prints `False`, the same stderr message the full pipeline would
emit is printed; the most common cause is a missing or stale
Bridge-generated password (re-copy from *Mailbox details* in the GUI,
or `info 0` in the CLI).

## Failure surfacing

Email failure is **not silent** at the exit-code level. If the pipeline
itself succeeded but SMTP delivery failed, `roigbiv-pipeline` exits with
status `3` and prints `Email FAILED — overlays remain on disk.` to
stderr. The full set of exit codes:

| Code | Meaning |
| --- | --- |
| `0` | Pipeline succeeded; email succeeded (or was not requested). |
| `1` | All FOVs failed. |
| `2` | Bad CLI input (e.g. `--input` path missing). |
| `3` | Pipeline succeeded but SMTP delivery failed (overlays preserved on disk under `inference/pipeline/{stem}/overlay/`). |

Use exit `3` to alert from a wrapper script when a remote run finishes
but the email never arrives.

## Idle SMTP timeouts during long runs

There aren't any. The SMTP socket is opened only after `run_pipeline`
returns, so a multi-hour compute phase never holds an idle connection.
The 30 s connect/send timeout in
`roigbiv.pipeline._email.send_email` only covers the post-run dispatch.

## Attachment size

PNG overlays larger than ~10 MiB are auto-downsampled before
attachment, and the running batch total is capped at 20 MiB to stay
comfortably under typical SMTP per-message limits (Proton's inbound
cap is 25 MiB; the 20 MiB ceiling leaves MIME-encoding headroom).
The body text notes
when downsampling occurred.
