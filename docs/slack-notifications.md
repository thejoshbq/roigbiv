# Slack notifications on pipeline completion

`roigbiv-pipeline` can post a run summary to a Slack channel and upload the
per-FOV overlay PNG(s) into that message's thread when a run finishes — pass
`--slack-channel <channel-id>` and the message goes out automatically. Use it
for unattended runs where you'd rather watch a channel than poll the host. The
Dash UI (`roigbiv-ui`) can do the same via an optional channel field on the
Process page.

Unlike email, Slack needs **no personal account and no SMTP bridge** — just a
one-time bot in your own workspace whose token lives in `ROIGBIV_SLACK_TOKEN`.

## CLI flags

| Flag | Default | Notes |
| --- | --- | --- |
| `--slack-channel` | — | Slack **channel ID** (e.g. `C0123ABCD`). Omit to skip Slack. |
| `--slack-token-env` | `ROIGBIV_SLACK_TOKEN` | Env-var name holding the bot token. **Never** pass the token on the command line. |
| `--no-slack` | off | Skip dispatch even if `--slack-channel` is set. |

Email and Slack are independent: use either, both, or neither. The overlay
filter (`--overlay-outcomes`) applies to both.

## One-time bot setup

1. **Create the app.** Go to <https://api.slack.com/apps> → *Create New App* →
   *From scratch*. Name it (e.g. "ROIGBIV") and pick your workspace.
2. **Add bot scopes.** *OAuth & Permissions* → *Scopes* → *Bot Token Scopes*,
   add exactly two:
   - `chat:write` — post the summary message.
   - `files:write` — upload the overlay PNGs.
3. **Install to workspace.** *OAuth & Permissions* → *Install to Workspace* →
   *Allow*. Copy the **Bot User OAuth Token** (starts with `xoxb-`).
4. **Invite the bot to the channel.** In Slack, open the target channel and
   type `/invite @ROIGBIV` (the bot can only post to channels it's a member
   of).
5. **Copy the channel ID.** In Slack, click the channel name → scroll to the
   bottom of the *About* tab → copy the ID (e.g. `C0123ABCD`). This is **not**
   the `#name` — the upload API requires the ID.
6. **Export the token** in the environment that runs the pipeline (or launches
   the UI):

   ```bash
   export ROIGBIV_SLACK_TOKEN='xoxb-…'
   # Optional: append to ~/.bashrc
   ```

## Run it

```bash
conda activate roigbiv
roigbiv-pipeline --input fov_dir/ --fs 7.5 --slack-channel C0123ABCD
```

A single summary message appears in the channel (accept/flag/reject counts,
duration, model/fs/tau, per-FOV lines), with one overlay PNG uploaded per FOV
into its thread.

**UI:** `export ROIGBIV_SLACK_TOKEN=…` before `roigbiv-ui`, then enter the
channel ID in the **Notifications** field on the Process page. The token is
never entered in the browser — it must be in the UI process's environment. The
run log shows `Slack: summary + overlays posted.` on success.

## Failure surfacing

Slack failure is **not silent** and does **not** disturb email's semantics:

| Code | Meaning |
| --- | --- |
| `0` | Pipeline succeeded; notifications succeeded (or none requested). |
| `1` | All FOVs failed. |
| `2` | Bad CLI input. |
| `3` | Pipeline succeeded but **email** (SMTP) delivery failed. |
| `4` | Pipeline succeeded but **Slack** delivery failed. |

When both email and Slack are requested and both fail, exit `3` (email) takes
precedence; pipeline-fail (`1`) and bad-input (`2`) always dominate. On any
Slack failure the overlay PNGs remain on disk and their paths are printed to
stderr. In the UI, failure surfaces as a `Slack FAILED …` log line (detailed
stderr goes to the server console).

## Attachment size

Overlay PNGs larger than ~10 MiB are downsampled before upload (shared with the
email path), well under Slack's per-file limit.
