# Dashboard — watch, scrub and share your agent's runs 📺

Three ways to look at a `.vcr` recording: a terminal debugger, a local web dashboard with a live
feed, and a self-hosted project store you can push recordings to from CI.

**TL;DR**

- `vcr run.vcr` opens a keyboard-driven debugger: step through frames, diff state, edit and export.
- `vcr serve` starts the API and web UI on `127.0.0.1:8000`. It **refuses** to listen on any other
  address without a token. With a token set, the browser signs in once at `/dashboard/login`.
- Add `--data-dir` plus an admin token and you get projects, scoped tokens and `vcr cloud push/pull`.

## Try it in 60 seconds

```bash
pip install "ai-agent-vcr[tui,dashboard]"

python - <<'EOF'
from agent_vcr import VCRRecorder

with VCRRecorder() as rec:
    rec.start_session("demo")
    rec.record_step("plan", {"task": "fix bug"}, {"plan": ["read", "patch", "test"]})
    rec.record_tool_call("run_tests", {"path": "tests/"}, {"passed": 41, "failed": 1},
                         latency_ms=820.0)
    rec.record_step("patch", {"failed": 1}, {"failed": 0})
EOF

vcr .vcr/demo.vcr        # terminal debugger: j/k to move, 3 for diff, q to quit
vcr serve                # then open http://127.0.0.1:8000  (Ctrl+C to stop)
```

## How it works

```text
 your agent ──record──▶ .vcr/*.vcr ◀──watch── vcr serve ──/ws/live──▶ browser
     │                                         ▲   │
     └── on_frame_recorded ── POST /api/push ──┘   └── /api/... JSON

 CI / laptop ── vcr cloud push ──▶ /api/projects/{id}/sessions
                                   (--data-dir, scoped tokens)
```

- `vcr serve` and `vcr-server` are the same program with the same flags (`vcr serve --help`).
- The server watches `--vcr-dir` (default `.vcr`, top level only) and broadcasts
  `session_created` and `frame_update` messages to every `/ws/live` client.
- Recorders flush to disk every 100 frames and on `save()`, so the file watcher can lag behind a
  long run. For instant updates, push each frame to `POST /api/push` (broadcast as `frame_push`).
- Without a built web UI in the package, `/` returns a JSON index of the API instead of
  redirecting to `/dashboard/`.

## Everyday recipes

### Debug a run in the terminal ⏪

```bash
vcr .vcr/demo.vcr         # or: vcr tui .vcr/demo.vcr
```

| Key | Action |
| --- | --- |
| `↑` / `k`, `↓` / `j` | Previous / next frame |
| `1` / `2` / `3` | Show input state / output state / diff |
| `s` | Filter frames by node name (`Enter` applies, `Esc` clears) |
| `e` | Edit this frame's output state as JSON (`Ctrl+S` or **Save** keeps it, `Esc` cancels) |
| `r` | Write `resume_state.json` next to the `.vcr` (session, frame index, state, edited flag) |
| `q` | Quit |

`r` doesn't re-run anything. It exports the (possibly edited) state so your own code can pick it
up with `json.load(...)`. Resuming for real is covered in
[recording-and-replay.md](recording-and-replay.md).

### Run the local dashboard

```bash
vcr serve --vcr-dir .vcr --port 8000
```

Useful API routes (all JSON):

```text
GET  /api/sessions?search=refund&tags=prod,checkout    # list, newest first
GET  /api/sessions/{id}                                # session + frames + stats
GET  /api/sessions/{id}/frames/{n}                     # one frame with input/output state
GET  /api/sessions/{id}/export?export_format=mermaid   # or json
POST /api/sessions/{id}/resume  {"from_frame": 2, "state_overrides": {...}}
```

`/resume` returns the state at that frame with your overrides applied. It doesn't execute
anything; the actual resume runs in the Python SDK.

### Put a token on it 🔐

| Setup | What happens |
| --- | --- |
| No token, `--host 127.0.0.1` (default) | Open local dev mode. Only creating projects and minting tokens is disabled (`403`). |
| No token, any other host | **Refuses to start** (exit code 2) and tells you which variable to set. |
| `AGENT_VCR_AUTH_TOKEN` and/or `AGENT_VCR_ADMIN_TOKEN` set | Every `/api` route except `/api/health` and `/api/hosted/*` needs a token. Missing or wrong → `401`. |
| `--allow-unauthenticated` / `AGENT_VCR_ALLOW_UNAUTHENTICATED=1` | Starts on a public host with no token and prints a big warning. Don't. |

```bash
export AGENT_VCR_ADMIN_TOKEN="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
vcr serve
curl -H "Authorization: Bearer $AGENT_VCR_ADMIN_TOKEN" http://127.0.0.1:8000/api/sessions
```

- Send the token as `Authorization: Bearer <token>` or `x-agent-vcr-token: <token>`. HTTP routes
  ignore `?token=`.
- `/ws/live` also accepts `?token=` (browsers can't set WebSocket headers), and it accepts only
  the two server tokens. A bad token closes the socket.
- Both server tokens have full admin rights. Prefer the environment variables over `--auth-token`
  / `--admin-token` so secrets stay out of the process list.

### Open the web UI on a token-protected server

The browser signs in once and gets a session cookie; the web UI, its API calls and its live
feed then just work, over HTTP or behind HTTPS.

1. Start the server with a token (as above) and open `http://127.0.0.1:8000/`. You land on
   `/dashboard/login`.
2. Paste `AGENT_VCR_ADMIN_TOKEN` (or `AGENT_VCR_AUTH_TOKEN`) and click **Sign in**.

For scripts or a bookmark, a link does the same thing and redirects to `/dashboard/` with the
token removed from the address bar (it still lands in your shell history and proxy logs, so
prefer the form on shared machines):

```bash
curl -s -c jar -o /dev/null -w '%{http_code} %{redirect_url}\n' \
  "http://127.0.0.1:8000/dashboard?token=$AGENT_VCR_ADMIN_TOKEN"
# 303 http://127.0.0.1:8000/dashboard/
curl -s -b jar http://127.0.0.1:8000/api/sessions | head -c 60; echo
curl -s -b jar -X POST http://127.0.0.1:8000/api/sessions/demo/resume \
  -H "Content-Type: application/json" -d '{"from_frame": 0}'
# {"detail":"Cross-origin request refused: ..."}   <- no Origin header, see below
```

- The cookie (`agent_vcr_session`) is `HttpOnly`, `SameSite=Strict`, valid for 8 hours, and
  `Secure` when the request came in over HTTPS (directly, or with `X-Forwarded-Proto: https`
  from your proxy). It's a signed expiry, never the token itself.
- Only the two server tokens can sign in. Project tokens get `401`, as on `/api/sessions`.
- Requests that change something (`POST`, `PUT`, `PATCH`, `DELETE`) authenticated only by the
  cookie must carry an `Origin` (or `Referer`) of the server itself: the browser does this
  for the web UI, a forged cross-site form doesn't get through. Requests with a token header
  skip this check. `/ws/live` accepts the cookie only on a same-origin handshake.
- Behind a proxy that rewrites `Host`, add `--allowed-origin https://your.host` (or have the
  proxy send `X-Forwarded-Host`) so the web UI's own requests count as same-origin.
- Sign out at `/dashboard/logout`. Restarting the server signs everyone out.

### Stream frames live

Point a recorder's `on_frame_recorded` callback at `POST /api/push`. The snippet is in
[integrations/README.md](integrations/README.md#live-forwarding-with-on_frame_recorded). On the
receiving side, any WebSocket client works:

```python
import asyncio
import json
import os

import websockets  # installed by the [dashboard] extra


async def main():
    url = "ws://127.0.0.1:8000/ws/live?token=" + os.environ.get("AGENT_VCR_ADMIN_TOKEN", "")
    async with websockets.connect(url) as ws:
        await ws.send(json.dumps({"action": "ping"}))   # -> {"type": "pong"}
        while True:
            msg = json.loads(await ws.recv())
            print(msg["type"], msg.get("session_id"))


asyncio.run(main())
```

### Self-host a project store for your team ☁️

Give the server a data directory and an admin token, then create a project and mint scoped tokens
(over HTTP — there is no CLI command for these two):

```bash
export AGENT_VCR_ADMIN_TOKEN=change-me-admin
vcr serve --data-dir .vcr/server &

curl -s -X POST http://127.0.0.1:8000/api/projects \
  -H "Authorization: Bearer $AGENT_VCR_ADMIN_TOKEN" -H "Content-Type: application/json" \
  -d '{"name": "Refunds", "id": "refunds"}'

curl -s -X POST http://127.0.0.1:8000/api/projects/refunds/tokens \
  -H "Authorization: Bearer $AGENT_VCR_ADMIN_TOKEN" -H "Content-Type: application/json" \
  -d '{"name": "ci", "role": "owner"}'
# {"id":"117edfdab2d3","token":"vcr_..."}   <- shown once; only a sha256 hash is stored
```

- `owner` tokens can read and write (upload, delete, list/revoke tokens). `member` tokens are
  read-only.
- Project tokens only work on `/api/projects/<their project>/...`. Everything else gets `403`.
- Revoke with `POST /api/projects/{id}/tokens/{token_id}/revoke`. It's idempotent.
- Uploads are capped at 100 MB, integrity-checked before they become visible, and deduplicated by
  sha256. Uploading a different file under an existing session id needs `--overwrite`.

### Push, pull and replay with `vcr cloud`

```bash
echo -n "vcr_..." | vcr cloud login http://127.0.0.1:8000 --token-stdin
vcr cloud status                                   # server health
vcr cloud push .vcr/demo.vcr --project refunds     # add --overwrite to replace
vcr cloud sessions refunds --search demo
vcr cloud pull refunds demo -o restored.vcr
vcr cloud replay refunds demo                      # server-side Ghost Replay, zero tokens
vcr cloud projects                                 # admin token only (project token → 403)
```

- `login` checks the token against the server before saving it to
  `~/.config/agent-vcr/credentials.json` (mode `0600`; override the folder with
  `AGENT_VCR_CONFIG_HOME`). Keep several servers apart with `--profile NAME`.
- For one-off calls, skip the profile: `--base-url URL --token -` reads the token from stdin.
- Every `vcr cloud` command exits `1` with `Error: cloud error <status>: ...` on failure.

Hosting this on the internet? See [deployment/railway.md](deployment/railway.md).

## What it can't do

- **The web UI signs in with a server token only.** Project tokens can't open it, and there are
  no user accounts: everyone signed in has admin rights. A sign-in lasts 8 hours or until the
  server restarts.
- **The project store is files on one disk.** Run a single replica. It's not a multi-tenant SaaS.
- **The live file watcher only sees `--vcr-dir` itself**, not subfolders, and only what has been
  flushed.
- **`replay` supports `--mode full` only.**
- **No TLS built in.** Put a reverse proxy in front for anything non-local.

## Cheat sheet

| Command | What it does |
| --- | --- |
| `vcr FILE.vcr` / `vcr tui FILE.vcr` | Terminal debugger (`[tui]` extra) |
| `vcr serve` / `vcr-server` | API + web UI + live feed (`[dashboard]` extra) |
| `--host` / `--port` | Bind address (default `127.0.0.1:8000`) |
| `--vcr-dir DIR` / `--data-dir DIR` | Recordings to serve (`.vcr`) / project store (default: same as `--vcr-dir`) |
| `--auth-token` / `--admin-token` | Tokens (default from `AGENT_VCR_AUTH_TOKEN` / `AGENT_VCR_ADMIN_TOKEN`) |
| `--allow-unauthenticated` | Allow a public bind without a token (insecure) |
| `--allowed-origin URL` | CORS origin, repeatable (default: the server's own localhost); also trusted as same-origin for the web UI's cookie |
| `/dashboard/login` · `/dashboard?token=…` · `/dashboard/logout` | Web UI sign-in (session cookie) and sign-out |
| `--workflow-handlers FILE` | Tools for HTTP workflows — see [integrations/n8n.md](integrations/n8n.md) |
| `--hosted` | Supabase/Stripe endpoints — see [deployment/railway.md](deployment/railway.md) |
| `vcr cloud login URL --token-stdin` | Verify and store credentials |
| `vcr cloud status \| projects \| sessions P` | Inspect the server |
| `vcr cloud push FILE --project P` / `pull P SID -o OUT` | Move recordings |
| `vcr cloud replay P SID` | Server-side Ghost Replay |

## Next

- [recording-and-replay.md](recording-and-replay.md) — time travel and Ghost Replay in depth.
- [deployment/railway.md](deployment/railway.md) — host the server with TLS and tokens.
- [integrations/README.md](integrations/README.md) — get frames in from your framework.
