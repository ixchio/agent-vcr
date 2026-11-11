# n8n — trigger recorded agent workflows from your automations

Point an n8n HTTP Request node at the Agent VCR server, start a workflow, and get the result back —
right away, or later through a webhook callback.

> 🧪 **Experimental.** Works and is tested, but the API may change between minor releases.

**TL;DR**

- `POST /api/workflows/{name}/trigger` runs a registered workflow; every run is recorded to `.vcr`.
- Add `callback_url` (n8n's `$execution.resumeUrl`) and you get `202` now, the result later.
- The callback URL must be **public** unless the server sets `AGENT_VCR_ALLOW_PRIVATE_CALLBACKS=1`.

## Try it in 60 seconds

Everything local, no API keys. You need `pip install "ai-agent-vcr[dashboard]"`.

1. A tool for the workflow to call — save as `handlers.py`:

   ```python
   def lookup_ticket(ticket_id: str = "") -> dict:
       return {"ticket_id": ticket_id, "priority": "high"}


   TOOLS = {"lookup_ticket": lookup_ticket}
   ```

2. A workflow definition — save as `triage.json`:

   ```json
   {
     "name": "triage",
     "triggers": [{"type": "webhook", "secret": "change-me"}],
     "steps": [
       {"id": "lookup", "kind": "tool", "ref": "lookup_ticket",
        "inputs": {"ticket_id": "$inputs.ticket_id"}}
     ]
   }
   ```

3. Start the server (loopback only), register the workflow, trigger it:

   ```bash
   export AGENT_VCR_ADMIN_TOKEN=dev-token
   vcr serve --workflow-handlers handlers.py &

   curl -s -X POST http://127.0.0.1:8000/api/workflows \
     -H "Authorization: Bearer $AGENT_VCR_ADMIN_TOKEN" -H "Content-Type: application/json" \
     -d @triage.json

   curl -s -X POST http://127.0.0.1:8000/api/workflows/triage/trigger \
     -H "Authorization: Bearer $AGENT_VCR_ADMIN_TOKEN" -H "Content-Type: application/json" \
     -d '{"inputs": {"ticket_id": "T-42"}, "secret": "change-me"}'
   # {"run_id":"f71a3f9a1092","status":"committed","session_id":"20261111_170605_63dffe72"}
   ```

That `session_id` is a normal recording in `.vcr/` — open it with `vcr .vcr/<session_id>.vcr`.
Stop the server with `kill %1`. The workflow format itself is documented in
[../experimental/workflow.md](../experimental/workflow.md).

## How it works

```text
 n8n                        Agent VCR server                 disk
 ─────────────────          ──────────────────────────       ──────────────
 HTTP Request ─ trigger ──▶ check token + workflow secret
                            run steps (TOOLS/AGENTS) ──────▶ .vcr/<sid>.vcr
   sync:  ◀── 200 {run_id, status, session_id}
   async: ◀── 202 accepted
 Wait node ◀── POST {run_id, status, session_id} to callback_url
```

- Workflows that call tools or agents need `--workflow-handlers file.py`, a Python file exporting
  `TOOLS` (and optionally `AGENTS`, `APPROVAL_REQUIRED`). Without it, such a trigger gets `409`.
- The workflow **secret** (in the definition's `webhook` trigger) is separate from the server's
  bearer token. A wrong secret gets `403`. An empty secret means no check.
- Steps that require human approval are **denied** on the server. There is no one to ask.
- Workflow routes are server-wide, so they need an **admin-level** token (`AGENT_VCR_ADMIN_TOKEN`
  or `AGENT_VCR_AUTH_TOKEN`). Project-scoped tokens get `403`. With no token configured, the
  server only listens on loopback and these routes are open — see [../dashboard.md](../dashboard.md).

## Everyday recipes

### Synchronous trigger (short runs)

An n8n **HTTP Request** node, body type JSON:

```text
POST {{vcr_url}}/api/workflows/triage/trigger
Authorization: Bearer <admin token, from an n8n credential>
Content-Type: application/json
```

```json
{
  "inputs": {"ticket_id": "{{$json.ticket_id}}"},
  "secret": "change-me"
}
```

The response is `{"run_id", "status", "session_id"}`. `status` is `committed` on success, `aborted`
when the workflow aborted.

### Async callback (long runs)

1. Trigger as above, adding `"callback_url": "{{$execution.resumeUrl}}"` to the body.
2. Follow it with a **Wait** node set to *Resume: On Webhook Call*.
3. Route on `{{$json.status}}` after the Wait node.

Agent VCR answers `202 {"run_id": "pending", "status": "accepted"}` immediately, runs the workflow,
then POSTs JSON to the callback URL:

```json
{"run_id": "f83105d37378", "status": "committed", "session_id": "20261111_170620_0e981f7a"}
```

If the run itself fails to start (for example a wrong secret, which is only checked in the
background in async mode), the callback body is
`{"run_id": "unknown", "status": "failed", "error": "..."}`.

### The callback URL rule (SSRF guard) 🛡️

The server makes an outbound request to `callback_url`, so it is checked twice: when you trigger,
and again right before sending. It must be `http://` or `https://`, have no `user:pass@`, be under
2048 characters, and resolve **only to public IP addresses**. Loopback, private (10/8, 172.16/12,
192.168/16), link-local, cloud-metadata (169.254.169.254) and CGNAT addresses are refused with `422`:

```json
{"detail": "callback_url resolves to a loopback, private, link-local or metadata address; set AGENT_VCR_ALLOW_PRIVATE_CALLBACKS=1 on the server to allow it"}
```

Running n8n on the same machine or in the same Docker network (`http://n8n:5678/...`,
`http://localhost:5678/...`)? Start the server with:

```bash
AGENT_VCR_ALLOW_PRIVATE_CALLBACKS=1 vcr serve --workflow-handlers handlers.py
```

Only do that when you trust everyone who holds the admin token. The connection is pinned to the
address that passed the check, and callbacks time out after 10 seconds. A refused or failed
callback is logged on the server, with the origin only — never the path.

### Check on a run, or resume it

```text
GET  /api/workflows/triage/runs           # recent runs for this workflow
GET  /api/workflow-runs/{run_id}          # full persisted run record
POST /api/workflow-runs/{run_id}/resume   # continue an interrupted run
```

### Ask why a run failed

If you also upload recordings to a project (see [../dashboard.md](../dashboard.md)), an HTTP
Request node can ask for a diagnosis:

```text
GET /api/projects/{project_id}/intel/why/{session_id}
Authorization: Bearer <admin token or a token for that project>
```

The answer is decision support built from the recorded evidence. It never runs a recovery by itself.

### Start from the template

[n8n-workflow-template.json](n8n-workflow-template.json) wires Trigger → Wait → IF
(`status == committed`) → diagnose. Import it, then fill in the **Config** node: base URL,
workflow name, project id, token and `webhook_secret`. The trigger already sends a JSON body with
`inputs` (an object — add your own keys), the `secret`, and n8n's resume URL as `callback_url`.
Keep tokens in n8n credentials, not in the exported JSON.

## What it can't do

- **No rollback of n8n's side effects.** Writes made by other n8n nodes or third-party services
  are not undone. Model them with compensation steps or approval gates.
- **No human approvals over HTTP.** Approval-gated steps are denied on the server.
- **Secrets are stored in plain JSON.** The workflow secret is in the saved definition and comes
  back from `GET /api/workflows/{name}` to any admin-token holder.
- **Callback delivery is one attempt.** No retries. If the callback fails, read the run record
  with `GET /api/workflow-runs/{run_id}`.
- **No tokens in query strings.** HTTP routes accept only `Authorization: Bearer` or
  `x-agent-vcr-token` (URLs end up in logs).
- **Public exposure is on you.** For a remote server, use TLS at a trusted proxy and a token — see
  [../deployment/railway.md](../deployment/railway.md).

## Cheat sheet

| Method | Endpoint | What it does |
| --- | --- | --- |
| `GET` | `/api/workflows` | List registered workflows |
| `POST` | `/api/workflows` | Validate and save a workflow (`201`) |
| `GET` | `/api/workflows/{name}` | Read a definition (`?version=N` for an older one) |
| `POST` | `/api/workflows/{name}/trigger` | Run it: `200` sync, `202` with `callback_url` |
| `GET` | `/api/workflows/{name}/runs` | List runs |
| `GET` | `/api/workflow-runs/{run_id}` | Read a run record |
| `POST` | `/api/workflow-runs/{run_id}/resume` | Resume an interrupted run |
| `GET` | `/api/projects/{pid}/intel/why/{sid}` | Diagnose a recorded session |

| Status | Meaning |
| --- | --- |
| `401` | Missing or wrong token |
| `403` | Wrong workflow secret, or a project token on a workflow route |
| `404` | Unknown workflow or run |
| `409` | Workflow needs tools/agents but the server has no `--workflow-handlers` |
| `422` | Invalid body, workflow or `callback_url` |

| Setting | Effect |
| --- | --- |
| `--workflow-handlers file.py` | Where `TOOLS` / `AGENTS` come from |
| `AGENT_VCR_ALLOW_PRIVATE_CALLBACKS=1` | Allow callbacks to loopback/private hosts |

## Next

- [../experimental/workflow.md](../experimental/workflow.md) — the workflow format and `vcr workflow`.
- [../dashboard.md](../dashboard.md) — server auth, tokens and the project store.
- [README.md](README.md) — every other integration.
