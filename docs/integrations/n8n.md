# n8n integration

n8n can call the optional Agent VCR server to trigger a registered workflow, wait for an async
callback, inspect run state, and request reliability diagnosis. Agent VCR does not automatically
roll back writes made by arbitrary n8n nodes or third-party services; model those effects with
workflow compensation or approval gates.

## Start the server

```bash
pip install "ai-agent-vcr[dashboard]"
vcr-server \
  --vcr-dir .vcr \
  --data-dir .vcr/server \
  --auth-token "$AGENT_VCR_TOKEN"
```

Keep the default loopback host for local n8n. For a remote deployment, terminate TLS at a trusted
proxy and do not expose an unauthenticated server to the public internet.

## Synchronous trigger

Use an n8n HTTP Request node:

```text
POST /api/workflows/{workflow_name}/trigger
Authorization: Bearer <token>
Content-Type: application/json
```

```json
{
  "inputs": {
    "ticket_id": "{{$json.ticket_id}}"
  },
  "secret": "registered-workflow-trigger-secret"
}
```

Without `callback_url`, the response contains the completed run record. The workflow trigger secret
is separate from the server bearer token and must match the registered workflow definition.

## Asynchronous callback

For a long run:

1. add an n8n Wait node configured to resume on webhook call;
2. pass `{{$execution.resumeUrl}}` as `callback_url` in the trigger body; and
3. route the returned callback body by its run status.

```json
{
  "inputs": {
    "ticket_id": "{{$json.ticket_id}}"
  },
  "secret": "registered-workflow-trigger-secret",
  "callback_url": "{{$execution.resumeUrl}}"
}
```

Agent VCR returns HTTP 202 immediately, then posts the final run record to that URL. Treat callback
URLs as secrets and allow them only from a trusted workflow definition; they cause an outbound HTTP
request from the Agent VCR server.

## Workflow endpoints

| Method | Endpoint | Purpose |
| --- | --- | --- |
| `GET` | `/api/workflows` | List registered workflows |
| `POST` | `/api/workflows` | Validate and save a workflow |
| `GET` | `/api/workflows/{name}` | Read one workflow definition |
| `POST` | `/api/workflows/{name}/trigger` | Start a workflow |
| `GET` | `/api/workflows/{name}/runs` | List runs for one workflow |
| `GET` | `/api/workflow-runs/{run_id}` | Read a persisted run record |
| `POST` | `/api/workflow-runs/{run_id}/resume` | Resume an interrupted run |

When `--auth-token` is configured, send it as `Authorization: Bearer ...` or
`x-agent-vcr-token: ...`. Do not put tokens in query strings because URLs are commonly logged.

## Diagnose a failed run

If the server also uses the project/session store, an HTTP Request node can call:

```text
GET /api/projects/{project_id}/intel/why/{session_id}
Authorization: Bearer <token>
```

The result is decision support based on recorded evidence. It does not automatically execute a
recovery action.

## Importable template

Start from [n8n-workflow-template.json](n8n-workflow-template.json), then replace the base URL,
workflow name, credentials, and input mapping. Store tokens in n8n credentials rather than directly
inside the exported workflow JSON.
