# n8n Integration

Agent VCR provides powerful testing capabilities for n8n workflows. It allows you to:
- Use **Ghost Replay** for reproducing past executions locally.
- Implement **ACID rollback** of your external service writes.
- Set up **Sentinel** to alert you of changes in agent responses.

## Zero-code Pattern

You can integrate Agent VCR in n8n without writing code using the **HTTP Request** node in a zero-code pattern:
1. Make an **HTTP Request** to `POST /api/workflows/{workflow_name}/trigger`
2. Specify the `callback_url` parameter using n8n's Wait node webhook URL (`{{$execution.resumeUrl}}`).
3. Add a **Wait** node (Wait for Webhook Call) immediately after the HTTP Request node.

## Available API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/workflows` | List all workflows |
| POST | `/api/workflows` | Save a workflow |
| GET | `/api/workflows/{name}` | Get a workflow |
| POST | `/api/workflows/{name}/trigger` | Trigger a workflow |
| GET | `/api/workflows/{name}/runs` | List runs for a workflow |
| GET | `/api/workflow-runs/{run_id}` | Get workflow run |
| POST | `/api/workflow-runs/{run_id}/resume` | Resume workflow run |

## Authentication Setup

To authenticate:
- Pass the token as a Bearer token in the `Authorization` header, or
- Pass the token using the `x-agent-vcr-token` custom header.

## Error Handling Patterns

If a run fails or exhibits unexpected behavior, you can implement self-healing error analysis by calling the Agent VCR Intelligence endpoint:
`GET /api/projects/{project_id}/intel/why/{session_id}`

## Async Callback Pattern

When you use the `callback_url` in the `trigger` endpoint, Agent VCR will immediately accept the request and run the workflow asynchronously. When the workflow finishes, Agent VCR will `POST` the run status back to your `callback_url`.

## Workflow Template

See the template here: [n8n-workflow-template.json](./n8n-workflow-template.json)
