# Agent VCR + SuperAgent
## Agent Reliability Cloud: System Design & Startup Blueprint

> **Status:** Historical internal design source. The current implementation is documented in
> [`../architecture.md`](../architecture.md) and the public run path is in
> [`../../README.md`](../../README.md).
>
> **Thesis:** Build the reliability, execution, recovery, and deployment layer for autonomous AI agents.
>
> **Positioning:** Time-travel infrastructure for autonomous AI.
>
> **Core loop:** Record → Checkpoint → Replay → Evaluate → Fork → Recover → Deploy.

---

## 1. Executive Summary

Agent VCR should evolve from a developer-focused time-travel/debugging library into an **Agent Reliability Cloud**.

The platform combines:

- **Agent VCR Core**: deterministic-ish execution recording, checkpoints, rollback, fork, replay, state diffs, Ghost Replay.
- **SuperAgent Runtime**: planner, sub-agents, tool execution, memory, policies, verification, recovery.
- **Workflow Engine**: durable DAG execution, triggers, schedules, approvals, retries, branching and compensation.
- **Reliability Cloud**: execution history, traces, evaluations, regression detection, replay datasets, analytics.
- **Agent Control Plane**: deployments, environments, secrets, permissions, budgets, policies, versions, audit logs.
- **Sandbox Runtime**: isolated execution for files, browsers, shells and external tools.
- **Automatic Recovery Engine**: detects failures and proposes or executes safe recovery strategies.

The strategic distinction is not “another AI agent builder.” The platform is designed to answer:

> **How do you run autonomous agents safely, reproducibly, and recoverably once they can affect real systems?**

---

# 2. Product Vision

## 2.1 One-line vision

**The reliability platform for autonomous AI.**

## 2.2 Product promise

An agent execution can be treated like a versioned software artifact:

- checkpoint it like a Git commit
- branch it like Git
- diff it like Git
- replay it like a test fixture
- rollback it like a database transaction
- evaluate it like CI
- deploy it like an application
- recover it like a resilient distributed workflow

## 2.3 Mental model

Traditional software:

```text
Git
 ├── commit
 ├── branch
 ├── diff
 ├── merge
 └── revert
```

Agent software:

```text
Agent VCR
 ├── checkpoint
 ├── branch
 ├── replay
 ├── state diff
 ├── fork
 ├── evaluate
 ├── rollback
 └── recover
```

---

# 3. Problem

Autonomous agents are fundamentally harder to debug than normal request/response applications.

An agent may:

1. call an LLM repeatedly
2. select tools dynamically
3. mutate files
4. browse the web
5. access APIs
6. invoke MCP tools
7. use multiple sub-agents
8. maintain changing memory/state
9. execute asynchronously
10. make decisions based on previous tool outputs
11. create side effects in external systems

When something fails, restarting from zero is expensive and often nondeterministic.

Typical observability systems can tell an engineer **what happened**. Agent VCR should let them **go back, change the execution, and resume**.

---

# 4. Core Differentiation

## 4.1 Observability vs Reliability

```text
Observability platform
    ↓
trace
    ↓
inspect
    ↓
evaluate
    ↓
understand failure
```

Agent Reliability platform:

```text
execution
    ↓
checkpoint
    ↓
observe
    ↓
detect failure
    ↓
rollback / fork
    ↓
modify state or strategy
    ↓
resume
    ↓
verify
```

## 4.2 Primary differentiation

**Active intervention instead of passive inspection.**

## 4.3 Secondary differentiation

The platform should create a reusable execution dataset from real agent runs. That dataset becomes the basis for:

- regression tests
- evaluations
- failure clustering
- model comparisons
- prompt comparisons
- tool reliability analysis
- recovery optimization
- cost optimization
- agent risk scoring

---

# 5. Product Architecture

```text
                         ┌─────────────────────────┐
                         │       Developer / Team   │
                         └────────────┬────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │                                   │
               CLI / SDK                         Web Console
                    │                                   │
                    └─────────────────┬─────────────────┘
                                      │
                              Control Plane API
                                      │
          ┌───────────────────────────┼───────────────────────────┐
          │                           │                           │
     Project Service            Policy Service             Auth/RBAC
          │                           │                           │
          └───────────────────────────┼───────────────────────────┘
                                      │
                              Execution Orchestrator
                                      │
                    ┌─────────────────┼─────────────────┐
                    │                 │                 │
               SuperAgent       Workflow Engine    Recovery Engine
                    │                 │                 │
                    └─────────────────┼─────────────────┘
                                      │
                              VCR Execution Kernel
                                      │
                 ┌────────────────────┼────────────────────┐
                 │                    │                    │
            State Store         Event Recorder        Checkpoint Store
                 │                    │                    │
                 └────────────────────┼────────────────────┘
                                      │
                              Sandbox Runtime
                                      │
          ┌──────────────┬────────────┼─────────────┬──────────────┐
          │              │            │             │              │
       Filesystem      Shell       Browser        MCP            APIs
          │              │            │             │              │
          └──────────────┴────────────┴─────────────┴──────────────┘
                                      │
                              External Services
```

---

# 6. Component Architecture

## 6.1 VCR Execution Kernel

The kernel is the heart of the platform.

Responsibilities:

- generate execution IDs
- assign frame/checkpoint IDs
- capture state transitions
- capture model requests/responses
- capture tool requests/responses
- capture filesystem diffs
- capture environment metadata
- record side effects
- create snapshots
- restore snapshots
- replay recorded interactions
- fork executions
- calculate execution diffs
- maintain lineage graph

Core abstraction:

```python
Execution
  ├── Run
  │    ├── Frame 0
  │    ├── Frame 1
  │    ├── Frame 2
  │    └── ...
  │
  ├── Checkpoints
  ├── Events
  ├── State snapshots
  ├── Tool calls
  ├── Model calls
  ├── Artifacts
  └── Side effects
```

### Execution identity

```text
project_id
agent_id
workflow_id
execution_id
parent_execution_id
branch_id
frame_id
checkpoint_id
```

---

# 7. State Model

Agent state must be split into layers because not all state has the same rollback semantics.

## 7.1 Logical state

Examples:

- planner state
- agent messages
- variables
- memory entries
- task status
- workflow state

## 7.2 Filesystem state

Examples:

- source code
- generated files
- temp files
- configuration
- artifacts

Use copy-on-write snapshots and content-addressed blobs where possible.

## 7.3 Runtime state

Examples:

- process metadata
- environment variables
- active sessions
- browser state
- credentials references

## 7.4 External side effects

Examples:

- database writes
- emails
- payments
- API mutations
- deployments
- messages

These cannot always be physically rolled back. The system therefore needs a **side-effect policy** and compensation model.

```text
reversible side effect
    ↓
rollback

irreversible side effect
    ↓
compensation / approval / block
```

---

# 8. SuperAgent Runtime

SuperAgent should be built on top of the VCR kernel rather than beside it.

## 8.1 SuperAgent responsibilities

- interpret task
- construct plan
- select sub-agents
- select tools
- execute actions
- maintain working memory
- request verification
- enforce budgets
- request human approval
- recover from failures

## 8.2 SuperAgent architecture

```text
                     SuperAgent
                         │
                    Task Intake
                         │
                         ▼
                     Planner
                         │
            ┌────────────┼─────────────┐
            ▼            ▼             ▼
       Researcher    Executor      Analyst
            │            │             │
            └────────────┼─────────────┘
                         ▼
                     Verifier
                         │
                 ┌───────┴────────┐
                 │                │
              Pass             Fail
                 │                │
                 ▼                ▼
              Commit        Recovery Engine
                                  │
                         ┌────────┼────────┐
                         ▼        ▼        ▼
                      Retry    Rollback   Fork
```

## 8.3 Planner

The planner produces an executable plan, not merely text.

```json
{
  "goal": "complete_customer_refund",
  "steps": [
    {
      "id": "classify",
      "agent": "classifier"
    },
    {
      "id": "lookup",
      "agent": "researcher"
    },
    {
      "id": "verify",
      "agent": "verifier"
    },
    {
      "id": "refund",
      "agent": "payment_executor",
      "requires_approval": true
    }
  ]
}
```

The planner should produce explicit checkpoints and policy boundaries.

---

# 9. Workflow Engine

The workflow engine turns agent executions into durable applications.

## 9.1 Workflow primitives

- triggers
- schedules
- steps
- agents
- tools
- conditions
- branches
- loops
- parallel execution
- retries
- timeouts
- checkpoints
- approvals
- compensations
- joins
- terminal states

## 9.2 Workflow definition example

```yaml
workflow:
  name: customer_support_resolution

  trigger:
    type: webhook

  agents:
    - classifier
    - researcher
    - resolver
    - verifier

  checkpoints:
    - after_classifier
    - before_external_action
    - before_payment

  policies:
    max_cost: 0.30
    max_runtime: 10m
    require_human_if:
      - confidence < 0.85
      - payment > 500
      - destructive_action == true
```

## 9.3 Workflow storage

A workflow is versioned.

```text
workflow:v13
   ├── definition
   ├── agent versions
   ├── tool versions
   ├── policy versions
   └── environment
```

Every execution records the exact version graph.

---

# 10. Agent CI

Agent CI is a major product feature.

Every pull request can run historical replay tests.

Example:

```text
PR #482

Agent Reliability CI

1,000 replay tests
✅ 973 passed
⚠️ 19 degraded
❌ 8 failed

Cost
$0.19 → $0.27

Latency
1.3s → 2.1s

Tool error rate
0.8% → 4.1%

Risk
LOW → HIGH
```

## 10.1 CI pipeline

```text
Git push
  ↓
build agent
  ↓
select replay dataset
  ↓
execute deterministic replays
  ↓
run model/tool policy
  ↓
evaluate outputs
  ↓
compare baseline
  ↓
generate reliability report
```

## 10.2 Regression dimensions

- output correctness
- tool correctness
- state transitions
- cost
- latency
- token usage
- failure rate
- policy violations
- human escalation rate

---

# 11. Ghost Replay

Ghost Replay should become a flagship feature.

## 11.1 Goal

Replay an agent execution without re-paying for external model/tool operations wherever the captured interaction is safe to reuse.

```text
Production Run
     ↓
Record
     ↓
Artifact
     ↓
Developer Replay
     ↓
Prompt / code / policy experiment
     ↓
Compare results
```

## 11.2 Benefits

- faster debugging
- lower API cost
- deterministic reproduction of failures
- safe development against captured production scenarios
- large-scale regression testing

## 11.3 Replay modes

```text
FULL_REPLAY
    Reproduce using captured interactions.

HYBRID_REPLAY
    Reuse selected interactions while re-running others.

LIVE_REPLAY
    Re-run selected external operations.

BRANCH_REPLAY
    Fork an execution at a chosen checkpoint.
```

---

# 12. Forking Model

An execution is a graph, not a linear log.

```text
                 frame 0
                   │
                   ▼
                 frame 1
                   │
                   ▼
                 frame 2
                /      \
               /        \
        original        fork-A
           │              │
        frame 3        frame 3'
           │              │
        frame 4        frame 4'
```

Each fork records:

```text
parent_execution_id
fork_frame_id
fork_reason
configuration_diff
agent_version
policy_version
```

This enables experiment trees.

---

# 13. Recovery Engine

The recovery engine should turn failures into structured decisions.

## 13.1 Failure taxonomy

```text
MODEL_FAILURE
TOOL_FAILURE
NETWORK_FAILURE
TIMEOUT
STATE_CORRUPTION
POLICY_VIOLATION
UNEXPECTED_OUTPUT
LOW_CONFIDENCE
RESOURCE_LIMIT
EXTERNAL_SIDE_EFFECT_FAILURE
```

## 13.2 Recovery strategies

```text
retry
↓
backoff
↓
switch tool
↓
switch model
↓
rollback
↓
modify state
↓
fork execution
↓
human approval
↓
abort
```

## 13.3 Recovery planning

```text
Execution failed
      ↓
Failure Classifier
      ↓
Context Builder
      ↓
Recovery Planner
      ↓
Candidate strategies
      ↓
Safety filter
      ↓
Cost estimator
      ↓
Best strategy
      ↓
Execute
      ↓
Verify
```

The system should avoid blindly retrying irreversible actions.

---

# 14. Agent Risk Engine

Every execution receives reliability metadata.

Example:

```text
Reliability: 94
Safety:      97
Cost:        81
Confidence:  92
Regression:   3%
```

## 14.1 Risk factors

- tool failure history
- model confidence
- output validation
- action reversibility
- privilege level
- external-system sensitivity
- historical failure rate
- cost anomaly
- policy violations
- unexpected state mutation

## 14.2 Risk levels

```text
LOW
MEDIUM
HIGH
CRITICAL
```

High-risk actions can require approval automatically.

---

# 15. Policy Engine

Policies should be executable and centrally managed.

Example:

```yaml
policy:
  name: payment_guardrail

  rules:
    - when: payment.amount > 500
      action: require_human

    - when: external_write == true
      action: checkpoint

    - when: confidence < 0.80
      action: escalate

    - when: tool == shell
      action: restricted_sandbox
```

Policies are versioned and attached to execution snapshots.

---

# 16. Sandbox System

Agents should run in controlled execution environments.

## 16.1 Sandbox components

- ephemeral container or microVM
- filesystem isolation
- network controls
- CPU limits
- memory limits
- process limits
- timeout
- secret proxy
- browser isolation
- tool permissions
- execution snapshots

## 16.2 Sandbox lifecycle

```text
request
  ↓
provision sandbox
  ↓
load agent
  ↓
attach VCR
  ↓
execute
  ↓
checkpoint
  ↓
commit artifacts
  ↓
freeze / destroy
```

## 16.3 Secrets

Agents should never receive unrestricted long-lived secrets.

Prefer:

```text
Agent
 ↓
credential reference
 ↓
policy check
 ↓
secret proxy
 ↓
short-lived credential
 ↓
external system
```

---

# 17. Tool Gateway

All external actions should pass through a unified tool gateway where possible.

```text
Agent
 ↓
Tool Router
 ↓
Policy
 ↓
Audit
 ↓
Rate limit
 ↓
Tool execution
 ↓
Result capture
```

Supported tool classes:

- HTTP APIs
- MCP tools
- shell
- filesystem
- browser
- databases
- queues
- messaging
- cloud APIs

This allows consistent observability, permissions and replay policies.

---

# 18. External Side Effects

Not everything is rollback-safe.

## 18.1 Side-effect classes

```text
PURE
READ_ONLY
REVERSIBLE
COMPENSATABLE
IRREVERSIBLE
```

Examples:

```text
read API       → READ_ONLY
file write     → REVERSIBLE
DB transaction → REVERSIBLE / COMPENSATABLE
email send     → IRREVERSIBLE
payment        → IRREVERSIBLE / COMPENSATABLE depending on system
```

## 18.2 Side-effect policy

Before an irreversible operation:

```text
checkpoint
  ↓
policy evaluation
  ↓
optional human approval
  ↓
execute
  ↓
record
```

---

# 19. Cloud Architecture

A practical initial architecture:

```text
                         Internet
                            │
                         API Gateway
                            │
                    ┌───────┴────────┐
                    │                │
                Control API      Ingest API
                    │                │
                    └───────┬────────┘
                            │
                     Event / Job Bus
                            │
       ┌────────────────────┼────────────────────┐
       │                    │                    │
  Orchestrator        Evaluator Workers     Recovery Workers
       │                    │                    │
       └────────────────────┼────────────────────┘
                            │
              ┌─────────────┼─────────────┐
              │             │             │
           Postgres      Object Store   Event Store
              │             │             │
              │             │             │
        metadata/index    snapshots     append log
              │             │             │
              └─────────────┼─────────────┘
                            │
                         Query API
                            │
                         Web App
```

## 19.1 Initial infrastructure choices

A sensible implementation could use:

- PostgreSQL for metadata, workflow definitions, users, indexes and execution metadata.
- S3-compatible object storage for snapshots, replay bundles, artifacts and large payloads.
- Redis for short-lived coordination/caching where useful.
- A durable queue or event bus for execution jobs.
- Container runtime or microVM infrastructure for agent sandboxes.
- OpenTelemetry-compatible event schemas for interoperability.

Exact infrastructure should remain replaceable behind internal interfaces.

---

# 20. Data Model

Core entities:

```text
User
Organization
Project
Environment
Agent
AgentVersion
Workflow
WorkflowVersion
Execution
Frame
Checkpoint
Event
ToolCall
ModelCall
Artifact
Snapshot
Policy
PolicyVersion
Evaluation
EvaluationRun
ReplayDataset
Fork
RecoveryAttempt
Sandbox
Deployment
SecretReference
AuditEvent
UsageRecord
BillingAccount
```

## 20.1 Execution record

```json
{
  "execution_id": "exec_123",
  "project_id": "proj_1",
  "agent_version": "agent:v42",
  "workflow_version": "workflow:v13",
  "policy_version": "policy:v8",
  "status": "failed",
  "started_at": "...",
  "ended_at": "...",
  "cost_usd": 0.27,
  "token_usage": 18342,
  "risk_score": 63,
  "parent_execution_id": null,
  "branch_id": "main"
}
```

---

# 21. Event Model

All meaningful state transitions should become structured events.

```json
{
  "event_id": "evt_123",
  "execution_id": "exec_123",
  "frame_id": "frame_17",
  "type": "tool.call",
  "timestamp": "...",
  "tool": "github.create_pr",
  "input_ref": "blob://...",
  "output_ref": "blob://...",
  "duration_ms": 820,
  "status": "success"
}
```

Potential event types:

```text
execution.started
execution.completed
execution.failed
checkpoint.created
checkpoint.restored
frame.created
model.request
model.response
tool.request
tool.response
filesystem.diff
browser.action
policy.check
approval.requested
approval.granted
approval.denied
recovery.started
recovery.completed
evaluation.completed
artifact.created
sandbox.created
sandbox.destroyed
```

---

# 22. Storage Strategy

The system will generate large amounts of execution data. Use tiered storage.

## Hot

Recent execution metadata and events.

## Warm

Replayable execution bundles and recent traces.

## Cold

Long-retention snapshots and archives.

Use content-addressed storage where possible:

```text
sha256(content) → blob ID
```

This reduces duplicate storage for repeated artifacts and frames.

---

# 23. Determinism Strategy

Full determinism is impossible for many agent workloads, so the system should target **controlled replay**.

Capture:

- model responses
- tool outputs
- timestamps where relevant
- random seeds where available
- environment metadata
- relevant file state
- selected browser state
- workflow version
- agent version
- policy version

Replay modes should explicitly indicate what was mocked and what was live.

---

# 24. Observability

Use OpenTelemetry-compatible concepts where possible.

The platform should provide:

- traces
- spans
- event timelines
- tool metrics
- model metrics
- cost metrics
- token metrics
- latency
- failure rates
- execution graphs
- state diffs
- checkpoint timelines

The differentiator is that these observability objects remain attached to mutable execution state.

---

# 25. Evaluation System

Evaluation should work at multiple levels.

## Step-level evaluation

Was a tool call correct?

## Agent-level evaluation

Did the agent solve the task?

## Workflow-level evaluation

Did the complete workflow produce the desired outcome?

## Reliability evaluation

Did the execution respect cost, latency and policies?

## Regression evaluation

Did a new version become worse than baseline?

Evaluator types:

- deterministic assertions
- schema validation
- rule-based evaluators
- model-based evaluators
- human evaluations
- domain-specific evaluators

---

# 26. Replay Dataset System

Every production execution can become a test case.

```text
production execution
       ↓
sanitize
       ↓
label
       ↓
add to dataset
       ↓
version dataset
       ↓
run CI
```

Dataset features:

- tags
- source
- expected output
- evaluator definitions
- failure category
- severity
- customer/environment metadata
- privacy classification

---

# 27. Reliability Intelligence

Build an intelligence layer over historical execution data.

Questions it should answer:

- Which agent versions fail most?
- Which tools cause failures?
- Which model performs best for each task?
- Which prompts introduce regressions?
- Where is cost increasing?
- Which workflows need more checkpoints?
- What failures are correlated?
- Which failure modes can be automatically recovered?

Potential outputs:

```text
Failure cluster: tool_timeout
Occurrences: 4,291
Primary tool: browser.search
Affected workflows: 17
Recommended action: increase timeout + fallback provider
Estimated reduction: 31%
```

---

# 28. Automatic Model Routing

The platform can eventually use execution data to select the best model per task.

Example:

```text
Task: SQL generation

Model A
quality 91
cost 0.07
latency 1.2s

Model B
quality 95
cost 0.19
latency 2.9s

Policy:
quality >= 92
cost <= 0.10

Selected: Model A
```

Routing should optimize for a configurable utility function.

```text
utility = quality_weight * quality
        - cost_weight * cost
        - latency_weight * latency
        - risk_weight * risk
```

---

# 29. Automatic Recovery Optimization

Recovery strategies should be evaluated historically.

```text
failure_type: API_TIMEOUT

retry
success: 62%
median cost: $0.04

alternate_tool
success: 81%
median cost: $0.07

rollback + alternate_tool
success: 94%
median cost: $0.08
```

The platform can gradually learn the most reliable recovery policy for each failure class.

---

# 30. Human-in-the-Loop System

Human approval should be treated as a first-class workflow state.

```text
Agent
 ↓
Risk Engine
 ↓
HIGH RISK
 ↓
Approval Request
 ↓
Slack / Web / API
 ↓
Approve or Reject
 ↓
Resume execution from checkpoint
```

This is especially useful for:

- payments
- production deployment
- destructive database actions
- account deletion
- outbound communications
- legal/financial actions

---

# 31. API Design

Example REST endpoints:

```text
POST   /v1/executions
GET    /v1/executions/{id}
GET    /v1/executions/{id}/events
POST   /v1/executions/{id}/replay
POST   /v1/executions/{id}/fork
POST   /v1/executions/{id}/rollback
POST   /v1/executions/{id}/evaluate
GET    /v1/executions/{id}/diff

POST   /v1/agents
GET    /v1/agents/{id}
POST   /v1/agents/{id}/deploy

POST   /v1/workflows
GET    /v1/workflows/{id}
POST   /v1/workflows/{id}/runs

POST   /v1/datasets
POST   /v1/datasets/{id}/evaluate

GET    /v1/projects/{id}/metrics
GET    /v1/projects/{id}/failures
GET    /v1/projects/{id}/risk
```

---

# 32. SDK Design

Python example:

```python
from agent_vcr import VCR

vcr = VCR(project="support-agent")

@vcr.agent
def support_agent(ticket):
    ...

run = support_agent(ticket)

run.checkpoint("before_refund")

if run.failed:
    recovered = run.recover()
```

Replay:

```python
run = vcr.get_run("exec_123")
replay = run.replay(mode="ghost")
```

Fork:

```python
fork = run.fork(frame="frame_17")
fork.set("model", "new-model")
fork.resume()
```

Evaluation:

```python
result = run.evaluate("support_quality")
```

---

# 33. CLI

Target CLI:

```bash
vcr login
vcr init
vcr run agent.py
vcr record agent.py
vcr list runs
vcr inspect exec_123
vcr replay exec_123
vcr replay exec_123 --mode ghost
vcr fork exec_123 --frame 17
vcr diff exec_123 exec_456
vcr rollback exec_123 --checkpoint before_refund
vcr evaluate exec_123
vcr dataset create production-failures
vcr deploy agent.py
vcr logs
vcr workflows deploy support.yaml
```

---

# 34. Web Console

## Core pages

### Dashboard

- agent health
- active executions
- cost
- failures
- regressions
- risk

### Execution Explorer

```text
Execution
  ├── timeline
  ├── DAG
  ├── state
  ├── tool calls
  ├── model calls
  ├── artifacts
  ├── policies
  └── checkpoints
```

### Replay Studio

Side-by-side:

```text
Original Run          Forked Run
────────────────      ────────────────
Model A               Model B
Tool X                Tool X
Prompt v12            Prompt v13
checkpoint 17         checkpoint 17
failure               success
```

### Workflow Builder

Visual DAG editor, but reliability-first rather than generic no-code automation.

### Evaluation Lab

- datasets
- graders
- experiment comparison
- regression analysis

### Deployments

- versions
- environments
- rollout
- rollback
- health

---

# 35. Integrations

Prioritize:

## Models

- OpenAI-compatible APIs
- Anthropic
- Google
- OpenRouter
- LiteLLM
- local models

## Agent frameworks

- LangGraph
- CrewAI
- OpenHands
- custom Python agents
- custom JavaScript agents

## Tools

- MCP
- GitHub
- Slack
- databases
- HTTP APIs
- browsers
- shell

## Dev workflow

- GitHub Actions
- GitLab CI
- webhook providers
- issue trackers

The integration strategy should focus on becoming the reliability layer above existing ecosystems rather than replacing every ecosystem.

---

# 36. Security Architecture

Security must be part of the platform, not a dashboard checkbox.

Core controls:

- tenant isolation
- sandbox isolation
- RBAC
- SSO/SAML for enterprise
- audit logging
- encrypted data at rest
- encrypted transport
- secret isolation
- scoped credentials
- network policy
- tool policy
- retention policies
- PII redaction
- configurable data residency

---

# 37. Privacy and Data Controls

Customers need configurable capture policies.

```yaml
capture:
  model_inputs: redact
  model_outputs: encrypted
  secrets: never
  pii: redact
  filesystem: allowlist
  browser: metadata_only
```

Replay bundles should have explicit sensitivity classifications.

---

# 38. Multitenancy

Logical isolation:

```text
Organization
 ├── Project
 │    ├── Environment
 │    ├── Agent
 │    ├── Workflow
 │    └── Dataset
```

Every resource should carry a tenant boundary.

For enterprise, support dedicated storage and dedicated execution pools.

---

# 39. Reliability and Scaling

The cloud must scale independently in these dimensions:

- control plane requests
- execution workers
- sandbox compute
- event ingestion
- object storage
- evaluation jobs
- replay jobs
- recovery jobs

Execution workers should be stateless where possible.

The durable state belongs in external persistence.

---

# 40. Queue Architecture

Use separate queues for different workload classes:

```text
execution.high_priority
execution.standard
replay
evaluation
recovery
sandbox.provision
artifact.processing
notifications
```

This prevents expensive evaluation/replay workloads from starving live customer traffic.

---

# 41. Billing Architecture

Three revenue engines:

## SaaS subscription

Recurring plan fees.

## Execution infrastructure

Usage-based fees for hosted agents and workflows.

## Reliability intelligence

Premium features for evaluation, recovery, risk, optimization and enterprise controls.

Potential billing dimensions:

- executions
- compute seconds
- sandbox runtime
- event volume
- replay storage
- evaluation runs
- workflow runs
- seats

Avoid pricing solely on “number of logs,” because customers will perceive raw observability as a commodity.

---

# 42. Proposed Pricing

## Free

- local VCR
- open-source runtime
- limited cloud storage
- 1 project
- basic replay

Purpose: developer adoption.

## Pro

Target range: `$29–$49/month`

- cloud replay
- workflow runner
- dashboards
- evaluation
- Git integration
- team features

## Team

Target range: `$199–$499/month`

- larger execution volume
- RBAC
- shared workspaces
- policy engine
- Slack alerts
- advanced evaluations
- failure intelligence

## Enterprise

Target range: `$2k–$10k+/month`

- private deployments
- VPC
- SSO/SAML
- SCIM
- audit
- custom retention
- dedicated compute
- data residency
- SLA

Usage pricing can sit on top for hosted compute and high-volume workloads.

---

# 43. Open-Source Strategy

Keep the execution kernel open.

Open-source candidates:

- record
- replay
- checkpoint
- rollback
- fork
- local CLI
- core integrations

Cloud-only candidates:

- team collaboration
- hosted execution
- centralized policy management
- analytics
- enterprise auth
- managed sandboxes
- automatic recovery intelligence
- fleet management
- hosted datasets

This creates the classic open-core funnel:

```text
GitHub
 ↓
install VCR
 ↓
local adoption
 ↓
cloud sync
 ↓
team usage
 ↓
production deployment
 ↓
enterprise
```

---

# 44. Startup Moat

The moat should not depend on a single LLM provider.

## Moat 1: Execution format

A standard agent execution record and replay format.

## Moat 2: Historical execution data

The system accumulates real failure/recovery data.

## Moat 3: Reliability engine

Recovery decisions improve with history.

## Moat 4: Ecosystem integrations

VCR becomes embedded in agent frameworks and CI pipelines.

## Moat 5: Workflow state

Production workflows become difficult to migrate because execution history, policies, datasets and deployment state live together.

---

# 45. Competitive Positioning

| Category | Typical strength | Agent VCR strategy |
|---|---|---|
| Workflow automation | Build workflows | Integrate with workflows and own reliability |
| Observability | Trace and inspect | Add replay, rollback, intervention |
| Evals | Measure quality | Turn real runs into replayable regression suites |
| Durable execution | Persist workflows | Add agent-native state and replay semantics |
| Agent security | Prevent unsafe actions | Combine prevention with rollback/recovery |
| Agent frameworks | Build agent loops | Provide a runtime layer above them |

The positioning should remain:

> **Not another agent framework. Not another tracing dashboard. The reliability layer underneath autonomous agent systems.**

---

# 46. Example End-to-End Scenario

## Customer support refund agent

Input:

```text
Customer requests $650 refund.
```

SuperAgent plan:

```text
classify
→ retrieve order
→ inspect policy
→ calculate refund
→ verify
→ request approval
→ refund
→ send confirmation
```

Execution:

```text
frame 0  task received
frame 1  classified
frame 2  order retrieved
frame 3  policy retrieved
frame 4  refund calculated
frame 5  verifier passed
frame 6  human approval
frame 7  refund submitted
frame 8  confirmation sent
```

Suppose the refund API fails at frame 7.

VCR stores:

```text
checkpoint_before_refund
```

Recovery engine evaluates:

```text
retry same API           52%
use alternate provider   77%
rollback + alternate     94%
```

The safest strategy is selected.

The agent resumes from frame 6 without recomputing the previous reasoning or repeating all previous API calls.

---

# 47. Example Coding Agent Scenario

A coding agent is asked to implement OAuth.

It:

```text
inspect repo
→ edit config
→ install package
→ modify routes
→ run tests
→ accidentally overwrite config
```

VCR detects the failure checkpoint.

Developer opens Replay Studio:

```text
Original
frame 13 → bad config

Fork A
frame 12 → use alternate package

Fork B
frame 12 → alter prompt

Fork C
frame 12 → switch model
```

Each branch can be tested.

The winning branch becomes a regression fixture.

---

# 48. Agent Reliability Graph

Each customer project should have a graph:

```text
Agent
 ├── Version
 ├── Workflow
 ├── Model
 ├── Tools
 ├── Policies
 └── Dataset
       │
       └── Executions
              ├── Failures
              ├── Recoveries
              └── Evaluations
```

This graph enables root-cause analysis.

---

# 49. “Why Did My Agent Fail?” Query Engine

Natural-language debugging should eventually be a product surface.

Example:

> Why did yesterday’s invoice agent failure rate increase?

System response structure:

```text
Failure rate increased from 1.7% to 5.4%.

Primary cause:
Invoice parser tool timeout.

Secondary cause:
Model version changed.

Affected workflows:
12 / 47

Recommended actions:
1. Add parser fallback.
2. Increase timeout from 5s → 8s.
3. Replay 4,820 historical failures.
```

The key is that the answer should be backed by actual execution evidence.

---

# 50. Product-Led Growth Strategy

The open-source CLI should provide a magical first experience.

Developer journey:

```text
pip install ai-agent-vcr
        ↓
vcr run agent.py
        ↓
agent fails
        ↓
vcr replay last
        ↓
“holy shit, I can time travel”
        ↓
cloud login
        ↓
team sharing
```

The first “aha” should happen locally before signup.

---

# 51. Initial ICP

Do not target everyone.

Best first users:

### 1. AI coding agents

High mutation risk, easy demonstration of rollback.

### 2. Research agents

Long-running, expensive, tool-heavy, easy to replay.

### 3. Customer support agents

High volume, measurable success, human approval boundaries.

### 4. Internal enterprise agents

Need audit, policy, approvals and reliability.

Avoid general-purpose consumer chat as an initial market.

---

# 52. Best Initial Wedge

The strongest initial wedge is:

# “GitHub Actions for agent reliability”

Install:

```text
Agent VCR
```

Add to CI:

```text
replay production runs
→ compare agent version
→ detect regression
→ block merge
```

Then expand into:

```text
CI
 ↓
Observability
 ↓
Replay
 ↓
Deployment
 ↓
Hosted execution
 ↓
Recovery
```

---

# 53. Development Roadmap

## Phase 0 — Existing Core Stabilization

- clean public API
- execution schema
- checkpoint API
- replay API
- fork API
- state diff API
- documentation
- tests
- benchmark suite

## Phase 1 — Agent VCR 1.0

- stable Python package
- reliable filesystem rollback
- Ghost Replay
- integrations
- execution explorer
- replay bundle format

## Phase 2 — Agent CI

- GitHub Action
- replay datasets
- baseline comparisons
- regression reports
- test annotations

## Phase 3 — VCR Cloud

- authentication
- projects
- execution ingestion
- object storage
- web dashboard
- cloud replay
- team collaboration

## Phase 4 — SuperAgent Runtime

- planner
- tool router
- memory
- verifier
- policies
- budgets
- human approval

## Phase 5 — Workflow Engine

- DAG
- schedules
- triggers
- durable state
- retries
- compensation

## Phase 6 — Reliability Intelligence

- failure clustering
- risk scoring
- recovery engine
- model routing
- cost optimization

## Phase 7 — Agent Cloud

- hosted sandboxes
- deployments
- autoscaling
- secrets
- environments
- enterprise controls

---

# 54. Suggested Repository Structure

```text
agent-vcr/
├── packages/
│   ├── vcr-core/
│   ├── vcr-sdk/
│   ├── vcr-cli/
│   ├── vcr-integrations/
│   └── superagent-runtime/
│
├── services/
│   ├── api/
│   ├── scheduler/
│   ├── orchestrator/
│   ├── evaluator/
│   ├── recovery/
│   ├── ingest/
│   └── billing/
│
├── workers/
│   ├── execution/
│   ├── replay/
│   ├── evaluation/
│   └── sandbox/
│
├── web/
│   ├── dashboard/
│   ├── execution-studio/
│   ├── replay-studio/
│   └── workflow-builder/
│
├── schemas/
│   ├── execution/
│   ├── events/
│   ├── workflows/
│   └── policies/
│
├── infra/
│   ├── docker/
│   ├── kubernetes/
│   └── terraform/
│
└── docs/
```

---

# 55. API Boundary Principles

Keep these boundaries stable:

```text
Agent code
   ↕
VCR SDK
   ↕
Execution Kernel
   ↕
Event Interface
   ↕
Storage Interface
```

Avoid letting cloud-specific implementation details leak into the local SDK.

This keeps local-first adoption viable.

---

# 56. Local-First Architecture

The open-source tool should remain useful without the cloud.

```text
Developer Machine

Agent
 ↓
VCR
 ↓
local event store
 ↓
local snapshots
 ↓
local replay
```

Cloud becomes optional synchronization and collaboration.

This is strategically valuable because the core developer experience does not depend on network connectivity.

---

# 57. Cloud Sync

Sync should be incremental.

```text
local execution
 ↓
content-addressed chunks
 ↓
hash comparison
 ↓
upload only missing objects
 ↓
cloud index
```

Possible command:

```bash
vcr cloud sync exec_123
```

---

# 58. Deployment Model

Agent deployment should be versioned.

```text
Agent v42
 ↓
Policy v8
 ↓
Workflow v13
 ↓
Environment production
 ↓
Sandbox pool
```

Rollout strategies:

- canary
- blue/green
- percentage rollout
- shadow replay
- automatic rollback

---

# 59. Shadow Deployment

Before releasing an agent version:

```text
production traffic
      │
      ├──────────────► current agent
      │
      └──────────────► shadow agent
                           │
                        replay
                           │
                        compare
```

The new agent does not produce real external side effects until approved.

---

# 60. Reliability SLOs

Track:

### Execution availability
Percent of runs successfully completed.

### Recovery success
Percent of failures successfully recovered.

### Replay fidelity
Percent of replays that match expected captured behavior within defined tolerances.

### Mean time to recovery
Time from failure to successful continuation.

### Regression rate
Percent of releases introducing statistically meaningful degradation.

---

# 61. Startup Metrics

North-star metric:

# **Successful autonomous task completions**

Supporting metrics:

- weekly active agents
- executions per customer
- replay runs
- checkpoint count
- recovery success rate
- production workflows
- hosted compute usage
- cloud retention
- expansion revenue
- gross margin

Do not optimize for raw trace volume.

---

# 62. Unit Economics

Monitor:

```text
Revenue / execution
- compute cost
- storage cost
- evaluation cost
- support cost
= contribution margin
```

Ghost Replay should improve margins by reducing repeated external model/tool spend.

Hosted execution should have explicit compute quotas to prevent runaway agent costs.

---

# 63. Failure Safety

Critical rule:

> Never automatically retry an unknown irreversible side effect solely because a request failed.

Every action should declare or infer:

```text
risk
reversibility
idempotency
required approval
```

---

# 64. Idempotency

Tool operations should support idempotency keys where possible.

```text
execution_id
+ tool_call_id
+ operation_key
```

This protects against duplicate actions during retries and recovery.

---

# 65. Compensation Model

Where rollback is impossible, create compensating actions.

Example:

```text
create_invoice
     ↓
compensation = cancel_invoice
```

The workflow engine should know that the original side effect is not physically reversible but may be compensatable.

---

# 66. Agent Memory

Memory must also be checkpoint-aware.

Memory categories:

- working memory
- episodic memory
- semantic memory
- tool memory
- workflow state

A rollback should clearly define whether memory changes are restored.

```text
ROLLBACK_SCOPE:
  workflow_state: true
  working_memory: true
  semantic_memory: false
  external_db: compensation
```

---

# 67. Multi-Agent Coordination

SuperAgent can coordinate specialized agents.

```text
                    Supervisor
                         │
          ┌──────────────┼──────────────┐
          ▼              ▼              ▼
      Researcher      Coder          Verifier
          │              │              │
          └──────────────┼──────────────┘
                         ▼
                     Synthesizer
```

Every sub-agent receives its own execution context while the supervisor maintains the parent execution graph.

---

# 68. Parent/Child Executions

```text
parent execution
├── child execution A
├── child execution B
└── child execution C
```

This allows:

- independent retries
- child-level evaluation
- child-level rollback
- aggregate workflow reasoning

---

# 69. Agent-to-Agent Contracts

Sub-agents should communicate through typed contracts.

```json
{
  "task": "research_company",
  "input": {
    "company": "..."
  },
  "output_schema": {
    "type": "object",
    "required": ["summary", "sources"]
  }
}
```

This improves validation and recovery.

---

# 70. Versioning

Everything important must be versioned:

```text
agent
workflow
prompt
model configuration
tool definition
policy
evaluation
replay dataset
runtime
```

Every production execution records the full version graph.

---

# 71. Reproducibility Bundle

A single execution should be exportable as a portable bundle.

```text
execution.bundle/
├── manifest.json
├── execution.json
├── events/
├── checkpoints/
├── artifacts/
├── model_calls/
├── tool_calls/
├── policies/
└── evaluation/
```

This enables local debugging and long-term archival.

---

# 72. Example Reproduction Command

```bash
vcr reproduce exec_123 --bundle run.bundle
```

Output:

```text
Loaded execution exec_123
Agent version: v42
Workflow: support:v13
Replay mode: ghost

Frame 0  OK
Frame 1  OK
Frame 2  OK
Frame 3  DIFFERENCE

State diff:
- customer_tier: gold → silver
```

---

# 73. Marketplace Potential

Later, the ecosystem could support:

- reusable recovery policies
- evaluator packs
- workflow templates
- agent runtime plugins
- integrations
- sandbox adapters
- model routing policies

Do not build this early. It becomes useful only after the execution platform gains users.

---

# 74. API-as-a-Product Potential

The VCR execution API itself can become infrastructure.

Example:

```text
POST /execute
POST /checkpoint
POST /fork
POST /resume
POST /evaluate
POST /recover
```

Third-party agent platforms can use the reliability kernel without adopting the entire SuperAgent runtime.

This is potentially the strongest long-term platform strategy.

---

# 75. Strategic Architecture: Kernel vs Product

Separate the company into two layers.

## Reliability Kernel

```text
record
checkpoint
rollback
replay
fork
state diff
```

## Product Layer

```text
SuperAgent
Workflow
Cloud
CI
Deployments
Analytics
Recovery
Billing
```

This separation preserves the ability to sell the kernel as infrastructure later.

---

# 76. Why SuperAgent Belongs Above VCR

SuperAgent generates complex, branching execution.

VCR gives it:

- checkpoint semantics
- safe experimentation
- recovery
- state lineage
- replay
- observability

Therefore:

```text
SuperAgent = intelligence
VCR = memory + control + recovery
Cloud = infrastructure + collaboration
```

That is a clean architectural story.

---

# 77. “Agent GitHub” Long-Term Vision

Potential future product:

```text
                     AgentCloud
                         │
         ┌───────────────┼────────────────┐
         │               │                │
      Code Repo      Agent Repo       Execution Repo
         │               │                │
       GitHub          VCR              VCR Cloud
         │               │                │
         └───────────────┼────────────────┘
                         │
                     Deployments
```

The company eventually owns the lifecycle:

```text
build
→ test
→ replay
→ evaluate
→ deploy
→ observe
→ recover
```

---

# 78. What NOT to Build

Do not begin by building:

- generic drag-and-drop automation for everyone
- another ChatGPT UI
- another generic agent framework
- another logging dashboard
- a giant integration catalog
- consumer autonomous assistants
- generic vector database infrastructure

These areas already have enormous competition.

The wedge is reliability.

---

# 79. MVP Definition

A credible first commercial MVP should include only:

```text
1. VCR Core
2. Ghost Replay
3. Execution Explorer
4. Agent CI
5. Replay Datasets
6. Basic Cloud Sync
7. GitHub integration
8. SuperAgent runtime with checkpoints
```

Everything else can follow.

---

# 80. MVP User Flow

```text
pip install ai-agent-vcr
        ↓
vcr init
        ↓
wrap agent
        ↓
vcr run
        ↓
execution recorded
        ↓
agent fails
        ↓
vcr inspect
        ↓
vcr fork --frame 13
        ↓
modify prompt/model/tool
        ↓
vcr replay
        ↓
fix confirmed
        ↓
vcr dataset add
        ↓
GitHub Action
        ↓
future regression blocked
```

This is a complete developer loop.

---

# 81. Example Architecture for a SuperAgent App

```text
                    User Task
                       │
                       ▼
                 SuperAgent API
                       │
                       ▼
                    Planner
                       │
                Execution Graph
                       │
      ┌────────────────┼────────────────┐
      ▼                ▼                ▼
  Researcher        Executor         Browser
      │                │                │
      └────────────────┼────────────────┘
                       ▼
                    Verifier
                       │
                Policy / Risk
                       │
          ┌────────────┴────────────┐
          │                         │
       continue                  recover
          │                         │
          ▼                  rollback / fork
      checkpoint                  │
          │                       ▼
          └──────────────► resume execution
```

---

# 82. Technical Priorities

Priority order:

### P0

- execution correctness
- checkpoint correctness
- rollback correctness
- replay correctness
- data integrity

### P1

- framework integrations
- SDK ergonomics
- execution viewer
- Ghost Replay
- Agent CI

### P2

- cloud
- workflow engine
- SuperAgent
- recovery engine

### P3

- advanced intelligence
- automated optimization
- enterprise features
- marketplace

Never sacrifice P0 correctness for feature velocity.

---

# 83. Testing Strategy

Test categories:

## Unit tests

State transitions, event encoding, diffs.

## Integration tests

Filesystem, tool adapters, model adapters.

## Replay fidelity tests

Recorded run versus expected replay.

## Chaos tests

Inject:

- network failures
- tool crashes
- timeouts
- malformed outputs
- corrupted state
- partial writes

## Recovery tests

Verify that recovery never duplicates unsafe side effects.

## Concurrency tests

Parallel agent/workflow executions and forks.

---

# 84. Benchmark Suite

Create public benchmarks around:

- time to reproduce failure
- replay fidelity
- rollback latency
- snapshot overhead
- event ingestion throughput
- storage overhead
- recovery success
- CI regression detection

This can build technical credibility around the project.

---

# 85. Open-Source Community Strategy

Repositories/issues should encourage:

- adapter contributions
- evaluator contributions
- sandbox integrations
- replay plugins
- workflow examples

Good first contributions:

```text
vcr integration for X
vcr evaluator for Y
vcr sandbox provider for Z
```

---

# 86. Brand Direction

The concept should remain technical and infrastructure-oriented.

Possible headline:

> **Run autonomous agents like production software.**

Subheadline:

> Record every execution. Rewind failures. Replay experiments. Recover automatically.

Supporting phrase:

> **Time-travel infrastructure for autonomous AI.**

---

# 87. Fundraising Story

The pitch should not be:

> “We built another agent platform.”

Instead:

> Autonomous AI is moving from chat to execution. Agents are now modifying code, calling APIs, operating browsers, handling customer workflows and touching real systems. Existing tooling can observe these executions, but reliable recovery and reproducible state manipulation remain difficult. Agent VCR provides the execution layer for checkpointing, replaying, branching, evaluating and recovering autonomous agents.

Then show:

```text
AI Agents
     ↓
real actions
     ↓
real failures
     ↓
Agent VCR
     ↓
replay + recovery
```

---

# 88. The Company Flywheel

```text
Open Source VCR
      ↓
Developer adoption
      ↓
Recorded executions
      ↓
Cloud sync
      ↓
Replay datasets
      ↓
CI / regression testing
      ↓
Production deployments
      ↓
Failure data
      ↓
Recovery intelligence
      ↓
More successful agents
      ↓
Higher cloud usage
      ↓
Revenue
      ↓
More engineering + ecosystem
      ↓
More adoption
```

The most important compounding asset is not logs.

It is **structured execution history plus recovery outcomes**.

---

# 89. End-State Architecture

```text
                           AGENT RELIABILITY CLOUD

 ┌─────────────────────────────────────────────────────────────────────┐
 │                         CONTROL PLANE                              │
 │                                                                     │
 │ Projects │ Agents │ Workflows │ Policies │ Deployments │ Billing   │
 └────────────────────────────────┬────────────────────────────────────┘
                                  │
 ┌────────────────────────────────┴────────────────────────────────────┐
 │                         EXECUTION PLANE                            │
 │                                                                     │
 │ SuperAgent │ Workflow Engine │ Tool Gateway │ Recovery Engine      │
 └────────────────────────────────┬────────────────────────────────────┘
                                  │
 ┌────────────────────────────────┴────────────────────────────────────┐
 │                          VCR KERNEL                                │
 │                                                                     │
 │ Record │ Checkpoint │ Snapshot │ Replay │ Fork │ Diff │ Rollback   │
 └────────────────────────────────┬────────────────────────────────────┘
                                  │
 ┌────────────────────────────────┴────────────────────────────────────┐
 │                          SANDBOX PLANE                             │
 │                                                                     │
 │ Containers │ MicroVMs │ Browser │ Shell │ MCP │ Network │ Secrets │
 └────────────────────────────────┬────────────────────────────────────┘
                                  │
 ┌────────────────────────────────┴────────────────────────────────────┐
 │                         INTELLIGENCE PLANE                         │
 │                                                                     │
 │ Evals │ Failure Clusters │ Risk │ Recovery │ Routing │ Optimization │
 └────────────────────────────────┬────────────────────────────────────┘
                                  │
                         Historical Executions
                                  │
                         Reliability Dataset
```

---

# 90. Final Strategic Position

The strongest version of the company is not:

**Agent VCR + SuperAgent + automation + cloud.**

It is:

# **A reliability operating system for autonomous agents.**

Where:

```text
VCR
= state + replay + rollback

SuperAgent
= planning + intelligence + execution

Workflow Engine
= durable orchestration

Sandbox
= safe environment

Reliability Cloud
= observability + evaluation + history

Recovery Engine
= autonomous failure handling

Control Plane
= deployment + policy + governance
```

Together:

```text
BUILD
   ↓
TEST
   ↓
RECORD
   ↓
REPLAY
   ↓
EVALUATE
   ↓
DEPLOY
   ↓
OBSERVE
   ↓
RECOVER
   ↓
IMPROVE
   ↺
```

## North-star statement

> **Agent VCR is the infrastructure layer that makes autonomous AI reliable enough to run in production.**

## Core product primitive

> **Every agent execution is a versioned, replayable, forkable state machine.**

## Startup thesis

> **As agents become more autonomous, execution reliability becomes infrastructure. Agent VCR owns that layer.**

---

# Appendix A — Recommended Build Order

```text
1. Stabilize existing Agent VCR core
2. Standardize execution/event schemas
3. Make Ghost Replay excellent
4. Add execution viewer
5. Add fork + state diff UX
6. Ship GitHub Action / Agent CI
7. Add cloud sync
8. Build SuperAgent runtime on VCR
9. Add workflow orchestration
10. Add sandbox execution
11. Add recovery engine
12. Add hosted agent deployment
13. Add reliability intelligence
14. Add enterprise controls
```

# Appendix B — The One-Sentence Product Test

Every feature should answer at least one of these:

```text
Can I reproduce the agent?
Can I understand the failure?
Can I safely change the execution?
Can I prevent the regression?
Can I recover without restarting everything?
Can I deploy the agent safely?
```

If a proposed feature does none of these, it probably belongs in the enormous graveyard of “cool AI features nobody needed.”
