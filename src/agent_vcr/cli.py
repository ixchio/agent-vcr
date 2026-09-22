"""Command-line entry point for Agent VCR."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

CLAUDE_CODE_EVENTS = (
    "PreToolUse",
    "PostToolUse",
    "FileChanged",
    "Stop",
    "SessionEnd",
)


def build_claude_code_hooks() -> dict[str, list[dict[str, Any]]]:
    """Build a Claude Code hooks config that records lifecycle events to VCR."""
    hooks: dict[str, list[dict[str, Any]]] = {}
    for event in CLAUDE_CODE_EVENTS:
        entry: dict[str, Any] = {
            "hooks": [
                {
                    "type": "command",
                    "command": f"python -m agent_vcr.hooks.claude_code {event}",
                }
            ]
        }
        if event in {"PreToolUse", "PostToolUse"}:
            entry["matcher"] = "*"
        hooks[event] = [entry]
    return hooks


def write_claude_code_hooks(project_dir: str | Path = ".") -> Path:
    """Create or update .claude/settings.json with Agent VCR hooks."""
    root = Path(project_dir)
    claude_dir = root / ".claude"
    settings_path = claude_dir / "settings.json"
    claude_dir.mkdir(parents=True, exist_ok=True)

    settings: dict[str, Any] = {}
    if settings_path.exists() and settings_path.stat().st_size > 0:
        with open(settings_path) as f:
            loaded = json.load(f)
        if not isinstance(loaded, dict):
            raise ValueError(f"{settings_path} must contain a JSON object")
        settings = loaded

    hooks = settings.setdefault("hooks", {})
    if not isinstance(hooks, dict):
        raise ValueError(f"{settings_path} field 'hooks' must be a JSON object")

    for event, entries in build_claude_code_hooks().items():
        existing = hooks.setdefault(event, [])
        if not isinstance(existing, list):
            raise ValueError(f"{settings_path} hooks.{event} must be a JSON array")

        existing_commands = {
            hook.get("command")
            for entry in existing
            if isinstance(entry, dict)
            for hook in entry.get("hooks", [])
            if isinstance(hook, dict)
        }
        for entry in entries:
            command = entry["hooks"][0]["command"]
            if command not in existing_commands:
                existing.append(entry)

    with open(settings_path, "w") as f:
        json.dump(settings, f, indent=2)
        f.write("\n")

    return settings_path


def _run_tui(filepath: str) -> None:
    from agent_vcr.tui import VCRApp

    path = Path(filepath)
    if not path.exists():
        print(f"Error: File not found: {path}", file=sys.stderr)
        raise SystemExit(1)

    app = VCRApp(str(path))
    app.run()


def _cmd_bundle_export(args: argparse.Namespace) -> None:
    from agent_vcr.bundle import export_bundle

    dest = export_bundle(args.file, output=args.output)
    print(f"Bundle exported: {dest}")


def _cmd_bundle_import(args: argparse.Namespace) -> None:
    from agent_vcr.bundle import import_bundle

    out = import_bundle(args.bundle, dest_dir=args.dest_dir)
    print(f"Bundle imported: {out}")


def _cmd_bundle_verify(args: argparse.Namespace) -> None:
    import json as _json

    from agent_vcr.bundle import verify_bundle

    print(_json.dumps(verify_bundle(args.bundle), indent=2))


def _cmd_reproduce(args: argparse.Namespace) -> None:
    import json as _json

    from agent_vcr.bundle import reproduce

    report = reproduce(args.bundle)
    print(_json.dumps(report, indent=2))
    if not report.get("reproduced"):
        raise SystemExit(1)


def _parse_csv(value: str | None) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()] if value else []


def _cmd_dataset_create(args: argparse.Namespace) -> None:
    from agent_vcr.datasets import ReplayDataset

    ds = ReplayDataset.create(args.name, datasets_dir=args.datasets_dir)
    print(f"Dataset created: {args.name} (v{ds.version})")


def _cmd_dataset_add(args: argparse.Namespace) -> None:
    import json as _json

    from agent_vcr.datasets import Expectation, ReplayDataset
    from agent_vcr.player import VCRPlayer
    from agent_vcr.recorder import VCRRecorder

    player = VCRPlayer.load(args.vcr)
    session_id = player.session.session_id
    # Re-record the loaded frames into a fresh recorder so the golden cache
    # owns an identical, self-contained copy of this exact recording.
    rec = VCRRecorder(output_dir=args.golden_dir, auto_save=False)
    rec.start_session(session_id=f"{session_id}-golden")
    for i, frame in enumerate(player.frames):
        from agent_vcr.models import StateSerializer

        rec.record_step(
            node_name=frame.node_name,
            input_state=player.get_input_state(i),
            output_state=StateSerializer.deserialize(frame.output_state),
            metadata=frame.metadata,
            frame_type=frame.frame_type,
        )
    rec.save()

    expect = Expectation(
        max_cost_usd=args.max_cost,
        max_latency_ms=args.max_latency_ms,
        max_tokens=args.max_tokens,
        require_no_errors=not args.allow_errors,
        expect_state=_json.loads(args.expect_state) if args.expect_state else {},
        forbid_tools=_parse_csv(args.forbid_tools),
        require_checkpoints=_parse_csv(args.require_checkpoints),
    )
    ds = ReplayDataset.load(args.name, datasets_dir=args.datasets_dir)
    from agent_vcr.golden_cache import GoldenRunCache

    case = ds.add_case(
        args.task,
        rec,
        name=args.case_name,
        tags=_parse_csv(args.tags),
        expect=expect,
        golden_cache=GoldenRunCache(cache_dir=args.golden_dir),
    )
    print(f"Case added: {case.name} (dataset v{ds.version}, golden {case.fingerprint})")


def _cmd_dataset_list(args: argparse.Namespace) -> None:
    from agent_vcr.datasets import ReplayDataset

    for name in ReplayDataset.list_datasets(datasets_dir=args.datasets_dir):
        print(name)


def _cmd_dataset_show(args: argparse.Namespace) -> None:
    import json as _json

    from agent_vcr.datasets import ReplayDataset

    ds = ReplayDataset.load(args.name, datasets_dir=args.datasets_dir)
    print(_json.dumps({"name": ds.name, "version": ds.version,
                       "cases": [c.model_dump() for c in ds.cases()]}, indent=2))


def _cmd_dataset_remove(args: argparse.Namespace) -> None:
    from agent_vcr.datasets import ReplayDataset

    ds = ReplayDataset.load(args.name, datasets_dir=args.datasets_dir)
    if not ds.remove_case(args.case):
        print(f"Case not found: {args.case}", file=sys.stderr)
        raise SystemExit(2)
    print(f"Case removed: {args.case} (dataset v{ds.version})")


def _cmd_ci_run(args: argparse.Namespace) -> None:
    from pathlib import Path as _Path

    from agent_vcr.datasets import ReplayDataset
    from agent_vcr.golden_cache import GoldenRunCache
    from agent_vcr.regression import (
        emit_github_annotations,
        run_regression,
        write_github_summary,
    )

    ds = ReplayDataset.load(args.dataset, datasets_dir=args.datasets_dir)
    cache = GoldenRunCache(cache_dir=args.golden_dir)
    cand_dir = _Path(args.candidates)
    candidates = {p.stem: str(p) for p in sorted(cand_dir.glob("*.vcr"))} if cand_dir.exists() else {}
    report = run_regression(ds, candidates, cache, tags=_parse_csv(args.tags) or None)

    markdown = report.to_markdown()
    if args.markdown:
        _Path(args.markdown).write_text(markdown)
    if args.report:
        report.save(args.report)
    if not args.quiet:
        print(markdown)
    if not args.no_summary:
        write_github_summary(markdown)
    emit_github_annotations(report)

    breached = (
        report.failed > 0
        or report.missing > 0
        or (args.fail_on == "degraded" and report.degraded > 0)
    )
    if args.fail_on != "never" and breached:
        raise SystemExit(1)


def _cloud_client(args: argparse.Namespace) -> Any:
    from agent_vcr.cloud import CloudClient

    if getattr(args, "base_url", None) and getattr(args, "token", None) is not None:
        token = args.token
        if token == "-":
            token = sys.stdin.read().strip()
        return CloudClient(args.base_url, token or None)
    profile = getattr(args, "profile", "default")
    if getattr(args, "base_url", None):
        return CloudClient(args.base_url, None)
    return CloudClient.from_profile(profile)


def _cmd_cloud_login(args: argparse.Namespace) -> None:
    from agent_vcr.cloud import CloudClient

    token = args.token
    if args.token_stdin:
        token = sys.stdin.read().strip()
    if not token:
        raise SystemExit("Provide --token or --token-stdin")
    client = CloudClient(args.base_url, token)
    path = client.login(profile=args.profile)
    print(f"Logged in: {args.base_url} (profile {args.profile}, credentials {path})")


def _cmd_cloud_status(args: argparse.Namespace) -> None:
    import json as _json

    client = _cloud_client(args)
    print(_json.dumps(client.health(), indent=2))


def _cmd_cloud_projects(args: argparse.Namespace) -> None:
    import json as _json

    client = _cloud_client(args)
    print(_json.dumps(client.list_projects(), indent=2))


def _cmd_cloud_sessions(args: argparse.Namespace) -> None:
    import json as _json

    client = _cloud_client(args)
    print(_json.dumps(
        client.list_sessions(args.project, limit=args.limit, offset=args.offset,
                             search=args.search), indent=2))


def _cmd_cloud_push(args: argparse.Namespace) -> None:
    import json as _json

    client = _cloud_client(args)
    print(_json.dumps(
        client.upload_session(args.project, args.file, overwrite=args.overwrite), indent=2))


def _cmd_cloud_pull(args: argparse.Namespace) -> None:
    client = _cloud_client(args)
    dest = args.output or f"{args.session_id}.vcr"
    out = client.download_session(args.project, args.session_id, dest)
    print(f"Pulled: {out}")


def _cmd_cloud_replay(args: argparse.Namespace) -> None:
    import json as _json

    client = _cloud_client(args)
    print(_json.dumps(
        client.replay(args.project, args.session_id, mode=args.mode, task=args.task), indent=2))


def _cmd_superagent_run(args: argparse.Namespace) -> None:
    import importlib.util
    import json as _json
    from pathlib import Path as _Path

    from agent_vcr.recorder import VCRRecorder
    from agent_vcr.superagent import (
        BudgetTracker,
        CallbackApprover,
        Policy,
        SuperAgent,
        ToolGateway,
        ToolSpec,
    )

    plan = _json.loads(_Path(args.plan).read_text())
    gateway = ToolGateway()
    agents: dict = {}
    if args.handlers:
        spec = importlib.util.spec_from_file_location("vcr_handlers", args.handlers)
        if spec is None or spec.loader is None:
            raise SystemExit(f"Cannot load handlers file: {args.handlers}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        approval_set = set(getattr(mod, "APPROVAL_REQUIRED", []))
        for name, handler in getattr(mod, "TOOLS", {}).items():
            if isinstance(handler, ToolSpec):
                gateway.register(handler)
            else:
                gateway.register(ToolSpec(
                    name=name, handler=handler,
                    parameters=getattr(handler, "PARAMS", {}),
                    requires_approval=name in approval_set,
                ))
        agents = dict(getattr(mod, "AGENTS", {}))

    def _ask(request: Any) -> bool:
        try:
            answer = input(f"Approve {request.kind}:{request.ref} "
                           f"(step {request.step_id}, {request.reason})? [y/N] ")
        except EOFError:
            return False
        return answer.strip().lower() in ("y", "yes")

    policy = Policy(
        forbid_tools=_parse_csv(args.forbid_tools),
        approval_required=_parse_csv(args.approval_required),
        max_steps=args.max_steps,
    )
    budget = BudgetTracker(max_total_cost_usd=args.max_cost)
    agent = SuperAgent(
        gateway, agents=agents, policy=policy, budget=budget,
        approver=CallbackApprover(_ask, name="stdin"),
        recorder=VCRRecorder(output_dir=args.vcr_dir),
    )
    try:
        report = agent.run(args.goal or plan.get("goal", "superagent run"), plan,
                           tags=["cli"])
    finally:
        agent.shutdown()
    print(report.model_dump_json(indent=2))
    if report.status != "committed":
        raise SystemExit(1)


def _load_handlers_file(path: str) -> tuple[Any, dict]:
    import importlib.util

    from agent_vcr.superagent import ToolGateway, ToolSpec

    gateway = ToolGateway()
    agents: dict = {}
    if not path:
        return gateway, agents
    spec = importlib.util.spec_from_file_location("vcr_handlers", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Cannot load handlers file: {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    approval_set = set(getattr(mod, "APPROVAL_REQUIRED", []))
    for name, handler in getattr(mod, "TOOLS", {}).items():
        if isinstance(handler, ToolSpec):
            gateway.register(handler)
        else:
            gateway.register(ToolSpec(
                name=name, handler=handler,
                parameters=getattr(handler, "PARAMS", {}),
                requires_approval=name in approval_set))
    return gateway, dict(getattr(mod, "AGENTS", {}))


def _cmd_workflow_validate(args: argparse.Namespace) -> None:
    import json as _json
    from pathlib import Path as _Path

    from agent_vcr.workflows.models import WorkflowDef

    definition = WorkflowDef(**_json.loads(_Path(args.file).read_text()))
    from agent_vcr.workflows.engine import WorkflowEngine

    order = WorkflowEngine._topo_order(definition)
    print(f"Valid: {definition.name} ({len(definition.steps)} steps, order: {', '.join(order)})")


def _cmd_workflow_run(args: argparse.Namespace) -> None:
    import json as _json
    from pathlib import Path as _Path

    from agent_vcr.superagent import BudgetTracker, CallbackApprover, Policy
    from agent_vcr.workflows.engine import WorkflowEngine
    from agent_vcr.workflows.models import WorkflowDef
    from agent_vcr.workflows.store import RunStore, WorkflowRegistry

    def _ask(request: Any) -> bool:
        try:
            answer = input(f"Approve {request.kind}:{request.ref} "
                           f"(step {request.step_id}, {request.reason})? [y/N] ")
        except EOFError:
            return False
        return answer.strip().lower() in ("y", "yes")

    registry = WorkflowRegistry(args.workflows_dir)
    if args.file:
        definition = registry.save(
            WorkflowDef(**_json.loads(_Path(args.file).read_text())))
    else:
        definition = registry.get(args.workflow)
    gateway, agents = _load_handlers_file(args.handlers)
    engine = WorkflowEngine(
        gateway, agents=agents, vcr_dir=args.vcr_dir,
        run_store=RunStore(args.runs_dir),
        policy=Policy(forbid_tools=_parse_csv(args.forbid_tools)),
        budget=BudgetTracker(max_total_cost_usd=args.max_cost),
        approver=CallbackApprover(_ask, name="stdin"),
    )
    try:
        record = engine.start(definition, _json.loads(args.inputs or "{}"), tags=["cli"])
    finally:
        engine.shutdown()
    print(f"Run {record.run_id}: {record.status.value} "
          f"({len(record.completed_top_steps)} steps, session {record.session_id})")
    if record.error:
        print(f"Error: {record.error}")
    if record.status.value != "committed":
        raise SystemExit(1)


def _cmd_workflow_list(args: argparse.Namespace) -> None:
    import json as _json

    from agent_vcr.workflows.store import WorkflowRegistry

    print(_json.dumps(WorkflowRegistry(args.workflows_dir).list(), indent=2))


def _cmd_workflow_runs(args: argparse.Namespace) -> None:
    import json as _json

    from agent_vcr.workflows.store import RunStore

    print(_json.dumps(RunStore(args.runs_dir).list(workflow=args.workflow), indent=2))


def _cmd_workflow_show(args: argparse.Namespace) -> None:
    import json as _json

    from agent_vcr.workflows.store import RunStore

    print(_json.dumps(RunStore(args.runs_dir).load(args.run_id).model_dump(), indent=2))


def _cmd_workflow_resume(args: argparse.Namespace) -> None:
    from agent_vcr.workflows.engine import WorkflowEngine
    from agent_vcr.workflows.store import RunStore

    gateway, agents = _load_handlers_file(args.handlers)
    engine = WorkflowEngine(gateway, agents=agents, vcr_dir=args.vcr_dir,
                            run_store=RunStore(args.runs_dir))
    try:
        record = engine.resume(args.run_id)
    finally:
        engine.shutdown()
    print(f"Run {record.run_id}: {record.status.value}")
    if record.status.value != "committed":
        raise SystemExit(1)


def _cmd_workflow_schedule_put(args: argparse.Namespace) -> None:
    from agent_vcr.workflows.schedules import Scheduler, ScheduleSpec

    schedule = Scheduler(args.schedules_dir).put(ScheduleSpec(
        name=args.name, workflow=args.workflow, kind=args.kind,
        interval_s=args.interval_s, cron=args.cron or "", at=args.at or "",
        enabled=not args.disabled,
        inputs=(__import__("json").loads(args.inputs or "{}"))))
    print(f"Schedule {schedule.name}: {schedule.kind} -> {schedule.workflow}")


def _cmd_workflow_schedule_list(args: argparse.Namespace) -> None:
    import json as _json

    from agent_vcr.workflows.schedules import Scheduler

    print(_json.dumps([s.model_dump() for s in Scheduler(args.schedules_dir).all()], indent=2))


def _cmd_workflow_schedule_remove(args: argparse.Namespace) -> None:
    from agent_vcr.workflows.schedules import Scheduler

    if not Scheduler(args.schedules_dir).remove(args.name):
        raise SystemExit(f"No schedule {args.name!r}")


def _cmd_intel_failures(args: argparse.Namespace) -> None:
    import json as _json

    from agent_vcr.intelligence import cluster_sessions

    print(_json.dumps(
        [c.model_dump() for c in cluster_sessions(args.dir, limit=args.limit)], indent=2))


def _cmd_intel_risk(args: argparse.Namespace) -> None:

    from agent_vcr.intelligence import score_agent

    print(score_agent(args.dir, baseline_session=args.baseline).model_dump_json(indent=2))


def _cmd_intel_why(args: argparse.Namespace) -> None:
    from agent_vcr.intelligence import diagnose

    report = diagnose(args.session, siblings_dir=args.siblings_dir, run_id=args.run_id)
    print(report.model_dump_json(indent=2))
    if report.verdict == "failed":
        raise SystemExit(1)


def _cmd_intel_recover(args: argparse.Namespace) -> None:
    import json as _json

    from agent_vcr.intelligence import auto_recover_workflow, plan_recovery
    from agent_vcr.player import VCRPlayer

    player = VCRPlayer.load(args.session)
    checkpoints = [n for n in player.list_nodes() if n.startswith("checkpoint:")]
    plan = plan_recovery(player, run_id=args.run_id, has_checkpoint=bool(checkpoints))
    if not args.execute:
        print(plan.model_dump_json(indent=2))
        return
    if not args.run_id:
        raise SystemExit("--execute needs --run-id (workflow runs only)")
    from agent_vcr.workflows.engine import WorkflowEngine
    from agent_vcr.workflows.store import RunStore

    gateway, agents = _load_handlers_file(args.handlers)
    engine = WorkflowEngine(gateway, agents=agents, vcr_dir=args.vcr_dir,
                            run_store=RunStore(args.runs_dir))
    try:
        print(_json.dumps(auto_recover_workflow(args.run_id, engine), indent=2))
    finally:
        engine.shutdown()


def _cmd_intel_routes(args: argparse.Namespace) -> None:
    import json as _json

    from agent_vcr.intelligence import ModelRegistry, route

    registry = ModelRegistry().ingest_dir(args.dir)
    decision = route(registry, prefer=_parse_csv(args.prefer),
                     min_success=args.min_success, max_avg_cost=args.max_cost)
    print(_json.dumps({**decision.model_dump(), "stats": registry.to_dict()}, indent=2))


def _cmd_intel_costs(args: argparse.Namespace) -> None:
    from agent_vcr.intelligence import breakdown

    print(breakdown(args.dir).model_dump_json(indent=2))


def _cmd_workflow_tick(args: argparse.Namespace) -> None:
    import json as _json

    from agent_vcr.workflows.engine import WorkflowEngine
    from agent_vcr.workflows.schedules import Scheduler
    from agent_vcr.workflows.store import RunStore, WorkflowRegistry

    registry = WorkflowRegistry(args.workflows_dir)
    gateway, agents = _load_handlers_file(args.handlers)
    engine = WorkflowEngine(gateway, agents=agents, vcr_dir=args.vcr_dir,
                            run_store=RunStore(args.runs_dir))
    try:
        results = Scheduler(args.schedules_dir).run_pending(engine, registry.get)
    finally:
        engine.shutdown()
    print(_json.dumps(results, indent=2))


def main(argv: list[str] | None = None) -> None:
    """Run the Agent VCR CLI."""
    raw_args = list(argv if argv is not None else sys.argv[1:])
    if raw_args and raw_args[0] not in {"init", "tui", "bundle", "dataset", "ci", "cloud",
                                       "superagent", "workflow", "intel",
                                       "reproduce", "-h", "--help"}:
        _run_tui(raw_args[0])
        return

    parser = argparse.ArgumentParser(description="Agent VCR")
    subparsers = parser.add_subparsers(dest="command")

    init_parser = subparsers.add_parser("init", help="Initialize Agent VCR integrations")
    init_parser.add_argument(
        "--claude-code",
        action="store_true",
        help="Install Claude Code hooks into .claude/settings.json",
    )
    init_parser.add_argument(
        "--project-dir",
        default=".",
        help="Project directory to initialize (default: current directory)",
    )

    tui_parser = subparsers.add_parser("tui", help="Open the terminal debugger")
    tui_parser.add_argument("file", help="Path to the .vcr file to replay")

    bundle_parser = subparsers.add_parser("bundle", help="Replay bundle operations")
    bundle_sub = bundle_parser.add_subparsers(dest="bundle_command", required=True)

    exp = bundle_sub.add_parser("export", help="Export a .vcr file to a .vcrb bundle")
    exp.add_argument("file", help="Path to the .vcr file")
    exp.add_argument("-o", "--output", default=None, help="Destination .vcrb path")
    exp.set_defaults(func=_cmd_bundle_export)

    imp = bundle_sub.add_parser("import", help="Import a .vcrb bundle to a .vcr file")
    imp.add_argument("bundle", help="Path to the .vcrb bundle")
    imp.add_argument("-d", "--dest-dir", default=".", help="Extraction directory")
    imp.set_defaults(func=_cmd_bundle_import)

    ver = bundle_sub.add_parser("verify", help="Verify a .vcrb bundle without extracting")
    ver.add_argument("bundle", help="Path to the .vcrb bundle")
    ver.set_defaults(func=_cmd_bundle_verify)

    rep = subparsers.add_parser("reproduce", help="Verify + load a bundle, print report")
    rep.add_argument("bundle", help="Path to the .vcrb bundle")
    rep.set_defaults(func=_cmd_reproduce)

    ds_parser = subparsers.add_parser("dataset", help="Replay dataset operations")
    ds_sub = ds_parser.add_subparsers(dest="dataset_command", required=True)

    stores = argparse.ArgumentParser(add_help=False)
    stores.add_argument("--datasets-dir", default=".vcr/datasets")
    stores.add_argument("--golden-dir", default=".vcr/golden")

    ds_create = ds_sub.add_parser("create", help="Create an empty dataset", parents=[stores])
    ds_create.add_argument("name")
    ds_create.set_defaults(func=_cmd_dataset_create)

    ds_add = ds_sub.add_parser("add", help="Pin a .vcr recording as a dataset case", parents=[stores])
    ds_add.add_argument("name", help="Dataset name")
    ds_add.add_argument("--task", required=True)
    ds_add.add_argument("--vcr", required=True, help="Recording to pin as golden")
    ds_add.add_argument("--case-name", default=None)
    ds_add.add_argument("--tags", default=None, help="Comma-separated tags")
    ds_add.add_argument("--max-cost", type=float, default=None)
    ds_add.add_argument("--max-latency-ms", type=float, default=None)
    ds_add.add_argument("--max-tokens", type=int, default=None)
    ds_add.add_argument("--expect-state", default=None, help="JSON partial final state")
    ds_add.add_argument("--forbid-tools", default=None, help="Comma-separated tool names")
    ds_add.add_argument("--require-checkpoints", default=None, help="Comma-separated labels")
    ds_add.add_argument("--allow-errors", action="store_true")
    ds_add.set_defaults(func=_cmd_dataset_add)

    ds_list = ds_sub.add_parser("list", help="List datasets", parents=[stores])
    ds_list.set_defaults(func=_cmd_dataset_list)

    ds_show = ds_sub.add_parser("show", help="Show dataset cases", parents=[stores])
    ds_show.add_argument("name")
    ds_show.set_defaults(func=_cmd_dataset_show)

    ds_rm = ds_sub.add_parser("remove", help="Remove a case from a dataset", parents=[stores])
    ds_rm.add_argument("name")
    ds_rm.add_argument("case")
    ds_rm.set_defaults(func=_cmd_dataset_remove)

    ci_parser = subparsers.add_parser("ci", help="Agent CI regression checks")
    ci_sub = ci_parser.add_subparsers(dest="ci_command", required=True)

    ci_run = ci_sub.add_parser("run", help="Run dataset regression, print report", parents=[stores])
    ci_run.add_argument("--dataset", required=True)
    ci_run.add_argument("--candidates", required=True, help="Dir of <case>.vcr files")
    ci_run.add_argument("--tags", default=None, help="Comma-separated tag filter")
    ci_run.add_argument("--fail-on", choices=["degraded", "failed", "never"], default="failed")
    ci_run.add_argument("--report", default=None, help="Write JSON report here")
    ci_run.add_argument("--markdown", default=None, help="Write markdown report here")
    ci_run.add_argument("--quiet", action="store_true")
    ci_run.add_argument("--no-summary", action="store_true",
                        help="Skip $GITHUB_STEP_SUMMARY append")
    ci_run.set_defaults(func=_cmd_ci_run)

    cloud_parser = subparsers.add_parser("cloud", help="VCR Cloud sync")
    cloud_sub = cloud_parser.add_subparsers(dest="cloud_command", required=True)

    conn = argparse.ArgumentParser(add_help=False)
    conn.add_argument("--base-url", default=None, help="Cloud URL (else stored profile)")
    conn.add_argument("--token", default=None, help="API token ('-' reads stdin)")
    conn.add_argument("--profile", default="default")

    c_login = cloud_sub.add_parser("login", help="Verify + store cloud credentials")
    c_login.add_argument("base_url")
    c_login.add_argument("--token", default=None)
    c_login.add_argument("--token-stdin", action="store_true")
    c_login.add_argument("--profile", default="default")
    c_login.set_defaults(func=_cmd_cloud_login)

    c_status = cloud_sub.add_parser("status", help="Show server health", parents=[conn])
    c_status.set_defaults(func=_cmd_cloud_status)

    c_projects = cloud_sub.add_parser("projects", help="List projects", parents=[conn])
    c_projects.set_defaults(func=_cmd_cloud_projects)

    c_sessions = cloud_sub.add_parser("sessions", help="List project sessions", parents=[conn])
    c_sessions.add_argument("project")
    c_sessions.add_argument("--limit", type=int, default=100)
    c_sessions.add_argument("--offset", type=int, default=0)
    c_sessions.add_argument("--search", default=None)
    c_sessions.set_defaults(func=_cmd_cloud_sessions)

    c_push = cloud_sub.add_parser("push", help="Upload a .vcr recording", parents=[conn])
    c_push.add_argument("file")
    c_push.add_argument("--project", required=True)
    c_push.add_argument("--overwrite", action="store_true")
    c_push.set_defaults(func=_cmd_cloud_push)

    c_pull = cloud_sub.add_parser("pull", help="Download a session", parents=[conn])
    c_pull.add_argument("project")
    c_pull.add_argument("session_id")
    c_pull.add_argument("-o", "--output", default=None)
    c_pull.set_defaults(func=_cmd_cloud_pull)

    c_replay = cloud_sub.add_parser("replay", help="Server-side Ghost Replay", parents=[conn])
    c_replay.add_argument("project")
    c_replay.add_argument("session_id")
    c_replay.add_argument("--mode", default="full")
    c_replay.add_argument("--task", default=None)
    c_replay.set_defaults(func=_cmd_cloud_replay)

    sa_parser = subparsers.add_parser("superagent", help="SuperAgent runtime")
    sa_sub = sa_parser.add_subparsers(dest="superagent_command", required=True)
    sa_run = sa_sub.add_parser("run", help="Execute a plan JSON file")
    sa_run.add_argument("--plan", required=True, help="Plan JSON file")
    sa_run.add_argument("--handlers", default=None, help="Python file with TOOLS/AGENTS dicts")
    sa_run.add_argument("--goal", default=None)
    sa_run.add_argument("--vcr-dir", default=".vcr")
    sa_run.add_argument("--forbid-tools", default=None)
    sa_run.add_argument("--approval-required", default=None)
    sa_run.add_argument("--max-steps", type=int, default=None)
    sa_run.add_argument("--max-cost", type=float, default=None)
    sa_run.set_defaults(func=_cmd_superagent_run)

    wf_parser = subparsers.add_parser("workflow", help="Workflow engine")
    wf_dirs = argparse.ArgumentParser(add_help=False)
    wf_dirs.add_argument("--workflows-dir", default=".vcr/workflows")
    wf_dirs.add_argument("--runs-dir", default=".vcr/workflow_runs")
    wf_dirs.add_argument("--schedules-dir", default=".vcr/schedules")
    wf_dirs.add_argument("--vcr-dir", default=".vcr")
    wf_sub = wf_parser.add_subparsers(dest="workflow_command", required=True)

    wf_validate = wf_sub.add_parser("validate", help="Validate a workflow file")
    wf_validate.add_argument("file")
    wf_validate.set_defaults(func=_cmd_workflow_validate)

    wf_run = wf_sub.add_parser("run", help="Execute a workflow", parents=[wf_dirs])
    src = wf_run.add_mutually_exclusive_group(required=True)
    src.add_argument("--workflow", default=None, help="Registered workflow name")
    src.add_argument("--file", default=None, help="Workflow JSON file (registers it)")
    wf_run.add_argument("--handlers", default=None)
    wf_run.add_argument("--inputs", default="{}")
    wf_run.add_argument("--forbid-tools", default=None)
    wf_run.add_argument("--max-cost", type=float, default=None)
    wf_run.set_defaults(func=_cmd_workflow_run)

    wf_list = wf_sub.add_parser("list", help="List workflows", parents=[wf_dirs])
    wf_list.set_defaults(func=_cmd_workflow_list)

    wf_runs = wf_sub.add_parser("runs", help="List runs", parents=[wf_dirs])
    wf_runs.add_argument("--workflow", default=None)
    wf_runs.set_defaults(func=_cmd_workflow_runs)

    wf_show = wf_sub.add_parser("show", help="Show a run", parents=[wf_dirs])
    wf_show.add_argument("run_id")
    wf_show.set_defaults(func=_cmd_workflow_show)

    wf_resume = wf_sub.add_parser("resume", help="Resume a run", parents=[wf_dirs])
    wf_resume.add_argument("run_id")
    wf_resume.add_argument("--handlers", default=None)
    wf_resume.set_defaults(func=_cmd_workflow_resume)

    wf_sched = wf_sub.add_parser("schedule", help="Schedules")
    wf_sched_sub = wf_sched.add_subparsers(dest="schedule_command", required=True)
    sched_put = wf_sched_sub.add_parser("put", parents=[wf_dirs])
    sched_put.add_argument("--name", required=True)
    sched_put.add_argument("--workflow", required=True)
    sched_put.add_argument("--kind", choices=["interval", "cron", "at"], default="interval")
    sched_put.add_argument("--interval-s", type=float, default=3600.0)
    sched_put.add_argument("--cron", default=None)
    sched_put.add_argument("--at", default=None)
    sched_put.add_argument("--inputs", default="{}")
    sched_put.add_argument("--disabled", action="store_true")
    sched_put.set_defaults(func=_cmd_workflow_schedule_put)
    sched_list = wf_sched_sub.add_parser("list", parents=[wf_dirs])
    sched_list.set_defaults(func=_cmd_workflow_schedule_list)
    sched_rm = wf_sched_sub.add_parser("remove", parents=[wf_dirs])
    sched_rm.add_argument("--name", required=True)
    sched_rm.set_defaults(func=_cmd_workflow_schedule_remove)
    sched_tick = wf_sched_sub.add_parser("tick", help="Run due schedules once", parents=[wf_dirs])
    sched_tick.add_argument("--handlers", default=None)
    sched_tick.set_defaults(func=_cmd_workflow_tick)

    intel_parser = subparsers.add_parser("intel", help="Reliability intelligence")
    intel_sub = intel_parser.add_subparsers(dest="intel_command", required=True)

    in_fail = intel_sub.add_parser("failures", help="Cluster failures across sessions")
    in_fail.add_argument("--dir", default=".vcr")
    in_fail.add_argument("--limit", type=int, default=500)
    in_fail.set_defaults(func=_cmd_intel_failures)

    in_risk = intel_sub.add_parser("risk", help="Score agent risk")
    in_risk.add_argument("--dir", default=".vcr")
    in_risk.add_argument("--baseline", default=None)
    in_risk.set_defaults(func=_cmd_intel_risk)

    in_why = intel_sub.add_parser("why", help="Diagnose a session")
    in_why.add_argument("session")
    in_why.add_argument("--siblings-dir", default=None)
    in_why.add_argument("--run-id", default="")
    in_why.set_defaults(func=_cmd_intel_why)

    in_rec = intel_sub.add_parser("recover", help="Plan (or execute) recovery")
    in_rec.add_argument("session")
    in_rec.add_argument("--run-id", default="")
    in_rec.add_argument("--execute", action="store_true",
                        help="Auto-resume interrupted workflow runs")
    in_rec.add_argument("--handlers", default=None)
    in_rec.add_argument("--runs-dir", default=".vcr/workflow_runs")
    in_rec.add_argument("--vcr-dir", default=".vcr")
    in_rec.set_defaults(func=_cmd_intel_recover)

    in_routes = intel_sub.add_parser("routes", help="Route a model from evidence")
    in_routes.add_argument("--dir", default=".vcr")
    in_routes.add_argument("--prefer", default=None)
    in_routes.add_argument("--min-success", type=float, default=0.8)
    in_routes.add_argument("--max-cost", type=float, default=None)
    in_routes.set_defaults(func=_cmd_intel_routes)

    in_costs = intel_sub.add_parser("costs", help="Cost breakdown + waste")
    in_costs.add_argument("--dir", default=".vcr")
    in_costs.set_defaults(func=_cmd_intel_costs)

    args = parser.parse_args(raw_args)

    if args.command == "init":
        project_dir = Path(args.project_dir)
        if not project_dir.is_dir():
            parser.error(f"project directory does not exist: {project_dir}")
        vcr_dir = project_dir / ".vcr"
        vcr_dir.mkdir(exist_ok=True)
        print(f"Agent VCR workspace ready: {vcr_dir}")
        if args.claude_code:
            path = write_claude_code_hooks(project_dir)
            print(f"Claude Code hooks installed: {path}")
        return

    if args.command == "tui":
        _run_tui(args.file)
        return

    if hasattr(args, "func"):
        from agent_vcr.cloud import CloudError

        try:
            args.func(args)
        except SystemExit:
            raise
        except (CloudError, KeyError, FileNotFoundError, PermissionError) as err:
            print(f"Error: {err}", file=sys.stderr)
            raise SystemExit(1) from err
        except (ValueError, json.JSONDecodeError) as err:
            print(f"Error: {err}", file=sys.stderr)
            raise SystemExit(1) from err
        return

    parser.print_help()


if __name__ == "__main__":
    main()
