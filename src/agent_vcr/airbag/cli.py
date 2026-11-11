"""``vcr undo`` / ``vcr redo`` / ``vcr rewind`` / ``vcr airbag ...`` commands."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any

from agent_vcr.airbag.classify import classify
from agent_vcr.airbag.core import Airbag, Policy
from agent_vcr.airbag.store import AirbagError, RestoreResult, Snapshot

LEVEL_ICON = {"read": "·", "write": "✎", "destructive": "🛟", "irreversible": "⚠"}


def register(subparsers: Any) -> None:
    """Add the airbag commands to the main ``vcr`` parser."""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--project-dir", default=".", help="Project to act on (default: .)")
    common.add_argument("--json", action="store_true", help="Machine-readable output")

    undo = subparsers.add_parser(
        "undo", parents=[common],
        help="Undo the agent's last file-changing action (airbag snapshot)",
    )
    undo.add_argument("steps", nargs="?", type=int, default=1, help="How many actions back")
    undo.add_argument("--dry-run", action="store_true", help="Show what would change")
    undo.set_defaults(func=cmd_undo)

    redo = subparsers.add_parser("redo", parents=[common], help="Redo the last `vcr undo`")
    redo.set_defaults(func=cmd_redo)

    rewind = subparsers.add_parser(
        "rewind", parents=[common], help="List airbag snapshots, or restore one: vcr rewind 12",
    )
    rewind.add_argument("snapshot", nargs="?", help="Snapshot number (e.g. 12 or #12)")
    rewind.add_argument("-n", "--limit", type=int, default=20)
    rewind.set_defaults(func=cmd_rewind)

    bag = subparsers.add_parser("airbag", help="Snapshot-before-every-tool-call safety net")
    sub = bag.add_subparsers(dest="airbag_command", required=True)

    inst = sub.add_parser("install", parents=[common], help="Install airbag hooks for agents")
    inst.add_argument("agents", nargs="*", help="claude, codex, gemini, cursor, opencode, ... "
                                               "(default: every agent detected)")
    inst.add_argument("--all", action="store_true", help="Install for every supported agent")
    inst.add_argument("--user", action="store_true",
                      help="Install in the user-level config instead of the project")
    inst.set_defaults(func=cmd_install)

    uninst = sub.add_parser("uninstall", parents=[common], help="Remove airbag hooks")
    uninst.add_argument("agents", nargs="*")
    uninst.add_argument("--user", action="store_true")
    uninst.set_defaults(func=cmd_uninstall)

    status = sub.add_parser("status", parents=[common], help="Show protection status")
    status.set_defaults(func=cmd_status)

    log = sub.add_parser("log", parents=[common], help="Timeline of protected agent actions")
    log.add_argument("-n", "--limit", type=int, default=20)
    log.set_defaults(func=cmd_rewind_list)

    check = sub.add_parser("check", parents=[common],
                           help='Classify a command without running it: vcr airbag check "rm -rf ."')
    check.add_argument("command", nargs=argparse.REMAINDER)
    check.set_defaults(func=cmd_check)

    snap = sub.add_parser("snapshot", parents=[common], help="Take a manual snapshot now")
    snap.add_argument("-m", "--message", default="manual snapshot")
    snap.set_defaults(func=cmd_snapshot)

    allow = sub.add_parser("allow", parents=[common],
                           help="Approve the last stopped command for one retry (10 minutes)")
    allow.set_defaults(func=cmd_allow)

    gc = sub.add_parser("gc", parents=[common], help="Delete old snapshots")
    gc.add_argument("--keep-days", type=float, default=7.0)
    gc.add_argument("--keep-last", type=int, default=50)
    gc.set_defaults(func=cmd_gc)

    run = sub.add_parser(
        "run", parents=[common],
        help="Run any agent with periodic snapshots (for agents without hooks): "
             "vcr airbag run -- aider",
    )
    run.add_argument("--interval", type=float, default=2.0, help="Seconds between snapshots")
    run.add_argument("argv", nargs=argparse.REMAINDER)
    run.set_defaults(func=cmd_run)

    hook = sub.add_parser("hook", help="(internal) hook entry point used by agent configs")
    hook.add_argument("agent")
    hook.add_argument("event", nargs="?", default="pre")
    hook.add_argument("option", nargs="?", default="")
    hook.set_defaults(func=cmd_hook)


COMMANDS = frozenset({"undo", "redo", "rewind", "airbag"})


def _bag(args: argparse.Namespace) -> Airbag:
    try:
        return Airbag(args.project_dir, policy=Policy.from_env())
    except AirbagError as err:
        _fail(str(err))
        raise  # unreachable


def _fail(message: str, code: int = 1) -> None:
    print(f"Error: {message}", file=sys.stderr)
    raise SystemExit(code)


def _ago(ts: float) -> str:
    delta = max(0.0, time.time() - ts)
    for unit, size in (("d", 86400), ("h", 3600), ("m", 60)):
        if delta >= size:
            return f"{int(delta // size)}{unit} ago"
    return f"{int(delta)}s ago"


def _summary(result: RestoreResult) -> str:
    parts = []
    if result.written:
        parts.append(f"restored {len(result.written)} file{'s' * (len(result.written) != 1)}")
    if result.deleted:
        parts.append(f"removed {len(result.deleted)} created since")
    if result.failed:
        parts.append(f"{len(result.failed)} could not be restored: {', '.join(result.failed[:5])}")
    if result.kept:
        parts.append(f"left {len(result.kept)} unprotected path(s) untouched "
                     f"({', '.join(result.kept[:3])}{'…' if len(result.kept) > 3 else ''})")
    return ", ".join(parts) or "files already matched the snapshot"


def _print_changes(result: RestoreResult, limit: int = 12) -> None:
    for rel in result.written[:limit]:
        print(f"    + {rel}")
    for rel in result.deleted[:limit]:
        print(f"    - {rel}")
    extra = len(result.written) + len(result.deleted) - 2 * limit
    if extra > 0:
        print(f"    … and {extra} more")


def cmd_undo(args: argparse.Namespace) -> None:
    bag = _bag(args)
    plan = bag.plan_undo(args.steps)
    if plan is None:
        print("Nothing to undo: no recorded agent action changed files since the last snapshot.")
        return
    action, snap = plan
    if args.dry_run:
        current = bag.store.latest()
        changes = bag.store.diff(current, snap) if current else []
        print(f"Would undo: {action.get('command') or action.get('tool')}  "
              f"({action.get('agent')}, {_ago(float(action.get('time', 0)))}) → snapshot {snap.short}")
        for status, rel in changes[:30]:
            # diff(current -> snapshot): A = comes back, D = will be removed (matches `vcr undo`)
            print(f"    {'+' if status == 'A' else '-' if status == 'D' else '~'} {rel}")
        if not changes:
            print("    (no file differences)")
        return
    result = bag.undo(args.steps)
    if result is None:
        print("Nothing to undo.")
        return
    action, restore = result
    if args.json:
        print(json.dumps({"undone": action, "snapshot": restore.target.seq,
                          "written": restore.written, "deleted": restore.deleted,
                          "failed": restore.failed}, default=str))
        return
    print(f"↩ Undid: {action.get('command') or action.get('tool')}  "
          f"({action.get('agent')}, {_ago(float(action.get('time', 0)))})")
    print(f"  {_summary(restore)} — back to snapshot {restore.target.short}.")
    _print_changes(restore)
    print("  Changed your mind? `vcr redo`")


def cmd_redo(args: argparse.Namespace) -> None:
    try:
        result = _bag(args).redo()
    except KeyError:
        _fail("the snapshot to redo to was deleted by `vcr airbag gc`")
        return
    if result is None:
        print("Nothing to redo.")
        return
    print(f"↪ Redid: {_summary(result)} — back to snapshot {result.target.short}.")
    _print_changes(result)


def _parse_seq(raw: str) -> int:
    try:
        return int(raw.lstrip("#"))
    except ValueError:
        _fail(f"not a snapshot number: {raw!r}", 2)
        raise


def cmd_rewind(args: argparse.Namespace) -> None:
    if not args.snapshot:
        cmd_rewind_list(args)
        return
    bag = _bag(args)
    seq = _parse_seq(args.snapshot)
    try:
        result = bag.rewind(seq)
    except KeyError as err:
        _fail(str(err.args[0] if err.args else err))
        return
    print(f"⏪ Rewound to snapshot #{seq}: {_summary(result)}.")
    _print_changes(result)
    print("  Undo this rewind with `vcr redo`.")


def cmd_rewind_list(args: argparse.Namespace) -> None:
    bag = _bag(args)
    actions = bag.actions()[-args.limit:]
    snaps = {s.seq: s for s in bag.store.snapshots()}
    if args.json:
        print(json.dumps(actions, default=str))
        return
    if not actions:
        print(f"No protected agent actions yet for {bag.root}.")
        print("Install hooks with `vcr airbag install`, or take one now: `vcr airbag snapshot`.")
        return
    print(f"Airbag timeline for {bag.root}  (newest last; restore with `vcr rewind <#>`)")
    print(f"  {'snap':>5}  {'when':>8}  {'agent':<12} {'':2}{'command'}")
    for action in actions:
        seq = action.get("snapshot")
        snap: Snapshot | None = snaps.get(int(seq)) if seq is not None else None
        icon = LEVEL_ICON.get(str(action.get("level")), "?")
        text = action.get("command") or ", ".join(action.get("paths") or []) or action.get("tool")
        mark = f"#{seq}" if snap else "  -"
        print(f"  {mark:>5}  {_ago(float(action.get('time', 0))):>8}  "
              f"{str(action.get('agent', ''))[:12]:<12} {icon} {str(text)[:70]}")
    state = bag.store.read_state()
    if state.get("undo_pos") is not None:
        print("  (you are in an undone state — `vcr redo` to go forward)")


def cmd_check(args: argparse.Namespace) -> None:
    command = " ".join(args.command).strip()
    if command.startswith("-- "):
        command = command[3:]
    if not command:
        _fail('usage: vcr airbag check "<shell command>"', 2)
    from agent_vcr.airbag.store import find_project_root

    cwd = os.path.abspath(args.project_dir)
    verdict = classify(command, cwd=cwd, root=find_project_root(cwd))
    if args.json:
        print(json.dumps({"level": verdict.level, "reasons": verdict.reasons,
                          "categories": verdict.categories, "undoable": verdict.undoable}))
        return
    icon = LEVEL_ICON[verdict.level]
    explain = {
        "read": "read-only — no snapshot needed",
        "write": "changes files — snapshot first, undo with `vcr undo`",
        "destructive": "destroys project data — snapshot first, undo with `vcr undo`",
        "irreversible": "cannot be undone by a snapshot — the airbag asks a human first",
    }[verdict.level]
    print(f"{icon} {verdict.level.upper()}: {explain}")
    for reason in verdict.reasons:
        print(f"   • {reason}")


def cmd_snapshot(args: argparse.Namespace) -> None:
    bag = _bag(args)
    snap = bag.store.snapshot(label=args.message, meta={"agent": "user", "manual": True})
    bag.store.log_event({"kind": "action", "agent": "user", "tool": "snapshot",
                         "command": args.message, "level": "write", "snapshot": snap.seq})
    print(f"📸 Snapshot {snap.short} of {bag.root} ({snap.files} files protected).")


def cmd_allow(args: argparse.Namespace) -> None:
    pending = _bag(args).approve_pending()
    if pending is None:
        print("No stopped command is waiting for approval (approvals expire after 10 minutes).")
        return
    print(f"✅ Approved once: {pending.get('command')}")
    print("   The agent may retry this exact command in the next 10 minutes.")


def cmd_gc(args: argparse.Namespace) -> None:
    bag = _bag(args)
    removed = bag.store.gc(keep_days=args.keep_days, keep_last=args.keep_last)
    print(f"Removed {removed} old snapshot(s); store uses "
          f"{bag.store.disk_usage() / 1e6:.1f} MB ({bag.store.dir}).")


def cmd_status(args: argparse.Namespace) -> None:
    from agent_vcr.airbag.agents import detect_installed

    bag = _bag(args)
    info = bag.store.info()
    policy = bag.policy
    if args.json:
        print(json.dumps({**info, "policy": policy.__dict__,
                          "hooks": detect_installed(bag.root)}, default=str))
        return
    print(f"Airbag for {info['root']}")
    print(f"  enabled: {policy.enabled}   mode: {policy.mode}   "
          f"irreversible commands: {policy.on_irreversible}")
    print(f"  snapshots: {info['snapshots']}   store: {info['store']}")
    hooks = detect_installed(bag.root)
    if hooks:
        for agent, where in hooks.items():
            print(f"  ✓ {agent}: {where}")
    else:
        print("  ✗ no agent hooks installed — run `vcr airbag install`")


def cmd_install(args: argparse.Namespace) -> None:
    from agent_vcr.airbag.agents import ADAPTERS, detect_available, install

    names = list(args.agents)
    if args.all:
        names = sorted(ADAPTERS)
    elif not names:
        names = detect_available()
        if not names:
            _fail("no supported agent found on PATH; name one: "
                  f"vcr airbag install {' | '.join(sorted(ADAPTERS))}")
    unknown = [n for n in names if n not in ADAPTERS]
    if unknown:
        _fail(f"unknown agent(s) {', '.join(unknown)}; choose from {', '.join(sorted(ADAPTERS))}",
              2)
    root = os.path.abspath(args.project_dir)
    for name in names:
        try:
            where, note = install(name, root, user=args.user)
        except (OSError, ValueError) as err:
            print(f"  ✗ {name}: {err}", file=sys.stderr)
            continue
        print(f"  ✓ {ADAPTERS[name].title}: {where}")
        if note:
            print(f"      {note}")
    print("Airbag armed. Try: vcr airbag check \"rm -rf .\"   ·   undo anything: vcr undo")


def cmd_uninstall(args: argparse.Namespace) -> None:
    from agent_vcr.airbag.agents import ADAPTERS, uninstall

    names = list(args.agents) or sorted(ADAPTERS)
    root = os.path.abspath(args.project_dir)
    for name in names:
        if name not in ADAPTERS:
            _fail(f"unknown agent {name!r}", 2)
        removed = uninstall(name, root, user=args.user)
        if removed:
            print(f"  ✓ removed {ADAPTERS[name].title} hooks: {removed}")


def cmd_run(args: argparse.Namespace) -> None:
    from agent_vcr.airbag.runner import run_with_snapshots

    argv = list(args.argv)
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv:
        _fail("usage: vcr airbag run -- <agent command ...>", 2)
    raise SystemExit(run_with_snapshots(argv, args.project_dir, interval=args.interval))


def cmd_hook(args: argparse.Namespace) -> None:
    from agent_vcr.airbag.hook import main as hook_main

    raise SystemExit(hook_main([args.agent, args.event, args.option]))
