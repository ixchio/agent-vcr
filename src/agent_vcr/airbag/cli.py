"""``vcr undo`` / ``vcr redo`` / ``vcr rewind`` / ``vcr airbag ...`` commands."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import time
from typing import Any

from agent_vcr.airbag.classify import classify
from agent_vcr.airbag.core import Airbag, Policy, describe_action
from agent_vcr.airbag.store import AirbagError, RestoreResult, Snapshot

LEVEL_ICON = {"read": "·", "write": "✎", "destructive": "🛟", "irreversible": "⚠"}
LEVEL_EXPLAIN = {
    "read": "read-only — no snapshot needed",
    "write": "changes files — snapshot first, undo with `vcr undo`",
    "destructive": "destroys project data — snapshot first, undo with `vcr undo`",
    "irreversible": "cannot be undone by a snapshot — the airbag asks a human first",
}
NOTHING_TO_UNDO = (
    "Nothing to undo: no recorded agent action changed files since the last snapshot."
)


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

    bis = subparsers.add_parser(
        "bisect", parents=[common],
        help="Find the agent action that broke your tests: vcr bisect -- pytest -q",
    )
    bis.add_argument("--good", help="Snapshot where the test passes "
                                    "(default: the oldest recorded action)")
    bis.add_argument("--bad", help="Snapshot where it fails (default: the current state)")
    bis.add_argument("--timeout", type=float, default=300.0,
                     help="Seconds per test run; a timeout counts as a failure (default: 300)")
    bis.add_argument("--copy-ignored", action="append", nargs="?", const="", metavar="DIR",
                     help="Symlink the project's ignored dependency directories (node_modules, "
                          ".venv, venv, env, vendor, .tox, bower_components) into each "
                          "temporary checkout; repeat with DIR to link another top-level "
                          "directory too (put `--` before the test command)")
    bis.add_argument("argv", nargs=argparse.REMAINDER,
                     help="Test command; one quoted string runs through the shell")
    bis.set_defaults(func=cmd_bisect)

    doc = subparsers.add_parser(
        "doctor", parents=[common],
        help="Check that the airbag really protects you, and print the exact fix",
        description="Checks Python and the `vcr` launchers on PATH, git, every agent's hooks "
                    "(installed where, interpreter still there, one-time trust steps), and "
                    "whether this folder is protected. Exit code 0 when healthy, 1 when not.",
    )
    doc.add_argument("--fix", action="store_true",
                     help="Fix what is safe to fix automatically: rewrite stale hook commands "
                          "and move project-level hooks to the user level")
    doc.set_defaults(func=cmd_doctor)

    bag = subparsers.add_parser("airbag", help="Snapshot-before-every-tool-call safety net")
    sub = bag.add_subparsers(dest="airbag_command", required=True)

    inst = sub.add_parser(
        "install", parents=[common], help="Install airbag hooks for agents",
        description="Install airbag hooks. By default they go into each agent's user-level "
                    "config, so one install protects every project and nothing is written "
                    "into your repository. Hooks from an older project-level install are "
                    "moved to the user level.",
    )
    inst.add_argument("agents", nargs="*", help="claude, codex, gemini, cursor, opencode, ... "
                                               "(default: every agent detected)")
    inst.add_argument("--all", action="store_true", help="Install for every supported agent")
    where = inst.add_mutually_exclusive_group()
    where.add_argument("--user", action="store_true",
                       help="User-level config: protects every project (the default)")
    where.add_argument("--project", action="store_true",
                       help="Per-developer config inside the project instead (Claude Code: "
                            ".claude/settings.local.json); do not commit it")
    inst.add_argument("--statusline", action="store_true",
                      help="Claude Code: also show the airbag in the status line, only if no "
                           "statusLine is configured yet")
    inst.set_defaults(func=cmd_install)

    uninst = sub.add_parser("uninstall", parents=[common],
                            help="Remove airbag hooks (user and project level)")
    uninst.add_argument("agents", nargs="*")
    which = uninst.add_mutually_exclusive_group()
    which.add_argument("--user", action="store_true", help="Only the user-level hooks")
    which.add_argument("--project", action="store_true", help="Only the project's hooks")
    uninst.set_defaults(func=cmd_uninstall)

    status = sub.add_parser("status", parents=[common], help="Show protection status")
    status.set_defaults(func=cmd_status)

    line = sub.add_parser(
        "statusline", help="One-line airbag status for Claude Code's statusLine (reads its "
                           "JSON on stdin)",
    )
    line.set_defaults(func=cmd_statusline)

    doctor = sub.add_parser("doctor", parents=[common],
                            help="Check the installation and print the exact fix (= vcr doctor)")
    doctor.add_argument("--fix", action="store_true",
                        help="Fix what is safe to fix automatically")
    doctor.set_defaults(func=cmd_doctor)

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

    db = sub.add_parser("db", help="Local Postgres snapshots: list | snapshot | restore")
    db_sub = db.add_subparsers(dest="db_command", required=True)
    db_list = db_sub.add_parser("list", parents=[common],
                                help="Configured local databases and their dumps")
    db_list.set_defaults(func=cmd_db_list)
    db_snap = db_sub.add_parser("snapshot", parents=[common],
                                help="Dump the configured database(s) now")
    db_snap.add_argument("--db", help="Only this database (name)")
    db_snap.add_argument("-m", "--message", default="manual database snapshot")
    db_snap.set_defaults(func=cmd_db_snapshot)
    db_restore = db_sub.add_parser(
        "restore", parents=[common],
        help="Put the database(s) back to snapshot N (files untouched): vcr airbag db restore 12",
    )
    db_restore.add_argument("snapshot", help="Snapshot number (e.g. 12 or #12)")
    db_restore.add_argument("--db", help="Only this database (name)")
    db_restore.set_defaults(func=cmd_db_restore)

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


COMMANDS = frozenset({"undo", "redo", "rewind", "bisect", "airbag", "doctor"})


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
    if result.databases:
        parts.append(f"restored database {', '.join(result.databases)}")
    if result.failed:
        parts.append(f"{len(result.failed)} could not be restored: {', '.join(result.failed[:5])}")
    if result.kept:
        parts.append(f"left {len(result.kept)} unprotected path(s) untouched "
                     f"({', '.join(result.kept[:3])}{'…' if len(result.kept) > 3 else ''})")
    return ", ".join(parts) or "files already matched the snapshot"


def _print_changes(result: RestoreResult, limit: int = 12) -> None:
    # Same signs as `vcr undo --dry-run`: + comes back, ~ changed back, - removed.
    modified = set(result.modified)
    for rel in result.written[:limit]:
        print(f"    {'~' if rel in modified else '+'} {rel}")
    for rel in result.deleted[:limit]:
        print(f"    - {rel}")
    extra = len(result.written) + len(result.deleted) - 2 * limit
    if extra > 0:
        print(f"    … and {extra} more")
    for name in result.databases:
        print(f"    ⟲ database {name}")
    for note in result.notes:
        print(f"    note: {note}")


def cmd_undo(args: argparse.Namespace) -> None:
    bag = _bag(args)
    if args.dry_run:
        preview = bag.preview_undo(args.steps)
        if preview is None:
            print(NOTHING_TO_UNDO)
            return
        action, snap, changes = preview
        print(f"Would undo: {describe_action(action, bag.root)}  "
              f"({action.get('agent')}, {_ago(float(action.get('time', 0)))}) → snapshot {snap.short}")
        for status, rel in changes[:30]:
            # diff(current -> snapshot): A = comes back, D = will be removed (matches `vcr undo`)
            print(f"    {'+' if status == 'A' else '-' if status == 'D' else '~'} {rel}")
        if not changes:
            print("    (no file differences)")
        for entry in bag.undo_database_plan(action):
            print(f"    ⟲ database {entry.get('name')} (from the dump at snapshot "
                  f"#{entry.get('snapshot')})")
        return
    result = bag.undo(args.steps)
    if result is None:
        print(NOTHING_TO_UNDO)
        return
    action, restore = result
    if args.json:
        print(json.dumps({"undone": action, "snapshot": restore.target.seq,
                          "written": restore.written, "deleted": restore.deleted,
                          "databases": restore.databases, "failed": restore.failed},
                         default=str))
        return
    print(f"↩ Undid: {describe_action(action, bag.root)}  "
          f"({action.get('agent')}, {_ago(float(action.get('time', 0)))})")
    print(f"  {_summary(restore)} — back to snapshot {restore.target.short}.")
    _print_changes(restore)
    print("  Changed your mind? `vcr redo`")


def _actions_to_redo(bag: Airbag) -> list[dict[str, Any]]:
    """The actions the next `vcr redo` brings back (empty after a rewind past the log)."""
    state = bag.store.read_state()
    stack = state.get("redo") or []
    if not stack:
        return []
    actions = bag.actions()
    cur, nxt = state.get("undo_pos"), stack[-1].get("pos")
    lo = len(actions) if cur is None else max(0, min(int(cur), len(actions)))
    hi = len(actions) if nxt is None else max(0, min(int(nxt), len(actions)))
    return actions[lo:hi]


def cmd_redo(args: argparse.Namespace) -> None:
    bag = _bag(args)
    redone = _actions_to_redo(bag)
    try:
        result = bag.redo()
    except KeyError:
        _fail("the snapshot to redo to was deleted by `vcr airbag gc`")
        return
    if result is None:
        print("Nothing to redo.")
        return
    if len(redone) == 1:
        action = redone[0]
        print(f"↪ Redid: {describe_action(action, bag.root)}  "
              f"({action.get('agent')}, {_ago(float(action.get('time', 0)))})")
    elif redone:
        print(f"↪ Redid {len(redone)} actions, up to: {describe_action(redone[-1], bag.root)}")
    else:
        print("↪ Redid the last undo.")
    print(f"  {_summary(result)} — now at snapshot {result.target.short}.")
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
        text = describe_action(action, bag.root)[:70]
        if action.get("databases"):
            marker = f"  [db: {', '.join(str(d) for d in action['databases'])}]"
            text = text[:max(20, 70 - len(marker))] + marker
        mark = f"#{seq}" if snap else "  -"
        print(f"  {mark:>5}  {_ago(float(action.get('time', 0))):>8}  "
              f"{str(action.get('agent', ''))[:12]:<12} {icon} {text}")
    state = bag.store.read_state()
    if state.get("undo_pos") is not None:
        print("  (you are in an undone state — `vcr redo` to go forward)")


def cmd_check(args: argparse.Namespace) -> None:
    command = " ".join(args.command).strip()
    if command.startswith("-- "):
        command = command[3:]
    if not command:
        _fail('usage: vcr airbag check "<shell command>"', 2)
    from agent_vcr.airbag.scope import resolve

    cwd = os.path.abspath(args.project_dir)
    scope = resolve(cwd)
    root = scope.root
    verdict = classify(command, cwd=cwd, root=root)
    if not scope.protected and verdict.level == "destructive":
        # What the hook does here: no snapshot, so a human decides (see airbag.scope).
        verdict.raise_to("irreversible", scope.unprotected_reason, "unprotected")
    if verdict.db_targets and scope.protected:
        from agent_vcr.airbag.core import downgrade_for_databases
        from agent_vcr.airbag.databases import plan_dumps

        plan = plan_dumps(verdict, root, command)
        if plan.problem:
            verdict.reasons.append(plan.problem)
        elif plan.databases and plan.caveat:
            names = ", ".join(db.dbname for db in plan.databases)
            verdict.reasons.append(f"local database {names} is dumped first, but {plan.caveat}")
        elif plan.databases and plan.covers_all and verdict.level == "irreversible":
            downgrade_for_databases(verdict, [db.dbname for db in plan.databases],
                                    note="dumped with pg_dump before it runs")
    if args.json:
        print(json.dumps({"level": verdict.level, "reasons": verdict.reasons,
                          "categories": verdict.categories, "undoable": verdict.undoable,
                          "protected": scope.protected}))
        return
    icon = LEVEL_ICON[verdict.level]
    print(f"{icon} {verdict.level.upper()}: {LEVEL_EXPLAIN[verdict.level]}")
    for reason in verdict.reasons:
        print(f"   • {reason}")


def cmd_snapshot(args: argparse.Namespace) -> None:
    bag = _bag(args)
    snap = bag.manual_snapshot(args.message)
    print(f"📸 Snapshot {snap.short} of {bag.root} ({snap.files} files protected).")


def _pending_time(bag: Airbag) -> float:
    pending = bag.store.read_state().get("pending")
    return float(pending.get("time", 0)) if isinstance(pending, dict) else -1.0


def cmd_allow(args: argparse.Namespace) -> None:
    from agent_vcr.airbag.scope import GuardStore, resolve
    from agent_vcr.airbag.store import ShadowStore

    policy = Policy.from_env()
    scope = resolve(args.project_dir)
    # Commands stopped in unprotected folders wait in the shared guard store.
    bags = [Airbag(scope.root, policy=policy, store=GuardStore(scope.root), scope=scope)]
    if scope.protected:
        with contextlib.suppress(AirbagError):
            bags.append(Airbag(scope.root, policy=policy, store=ShadowStore(scope.root)))
    bag = max(bags, key=_pending_time)
    pending = bag.approve_pending()
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


def _short_path(path: str | os.PathLike[str]) -> str:
    text = str(path)
    home = os.path.expanduser("~")
    return "~" + text[len(home):] if text.startswith(home + os.sep) else text


def _roots(project_dir: str) -> tuple[Any, list[Any]]:
    """The scope of ``project_dir`` and the roots whose project-level hooks count."""
    from pathlib import Path

    from agent_vcr.airbag.scope import resolve

    here = Path(os.path.abspath(project_dir))
    scope = resolve(here)
    return scope, [here] + ([scope.root] if scope.root != here else [])


def protection_line(scope: Any) -> str:
    if scope.protected:
        return f"✓ protected: {scope.root} ({scope.why})"
    return (f"✗ not protected: {scope.root} ({scope.why}). Nothing is snapshotted here; "
            f"commands that would destroy data are stopped for a human. To protect it, "
            f"{scope.opt_in_hint}.")


def cmd_status(args: argparse.Namespace) -> None:
    from agent_vcr.airbag.agents import find_hooks
    from agent_vcr.airbag.store import ShadowStore, state_home

    scope, roots = _roots(args.project_dir)
    policy = Policy.from_env()
    info: dict[str, Any] = {"root": str(scope.root), "store": str(state_home()),
                            "snapshots": 0, "latest": None}
    if scope.protected:
        with contextlib.suppress(AirbagError):
            info = ShadowStore(scope.root).info()
    sites = {s.path: s for root in roots for s in find_hooks(root)}.values()
    hooks: dict[str, str] = {}
    for site in sorted(sites, key=lambda s: s.scope != "project"):
        hooks.setdefault(site.agent, str(site.path))
    if args.json:
        print(json.dumps({**info, "policy": policy.__dict__, "hooks": hooks,
                          "protected": scope.protected, "protection": scope.to_dict(),
                          "hook_sites": [s.to_dict() for s in sites]}, default=str))
        return
    print(f"Airbag for {info['root']}")
    print(f"  {protection_line(scope)}")
    print(f"  enabled: {policy.enabled}   mode: {policy.mode}   "
          f"irreversible commands: {policy.on_irreversible}")
    print(f"  snapshots: {info['snapshots']}   store: {info['store']}")
    if not sites:
        print("  ✗ no agent hooks installed — run `vcr airbag install`")
    for site in sites:
        mark = "✗" if site.missing_interpreters else "!" if site.scope == "project" else "✓"
        extra = ""
        if site.missing_interpreters:
            extra = "  (its Python is gone: `vcr doctor --fix`)"
        elif site.scope == "project":
            extra = "  (in the project: `vcr doctor --fix` moves it to user level)"
        print(f"  {mark} {site.agent} ({site.scope}): {_short_path(site.path)}{extra}")
    print("  Protected folders: git repositories, and folders with a .vcr/ folder "
          "(`mkdir .vcr`). Elsewhere nothing is snapshotted, and destructive commands are "
          "stopped for a human.")


def cmd_install(args: argparse.Namespace) -> None:
    from agent_vcr.airbag import agents as ag
    from agent_vcr.airbag.doctor import migrate

    user = not args.project
    scope, roots = _roots(args.project_dir)
    leftovers = sorted({s.agent for root in roots for s in ag.find_hooks(root)
                        if s.scope == "project"}) if user else []
    names = list(args.agents)
    if args.all:
        names = sorted(ag.ADAPTERS)
    elif not names:
        names = ag.detect_available()
        if not names and not leftovers:
            _fail("no supported agent found on PATH; name one: "
                  f"vcr airbag install {' | '.join(sorted(ag.ADAPTERS))}")
    # Project-level hooks from an older install are always moved: left in the repository,
    # they break every teammate's agent once committed.
    names += [n for n in leftovers if n not in names]
    unknown = [n for n in names if n not in ag.ADAPTERS]
    if unknown:
        _fail(f"unknown agent(s) {', '.join(unknown)}; choose from "
              f"{', '.join(sorted(ag.ADAPTERS))}", 2)
    root = roots[0]
    installed: list[dict[str, Any]] = []
    failed: list[dict[str, str]] = []
    moved: list[str] = []
    statusline: str | None = None
    for name in names:
        adapter = ag.ADAPTERS[name]
        wants_line = args.statusline and name == "claude"
        try:
            if user and name in leftovers:
                moved.extend(migrate(name, roots))
            where, note = ag.install(name, root, user=user, statusline=wants_line)
        except (OSError, ValueError) as err:
            failed.append({"agent": name, "error": str(err)})
            continue
        if wants_line:
            current = ag._read_json(where).get("statusLine")
            statusline = "added" if ag._is_our_statusline(current) else "kept-existing"
        installed.append({"agent": name, "title": adapter.title, "path": str(where),
                          "scope": "user" if user else "project", "note": note,
                          "required_step": adapter.required_step})
    steps = [i["required_step"] for i in installed if i["required_step"]]
    if args.json:
        print(json.dumps({"scope": "user" if user else "project", "installed": installed,
                          "failed": failed, "moved": moved, "required_steps": steps,
                          "protection": scope.to_dict(), "statusline": statusline}))
        if failed and not installed:
            raise SystemExit(1)
        return
    for item in failed:
        print(f"  ✗ {item['agent']}: {item['error']}", file=sys.stderr)
    if not installed:
        raise SystemExit(1)
    n = len(installed)
    if user:
        print(f"🛟 Airbag armed for {n} agent{'s' * (n != 1)}, user level: every project, "
              "nothing written into your repository.")
    else:
        print(f"🛟 Airbag armed for {n} agent{'s' * (n != 1)} in this project only.")
    width = max(len(i["title"]) for i in installed)
    for item in installed:
        print(f"  ✓ {item['title']:<{width}}  {_short_path(item['path'])}")
        if item["note"]:
            print(f"      {item['note']}")
    if not user:
        print("  ⚠ These files are per-developer (they hold this machine's Python path): "
              "don't commit them; add them to your .gitignore.")
    if moved:
        print("Moved from the project to your user-level config (commit the removal, so "
              "teammates' agents stop calling a path from your machine):")
        for line in moved:
            print(f"  ↪ {line}")
    if statusline == "added":
        print("Status line: Claude Code now shows the airbag (`vcr airbag statusline`).")
    elif statusline == "kept-existing":
        print("Status line: you already have one, left as is; see `vcr airbag statusline` in "
              "docs/airbag.md to add it yourself.")
    print("What your agent sees: nothing for safe commands; a one-line notice "
          "(\"snapshot #N saved … Undo with `vcr undo`\") before destructive ones; a "
          "confirmation, or a block, before anything a snapshot can't undo.")
    print(f"Protected: git repositories and folders with a .vcr/ folder. Here: "
          f"{protection_line(scope)}")
    if steps:
        print("Required once:")
        for step in steps:
            print(f"  • {step}")
    print('Try it: vcr airbag check "rm -rf ."   ·   undo anything: vcr undo   ·   '
          "health check: vcr doctor")


def cmd_uninstall(args: argparse.Namespace) -> None:
    from agent_vcr.airbag.agents import ADAPTERS, uninstall_paths

    names = list(args.agents) or sorted(ADAPTERS)
    _, roots = _roots(args.project_dir)
    scopes = [True] if args.user else [False] if args.project else [False, True]
    for name in names:
        if name not in ADAPTERS:
            _fail(f"unknown agent {name!r}", 2)
        removed: list[Any] = []
        for user in scopes:
            for root in roots[:1] if user else roots:
                removed.extend(p for p in uninstall_paths(name, root, user=user)
                               if p not in removed)
        for path in removed:
            print(f"  ✓ removed {ADAPTERS[name].title} hooks: {_short_path(path)}")


def cmd_statusline(args: argparse.Namespace) -> None:
    from agent_vcr.airbag.statusline import main as statusline_main

    raise SystemExit(statusline_main([]))


def cmd_doctor(args: argparse.Namespace) -> None:
    from agent_vcr.airbag import doctor

    fixed = doctor.fix(args.project_dir) if args.fix else []
    report = doctor.diagnose(args.project_dir)
    report.fixed = fixed
    if args.json:
        print(json.dumps(report.to_dict(), default=str))
    else:
        if args.fix:
            for line in fixed or ["nothing to fix automatically"]:
                print(f"🔧 {line}")
        print(doctor.format_report(report))
    raise SystemExit(0 if report.healthy else 1)


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


# ── vcr bisect ───────────────────────────────────────────────────────────

_OUTCOME = {"pass": "✓ pass", "fail": "✗ fail", "timeout": "⏱ timeout", "error": "✗ error"}


def cmd_bisect(args: argparse.Namespace) -> None:
    from pathlib import Path

    from agent_vcr.airbag.bisect import TestRun, bisect

    argv = list(args.argv)
    if argv and argv[0] == "--":
        argv = argv[1:]
    if not argv:
        _fail("usage: vcr bisect [--good N] [--bad N] -- <test command ...>", 2)
    if args.timeout <= 0:
        _fail("--timeout must be positive", 2)
    good = _parse_seq(args.good) if args.good else None
    bad = _parse_seq(args.bad) if args.bad else None
    bag = _bag(args)

    def progress(run: TestRun, note: str) -> None:
        label = f"#{run.seq}{' (current)' if note == 'current' else ''}"
        timing = "same files as an earlier run" if run.cached else f"{run.seconds:.1f}s"
        extra = f" after {args.timeout:g}s (counted as a failure)" if run.outcome == "timeout" \
            else ""
        print(f"  {label:<14} {_OUTCOME.get(run.outcome, run.outcome)}{extra}   ({timing})",
              flush=True)

    if not args.json:
        print(f"🔍 Bisecting the airbag timeline of {bag.root}")
        print(f"   test: {' '.join(argv)}   (each run in a temporary checkout)")
    links = list(args.copy_ignored or [])
    try:
        report = bisect(bag, argv, good=good, bad=bad, timeout=args.timeout,
                        copy_ignored=bool(links), extra_links=[d for d in links if d],
                        progress=None if args.json else progress)
    except ValueError as err:
        _fail(str(err), 2)
        return
    rewind = report.rewind
    if rewind:
        here = Path.cwd().resolve()
        if here != bag.root and bag.root not in here.parents:
            rewind += f" --project-dir {bag.root}"
            report.rewind = rewind
    if args.json:
        print(json.dumps(report.to_dict(), default=str))
        raise SystemExit(0 if report.status == "found" else 1)
    if report.status != "found":
        print(f"\n{report.message}")
        last = report.runs[-1] if report.runs else None
        if last is not None and last.output and report.status == "good-fails":
            print("  last lines of that run:")
            for line in last.output.splitlines()[-12:]:
                print(f"    {line}")
        raise SystemExit(2 if report.status == "invalid" else 1)

    event = report.culprit or {}
    when = float(event.get("time", 0))
    stamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(when)) if when else "?"
    print(f"\n🎯 {report.message}")
    if report.culprit_kind == "action":
        who = " · ".join(str(x) for x in (event.get("agent"), event.get("tool")) if x)
        print(f"   {who} · {stamp} ({_ago(when)})")
        if event.get("agent") == "user":
            print("   (a manual save point: the change came from you or a process "
                  "without airbag hooks)")
    else:
        print(f"   {stamp} ({_ago(when)})")
    after = f"#{report.after}{' (current state)' if report.after_is_current else ''}"
    print(f"   snapshot #{report.before} passes → {after} fails   "
          f"({report.test_runs} test run{'s' * (report.test_runs != 1)})")
    if report.timeouts:
        print(f"   note: {report.timeouts} run(s) timed out after {args.timeout:g}s and "
              "counted as failures")
    print("   files it changed (+ added, - deleted, ~ modified):")
    for status, rel in report.changes[:30]:
        print(f"     {'+' if status == 'A' else '-' if status == 'D' else '~'} {rel}")
    if len(report.changes) > 30:
        print(f"     … and {len(report.changes) - 30} more")
    if not report.changes:
        print("     (no file differences)")
    failing = next((r for r in report.runs if r.seq == report.after), None)
    if failing is not None and failing.output.strip():
        print("   failing run (last lines):")
        tail = failing.output.splitlines()[-10:]
        while tail and not tail[0].strip():
            tail.pop(0)
        for line in tail:
            print(f"     {line}")
    print(f"   Go back to just before it:  {rewind}")


# ── vcr airbag db ────────────────────────────────────────────────────────

def cmd_db_list(args: argparse.Namespace) -> None:
    from agent_vcr.airbag import databases as dbs

    bag = _bag(args)
    configs, problems = dbs.load_databases(bag.root)
    dumps = [(s, e) for s in bag.store.snapshots() for e in (s.meta.get("databases") or [])
             if isinstance(e, dict)]
    if args.json:
        print(json.dumps({
            "databases": [db.info() for db in configs], "refused": problems,
            "dumps": [{"snapshot": s.seq, "time": s.time, "label": s.label, **e}
                      for s, e in dumps],
            "pg_dump": dbs.pg_tool("pg_dump"), "pg_restore": dbs.pg_tool("pg_restore"),
        }, default=str))
        return
    if not configs and not problems:
        print("No local databases configured. Add one with "
              f"{dbs.ENV_VAR}=postgresql://localhost/app or {dbs.CONFIG_FILE}.")
    for db in configs:
        print(f"  ✓ {db.display}")
    for problem in problems:
        print(f"  ✗ refused: {problem}")
    missing = dbs.missing_tools("pg_dump", "pg_restore")
    if missing:
        print(f"  ⚠ {missing}")
    if dumps:
        print("Dumps (restore with `vcr undo`, `vcr rewind <#>` or `vcr airbag db restore <#>`):")
        for snap, entry in dumps:
            size = float(entry.get("bytes") or 0)
            print(f"  #{snap.seq:<5} {str(entry.get('name')):<16} {size / 1024:8.1f} KB  "
                  f"{_ago(snap.time):>8}  {snap.label}")
    elif configs:
        print("No dumps yet (taken before destructive SQL, or `vcr airbag db snapshot`).")


def cmd_db_snapshot(args: argparse.Namespace) -> None:
    bag = _bag(args)
    try:
        snap, names = bag.snapshot_databases(only=args.db, label=args.message)
    except AirbagError as err:
        _fail(str(err))
        return
    if args.json:
        print(json.dumps({"snapshot": snap.seq, "databases": names}))
        return
    print(f"🗄  Snapshot {snap.short}: dumped database {', '.join(names)}. "
          f"Restore with `vcr airbag db restore {snap.seq}`.")


def cmd_db_restore(args: argparse.Namespace) -> None:
    bag = _bag(args)
    seq = _parse_seq(args.snapshot)
    try:
        result = bag.restore_databases(seq, only=args.db)
    except KeyError as err:
        _fail(str(err.args[0] if err.args else err))
        return
    if result is None:
        _fail(f"no database dump at or after snapshot #{seq}: the database has not been "
              "dumped since then (see `vcr airbag db list`)")
        return
    if args.json:
        print(json.dumps({"snapshot": seq, "before": result.before.seq,
                          "databases": result.databases, "failed": result.failed}))
        raise SystemExit(0 if not result.failed else 1)
    if result.databases:
        print(f"⏪ Restored database {', '.join(result.databases)} to snapshot #{seq}. "
              f"Undo with `vcr airbag db restore {result.before.seq}`.")
    for problem in result.failed:
        print(f"  ✗ {problem}", file=sys.stderr)
    if result.failed:
        raise SystemExit(1)
