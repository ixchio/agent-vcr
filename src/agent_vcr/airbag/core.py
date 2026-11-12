"""The airbag engine: classify → snapshot → decide, plus undo/redo/rewind.

Agent-neutral. Every integration (Claude Code, Codex, Gemini CLI, Cursor,
OpenCode, OpenHands, a plain Python agent...) converts its tool call into a
:class:`ToolCall`, asks :meth:`Airbag.before` for a :class:`Decision`, and
renders that decision in its own hook format.

Safety rule: the airbag never *grants* permission. It either stays silent
(the agent's normal permission flow continues), adds a notice, or escalates
to "ask"/"deny" for commands it cannot undo.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_vcr.airbag.classify import Verdict, classify, classify_file_write
from agent_vcr.airbag.store import (
    AirbagError,
    DatabaseDump,
    RestoreResult,
    ShadowStore,
    Snapshot,
    find_project_root,
)

APPROVAL_TTL_SECONDS = 600
#: Cursor / Copilot CLI / Devin also run .claude/settings.json hooks, so one tool
#: call can reach the airbag twice; repeats inside this window reuse the first answer.
DUPLICATE_WINDOW_SECONDS = 5.0


@dataclass
class ToolCall:
    """One agent tool call, normalized across agents."""

    agent: str
    tool_name: str
    kind: str  # "shell" | "edit" | "other"
    cwd: str
    command: str | None = None
    paths: list[str] = field(default_factory=list)
    project_dir: str | None = None
    session_id: str | None = None
    call_id: str | None = None


@dataclass
class Policy:
    """How the airbag behaves. Defaults are safe and quiet."""

    enabled: bool = True
    #: "every": snapshot before every non-read tool call (full undo history).
    #: "risky": only before destructive/irreversible commands (fastest).
    mode: str = "every"
    #: What to do with commands a snapshot cannot undo: "ask", "deny" or "warn".
    on_irreversible: str = "ask"
    #: Show a notice to the user when a destructive command was snapshotted.
    notify_destructive: bool = True

    @classmethod
    def from_env(cls) -> Policy:
        env = os.environ
        policy = cls()
        if env.get("AGENT_VCR_AIRBAG", "").lower() in {"0", "off", "false", "no", "disabled"}:
            policy.enabled = False
        if env.get("AGENT_VCR_AIRBAG_MODE") in {"every", "risky"}:
            policy.mode = env["AGENT_VCR_AIRBAG_MODE"]
        if env.get("AGENT_VCR_AIRBAG_ON_IRREVERSIBLE") in {"ask", "deny", "warn"}:
            policy.on_irreversible = env["AGENT_VCR_AIRBAG_ON_IRREVERSIBLE"]
        return policy


@dataclass
class Decision:
    """What the integration should tell its agent."""

    action: str  # "continue" | "ask" | "deny"
    verdict: Verdict
    snapshot: Snapshot | None = None
    #: Shown to the human (system message / notification).
    user_message: str | None = None
    #: Fed back to the model (reason for ask/deny, or a hint).
    agent_message: str | None = None
    error: str | None = None
    #: Local databases dumped for this call (attached to ``snapshot``).
    databases: list[str] = field(default_factory=list)


class Airbag:
    """Protect one project directory."""

    def __init__(self, project_dir: str | Path, policy: Policy | None = None,
                 store: ShadowStore | None = None) -> None:
        self.policy = policy or Policy.from_env()
        self.store = store or ShadowStore(find_project_root(project_dir))

    @classmethod
    def for_call(cls, call: ToolCall, policy: Policy | None = None) -> Airbag:
        return cls(call.project_dir or call.cwd, policy=policy)

    @property
    def root(self) -> Path:
        return self.store.root

    # ── before a tool call ───────────────────────────────────────────────

    def classify(self, call: ToolCall) -> Verdict:
        if call.kind == "shell" and call.command:
            return classify(call.command, cwd=call.cwd, root=self.root)
        if call.kind == "edit":
            verdict = Verdict(level="write")
            for path in call.paths or [""]:
                verdict.merge(classify_file_write(path, cwd=call.cwd, root=self.root))
            return verdict
        return Verdict(level="read")

    def before(self, call: ToolCall) -> Decision:
        """Classify, snapshot if needed, and decide. Never raises."""
        verdict = self.classify(call)
        decision = Decision(action="continue", verdict=verdict)
        if not self.policy.enabled or verdict.level == "read":
            return decision
        if self.policy.mode == "risky" and verdict.level == "write":
            return decision
        duplicate = self._recent_duplicate(call)
        if duplicate is not None:
            decision.action = str(duplicate.get("decision") or "continue")
            decision.agent_message = duplicate.get("agent_message")
            return decision
        self._decide(call, verdict, decision)
        self._record_action(call, verdict, decision)
        return decision

    def _decide(self, call: ToolCall, verdict: Verdict, decision: Decision) -> None:

        label = _short(call.command or ", ".join(call.paths) or call.tool_name, 120)
        dumps, covers_all, db_problem = self._dump_databases(verdict, call.command)
        try:
            decision.snapshot = self.store.snapshot(
                label=f"before {call.tool_name}",
                meta={"agent": call.agent, "tool": call.tool_name, "command": label,
                      "level": verdict.level},
                backup_git_dir="git-dir" in verdict.categories,
                databases=dumps or None,
            )
        except (AirbagError, OSError, ValueError) as err:
            decision.error = f"airbag snapshot failed: {err}"
        except Exception as err:  # noqa: BLE001 - a hook must never crash the agent
            decision.error = f"airbag snapshot failed unexpectedly: {err!r}"
        finally:
            _discard_dumps(dumps)  # only the ones not attached to a snapshot

        snap = decision.snapshot
        dumped = [str(e.get("name")) for e in (snap.meta.get("databases") or [])] \
            if snap is not None and dumps else []
        decision.databases = dumped
        if dumped and covers_all and not db_problem and verdict.level == "irreversible":
            downgrade_for_databases(verdict, dumped)
        elif db_problem and db_problem not in verdict.reasons:
            verdict.reasons.append(db_problem)
        nested = self._nested_repos_hit(call, verdict, snap)
        if nested:
            verdict.raise_to(
                "irreversible",
                f"{', '.join(nested[:3])} {'is a nested git repository' if len(nested) == 1 else 'are nested git repositories'} "
                "— its files are not in the airbag snapshot and cannot be restored",
                "nested-repo",
            )

        if verdict.level == "irreversible":
            reasons = "; ".join(verdict.reasons) or "this command cannot be undone"
            if self._consume_approval(call):
                decision.user_message = f"Agent VCR airbag: approved irreversible command ({reasons})."
                return
            if self.policy.on_irreversible == "warn":
                decision.user_message = f"⚠ Agent VCR airbag: {reasons}. A snapshot cannot undo this."
                return
            decision.action = self.policy.on_irreversible  # "ask" | "deny"
            self._remember_pending(call, reasons)
            decision.user_message = (
                f"⚠ Agent VCR airbag: `{_short(call.command or label, 80)}` — {reasons}."
            )
            decision.agent_message = (
                f"Agent VCR airbag stopped this command: {reasons}. A local snapshot cannot undo "
                "it. Explain to the user what it will do and let them confirm; if they approve, "
                "they can run `vcr airbag allow` and you may retry the exact same command once."
            )
            return

        if verdict.level == "destructive" and self.policy.notify_destructive and snap:
            extra = f" (database {', '.join(dumped)} snapshotted)" if dumped else ""
            decision.user_message = (
                f"🛟 Agent VCR airbag: snapshot {snap.short} saved before "
                f"`{_short(call.command or label, 60)}`{extra}. Undo with `vcr undo`."
            )
        if decision.error and verdict.level in {"destructive", "irreversible"}:
            decision.user_message = (
                f"⚠ Agent VCR airbag could not snapshot before a {verdict.level} command: "
                f"{decision.error}"
            )

    # ── local databases (opt-in, see airbag.databases) ───────────────────

    def _dump_databases(self, verdict: Verdict, command: str | None = None,
                        ) -> tuple[list[DatabaseDump], bool, str | None]:
        """Dump the configured local databases a destructive SQL command would hit.

        Returns (dumps, whether they make every irreversible part undoable, problem).
        """
        if not verdict.db_targets:
            return [], False, None
        dumps: list[DatabaseDump] = []
        try:
            from agent_vcr.airbag import databases as dbs

            plan = dbs.plan_dumps(verdict, self.root, command)
            if not plan.databases or plan.problem:
                return [], False, plan.problem
            for db in plan.databases:
                path = self.store.new_dump_path(db.dbname)
                try:
                    dbs.dump(db, path)
                except dbs.DatabaseError as err:
                    _discard_dumps([DatabaseDump(path=path, name=db.dbname)])
                    return dumps, False, (f"could not snapshot the local database {db.display}: "
                                          f"{err}")
                dumps.append(DatabaseDump(path=path, name=db.dbname, info=db.info()))
            if plan.caveat:
                names = ", ".join(d.name for d in dumps)
                return dumps, False, f"local database {names} was dumped, but {plan.caveat}"
            return dumps, plan.covers_all, None
        except Exception as err:  # noqa: BLE001 - a hook must never crash the agent
            _discard_dumps(dumps)
            return [], False, f"could not snapshot the local database: {err!r}"

    def database_restore_plan(self, target: Snapshot,
                              only: str | None = None) -> list[dict[str, Any]]:
        """Database dumps that bring each database back to how it was at ``target``.

        A database's state at a snapshot is the first dump taken at or after it
        (nothing dumped it in between, so nothing destructive ran on it). No
        dump after ``target`` means the database has not been touched since.
        """
        chosen: dict[str, dict[str, Any]] = {}
        for snap in self.store.snapshots():
            if snap.seq < target.seq:
                continue
            for entry in snap.meta.get("databases") or []:
                if not isinstance(entry, dict):
                    continue
                key = str(entry.get("key") or entry.get("name"))
                if only and only not in {entry.get("name"), key}:
                    continue
                if key not in chosen:
                    chosen[key] = {**entry, "snapshot": snap.seq}
        return list(chosen.values())

    def _prepare_database_restore(
        self, plan: list[dict[str, Any]],
    ) -> tuple[list[DatabaseDump], list[tuple[Any, Path, str]], list[str]]:
        """Safety-dump every database about to be restored (never restore without one)."""
        from agent_vcr.airbag import databases as dbs

        configs = {db.key: db for db in dbs.load_databases(self.root)[0]}
        pre: list[DatabaseDump] = []
        ready: list[tuple[Any, Path, str]] = []
        problems: list[str] = []
        missing = dbs.missing_tools("pg_dump", "pg_restore")
        for entry in plan:
            name = str(entry.get("name"))
            db = configs.get(str(entry.get("key")))
            src = self.store.dump_file(entry)
            if db is None:
                problems.append(f"database {name}: no longer configured in "
                                f"{dbs.ENV_VAR} or {dbs.CONFIG_FILE}; left untouched")
                continue
            if src is None:
                problems.append(f"database {name}: dump file is missing; left untouched")
                continue
            if missing:
                problems.append(f"database {name}: {missing}; left untouched")
                continue
            path = self.store.new_dump_path(db.dbname)
            try:
                dbs.dump(db, path)
            except dbs.DatabaseError as err:
                _discard_dumps([DatabaseDump(path=path, name=name)])
                problems.append(f"database {name}: could not dump its current state first "
                                f"({err}); left untouched")
                continue
            pre.append(DatabaseDump(path=path, name=db.dbname, info=db.info()))
            ready.append((db, src, name))
        return pre, ready, problems

    def _apply_database_restore(self, ready: list[tuple[Any, Path, str]],
                                result: RestoreResult) -> None:
        from agent_vcr.airbag import databases as dbs

        for db, src, name in ready:
            try:
                note = dbs.restore(db, src)
            except dbs.DatabaseError as err:
                result.failed.append(f"database {name}: {err}")
                continue
            result.databases.append(name)
            if note:
                result.notes.append(f"database {name}: {note}")

    def undo_database_plan(self, action: dict[str, Any]) -> list[dict[str, Any]]:
        """Dumps ``vcr undo`` restores when it reverts back to ``action``.

        Only databases that one of the reverted actions (``action`` and the
        ones after it, up to the current undo position) dumped itself: undoing
        a file edit never touches a database.
        """
        actions = self.actions()
        pos = _pos(self.store.read_state(), len(actions))
        start = next((i for i, a in enumerate(actions) if a is action or a == action), None)
        if start is None:
            return []
        snaps = {s.seq: s for s in self.store.snapshots()}
        chosen: dict[str, dict[str, Any]] = {}
        for undone in actions[start:max(pos, start + 1)]:
            snap = snaps.get(int(undone["snapshot"]))
            if snap is None or not _dumped_by(undone, snap):
                continue
            for entry in snap.meta.get("databases") or []:
                if isinstance(entry, dict):
                    key = str(entry.get("key") or entry.get("name"))
                    chosen.setdefault(key, {**entry, "snapshot": snap.seq})
        return list(chosen.values())

    def _restore(self, target: Snapshot,
                 db_plan: list[dict[str, Any]] | None = None) -> RestoreResult:
        """Files (and configured local databases) back to ``target``.

        ``db_plan`` defaults to the databases' state at ``target`` (rewind, redo).
        """
        plan = self.database_restore_plan(target) if db_plan is None else db_plan
        if not plan:
            return self.store.restore(target)
        pre, ready, problems = self._prepare_database_restore(plan)
        try:
            result = self.store.restore(target, databases=pre or None)
        finally:
            _discard_dumps(pre)
        self._apply_database_restore(ready, result)
        result.failed.extend(problems)
        return result

    def restore_databases(self, seq: int, only: str | None = None) -> RestoreResult | None:
        """Only the databases back to snapshot ``seq`` (files untouched); None = no dump."""
        target = self.store.get(seq)
        plan = self.database_restore_plan(target, only=only)
        if not plan:
            return None
        pre, ready, problems = self._prepare_database_restore(plan)
        try:
            before = self.store.snapshot(label="before database restore",
                                         meta={"restoring_to": seq}, databases=pre or None)
        finally:
            _discard_dumps(pre)
        result = RestoreResult(target=target, before=before)
        self._apply_database_restore(ready, result)
        result.failed.extend(problems)
        self.store.log_event({"kind": "db-restore", "to": seq, "before": before.seq,
                              "databases": result.databases, "failed": result.failed[:20]})
        return result

    def snapshot_databases(self, only: str | None = None,
                           label: str = "database snapshot") -> tuple[Snapshot, list[str]]:
        """Dump the configured databases now (``vcr airbag db snapshot``)."""
        from agent_vcr.airbag import databases as dbs

        configs, _ = dbs.load_databases(self.root)
        configs = [db for db in configs if not only or only in {db.dbname, db.key}]
        if not configs:
            raise AirbagError("no matching local database configured")
        missing = dbs.missing_tools("pg_dump")
        if missing:
            raise AirbagError(missing)
        dumps: list[DatabaseDump] = []
        try:
            for db in configs:
                path = self.store.new_dump_path(db.dbname)
                dumps.append(DatabaseDump(path=path, name=db.dbname, info=db.info()))
                try:
                    dbs.dump(db, path)
                except dbs.DatabaseError as err:
                    raise AirbagError(f"{db.display}: {err}") from None
            snap = self.store.snapshot(label=label, meta={"agent": "user", "manual": True},
                                       databases=dumps)
        finally:
            _discard_dumps(dumps)
        names = [db.dbname for db in configs]
        self.store.log_event({"kind": "action", "agent": "user", "tool": "db snapshot",
                              "command": label, "level": "write", "snapshot": snap.seq,
                              "databases": names})
        return snap, names

    # ── undo / redo / rewind ─────────────────────────────────────────────

    def actions(self) -> list[dict[str, Any]]:
        """Recorded agent actions (oldest first) that have a snapshot."""
        return [e for e in self.store.events()
                if e.get("kind") == "action" and e.get("snapshot") is not None]

    def plan_undo(self, steps: int = 1) -> tuple[dict[str, Any], Snapshot] | None:
        """Which action ``vcr undo`` would revert, without touching anything."""
        actions = self.actions()
        state = self.store.read_state()
        pos = _pos(state, len(actions))
        current = self.store.snapshot(label="current")
        git_damaged = self.store.git_dir_needs_repair()
        remaining = steps
        idx = pos - 1
        while idx >= 0:
            action = actions[idx]
            try:
                snap = self.store.get(int(action["snapshot"]))
            except KeyError:
                idx -= 1
                continue
            repairs_git = git_damaged and "git-dir" in (action.get("categories") or [])
            if snap.tree != current.tree or repairs_git or _dumped_by(action, snap):
                remaining -= 1
                if remaining == 0:
                    return action, snap
            idx -= 1
        return None

    def preview_undo(
        self, steps: int = 1,
    ) -> tuple[dict[str, Any], Snapshot, list[tuple[str, str]]] | None:
        """``plan_undo`` plus the ``(status, path)`` changes ``undo`` would make.

        Status is from current → snapshot: ``A`` comes back, ``D`` is removed,
        ``M``/``T`` is rewritten. Only the shadow store is written (a "current"
        snapshot); project files are untouched.
        """
        plan = self.plan_undo(steps)
        if plan is None:
            return None
        action, snap = plan
        current = self.store.latest()
        return action, snap, self.store.diff(current, snap) if current else []

    def preview_rewind(self, seq: int) -> tuple[Snapshot, list[tuple[str, str]]]:
        """The snapshot ``rewind(seq)`` restores and the changes it would make."""
        target = self.store.get(seq)
        current = self.store.snapshot(label="current")
        return target, self.store.diff(current, target)

    def manual_snapshot(self, message: str, agent: str = "user") -> Snapshot:
        """Take a save point now and log it on the timeline (``vcr airbag snapshot``)."""
        snap = self.store.snapshot(label=message, meta={"agent": agent, "manual": True})
        self.store.log_event({"kind": "action", "agent": agent, "tool": "snapshot",
                              "command": message, "level": "write", "snapshot": snap.seq})
        return snap

    def undo(self, steps: int = 1) -> tuple[dict[str, Any], RestoreResult] | None:
        """Revert the files to just before the last ``steps`` file-changing actions."""
        plan = self.plan_undo(steps)
        if plan is None:
            return None
        action, snap = plan
        actions = self.actions()
        state = self.store.read_state()
        old_pos = _pos(state, len(actions))
        result = self._restore(snap, db_plan=self.undo_database_plan(action))
        new_pos = next((i for i, a in enumerate(actions) if a is action or a == action), old_pos)
        redo = list(state.get("redo", []))
        redo.append({"before": result.before.seq, "pos": old_pos})
        self.store.update_state({"undo_pos": new_pos, "redo": redo[-100:]})
        return action, result

    def redo(self) -> RestoreResult | None:
        state = self.store.read_state()
        redo = list(state.get("redo", []))
        if not redo:
            return None
        item = redo.pop()
        try:
            target = self.store.get(int(item["before"]))
        except KeyError:
            # The snapshot was garbage-collected; drop this redo entry.
            self.store.update_state({"redo": redo})
            raise
        result = self._restore(target)
        update: dict[str, Any] = {"redo": redo}
        update["undo_pos"] = item.get("pos")
        self.store.update_state(update)
        return result

    def rewind(self, seq: int) -> RestoreResult:
        state = self.store.read_state()
        result = self._restore(self.store.get(seq))
        redo = list(state.get("redo", []))
        redo.append({"before": result.before.seq, "pos": state.get("undo_pos")})
        actions = self.actions()
        pos = next((i for i, a in enumerate(actions) if int(a["snapshot"]) == seq), None)
        self.store.update_state({"redo": redo[-100:], "undo_pos": pos})
        return result

    # ── approvals ────────────────────────────────────────────────────────

    def approve_pending(self) -> dict[str, Any] | None:
        """Approve the most recent stopped command for one retry (10 minutes)."""
        state = self.store.read_state()
        pending = state.get("pending")
        if not pending or time.time() - float(pending.get("time", 0)) > APPROVAL_TTL_SECONDS:
            return None
        approvals = [a for a in state.get("approvals", [])
                     if time.time() - float(a.get("time", 0)) < APPROVAL_TTL_SECONDS]
        approvals.append({"hash": pending["hash"], "time": time.time(),
                          "command": pending.get("command")})
        self.store.update_state({"approvals": approvals, "pending": None})
        self.store.log_event({"kind": "approval", "command": pending.get("command")})
        return pending

    def _nested_repos_hit(self, call: ToolCall, verdict: Verdict,
                          snap: Snapshot | None) -> list[str]:
        """Nested repos (submodules, cloned deps) that a deleting command would remove."""
        if snap is None or verdict.level != "destructive" or not verdict.paths:
            return []
        nested = [str(n) for n in snap.meta.get("nested_repos") or []]
        if not nested:
            return []
        cwd = Path(call.cwd).expanduser()
        targets = []
        for raw in verdict.paths:
            path = Path(os.path.expanduser(raw))
            targets.append(Path(os.path.normpath(str(path if path.is_absolute() else cwd / path))))
        hit = []
        for rel in nested:
            repo = Path(os.path.normpath(str(self.root / rel)))
            if any(t == repo or t in repo.parents or repo in t.parents for t in targets):
                hit.append(rel)
        return hit

    def _call_hash(self, call: ToolCall) -> str:
        """Identity of *what* runs — independent of how each agent names its tool."""
        cwd = str(Path(call.cwd).expanduser().resolve()) if call.cwd else ""
        payload = json.dumps([call.kind, call.command, sorted(call.paths), cwd], default=str)
        return hashlib.sha256(payload.encode()).hexdigest()[:24]

    def _recent_duplicate(self, call: ToolCall) -> dict[str, Any] | None:
        digest = self._call_hash(call)
        now = time.time()
        for event in reversed(self.store.recent_events()):
            if now - float(event.get("time", 0)) > DUPLICATE_WINDOW_SECONDS:
                break
            if event.get("kind") == "approval":
                return None  # the user approved since: this is the approved retry
            if event.get("kind") != "action" or event.get("hash") != digest:
                continue
            other = event.get("call_id")
            if other and call.call_id and other != call.call_id:
                continue  # same command, genuinely a second call
            return event
        return None

    def _remember_pending(self, call: ToolCall, reasons: str) -> None:
        self.store.update_state({"pending": {
            "hash": self._call_hash(call), "time": time.time(), "agent": call.agent,
            "command": _short(call.command or call.tool_name, 300), "reasons": reasons,
        }})

    def _consume_approval(self, call: ToolCall) -> bool:
        state = self.store.read_state()
        approvals = state.get("approvals", [])
        digest = self._call_hash(call)
        now = time.time()
        for i, approval in enumerate(approvals):
            if approval.get("hash") == digest and now - float(approval.get("time", 0)) < \
                    APPROVAL_TTL_SECONDS:
                del approvals[i]
                self.store.update_state({"approvals": approvals})
                return True
        return False

    def _record_action(self, call: ToolCall, verdict: Verdict, decision: Decision) -> None:
        snap, error = decision.snapshot, decision.error
        try:
            state = self.store.read_state()
            if state.get("undo_pos") is not None or state.get("redo"):
                # A new action after an undo starts a new timeline (like an editor).
                self.store.update_state({"undo_pos": None, "redo": []})
            self.store.log_event({
                "kind": "action", "agent": call.agent, "tool": call.tool_name,
                "command": _short(call.command or "", 500), "paths": call.paths[:20],
                "cwd": call.cwd, "session": call.session_id, "call_id": call.call_id,
                "level": verdict.level, "reasons": verdict.reasons,
                "categories": verdict.categories,
                "snapshot": snap.seq if snap else None, "error": error,
                "hash": self._call_hash(call), "decision": decision.action,
                "agent_message": decision.agent_message,
                "databases": decision.databases,
            })
        except OSError:
            pass


def _dumped_by(action: dict[str, Any], snap: Snapshot) -> bool:
    """Did this action itself dump a database (not just share a snapshot with one that did)?"""
    return bool(action.get("databases")) and bool(snap.meta.get("databases"))


def _discard_dumps(dumps: list[DatabaseDump]) -> None:
    """Delete dump files that were not attached to a snapshot (attached ones were moved)."""
    for item in dumps:
        with contextlib.suppress(OSError):
            item.path.unlink()


def downgrade_for_databases(verdict: Verdict, names: list[str],
                            note: str = "database snapshotted") -> None:
    """Every irreversible part was a configured local database, now dumped: undoable."""
    for reason in verdict.irreversible_reasons:
        if reason in verdict.reasons:
            verdict.reasons.remove(reason)
    verdict.irreversible_reasons.clear()
    verdict.level = "destructive"
    labels = sorted({str(t.get("label") or "destructive SQL") for t in verdict.db_targets})
    verdict.raise_to("destructive", f"{' / '.join(labels)} on local database "
                     f"{', '.join(names)} ({note})", "database")


def describe_action(event: dict[str, Any], root: str | os.PathLike[str] | None = None) -> str:
    """A short human name for a logged action: its command, or ``Edit src/app.py``.

    Paths inside ``root`` (default: the action's cwd) are shown relative to it.
    """
    command = str(event.get("command") or "").strip()
    if command:
        return command
    base = str(root or event.get("cwd") or "")
    shown = []
    for raw in list(event.get("paths") or [])[:3]:
        path = str(raw)
        if base:
            try:
                rel = os.path.relpath(path, base)
                if not rel.startswith(".."):
                    path = rel
            except ValueError:  # different drives on Windows
                pass
        shown.append(path.replace(os.sep, "/"))
    if len(event.get("paths") or []) > 3:
        shown.append("…")
    tool = str(event.get("tool") or "").strip()
    return " ".join(part for part in (tool, ", ".join(shown)) if part) or "?"


def _pos(state: dict[str, Any], count: int) -> int:
    raw = state.get("undo_pos")
    return count if raw is None else max(0, min(int(raw), count))


def _short(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
