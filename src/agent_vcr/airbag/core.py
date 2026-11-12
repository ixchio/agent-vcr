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
        try:
            decision.snapshot = self.store.snapshot(
                label=f"before {call.tool_name}",
                meta={"agent": call.agent, "tool": call.tool_name, "command": label,
                      "level": verdict.level},
                backup_git_dir="git-dir" in verdict.categories,
            )
        except (AirbagError, OSError, ValueError) as err:
            decision.error = f"airbag snapshot failed: {err}"
        except Exception as err:  # noqa: BLE001 - a hook must never crash the agent
            decision.error = f"airbag snapshot failed unexpectedly: {err!r}"

        snap = decision.snapshot
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
            decision.user_message = (
                f"🛟 Agent VCR airbag: snapshot {snap.short} saved before "
                f"`{_short(call.command or label, 60)}`. Undo with `vcr undo`."
            )
        if decision.error and verdict.level in {"destructive", "irreversible"}:
            decision.user_message = (
                f"⚠ Agent VCR airbag could not snapshot before a {verdict.level} command: "
                f"{decision.error}"
            )

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
            if snap.tree != current.tree or repairs_git:
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
        result = self.store.restore(snap)
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
        result = self.store.restore(target)
        update: dict[str, Any] = {"redo": redo}
        update["undo_pos"] = item.get("pos")
        self.store.update_state(update)
        return result

    def rewind(self, seq: int) -> RestoreResult:
        state = self.store.read_state()
        result = self.store.restore(self.store.get(seq))
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
            })
        except OSError:
            pass


def _pos(state: dict[str, Any], count: int) -> int:
    raw = state.get("undo_pos")
    return count if raw is None else max(0, min(int(raw), count))


def _short(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"
