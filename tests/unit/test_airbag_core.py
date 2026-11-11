"""Airbag engine tests: classifier, shadow-git store, undo/redo, approvals.

Everything runs against real git and real SQLite in temporary directories.
"""

from __future__ import annotations

import multiprocessing
import os
import shutil
import sqlite3
import subprocess
from pathlib import Path

import pytest

from agent_vcr.airbag.classify import classify
from agent_vcr.airbag.core import Airbag, Policy, ToolCall
from agent_vcr.airbag.store import AirbagError, ShadowStore

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git required")

ROOT = "/home/someone/proj"  # classification needs no real directory


@pytest.fixture(autouse=True)
def _airbag_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_VCR_AIRBAG_HOME", str(tmp_path / "airbag-home"))
    for var in ("AGENT_VCR_AIRBAG", "AGENT_VCR_AIRBAG_MODE", "AGENT_VCR_AIRBAG_ON_IRREVERSIBLE",
                "DATABASE_URL", "PGHOST"):
        monkeypatch.delenv(var, raising=False)


def _git(cwd: Path, *args: str) -> str:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True,
                          env=env).stdout


@pytest.fixture
def project(tmp_path: Path) -> Path:
    proj = tmp_path / "proj"
    (proj / "src").mkdir(parents=True)
    _git(proj, "init", "-q", "-b", "main")
    _git(proj, "config", "user.email", "u@example.com")
    _git(proj, "config", "user.name", "u")
    (proj / "src" / "app.py").write_text("print('v1')\n")
    (proj / ".gitignore").write_text(".env\n*.db\nnode_modules/\nbig.bin\n")
    _git(proj, "add", "-A")
    _git(proj, "commit", "-qm", "init")
    return proj


def _call(proj: Path, command: str) -> ToolCall:
    return ToolCall(agent="test", tool_name="Bash", kind="shell", cwd=str(proj), command=command)


# ── classifier ───────────────────────────────────────────────────────────

@pytest.mark.parametrize("command,level", [
    ("ls -la && git status", "read"),
    ("grep -rn 'DROP TABLE' migrations/", "read"),
    ("pytest -q", "write"),
    ("echo hi >> notes.txt", "write"),
    ("rm -rf src", "destructive"),
    ("rm -rf .", "destructive"),
    ("git reset --hard HEAD~2", "destructive"),
    ("git clean -fdx", "destructive"),
    ("find . -name '*.log' -delete", "destructive"),
    ("bash -c 'rm -rf build'", "destructive"),
    ("sqlite3 app.db 'DROP TABLE users'", "destructive"),
    ("rm -rf ~/other-project", "irreversible"),
    ("cd /etc && rm -rf nginx", "irreversible"),
    ("sudo rm -rf --no-preserve-root /", "irreversible"),
    ("git push --force origin main", "irreversible"),
    ("git branch -D feature", "irreversible"),
    ("psql -h prod.example.com -c 'DROP TABLE users'", "irreversible"),
    ("echo 'TRUNCATE orders;' | psql", "irreversible"),
    ("DATABASE_URL=postgres://u@db.internal/app npx prisma db push --accept-data-loss",
     "irreversible"),
    ("terraform destroy -auto-approve", "irreversible"),
    ("kubectl delete namespace prod", "irreversible"),
    ("railway volume delete data", "irreversible"),
    ("docker compose down -v", "irreversible"),
    ("curl -X DELETE https://api.example.com/v1/users/7", "irreversible"),
    ("npm publish", "irreversible"),
    ("echo $(rm -rf src)", "destructive"),
])
def test_classifier_levels(command: str, level: str) -> None:
    assert classify(command, cwd=ROOT, root=ROOT).level == level


def test_classifier_reports_remote_database_host() -> None:
    verdict = classify("psql -h prod.example.com -c 'DROP TABLE users'", cwd=ROOT, root=ROOT)
    assert "prod.example.com" in " ".join(verdict.reasons)
    assert "database" in verdict.categories


def test_deleting_the_project_marks_git_dir() -> None:
    assert "git-dir" in classify(f"rm -rf {ROOT}", cwd="/home/someone", root=ROOT).categories
    assert "git-dir" in classify("rm -rf .git", cwd=ROOT, root=ROOT).categories


# ── store ────────────────────────────────────────────────────────────────

def test_store_lives_outside_the_project_and_never_touches_user_git(project: Path) -> None:
    (project / "src" / "app.py").write_text("print('dirty')\n")
    _git(project, "stash", "list")
    head, status = _git(project, "rev-parse", "HEAD"), _git(project, "status", "--porcelain")
    store = ShadowStore(project)
    snap = store.snapshot("t")
    assert project not in store.dir.parents
    (project / "src" / "app.py").write_text("print('changed by agent')\n")
    store.restore(snap)
    assert (project / "src" / "app.py").read_text() == "print('dirty')\n"
    assert _git(project, "rev-parse", "HEAD") == head
    assert _git(project, "status", "--porcelain") == status
    assert _git(project, "for-each-ref", "refs/airbag") == ""


def test_snapshot_dedups_when_nothing_changed(project: Path) -> None:
    store = ShadowStore(project)
    first = store.snapshot("a")
    assert store.snapshot("b").seq == first.seq
    (project / "new.txt").write_text("x")
    assert store.snapshot("c").seq == first.seq + 1


def test_protects_untracked_and_precious_but_not_dependency_trees(project: Path) -> None:
    (project / ".env").write_text("API_KEY=secret\n")
    (project / "draft.md").write_text("untracked\n")
    (project / "node_modules" / "pkg").mkdir(parents=True)
    (project / "node_modules" / "pkg" / "index.js").write_text("x")
    store = ShadowStore(project)
    snap = store.snapshot("t")
    for path in (".env", "draft.md", "node_modules"):
        target = project / path
        shutil.rmtree(target) if target.is_dir() else target.unlink()
    store.restore(snap)
    assert (project / ".env").read_text() == "API_KEY=secret\n"
    assert (project / "draft.md").exists()
    assert not (project / "node_modules").exists()  # regenerable, never snapshotted


def test_restore_never_deletes_ignored_files(project: Path) -> None:
    store = ShadowStore(project)
    snap = store.snapshot("t")
    (project / "node_modules").mkdir()
    (project / "node_modules" / "keep.js").write_text("keep")
    (project / "created.txt").write_text("agent made this")
    result = store.restore(snap)
    assert "created.txt" in result.deleted and not (project / "created.txt").exists()
    assert (project / "node_modules" / "keep.js").exists()


def test_large_files_are_skipped(project: Path) -> None:
    (project / "huge.dat").write_bytes(b"x" * 4096)
    store = ShadowStore(project, max_file_bytes=1024)
    snap = store.snapshot("t")
    assert "huge.dat" in snap.meta["skipped_large"]


def test_sqlite_snapshot_is_consistent_in_wal_mode(project: Path) -> None:
    db_path = project / "app.db"
    db = sqlite3.connect(db_path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("create table users(id int)")
    db.executemany("insert into users values (?)", [(i,) for i in range(100)])
    db.commit()  # rows live in app.db-wal until checkpoint; db stays open
    store = ShadowStore(project)
    snap = store.snapshot("before drop")
    db.execute("drop table users")
    db.commit()
    db.close()
    store.restore(snap)
    rows = sqlite3.connect(db_path).execute("select count(*) from users").fetchone()
    assert rows == (100,)


def test_full_project_deletion_restores_files_and_git_history(project: Path) -> None:
    (project / "src" / "app.py").write_text("print('uncommitted')\n")
    (project / ".env").write_text("SECRET=1\n")
    head = _git(project, "rev-parse", "HEAD")
    bag = Airbag(project, policy=Policy())
    decision = bag.before(_call(project.parent, f"rm -rf {project}"))
    assert decision.verdict.level == "destructive" and decision.snapshot is not None
    shutil.rmtree(project)
    undone = bag.undo()
    assert undone is not None
    assert (project / "src" / "app.py").read_text() == "print('uncommitted')\n"
    assert (project / ".env").read_text() == "SECRET=1\n"
    assert _git(project, "rev-parse", "HEAD") == head
    assert "app.py" in _git(project, "status", "--porcelain")


def test_refuses_home_directory(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(AirbagError):
        ShadowStore(Path.home())


def _parallel_snapshot(args: tuple[str, int]) -> int:
    root, i = args
    (Path(root) / f"file{i}.txt").write_text(str(i))
    return ShadowStore(root).snapshot(f"p{i}").seq


def test_concurrent_hook_processes_do_not_corrupt_the_store(project: Path) -> None:
    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(4) as pool:
        pool.map(_parallel_snapshot, [(str(project), i) for i in range(12)])
    store = ShadowStore(project)
    seqs = [s.seq for s in store.snapshots()]
    assert len(seqs) == len(set(seqs))
    latest = store.snapshot("final")
    assert latest.files >= 13  # app.py + .gitignore + 12 new files (no lost writes)


def test_gc_removes_old_snapshots(project: Path) -> None:
    store = ShadowStore(project)
    for i in range(5):
        (project / f"f{i}").write_text(str(i))
        store.snapshot(f"s{i}")
    assert store.gc(keep_days=0, keep_last=2) == 3
    assert len(store.snapshots()) == 2


# ── undo / redo / rewind ─────────────────────────────────────────────────

def test_undo_skips_actions_that_changed_nothing(project: Path) -> None:
    bag = Airbag(project, policy=Policy())
    bag.before(_call(project, "rm src/app.py"))
    (project / "src" / "app.py").unlink()
    bag.before(_call(project, "pytest -q"))  # changed nothing afterwards
    action, result = bag.undo()  # type: ignore[misc]
    assert action["command"] == "rm src/app.py"
    assert (project / "src" / "app.py").exists()
    assert "src/app.py" in result.written


def test_undo_twice_then_redo(project: Path) -> None:
    bag = Airbag(project, policy=Policy())
    app = project / "src" / "app.py"
    bag.before(_call(project, "sed -i s/v1/v2/ src/app.py"))
    app.write_text("print('v2')\n")
    bag.before(_call(project, "sed -i s/v2/v3/ src/app.py"))
    app.write_text("print('v3')\n")
    bag.undo()
    assert app.read_text() == "print('v2')\n"
    bag.undo()
    assert app.read_text() == "print('v1')\n"
    bag.redo()
    assert app.read_text() == "print('v2')\n"
    bag.redo()
    assert app.read_text() == "print('v3')\n"
    assert bag.redo() is None


def test_new_action_after_undo_clears_redo(project: Path) -> None:
    bag = Airbag(project, policy=Policy())
    bag.before(_call(project, "rm src/app.py"))
    (project / "src" / "app.py").unlink()
    bag.undo()
    bag.before(_call(project, "touch other.txt"))
    assert bag.store.read_state().get("redo") == []


def test_rewind_to_any_snapshot(project: Path) -> None:
    bag = Airbag(project, policy=Policy())
    first = bag.store.snapshot("start")
    for i in range(3):
        (project / f"step{i}.txt").write_text(str(i))
        bag.store.snapshot(f"s{i}")
    bag.rewind(first.seq)
    assert not any(project.glob("step*.txt"))
    bag.redo()
    assert len(list(project.glob("step*.txt"))) == 3


# ── decisions / approvals / policy ───────────────────────────────────────

def test_destructive_continues_with_notice_and_never_grants_permission(project: Path) -> None:
    decision = Airbag(project, policy=Policy()).before(_call(project, "rm -rf src"))
    assert decision.action == "continue"
    assert decision.snapshot is not None
    assert "vcr undo" in (decision.user_message or "")


def test_irreversible_asks_and_one_time_approval(project: Path) -> None:
    bag = Airbag(project, policy=Policy())

    def call(call_id: str) -> ToolCall:
        c = _call(project, "git push --force origin main")
        c.call_id = call_id
        return c

    first = bag.before(call("t1"))
    assert first.action == "ask" and first.agent_message
    assert bag.approve_pending() is not None
    assert bag.before(call("t2")).action == "continue"  # approved once
    assert bag.before(call("t3")).action == "ask"  # ...and only once


def test_same_call_reaching_two_hooks_is_handled_once(project: Path) -> None:
    """Cursor / Copilot also run .claude/settings.json hooks: one call, two hook runs."""
    bag = Airbag(project, policy=Policy())
    via_claude_hook = ToolCall(agent="claude", tool_name="Bash", kind="shell",
                               cwd=str(project), command="rm -rf src")
    via_cursor_hook = ToolCall(agent="cursor", tool_name="Shell", kind="shell",
                               cwd=str(project), command="rm -rf src")
    first = bag.before(via_claude_hook)
    second = bag.before(via_cursor_hook)
    assert first.snapshot is not None and second.snapshot is None
    assert len(bag.actions()) == 1
    ask_a = bag.before(_call(project, "terraform destroy"))
    ask_b = bag.before(_call(project, "terraform destroy"))
    assert ask_a.action == ask_b.action == "ask"  # duplicate keeps the escalation


def test_policy_deny_and_disabled(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_VCR_AIRBAG_ON_IRREVERSIBLE", "deny")
    assert Airbag(project).before(_call(project, "terraform destroy")).action == "deny"
    monkeypatch.setenv("AGENT_VCR_AIRBAG", "off")
    decision = Airbag(project).before(_call(project, "rm -rf src"))
    assert decision.action == "continue" and decision.snapshot is None


def test_risky_mode_skips_plain_writes(project: Path) -> None:
    bag = Airbag(project, policy=Policy(mode="risky"))
    assert bag.before(_call(project, "pytest")).snapshot is None
    assert bag.before(_call(project, "rm -rf src")).snapshot is not None


def test_snapshot_failure_never_raises(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    bag = Airbag(project, policy=Policy())

    def boom(*args: object, **kwargs: object) -> None:
        raise AirbagError("disk full")

    monkeypatch.setattr(bag.store, "snapshot", boom)
    decision = bag.before(_call(project, "rm -rf src"))
    assert decision.action == "continue"
    assert decision.error and "disk full" in decision.error
    assert "could not snapshot" in (decision.user_message or "")


# ── regressions from the adversarial safety review ───────────────────────

@pytest.mark.parametrize("command", [
    "sort -o important.txt data.txt", "yq -i '.a=1' deploy.yaml", "uniq in.txt out.txt",
    "fd -e log -x rm {}",
])
def test_read_only_programs_that_write_get_a_snapshot(command: str) -> None:
    assert classify(command, cwd=ROOT, root=ROOT).level in {"write", "destructive"}


def test_file_that_grew_past_the_limit_is_never_clobbered(project: Path) -> None:
    store = ShadowStore(project, max_file_bytes=1024)
    data = project / "data.csv"
    data.write_text("version-A\n")
    first = store.snapshot("A")
    data.write_text("version-B\n")
    store.snapshot("B")
    real = "".join(f"row {i} of irreplaceable results\n" for i in range(300))
    data.write_text(real)
    grown = store.snapshot("C")
    assert "data.csv" in grown.meta["skipped_large"]
    result = store.restore(first)
    assert data.read_text() == real and "data.csv" in result.kept
    store.restore(result.before)
    assert data.read_text() == real


@pytest.mark.parametrize("with_commit", [True, False])
def test_deleting_a_nested_repo_is_escalated_and_never_faked(project: Path,
                                                             with_commit: bool) -> None:
    nested = project / "vendor" / "lib"
    nested.mkdir(parents=True)
    _git(nested, "init", "-q")
    _git(nested, "config", "user.email", "u@example.com")
    _git(nested, "config", "user.name", "u")
    (nested / "core.c").write_text("int main(){}\n")
    if with_commit:
        _git(nested, "add", "-A")
        _git(nested, "commit", "-qm", "dep")
    bag = Airbag(project, policy=Policy())
    decision = bag.before(_call(project, "rm -rf vendor/lib"))
    assert decision.verdict.level == "irreversible" and decision.action == "ask"
    assert "nested git repositor" in " ".join(decision.verdict.reasons)
    shutil.rmtree(nested)
    undone = bag.undo()
    if undone is not None:
        assert "vendor/lib" in undone[1].kept
    assert not nested.exists()  # no misleading empty directory


def test_private_file_permissions_survive_restore(project: Path) -> None:
    secret = project / ".env"
    secret.write_text("KEY=1\n")
    secret.chmod(0o600)
    store = ShadowStore(project)
    snap = store.snapshot("t")
    secret.unlink()
    store.restore(snap)
    assert secret.stat().st_mode & 0o777 == 0o600


def test_gitattributes_cannot_change_restored_bytes(project: Path) -> None:
    (project / ".gitattributes").write_text("* text=auto\n")
    crlf = b"id,name\r\n1,alice\r\n"
    (project / "fixture.csv").write_bytes(crlf)
    store = ShadowStore(project)
    snap = store.snapshot("t")
    (project / "fixture.csv").unlink()
    store.restore(snap)
    assert (project / "fixture.csv").read_bytes() == crlf


def test_undo_repairs_damaged_or_replaced_git_dir(project: Path) -> None:
    head = _git(project, "rev-parse", "HEAD")
    bag = Airbag(project, policy=Policy())
    bag.before(_call(project, "rm -rf .git/objects"))
    shutil.rmtree(project / ".git" / "objects")
    assert bag.undo() is not None
    assert _git(project, "rev-parse", "HEAD") == head
    bag2 = Airbag(project, policy=Policy())
    bag2.before(_call(project, "rm -rf .git && git init"))
    shutil.rmtree(project / ".git")
    _git(project, "init", "-q")
    assert bag2.undo() is not None
    assert _git(project, "rev-parse", "HEAD") == head


def test_worktree_git_file_is_restored(project: Path, tmp_path: Path) -> None:
    worktree = tmp_path / "wt"
    _git(project, "worktree", "add", "-q", str(worktree))
    bag = Airbag(worktree, policy=Policy())
    bag.before(_call(tmp_path, f"rm -rf {worktree}"))
    shutil.rmtree(worktree)
    assert bag.undo() is not None
    assert (worktree / ".git").is_file()
    assert _git(worktree, "status", "--porcelain") == ""


def test_backslash_filename_is_restored_on_posix(project: Path) -> None:
    if os.name == "nt":
        pytest.skip("backslash is a path separator on Windows")
    odd = project / "a\\..\\b.txt"
    odd.write_text("odd")
    store = ShadowStore(project)
    snap = store.snapshot("t")
    odd.unlink()
    result = store.restore(snap)
    assert odd.read_text() == "odd" and not result.failed


def test_redo_after_gc_raises_cleanly(project: Path) -> None:
    bag = Airbag(project, policy=Policy())
    bag.before(_call(project, "rm src/app.py"))
    (project / "src" / "app.py").unlink()
    bag.undo()
    bag.store.gc(keep_days=0, keep_last=0)
    with pytest.raises(KeyError):
        bag.redo()
    assert bag.store.read_state()["redo"] == []


def test_size_limits_come_from_environment(monkeypatch: pytest.MonkeyPatch,
                                          project: Path) -> None:
    monkeypatch.setenv("AGENT_VCR_AIRBAG_MAX_FILES", "7")
    monkeypatch.setenv("AGENT_VCR_AIRBAG_MAX_FILE_MB", "0.5")
    store = ShadowStore(project)
    assert store.max_files == 7 and store.max_file_bytes == 524288


def test_cli_undo_finds_a_deleted_project(project: Path, capsys: pytest.CaptureFixture) -> None:
    """The headline demo: the agent deletes the project, a *fresh* `vcr undo` restores it."""
    from agent_vcr import cli

    (project / ".env").write_text("SECRET=1\n")
    head = _git(project, "rev-parse", "HEAD")
    Airbag(project, policy=Policy()).before(_call(project.parent, f"rm -rf {project}"))
    shutil.rmtree(project)
    cli.main(["undo", "--project-dir", str(project)])
    assert "Undid" in capsys.readouterr().out
    assert (project / ".env").read_text() == "SECRET=1\n"
    assert _git(project, "rev-parse", "HEAD") == head
    # ...and from a subdirectory of a project whose .git is gone:
    Airbag(project, policy=Policy()).before(_call(project, "rm -rf .git"))
    shutil.rmtree(project / ".git")
    cli.main(["undo", "--project-dir", str(project / "src")])
    assert _git(project, "rev-parse", "HEAD") == head
