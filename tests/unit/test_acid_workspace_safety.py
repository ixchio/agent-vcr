"""Data-safety tests for ACIDWorkspace using real git repositories."""

from __future__ import annotations

import hashlib
import logging
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent_vcr.integrations import openhands
from agent_vcr.integrations.openhands import ACIDWorkspace
from agent_vcr.recorder import VCRRecorder

_IDENT_ENV = (
    "GIT_AUTHOR_NAME",
    "GIT_AUTHOR_EMAIL",
    "GIT_COMMITTER_NAME",
    "GIT_COMMITTER_EMAIL",
    "EMAIL",
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
)


@pytest.fixture(autouse=True)
def isolated_git_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """No global identity, no system config: commits only work if the code supplies one."""
    home = tmp_path / "home"
    home.mkdir()
    global_config = home / ".gitconfig"
    global_config.write_text("[user]\n\tuseConfigOnly = true\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / ".config"))
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", str(global_config))
    monkeypatch.setenv("GIT_CONFIG_NOSYSTEM", "1")
    for name in _IDENT_ENV:
        monkeypatch.delenv(name, raising=False)


def git(cwd: Path, *args: str, check: bool = True) -> str:
    """Run git as the *user* would (with an explicit identity for their own commits)."""
    result = subprocess.run(
        ["git", "-c", "user.name=user", "-c", "user.email=user@example.com", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        check=check, encoding="utf-8", errors="replace",
    )
    return result.stdout


def make_repo(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    git(path, "init", "-q")
    git(path, "symbolic-ref", "HEAD", "refs/heads/main")
    (path / "tracked.txt").write_text("v1\n", encoding="utf-8")
    (path / "other.txt").write_text("other v1\n", encoding="utf-8")
    (path / "doomed.txt").write_text("doomed\n", encoding="utf-8")
    (path / ".gitignore").write_text("*.log\n", encoding="utf-8")
    git(path, "add", "-A")
    git(path, "commit", "-qm", "init")
    return path


def make_subdir_repo(root: Path) -> Path:
    make_repo(root)
    sub = root / "sub"
    sub.mkdir()
    (sub / "keep.txt").write_text("keep v1\n", encoding="utf-8")
    (sub / "edit.txt").write_text("edit v1\n", encoding="utf-8")
    git(root, "add", "-A")
    git(root, "commit", "-qm", "sub")
    return sub


def recorder(tmp_path: Path) -> VCRRecorder:
    return VCRRecorder(output_dir=str(tmp_path / "vcr-out"), auto_save=False)


def worktree_state(root: Path) -> dict[str, bytes]:
    state: dict[str, bytes] = {}
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if name != ".git"]
        for name in filenames:
            path = Path(dirpath) / name
            state[path.relative_to(root).as_posix()] = path.read_bytes()
    return state


def repo_state(root: Path) -> dict[str, Any]:
    """Everything the user could lose: files, index, status, branch, refs."""
    return {
        "files": worktree_state(root),
        "index": git(root, "ls-files", "-s"),
        "status": git(root, "status", "--porcelain=v1", "-uall", "--ignored"),
        "branch": git(root, "symbolic-ref", "-q", "HEAD", check=False).strip(),
        "head": git(root, "rev-parse", "HEAD").strip(),
        "branches": git(root, "for-each-ref", "--format=%(refname)", "refs/heads"),
    }


def all_committed_paths(root: Path) -> set[str]:
    out = git(root, "log", "--all", "--format=", "--name-only")
    return {line for line in out.splitlines() if line}


def make_user_mess(root: Path, prefix: str = "") -> None:
    """Unstaged, staged, staged+unstaged, deleted, untracked and ignored user changes."""
    base = root / prefix if prefix else root
    (base / ("edit.txt" if prefix else "tracked.txt")).write_text("USER UNSAVED WORK\n",
                                                                  encoding="utf-8")
    staged = base / ("keep.txt" if prefix else "other.txt")
    staged.write_text("USER STAGED\n", encoding="utf-8")
    git(root, "add", "--", staged.relative_to(root).as_posix())
    staged.write_text("USER STAGED THEN EDITED\n", encoding="utf-8")
    if not prefix:
        (root / "doomed.txt").unlink()
    (base / "notes.local").write_text("user notes\n", encoding="utf-8")
    (base / "debug.log").write_text("user log\n", encoding="utf-8")
    (base / "with space ü.txt").write_text("unicode name\n", encoding="utf-8")


# ──────────────────────────────────────────────
#  Bug 1: abort must preserve pre-existing dirty state
# ──────────────────────────────────────────────


class TestAbortPreservesDirtyState:
    def test_abort_restores_preexisting_dirty_tracked_edits(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path / "repo")
        make_user_mess(repo)
        before = repo_state(repo)

        ws = ACIDWorkspace(str(repo), recorder=recorder(tmp_path), dirty_worktree_policy="allow")
        ws.begin("dirty-abort")
        (repo / "tracked.txt").write_text("agent overwrite\n", encoding="utf-8")
        (repo / "agent.py").write_text("print('agent')\n", encoding="utf-8")
        (repo / "notes.local").write_text("agent clobbered notes\n", encoding="utf-8")
        ws.savepoint({"step": 1})
        (repo / "other.txt").unlink()
        (repo / "later.py").write_text("later\n", encoding="utf-8")
        ws.abort()

        assert repo_state(repo) == before
        assert (repo / "tracked.txt").read_text(encoding="utf-8") == "USER UNSAVED WORK\n"

    def test_rollback_keeps_preexisting_dirty_edits(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path / "repo")
        (repo / "tracked.txt").write_text("USER UNSAVED WORK\n", encoding="utf-8")

        ws = ACIDWorkspace(str(repo), recorder=recorder(tmp_path), dirty_worktree_policy="allow")
        ws.begin("dirty-rollback")
        ws.savepoint({"step": 0})
        (repo / "tracked.txt").write_text("agent overwrite\n", encoding="utf-8")
        ws.savepoint({"step": 1})
        ws.rollback(0)

        assert (repo / "tracked.txt").read_text(encoding="utf-8") == "USER UNSAVED WORK\n"
        ws.abort()
        assert (repo / "tracked.txt").read_text(encoding="utf-8") == "USER UNSAVED WORK\n"
        assert git(repo, "status", "--porcelain").strip() == "M tracked.txt"

    def test_abort_without_savepoints_restores_everything(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path / "repo")
        make_user_mess(repo)
        before = repo_state(repo)

        ws = ACIDWorkspace(str(repo), recorder=recorder(tmp_path), dirty_worktree_policy="allow")
        ws.begin("no-savepoints")
        (repo / "tracked.txt").write_text("agent\n", encoding="utf-8")
        (repo / "new.txt").write_text("agent\n", encoding="utf-8")
        ws.abort()

        assert repo_state(repo) == before


# ──────────────────────────────────────────────
#  Bug 2: workspace inside a larger repository
# ──────────────────────────────────────────────


class TestSubdirectoryWorkspace:
    def test_abort_never_touches_paths_outside_workspace(self, tmp_path: Path) -> None:
        root = tmp_path / "repo"
        sub = make_subdir_repo(root)
        make_user_mess(root)  # outside the workspace
        make_user_mess(root, prefix="sub")  # inside the workspace
        before = repo_state(root)

        ws = ACIDWorkspace(str(sub), recorder=recorder(tmp_path), dirty_worktree_policy="allow")
        ws.begin("sub-abort")
        (sub / "agent.py").write_text("agent\n", encoding="utf-8")
        (sub / "keep.txt").write_text("agent edit\n", encoding="utf-8")
        frame = ws.savepoint({"step": 1})

        savepoint_paths = set(
            git(root, "ls-tree", "-r", "--name-only", frame.metadata.custom["acid_commit_hash"])
            .splitlines()
        )
        assert "sub/agent.py" in savepoint_paths
        assert "sub/notes.local" not in savepoint_paths
        assert "sub/with space ü.txt" not in savepoint_paths
        assert "doomed.txt" in savepoint_paths  # user's outside deletion was not committed
        outside_blob = git(root, "show", f"{frame.metadata.custom['acid_commit_hash']}:other.txt")
        assert outside_blob == "other v1\n"  # user's outside staged edit was not committed

        (sub / "later.py").write_text("later\n", encoding="utf-8")
        ws.abort()

        assert repo_state(root) == before
        assert "sub/notes.local" not in all_committed_paths(root)

    def test_begin_with_dirty_files_only_outside_workspace_is_allowed(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "repo"
        sub = make_subdir_repo(root)
        (root / "tracked.txt").write_text("USER UNSAVED WORK\n", encoding="utf-8")

        ws = ACIDWorkspace(str(sub), recorder=recorder(tmp_path))
        ws.begin("outside-dirty")
        ws.abort()
        assert (root / "tracked.txt").read_text(encoding="utf-8") == "USER UNSAVED WORK\n"

    def test_dirty_files_inside_subdir_workspace_still_fail_by_default(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "repo"
        sub = make_subdir_repo(root)
        (sub / "edit.txt").write_text("dirty\n", encoding="utf-8")

        ws = ACIDWorkspace(str(sub), recorder=recorder(tmp_path))
        with pytest.raises(RuntimeError, match=r"dirty tracked files.*edit\.txt"):
            ws.begin("inside-dirty")

    def test_commit_only_commits_workspace_and_preserves_outside_state(
        self, tmp_path: Path
    ) -> None:
        root = tmp_path / "repo"
        sub = make_subdir_repo(root)
        (root / "tracked.txt").write_text("USER UNSAVED WORK\n", encoding="utf-8")
        (root / "other.txt").write_text("USER STAGED\n", encoding="utf-8")
        git(root, "add", "other.txt")
        (root / "root_untracked.txt").write_text("user notes\n", encoding="utf-8")
        (sub / "notes.local").write_text("sub notes\n", encoding="utf-8")

        ws = ACIDWorkspace(str(sub), recorder=recorder(tmp_path))
        ws.begin("sub-commit")
        (sub / "agent.py").write_text("agent\n", encoding="utf-8")
        ws.savepoint({"step": 1})
        (sub / "edit.txt").write_text("edit v2\n", encoding="utf-8")
        ws.commit()

        head_paths = set(git(root, "ls-tree", "-r", "--name-only", "HEAD").splitlines())
        assert {"sub/agent.py", "sub/edit.txt"} <= head_paths
        assert "sub/notes.local" not in head_paths
        assert "root_untracked.txt" not in head_paths
        assert git(root, "show", "HEAD:tracked.txt") == "v1\n"
        assert git(root, "show", "HEAD:other.txt") == "other v1\n"
        assert git(root, "show", "HEAD:sub/edit.txt") == "edit v2\n"
        status = git(root, "status", "--porcelain=v1").splitlines()
        assert sorted(status) == sorted(
            ["M  other.txt", " M tracked.txt", "?? root_untracked.txt", "?? sub/notes.local"]
        )
        assert (root / "tracked.txt").read_text(encoding="utf-8") == "USER UNSAVED WORK\n"

    def test_rollback_in_subdir_is_scoped(self, tmp_path: Path) -> None:
        root = tmp_path / "repo"
        sub = make_subdir_repo(root)
        (root / "tracked.txt").write_text("USER UNSAVED WORK\n", encoding="utf-8")
        (sub / "notes.local").write_text("sub notes\n", encoding="utf-8")

        ws = ACIDWorkspace(str(sub), recorder=recorder(tmp_path))
        ws.begin("sub-rollback")
        (sub / "good.py").write_text("good\n", encoding="utf-8")
        ws.savepoint({"step": 0})
        (sub / "bad.py").write_text("bad\n", encoding="utf-8")
        (sub / "keep.txt").write_text("bad edit\n", encoding="utf-8")
        ws.savepoint({"step": 1})
        (sub / "junk.tmp").write_text("junk\n", encoding="utf-8")
        ws.rollback(0)

        assert (sub / "good.py").exists()
        assert not (sub / "bad.py").exists()
        assert not (sub / "junk.tmp").exists()
        assert (sub / "keep.txt").read_text(encoding="utf-8") == "keep v1\n"
        assert (sub / "notes.local").read_text(encoding="utf-8") == "sub notes\n"
        assert (root / "tracked.txt").read_text(encoding="utf-8") == "USER UNSAVED WORK\n"
        ws.abort()


# ──────────────────────────────────────────────
#  Bug 3: commit() includes work after the last savepoint
# ──────────────────────────────────────────────


class TestCommitIncludesTrailingWork:
    def test_changes_after_last_savepoint_are_committed(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path / "repo")
        ws = ACIDWorkspace(str(repo), recorder=recorder(tmp_path))
        ws.begin("trailing")
        (repo / "a.py").write_text("print(1)\n", encoding="utf-8")
        ws.savepoint({"s": 1})
        (repo / "b.py").write_text("print(2)\n", encoding="utf-8")
        (repo / "tracked.txt").write_text("v2 after last savepoint\n", encoding="utf-8")
        ws.commit()

        head_paths = git(repo, "ls-tree", "--name-only", "HEAD").split()
        assert {"a.py", "b.py"} <= set(head_paths)
        assert git(repo, "show", "HEAD:tracked.txt") == "v2 after last savepoint\n"
        assert git(repo, "status", "--porcelain") == ""
        assert git(repo, "symbolic-ref", "--short", "HEAD").strip() == "main"
        assert git(repo, "log", "-1", "--format=%s").strip() == "ACID COMMIT: trailing"

    def test_no_extra_commit_when_nothing_changed_after_savepoint(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path / "repo")
        ws = ACIDWorkspace(str(repo), recorder=recorder(tmp_path))
        ws.begin("no-trailing")
        (repo / "a.py").write_text("print(1)\n", encoding="utf-8")
        ws.savepoint({"s": 1})
        ws.commit()

        # init + savepoint + merge
        assert git(repo, "rev-list", "--count", "HEAD").strip() == "3"

    def test_commit_without_savepoints_commits_changes(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path / "repo")
        with ACIDWorkspace(str(repo), recorder=recorder(tmp_path)):
            (repo / "only.py").write_text("x\n", encoding="utf-8")
        assert "only.py" in git(repo, "ls-tree", "--name-only", "HEAD").split()
        assert git(repo, "status", "--porcelain") == ""


# ──────────────────────────────────────────────
#  Bug 4: transaction branch lifecycle
# ──────────────────────────────────────────────


class TestBranchLifecycle:
    def test_commit_deletes_branch_and_same_id_can_begin_again(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path / "repo")
        ws = ACIDWorkspace(str(repo), recorder=recorder(tmp_path))
        ws.begin("task1")
        (repo / "a.py").write_text("1\n", encoding="utf-8")
        ws.savepoint({})
        ws.commit()
        assert git(repo, "branch", "--list", "acid/*") == ""

        ws2 = ACIDWorkspace(str(repo), recorder=recorder(tmp_path))
        ws2.begin("task1")
        assert ws2.branch_name == "acid/task1"
        ws2.abort()
        assert git(repo, "branch", "--list", "acid/*") == ""

    def test_stale_branch_is_left_alone_and_unique_name_used(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path / "repo")
        git(repo, "branch", "acid/task1")
        stale_tip = git(repo, "rev-parse", "acid/task1").strip()

        ws = ACIDWorkspace(str(repo), recorder=recorder(tmp_path))
        ws.begin("task1")
        assert ws.branch_name == "acid/task1-2"
        (repo / "a.py").write_text("1\n", encoding="utf-8")
        ws.commit()

        assert git(repo, "rev-parse", "acid/task1").strip() == stale_tip
        assert git(repo, "branch", "--list", "acid/task1-2") == ""
        assert "a.py" in git(repo, "ls-tree", "--name-only", "main").split()


# ──────────────────────────────────────────────
#  Bug 5: detached HEAD
# ──────────────────────────────────────────────


class TestDetachedHead:
    def test_begin_refuses_detached_head(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path / "repo")
        git(repo, "checkout", "-q", "--detach")
        before = repo_state(repo)

        ws = ACIDWorkspace(str(repo), recorder=recorder(tmp_path))
        with pytest.raises(RuntimeError, match="detached HEAD"):
            ws.begin("detached")

        assert repo_state(repo) == before
        assert ws.branch_name is None


# ──────────────────────────────────────────────
#  Bug 6: subprocess hygiene
# ──────────────────────────────────────────────


class TestGitInvocation:
    def test_git_output_never_reaches_stdout(
        self, tmp_path: Path, capfd: pytest.CaptureFixture[str]
    ) -> None:
        repo = make_repo(tmp_path / "repo")
        capfd.readouterr()
        ws = ACIDWorkspace(str(repo), recorder=recorder(tmp_path))
        ws.begin("quiet")
        (repo / "a.py").write_text("1\n", encoding="utf-8")
        ws.savepoint({})
        (repo / "b.py").write_text("2\n", encoding="utf-8")
        ws.savepoint({})
        ws.rollback(0)
        ws.commit()
        ws.begin("quiet2")
        (repo / "c.py").write_text("3\n", encoding="utf-8")
        ws.abort()

        captured = capfd.readouterr()
        assert captured.out == ""

    def test_fresh_workspace_init_is_quiet(
        self, tmp_path: Path, capfd: pytest.CaptureFixture[str]
    ) -> None:
        workspace = tmp_path / "fresh"
        ws = ACIDWorkspace(str(workspace), recorder=recorder(tmp_path))
        ws.begin("fresh")
        ws.commit()
        assert capfd.readouterr().out == ""

    def test_user_hooks_signing_and_missing_identity_do_not_break_commits(
        self, tmp_path: Path
    ) -> None:
        repo = make_repo(tmp_path / "repo")
        marker = tmp_path / "hook-ran"
        hooks = repo / ".git" / "hooks"
        for hook in ("pre-commit", "commit-msg", "post-commit", "post-checkout", "pre-merge-commit"):
            script = hooks / hook
            script.write_text(f"#!/bin/sh\necho {hook} >> '{marker}'\nexit 1\n", encoding="utf-8")
            script.chmod(0o755)
        git(repo, "config", "commit.gpgSign", "true")
        git(repo, "config", "gpg.program", "false")
        # The isolated global config sets user.useConfigOnly with no identity at all.
        assert subprocess.run(
            ["git", "var", "GIT_COMMITTER_IDENT"], cwd=repo, capture_output=True
        ).returncode != 0

        ws = ACIDWorkspace(str(repo), recorder=recorder(tmp_path))
        ws.begin("hooks")
        (repo / "a.py").write_text("1\n", encoding="utf-8")
        frame = ws.savepoint({})
        ws.commit()

        assert not marker.exists()
        savepoint = frame.metadata.custom["acid_commit_hash"]
        author = git(repo, "log", "-1", "--format=%an <%ae>", savepoint).strip()
        assert author == "agent-vcr <agent-vcr@localhost>"
        assert "a.py" in git(repo, "ls-tree", "--name-only", "HEAD").split()

    def test_merge_commit_uses_user_identity_when_configured(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path / "repo")
        git(repo, "config", "user.name", "Real User")
        git(repo, "config", "user.email", "real@example.com")
        ws = ACIDWorkspace(str(repo), recorder=recorder(tmp_path))
        ws.begin("ident")
        (repo / "a.py").write_text("1\n", encoding="utf-8")
        ws.commit()
        assert git(repo, "log", "-1", "--format=%an").strip() == "Real User"

    def test_every_git_call_has_timeout_and_captured_output(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = make_repo(tmp_path / "repo")
        calls: list[dict[str, Any]] = []
        real_run = subprocess.run

        def spy(*args: Any, **kwargs: Any) -> Any:
            command = args[0] if args else kwargs.get("args")
            if isinstance(command, list) and command and command[0] == "git":
                calls.append(kwargs)
            return real_run(*args, **kwargs)

        monkeypatch.setattr(openhands.subprocess, "run", spy)
        ws = ACIDWorkspace(str(repo), recorder=recorder(tmp_path), git_timeout=17)
        ws.begin("timeouts")
        (repo / "a.py").write_text("1\n", encoding="utf-8")
        ws.savepoint({})
        ws.rollback(0)
        ws.commit()

        assert calls
        for kwargs in calls:
            assert kwargs.get("timeout") == 17
            assert kwargs.get("capture_output") is True
            assert kwargs.get("stdin") is subprocess.DEVNULL

    def test_git_timeout_must_be_positive(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="git_timeout"):
            ACIDWorkspace(str(tmp_path / "w"), recorder=recorder(tmp_path), git_timeout=0)


# ──────────────────────────────────────────────
#  Bug 7: savepoint robustness against racing files
# ──────────────────────────────────────────────


class TestSavepointRobustness:
    def _commit_count(self, repo: Path, ws: ACIDWorkspace) -> int:
        assert ws.branch_name is not None
        return int(git(repo, "rev-list", "--count", f"main..{ws.branch_name}").strip())

    def test_file_vanishing_during_escrow_does_not_fail_savepoint(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        repo = make_repo(tmp_path / "repo")
        ws = ACIDWorkspace(str(repo), recorder=recorder(tmp_path))
        ws.begin("vanish")
        ephemeral = repo / "ephemeral.log"  # ignored, so it goes through the escrow
        ephemeral.write_text("temp\n", encoding="utf-8")

        real_scan = ws._scan_local_paths

        def scan_then_delete() -> Any:
            result = real_scan()
            if ephemeral.exists():
                ephemeral.unlink()
            return result

        monkeypatch.setattr(ws, "_scan_local_paths", scan_then_delete)
        with caplog.at_level(logging.WARNING, logger=openhands.__name__):
            frame = ws.savepoint({"step": 1})

        assert frame is not None
        assert "vanished" in caplog.text
        assert self._commit_count(repo, ws) == len(ws.recorder.get_frames()) == 1
        ws.abort()

    def test_file_changing_during_escrow_is_skipped_with_warning(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        repo = make_repo(tmp_path / "repo")
        ws = ACIDWorkspace(str(repo), recorder=recorder(tmp_path))
        ws.begin("unstable")
        busy = repo / "busy.log"
        busy.write_text("start\n", encoding="utf-8")

        real_sha256 = hashlib.sha256

        class GrowingDigest:
            """Appends to the file being escrowed every time a chunk is hashed."""

            def __init__(self, data: bytes = b"") -> None:
                self._digest = real_sha256(data)
                self._touched = False

            def update(self, chunk: bytes) -> None:
                if not self._touched:
                    self._touched = True
                    with busy.open("a", encoding="utf-8") as handle:
                        handle.write("more\n")
                self._digest.update(chunk)

            def hexdigest(self) -> str:
                return self._digest.hexdigest()

        monkeypatch.setattr(openhands, "hashlib", SimpleNamespace(sha256=GrowingDigest))
        with caplog.at_level(logging.WARNING, logger=openhands.__name__):
            frame = ws.savepoint({"step": 1})
        monkeypatch.setattr(openhands, "hashlib", hashlib)

        assert frame is not None
        snapshot = ws._local_snapshots[frame.frame_id]
        assert snapshot.skipped_paths.get("busy.log") == "unstable"
        assert "unstable" in caplog.text
        assert self._commit_count(repo, ws) == 1
        ws.abort()

    def test_failed_frame_recording_leaves_no_orphan_commit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repo = make_repo(tmp_path / "repo")
        ws = ACIDWorkspace(str(repo), recorder=recorder(tmp_path))
        ws.begin("orphan")
        (repo / "a.py").write_text("1\n", encoding="utf-8")
        tip_before = git(repo, "rev-parse", "HEAD").strip()

        def explode(*args: Any, **kwargs: Any) -> Any:
            raise OSError("disk full")

        real_record_step = ws.recorder.record_step
        monkeypatch.setattr(ws.recorder, "record_step", explode)
        with pytest.raises(OSError, match="disk full"):
            ws.savepoint({"step": 1})
        monkeypatch.setattr(ws.recorder, "record_step", real_record_step)

        assert git(repo, "rev-parse", "HEAD").strip() == tip_before
        assert git(repo, "status", "--porcelain").strip() == "?? a.py"
        ws.savepoint({"step": 1})
        assert self._commit_count(repo, ws) == 1
        ws.commit()
        assert "a.py" in git(repo, "ls-tree", "--name-only", "HEAD").split()

    def test_savepoint_without_begin_is_rejected(self, tmp_path: Path) -> None:
        repo = make_repo(tmp_path / "repo")
        ws = ACIDWorkspace(str(repo), recorder=recorder(tmp_path))
        with pytest.raises(RuntimeError, match="begin"):
            ws.savepoint({})
        assert git(repo, "rev-list", "--count", "HEAD").strip() == "1"
