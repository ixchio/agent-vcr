"""1.7.1 install story: user-level by default, the protection rule, self-healing hook
commands, migration from 1.7.0 project hooks, ``vcr doctor``, the status line, help.

Everything runs with ``HOME`` and ``AGENT_VCR_AIRBAG_HOME`` in a tmp dir: no real
agent config is ever read or written.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from agent_vcr import cli as main_cli
from agent_vcr.airbag import agents as ag
from agent_vcr.airbag import doctor
from agent_vcr.airbag import statusline as sl
from agent_vcr.airbag.core import Airbag, Policy, ToolCall
from agent_vcr.airbag.scope import GUARD_DIR_NAME, resolve
from agent_vcr.airbag.store import ShadowStore

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git required")
SRC = str(Path(__file__).resolve().parents[2] / "src")
POSIX = sys.platform != "win32"
_GIT_ENV = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}


def _git_init(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(path)], check=True, capture_output=True,
                   timeout=60, env=_GIT_ENV)
    return path


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fresh user: tmp HOME, tmp airbag store, no agent project-dir variables."""
    user = tmp_path / "home"
    user.mkdir()
    monkeypatch.setenv("HOME", str(user))
    monkeypatch.setenv("USERPROFILE", str(user))
    monkeypatch.setenv("AGENT_VCR_AIRBAG_HOME", str(tmp_path / "airbag"))
    for var in ("XDG_CONFIG_HOME", "COPILOT_HOME", "APPDATA", "CLAUDE_PROJECT_DIR",
                "GEMINI_PROJECT_DIR", "CURSOR_PROJECT_DIR", "OPENHANDS_PROJECT_DIR",
                "AGENT_VCR_AIRBAG", "AGENT_VCR_AIRBAG_MODE", "AGENT_VCR_AIRBAG_ON_IRREVERSIBLE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setattr(ag, "_gemini_supports_ask", lambda: True)
    return user


@pytest.fixture
def repo(home: Path) -> Path:
    proj = _git_init(home / "code" / "app")
    (proj / "a.txt").write_text("a\n", encoding="utf-8")
    return proj


def _call(cwd: Path, command: str, project_dir: Path | None = None,
          call_id: str | None = None) -> ToolCall:
    return ToolCall(agent="claude", tool_name="Bash", kind="shell", cwd=str(cwd),
                    command=command, project_dir=str(project_dir) if project_dir else None,
                    call_id=call_id)


def _stores(home: Path) -> list[str]:
    base = home.parent / "airbag"
    return sorted(p.name for p in base.iterdir()) if base.exists() else []


# ── the protection rule ──────────────────────────────────────────────────

def test_git_repositories_are_protected(repo: Path) -> None:
    (repo / "sub").mkdir()
    scope = resolve(repo / "sub")
    assert scope.protected and scope.kind == "git" and scope.root == repo.resolve()


def test_plain_directories_home_and_root_are_not(home: Path) -> None:
    plain = home / "notes"
    plain.mkdir()
    assert not resolve(plain).protected and resolve(plain).kind == "plain"
    assert resolve(home).kind == "unsafe" and not resolve(home).protected
    assert not resolve(Path(home.anchor)).protected
    downloads = home / "Downloads"
    downloads.mkdir()
    assert resolve(downloads).kind == "unsafe"


def test_vcr_folder_opts_a_directory_in(home: Path) -> None:
    project = home / "scratch" / "tool"
    (project / "deep").mkdir(parents=True)
    (project / ".vcr").mkdir()
    scope = resolve(project / "deep")
    assert scope.protected and scope.kind == "opt-in" and scope.root == project.resolve()
    # A .vcr/ in the home directory (the recorder ran there) never opts in all of ~.
    (home / ".vcr").mkdir()
    other = home / "elsewhere"
    other.mkdir()
    assert not resolve(other).protected


def test_a_home_directory_git_repo_is_never_snapshotted(home: Path) -> None:
    _git_init(home)  # dotfiles repo
    (home / "notes").mkdir()
    scope = resolve(home / "notes")
    assert scope.kind == "unsafe" and not scope.protected


def test_existing_store_keeps_a_project_protected_after_git_is_gone(repo: Path) -> None:
    Airbag(repo).manual_snapshot("save")
    shutil.rmtree(repo / ".git")
    assert resolve(repo).protected and resolve(repo).kind == "store"
    shutil.rmtree(repo)
    assert resolve(repo).protected  # so `vcr undo` and the hook still find the snapshots


def test_unprotected_dir_never_snapshots_but_stops_destruction(home: Path) -> None:
    notes = home / "notes"
    notes.mkdir()
    (notes / "old").mkdir()
    bag = Airbag.for_call(_call(notes, "rm -rf old"))
    assert not bag.protected
    decision = bag.before(_call(notes, "rm -rf old"))
    assert decision.action == "ask" and decision.snapshot is None
    assert "not protected" in (decision.agent_message or "")
    # Writes go ahead silently; reads too; irreversible is stopped as everywhere.
    assert Airbag.for_call(_call(notes, "echo x > n.txt")).before(
        _call(notes, "echo x > n.txt")).action == "continue"
    assert Airbag.for_call(_call(notes, "ls")).before(_call(notes, "ls")).action == "continue"
    assert Airbag.for_call(_call(home, "git push --force")).before(
        _call(home, "git push --force")).action == "ask"
    assert _stores(home) == [GUARD_DIR_NAME]  # no snapshot store for ~ or ~/notes
    assert not list((home.parent / "airbag").glob("*/root.txt"))


def test_unprotected_never_grants_and_respects_policy(home: Path) -> None:
    notes = home / "notes"
    notes.mkdir()
    deny = Airbag.for_call(_call(notes, "rm -rf x"), policy=Policy(on_irreversible="deny"))
    assert deny.before(_call(notes, "rm -rf x")).action == "deny"
    off = Airbag.for_call(_call(notes, "rm -rf x"), policy=Policy(enabled=False))
    assert off.before(_call(notes, "rm -rf x")).action == "continue"


def test_allow_works_in_unprotected_folders(home: Path, capsys: pytest.CaptureFixture[str]
                                            ) -> None:
    notes = home / "notes"
    notes.mkdir()
    call = _call(notes, "rm -rf build", call_id="1")
    assert Airbag.for_call(call).before(call).action == "ask"
    main_cli.main(["airbag", "allow", "--project-dir", str(notes)])
    assert "Approved once: rm -rf build" in capsys.readouterr().out
    retry = _call(notes, "rm -rf build", call_id="2")
    assert Airbag.for_call(retry).before(retry).action == "continue"


def test_agent_started_in_home_working_in_a_repo_is_protected(home: Path, repo: Path) -> None:
    call = _call(repo, "rm -rf a.txt", project_dir=home)
    bag = Airbag.for_call(call)
    decision = bag.before(call)
    assert bag.protected and decision.snapshot is not None and decision.action == "continue"


def test_check_matches_the_hook_in_unprotected_folders(home: Path, repo: Path,
                                                       capsys: pytest.CaptureFixture[str]) -> None:
    notes = home / "notes"
    notes.mkdir()
    main_cli.main(["airbag", "check", "--json", "--project-dir", str(notes), "rm -rf old"])
    out = json.loads(capsys.readouterr().out)
    assert out["level"] == "irreversible" and out["protected"] is False
    main_cli.main(["airbag", "check", "--json", "--project-dir", str(repo), "rm -rf old"])
    assert json.loads(capsys.readouterr().out)["level"] == "destructive"


# ── hook commands that survive a missing interpreter ─────────────────────

def _sh(command: str, payload: dict[str, Any], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, shell=True, input=json.dumps(payload), capture_output=True,
                          encoding="utf-8", timeout=60, cwd=str(cwd),
                          env={**os.environ, "PYTHONPATH": SRC})


@pytest.mark.skipif(not POSIX, reason="the guard is POSIX sh")
@pytest.mark.parametrize("agent", ["claude", "codex", "gemini", "devin", "openhands", "copilot"])
def test_missing_interpreter_is_a_silent_noop(agent: str, home: Path) -> None:
    command = ag.hook_command(agent, python="/nonexistent/venv/bin/python")
    assert ag.MARKER.strip() in command  # still found (and removed) by uninstall
    proc = _sh(command, {"tool_name": "Bash", "tool_input": {"command": "rm -rf /"}}, home)
    assert proc.returncode == 0 and proc.stdout == "" and proc.stderr == ""


@pytest.mark.skipif(not POSIX, reason="the guard is POSIX sh")
def test_cursor_still_gets_its_mandatory_answer(home: Path) -> None:
    command = ag.hook_command("cursor", "shell", python="/nonexistent/python")
    proc = _sh(command, {"command": "ls"}, home)
    assert proc.returncode == 0 and json.loads(proc.stdout) == {"permission": "allow"}


@pytest.mark.skipif(not POSIX, reason="Cline hooks are POSIX scripts")
def test_cline_script_survives_a_missing_interpreter(repo: Path, monkeypatch: pytest.MonkeyPatch
                                                     ) -> None:
    path, _ = ag.install("cline", repo)
    path.write_text(path.read_text(encoding="utf-8").replace(
        ag._interpreter("default"), "/nonexistent/python"), encoding="utf-8")
    proc = subprocess.run([str(path)], input="{}", capture_output=True, text=True, timeout=30)
    assert proc.returncode == 0 and json.loads(proc.stdout) == {"cancel": False}
    site = next(s for s in ag.find_hooks(repo) if s.agent == "cline")
    assert site.missing_interpreters == ["/nonexistent/python"] and site.guarded


@pytest.mark.parametrize("command, python", [
    ("test -x /v/bin/python && exec /v/bin/python -m agent_vcr.airbag.hook claude pre || true",
     "/v/bin/python"),
    ("test -x '/my env/python' && exec '/my env/python' -m agent_vcr.airbag.hook codex pre "
     "|| true", "/my env/python"),
    ("/old/bin/python3 -m agent_vcr.airbag.hook claude pre", "/old/bin/python3"),  # 1.7.0
    ('"C:/Program Files/Python312/python.exe" -m agent_vcr.airbag.hook claude pre',
     "C:/Program Files/Python312/python.exe"),
    ("C:/Python312/python.exe -m agent_vcr.airbag.hook gemini pre ask", "C:/Python312/python.exe"),
    ("& 'C:/It''s/python.exe' -m agent_vcr.airbag.hook copilot pre", "C:/It's/python.exe"),
    ("test -x /v/py && exec /v/py -m agent_vcr.airbag.statusline || true", "/v/py"),
])
def test_hook_interpreter_reads_every_spelling(command: str, python: str) -> None:
    assert ag.hook_interpreter(command) == python


def test_windows_commands_stay_plain_and_quoted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "win32")
    monkeypatch.setattr(sys, "executable", r"C:\Program Files\Python312\python.exe")
    monkeypatch.setattr(ag.os, "sep", "\\")
    command = ag.hook_command("claude")
    assert command == '"C:/Program Files/Python312/python.exe" -m agent_vcr.airbag.hook claude pre'
    assert ag.hook_command("copilot", shell="powershell").startswith(
        "& 'C:\\Program Files\\Python312\\python.exe'")
    assert ag.hook_interpreter(command) == "C:/Program Files/Python312/python.exe"


# ── where hooks go ───────────────────────────────────────────────────────

def test_user_level_paths_per_agent(home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    paths = {n: a.config_path("/proj", True) for n, a in ag.ADAPTERS.items()}
    assert paths["claude"] == home / ".claude" / "settings.json"
    assert paths["codex"] == home / ".codex" / "hooks.json"
    assert paths["gemini"] == home / ".gemini" / "settings.json"
    assert paths["cursor"] == home / ".cursor" / "hooks.json"
    assert paths["copilot"] == home / ".copilot" / "hooks" / "agent-vcr-airbag.json"
    assert paths["openhands"] == home / ".openhands" / "hooks.json"
    assert paths["opencode"] == home / ".config" / "opencode" / "plugin" / "agent-vcr-airbag.js"
    assert ag.ADAPTERS["claude"].config_path("/proj", False) == \
        Path("/proj/.claude/settings.local.json")  # personal, not the team's settings.json
    if POSIX:
        assert paths["devin"] == home / ".config" / "devin" / "config.json"
        assert paths["cline"] == home / "Documents" / "Cline" / "Hooks" / "PreToolUse"
    monkeypatch.setenv("COPILOT_HOME", str(home / "cop"))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home / "xdg"))
    assert ag.ADAPTERS["copilot"].config_path(".", True) == \
        home / "cop" / "hooks" / "agent-vcr-airbag.json"
    assert ag.ADAPTERS["opencode"].config_path(".", True).is_relative_to(home / "xdg")
    if sys.platform.startswith("linux"):
        (home / "xdg").mkdir()
        (home / "xdg" / "user-dirs.dirs").write_text(
            'XDG_DOCUMENTS_DIR="$HOME/Dokumente"\n', encoding="utf-8")
        assert ag.ADAPTERS["cline"].config_path(".", True) == \
            home / "Dokumente" / "Cline" / "Hooks" / "PreToolUse"


def test_devin_user_config_keeps_other_settings(home: Path) -> None:
    path = ag.ADAPTERS["devin"].config_path(".", True)
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"model": "x"}), encoding="utf-8")
    ag.install("devin", ".", user=True)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["model"] == "x" and ag.MARKER.strip() in json.dumps(data["hooks"])
    assert ag.uninstall("devin", ".", user=True) == path
    assert json.loads(path.read_text(encoding="utf-8")) == {"model": "x"}


def test_default_install_is_user_level_and_writes_nothing_in_the_repo(
        repo: Path, home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main_cli.main(["airbag", "install", "claude", "codex", "--project-dir", str(repo)])
    out = capsys.readouterr().out
    assert "user level" in out and "Codex: run /hooks" in out
    assert 'vcr airbag check "rm -rf ."' in out and "vcr undo" in out
    assert "✓ protected" in out
    assert (home / ".claude" / "settings.json").exists()
    assert (home / ".codex" / "hooks.json").exists()
    assert sorted(p.name for p in repo.iterdir()) == [".git", "a.txt"]


def test_install_json_is_machine_readable(repo: Path, home: Path,
                                          capsys: pytest.CaptureFixture[str]) -> None:
    main_cli.main(["airbag", "install", "--json", "gemini", "--project-dir", str(repo)])
    data = json.loads(capsys.readouterr().out)
    assert data["scope"] == "user" and data["installed"][0]["agent"] == "gemini"
    assert data["installed"][0]["path"] == str(home / ".gemini" / "settings.json")
    assert data["required_steps"] and "/trust" in data["required_steps"][0]
    assert data["protection"]["protected"] is True and data["moved"] == []


def test_project_install_is_personal_and_says_do_not_commit(
        repo: Path, capsys: pytest.CaptureFixture[str]) -> None:
    main_cli.main(["airbag", "install", "claude", "--project", "--project-dir", str(repo)])
    out = capsys.readouterr().out
    assert "don't commit" in out
    assert (repo / ".claude" / "settings.local.json").exists()
    assert not (repo / ".claude" / "settings.json").exists()


def test_migration_from_170_project_hooks(repo: Path, home: Path,
                                          capsys: pytest.CaptureFixture[str]) -> None:
    team_hook = {"type": "command", "command": "npm run lint"}
    stale = "/home/someone/.local/share/pipx/venvs/ai-agent-vcr/bin/python -m " \
            "agent_vcr.airbag.hook claude pre"
    (repo / ".claude").mkdir()
    (repo / ".claude" / "settings.json").write_text(json.dumps({
        "permissions": {"allow": ["Bash(ls)"]},
        "hooks": {"PreToolUse": [{"matcher": "Bash", "hooks": [
            {"type": "command", "command": stale, "timeout": 30}, team_hook]}]},
    }), encoding="utf-8")
    (repo / ".codex").mkdir()
    (repo / ".codex" / "hooks.json").write_text(json.dumps({"hooks": {"PreToolUse": [
        {"matcher": "Bash", "hooks": [{"type": "command",
                                       "command": stale.replace("claude", "codex")}]}]}}),
        encoding="utf-8")
    # Codex is not even on PATH: its leftover is still moved.
    main_cli.main(["airbag", "install", "claude", "--project-dir", str(repo)])
    out = capsys.readouterr().out
    assert "Moved from the project" in out and "commit the removal" in out
    settings = json.loads((repo / ".claude" / "settings.json").read_text(encoding="utf-8"))
    assert settings["permissions"] == {"allow": ["Bash(ls)"]}
    assert settings["hooks"]["PreToolUse"][0]["hooks"] == [team_hook]
    assert not (repo / ".codex").exists()  # only our hook was in it: nothing left behind
    assert ag.MARKER.strip() in (home / ".codex" / "hooks.json").read_text(encoding="utf-8")
    assert not [s for s in ag.find_hooks(repo) if s.scope == "project"]


def test_uninstall_removes_both_levels_and_only_ours(repo: Path, home: Path) -> None:
    ag.install("claude", repo, user=True)
    ag.install("claude", repo, user=False)
    main_cli.main(["airbag", "uninstall", "claude", "--project-dir", str(repo)])
    assert not ag.find_hooks(repo)
    assert json.loads((home / ".claude" / "settings.json").read_text(encoding="utf-8")) == {}


# ── status line ──────────────────────────────────────────────────────────

def test_statusline_is_added_only_when_none_exists(repo: Path, home: Path,
                                                   capsys: pytest.CaptureFixture[str]) -> None:
    settings = home / ".claude" / "settings.json"
    main_cli.main(["airbag", "install", "claude", "--statusline", "--project-dir", str(repo)])
    line = json.loads(settings.read_text(encoding="utf-8"))["statusLine"]
    assert line["type"] == "command" and "agent_vcr.airbag.statusline" in line["command"]
    assert "Status line: Claude Code now shows" in capsys.readouterr().out
    ag.uninstall("claude", repo, user=True)
    assert "statusLine" not in json.loads(settings.read_text(encoding="utf-8"))
    mine = {"type": "command", "command": "~/bin/my-line"}
    settings.write_text(json.dumps({"statusLine": mine}), encoding="utf-8")
    main_cli.main(["airbag", "install", "claude", "--statusline", "--project-dir", str(repo)])
    assert json.loads(settings.read_text(encoding="utf-8"))["statusLine"] == mine
    assert "you already have one" in capsys.readouterr().out
    main_cli.main(["airbag", "install", "claude", "--project-dir", str(repo)])  # no flag
    assert json.loads(settings.read_text(encoding="utf-8"))["statusLine"] == mine


def test_statusline_text(repo: Path, home: Path) -> None:
    assert sl.render(str(repo)) == "🛟 airbag · armed · no snapshots yet"
    assert _stores(home) == []  # reading the status never creates a store
    call = _call(repo, "rm -rf a.txt")
    Airbag.for_call(call).before(call)
    text = sl.render(str(repo))
    assert text.startswith("🛟 airbag · 1 snapshot · last: rm -rf a.txt (") and "ago)" in text
    notes = home / "notes"
    notes.mkdir()
    assert sl.render(str(notes)) == "🛟 airbag off here (not a git repo)"
    assert sl.render(str(home)) == "🛟 airbag off here (your home directory)"


def test_statusline_reads_claude_code_json(repo: Path, home: Path) -> None:
    payload = {"cwd": str(home), "workspace": {"current_dir": str(repo),
                                               "project_dir": str(home)}}
    proc = subprocess.run([sys.executable, "-m", "agent_vcr.airbag.statusline"],
                          input=json.dumps(payload), capture_output=True, text=True, timeout=30,
                          env={**os.environ, "PYTHONPATH": SRC}, encoding="utf-8")
    assert proc.returncode == 0 and proc.stdout.strip() == "🛟 airbag · armed · no snapshots yet"
    proc = subprocess.run([sys.executable, "-m", "agent_vcr.cli", "airbag", "statusline"],
                          input="{}", capture_output=True, text=True, timeout=30, cwd=str(home),
                          env={**os.environ, "PYTHONPATH": SRC}, encoding="utf-8")
    assert proc.stdout.strip().startswith("🛟 airbag off here")


# ── doctor ───────────────────────────────────────────────────────────────

def _no_agents_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ag, "detect_available", lambda: [])
    monkeypatch.setattr(ag, "gemini_version", lambda: (0, 37))


def test_doctor_finds_and_fixes_a_stale_interpreter(repo: Path, home: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    _no_agents_on_path(monkeypatch)
    report = doctor.diagnose(repo, path_env="")
    assert not report.healthy and any(c.id == "agents" for c in report.checks)
    ag.install("claude", repo, user=True)
    settings = home / ".claude" / "settings.json"
    settings.write_text(settings.read_text(encoding="utf-8").replace(
        ag._interpreter("default"), "/gone/venv/bin/python"), encoding="utf-8")
    report = doctor.diagnose(repo, path_env="")
    stale = [c for c in report.checks if c.status == "fail"]
    assert len(stale) == 1 and "no longer exists" in stale[0].title and stale[0].fixable
    fixed = doctor.fix(repo)
    assert fixed and "rewrote the Claude Code hook" in fixed[0]
    assert doctor.diagnose(repo, path_env="").healthy
    assert "/gone/" not in settings.read_text(encoding="utf-8")


def test_doctor_fix_moves_project_hooks_and_upgrades_170_commands(
        repo: Path, home: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _no_agents_on_path(monkeypatch)
    old = f"{sys.executable} -m agent_vcr.airbag.hook codex pre"  # 1.7.0: unguarded
    user_file = home / ".codex" / "hooks.json"
    user_file.parent.mkdir()
    user_file.write_text(json.dumps({"hooks": {"PreToolUse": [
        {"matcher": "Bash", "hooks": [{"type": "command", "command": old}]}]}}), encoding="utf-8")
    ag.install("claude", repo, user=False)
    checks = {c.id: c for c in doctor.diagnose(repo, path_env="").checks}
    assert checks[f"hook:{repo / '.claude' / 'settings.local.json'}"].status == "warn"
    if POSIX:
        assert checks[f"hook:{user_file}"].status == "warn"
    done = doctor.fix(repo)
    assert any("moved the Claude Code hook" in d for d in done)
    assert not (repo / ".claude").exists()
    sites = ag.find_hooks(repo)
    assert {s.scope for s in sites} == {"user"} and all(s.guarded for s in sites)


def test_doctor_flags_stale_launchers(home: Path, tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    gone = bin_dir / "vcr"
    gone.write_text("#!/nonexistent/python3.11\nfrom agent_vcr.cli import main\n",
                    encoding="utf-8")
    gone.chmod(0o755)
    checks = doctor._launcher_checks(str(bin_dir))
    assert checks[0].status == "fail" and "no longer exists" in checks[0].title
    assert "rm " in checks[0].fix
    # A Python that exists but whose agent_vcr package was removed (the 0.6.0 case).
    empty = tmp_path / "empty-venv"
    subprocess.run([sys.executable, "-m", "venv", "--without-pip", str(empty)], check=True,
                   capture_output=True, timeout=120)
    python = empty / ("Scripts/python.exe" if not POSIX else "bin/python")
    gone.write_text(f"#!{python}\nfrom agent_vcr.cli import main\n", encoding="utf-8")
    if POSIX:
        checks = doctor._launcher_checks(str(bin_dir))
        assert checks[0].status == "fail" and "stale" in checks[0].title
    assert doctor._launcher_checks(str(tmp_path / "nothing"))[0].status == "warn"


def test_doctor_cli_json_and_exit_codes(repo: Path, home: Path, monkeypatch: pytest.MonkeyPatch,
                                        capsys: pytest.CaptureFixture[str]) -> None:
    _no_agents_on_path(monkeypatch)
    monkeypatch.setattr(doctor, "_launcher_checks", lambda path_env=None: [])
    with pytest.raises(SystemExit) as exc:
        main_cli.main(["doctor", "--json", "--project-dir", str(repo)])
    data = json.loads(capsys.readouterr().out)
    assert exc.value.code == 1 and data["healthy"] is False
    ag.install("claude", repo, user=True)
    with pytest.raises(SystemExit) as exc:
        main_cli.main(["airbag", "doctor", "--project-dir", str(repo)])
    out = capsys.readouterr().out
    assert exc.value.code == 0 and "✓ Healthy." in out and "This folder is protected" in out
    assert "python" in {c["id"] for c in data["checks"]}


def test_install_kind() -> None:
    assert doctor.install_kind("/home/u/.local/share/pipx/venvs/ai-agent-vcr",
                               "/x/site-packages/agent_vcr/__init__.py") == "pipx"
    assert doctor.install_kind(sys.prefix, "/src/agent_vcr/__init__.py").startswith("editable")


# ── help ─────────────────────────────────────────────────────────────────

def test_top_level_help_leads_with_undo_and_groups_commands(
        capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        main_cli.main(["--help"])
    text = capsys.readouterr().out
    assert text.split("\n\n")[1].startswith("Undo anything your coding agent did")
    assert "pipx install ai-agent-vcr" in text and "vcr airbag install" in text
    order = [text.index(f"{title}:") for title, _ in main_cli.COMMAND_GROUPS]
    assert order == sorted(order) and text.index("Undo & safety:") < text.index("  tui ")
    listed = [name for _, names in main_cli.COMMAND_GROUPS for name in names]
    assert sorted(listed) == sorted(main_cli.COMMANDS - main_cli.HIDDEN_COMMANDS)
    assert len(listed) == len(set(listed))
    assert "init" not in [line.split()[0] for line in text.splitlines() if line.startswith("  ")
                          and line.split()]


@pytest.mark.parametrize("command", sorted(main_cli.COMMANDS - {"serve"}))
def test_every_command_help_still_works(command: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        main_cli.main([command, "--help"])
    assert exc.value.code == 0
    out = " ".join(capsys.readouterr().out.split())
    assert out.startswith(f"usage: vcr {command}")
    if command == "init":
        assert "Deprecated" in out and "vcr airbag install" in out


def test_init_still_works_and_says_it_is_deprecated(tmp_path: Path,
                                                    capsys: pytest.CaptureFixture[str]) -> None:
    main_cli.main(["init", "--project-dir", str(tmp_path)])
    captured = capsys.readouterr()
    assert (tmp_path / ".vcr").is_dir() and "deprecated" in captured.err


# ── fresh HOME, end to end, real subprocesses ────────────────────────────

def _vcr(*args: str, env: dict[str, str], cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-m", "agent_vcr.cli", *args], capture_output=True,
                          text=True, timeout=120, env=env, cwd=str(cwd), encoding="utf-8",
                          errors="replace")


@pytest.mark.skipif(not POSIX, reason="runs the hook command through /bin/sh")
def test_fresh_home_end_to_end(tmp_path: Path) -> None:
    home = tmp_path / "home"
    project = _git_init(home / "src" / "app")
    (project / "app.py").write_text("print('v1')\n", encoding="utf-8")
    plain = home / "Notes"
    (plain / "drafts").mkdir(parents=True)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    (bin_dir / "git").symlink_to(shutil.which("git") or "git")
    env = {"HOME": str(home), "AGENT_VCR_AIRBAG_HOME": str(tmp_path / "airbag"),
           "PATH": str(bin_dir), "PYTHONPATH": SRC, "LANG": "C.UTF-8"}

    # 1. Install (user level): nothing lands in the project.
    proc = _vcr("airbag", "install", "claude", env=env, cwd=project)
    assert proc.returncode == 0, proc.stderr
    assert "user level" in proc.stdout
    assert sorted(p.name for p in project.iterdir()) == [".git", "app.py"]
    settings = home / ".claude" / "settings.json"
    command = json.loads(settings.read_text(encoding="utf-8"))[
        "hooks"]["PreToolUse"][0]["hooks"][0]["command"]

    def hook(cwd: Path, shell_command: str) -> subprocess.CompletedProcess[str]:
        payload = {"session_id": "s", "cwd": str(cwd), "hook_event_name": "PreToolUse",
                   "tool_name": "Bash", "tool_input": {"command": shell_command}}
        return subprocess.run(["/bin/sh", "-c", command], input=json.dumps(payload),
                              capture_output=True, text=True, timeout=120, env=env,
                              cwd=str(cwd), encoding="utf-8")

    # 2. In the git repo: snapshot, then the change is undoable.
    out = hook(project, "rm -f app.py")
    assert out.returncode == 0 and "snapshot #1" in json.loads(out.stdout)["systemMessage"]
    (project / "app.py").unlink()
    assert _vcr("undo", env=env, cwd=project).returncode == 0
    assert (project / "app.py").read_text(encoding="utf-8") == "print('v1')\n"

    # 3. In a plain non-repo directory under home: no snapshot, destruction still stopped.
    out = hook(plain, "rm -rf drafts")
    decision = json.loads(out.stdout)["hookSpecificOutput"]
    assert decision["permissionDecision"] == "ask"
    out = hook(home, "git push --force")
    assert json.loads(out.stdout)["hookSpecificOutput"]["permissionDecision"] == "ask"
    assert hook(home, "ls").stdout == ""
    stores = sorted(p.name for p in (tmp_path / "airbag").iterdir())
    assert stores == [GUARD_DIR_NAME, ShadowStore(project).dir.name]
    status = _vcr("airbag", "status", env=env, cwd=plain)
    assert "not protected" in status.stdout and "mkdir .vcr" in status.stdout

    # 4. The interpreter disappears (pipx reinstall): the hook is a no-op, doctor sees it.
    settings.write_text(settings.read_text(encoding="utf-8").replace(
        ag._interpreter("default"), "/gone/venv/bin/python"), encoding="utf-8")
    command = json.loads(settings.read_text(encoding="utf-8"))[
        "hooks"]["PreToolUse"][0]["hooks"][0]["command"]
    out = hook(project, "rm -rf .")
    assert out.returncode == 0 and out.stdout == "" and out.stderr == ""
    doc = _vcr("doctor", "--json", env=env, cwd=project)
    assert doc.returncode == 1
    failing = [c for c in json.loads(doc.stdout)["checks"] if c["status"] == "fail"]
    assert any("no longer exists" in c["title"] and c["fixable"] for c in failing)
    fixed = _vcr("doctor", "--fix", env=env, cwd=project)
    assert "rewrote the Claude Code hook" in fixed.stdout
    assert "/gone/" not in settings.read_text(encoding="utf-8")
