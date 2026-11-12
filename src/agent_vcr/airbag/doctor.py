"""``vcr doctor``: is the airbag really protecting you, and if not, the exact fix.

Read-only by default. ``--fix`` does only what is safe to do automatically:
rewrite airbag hook commands whose interpreter is gone (or that predate the
self-healing form), and move project-level airbag hooks to the user level.
It never deletes a launcher, never touches another tool's hooks, and never
writes a ``.git`` directory.
"""

from __future__ import annotations

import contextlib
import os
import re
import shutil
import site
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agent_vcr.airbag import agents as ag
from agent_vcr.airbag.scope import OPT_IN_DIR, Scope, marker_path, resolve

ICONS = {"ok": "✓", "fail": "✗", "warn": "!", "info": "·"}


@dataclass
class Check:
    id: str
    status: str  # "ok" | "warn" | "fail" | "info"
    title: str
    detail: str = ""
    fix: str = ""
    #: ``vcr doctor --fix`` can repair it.
    fixable: bool = False


@dataclass
class Report:
    checks: list[Check] = field(default_factory=list)
    fixed: list[str] = field(default_factory=list)

    @property
    def healthy(self) -> bool:
        return not any(c.status == "fail" for c in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {"healthy": self.healthy, "checks": [asdict(c) for c in self.checks],
                "fixed": self.fixed}


def _home_short(path: str | Path) -> str:
    text = str(path)
    home = str(Path.home())
    return "~" + text[len(home):] if home and text.startswith(home + os.sep) else text


# ── Python / launchers / git ─────────────────────────────────────────────

def install_kind(prefix: str | None = None, module_file: str | None = None) -> str:
    """How this copy of Agent VCR was installed: pipx, venv, pip --user, editable, system."""
    prefix = prefix if prefix is not None else sys.prefix
    if module_file is None:
        import agent_vcr

        module_file = str(agent_vcr.__file__ or "")
    parts = Path(prefix).parts
    if "pipx" in parts and "venvs" in parts:
        return "pipx"
    if "site-packages" not in module_file and "dist-packages" not in module_file:
        return "editable (source checkout)"
    if prefix != getattr(sys, "base_prefix", prefix):
        return "virtualenv"
    with contextlib.suppress(Exception):
        if module_file.startswith(site.getuserbase()):
            return "pip --user"
    return "system pip"


def _python_check() -> Check:
    from agent_vcr import __version__

    kind = install_kind()
    return Check("python", "ok", f"Agent VCR {__version__} on Python {sys.version.split()[0]}",
                 f"{kind}: {sys.executable}")




#: pip/distlib launchers for interpreter paths with spaces or over 127 characters:
#: ``#!/bin/sh`` then ``'''exec' "/path/to/python" "$0" "$@"``.
_SH_EXEC = re.compile(r"""^'''exec' (?:"([^"]+)"|(\S+))""")


def launcher_python(path: Path) -> str | None:
    """The interpreter a ``vcr`` launcher script runs (for display), if it is a script."""
    try:
        with open(path, "rb") as fh:
            head = fh.read(4096).decode("utf-8", "replace").splitlines()
    except OSError:
        return None
    if not head or not head[0].startswith("#!"):
        return None
    words = head[0][2:].split()
    if words and os.path.basename(words[0]) in {"sh", "bash", "dash"}:
        for line in head[1:4]:
            match = _SH_EXEC.match(line.strip())
            if match:
                return match.group(1) or match.group(2)
        return None
    if words and os.path.basename(words[0]) == "env" and len(words) > 1:
        return shutil.which(words[-1]) or words[-1]
    return words[0] if words else None


def run_launcher(path: Path) -> tuple[int | None, str]:
    """Run ``path --version`` as your shell would; (exit code or None if it can't start, output)."""
    env = {k: v for k, v in os.environ.items() if not k.startswith("PYTHON")}
    try:
        out = subprocess.run([str(path), "--version"], capture_output=True, text=True,
                             timeout=20, stdin=subprocess.DEVNULL, env=env, encoding="utf-8",
                             errors="replace")
    except subprocess.TimeoutExpired:
        return 1, "timed out"
    except OSError as err:
        return None, str(err)
    return out.returncode, (out.stdout + out.stderr).strip()




def launchers(path_env: str | None = None) -> list[Path]:
    """Every ``vcr`` executable on PATH, in PATH order (first one wins)."""
    names = ("vcr.exe", "vcr.cmd", "vcr") if sys.platform == "win32" else ("vcr",)
    found: list[Path] = []
    for directory in (path_env if path_env is not None else os.environ.get("PATH", "")
                      ).split(os.pathsep):
        if not directory:
            continue
        for name in names:
            candidate = Path(directory) / name
            if (candidate.is_symlink() or candidate.exists()) and candidate not in found:
                found.append(candidate)
    return found


_VERSION = re.compile(r"(\d+\.\d+\.\d+\S*)")
_IMPORT_FAILED = ("No module named", "ModuleNotFoundError", "ImportError", "agent_vcr")


def _launcher_checks(path_env: str | None = None) -> list[Check]:
    """Each ``vcr`` on PATH, judged by running it: a removal is only suggested when it fails."""
    from agent_vcr import __version__

    found = launchers(path_env)
    if not found:
        return [Check("launcher", "warn", "`vcr` is not on your PATH",
                      "Agent hooks still work (they call Python directly).",
                      "pipx ensurepath   (then open a new terminal)")]
    checks: list[Check] = []
    for index, launcher in enumerate(found):
        where = _home_short(launcher)
        first = index == 0
        role = "runs when you type `vcr`" if first else "shadowed by an earlier `vcr` on PATH"
        remove = f"rm {where}   (a leftover of an old install; `pipx install ai-agent-vcr` " \
                 "for a fresh one)"
        bad = "fail" if first else "warn"
        cid = f"launcher:{launcher}"
        if launcher.is_symlink() and not launcher.exists():
            checks.append(Check(cid, bad, f"{where} is a broken link", role, remove))
            continue
        python = launcher_python(launcher.resolve())
        detail = f"{python}; {role}" if python else role
        code, output = run_launcher(launcher)
        if code is None:
            gone = python is not None and not os.path.exists(python)
            title = f"{where} points to a Python that no longer exists" if gone \
                else f"{where} cannot be started"
            checks.append(Check(cid, bad, title, f"{detail} ({output})", remove))
            continue
        if code != 0:
            last = output.splitlines()[-1] if output else f"exit code {code}"
            if python is not None and not os.path.exists(python):
                checks.append(Check(cid, bad, f"{where} points to a Python that no longer "
                                    "exists", f"{detail}; {last}", remove))
            elif any(word in output for word in _IMPORT_FAILED):
                checks.append(Check(cid, bad, f"{where} is stale: its Python has no agent_vcr "
                                    "package", f"{detail}; {last}", remove))
            else:
                checks.append(Check(cid, "warn", f"{where} exits with an error",
                                    f"{detail}; {last}",
                                    f"run `{where} --version` to see why; reinstall with "
                                    "`pipx install --force ai-agent-vcr`"))
            continue
        match = _VERSION.search(output)
        version = match.group(1) if match else None
        if version is None:
            checks.append(Check(cid, "info", f"{where} (not Agent VCR?)", f"{detail}; {output}"))
        elif first and version != __version__:
            checks.append(Check(cid, "warn", f"{where} runs Agent VCR {version}, not "
                                f"{__version__}", detail,
                                "pipx upgrade ai-agent-vcr, or remove the older install"))
        else:
            checks.append(Check(cid, "ok", f"{where} → Agent VCR {version}", role))
    return checks




def _git_check() -> Check:
    git = shutil.which("git")
    if git is None:
        return Check("git", "fail", "git is not installed",
                     "The airbag stores snapshots with git.",
                     "install git (https://git-scm.com/downloads)")
    try:
        out = subprocess.run([git, "--version"], capture_output=True, text=True, timeout=10,
                             stdin=subprocess.DEVNULL, encoding="utf-8", errors="replace")
        version = out.stdout.strip() or "git"
    except (OSError, subprocess.SubprocessError) as err:
        return Check("git", "fail", "git does not run", str(err), "reinstall git")
    return Check("git", "ok", version, git)


# ── agents ───────────────────────────────────────────────────────────────

def _sites(roots: list[Path]) -> list[ag.HookSite]:
    sites: list[ag.HookSite] = []
    seen: set[Path] = set()
    for root in roots:
        for site_ in ag.find_hooks(root):
            if site_.path not in seen:
                seen.add(site_.path)
                sites.append(site_)
    return sites


def _agent_checks(roots: list[Path], sites: list[ag.HookSite]) -> list[Check]:
    checks: list[Check] = []
    available = set(ag.detect_available())
    names = [n for n in ag.ADAPTERS if n in available or any(s.agent == n for s in sites)]
    if not sites:
        checks.append(Check("agents", "fail", "No agent is protected: no airbag hooks installed",
                            f"Agents on your PATH: {', '.join(sorted(available)) or 'none'}",
                            "vcr airbag install"))
    for name in names:
        adapter = ag.ADAPTERS[name]
        mine = [s for s in sites if s.agent == name]
        if not mine:
            checks.append(Check(f"agent:{name}", "warn",
                                f"{adapter.title}: on your PATH but not protected",
                                fix=f"vcr airbag install {name}"))
        for s in mine:
            where = _home_short(s.path)
            missing = s.missing_interpreters
            if missing:
                checks.append(Check(
                    f"hook:{s.path}", "fail",
                    f"{adapter.title}: hook calls a Python that no longer exists",
                    f"{where} → {', '.join(missing)}"
                    + ("" if s.guarded else " (every tool call fails until fixed)"),
                    "vcr doctor --fix   (or re-run `vcr airbag install`)", True))
            elif s.scope == "project":
                checks.append(Check(
                    f"hook:{s.path}", "fail" if s.legacy else "warn",
                    f"{adapter.title}: hook is in the project ({where})",
                    "It holds a path from this machine; committed, it breaks teammates' agents.",
                    "vcr doctor --fix   (moves it to your user-level config)", True))
            elif s.legacy or (not s.guarded and sys.platform != "win32"):
                checks.append(Check(
                    f"hook:{s.path}", "warn",
                    f"{adapter.title}: hook predates 1.7.1 ({where})",
                    "It fails every tool call if this Python moves; the new form degrades to "
                    "a no-op.", "vcr doctor --fix", True))
            else:
                checks.append(Check(f"hook:{s.path}", "ok",
                                    f"{adapter.title}: protected ({s.scope}-level)", where))
        checks.extend(_agent_specific(name, mine, roots))
    return checks


def _agent_specific(name: str, mine: list[ag.HookSite], roots: list[Path]) -> list[Check]:
    out: list[Check] = []
    if not mine:
        return out
    if name == "codex":
        out.append(Check("codex:trust", "info",
                         "Codex runs new or changed hooks only after you trust them",
                         fix="run /hooks in Codex and trust the Agent VCR hook"))
    elif name == "gemini":
        version = ag.gemini_version()
        if version is None:
            out.append(Check("gemini:version", "info", "Gemini CLI version unknown",
                             "≥ 0.36 is needed to ask before irreversible commands "
                             "(older versions block them)."))
        elif version >= (0, 36):
            out.append(Check("gemini:version", "ok",
                             f"Gemini CLI {version[0]}.{version[1]}: asks before irreversible "
                             "commands"))
        else:
            out.append(Check("gemini:version", "warn",
                             f"Gemini CLI {version[0]}.{version[1]} < 0.36: irreversible "
                             "commands are blocked, not asked",
                             fix="npm install -g @google/gemini-cli@latest, then "
                                 "vcr airbag install gemini"))
        out.append(Check("gemini:trust", "info",
                         "Gemini runs hooks only in trusted folders (user-level ones too)",
                         fix="run /trust in Gemini in each project folder"))
    elif name == "cline" and sys.platform == "win32":
        out.append(Check("cline:windows", "fail", "Cline hooks are not supported on Windows",
                         "Cline runs PowerShell .ps1 hooks there; the airbag installs a POSIX "
                         "script.", "vcr airbag uninstall cline --user; use `vcr airbag run --` "
                                    "around your agent instead"))
    elif name == "openhands" and any(s.scope == "user" for s in mine):
        for root in roots:
            project = ag.ADAPTERS["openhands"].config_path(root, False)
            if project.exists() and not any(s.path == project for s in mine):
                out.append(Check("openhands:shadow", "warn",
                                 f"{project} hides your user-level OpenHands hooks",
                                 "OpenHands reads only the first hooks.json it finds.",
                                 f"vcr airbag install openhands --project --project-dir {root}"))
    return out


# ── store / protection ───────────────────────────────────────────────────

def _store_checks(scope: Scope) -> list[Check]:
    from agent_vcr.airbag.statusline import count_snapshots
    from agent_vcr.airbag.store import AirbagError, ShadowStore, state_home

    if scope.protected:
        detail = f"{scope.root} ({scope.why})"
        checks = [Check("protection", "ok", "This folder is protected", detail)]
    else:
        checks = [Check("protection", "warn", "This folder is not protected",
                        f"{scope.root}: {scope.why}. Nothing is snapshotted; irreversible "
                        "commands and deleting a directory tree are stopped for a human.",
                        scope.opt_in_hint)]
    bare = scope.path / OPT_IN_DIR
    if not scope.protected and bare.is_dir() and not marker_path(scope.path).is_file():
        checks.append(Check("protection:bare-vcr", "info",
                            f"{bare} exists but is not an airbag opt-in",
                            "The recorder and `vcr init` create .vcr/ too, so it no longer "
                            "protects a folder by itself.", "vcr airbag protect"))
    if not scope.protected:
        checks.append(Check("store", "info", "Snapshot stores", str(state_home())))
        return checks
    try:
        store = ShadowStore(scope.root)
    except AirbagError as err:
        checks.append(Check("store", "warn", "No store for this folder", str(err)))
        return checks
    count = count_snapshots(store.git_dir)
    size = store.disk_usage() if store.dir.exists() else 0
    checks.append(Check("store", "info",
                        f"{count} snapshot{'s' * (count != 1)} for this project, "
                        f"{size / 1e6:.1f} MB", str(store.dir)))
    return checks


# ── run / fix ────────────────────────────────────────────────────────────

def _roots(project_dir: str | Path) -> tuple[Scope, list[Path]]:
    here = Path(os.path.abspath(Path(project_dir).expanduser()))
    scope = resolve(here)
    roots = [here]
    if scope.root != here:
        roots.append(scope.root)
    return scope, roots


def diagnose(project_dir: str | Path = ".", path_env: str | None = None) -> Report:
    scope, roots = _roots(project_dir)
    sites = _sites(roots)
    report = Report()
    report.checks.append(_python_check())
    report.checks.extend(_launcher_checks(path_env))
    report.checks.append(_git_check())
    report.checks.extend(_agent_checks(roots, sites))
    report.checks.extend(_store_checks(scope))
    return report


def fix(project_dir: str | Path = ".") -> list[str]:
    """Repair what is safe to repair; returns what was done (one line each)."""
    _, roots = _roots(project_dir)
    done: list[str] = []
    sites = _sites(roots)
    reinstalled: set[str] = set()
    for s in sites:
        stale = bool(s.missing_interpreters) or s.legacy or \
            (not s.guarded and sys.platform != "win32")
        if s.scope != "user" or not stale or s.agent in reinstalled:
            continue
        title = ag.ADAPTERS[s.agent].title
        reinstalled.add(s.agent)
        try:
            path = install_user_hook(s.agent, roots[0])
        except (OSError, ValueError) as err:
            done.append(f"{FAILED} could not rewrite the {title} hook: {err}. Nothing was "
                        "changed for it.")
            continue
        done.append(f"rewrote the {title} hook for {sys.executable}: {_home_short(path)}"
                    + (" (run /hooks in Codex again to trust it)" if s.agent == "codex"
                       else ""))
        for old in sites:  # an older user-level location, only now that the new one works
            if old.agent == s.agent and old.legacy and old.scope == "user" and \
                    old.path != path and ag.remove_site(old):
                done.append(f"removed the old {title} hook: {_home_short(old.path)}")
    migrated: set[str] = set()
    for s in sites:
        if s.scope == "project" and s.agent not in migrated:
            migrated.add(s.agent)
            try:
                done.extend(migrate(s.agent, roots))
            except (OSError, ValueError) as err:
                done.append(f"{FAILED} could not move the {ag.ADAPTERS[s.agent].title} hook "
                            f"to the user level: {err}. Nothing was changed; the project hook "
                            f"in {_home_short(s.path)} is still there.")
    return done


#: Prefix of a line in :func:`fix`'s result that reports a failure.
FAILED = "✗"


def install_user_hook(name: str, root: Path) -> Path:
    """Install ``name`` at the user level and check it really is there (raises if not)."""
    path, _ = ag.install(name, root, user=True)
    if not ag.installed_ok(name, path):
        raise ValueError(f"{path}: the airbag hook is not there after writing it")
    return path


def migrate(name: str, roots: list[Path]) -> list[str]:
    """Move ``name``'s project-level airbag hooks under ``roots`` to the user level.

    The user-level hook is written (and checked) first; only then are the
    airbag's entries removed from the project files. If the user-level write
    fails, this raises and nothing in the project has changed.
    """
    sites: list[ag.HookSite] = []
    for root in roots:
        for found in ag.find_hooks(root):
            if found.agent == name and found.scope == "project" and \
                    all(found.path != s.path for s in sites):
                sites.append(found)
    if not sites:
        return []
    path = install_user_hook(name, roots[0])
    title = ag.ADAPTERS[name].title
    done: list[str] = []
    for old in sites:
        if not ag.remove_site(old):
            continue
        shown = next((os.path.relpath(old.path, r) for r in roots
                      if old.path.is_relative_to(r)), _home_short(old.path))
        done.append(f"moved the {title} hook from {shown} to {_home_short(path)}")
    return done


def format_report(report: Report) -> str:
    lines: list[str] = []
    for check in report.checks:
        lines.append(f"  {ICONS.get(check.status, '?')} {check.title}")
        if check.detail:
            lines.append(f"      {check.detail}")
        if check.fix and check.status in {"fail", "warn", "info"}:
            lines.append(f"      fix: {check.fix}")
    problems = sum(c.status == "fail" for c in report.checks)
    if problems:
        fixable = sum(c.status == "fail" and c.fixable for c in report.checks)
        tail = f"; {fixable} can be fixed with `vcr doctor --fix`" if fixable else ""
        lines.append(f"✗ {problems} problem{'s' * (problems != 1)}{tail}.")
    else:
        lines.append("✓ Healthy.")
    return "\n".join(lines)
