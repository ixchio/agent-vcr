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
import shutil
import site
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agent_vcr.airbag import agents as ag
from agent_vcr.airbag.scope import Scope, resolve

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


def _shebang(path: Path) -> str | None:
    try:
        with open(path, "rb") as fh:
            first = fh.readline(512)
    except OSError:
        return None
    if not first.startswith(b"#!"):
        return None
    text = first[2:].decode("utf-8", "replace").strip()
    if text.startswith("/usr/bin/env "):
        name = text.split()[1] if len(text.split()) > 1 else ""
        return shutil.which(name) or name
    return text.split()[0] if text else None


def _package_version(python: str) -> str | None:
    try:
        out = subprocess.run(
            [python, "-c", "import agent_vcr; print(agent_vcr.__version__)"],
            capture_output=True, text=True, timeout=15, stdin=subprocess.DEVNULL,
            encoding="utf-8", errors="replace",
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else None


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


def _launcher_checks(path_env: str | None = None) -> list[Check]:
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
        if launcher.is_symlink() and not launcher.exists():
            checks.append(Check(f"launcher:{launcher}", "fail" if first else "warn",
                                f"{where} is a broken link", role, remove))
            continue
        if sys.platform == "win32":
            checks.append(Check(f"launcher:{launcher}", "info", f"{where}", role))
            continue
        python = _shebang(launcher.resolve())
        if python is None:
            checks.append(Check(f"launcher:{launcher}", "info", f"{where} (not a Python script)",
                                role))
            continue
        if not os.path.exists(python):
            checks.append(Check(f"launcher:{launcher}", "fail" if first else "warn",
                                f"{where} points to a Python that no longer exists",
                                f"{python}; {role}", remove))
            continue
        version = _package_version(python)
        if version is None:
            checks.append(Check(f"launcher:{launcher}", "fail" if first else "warn",
                                f"{where} is stale: its Python has no agent_vcr package",
                                f"{python}; {role}", remove))
        elif first and version != __version__:
            checks.append(Check(f"launcher:{launcher}", "warn",
                                f"{where} runs Agent VCR {version}, not {__version__}",
                                f"{python}; {role}",
                                "pipx upgrade ai-agent-vcr, or remove the older install"))
        else:
            checks.append(Check(f"launcher:{launcher}", "ok", f"{where} → Agent VCR {version}",
                                role))
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
                        f"{scope.root}: {scope.why}. Irreversible and destructive commands are "
                        "still stopped for a human, but nothing is snapshotted.",
                        scope.opt_in_hint)]
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
        if s.scope != "user" or not stale:
            continue
        title = ag.ADAPTERS[s.agent].title
        if s.legacy and ag.remove_site(s):
            done.append(f"removed the old {title} hook: {_home_short(s.path)}")
        if s.agent not in reinstalled:
            path, _ = ag.install(s.agent, roots[0], user=True)
            reinstalled.add(s.agent)
            done.append(f"rewrote the {title} hook for {sys.executable}: {_home_short(path)}"
                        + (" (run /hooks in Codex again to trust it)" if s.agent == "codex"
                           else ""))
    migrated: set[str] = set()
    for s in sites:
        if s.scope == "project" and s.agent not in migrated:
            migrated.add(s.agent)
            done.extend(migrate(s.agent, roots))
    return done


def migrate(name: str, roots: list[Path]) -> list[str]:
    """Move ``name``'s project-level airbag hooks under ``roots`` to the user level."""
    done: list[str] = []
    removed: list[Path] = []
    for root in roots:
        removed.extend(p for p in ag.uninstall_paths(name, root, user=False) if p not in removed)
    if not removed:
        return done
    path, _ = ag.install(name, roots[0], user=True)
    title = ag.ADAPTERS[name].title
    for old in removed:
        done.append(f"moved the {title} hook from {old} to {_home_short(path)}")
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
