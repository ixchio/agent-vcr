"""
Sentinel CLI — Standalone code quality analysis for AI agent workspaces.

Run Sentinel directly on any directory without OpenHands:

    sentinel scan ./my-project
    sentinel watch ./my-project  (live mode)

Every scan is recorded to agent-vcr for audit trails.

Exit codes (``scan``; ``watch`` uses 0/1/2/4):
    0  passed — no critical or blocker issues
    1  critical issue(s) found
    2  blocker(s) found (e.g. syntax errors) — the agent should not proceed
    3  no Python files found in the target (nothing was checked)
    4  target directory does not exist or is not a directory
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from agent_vcr import VCRRecorder
from openhands_sentinel.analyzer import Severity
from openhands_sentinel.sentinel import Sentinel, SentinelConfig

EXIT_PASSED = 0
EXIT_CRITICAL = 1
EXIT_BLOCKED = 2
EXIT_NO_FILES = 3
EXIT_BAD_TARGET = 4

# Directory names never scanned (matched against paths *relative to the scan root*).
_EXCLUDED_DIRS = frozenset({"node_modules", "__pycache__", ".venv", "venv"})

# ──────────────────────────────────────────────
#  Colors for terminal output
# ──────────────────────────────────────────────

class Colors:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    GREEN = "\033[92m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"
    WHITE = "\033[97m"
    BG_RED = "\033[41m"
    BG_GREEN = "\033[42m"
    BG_YELLOW = "\033[43m"


def severity_color(severity: Severity) -> str:
    return {
        Severity.INFO: Colors.DIM,
        Severity.WARNING: Colors.YELLOW,
        Severity.CRITICAL: Colors.RED,
        Severity.BLOCKER: f"{Colors.BG_RED}{Colors.WHITE}{Colors.BOLD}",
    }[severity]


def print_banner() -> None:
    banner = f"""{Colors.CYAN}{Colors.BOLD}
╔═══════════════════════════════════════════════════════╗
║                                                       ║
║   🛡️  OPENHANDS SENTINEL                              ║
║   Real-time code quality guardian for AI agents        ║
║                                                       ║
║   Watches. Analyzes. Warns. Records.                   ║
║                                                       ║
╚═══════════════════════════════════════════════════════╝{Colors.RESET}
"""
    print(banner)


def _check_target(directory: str) -> Path | None:
    """Return the target as a Path, or print an error and return None if unusable."""
    target = Path(directory)
    if not target.exists():
        print(f"{Colors.RED}Error: Directory {directory} does not exist{Colors.RESET}")
        return None
    if not target.is_dir():
        print(f"{Colors.RED}Error: {directory} is not a directory{Colors.RESET}")
        return None
    return target


def _iter_python_files(target: Path) -> list[Path]:
    """Collect Python files under ``target``, skipping hidden/vendored dirs.

    The exclusion rules only look at path components *below* the scan root,
    so a target that itself lives under a dot-directory (``.claude/worktrees/x``,
    ``../proj``, ``/home/me/.src/proj``) is still scanned.
    """
    files: list[Path] = []
    for f in sorted(target.rglob("*.py")):
        rel_parts = f.relative_to(target).parts
        if any(part.startswith(".") or part in _EXCLUDED_DIRS for part in rel_parts):
            continue
        files.append(f)
    return files


def scan_directory(directory: str, config: SentinelConfig) -> int:
    """Scan all Python files in a directory and report violations.

    Returns one of the ``EXIT_*`` codes documented in the module docstring.
    """
    print_banner()

    # Validate before anything touches the filesystem (VCRRecorder creates
    # its output dir, which would otherwise mask a mistyped target).
    target = _check_target(directory)
    if target is None:
        return EXIT_BAD_TARGET

    py_files = _iter_python_files(target)
    if not py_files:
        print(f"  {Colors.BG_YELLOW}{Colors.BOLD} NO FILES {Colors.RESET} "
              f"No Python files found in {directory} — nothing was checked.\n")
        return EXIT_NO_FILES

    recorder = VCRRecorder(output_dir=os.path.join(directory, ".vcr"))
    sentinel = Sentinel(config=config, recorder=recorder)
    sentinel.start_session("sentinel-scan")

    print(f"{Colors.DIM}  Scanning {len(py_files)} Python files in {directory}{Colors.RESET}\n")

    start = time.perf_counter()

    for py_file in py_files:
        try:
            content = py_file.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

        rel_path = str(py_file.relative_to(target))
        result = sentinel.check_file(rel_path, content)

        if result.violations:
            print(f"  {Colors.RED}✗{Colors.RESET} {Colors.BOLD}{rel_path}{Colors.RESET}")
            for v in result.violations:
                color = severity_color(v.severity)
                loc = f"Line {v.line}: " if v.line else ""
                print(f"    {color}{v.severity.value.upper():>8}{Colors.RESET}  {loc}{v.message}")
        else:
            print(f"  {Colors.GREEN}✓{Colors.RESET} {Colors.DIM}{rel_path}{Colors.RESET}")

    elapsed = time.perf_counter() - start

    # Print report
    print(f"\n{sentinel.get_report()}")
    print(f"  {Colors.DIM}Completed in {elapsed:.2f}s{Colors.RESET}")

    # Save recording
    path = sentinel.save()
    print(f"  {Colors.DIM}Audit trail saved to {path}{Colors.RESET}\n")

    # Exit code
    if sentinel.stats.blockers > 0:
        print(f"  {Colors.BG_RED}{Colors.WHITE}{Colors.BOLD} BLOCKED {Colors.RESET} "
              f"{sentinel.stats.blockers} blocker(s) found. Agent should not proceed.\n")
        return EXIT_BLOCKED
    elif sentinel.stats.criticals > 0:
        print(f"  {Colors.RED}{Colors.BOLD} CRITICAL {Colors.RESET} "
              f"{sentinel.stats.criticals} critical issue(s) found.\n")
        return EXIT_CRITICAL
    else:
        print(f"  {Colors.BG_GREEN}{Colors.WHITE}{Colors.BOLD} PASSED {Colors.RESET} "
              f"No critical issues in {len(py_files)} file(s).\n")
        return EXIT_PASSED


def watch_directory(
    directory: str,
    config: SentinelConfig,
    interval: float = 1.0,
    once: bool = False,
) -> int:
    """Watch Python files and rescan changed files until interrupted."""
    print_banner()
    target = _check_target(directory)
    if target is None:
        return EXIT_BAD_TARGET

    recorder = VCRRecorder(output_dir=os.path.join(directory, ".vcr"))
    sentinel = Sentinel(config=config, recorder=recorder)
    sentinel.start_session("sentinel-watch")
    mtimes: dict[Path, float] = {}
    exit_code = 0

    print(f"{Colors.DIM}  Watching Python files in {directory}{Colors.RESET}\n")
    try:
        while True:
            changed: list[Path] = []
            current_files = _iter_python_files(target)
            current_set = set(current_files)
            for gone in [p for p in mtimes if p not in current_set]:
                del mtimes[gone]
                sentinel.forget_file(str(gone.relative_to(target)))
            for py_file in current_files:
                try:
                    mtime = py_file.stat().st_mtime
                except OSError:
                    continue
                if mtimes.get(py_file) != mtime:
                    mtimes[py_file] = mtime
                    changed.append(py_file)

            for py_file in changed:
                try:
                    content = py_file.read_text(encoding="utf-8", errors="ignore")
                except Exception:
                    continue
                rel_path = str(py_file.relative_to(target))
                result, warning = sentinel.check_and_warn(rel_path, content)
                status = f"{Colors.GREEN}clean{Colors.RESET}" if result.passed else f"{Colors.RED}{len(result.violations)} issue(s){Colors.RESET}"
                print(f"  {rel_path}: {status}")
                if warning:
                    code = EXIT_BLOCKED if result.blocker_count else EXIT_CRITICAL
                    exit_code = max(exit_code, code)

            if once:
                break
            time.sleep(interval)
    except KeyboardInterrupt:
        pass
    finally:
        path = sentinel.save()
        print(f"\n  {Colors.DIM}Audit trail saved to {path}{Colors.RESET}")

    return exit_code


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenHands Sentinel — Code quality guardian for AI agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  sentinel scan ./my-project          Scan a project directory
  sentinel scan . --max-lines 80      Custom function length threshold
  sentinel scan . --max-complexity 5  Strict complexity threshold

Exit codes (scan):
  0 passed   1 critical issues   2 blockers   3 no Python files found
  4 target missing / not a directory
        """,
    )

    subparsers = parser.add_subparsers(dest="command")

    # scan command
    scan_parser = subparsers.add_parser("scan", help="Scan a directory for code quality issues")
    scan_parser.add_argument("directory", default=".", nargs="?", help="Directory to scan")
    scan_parser.add_argument("--max-lines", type=int, default=50, help="Max function lines")
    scan_parser.add_argument("--max-complexity", type=int, default=10, help="Max cyclomatic complexity")
    scan_parser.add_argument("--max-file-lines", type=int, default=500, help="Max file lines")
    scan_parser.add_argument("--max-params", type=int, default=7, help="Max function parameters")

    watch_parser = subparsers.add_parser("watch", help="Watch a directory and scan changed Python files")
    watch_parser.add_argument("directory", default=".", nargs="?", help="Directory to watch")
    watch_parser.add_argument("--max-lines", type=int, default=50, help="Max function lines")
    watch_parser.add_argument("--max-complexity", type=int, default=10, help="Max cyclomatic complexity")
    watch_parser.add_argument("--max-file-lines", type=int, default=500, help="Max file lines")
    watch_parser.add_argument("--max-params", type=int, default=7, help="Max function parameters")
    watch_parser.add_argument("--interval", type=float, default=1.0, help="Polling interval in seconds")
    watch_parser.add_argument("--once", action="store_true", help="Run one watch iteration and exit")

    args = parser.parse_args()

    if args.command == "scan":
        config = SentinelConfig(
            max_function_lines=args.max_lines,
            max_complexity=args.max_complexity,
            max_file_lines=args.max_file_lines,
            max_function_params=args.max_params,
        )
        exit_code = scan_directory(args.directory, config)
        sys.exit(exit_code)
    elif args.command == "watch":
        config = SentinelConfig(
            max_function_lines=args.max_lines,
            max_complexity=args.max_complexity,
            max_file_lines=args.max_file_lines,
            max_function_params=args.max_params,
        )
        exit_code = watch_directory(args.directory, config, interval=args.interval, once=args.once)
        sys.exit(exit_code)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
