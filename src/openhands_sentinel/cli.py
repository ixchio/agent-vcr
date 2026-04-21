"""
Sentinel CLI — Standalone code quality analysis for AI agent workspaces.

Run Sentinel directly on any directory without OpenHands:

    sentinel scan ./my-project
    sentinel watch ./my-project  (live mode)

Every scan is recorded to agent-vcr for audit trails.
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


def scan_directory(directory: str, config: SentinelConfig) -> int:
    """Scan all Python files in a directory and report violations."""
    print_banner()

    recorder = VCRRecorder(output_dir=os.path.join(directory, ".vcr"))
    sentinel = Sentinel(config=config, recorder=recorder)
    sentinel.start_session("sentinel-scan")

    target = Path(directory)
    if not target.exists():
        print(f"{Colors.RED}Error: Directory {directory} does not exist{Colors.RESET}")
        return 1

    # Collect Python files
    py_files = sorted(target.rglob("*.py"))
    py_files = [
        f for f in py_files
        if not any(part.startswith(".") for part in f.parts)
        and "node_modules" not in f.parts
        and "__pycache__" not in f.parts
        and ".venv" not in f.parts
        and "venv" not in f.parts
    ]

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
        return 2
    elif sentinel.stats.criticals > 0:
        print(f"  {Colors.RED}{Colors.BOLD} CRITICAL {Colors.RESET} "
              f"{sentinel.stats.criticals} critical issue(s) found.\n")
        return 1
    else:
        print(f"  {Colors.BG_GREEN}{Colors.WHITE}{Colors.BOLD} PASSED {Colors.RESET} "
              f"No critical issues.\n")
        return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="OpenHands Sentinel — Code quality guardian for AI agents",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  sentinel scan ./my-project          Scan a project directory
  sentinel scan . --max-lines 80      Custom function length threshold
  sentinel scan . --max-complexity 5  Strict complexity threshold
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
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
