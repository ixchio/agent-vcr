"""Decide how dangerous a shell command is, and whether a snapshot can undo it.

Four levels:

* ``read``          — no side effects worth snapshotting (``ls``, ``git status``).
* ``write``         — changes files in the project; a snapshot makes it undoable.
* ``destructive``   — deletes/overwrites project data (``rm -rf``, ``git reset
  --hard``, ``sed -i``, ``DROP TABLE`` on a SQLite file in the repo); the
  snapshot taken first makes it undoable.
* ``irreversible``  — the airbag *cannot* undo it: remote databases, cloud and
  infra deletions, force-pushes, publishing, writes outside the project,
  system commands. These should be confirmed by a human.

This is a heuristic guard, not a sandbox: it parses common shells forms
(pipelines, ``&&``, ``sudo``, ``bash -c "..."``, redirects) and errs towards
``write`` (snapshot) when unsure.
"""

from __future__ import annotations

import os
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path

LEVELS = ("read", "write", "destructive", "irreversible")
_RANK = {name: i for i, name in enumerate(LEVELS)}


@dataclass
class Verdict:
    level: str = "read"
    reasons: list[str] = field(default_factory=list)
    categories: list[str] = field(default_factory=list)
    paths: list[str] = field(default_factory=list)

    @property
    def needs_snapshot(self) -> bool:
        return self.level != "read"

    @property
    def undoable(self) -> bool:
        return self.level in {"write", "destructive"}

    def raise_to(self, level: str, reason: str | None = None, category: str | None = None) -> None:
        if _RANK[level] > _RANK[self.level]:
            self.level = level
        if reason and reason not in self.reasons:
            self.reasons.append(reason)
        if category and category not in self.categories:
            self.categories.append(category)

    def merge(self, other: Verdict) -> None:
        for reason in other.reasons:
            if reason not in self.reasons:
                self.reasons.append(reason)
        for cat in other.categories:
            if cat not in self.categories:
                self.categories.append(cat)
        self.paths.extend(p for p in other.paths if p not in self.paths)
        if _RANK[other.level] > _RANK[self.level]:
            self.level = other.level


READ_ONLY = frozenset({
    "ls", "ll", "la", "cat", "bat", "head", "tail", "less", "more", "grep", "egrep", "fgrep",
    "rg", "ag", "ack", "fd", "tree", "pwd", "echo", "printf", "which", "type", "whereis",
    "file", "stat", "wc", "du", "df", "printenv", "date", "whoami", "id", "uname", "hostname",
    "ps", "pgrep", "jq", "yq", "sort", "uniq", "cut", "tr", "diff", "cmp", "comm", "md5sum",
    "sha1sum", "sha256sum", "shasum", "basename", "dirname", "realpath", "readlink", "test",
    "[", "true", "false", "sleep", "man", "help", "cd", "nl", "column", "od", "xxd",
    "hexdump", "strings", "lsof", "env", "uptime", "free", "nproc", "seq", "expr",
})
# NB: a few of these can write with specific flags (sort -o, yq -i, uniq in out, ...);
# see _read_only_exceptions.

GIT_READ = frozenset({
    "status", "log", "diff", "show", "blame", "shortlog", "describe", "rev-parse", "ls-files",
    "ls-tree", "grep", "reflog", "cat-file", "whatchanged", "show-ref", "for-each-ref",
    "merge-base", "name-rev", "version", "help", "count-objects", "check-ignore", "fetch",
})

DB_CLIENTS = frozenset({
    "psql", "mysql", "mariadb", "mongosh", "mongo", "redis-cli", "cockroach", "sqlcmd",
    "clickhouse-client", "clickhouse", "cqlsh", "pgcli", "mycli", "litecli", "duckdb",
})

CLOUD_CLIS = frozenset({
    "aws", "gcloud", "gsutil", "az", "doctl", "fly", "flyctl", "heroku", "railway", "vercel",
    "netlify", "supabase", "firebase", "kubectl", "helm", "terraform", "tofu", "pulumi",
    "gh", "stripe", "wrangler", "neonctl", "planetscale", "pscale", "render", "linode-cli",
    "hcloud", "oci", "ibmcloud", "eksctl", "serverless", "sls", "cdk", "sam", "ansible",
    "docker", "podman", "nomad", "consul", "vault", "turso", "upstash", "mongocli", "atlas",
})

_DANGER_VERB = re.compile(
    r"^(?:-{0,2})(delete|destroy|terminate|remove|rm|rb|purge|prune|drop|reset|wipe|"
    r"uninstall|deprovision|decommission|nuke|truncate|erase|evict|drain|scale-to-zero)"
    r"(?:[-_:].*)?$",
    re.IGNORECASE,
)

_PUBLISH = {
    "npm": {"publish", "unpublish", "deprecate"}, "pnpm": {"publish"}, "yarn": {"publish"},
    "twine": {"upload"}, "cargo": {"publish", "yank"}, "gem": {"push", "yank"},
    "poetry": {"publish"}, "uv": {"publish"}, "flit": {"publish"}, "hatch": {"publish"},
    "mvn": {"deploy"}, "dotnet": {"nuget"}, "docker": {"push"}, "podman": {"push"},
}

_SQL_DANGER = [
    (re.compile(r"\bdrop\s+(table|database|schema|collection|index|view|user|role)\b", re.I),
     "SQL DROP"),
    (re.compile(r"\btruncate\s+(table\s+)?[\w.\"`]+", re.I), "SQL TRUNCATE"),
    (re.compile(r"\bdelete\s+from\s+[\w.\"`]+\s*(;|$|\"|')", re.I), "SQL DELETE without WHERE"),
    (re.compile(r"\bupdate\s+[\w.\"`]+\s+set\b(?![^;]*\bwhere\b)", re.I),
     "SQL UPDATE without WHERE"),
    (re.compile(r"\balter\s+table\s+[\w.\"`]+\s+drop\b", re.I), "SQL ALTER TABLE ... DROP"),
    (re.compile(r"\b(dropDatabase|deleteMany\(\s*\{\s*\}\s*\)|drop\(\))", re.I),
     "MongoDB drop/deleteMany({})"),
    (re.compile(r"\bflush(all|db)\b", re.I), "Redis FLUSHALL/FLUSHDB"),
]

_ORM_DESTRUCTIVE = [
    (("prisma", "migrate", "reset"), "prisma migrate reset"),
    (("prisma", "db", "push", "--force-reset"), "prisma db push --force-reset"),
    (("prisma", "db", "push", "--accept-data-loss"), "prisma db push --accept-data-loss"),
    (("rails", "db:drop"), "rails db:drop"), (("rails", "db:reset"), "rails db:reset"),
    (("rails", "db:schema:load"), "rails db:schema:load"), (("rake", "db:drop"), "rake db:drop"),
    (("rake", "db:reset"), "rake db:reset"), (("manage.py", "flush"), "django flush"),
    (("manage.py", "reset_db"), "django reset_db"), (("django-admin", "flush"), "django flush"),
    (("supabase", "db", "reset"), "supabase db reset"), (("dropdb",), "dropdb"),
    (("alembic", "downgrade", "base"), "alembic downgrade base"),
    (("knex", "migrate:rollback", "--all"), "knex rollback --all"),
    (("sequelize", "db:drop"), "sequelize db:drop"), (("drizzle-kit", "drop"), "drizzle-kit drop"),
    (("php", "artisan", "migrate:fresh"), "artisan migrate:fresh"),
    (("php", "artisan", "db:wipe"), "artisan db:wipe"),
]

_SYSTEM = frozenset({
    "shutdown", "reboot", "halt", "poweroff", "mkfs", "fdisk", "parted", "wipefs", "sfdisk",
    "mkswap", "diskutil", "format",
})

_WRAPPERS = frozenset({
    "sudo", "doas", "nohup", "time", "nice", "ionice", "command", "builtin", "exec", "stdbuf",
    "unbuffer", "caffeinate", "chronic", "env", "npx", "bunx", "pnpx", "uvx", "pipx",
})

_SHELLS = frozenset({"bash", "sh", "zsh", "dash", "ksh", "fish", "pwsh", "powershell"})
_OPERATORS = frozenset({";", "&&", "||", "|", "&", "|&", ";;", "(", ")", "\n"})
_REDIRECT_WRITE = frozenset({">", ">|", "&>"})
_SAFE_SINKS = frozenset({"/dev/null", "/dev/stdout", "/dev/stderr", "/dev/tty"})
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "0.0.0.0", ""})


def classify(command: str, cwd: str | Path | None = None, root: str | Path | None = None) -> Verdict:
    """Classify a shell command string run in ``cwd`` for project ``root``."""
    ctx = _Context(
        cwd=Path(cwd).expanduser().resolve() if cwd else Path.cwd(),
        root=Path(root).expanduser().resolve() if root else None,
    )
    verdict = Verdict()
    _classify_into(command, ctx, verdict, depth=0)
    return verdict


def classify_file_write(path: str, cwd: str | Path | None = None,
                        root: str | Path | None = None) -> Verdict:
    """Verdict for an agent's file-edit tool writing ``path``."""
    ctx = _Context(
        cwd=Path(cwd).expanduser().resolve() if cwd else Path.cwd(),
        root=Path(root).expanduser().resolve() if root else None,
    )
    verdict = Verdict(level="write", paths=[path])
    if not ctx.inside(path):
        verdict.raise_to("write", "file is outside the project (not protected by snapshots)",
                         "outside-project")
    return verdict


@dataclass
class _Context:
    cwd: Path
    root: Path | None
    #: Destructive SQL seen anywhere in the command text (e.g. piped into a
    #: client or in a heredoc); applied only if a database client runs it.
    sql_hint: str | None = None
    #: ``VAR=value`` assignments preceding the current simple command.
    env: dict[str, str] = field(default_factory=dict)

    def resolve(self, raw: str) -> Path:
        text = os.path.expandvars(raw.replace("${HOME}", "~").replace("$HOME", "~"))
        path = Path(text).expanduser()
        if not path.is_absolute():
            path = self.cwd / path
        return Path(os.path.normpath(str(path)))

    def inside(self, raw: str) -> bool:
        if self.root is None:
            return True
        path = self.resolve(raw)
        return path == self.root or self.root in path.parents

    def is_temp(self, raw: str) -> bool:
        path = str(self.resolve(raw))
        temps = {"/tmp", "/var/tmp", "/private/tmp", "/var/folders", "/dev/shm",
                 os.environ.get("TMPDIR", "/tmp").rstrip("/")}
        return any(path == t or path.startswith(t + "/") for t in temps if t)


def _classify_into(command: str, ctx: _Context, verdict: Verdict, depth: int) -> None:
    if depth > 4 or not command.strip():
        return
    if ctx.sql_hint is None:
        ctx.sql_hint = next((label for pat, label in _SQL_DANGER if pat.search(command)), None)
    for inner in re.findall(r"\$\(([^()]*)\)|`([^`]*)`", command):
        _classify_into(inner[0] or inner[1], ctx, verdict, depth + 1)
    try:
        lexer = shlex.shlex(command, posix=True, punctuation_chars=True)
        lexer.whitespace_split = True
        lexer.commenters = ""
        tokens = list(lexer)
    except ValueError:
        verdict.raise_to("write", "could not parse command; snapshot taken to be safe")
        _regex_fallback(command, ctx, verdict)
        return
    segment: list[str] = []
    redirects: list[tuple[str, str]] = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok in _OPERATORS or tok == ";":
            _classify_segment(segment, redirects, ctx, verdict, depth)
            segment, redirects = [], []
        elif tok in _REDIRECT_WRITE or tok == ">>" or (tok.endswith(">") and tok[:-1].isdigit()):
            target = tokens[i + 1] if i + 1 < len(tokens) else ""
            redirects.append((tok, target))
            i += 1
        elif tok in {"<", "<<", "<<<", "<&", ">&"}:
            i += 1  # skip operand
        else:
            segment.append(tok)
        i += 1
    _classify_segment(segment, redirects, ctx, verdict, depth)


def _strip_wrappers(argv: list[str]) -> list[str]:
    while argv:
        head = os.path.basename(argv[0])
        if re.match(r"^[A-Za-z_][A-Za-z0-9_]*=", argv[0]):
            argv = argv[1:]
        elif head in _WRAPPERS:
            argv = argv[1:]
            while argv and argv[0].startswith("-"):
                argv = argv[1:]  # flags of the wrapper (sudo -u x is rare in agents)
        elif head == "timeout" and len(argv) > 1:
            argv = argv[2:] if not argv[1].startswith("-") else argv[1:]
        elif head == "xargs":
            argv = argv[1:]
            while argv and argv[0].startswith("-"):
                argv = argv[1:]
        else:
            break
    return argv


def _classify_segment(argv: list[str], redirects: list[tuple[str, str]], ctx: _Context,
                      verdict: Verdict, depth: int) -> None:
    for op, target in redirects:
        if not target or target in _SAFE_SINKS or target.startswith("&"):
            continue
        if not ctx.inside(target) and not ctx.is_temp(target):
            verdict.raise_to("irreversible", f"writes {target} outside the project",
                             "outside-project")
        elif op in _REDIRECT_WRITE or op.endswith(">") and op != ">>":
            if ctx.resolve(target).exists():
                verdict.raise_to("destructive", f"overwrites {target}", "files")
            else:
                verdict.raise_to("write")
        else:
            verdict.raise_to("write")
        verdict.paths.append(target)
    ctx.env = {}
    for word in argv:
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", word)
        if not match:
            break
        ctx.env[match.group(1)] = match.group(2)
    argv = _strip_wrappers(argv)
    if not argv:
        return
    prog = os.path.basename(argv[0])
    args = argv[1:]

    if prog in {"cd", "pushd"}:
        # Later segments of `cd dir && rm -rf x` run inside dir.
        target = next((a for a in args if not a.startswith("-")), "~")
        ctx.cwd = ctx.resolve(target)
        return

    if prog in _SHELLS and "-c" in args:
        idx = args.index("-c")
        if idx + 1 < len(args):
            _classify_into(args[idx + 1], ctx, verdict, depth + 1)
        return
    if prog == "eval":
        _classify_into(" ".join(args), ctx, verdict, depth + 1)
        return

    if prog in READ_ONLY:
        _read_only_exceptions(prog, args, ctx, verdict, depth)
        return
    if prog == "find":
        if "-delete" in args or any(a in {"-exec", "-execdir", "-ok"} for a in args) and any(
            os.path.basename(a) in {"rm", "shred", "unlink"} for a in args
        ):
            verdict.raise_to("destructive", "find deletes matching files", "files")
            _check_targets([a for a in args if not a.startswith("-")][:1] or ["."],
                           ctx, verdict, "delete")
        elif any(a in {"-exec", "-execdir"} for a in args):
            verdict.raise_to("write")
        return
    if prog == "git":
        _classify_git(args, ctx, verdict)
        return
    if prog in _SYSTEM or prog.startswith("mkfs"):
        verdict.raise_to("irreversible", f"system command: {prog}", "system")
        return
    if prog in {"rm", "unlink", "shred", "srm", "rmdir"}:
        targets = [a for a in args if not a.startswith("-") or a == "-"]
        if "--no-preserve-root" in args or any(ctx.resolve(t) == Path("/") for t in targets):
            verdict.raise_to("irreversible", "deletes the filesystem root", "system")
            return
        verdict.raise_to("destructive", f"{prog} deletes files", "files")
        _check_targets(targets, ctx, verdict, "delete")
        return
    if prog in {"mv", "cp", "rsync", "install", "ln", "tee", "touch", "mkdir", "dd",
                "truncate", "chmod", "chown", "chgrp", "unzip", "tar", "patch"}:
        _classify_file_tool(prog, args, ctx, verdict)
        return
    if prog in {"sed", "perl", "ruby"} and any(a.startswith("-i") or a == "--in-place"
                                                for a in args):
        verdict.raise_to("destructive", f"{prog} edits files in place", "files")
        return
    if prog in {"kill", "killall", "pkill"}:
        if "1" in args or "-1" in args:
            verdict.raise_to("irreversible", "kills init / all processes", "system")
        else:
            verdict.raise_to("write")
        return
    if prog == "crontab" and "-r" in args:
        verdict.raise_to("irreversible", "crontab -r deletes all cron jobs", "system")
        return
    if prog in {"systemctl", "launchctl", "service"} and any(
        a in {"stop", "disable", "mask", "unload", "remove"} for a in args
    ):
        verdict.raise_to("irreversible", f"{prog} stops/disables a system service", "system")
        return

    joined = " ".join(argv)
    for words, label in _ORM_DESTRUCTIVE:
        if _contains_sequence([os.path.basename(a) for a in argv], words):
            _database_verdict(label, argv, ctx, verdict)
            return
    if prog in _PUBLISH and any(a in _PUBLISH[prog] for a in args[:2]):
        verdict.raise_to("irreversible", f"{prog} {args[0]} publishes outside this machine",
                         "publish")
        return
    if prog in DB_CLIENTS or prog in {"sqlite3", "sqlite"}:
        sql_hit = next((label for pat, label in _SQL_DANGER if pat.search(joined)), None)
        if sql_hit or ctx.sql_hint:
            _database_verdict(sql_hit or ctx.sql_hint or "destructive SQL", argv, ctx, verdict)
        else:
            verdict.raise_to("write")
        return
    if prog in CLOUD_CLIS:
        _classify_cloud(prog, args, verdict)
        return
    if prog in {"curl", "http", "https", "xh", "wget"}:
        _classify_http(args, verdict)
        return
    # Unknown programs (build tools, scripts, test runners): assume they may
    # write files — snapshot first, no prompt.
    verdict.raise_to("write")


def _option_value(args: list[str], short: str, long: str) -> str | None:
    """Value of ``-o X`` / ``-oX`` / ``--output=X`` / ``--output X``."""
    for i, arg in enumerate(args):
        if arg in {short, long} and i + 1 < len(args):
            return args[i + 1]
        if long and arg.startswith(long + "="):
            return arg.split("=", 1)[1]
        if short and arg.startswith(short) and len(arg) > len(short) and not arg.startswith("--"):
            return arg[len(short):]
    return None


def _write_target(target: str, label: str, ctx: _Context, verdict: Verdict) -> None:
    if ctx.resolve(target).exists():
        verdict.raise_to("destructive", f"{label} overwrites {target}", "files")
    else:
        verdict.raise_to("write")
    _check_targets([target], ctx, verdict, "write")


def _read_only_exceptions(prog: str, args: list[str], ctx: _Context, verdict: Verdict,
                          depth: int) -> None:
    """Usually read-only programs that write files or change the system with some flags."""
    operands = [a for a in args if not a.startswith("-")]
    if prog == "sort":
        out = _option_value(args, "-o", "--output")
        if out:
            _write_target(out, "sort -o", ctx, verdict)
    elif prog == "yq" and any(a in {"-i", "--inplace"} or a.startswith("--inplace=")
                              or (a.startswith("-") and not a.startswith("--") and "i" in a[1:])
                              for a in args):
        verdict.raise_to("destructive", "yq edits files in place", "files")
        _check_targets(operands[-1:], ctx, verdict, "write")
    elif prog == "uniq" and len(operands) >= 2:
        _write_target(operands[1], "uniq", ctx, verdict)
    elif prog == "tree":
        out = _option_value(args, "-o", "")
        if out:
            _write_target(out, "tree -o", ctx, verdict)
    elif prog == "xxd" and "-r" in args and len(operands) >= 2:
        _write_target(operands[1], "xxd -r", ctx, verdict)
    elif prog == "fd":
        for flag in ("-x", "--exec", "-X", "--exec-batch"):
            if flag in args:
                rest = args[args.index(flag) + 1:]
                _classify_into(" ".join(shlex.quote(a) for a in rest if a != ";"), ctx,
                               verdict, depth + 1)
                verdict.raise_to("write")
    elif prog == "date" and any(a in {"-s", "--set"} or a.startswith("--set=") for a in args):
        verdict.raise_to("irreversible", "date -s changes the system clock", "system")
    elif prog == "hostname" and operands:
        verdict.raise_to("irreversible", "hostname changes the system hostname", "system")


def _classify_git(args: list[str], ctx: _Context, verdict: Verdict) -> None:
    sub_idx = 0
    while sub_idx < len(args) and args[sub_idx].startswith("-"):
        sub_idx += 2 if args[sub_idx] in {"-C", "-c", "--git-dir", "--work-tree"} else 1
    if sub_idx >= len(args):
        return
    sub, rest = args[sub_idx], args[sub_idx + 1:]
    flags = set(rest)
    if sub in GIT_READ:
        return
    if sub == "branch":
        force_delete = "-D" in flags or ({"--delete", "-d"} & flags and {"--force", "-f"} & flags)
        if force_delete:
            verdict.raise_to("irreversible", "git branch -D can drop unmerged commits", "git")
        elif rest and not flags <= {"-a", "-r", "-v", "-vv", "--list", "--show-current", "-l"}:
            verdict.raise_to("write")
        return
    if sub == "push":
        if flags & {"--force", "-f", "--force-with-lease", "--mirror", "--delete", "-d"} or any(
            a.startswith("+") or a.startswith(":") or a.startswith("--force") for a in rest
        ):
            verdict.raise_to("irreversible", "force-push / remote delete rewrites shared history",
                             "git-remote")
        else:
            verdict.raise_to("write")
        return
    if sub == "reset" and "--hard" in flags:
        verdict.raise_to("destructive", "git reset --hard discards uncommitted changes", "files")
        return
    if sub == "clean" and any(a.startswith("-") and "f" in a for a in rest):
        verdict.raise_to("destructive", "git clean deletes untracked files", "files")
        return
    if sub in {"checkout", "restore"} and ("." in rest or "--" in rest or "-f" in flags
                                            or "--force" in flags):
        verdict.raise_to("destructive", f"git {sub} discards local edits", "files")
        return
    if sub == "stash" and rest[:1] in (["drop"], ["clear"]):
        verdict.raise_to("irreversible", f"git stash {rest[0]} deletes stashed work", "git")
        return
    if sub in {"filter-branch", "filter-repo"} or (sub == "reflog" and "expire" in rest) or (
        sub == "gc" and any(a.startswith("--prune") for a in rest)
    ):
        verdict.raise_to("irreversible", f"git {sub} permanently rewrites/prunes history", "git")
        return
    verdict.raise_to("write")


def _classify_file_tool(prog: str, args: list[str], ctx: _Context, verdict: Verdict) -> None:
    operands = [a for a in args if not a.startswith("-")]
    if prog == "dd":
        out = next((a[3:] for a in args if a.startswith("of=")), "")
        if out.startswith("/dev/") and out not in _SAFE_SINKS:
            verdict.raise_to("irreversible", f"dd writes the device {out}", "system")
            return
        operands = [out] if out else []
    if prog in {"rsync"} and any(a.startswith("--delete") for a in args):
        verdict.raise_to("destructive", "rsync --delete removes files", "files")
    elif prog in {"mv", "truncate", "dd"}:
        verdict.raise_to("destructive", f"{prog} can overwrite files", "files")
    elif prog in {"chmod", "chown", "chgrp"} and any(a in {"-R", "--recursive"} for a in args):
        verdict.raise_to("destructive", f"recursive {prog}", "files")
    else:
        verdict.raise_to("write")
    # Destination for cp/mv/rsync/install/ln is the last operand; others: all.
    dests = operands[-1:] if prog in {"cp", "mv", "rsync", "install", "ln"} else operands
    _check_targets(dests, ctx, verdict, "write")


def _check_targets(targets: list[str], ctx: _Context, verdict: Verdict, action: str) -> None:
    for target in targets:
        if not target or target.startswith("-") or "://" in target:
            continue
        verdict.paths.append(target)
        if not ctx.inside(target) and ctx.is_temp(target):
            continue  # scratch files outside the project (e.g. /tmp/build-cache)
        if not ctx.inside(target):
            verdict.raise_to(
                "irreversible",
                f"{action}s {target} outside the project (snapshots only cover the project)",
                "outside-project",
            )
        elif ctx.root is not None and action == "delete":
            resolved = ctx.resolve(target)
            git_dir = ctx.root / ".git"
            if resolved == ctx.root:
                verdict.raise_to("destructive", "deletes the whole project directory", "git-dir")
            elif resolved == git_dir or git_dir in resolved.parents:
                verdict.raise_to("destructive", "deletes the project's .git history", "git-dir")
            elif any(ch in target for ch in "*?[") and resolved.parent == ctx.root and \
                    target.rsplit("/", 1)[-1].startswith("."):
                verdict.raise_to("destructive", "glob may match .git", "git-dir")


def _database_verdict(label: str, argv: list[str], ctx: _Context, verdict: Verdict) -> None:
    prog = os.path.basename(argv[0])
    if prog in {"sqlite3", "sqlite", "litecli", "duckdb"}:
        db = next((a for a in argv[1:] if not a.startswith("-")), "")
        if db and db != ":memory:" and ctx.inside(db):
            verdict.raise_to("destructive", f"{label} on {db} (file snapshotted first)",
                             "database")
        elif db == ":memory:":
            verdict.raise_to("write")
        else:
            verdict.raise_to("irreversible", f"{label} on {db or 'a database'} outside the project",
                             "database")
        return
    host = _db_host(argv, ctx.env)
    where = "remote" if host not in _LOCAL_HOSTS else "local server"
    verdict.raise_to(
        "irreversible",
        f"{label} on a {where} database{f' ({host})' if host else ''} — server data is not in "
        "your project files, so the airbag cannot undo it",
        "database",
    )


def _db_host(argv: list[str], inline_env: dict[str, str] | None = None) -> str:
    for i, arg in enumerate(argv):
        if arg in {"-h", "--host", "-H"} and i + 1 < len(argv):
            return argv[i + 1]
        if arg.startswith("--host="):
            return arg.split("=", 1)[1]
        match = re.match(r"^[a-z0-9+]+://(?:[^@/]*@)?(\[[^\]]+\]|[^:/?]+)", arg, re.I)
        if match:
            return match.group(1).strip("[]")
        match = re.match(r"^(?:PGHOST|MYSQL_HOST|DB_HOST)=(.+)$", arg)
        if match:
            return match.group(1)
    env = {**os.environ, **(inline_env or {})}
    for var in ("PGHOST", "MYSQL_HOST", "DB_HOST"):
        if env.get(var):
            return env[var]
    url = env.get("DATABASE_URL", "") or env.get("POSTGRES_URL", "") or env.get("MYSQL_URL", "")
    match = re.match(r"^[a-z0-9+]+://(?:[^@/]*@)?(\[[^\]]+\]|[^:/?]+)", url, re.I)
    return match.group(1).strip("[]") if match else ""


def _classify_cloud(prog: str, args: list[str], verdict: Verdict) -> None:
    if prog in {"docker", "podman"}:
        if (len(args) >= 2 and args[0] in {"volume", "system", "image", "network"}
                and args[1] in {"rm", "prune", "remove"}) or (
                "compose" in args[:1] and "down" in args and any(a in {"-v", "--volumes"}
                                                               for a in args)):
            verdict.raise_to("irreversible", f"{prog} deletes volumes/data outside the project",
                             "cloud")
        else:
            verdict.raise_to("write")
        return
    if prog in {"terraform", "tofu", "pulumi"} and (
        "destroy" in args or "-destroy" in args or "--destroy" in args
    ):
        verdict.raise_to("irreversible", f"{prog} destroys infrastructure", "cloud")
        return
    if prog == "gh" and len(args) >= 2 and args[0] in {"repo", "release", "secret", "variable",
                                                       "issue", "pr", "workflow", "run", "gist",
                                                       "cache"}:
        if _DANGER_VERB.match(args[1]):
            verdict.raise_to("irreversible", f"gh {args[0]} {args[1]} changes GitHub", "cloud")
        else:
            verdict.raise_to("write")
        return
    if any(_DANGER_VERB.match(a) for a in args[:6]):
        verdict.raise_to("irreversible", f"{prog} {' '.join(args[:3])} deletes cloud resources",
                         "cloud")
        return
    verdict.raise_to("write")


def _classify_http(args: list[str], verdict: Verdict) -> None:
    joined = " ".join(args)
    if re.search(r"(-X\s*|--request[ =]|^|\s)(DELETE|PURGE)\b", joined):
        verdict.raise_to("irreversible", "HTTP DELETE request to a remote service", "network")
    elif re.search(r"(-X\s*|--request[ =])(POST|PUT|PATCH)\b", joined) or any(
        a in {"-d", "--data", "--data-raw", "-F", "--form", "--json"} for a in args
    ):
        verdict.raise_to("write", "HTTP request that changes remote state", "network")
    elif any(a in {"-o", "-O", "--output"} for a in args):
        verdict.raise_to("write")


def _contains_sequence(words: list[str], seq: tuple[str, ...]) -> bool:
    it = iter(words)
    return all(any(w == s for w in it) for s in seq)


def _regex_fallback(command: str, ctx: _Context, verdict: Verdict) -> None:
    if re.search(r"\brm\s+-[a-z]*r", command):
        verdict.raise_to("destructive", "recursive rm", "files")
    if re.search(r"\bgit\s+push\b.*(--force|-f\b)", command):
        verdict.raise_to("irreversible", "force-push", "git-remote")
