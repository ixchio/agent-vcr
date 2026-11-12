"""Rewind a *local* Postgres database too (opt-in).

A server database is not in your project files, so a file snapshot cannot undo
``DROP TABLE``. For local development databases you list explicitly, the airbag
runs ``pg_dump --format=custom`` before a destructive SQL command and stores the
dump next to the file snapshot; ``vcr undo`` / ``vcr rewind`` then put the
database back from that dump (after dumping the current state first, so the
restore can be undone too). The restore runs in one transaction: user tables,
views and sequences are dropped and the dump is replayed with ``psql``, or, if
``psql`` is missing or that fails, ``pg_restore --clean --if-exists --no-owner``.

Configuration (either or both):

* ``AGENT_VCR_AIRBAG_DATABASES`` — comma-separated ``postgresql://`` URLs;
* ``.vcr/airbag-databases.json`` in the project — ``{"databases": ["postgresql://..."]}``.

Only ``localhost``, ``127.0.0.1``, ``::1`` and unix-socket URLs are accepted:
the airbag never dumps (or restores into) a remote or production database.
Passwords are passed to ``pg_dump``/``pg_restore`` through ``PGPASSWORD``,
never on the command line, and are never written into the store.

Standard library only: this module is imported by the hook.
"""

from __future__ import annotations

import getpass
import json
import os
import posixpath
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import parse_qsl, unquote

ENV_VAR = "AGENT_VCR_AIRBAG_DATABASES"
CONFIG_FILE = Path(".vcr") / "airbag-databases.json"
PG_BIN_ENV = "AGENT_VCR_AIRBAG_PG_BIN"
TIMEOUT_ENV = "AGENT_VCR_AIRBAG_DB_TIMEOUT"
#: Hooks run with a ~30 s budget, so a dump taken inside one must be quicker.
DEFAULT_DUMP_TIMEOUT = 20.0
DEFAULT_RESTORE_TIMEOUT = 600.0

LOCAL_TCP_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})
#: Where libpq looks for the server socket when no host is given (build dependent).
DEFAULT_SOCKET_DIRS = frozenset({"/var/run/postgresql", "/run/postgresql", "/tmp"})
PG_CLIENTS = frozenset({"psql", "pgcli"})
#: Clients of other database servers: never matched to a Postgres config.
OTHER_CLIENTS = frozenset({
    "mysql", "mariadb", "mongosh", "mongo", "redis-cli", "cockroach", "sqlcmd",
    "clickhouse-client", "clickhouse", "cqlsh", "mycli", "litecli", "duckdb", "sqlite3", "sqlite",
})
#: Connection-target variables scrubbed from the environment of pg_dump/pg_restore,
#: so an inherited PGHOSTADDR/PGSERVICE cannot point the tool at another server.
_TARGET_ENV = ("PGHOST", "PGHOSTADDR", "PGPORT", "PGDATABASE", "PGSERVICE", "PGSERVICEFILE")
_URL = re.compile(r"^(?:postgres|postgresql)://(?P<netloc>[^/?#]*)(?P<path>/[^?#]*)?"
                  r"(?:\?(?P<query>[^#]*))?$", re.I)
_CONNINFO = re.compile(r"(\w+)\s*=\s*('(?:[^'\\]|\\.)*'|\S+)")
#: psql/pgcli options that take a value (so the value is not a positional dbname).
_VALUE_OPTS = frozenset({
    "-c", "--command", "-f", "--file", "-v", "--set", "--variable", "-o", "--output",
    "-L", "--log-file", "-P", "--pset", "-T", "--table-attr", "-F", "--field-separator",
    "-R", "--record-separator", "-h", "--host", "-p", "--port", "-U", "--username",
    "-d", "--dbname", "-D",
})


class DatabaseError(RuntimeError):
    """pg_dump / pg_restore failed, timed out, or is not installed."""


@dataclass(frozen=True)
class PgDatabase:
    """One local Postgres database the airbag may dump and restore."""

    dbname: str
    #: None = libpq's default unix socket; "/dir" = socket directory; else a local TCP host.
    host: str | None = None
    port: int = 5432
    user: str | None = None
    password: str | None = field(default=None, repr=False, compare=False)
    sslmode: str | None = field(default=None, compare=False)

    @property
    def key(self) -> str:
        """Stable identity (no credentials) used to match dumps to the config."""
        return f"postgresql://{self.user + '@' if self.user else ''}" \
               f"{self.host or ''}:{self.port}/{self.dbname}"

    @property
    def display(self) -> str:
        where = self.host or "default socket"
        return f"{self.dbname} ({where}:{self.port}{', user ' + self.user if self.user else ''})"

    def conninfo(self) -> str:
        """libpq conninfo without the password (it goes in the environment)."""
        parts = {"dbname": self.dbname, "port": str(self.port)}
        if self.host:
            parts["host"] = self.host
        if self.user:
            parts["user"] = self.user
        if self.sslmode:
            parts["sslmode"] = self.sslmode
        return " ".join(f"{k}={_quote(v)}" for k, v in parts.items())

    def env(self) -> dict[str, str]:
        env = {k: v for k, v in os.environ.items() if k not in _TARGET_ENV}
        if self.password is not None:
            env["PGPASSWORD"] = self.password
        env.setdefault("PGCONNECT_TIMEOUT", "5")
        return env

    def info(self) -> dict[str, Any]:
        return {"key": self.key, "dbname": self.dbname, "host": self.host, "port": self.port,
                "user": self.user}


def _quote(value: str) -> str:
    return "'" + value.replace("\\", "\\\\").replace("'", "\\'") + "'"


def redact(url: str) -> str:
    """``url`` with any password replaced by ``***`` (for messages)."""
    match = re.match(r"^([a-z0-9+.-]*://)([^/?#]*)(.*)$", url, re.I | re.S)
    if match:
        scheme, netloc, rest = match.groups()
        userinfo, at, hostport = netloc.rpartition("@")  # the host follows the LAST "@"
        if at and ":" in userinfo:
            netloc = f"{userinfo.partition(':')[0]}:***@{hostport}"
        url = scheme + netloc + rest
    return re.sub(r"(?i)([?&]password=)[^&#]*", r"\1***", url)


def is_local_host(host: str | None) -> bool:
    return not host or host.startswith("/") or host.lower() in LOCAL_TCP_HOSTS


def _refuse(url: str, host: str) -> ValueError:
    return ValueError(
        f"{redact(url)}: {host} is not a local database. The airbag only snapshots databases on "
        "localhost, 127.0.0.1, ::1 or a unix socket — it never dumps remote or production "
        "databases"
    )


def parse_url(url: str) -> PgDatabase:
    """Parse a ``postgresql://`` URL of a *local* database; ``ValueError`` otherwise."""
    url = url.strip()
    match = _URL.match(url)
    if not match:
        raise ValueError(f"{redact(url)}: not a postgresql:// URL")
    netloc = match.group("netloc") or ""
    userinfo, _, hostport = netloc.rpartition("@")
    user_raw, has_pw, pw_raw = userinfo.partition(":")
    user = unquote(user_raw) or None
    password = unquote(pw_raw) if has_pw else None
    if "," in hostport:
        raise ValueError(f"{redact(url)}: multi-host URLs are not supported")
    if hostport.startswith("["):
        host_raw, _, rest = hostport[1:].partition("]")
        port_raw = rest[1:] if rest.startswith(":") else ""
    elif ":" in hostport:
        host_raw, _, port_raw = hostport.rpartition(":")
    else:
        host_raw, port_raw = hostport, ""
    host: str | None = unquote(host_raw) or None
    dbname = unquote((match.group("path") or "").lstrip("/"))
    params = dict(parse_qsl(match.group("query") or "", keep_blank_values=True))
    if "service" in params:
        raise ValueError(f"{redact(url)}: service= is not supported; spell out host and dbname")
    host = params.get("host") or host
    port_raw = params.get("port") or port_raw
    dbname = params.get("dbname") or dbname
    user = params.get("user") or user
    if "password" in params:
        password = params["password"]
    hostaddr = params.get("hostaddr")
    if hostaddr:
        if hostaddr not in {"127.0.0.1", "::1"}:
            raise _refuse(url, hostaddr)
        host = host or hostaddr
    if host and "," in host:
        raise ValueError(f"{redact(url)}: multi-host URLs are not supported")
    if not is_local_host(host):
        raise _refuse(url, str(host))
    if host and host.startswith("/"):
        host = posixpath.normpath(host)  # a libpq socket directory: always POSIX syntax
    if not dbname:
        raise ValueError(f"{redact(url)}: no database name (postgresql://.../<dbname>)")
    try:
        port = int(port_raw) if port_raw else 5432
    except ValueError:
        raise ValueError(f"{redact(url)}: bad port {port_raw!r}") from None
    return PgDatabase(dbname=dbname, host=host, port=port, user=user, password=password,
                      sslmode=params.get("sslmode") or None)


def load_databases(root: str | Path | None) -> tuple[list[PgDatabase], list[str]]:
    """Configured local databases, plus a message for every refused/invalid entry."""
    raw: list[str] = []
    problems: list[str] = []
    env = os.environ.get(ENV_VAR, "")
    raw.extend(part for part in (p.strip() for p in env.split(",")) if part)
    if root is not None:
        path = Path(root) / CONFIG_FILE
        if path.is_file():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                entries = data.get("databases") if isinstance(data, dict) else None
                if not isinstance(entries, list) or not all(isinstance(e, str) for e in entries):
                    raise ValueError('expected {"databases": ["postgresql://..."]}')
                raw.extend(e for e in entries if e.strip())
            except (OSError, ValueError) as err:
                problems.append(f"{path}: {err}")
    found: dict[str, PgDatabase] = {}
    for url in raw:
        try:
            db = parse_url(url)
        except ValueError as err:
            problems.append(str(err))
            continue
        found.setdefault(db.key, db)
    return list(found.values()), problems


# ── which configured database does a command hit? ───────────────────────

def _same_server(a: str | None, b: str | None) -> bool:
    """Same local server for certain: both loopback TCP, or the same socket directory."""
    a_sock = a is None or a.startswith("/")
    b_sock = b is None or b.startswith("/")
    if a_sock != b_sock:
        return False
    if a_sock:
        if a is None or b is None:
            return a is None and b is None
        return posixpath.normpath(a) == posixpath.normpath(b)
    return True  # localhost / 127.0.0.1 / ::1: the port decides


def _parse_conninfo(text: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for key, value in _CONNINFO.findall(text):
        if value.startswith("'"):
            value = re.sub(r"\\(.)", r"\1", value[1:-1])
        out[key] = value
    return out


_LONG_OPTS = {"--host": "-h", "--port": "-p", "--username": "-U", "--dbname": "-d",
              "--command": "-c", "--file": "-f"}
#: libpq variables that can send a client somewhere its arguments don't say.
_REDIRECT_VARS = ("PGHOST", "PGHOSTADDR", "PGPORT", "PGDATABASE", "PGSERVICE", "PGSERVICEFILE",
                  "PGSYSCONFDIR")
_REDIRECT_RE = re.compile(r"\b(" + "|".join(_REDIRECT_VARS) + r")\b")


def _url_params(url: str) -> dict[str, str] | None:
    """The connection parameters a ``postgresql://`` URL *sets* (no defaults filled in)."""
    match = _URL.match(url.strip())
    if not match:
        return None
    out: dict[str, str] = {}
    userinfo, _, hostport = (match.group("netloc") or "").rpartition("@")
    user = unquote(userinfo.partition(":")[0])
    if user:
        out["user"] = user
    if hostport.startswith("["):
        host, _, rest = hostport[1:].partition("]")
        port = rest[1:] if rest.startswith(":") else ""
    elif ":" in hostport and "," not in hostport:
        host, _, port = hostport.rpartition(":")
    else:
        host, port = hostport, ""
    if unquote(host):
        out["host"] = unquote(host)
    if port:
        out["port"] = port
    dbname = unquote((match.group("path") or "").lstrip("/"))
    if dbname:
        out["dbname"] = dbname
    out.update(dict(parse_qsl(match.group("query") or "", keep_blank_values=True)))
    return out


def _psql_target(args: list[str], inline_env: dict[str, str],
                 command: str = "") -> tuple[PgDatabase | None, bool]:
    """Where ``psql`` with these arguments connects, and whether that is *certain*.

    Follows libpq: parameters inside a conninfo/URL database argument override
    ``-h``/``-p``/``-U``, which override ``PG*`` environment variables. The
    target is certain only when host, database and port come from the command
    itself, every host named anywhere is local, nothing (``PGHOSTADDR``,
    ``PGSERVICE``, a service file) can redirect it, and the SQL is in ``-c``
    (not a file or stdin). ``(None, False)`` = unknown or not local.
    """
    opts: dict[str, list[str]] = {}
    positional: list[str] = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--":
            positional.extend(args[i + 1:])
            break
        if arg.startswith("--") and "=" in arg:
            name, _, value = arg.partition("=")
            opts.setdefault(_LONG_OPTS.get(name, name), []).append(value)
        elif arg in _VALUE_OPTS:
            if i + 1 < len(args):
                opts.setdefault(_LONG_OPTS.get(arg, arg), []).append(args[i + 1])
            i += 1
        elif re.match(r"^-[hpUdcf].+", arg):
            opts.setdefault(arg[:2], []).append(arg[2:])
        elif not arg.startswith("-"):
            positional.append(arg)
        i += 1
    certain = all(len(opts.get(k, [])) <= 1 for k in ("-h", "-p", "-U", "-d"))
    certain = certain and bool(opts.get("-c")) and not opts.get("-f") and len(positional) <= 2
    def last(key: str) -> str | None:
        values = opts.get(key)
        return values[-1] if values else None

    host = last("-h")
    port = last("-p")
    user = last("-U") or (positional[1] if len(positional) > 1 else None)
    dbname = last("-d")
    if dbname and positional:
        certain = False
    dbname = dbname or (positional[0] if positional else None)
    hosts: list[str | None] = [host]
    if dbname and re.match(r"^postgres(ql)?://", dbname, re.I):
        conn = _url_params(dbname)
        if conn is None:
            return None, False
    elif dbname and "=" in dbname:
        conn = _parse_conninfo(dbname)
    else:
        conn = None
    if conn is not None:
        if "service" in conn:
            return None, False
        hosts += [conn.get("host"), conn.get("hostaddr")]
        host = conn.get("hostaddr") or conn.get("host") or host
        port = conn.get("port") or port
        user = conn.get("user") or user
        dbname = conn.get("dbname")  # the expanded argument replaces the database name
    env = {**os.environ, **inline_env}
    if any(env.get(v) for v in ("PGSERVICE", "PGSERVICEFILE")):
        return None, False
    if env.get("PGHOSTADDR") and not (conn and conn.get("hostaddr")):
        hosts.append(env["PGHOSTADDR"])
        certain = False
    if not host:
        host = env.get("PGHOST") or None
        hosts.append(host)
        certain = certain and "PGHOST" in inline_env
    for name in set(_REDIRECT_RE.findall(command)):
        if name not in inline_env:
            certain = False  # e.g. `export PGHOST=prod && psql ...`
    for candidate in hosts:
        if candidate and ("," in candidate or not is_local_host(candidate)):
            return None, False
    if not host:
        certain = False  # libpq's default socket: local, but not stated
    if not port:
        port = env.get("PGPORT")
        if port and "PGPORT" not in inline_env:
            certain = False
    user = user or env.get("PGUSER") or _os_user()
    if not dbname:
        dbname = env.get("PGDATABASE") or user
        certain = certain and "PGDATABASE" in inline_env
    try:
        port_num = int(port or 5432)
    except ValueError:
        return None, False
    if not dbname:
        return None, False
    if host and host.startswith("/"):
        host = posixpath.normpath(host)  # a libpq socket directory: always POSIX syntax
    return PgDatabase(dbname=dbname, host=host, port=port_num, user=user), certain


def _os_user() -> str | None:
    try:
        return getpass.getuser()
    except Exception:  # noqa: BLE001 - no user database entry
        return None


def _dotenv_url(root: Path | None) -> str | None:
    if root is None:
        return None
    try:
        text = (root / ".env").read_text(errors="replace", encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        match = re.match(r"^\s*(?:export\s+)?(DATABASE_URL|POSTGRES_URL)\s*=\s*(.*)$", line)
        if match:
            return match.group(2).strip().strip("'\"") or None
    return None


def _find(dest: PgDatabase, configs: list[PgDatabase]) -> PgDatabase | None:
    for db in configs:
        if db.dbname == dest.dbname and db.port == dest.port and _same_server(db.host, dest.host):
            return db
    return None


def resolve_target(target: dict[str, Any], configs: list[PgDatabase],
                   root: str | Path | None = None,
                   command: str = "") -> tuple[PgDatabase | None, bool]:
    """The configured database a recorded client call (``Verdict.db_targets``) acts on.

    The second value is True only when that is certain from the command itself
    (a ``psql`` call whose target is fully stated and local); only then may the
    airbag stop asking. Best guesses (ORM commands, the only configured
    database) are good enough for taking a dump, never for skipping the prompt.
    """
    if not configs:
        return None, False
    prog = os.path.basename(str(target.get("prog") or ""))
    args = [str(a) for a in target.get("args") or []]
    inline = {str(k): str(v) for k, v in (target.get("env") or {}).items()}
    env = {**os.environ, **inline}
    if prog in OTHER_CLIENTS:
        return None, False
    if prog in PG_CLIENTS:
        dest, certain = _psql_target(args, inline, command)
        if dest is None:
            return None, False
        db = _find(dest, configs)
        return db, certain and db is not None and prog == "psql"
    # ORMs and framework commands (prisma migrate reset, rails db:drop, ...) read their own
    # configuration (schema files, database.yml, --env, linked projects): never certain.
    from agent_vcr.airbag.classify import _db_host

    host = _db_host([prog, *args], inline)
    if host and not is_local_host(host):
        return None, False
    url = next((a for a in args if re.match(r"^postgres(ql)?://", a, re.I)), None) or \
        env.get("DATABASE_URL") or env.get("POSTGRES_URL") or \
        _dotenv_url(Path(root) if root is not None else None)
    if url:
        if not re.match(r"^postgres(ql)?://", url, re.I):
            return None, False  # mysql://, sqlite:, ... — not a Postgres database
        try:
            dest = parse_url(url)
        except ValueError:
            return None, False
        return _find(dest, configs), False
    return (configs[0] if len(configs) == 1 else None), False


def match_target(target: dict[str, Any], configs: list[PgDatabase],
                 root: str | Path | None = None) -> PgDatabase | None:
    """The configured database a client call probably acts on (see :func:`resolve_target`)."""
    return resolve_target(target, configs, root)[0]


@dataclass
class DumpPlan:
    """What the airbag should do about the databases a command touches."""

    #: Configured databases to dump before the command runs.
    databases: list[PgDatabase] = field(default_factory=list)
    #: True only when every database the command can reach is certainly a configured local
    #: one being dumped, and nothing else about the command is irreversible.
    covers_all: bool = False
    #: Why no dump is possible (shown to the user), when a configured database is hit.
    problem: str | None = None
    #: Why the dump cannot make the command undoable (it acts beyond the dumped database).
    caveat: str | None = None


#: SQL that acts on the whole cluster or another database: a dump of one database can't undo it.
#: Also psql meta-commands that switch connection or run unseen SQL/shell (\connect, \c, \i,
#: \!), and SQL that reaches other servers or the shell (dblink, foreign servers, COPY PROGRAM).
_BEYOND_ONE_DB = re.compile(
    r"\bdrop\s+(database|role|user|group|tablespace|owned)\b|\balter\s+(system|role|user)\b"
    r"|\bcreate\s+or\s+replace\s+database\b"
    r"|\\(c|connect|i|ir|include|include_relative)\b|\\!"
    r"|\bdblink\w*\s*\(|\bcreate\s+server\b|\bimport\s+foreign\s+schema\b"
    r"|\bcopy\b[^;]*\bprogram\b|\blo_(import|export)\b", re.I)
#: Commands that delete the database itself: the restore needs it to exist.
_DROPS_DATABASE = frozenset({"dropdb", "rails db:drop", "rake db:drop", "sequelize db:drop"})


def plan_dumps(verdict: Any, root: str | Path | None, command: str | None = None) -> DumpPlan:
    """Configured local databases a command's destructive SQL would hit."""
    targets = list(getattr(verdict, "db_targets", None) or [])
    if not targets:
        return DumpPlan()
    configs, _problems = load_databases(root)
    if not configs:
        return DumpPlan()
    chosen: dict[str, PgDatabase] = {}
    covered: set[str] = set()
    all_certain = True
    for target in targets:
        db, certain = resolve_target(target, configs, root, command or "")
        if not certain:
            all_certain = False
        if db is None:
            continue
        chosen.setdefault(db.key, db)
        if certain:
            covered.add(str(target.get("reason") or ""))
    plan = DumpPlan(databases=list(chosen.values()))
    if not plan.databases:
        return plan
    irreversible = dict(getattr(verdict, "irreversible_reasons", None) or {})
    plan.covers_all = all_certain and bool(irreversible) and all(
        category == "database" and reason in covered for reason, category in irreversible.items()
    )
    texts = [command or ""] + [" ".join(str(a) for a in t.get("args") or []) for t in targets]
    if any(str(t.get("label")) in _DROPS_DATABASE for t in targets) or \
            any(_BEYOND_ONE_DB.search(text) for text in texts):
        plan.covers_all = False
        plan.caveat = ("the command drops a whole database or acts on roles/other databases, "
                       "which a dump of one database cannot undo automatically")
    missing = missing_tools("pg_dump")
    if missing:
        plan.problem = missing
    return plan


# ── pg_dump / pg_restore ────────────────────────────────────────────────

def pg_tool(name: str) -> str | None:
    """Full path of a Postgres client tool (``AGENT_VCR_AIRBAG_PG_BIN`` first)."""
    bin_dir = os.environ.get(PG_BIN_ENV)
    if bin_dir:
        candidate = Path(bin_dir).expanduser() / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return shutil.which(name)


def missing_tools(*names: str) -> str | None:
    missing = [n for n in names if pg_tool(n) is None]
    if not missing:
        return None
    return (f"{' and '.join(missing)} not found on PATH (install the PostgreSQL client tools or "
            f"set {PG_BIN_ENV}), so the local database could not be snapshotted")


def _timeout(default: float) -> float:
    try:
        return float(os.environ.get(TIMEOUT_ENV) or default)
    except ValueError:
        return default


def _run(cmd: list[str], db: PgDatabase, timeout: float, what: str) -> None:
    try:
        proc = subprocess.run(cmd, capture_output=True, timeout=timeout, env=db.env(),
                              stdin=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        raise DatabaseError(f"{what} timed out after {timeout:g}s") from None
    except OSError as err:
        raise DatabaseError(f"{what} could not start: {err}") from None
    if proc.returncode != 0:
        lines = [ln for ln in proc.stderr.decode(errors="replace").splitlines() if ln.strip()]
        detail = "; ".join(lines[-3:]) or f"exit code {proc.returncode}"
        raise DatabaseError(f"{what} failed: {detail}")


def dump(db: PgDatabase, dest: str | Path, timeout: float | None = None) -> None:
    """``pg_dump --format=custom`` of ``db`` into ``dest`` (an existing 0600 file)."""
    exe = pg_tool("pg_dump")
    if exe is None:
        raise DatabaseError(missing_tools("pg_dump") or "pg_dump not found")
    _run([exe, "--format=custom", "--no-password", f"--file={dest}",
          f"--dbname={db.conninfo()}"], db, timeout or _timeout(DEFAULT_DUMP_TIMEOUT), "pg_dump")


#: Drops every user table, view, materialized view, sequence and foreign table (not ones that
#: belong to an extension), so tables created after the dump do not survive the restore.
#: Schemas — and their grants — stay.
_DROP_RELATIONS = """
DO $airbag$
DECLARE r record;
BEGIN
  -- Tables, views and foreign tables first: CASCADE takes their serial/identity sequences along.
  FOR r IN
    SELECT n.nspname, c.relname, c.relkind
      FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
     WHERE c.relkind IN ('r', 'p', 'v', 'm', 'f') AND NOT c.relispartition
       AND n.nspname !~ '^pg_' AND n.nspname <> 'information_schema'
       AND NOT EXISTS (SELECT 1 FROM pg_catalog.pg_depend d
                        WHERE d.classid = 'pg_catalog.pg_class'::regclass
                          AND d.objid = c.oid AND d.deptype = 'e')
  LOOP
    EXECUTE format('DROP %s IF EXISTS %I.%I CASCADE',
      CASE r.relkind WHEN 'v' THEN 'VIEW' WHEN 'm' THEN 'MATERIALIZED VIEW'
                     WHEN 'f' THEN 'FOREIGN TABLE' ELSE 'TABLE' END,
      r.nspname, r.relname);
  END LOOP;
  -- Then free-standing sequences only (owned 'a' / identity 'i' ones cannot be dropped alone).
  FOR r IN
    SELECT n.nspname, c.relname
      FROM pg_catalog.pg_class c JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
     WHERE c.relkind = 'S'
       AND n.nspname !~ '^pg_' AND n.nspname <> 'information_schema'
       AND NOT EXISTS (SELECT 1 FROM pg_catalog.pg_depend d
                        WHERE d.classid = 'pg_catalog.pg_class'::regclass
                          AND d.objid = c.oid AND d.deptype IN ('a', 'i', 'e'))
  LOOP
    EXECUTE format('DROP SEQUENCE IF EXISTS %I.%I CASCADE', r.nspname, r.relname);
  END LOOP;
END
$airbag$;
"""


def restore(db: PgDatabase, src: str | Path, timeout: float | None = None) -> str | None:
    """Put ``db`` back to the dump in ``src``, in one transaction.

    Exact when ``psql`` is available: user tables/views/sequences are dropped
    first, then the dump is replayed. Otherwise (or if that fails) falls back to
    ``pg_restore --clean --if-exists``, which recreates everything in the dump
    but leaves objects created after it; the returned note says so.
    """
    exe = pg_tool("pg_restore")
    if exe is None:
        raise DatabaseError(missing_tools("pg_restore") or "pg_restore not found")
    limit = timeout or max(_timeout(DEFAULT_DUMP_TIMEOUT), DEFAULT_RESTORE_TIMEOUT)
    psql = pg_tool("psql")
    reason = "psql not found"
    if psql is not None:
        try:
            _restore_exact(db, src, exe, psql, limit)
            return None
        except DatabaseError as err:
            reason = str(err)
    _run([exe, "--clean", "--if-exists", "--no-owner", "--single-transaction", "--no-password",
          f"--dbname={db.conninfo()}", str(src)], db, limit, "pg_restore")
    return f"tables created after the dump were left in place ({reason})"


def _restore_exact(db: PgDatabase, src: str | Path, pg_restore: str, psql: str,
                   timeout: float) -> None:
    try:
        script = subprocess.run(
            [pg_restore, "--clean", "--if-exists", "--no-owner", "--file=-", str(src)],
            capture_output=True, timeout=timeout, env=db.env(), stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError) as err:
        raise DatabaseError(f"pg_restore could not read the dump: {err}") from None
    if script.returncode != 0:
        detail = script.stderr.decode(errors="replace").strip().splitlines()[-1:]
        raise DatabaseError(f"pg_restore could not read the dump: {'; '.join(detail)}")
    payload = b"SET client_min_messages = warning;\n" + _DROP_RELATIONS.encode() + script.stdout
    try:
        proc = subprocess.run(
            [psql, "-X", "-q", "--no-password", "--single-transaction", "-v", "ON_ERROR_STOP=1",
             f"--dbname={db.conninfo()}", "-f", "-"],
            input=payload, capture_output=True, timeout=timeout, env=db.env(),
        )
    except subprocess.TimeoutExpired:
        raise DatabaseError(f"psql timed out after {timeout:g}s") from None
    except OSError as err:
        raise DatabaseError(f"psql could not start: {err}") from None
    if proc.returncode != 0:
        lines = [ln for ln in proc.stderr.decode(errors="replace").splitlines() if ln.strip()]
        raise DatabaseError(f"psql failed: {'; '.join(lines[-3:]) or proc.returncode}")
