"""Airbag database rewind: local Postgres dumped before destructive SQL, restored by undo.

The parsing/matching tests need nothing. The end-to-end tests start a REAL
throwaway Postgres cluster (unix socket only, a free port, this user) and stop
it afterwards; they skip when the PostgreSQL server binaries are not installed.
The system cluster (port 5432) is never touched.
"""

from __future__ import annotations

import getpass
import glob
import json
import os
import shutil
import socket
import stat
import subprocess
import tempfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from agent_vcr import cli
from agent_vcr.airbag import databases as dbs
from agent_vcr.airbag.classify import classify
from agent_vcr.airbag.core import Airbag, Policy, ToolCall

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git required")


def _pg_bin() -> Path | None:
    found = sorted(glob.glob("/usr/lib/postgresql/*/bin/pg_ctl"))
    if found:
        return Path(found[-1]).parent
    which = shutil.which("pg_ctl")
    return Path(which).parent if which else None


PG_BIN = _pg_bin()
needs_pg = pytest.mark.skipif(
    PG_BIN is None or not all((PG_BIN / t).exists() for t in ("initdb", "pg_dump", "psql")),
    reason="PostgreSQL server binaries (initdb, pg_ctl, pg_dump) not installed",
)


@pytest.fixture(autouse=True)
def _env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENT_VCR_AIRBAG_HOME", str(tmp_path / "airbag-home"))
    for var in ("AGENT_VCR_AIRBAG", "AGENT_VCR_AIRBAG_MODE", "AGENT_VCR_AIRBAG_ON_IRREVERSIBLE",
                dbs.ENV_VAR, dbs.PG_BIN_ENV, "DATABASE_URL", "POSTGRES_URL", "PGHOST",
                "PGPORT", "PGDATABASE", "PGHOSTADDR", "PGSERVICE"):
        monkeypatch.delenv(var, raising=False)


def _git(cwd: Path, *args: str) -> str:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True,
                          env=env, encoding="utf-8", errors="replace").stdout


@pytest.fixture
def project(tmp_path: Path) -> Path:
    proj = tmp_path / "proj"
    proj.mkdir()
    _git(proj, "init", "-q", "-b", "main")
    (proj / "app.py").write_text("print('v1')\n", encoding="utf-8")
    return proj


# ── URL parsing and the local-only rule ──────────────────────────────────

@pytest.mark.parametrize("url,host,port,dbname,user", [
    ("postgresql://localhost/app", "localhost", 5432, "app", None),
    ("postgres://me:pw@127.0.0.1:5433/app", "127.0.0.1", 5433, "app", "me"),
    ("postgresql://[::1]:6000/app", "::1", 6000, "app", None),
    ("postgresql:///app?host=/tmp/pgsock&port=7000", "/tmp/pgsock", 7000, "app", None),
    ("postgresql://%2Fvar%2Frun%2Fpostgresql/app", "/var/run/postgresql", 5432, "app", None),
    ("postgresql:///app", None, 5432, "app", None),
])
def test_parse_local_urls(url: str, host: str | None, port: int, dbname: str,
                          user: str | None) -> None:
    db = dbs.parse_url(url)
    assert (db.host, db.port, db.dbname, db.user) == (host, port, dbname, user)


@pytest.mark.parametrize("url", [
    "postgresql://prod.example.com/app",
    "postgresql://admin:hunter2@10.0.0.5:5432/app",
    "postgresql://localhost/app?host=db.internal",
    "postgresql://localhost/app?hostaddr=10.1.2.3",
    "postgresql://localhost,prod/app",
    "postgresql://localhost/app?service=prod",
    "mysql://localhost/app",
    "postgresql://localhost/",
])
def test_refuses_remote_and_odd_urls(url: str) -> None:
    with pytest.raises(ValueError) as err:
        dbs.parse_url(url)
    assert "hunter2" not in str(err.value)


def test_remote_refusal_message_is_clear() -> None:
    with pytest.raises(ValueError, match="never dumps remote or production"):
        dbs.parse_url("postgresql://u:secret@prod.example.com/app")


def test_password_goes_in_env_not_argv() -> None:
    db = dbs.parse_url("postgresql://me:s3cr%40t@localhost:5433/app")
    assert "s3cr" not in db.conninfo() and "s3cr" not in db.key and "s3cr" not in repr(db)
    assert db.env()["PGPASSWORD"] == "s3cr@t"
    assert dbs.redact("postgresql://me:s3cret@localhost/app") == "postgresql://me:***@localhost/app"


def test_load_from_env_and_project_file(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(dbs.ENV_VAR, "postgresql://localhost/one, postgresql://prod.example.com/x")
    (project / ".vcr").mkdir()
    (project / ".vcr" / "airbag-databases.json").write_text(
        json.dumps({"databases": ["postgresql:///two?host=/tmp/s", "postgresql://localhost/one"]}),
            encoding="utf-8")
    configs, problems = dbs.load_databases(project)
    assert [db.dbname for db in configs] == ["one", "two"]
    assert len(problems) == 1 and "prod.example.com" in problems[0]


def test_bad_project_file_is_reported(project: Path) -> None:
    (project / ".vcr").mkdir()
    (project / ".vcr" / "airbag-databases.json").write_text('{"databases": "nope"}',
                                                            encoding="utf-8")
    configs, problems = dbs.load_databases(project)
    assert configs == [] and "airbag-databases.json" in problems[0]


# ── which configured database does a command hit? ────────────────────────

CONFIG = [dbs.parse_url("postgresql:///app?host=/tmp/pgsock&port=7000")]


def _target(command: str) -> list[dbs.PgDatabase | None]:
    verdict = classify(command, cwd="/tmp", root="/home/someone/proj")
    return [dbs.match_target(t, CONFIG) for t in verdict.db_targets]


@pytest.mark.parametrize("command,hit", [
    ("psql -h /tmp/pgsock -p 7000 -d app -c 'DROP TABLE users'", True),
    ("psql -h/tmp/pgsock -p7000 app -c 'TRUNCATE users'", True),
    ("psql --host=/tmp/pgsock --port=7000 --dbname=app -c 'DROP TABLE users'", True),
    ("psql 'postgresql:///app?host=/tmp/pgsock&port=7000' -c 'DROP TABLE users'", True),
    ("psql -d 'host=/tmp/pgsock port=7000 dbname=app' -c 'DROP TABLE users'", True),
    ("PGHOST=/tmp/pgsock PGPORT=7000 psql -d app -c 'DROP TABLE users'", True),
    ("bash -c \"psql -h /tmp/pgsock -p 7000 -d app -c 'DROP TABLE users'\"", True),
    ("psql -h /tmp/pgsock -p 7001 -d app -c 'DROP TABLE users'", False),  # other server
    ("psql -h /tmp/pgsock -p 7000 -d other -c 'DROP TABLE users'", False),  # other db
    ("psql -h prod.example.com -p 7000 -d app -c 'DROP TABLE users'", False),  # remote
    ("psql -d app -c 'DROP TABLE users'", False),  # default socket + port 5432
    ("mysql -h localhost app -e 'DROP TABLE users'", False),  # not Postgres
])
def test_match_command_target(command: str, hit: bool) -> None:
    targets = _target(command)
    assert targets, command
    assert (targets[0] is not None) == hit


def test_orm_commands_use_database_url_or_the_only_config(
        monkeypatch: pytest.MonkeyPatch, project: Path) -> None:
    target = {"prog": "npx", "args": ["prisma", "migrate", "reset"], "env": {}}
    assert dbs.match_target(target, CONFIG, project) == CONFIG[0]  # the only configured db
    monkeypatch.setenv("DATABASE_URL", "postgresql://prod.example.com/app")
    assert dbs.match_target(target, CONFIG, project) is None
    monkeypatch.setenv("DATABASE_URL", "mysql://localhost/app")
    assert dbs.match_target(target, CONFIG, project) is None
    monkeypatch.delenv("DATABASE_URL")
    (project / ".env").write_text("DATABASE_URL=postgresql://localhost:5432/app\n",
                                  encoding="utf-8")
    assert dbs.match_target(target, CONFIG, project) is None  # .env says another server


def test_plan_only_downgrades_when_every_irreversible_part_is_covered(
        monkeypatch: pytest.MonkeyPatch, project: Path) -> None:
    monkeypatch.setenv(dbs.ENV_VAR, "postgresql:///app?host=/tmp/pgsock&port=7000")
    drop = "psql -h /tmp/pgsock -p 7000 -d app -c 'DROP TABLE users'"
    plan = dbs.plan_dumps(classify(drop, cwd=str(project), root=project), project)
    assert [db.dbname for db in plan.databases] == ["app"] and plan.covers_all
    mixed = drop + " && git push --force origin main"
    plan = dbs.plan_dumps(classify(mixed, cwd=str(project), root=project), project)
    assert plan.databases and not plan.covers_all
    both = drop + " && psql -h prod.example.com -d app -c 'DROP TABLE users'"
    plan = dbs.plan_dumps(classify(both, cwd=str(project), root=project), project)
    assert plan.databases and not plan.covers_all


@pytest.mark.parametrize("command", [
    "psql -h /tmp/pgsock -p 7000 -d app -c 'DROP DATABASE other'",
    "psql -h /tmp/pgsock -p 7000 -d app -c 'DROP ROLE deploy'",
    "dropdb app",
    "bin/rails db:drop",
])
def test_whole_database_or_role_drops_are_never_downgraded(
        command: str, monkeypatch: pytest.MonkeyPatch, project: Path) -> None:
    monkeypatch.setenv(dbs.ENV_VAR, "postgresql:///app?host=/tmp/pgsock&port=7000")
    verdict = classify(command, cwd=str(project), root=project)
    assert verdict.level == "irreversible", command
    plan = dbs.plan_dumps(verdict, project, command)
    assert plan.databases and not plan.covers_all and plan.caveat


def test_caveat_keeps_the_ask_but_still_dumps(project: Path,
                                              monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(dbs.ENV_VAR, "postgresql:///app?host=/tmp/pgsock&port=7000")
    monkeypatch.setattr(dbs, "pg_tool", lambda name: "/usr/bin/" + name)

    def fake_dump(db: dbs.PgDatabase, dest: object, timeout: float | None = None) -> None:
        Path(str(dest)).write_bytes(b"PGDMP fake")

    monkeypatch.setattr(dbs, "dump", fake_dump)
    bag = Airbag(project, policy=Policy())
    decision = bag.before(ToolCall(
        agent="t", tool_name="Bash", kind="shell", cwd=str(project),
        command="psql -h /tmp/pgsock -p 7000 -d app -c 'DROP DATABASE other'"))
    assert decision.action == "ask"
    assert any("was dumped, but" in r for r in decision.verdict.reasons)
    assert decision.snapshot is not None and decision.snapshot.meta["databases"]


def test_without_config_nothing_changes(project: Path) -> None:
    bag = Airbag(project, policy=Policy())
    decision = bag.before(ToolCall(agent="t", tool_name="Bash", kind="shell", cwd=str(project),
                                   command="psql -h localhost -d app -c 'DROP TABLE users'"))
    assert decision.action == "ask"
    assert decision.verdict.level == "irreversible"


def test_missing_pg_dump_falls_back_to_ask(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(dbs.ENV_VAR, "postgresql:///app?host=/tmp/pgsock&port=7000")
    monkeypatch.setattr(dbs, "pg_tool", lambda name: None)
    bag = Airbag(project, policy=Policy())
    decision = bag.before(ToolCall(
        agent="t", tool_name="Bash", kind="shell", cwd=str(project),
        command="psql -h /tmp/pgsock -p 7000 -d app -c 'DROP TABLE users'"))
    assert decision.action == "ask"
    assert any("pg_dump not found" in r for r in decision.verdict.reasons)
    assert not (bag.store.db_dir.exists() and list(bag.store.db_dir.iterdir()))


def test_check_shows_the_database_plan(project: Path, monkeypatch: pytest.MonkeyPatch,
                                       capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setenv(dbs.ENV_VAR, "postgresql:///app?host=/tmp/pgsock&port=7000")
    monkeypatch.setattr(dbs, "pg_tool", lambda name: "/usr/bin/" + name)
    cli.main(["airbag", "check", "--project-dir", str(project),
              "psql -h /tmp/pgsock -p 7000 -d app -c 'DROP TABLE users'"])
    out = capsys.readouterr().out
    assert "DESTRUCTIVE" in out and "dumped with pg_dump before it runs" in out


# ── a real throwaway Postgres ────────────────────────────────────────────

@dataclass
class Cluster:
    sock: str
    port: int
    bin: Path

    @property
    def url(self) -> str:
        return f"postgresql:///app?host={self.sock}&port={self.port}"

    def psql(self, sql: str, db: str = "app") -> str:
        out = subprocess.run(
            [str(self.bin / "psql"), "-X", "-q", "-tA", "-h", self.sock, "-p", str(self.port),
             "-d", db, "-v", "ON_ERROR_STOP=1", "-c", sql],
            check=True, capture_output=True, text=True, timeout=30,
                encoding="utf-8", errors="replace")
        return out.stdout.strip()

    def drop_command(self, table: str = "users") -> str:
        return f"psql -h {self.sock} -p {self.port} -d app -c 'DROP TABLE {table}'"


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.fixture
def pg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Cluster]:
    assert PG_BIN is not None
    data = tmp_path / "pgdata"
    # Unix socket paths are limited to ~107 bytes, so the socket gets a short directory.
    sock = tempfile.mkdtemp(prefix="vcrpg-", dir="/tmp")
    port = _free_port()
    assert port != 5432
    env = {k: v for k, v in os.environ.items() if not k.startswith("PG")}
    subprocess.run([str(PG_BIN / "initdb"), "-D", str(data), "-A", "trust", "-U",
                    getpass.getuser(), "--no-sync"], check=True, capture_output=True, env=env,
                   timeout=120)
    ctl = [str(PG_BIN / "pg_ctl"), "-D", str(data)]
    subprocess.run([*ctl, "-o", f"-p {port} -k {sock} -c listen_addresses='' -c fsync=off",
                    "-l", str(tmp_path / "pg.log"), "-w", "start"],
                   check=True, capture_output=True, env=env, timeout=120)
    try:
        cluster = Cluster(sock=sock, port=port, bin=PG_BIN)
        cluster.psql("CREATE DATABASE app", db="postgres")
        cluster.psql("CREATE TABLE users (id int PRIMARY KEY, name text);"
                     "INSERT INTO users VALUES (1, 'ada'), (2, 'linus')")
        monkeypatch.setenv(dbs.PG_BIN_ENV, str(PG_BIN))
        monkeypatch.setenv(dbs.ENV_VAR, cluster.url)
        yield cluster
    finally:
        subprocess.run([*ctl, "-m", "immediate", "stop"], capture_output=True, env=env,
                       timeout=120)
        shutil.rmtree(sock, ignore_errors=True)


def _shell(command: str, cwd: Path) -> None:
    subprocess.run(command, shell=True, cwd=cwd, check=True, capture_output=True, timeout=60)


def _tables(pg: Cluster) -> str:
    return pg.psql("SELECT coalesce(string_agg(tablename, ',' ORDER BY tablename), '') "
                   "FROM pg_tables WHERE schemaname = 'public'")


@needs_pg
def test_drop_table_then_vcr_undo_brings_rows_back(
        pg: Cluster, project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bag = Airbag(project, policy=Policy())
    command = pg.drop_command()
    decision = bag.before(ToolCall(agent="claude", tool_name="Bash", kind="shell",
                                   cwd=str(project), command=command))
    assert decision.action == "continue"  # was "ask" without the database config
    assert decision.verdict.level == "destructive"
    assert any("database snapshotted" in r for r in decision.verdict.reasons)
    assert decision.user_message and "database app snapshotted" in decision.user_message
    snap = decision.snapshot
    assert snap is not None
    entry = snap.meta["databases"][0]
    dump = bag.store.dump_file(entry)
    assert dump is not None and dump.name == f"{snap.seq:08d}-app.dump"
    assert stat.S_IMODE(dump.stat().st_mode) == 0o600
    assert "password" not in json.dumps(entry)

    _shell(command, project)  # the agent's command really runs
    assert _tables(pg) == ""
    pg.psql("CREATE TABLE later (x int)")  # made after the dump: undo removes it too

    cli.main(["undo", "--project-dir", str(project)])
    out = capsys.readouterr().out
    assert "restored database app" in out
    assert pg.psql("SELECT string_agg(name, ',' ORDER BY id) FROM users") == "ada,linus"
    assert _tables(pg) == "users"

    cli.main(["redo", "--project-dir", str(project)])  # the pre-restore dump makes it undoable
    assert _tables(pg) == "later"
    cli.main(["undo", "--project-dir", str(project)])
    assert _tables(pg) == "users"


@needs_pg
def test_rewind_restores_files_and_database(pg: Cluster, project: Path) -> None:
    bag = Airbag(project, policy=Policy())
    edit = "echo v2 > app.py"
    bag.before(ToolCall(agent="claude", tool_name="Bash", kind="shell", cwd=str(project),
                        command=edit))
    _shell(edit, project)
    first = bag.actions()[0]["snapshot"]
    truncate = f"psql -h {pg.sock} -p {pg.port} -d app -c 'TRUNCATE users'"
    decision = bag.before(ToolCall(agent="claude", tool_name="Bash", kind="shell",
                                   cwd=str(project), command=truncate))
    assert decision.action == "continue"
    _shell(truncate, project)
    assert pg.psql("SELECT count(*) FROM users") == "0"

    result = bag.rewind(int(first))  # before both actions
    assert result.databases == ["app"] and result.failed == []
    assert (project / "app.py").read_text(encoding="utf-8") == "print('v1')\n"
    assert pg.psql("SELECT count(*) FROM users") == "2"


@needs_pg
def test_db_cli_snapshot_restore_list_and_gc(
        pg: Cluster, project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cli.main(["airbag", "db", "snapshot", "--project-dir", str(project), "--json"])
    seq = json.loads(capsys.readouterr().out)["snapshot"]
    pg.psql("DELETE FROM users WHERE id = 1")

    cli.main(["airbag", "db", "restore", str(seq), "--project-dir", str(project)])
    assert "Restored database app" in capsys.readouterr().out
    assert pg.psql("SELECT count(*) FROM users") == "2"

    cli.main(["airbag", "db", "list", "--project-dir", str(project), "--json"])
    listing = json.loads(capsys.readouterr().out)
    assert [d["dbname"] for d in listing["databases"]] == ["app"]
    assert len(listing["dumps"]) == 2  # the manual one + the pre-restore safety dump

    bag = Airbag(project, policy=Policy())
    files = list(bag.store.db_dir.glob("*.dump"))
    assert len(files) == 2
    cli.main(["airbag", "gc", "--project-dir", str(project), "--keep-days", "0",
              "--keep-last", "0"])
    assert list(bag.store.db_dir.glob("*.dump")) == []


@needs_pg
def test_failed_dump_keeps_the_ask(pg: Cluster, project: Path,
                                   monkeypatch: pytest.MonkeyPatch) -> None:
    # Configured, but the database does not exist: pg_dump fails, so no downgrade.
    monkeypatch.setenv(dbs.ENV_VAR, f"postgresql:///ghost?host={pg.sock}&port={pg.port}")
    bag = Airbag(project, policy=Policy())
    decision = bag.before(ToolCall(
        agent="claude", tool_name="Bash", kind="shell", cwd=str(project),
        command=f"psql -h {pg.sock} -p {pg.port} -d ghost -c 'DROP TABLE users'"))
    assert decision.action == "ask"
    assert any("could not snapshot the local database" in r for r in decision.verdict.reasons)
    assert list(bag.store.db_dir.iterdir()) == []  # no half-written dump left behind


@needs_pg
def test_restore_refuses_without_a_safety_dump(pg: Cluster, project: Path,
                                               monkeypatch: pytest.MonkeyPatch) -> None:
    bag = Airbag(project, policy=Policy())
    bag.before(ToolCall(agent="claude", tool_name="Bash", kind="shell", cwd=str(project),
                        command=pg.drop_command()))
    _shell(pg.drop_command(), project)
    pg.psql("CREATE TABLE precious (x int); INSERT INTO precious VALUES (42)")

    def broken_dump(db: dbs.PgDatabase, dest: object, timeout: float | None = None) -> None:
        raise dbs.DatabaseError("disk full")

    monkeypatch.setattr(dbs, "dump", broken_dump)
    result = bag.undo()
    assert result is not None
    _, restore = result
    assert restore.databases == []
    assert any("could not dump its current state first" in f for f in restore.failed)
    assert pg.psql("SELECT x FROM precious") == "42"  # nothing overwritten


# ── review regressions: the downgrade needs certainty ────────────────────

def _fake_dumps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(dbs, "pg_tool", lambda name: "/usr/bin/" + name)

    def fake_dump(db: dbs.PgDatabase, dest: object, timeout: float | None = None) -> None:
        Path(str(dest)).write_bytes(b"PGDMP fake")

    monkeypatch.setattr(dbs, "dump", fake_dump)


def _decide(project: Path, command: str) -> Any:
    bag = Airbag(project, policy=Policy())
    return bag.before(ToolCall(agent="t", tool_name="Bash", kind="shell", cwd=str(project),
                               command=command))


@pytest.mark.parametrize("command", [
    "npx prisma migrate reset --force",
    "supabase db reset --linked",
    "RAILS_ENV=production rails db:reset",
    "php artisan migrate:fresh --env=production",
    "python manage.py flush",
])
def test_orm_commands_never_skip_the_prompt(command: str, project: Path,
                                            monkeypatch: pytest.MonkeyPatch) -> None:
    # One local database configured, but the ORM reads its own config (here: prod).
    monkeypatch.setenv(dbs.ENV_VAR, "postgresql://localhost:5433/app")
    (project / "prisma").mkdir()
    (project / "prisma" / ".env").write_text("DATABASE_URL=postgresql://prod.example.com/app\n",
                                             encoding="utf-8")
    _fake_dumps(monkeypatch)
    decision = _decide(project, command)
    assert decision.verdict.level == "irreversible", command
    assert decision.action == "ask"


@pytest.mark.parametrize("command", [
    # libpq: host in a conninfo / URL database argument beats -h.
    "psql -h 127.0.0.1 -p 5433 -U postgres 'host=prod.example.com dbname=app' "
    "-c 'drop table users'",
    "psql -h 127.0.0.1 -p 5433 'postgresql://prod.example.com/app' -c 'drop table users'",
    "psql -h 127.0.0.1 -p 5433 'hostaddr=10.0.0.9 dbname=app' -c 'drop table users'",
    "psql -h 127.0.0.1 -p 5433 'service=prod' -c 'drop table users'",
    "PGHOSTADDR=10.0.0.9 psql -h 127.0.0.1 -p 5433 -d app -c 'drop table users'",
    "PGSERVICE=prod psql -h 127.0.0.1 -p 5433 -d app -c 'drop table users'",
    "export PGHOST=prod.example.com; psql -p 5433 -d app -c 'drop table users'",
    "export PGHOSTADDR=10.0.0.9 && psql -h 127.0.0.1 -p 5433 -d app -c 'drop table users'",
    "psql -h 127.0.0.1 -h prod.example.com -p 5433 -d app -c 'drop table users'",
    "psql -p 5433 -d app -c 'drop table users'",  # no host: default socket, not stated
    "psql -h 127.0.0.1 -p 5433 -d app -c 'drop table users' -f more.sql",
    "psql -h 127.0.0.1 -p 5433 -d app -f drop.sql -c 'drop table users'",
    "psql -h 127.0.0.1 -p 5433 app -c '\\connect postgresql://prod.example.com/app' "
    "-c 'drop table users'",
    "psql -h 127.0.0.1 -p 5433 -d app -c '\\c prod' -c 'drop table users'",
    "psql -h 127.0.0.1 -p 5433 -d app -c '\\i other.sql' -c 'drop table users'",
    "psql -h 127.0.0.1 -p 5433 -d app -c \"select dblink_exec('host=prod', 'x'); "
    "drop table users\"",
    "pgcli -h 127.0.0.1 -p 5433 -d app -c 'drop table users'",
])
def test_uncertain_psql_targets_keep_the_prompt(command: str, project: Path,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(dbs.ENV_VAR, "postgresql://127.0.0.1:5433/app")
    _fake_dumps(monkeypatch)
    plan = dbs.plan_dumps(classify(command, cwd=str(project), root=project), project, command)
    assert not plan.covers_all, command
    assert _decide(project, command).action == "ask"


def test_environment_redirects_are_honored(project: Path,
                                           monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(dbs.ENV_VAR, "postgresql://127.0.0.1:5433/app")
    _fake_dumps(monkeypatch)
    command = "psql -h 127.0.0.1 -p 5433 -d app -c 'drop table users'"
    assert _decide(project, command).action == "continue"
    monkeypatch.setenv("PGHOSTADDR", "10.0.0.9")
    assert _decide(project, command + " ").action == "ask"
    monkeypatch.delenv("PGHOSTADDR")
    monkeypatch.setenv("PGSERVICE", "prod")
    assert _decide(project, command + "  ").action == "ask"


@pytest.mark.parametrize("command,covered", [
    ("psql -h 127.0.0.1 -p 5433 -d app -c 'DROP TABLE users'", True),
    ("psql 'host=localhost port=5433 dbname=app' -c 'DROP TABLE users'", True),
    ("PGHOST=127.0.0.1 PGPORT=5433 psql -d app -c 'TRUNCATE users'", True),
    # The conninfo host wins, but a non-local host named anywhere means: ask anyway.
    ("psql -h prod.example.com 'host=127.0.0.1 port=5433 dbname=app' -c 'drop table users'",
     False),
])
def test_fully_stated_local_targets_still_downgrade(command: str, covered: bool, project: Path,
                                                    monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(dbs.ENV_VAR, "postgresql://127.0.0.1:5433/app")
    plan = dbs.plan_dumps(classify(command, cwd=str(project), root=project), project, command)
    assert plan.covers_all is covered, command


def test_socket_directories_must_match() -> None:
    tcp = [dbs.parse_url("postgresql://localhost:5432/app")]
    other_socket = {"prog": "psql", "env": {},
                    "args": ["-h", "/some/other/socket", "-p", "5432", "-d", "app", "-c", "x"]}
    assert dbs.match_target(other_socket, tcp) is None
    sock = [dbs.parse_url("postgresql:///app?host=/run/a&port=5432")]
    assert dbs.match_target({**other_socket, "args": ["-h", "/run/b", "-d", "app"]}, sock) is None
    assert dbs.match_target({**other_socket, "args": ["-h", "/run/a/", "-d", "app"]}, sock)


def test_redact_uses_the_last_at_sign() -> None:
    assert dbs.redact("postgresql://u:p@ss@evil.com/app") == "postgresql://u:***@evil.com/app"
    assert dbs.redact("postgresql://u@localhost/app?password=x&a=1") == \
        "postgresql://u@localhost/app?password=***&a=1"


@needs_pg
def test_conninfo_host_beats_dash_h_like_real_psql(pg: Cluster) -> None:
    # Real psql connects to the conninfo host even though -h names a socket that doesn't exist.
    args = ["-h", "/nonexistent/socket", "-p", str(pg.port), f"host={pg.sock} dbname=app",
            "-c", "select 1"]
    out = subprocess.run([str(pg.bin / "psql"), "-X", "-tA", *args], capture_output=True,
                         text=True, timeout=30, encoding="utf-8", errors="replace")
    assert out.returncode == 0 and out.stdout.strip() == "1"
    db, certain = dbs.resolve_target({"prog": "psql", "args": args, "env": {}},
                                     [dbs.parse_url(pg.url)])
    assert db is not None and db.host == pg.sock and certain


@needs_pg
def test_undoing_a_later_edit_leaves_the_database_alone(
        pg: Cluster, project: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bag = Airbag(project, policy=Policy())
    drop = bag.before(ToolCall(agent="claude", tool_name="Bash", kind="shell", cwd=str(project),
                               command=pg.drop_command()))
    assert drop.action == "continue" and drop.databases == ["app"]
    _shell(pg.drop_command(), project)
    pg.psql("CREATE TABLE t2 (x int); INSERT INTO t2 VALUES (42)")
    edit = bag.before(ToolCall(agent="claude", tool_name="Edit", kind="edit", cwd=str(project),
                               paths=[str(project / "app.py")]))
    assert edit.snapshot is not None and drop.snapshot is not None
    assert edit.snapshot.seq != drop.snapshot.seq  # a snapshot with dumps is never shared
    assert edit.databases == []
    (project / "app.py").write_text("print('v2')\n", encoding="utf-8")

    cli.main(["rewind", "--project-dir", str(project)])
    timeline = capsys.readouterr().out.splitlines()
    assert sum("[db: app]" in line for line in timeline) == 1

    result = bag.undo(1)
    assert result is not None
    action, restore = result
    assert action["tool"] == "Edit" and restore.databases == []
    assert (project / "app.py").read_text(encoding="utf-8") == "print('v1')\n"
    assert _tables(pg) == "t2"  # the database was not touched

    result = bag.undo(1)  # now the DROP itself
    assert result is not None and result[1].databases == ["app"]
    assert _tables(pg) == "users"


@needs_pg
def test_exact_restore_with_identity_and_serial_columns(pg: Cluster) -> None:
    db = dbs.parse_url(pg.url)
    pg.psql("CREATE TABLE ident (id int GENERATED ALWAYS AS IDENTITY PRIMARY KEY, v text);"
            "CREATE TABLE ser (id serial PRIMARY KEY, v text);"
            "CREATE SEQUENCE free_seq;"
            "INSERT INTO ident (v) VALUES ('a'), ('b'); INSERT INTO ser (v) VALUES ('c')")
    with tempfile.TemporaryDirectory() as tmp:
        dump = Path(tmp) / "app.dump"
        dbs.dump(db, dump)
        pg.psql("DROP TABLE ser; CREATE SEQUENCE later_seq;"
                "CREATE TABLE later (id int GENERATED BY DEFAULT AS IDENTITY, x serial)")
        assert dbs.restore(db, dump) is None  # exact, no pg_restore --clean fallback
    rels = pg.psql("SELECT string_agg(relname, ',' ORDER BY relname) FROM pg_class "
                   "WHERE relkind IN ('r', 'S') AND relnamespace = 'public'::regnamespace")
    assert rels == "free_seq,ident,ident_id_seq,ser,ser_id_seq,users"
    assert pg.psql("SELECT string_agg(v, ',' ORDER BY id) FROM ident") == "a,b"
    pg.psql("INSERT INTO ident (v) VALUES ('c')")  # the identity sequence still works
    assert pg.psql("SELECT max(id) FROM ident") == "3"
