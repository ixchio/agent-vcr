"""``vcr mcp``: the stdio MCP server, driven over real pipes and by the official SDK client."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from agent_vcr import __version__
from agent_vcr.airbag.core import Airbag, ToolCall
from agent_vcr.cli import COMMANDS
from agent_vcr.mcp_server import (
    LATEST_PROTOCOL_VERSION,
    TOOLS,
    MCPServer,
)
from agent_vcr.models import FrameMetadata
from agent_vcr.recorder import VCRRecorder

SRC = str(Path(__file__).resolve().parents[2] / "src")
TOOL_NAMES = {
    "airbag_status", "airbag_timeline", "airbag_check", "airbag_snapshot", "airbag_undo",
    "airbag_rewind", "recordings_list", "recording_summary",
}


@pytest.fixture
def project(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("AGENT_VCR_AIRBAG_HOME", str(tmp_path / "airbag-home"))
    for var in ("AGENT_VCR_AIRBAG", "AGENT_VCR_AIRBAG_MODE", "AGENT_VCR_AIRBAG_ON_IRREVERSIBLE"):
        monkeypatch.delenv(var, raising=False)
    proj = tmp_path / "proj"
    proj.mkdir()
    subprocess.run(["git", "init", "-q", str(proj)], check=True, timeout=30)
    (proj / "a.txt").write_text("precious work\n")
    (proj / "b.txt").write_text("b\n")
    return proj


def _env() -> dict[str, str]:
    return {**os.environ, "PYTHONPATH": SRC + os.pathsep + os.environ.get("PYTHONPATH", "")}


class Pipe:
    """A real ``vcr mcp`` subprocess, spoken to line by line."""

    def __init__(self, project: Path, argv: list[str] | None = None) -> None:
        cmd = argv or [sys.executable, "-m", "agent_vcr.cli", "mcp", "--project-dir", str(project)]
        self.proc = subprocess.Popen(
            cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            cwd=str(project), env=_env(),
        )
        self.next_id = 0

    def send(self, message: dict[str, Any] | str) -> None:
        raw = message if isinstance(message, str) else json.dumps(message)
        assert self.proc.stdin is not None
        self.proc.stdin.write(raw.encode() + b"\n")
        self.proc.stdin.flush()

    def send_raw(self, data: bytes) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(data)
        self.proc.stdin.flush()

    def recv(self) -> dict[str, Any]:
        assert self.proc.stdout is not None
        line = self.proc.stdout.readline()
        assert line.endswith(b"\n"), f"server closed stdout; stderr: {self.stderr()}"
        return json.loads(line)

    def request(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        self.next_id += 1
        msg: dict[str, Any] = {"jsonrpc": "2.0", "id": self.next_id, "method": method}
        if params is not None:
            msg["params"] = params
        self.send(msg)
        reply = self.recv()
        assert reply["id"] == self.next_id
        return reply

    def call(self, name: str, **arguments: Any) -> dict[str, Any]:
        reply = self.request("tools/call", {"name": name, "arguments": arguments})
        assert "result" in reply, reply
        return reply["result"]

    def initialize(self, version: str = LATEST_PROTOCOL_VERSION) -> dict[str, Any]:
        reply = self.request("initialize", {
            "protocolVersion": version, "capabilities": {},
            "clientInfo": {"name": "pytest", "version": "0"},
        })
        self.send({"jsonrpc": "2.0", "method": "notifications/initialized"})
        return reply

    def close(self) -> int:
        assert self.proc.stdin is not None
        self.proc.stdin.close()
        code = self.proc.wait(timeout=30)
        assert self.proc.stdout is not None
        rest = self.proc.stdout.read()
        assert rest == b"", f"unexpected stdout after EOF: {rest!r}"
        return code

    def stderr(self) -> str:
        assert self.proc.stderr is not None
        return self.proc.stderr.read().decode(errors="replace") if self.proc.poll() is not None \
            else ""


@pytest.fixture
def pipe(project: Path) -> Any:
    server = Pipe(project)
    yield server
    if server.proc.poll() is None:
        server.proc.kill()
        server.proc.wait(timeout=30)


def _payload(result: dict[str, Any]) -> dict[str, Any]:
    assert result["isError"] is False, result["content"][0]["text"]
    text = json.loads(result["content"][0]["text"])
    assert text == result["structuredContent"]
    return text


def _agent_rm(project: Path, rel: str) -> None:
    """What a hooked agent does: the airbag snapshots, then the command deletes the file."""
    decision = Airbag(project).before(ToolCall(
        agent="claude", tool_name="Bash", kind="shell", cwd=str(project), command=f"rm {rel}"))
    assert decision.snapshot is not None
    (project / rel).unlink()


# ── protocol ─────────────────────────────────────────────────────────────

def test_initialize_negotiates_version_and_declares_tools(pipe: Pipe) -> None:
    reply = pipe.initialize("2025-06-18")
    result = reply["result"]
    assert reply["jsonrpc"] == "2.0"
    assert result["protocolVersion"] == "2025-06-18"  # supported: echoed
    assert result["capabilities"] == {"tools": {}}
    assert result["serverInfo"]["name"] == "agent-vcr"
    assert result["serverInfo"]["version"] == __version__
    again = pipe.request("initialize", {"protocolVersion": "1999-01-01", "capabilities": {},
                                        "clientInfo": {"name": "old", "version": "0"}})
    assert again["result"]["protocolVersion"] == LATEST_PROTOCOL_VERSION
    assert pipe.close() == 0


def test_notifications_get_no_reply_and_errors_use_jsonrpc_codes(pipe: Pipe) -> None:
    pipe.initialize()
    pipe.send({"jsonrpc": "2.0", "method": "notifications/cancelled", "params": {}})
    assert pipe.request("ping") == {"jsonrpc": "2.0", "id": 2, "result": {}}
    assert pipe.request("resources/list")["error"]["code"] == -32601
    assert pipe.request("no/such/method")["error"]["code"] == -32601
    unknown = pipe.request("tools/call", {"name": "rm_everything", "arguments": {}})
    assert unknown["error"]["code"] == -32602
    pipe.send("this is not json")
    parse = pipe.recv()
    assert parse["id"] is None and parse["error"]["code"] == -32700
    pipe.send(json.dumps([{"jsonrpc": "2.0", "id": 99, "method": "ping"}]))
    assert pipe.recv()["error"]["code"] == -32600
    pipe.send({"id": 100, "method": "ping"})  # missing jsonrpc
    assert pipe.recv() == {"jsonrpc": "2.0", "id": 100,
                           "error": {"code": -32600, "message": "invalid JSON-RPC 2.0 request"}}
    assert pipe.request("ping")["result"] == {}  # still serving
    assert pipe.close() == 0


def test_parse_error_reply(pipe: Pipe) -> None:
    pipe.send("{oops")
    reply = pipe.recv()
    assert reply["id"] is None and reply["error"]["code"] == -32700
    assert pipe.close() == 0


def test_tools_list_schemas(pipe: Pipe) -> None:
    pipe.initialize()
    tools = pipe.request("tools/list")["result"]["tools"]
    assert {t["name"] for t in tools} == TOOL_NAMES
    by_name = {t["name"]: t for t in tools}
    for tool in tools:
        assert tool["inputSchema"]["type"] == "object"
        assert tool["description"]
    assert by_name["airbag_check"]["inputSchema"]["required"] == ["command"]
    assert by_name["airbag_check"]["annotations"]["readOnlyHint"] is True
    undo = by_name["airbag_undo"]
    assert undo["annotations"]["destructiveHint"] is True
    assert undo["inputSchema"]["properties"]["dry_run"]["default"] is True
    assert "confirm=true" in undo["description"] and "user" in undo["description"]
    assert by_name["airbag_rewind"]["inputSchema"]["required"] == ["snapshot"]
    assert pipe.close() == 0


# ── airbag tools ─────────────────────────────────────────────────────────

def test_status_check_and_snapshot(pipe: Pipe, project: Path) -> None:
    pipe.initialize()
    status = _payload(pipe.call("airbag_status"))
    assert status["project"] == str(project.resolve())
    assert status["snapshots"] == 0 and status["protected"] is False
    assert "vcr airbag install" in status["hint"]

    check = _payload(pipe.call("airbag_check", command="git push --force origin main"))
    assert check["level"] == "irreversible" and check["airbag_would_stop"] is True
    assert check["reasons"]
    safe = _payload(pipe.call("airbag_check", command="ls -la"))
    assert safe["level"] == "read" and safe["undoable"] is False

    snap = _payload(pipe.call("airbag_snapshot", message="before refactor"))
    assert snap["snapshot"] == 1 and snap["new"] is True and snap["label"] == "before refactor"
    same = _payload(pipe.call("airbag_snapshot", message="again"))
    assert same["snapshot"] == 1 and same["new"] is False  # nothing changed: idempotent
    assert _payload(pipe.call("airbag_status"))["snapshots"] == 1
    assert (project / "a.txt").read_text() == "precious work\n"
    assert pipe.close() == 0


def test_undo_previews_by_default_and_requires_confirm(pipe: Pipe, project: Path) -> None:
    _agent_rm(project, "a.txt")
    pipe.initialize()
    timeline = _payload(pipe.call("airbag_timeline", limit=5))
    assert timeline["total"] == 1
    assert timeline["actions"][0]["command"] == "rm a.txt"
    assert timeline["actions"][0]["agent"] == "claude"

    preview = _payload(pipe.call("airbag_undo"))
    assert preview["dry_run"] is True
    assert {"change": "restore", "path": "a.txt"} in preview["changes"]
    assert not (project / "a.txt").exists()

    for args in ({"dry_run": False}, {"dry_run": False, "confirm": False}):
        refused = pipe.call("airbag_undo", **args)
        assert refused["isError"] is True
        assert "confirm=true" in refused["content"][0]["text"]
        assert not (project / "a.txt").exists()

    bad = pipe.call("airbag_undo", dry_run="false", confirm="true")  # strings are not booleans
    assert bad["isError"] is True and not (project / "a.txt").exists()
    confirmed_only = pipe.call("airbag_undo", confirm=True)  # dry_run still defaults to true
    assert _payload(confirmed_only)["dry_run"] is True and not (project / "a.txt").exists()

    done = _payload(pipe.call("airbag_undo", dry_run=False, confirm=True))
    assert done["dry_run"] is False and "a.txt" in done["written"]
    assert (project / "a.txt").read_text() == "precious work\n"
    assert "vcr redo" in done["redo"]
    assert pipe.close() == 0


def test_rewind_previews_and_requires_confirm(pipe: Pipe, project: Path) -> None:
    pipe.initialize()
    first = _payload(pipe.call("airbag_snapshot", message="start"))["snapshot"]
    (project / "b.txt").write_text("changed by the agent\n")
    (project / "new.txt").write_text("new\n")
    preview = _payload(pipe.call("airbag_rewind", snapshot=first))
    assert preview["dry_run"] is True
    changes = {(c["change"], c["path"]) for c in preview["changes"]}
    assert ("rewrite", "b.txt") in changes and ("remove", "new.txt") in changes
    assert (project / "b.txt").read_text() == "changed by the agent\n"

    refused = pipe.call("airbag_rewind", snapshot=first, dry_run=False)
    assert refused["isError"] is True and (project / "new.txt").exists()

    missing = pipe.call("airbag_rewind", snapshot=999)
    assert missing["isError"] is True and "#999" in missing["content"][0]["text"]

    done = _payload(pipe.call("airbag_rewind", snapshot=first, dry_run=False, confirm=True))
    assert (project / "b.txt").read_text() == "b\n" and not (project / "new.txt").exists()
    assert "b.txt" in done["written"] and "new.txt" in done["deleted"]
    assert pipe.close() == 0


def test_bad_arguments_are_tool_errors_not_crashes(pipe: Pipe) -> None:
    pipe.initialize()
    for name, args in (
        ("airbag_undo", {"steps": "two"}),
        ("airbag_undo", {"steps": 0}),
        ("airbag_timeline", {"limit": 10_000}),
        ("airbag_check", {}),
        ("airbag_check", {"command": "ls", "typo": 1}),
        ("airbag_rewind", {"snapshot": True}),
        ("recording_summary", {"path": "nope.vcr"}),
        ("recordings_list", {"directory": "does/not/exist"}),
    ):
        result = pipe.call(name, **args)
        assert result["isError"] is True, (name, args, result)
        assert result["content"][0]["type"] == "text"
    assert pipe.request("ping")["result"] == {}
    assert pipe.close() == 0


def test_unsafe_root_is_a_tool_error(pipe: Pipe) -> None:
    pipe.initialize()
    result = pipe.call("airbag_status", project_dir=str(Path.home()))
    assert result["isError"] is True and "refusing" in result["content"][0]["text"]
    assert pipe.close() == 0


# ── recordings ───────────────────────────────────────────────────────────

def _record(directory: Path) -> Path:
    recorder = VCRRecorder(output_dir=str(directory), auto_save=False)
    recorder.start_session(session_id="mcp-demo")
    recorder.record_step(node_name="plan", input_state={"q": 1}, output_state={"q": 2},
                         metadata=FrameMetadata(latency_ms=10))
    recorder.record_llm_call(model="gpt-test", messages=[{"role": "user", "content": "hi"}],
                             response={"content": "hello"}, tokens_input=12, tokens_output=8,
                             latency_ms=20, cost_usd=0.25)
    recorder.record_error(node_name="tool", input_state={}, error=RuntimeError("boom"),
                          latency_ms=5)
    recorder.record_checkpoint({"q": 2}, label="done")
    return recorder.save()


def test_recordings_list_and_summary(pipe: Pipe, project: Path) -> None:
    path = _record(project / ".vcr")
    junk = project / "node_modules" / "x"
    junk.mkdir(parents=True)
    (junk / "ignored.vcr").write_text("{}\n")
    pipe.initialize()
    listing = _payload(pipe.call("recordings_list"))
    assert listing["count"] == 1
    entry = listing["recordings"][0]
    assert entry["path"] == str(path) and entry["session_id"] == "mcp-demo"
    assert entry["relative"] == os.path.join(".vcr", path.name)

    summary = _payload(pipe.call("recording_summary", path=entry["relative"]))
    assert summary["session_id"] == "mcp-demo"
    assert summary["frames"] == 4
    assert summary["tokens"] == 20 and summary["tokens_input"] == 12
    assert summary["cost_usd"] == pytest.approx(0.25)
    assert summary["error_count"] == 1 and "boom" in summary["errors"][0]["message"]
    assert summary["checkpoints"] == ["done"]
    assert summary["models"] == ["gpt-test"]
    assert {n["name"] for n in summary["nodes"]} >= {"plan", "tool"}
    assert summary["integrity"]["ok"] is True
    assert pipe.close() == 0


def test_recording_summary_of_garbage_file_is_tool_error(pipe: Pipe, project: Path) -> None:
    (project / "bad.vcr").write_text("not json at all\n")
    pipe.initialize()
    result = pipe.call("recording_summary", path="bad.vcr")
    assert result["isError"] is True and "session header" in result["content"][0]["text"]
    assert pipe.close() == 0


# ── stdout hygiene, CLI ──────────────────────────────────────────────────

NOISY = r"""
import os, subprocess, sys
import agent_vcr.mcp_server as m

def noisy(server, args):
    print("stray print from a tool")
    subprocess.run(["git", "--version"], timeout=30)  # child inherits fd 1
    leaked = sys.stdin.read() if not sys.stdin.closed else ""
    return {"leaked_stdin": leaked}

m.TOOLS = m.TOOLS + (m.Tool(name="noisy", title="noisy", description="noisy",
                            properties={}, handler=noisy),)
raise SystemExit(m.run(sys.argv[1]))
"""


def test_stdout_carries_only_protocol_json(project: Path) -> None:
    pipe = Pipe(project, [sys.executable, "-c", NOISY, str(project)])
    try:
        pipe.initialize()
        result = _payload(pipe.call("noisy"))
        assert result == {"leaked_stdin": ""}  # fd 0 is /dev/null for tools and children
        assert pipe.request("ping")["result"] == {}
        assert pipe.close() == 0
        err = pipe.stderr()
        assert "stray print from a tool" in err and "git version" in err
    finally:
        if pipe.proc.poll() is None:
            pipe.proc.kill()


def test_cli_registers_mcp_and_keeps_it_lazy(tmp_path: Path) -> None:
    assert "mcp" in COMMANDS
    code = (
        "import sys, agent_vcr.cli as c; "
        "assert 'agent_vcr.mcp_server' not in sys.modules; "
        "import agent_vcr.mcp_server; "
        "assert 'pydantic' not in sys.modules, 'mcp server import pulled in pydantic'"
    )
    proc = subprocess.run([sys.executable, "-c", code], env=_env(), capture_output=True,
                          text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    missing = subprocess.run(
        [sys.executable, "-m", "agent_vcr.cli", "mcp", "--project-dir", str(tmp_path / "nope")],
        env=_env(), capture_output=True, text=True, timeout=60, stdin=subprocess.DEVNULL,
    )
    assert missing.returncode == 2 and missing.stdout == ""
    assert "does not exist" in missing.stderr


def test_in_process_handle_without_transport(project: Path) -> None:
    server = MCPServer(project)
    assert server.handle({"jsonrpc": "2.0", "method": "notifications/initialized"}) is None
    assert server.handle({"jsonrpc": "2.0", "id": 1, "result": {}}) is None  # stray response
    listed = server.handle({"jsonrpc": "2.0", "id": "a", "method": "tools/list"})
    assert listed is not None and len(listed["result"]["tools"]) == len(TOOLS)
    bad_params = server.handle({"jsonrpc": "2.0", "id": 2, "method": "tools/list",
                                "params": [1]})
    assert bad_params is not None and bad_params["error"]["code"] == -32602
    bool_id = server.handle({"jsonrpc": "2.0", "id": True, "method": "ping"})
    assert bool_id is not None and bool_id["error"]["code"] == -32600


# ── the official MCP Python SDK as the client ────────────────────────────

def test_official_sdk_client_round_trip(project: Path) -> None:
    pytest.importorskip("mcp")
    from mcp import ClientSession
    from mcp.client.stdio import StdioServerParameters, stdio_client

    _agent_rm(project, "a.txt")
    params = StdioServerParameters(
        command=sys.executable,
        args=["-m", "agent_vcr.cli", "mcp", "--project-dir", str(project)],
        env=_env(), cwd=str(project),
    )

    async def scenario() -> None:
        async with stdio_client(params) as (read, write), ClientSession(read, write) as session:
            init = await session.initialize()
            assert init.serverInfo.name == "agent-vcr"
            assert init.serverInfo.version == __version__
            assert init.capabilities.tools is not None
            tools = await session.list_tools()
            assert {t.name for t in tools.tools} == TOOL_NAMES
            check = await session.call_tool("airbag_check", {"command": "rm -rf ."})
            assert check.isError is False
            assert check.structuredContent is not None
            assert check.structuredContent["level"] in {"destructive", "irreversible"}
            preview = await session.call_tool("airbag_undo", {})
            assert preview.isError is False and not (project / "a.txt").exists()
            refused = await session.call_tool("airbag_undo", {"dry_run": False})
            assert refused.isError is True and not (project / "a.txt").exists()
            done = await session.call_tool("airbag_undo", {"dry_run": False, "confirm": True})
            assert done.isError is False
            assert (project / "a.txt").read_text() == "precious work\n"
            unicode = await session.call_tool("airbag_check", {"command": "echo héllo 🎉"})
            assert unicode.structuredContent is not None
            assert unicode.structuredContent["command"] == "echo héllo 🎉"
            await session.send_ping()

    asyncio.run(asyncio.wait_for(scenario(), timeout=120))


# ── review regressions: odd strings, scope, limits ───────────────────────

def _ping_still_works(pipe: Pipe) -> None:
    assert pipe.request("ping")["result"] == {}


def _make_non_utf8_file(directory: Path, name: bytes) -> Path:
    if sys.platform != "linux":
        pytest.skip("non-UTF-8 file names need a byte-oriented file system")
    raw = os.path.join(os.fsencode(str(directory)), name)
    try:
        with open(raw, "wb") as f:
            f.write(b"x\n")
    except OSError:
        pytest.skip("file system rejects non-UTF-8 names")
    return Path(os.fsdecode(raw))


def test_unencodable_tool_name_and_id_do_not_kill_the_server(pipe: Pipe) -> None:
    pipe.initialize()
    pipe.send_raw(b'{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"\\ud800"}}\n')
    reply = pipe.recv()
    assert reply["id"] == 7 and reply["error"]["code"] == -32602
    pipe.send_raw(b'{"jsonrpc":"2.0","id":"\\ud800","method":"ping"}\n')
    reply = pipe.recv()
    assert reply["id"] == "\ud800" and reply["result"] == {}
    pipe.send_raw(b'{"jsonrpc":"2.0","id":8,"method":"tools/call","params":{"name":"airbag_check",'
                  b'"arguments":{"command":"echo \\udcff \\ud800"}}}\n')
    result = pipe.recv()["result"]
    assert result["isError"] is False
    _ping_still_works(pipe)
    assert pipe.close() == 0


def test_non_utf8_recording_name_does_not_kill_the_server(pipe: Pipe, project: Path) -> None:
    _make_non_utf8_file(project, b"bad\xff.vcr")
    pipe.initialize()
    listing = _payload(pipe.call("recordings_list"))
    assert listing["count"] == 1
    assert listing["recordings"][0]["relative"] == "bad\\xff.vcr"  # bytes shown as text
    _ping_still_works(pipe)
    assert pipe.close() == 0


def test_non_utf8_file_in_rewind_preview_does_not_kill_the_server(
    pipe: Pipe, project: Path,
) -> None:
    pipe.initialize()
    first = _payload(pipe.call("airbag_snapshot", message="clean"))["snapshot"]
    bad = _make_non_utf8_file(project, b"caf\xe9.txt")
    preview = _payload(pipe.call("airbag_rewind", snapshot=first))
    assert {"change": "remove", "path": "caf\\xe9.txt"} in preview["changes"]
    assert bad.exists()
    _ping_still_works(pipe)
    assert pipe.close() == 0


@pytest.fixture
def other_project(tmp_path: Path, project: Path) -> Path:
    other = tmp_path / "other"
    other.mkdir()
    subprocess.run(["git", "init", "-q", str(other)], check=True, timeout=30)
    (other / "secret.txt").write_text("other project\n")
    _agent_rm(other, "secret.txt")
    return other


def test_restoring_tools_stay_inside_the_server_project(
    pipe: Pipe, project: Path, other_project: Path,
) -> None:
    pipe.initialize()
    for name, args in (
        ("airbag_undo", {"dry_run": False, "confirm": True}),
        ("airbag_undo", {}),
        ("airbag_rewind", {"snapshot": 1, "dry_run": False, "confirm": True}),
        ("airbag_snapshot", {"message": "x"}),
    ):
        result = pipe.call(name, project_dir=str(other_project), **args)
        assert result["isError"] is True, (name, result)
        assert "outside this server's project" in result["content"][0]["text"]
    assert not (other_project / "secret.txt").exists()
    # read-only tools may still look elsewhere
    status = _payload(pipe.call("airbag_status", project_dir=str(other_project)))
    assert status["snapshots"] >= 1
    timeline = _payload(pipe.call("airbag_timeline", project_dir=str(other_project)))
    assert timeline["total"] == 1
    # subdirectories of the server's project are fine
    (project / "sub").mkdir()
    sub = _payload(pipe.call("airbag_snapshot", project_dir="sub", message="from sub"))
    assert sub["project"] == str(project.resolve())
    assert pipe.close() == 0


def test_symlink_out_of_the_project_is_refused(
    pipe: Pipe, project: Path, other_project: Path,
) -> None:
    try:
        (project / "link").symlink_to(other_project, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks not available")
    pipe.initialize()
    result = pipe.call("airbag_undo", project_dir="link", dry_run=False, confirm=True)
    assert result["isError"] is True and not (other_project / "secret.txt").exists()
    assert pipe.close() == 0


def test_allow_any_project_flag_lifts_the_scope(project: Path, other_project: Path) -> None:
    pipe = Pipe(project, [sys.executable, "-m", "agent_vcr.cli", "mcp", "--project-dir",
                          str(project), "--allow-any-project"])
    try:
        pipe.initialize()
        done = _payload(pipe.call("airbag_undo", project_dir=str(other_project),
                                  dry_run=False, confirm=True))
        assert "secret.txt" in done["written"]
        assert (other_project / "secret.txt").read_text() == "other project\n"
        assert pipe.close() == 0
    finally:
        if pipe.proc.poll() is None:
            pipe.proc.kill()


def test_recording_summary_requires_a_recording_suffix(pipe: Pipe, project: Path) -> None:
    (project / "notes.txt").write_text("not a recording\n")
    pipe.initialize()
    for path in ("notes.txt", "/etc/passwd", str(project / "a.txt")):
        result = pipe.call("recording_summary", path=path)
        assert result["isError"] is True
        assert "not a recording" in result["content"][0]["text"]
    recorded = _record(project / ".vcr")
    jsonl = recorded.with_suffix(".jsonl")
    jsonl.write_bytes(recorded.read_bytes())
    assert _payload(pipe.call("recording_summary", path=str(jsonl)))["frames"] == 4
    assert pipe.close() == 0


def test_null_arguments_mean_not_given(pipe: Pipe, project: Path) -> None:
    pipe.initialize()
    first = _payload(pipe.call("airbag_snapshot", message=None))
    assert first["label"] == "manual snapshot (MCP)"
    assert _payload(pipe.call("airbag_timeline", limit=None))["total"] == 1
    (project / "b.txt").write_text("changed\n")
    preview = _payload(pipe.call("airbag_rewind", snapshot=first["snapshot"], dry_run=None,
                                 confirm=None))
    assert preview["dry_run"] is True and (project / "b.txt").read_text() == "changed\n"
    assert pipe.close() == 0


def test_bom_on_first_line_and_oversized_lines(pipe: Pipe) -> None:
    pipe.send_raw(b'\xef\xbb\xbf{"jsonrpc":"2.0","id":1,"method":"ping"}\r\n')
    assert pipe.recv() == {"jsonrpc": "2.0", "id": 1, "result": {}}
    huge = b'{"jsonrpc":"2.0","id":2,"method":"ping","params":{"x":"' + b"a" * (5 << 20) + b'"}}\n'
    pipe.send_raw(huge)
    reply = pipe.recv()
    assert reply["id"] is None and reply["error"]["code"] == -32600
    assert "larger than 4 MB" in reply["error"]["message"]
    pipe.next_id = 2
    _ping_still_works(pipe)
    assert pipe.close() == 0


def test_check_cwd_resolves_against_project_dir(
    pipe: Pipe, project: Path, other_project: Path,
) -> None:
    pipe.initialize()
    check = _payload(pipe.call("airbag_check", command="ls", project_dir=str(other_project),
                               cwd="src"))
    assert check["cwd"] == str(other_project / "src")
    default = _payload(pipe.call("airbag_check", command="ls"))
    assert default["cwd"] == str(project)
    assert pipe.close() == 0


def test_recording_summary_caps_long_lists(project: Path) -> None:
    recorder = VCRRecorder(output_dir=str(project / ".vcr"), auto_save=False)
    recorder.start_session(session_id="many-nodes")
    for i in range(250):
        recorder.record_step(node_name=f"node-{i}", input_state={}, output_state={"i": i},
                             metadata=FrameMetadata(model=f"model-{i}"))
    path = recorder.save()
    server = MCPServer(project)
    reply = server.handle({"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {
        "name": "recording_summary", "arguments": {"path": str(path)}}})
    assert reply is not None
    summary = reply["result"]["structuredContent"]
    assert summary["node_count"] == 250 and len(summary["nodes"]) == 200
    assert len(summary["models"]) == 200 and summary["lists_truncated"] is True
