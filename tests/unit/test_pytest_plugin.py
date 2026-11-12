"""The pytest plugin (agent_vcr.pytest_plugin), exercised through real inner pytest runs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from agent_vcr.pytest_plugin import RESULT_NODE, node_slug

pytest_plugins = ["pytester"]

REPO = Path(__file__).resolve().parents[2]

AGENT_TESTS = '''
import os

import pytest


def triage(rec):
    with open("calls.txt", "a") as f:
        f.write("call\\n")
    careless = os.environ.get("CARELESS") == "1"
    rec.record_llm_call(
        "demo-model", [{"role": "user", "content": "charged twice"}], {"content": "billing"},
        tokens_input=500, tokens_output=900 if careless else 300, latency_ms=400.0,
        cost_usd=0.008,
    )
    rec.record_tool_call("lookup_payments", {"order": "A-1"}, {"duplicate": True}, latency_ms=40)
    if careless:
        rec.record_tool_call("delete_customer", {"order": "A-1"}, {"ok": True}, latency_ms=10)
    return {"queue": "billing", "status": "deleted" if careless else "refunded"}


@pytest.mark.agent_vcr(task="triage ticket t_123", identity={"model": "demo-model"})
def test_triage(agent_vcr):
    run = agent_vcr.run(triage)
    assert run.output == {"queue": "billing", "status": run.output["status"]}
    with open("modes.txt", "a") as f:
        f.write(("replayed" if run.replayed else "recorded") + "\\n")
    agent_vcr.assert_matches_golden()
'''


@pytest.fixture
def inner(pytester: pytest.Pytester, monkeypatch: pytest.MonkeyPatch):
    """Run an isolated inner pytest with only this plugin loaded."""
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    monkeypatch.delenv("CARELESS", raising=False)

    # This module already imported the plugin, so pytest can't assert-rewrite it again.
    quiet = "ignore::pytest.PytestAssertRewriteWarning"

    def run(*args: str) -> pytest.RunResult:
        return pytester.runpytest(
            "-p", "agent_vcr.pytest_plugin", "-p", "no:cacheprovider", "-W", quiet, *args
        )

    return run


def _calls(pytester: pytest.Pytester) -> int:
    path = pytester.path / "calls.txt"
    return len(path.read_text().splitlines()) if path.exists() else 0


def _golden_dir(pytester: pytest.Pytester, nodeid: str) -> Path:
    return pytester.path / ".vcr" / "pytest" / "golden" / node_slug(nodeid)


def _golden_files(pytester: pytest.Pytester, nodeid: str) -> list[Path]:
    return sorted(_golden_dir(pytester, nodeid).glob("*.vcr"))


# -- record / replay ------------------------------------------------------------


def test_records_once_then_replays_with_zero_agent_calls(pytester, inner) -> None:
    pytester.makepyfile(test_agent=AGENT_TESTS)

    first = inner()
    first.assert_outcomes(passed=1)
    assert _calls(pytester) == 1
    goldens = _golden_files(pytester, "test_agent.py::test_triage")
    assert len(goldens) == 1
    first.stdout.fnmatch_lines(["*agent-vcr: 1 run(s) recorded, 1 saved as golden*"])

    for _ in range(2):
        again = inner()
        again.assert_outcomes(passed=1)
        again.stdout.fnmatch_lines(["*1 run(s) replayed (agent not called, 800 tokens*"])
    assert _calls(pytester) == 1  # the agent was never called again
    assert (pytester.path / "modes.txt").read_text().split() == ["recorded", "replayed", "replayed"]


def test_replayed_output_matches_recorded_output(pytester, inner) -> None:
    pytester.makepyfile(
        test_out='''
import json
from pathlib import Path


def agent(rec):
    rec.record_step("plan", {"q": 1}, {"plan": ["a", "b"]})
    return ["done", 3]


def test_out(agent_vcr):
    run = agent_vcr.run("list result", agent)
    Path("out.jsonl").open("a").write(json.dumps([run.output, len(run.outputs)]) + "\\n")
'''
    )
    inner().assert_outcomes(passed=1)
    inner().assert_outcomes(passed=1)
    lines = (pytester.path / "out.jsonl").read_text().splitlines()
    assert lines[0] == lines[1] == json.dumps([["done", 3], 2])
    golden = _golden_files(pytester, "test_out.py::test_out")[0]
    frames = [json.loads(line) for line in golden.read_text().splitlines()[1:]]
    assert frames[-1]["data"]["node_name"] == RESULT_NODE
    assert frames[-1]["data"]["metadata"]["latency_ms"] == 0.0


def test_step_pipeline_records_usage_and_input_is_part_of_identity(pytester, inner) -> None:
    pytester.makepyfile(
        test_pipe='''
import os


def classify(state):
    open("calls.txt", "a").write("call\\n")
    return {**state, "queue": "billing"}, {"tokens_used": 120, "cost_usd": 0.001, "latency_ms": 5}


def draft(state):
    return {**state, "reply": "Refund on its way"}


def test_pipe(agent_vcr):
    text = os.environ.get("TICKET", "charged twice")
    run = agent_vcr.run("triage", [("classify", classify), ("draft", draft)],
                        input={"text": text})
    assert run.output["reply"] == "Refund on its way"
    assert run.output["text"] == text
    assert "input_sha256" in run.identity
'''
    )
    inner().assert_outcomes(passed=1)
    inner().assert_outcomes(passed=1)
    assert _calls(pytester) == 1
    golden = _golden_files(pytester, "test_pipe.py::test_pipe")[0]
    frames = [json.loads(line)["data"] for line in golden.read_text().splitlines()[1:]]
    assert [f["node_name"] for f in frames] == ["classify", "draft"]
    assert frames[0]["metadata"]["tokens_used"] == 120

    os.environ["TICKET"] = "card declined"
    try:
        missing = inner("--vcr-record=none")
        missing.assert_outcomes(failed=1)
        missing.stdout.fnmatch_lines(["*different identity (changed: input_sha256)*"])
        inner().assert_outcomes(passed=1)  # once: records the new input
    finally:
        del os.environ["TICKET"]
    assert _calls(pytester) == 2
    assert len(_golden_files(pytester, "test_pipe.py::test_pipe")) == 2


# -- regression detection -----------------------------------------------------------


def test_rerecord_detects_regression_and_keeps_the_golden(pytester, inner, monkeypatch) -> None:
    pytester.makepyfile(test_agent=AGENT_TESTS)
    inner().assert_outcomes(passed=1)
    golden = _golden_files(pytester, "test_agent.py::test_triage")[0]
    before = golden.read_bytes()

    monkeypatch.setenv("CARELESS", "1")
    result = inner("--vcr-record=all")
    result.assert_outcomes(failed=1)
    assert _calls(pytester) == 2
    result.stdout.fnmatch_lines(
        [
            "*AssertionError: agent_vcr: DEGRADED vs golden (fail_on=degraded)",
            "*test:      test_agent.py::test_triage",
            "*candidate: *candidate-0.vcr",
            "*- tokens +75% vs baseline",
            "*- new tool(s) vs baseline: delete_customer",
            "*- final state differs from baseline",
            "*tokens   800 -> 1,400  (+75%)",
            "*tools    +delete_customer",
            "*~ status: 'refunded' -> 'deleted'",
            "*pytest --vcr-record=all --vcr-fail-on=never 'test_agent.py::test_triage'",
            "*1 not saved because the test failed*",
        ]
    )
    assert golden.read_bytes() == before  # a failing run never becomes the golden

    # Only rule breaks fail under --vcr-fail-on=failed; drift alone passes.
    inner("--vcr-record=all", "--vcr-fail-on=failed").assert_outcomes(passed=1)

    monkeypatch.delenv("CARELESS")
    replay = inner("--vcr-record=none")
    replay.assert_outcomes(passed=1)
    assert _calls(pytester) == 3  # replay did not call the agent


def test_fail_on_never_accepts_the_new_golden_with_a_warning(pytester, inner, monkeypatch) -> None:
    pytester.makepyfile(test_agent=AGENT_TESTS)
    inner().assert_outcomes(passed=1)
    golden = _golden_files(pytester, "test_agent.py::test_triage")[0]

    monkeypatch.setenv("CARELESS", "1")
    accepted = inner("--vcr-record=all", "--vcr-fail-on=never")
    accepted.assert_outcomes(passed=1, warnings=1)
    accepted.stdout.fnmatch_lines(["*AgentVCRWarning: agent_vcr: DEGRADED vs golden*"])
    assert "delete_customer" in golden.read_text()


def test_expectations_fail_on_rule_breaks_even_when_replaying(pytester, inner) -> None:
    pytester.makepyfile(
        test_budget='''
def agent(rec):
    rec.record_llm_call("m", [], {"content": "x"}, tokens_input=900, tokens_output=100,
                        latency_ms=10.0, cost_usd=0.01)
    rec.record_tool_call("delete_customer", {}, {"ok": True}, latency_ms=1)
    return {"done": True}


def test_budget(agent_vcr):
    agent_vcr.run("budget task", agent)
    agent_vcr.assert_matches_golden(max_tokens=500, forbid_tools=["delete_customer"],
                                    expect_state={"done": True})
'''
    )
    first = inner()
    first.assert_outcomes(failed=1)
    first.stdout.fnmatch_lines(
        [
            "*agent_vcr: FAIL vs golden*",
            "*- tokens 1000 > budget 500",
            "*- forbidden tool(s) used: delete_customer",
        ]
    )
    assert not _golden_dir(pytester, "test_budget.py::test_budget").exists()


def test_latency_drift_below_noise_floor_is_ignored_unless_asked(pytester, inner) -> None:
    pytester.makepyfile(
        test_lat='''
import os

import pytest


def agent(rec):
    ms = float(os.environ.get("MS", "1"))
    rec.record_tool_call("search", {}, {"hits": 1}, latency_ms=ms)
    return {"ok": True}


@pytest.mark.parametrize("delta", [None, 1.0])
def test_lat(agent_vcr, delta):
    agent_vcr.run("latency task", agent)
    if delta is None:
        agent_vcr.assert_matches_golden()
    else:
        agent_vcr.assert_matches_golden(max_latency_delta=delta)
'''
    )
    inner().assert_outcomes(passed=2)
    os.environ["MS"] = "9"
    try:
        result = inner("--vcr-record=all")
    finally:
        del os.environ["MS"]
    result.assert_outcomes(passed=1, failed=1)
    result.stdout.fnmatch_lines(["*test_lat?1.0?*", "*- latency +800% vs baseline"])


def test_new_identity_is_compared_with_the_previous_golden(pytester, inner, monkeypatch) -> None:
    pytester.makepyfile(
        test_ident='''
import os


def agent(rec):
    model = os.environ.get("MODEL", "m1")
    rec.record_llm_call(model, [], {"content": "x"}, tokens_input=100,
                        tokens_output=900 if model == "m2" else 100, latency_ms=100.0)
    return {"ok": True}


def test_ident(agent_vcr):
    agent_vcr.run("ident task", agent, identity={"model": os.environ.get("MODEL", "m1")})
    agent_vcr.assert_matches_golden()
'''
    )
    inner().assert_outcomes(passed=1)
    monkeypatch.setenv("MODEL", "m2")
    result = inner()  # once: no golden for m2, so it records and compares with m1's golden
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(
        [
            "*baseline:  previous golden for this task (identity changed: model)",
            "*- tokens +400% vs baseline",
            "*- new model(s) vs baseline: m2",
        ]
    )
    assert len(_golden_files(pytester, "test_ident.py::test_ident")) == 1


def test_goldens_resolve_against_rootdir_from_any_cwd(pytester, inner, monkeypatch) -> None:
    pytester.makeini("[pytest]\naddopts = --vcr-dir goldens\n")
    pytester.makepyfile(test_agent=AGENT_TESTS)
    inner().assert_outcomes(passed=1)
    assert (pytester.path / "goldens" / "golden").is_dir()
    sub = pytester.mkdir("sub")
    monkeypatch.chdir(sub)
    inner("--vcr-record=none", str(pytester.path)).assert_outcomes(passed=1)
    assert _calls(pytester) == 1
    assert not (sub / "goldens").exists()


# -- recording forbidden ------------------------------------------------------------


def test_none_mode_without_golden_fails_and_never_calls_the_agent(pytester, inner) -> None:
    pytester.makepyfile(test_agent=AGENT_TESTS)
    result = inner("--vcr-record=none")
    result.assert_outcomes(failed=1)
    result.stdout.fnmatch_lines(
        [
            "*agent_vcr: no golden recording for this test, and recording is off "
            "(--vcr-record=none), so the agent was not called.",
            "*test:      test_agent.py::test_triage",
            "*task:      'triage ticket t_123'",
            '*identity:  {"model": "demo-model"}',
            "*looked in: .vcr*pytest*golden*test_agent.py--test_triage",
            "*pytest --vcr-record=once 'test_agent.py::test_triage'",
        ]
    )
    assert _calls(pytester) == 0
    assert not (pytester.path / ".vcr").exists()


def test_marker_record_mode_and_command_line_precedence(pytester, inner) -> None:
    pytester.makepyfile(
        test_mark='''
import pytest


def agent(rec):
    open("calls.txt", "a").write("call\\n")
    return {"ok": True}


@pytest.mark.agent_vcr("marked task", record="none")
def test_mark(agent_vcr):
    assert agent_vcr.mode in ("none", "once")
    agent_vcr.run(agent)
'''
    )
    inner().assert_outcomes(failed=1)  # the marker forbids recording
    assert _calls(pytester) == 0
    inner("--vcr-record=once").assert_outcomes(passed=1)  # command line wins
    inner().assert_outcomes(passed=1)  # marker none + golden: replay
    assert _calls(pytester) == 1


def test_bad_marker_arguments_error_at_setup(pytester, inner) -> None:
    pytester.makepyfile(
        test_bad='''
import pytest


@pytest.mark.agent_vcr(task="t", record="sometimes")
def test_bad_mode(agent_vcr):
    pass


@pytest.mark.agent_vcr(tsk="t")
def test_typo(agent_vcr):
    pass
'''
    )
    result = inner()
    result.assert_outcomes(errors=2)
    result.stdout.fnmatch_lines(["*(record=) must be one of*", "*unknown argument(s): tsk*"])


def test_run_argument_errors(pytester, inner) -> None:
    pytester.makepyfile(
        test_args='''
import pytest


def test_no_task(agent_vcr):
    with pytest.raises(TypeError, match="needs a task"):
        agent_vcr.run(lambda rec: 1)


def test_bad_pipeline(agent_vcr):
    with pytest.raises(TypeError, match="list of"):
        agent_vcr.run("t", [("a",)])
    with pytest.raises(TypeError, match="input= is for step pipelines"):
        agent_vcr.run("t", lambda rec: 1, input={})


def test_bad_expectation(agent_vcr):
    agent_vcr.run("t", lambda rec: {"x": 1})
    with pytest.raises(TypeError, match="unknown expectation"):
        agent_vcr.assert_matches_golden(max_tokenz=1)


def test_nothing_to_replay(agent_vcr):
    agent_vcr.run("t", lambda rec: None)
'''
    )
    result = inner()
    result.assert_outcomes(passed=3, failed=1)
    result.stdout.fnmatch_lines(["*recorded no steps and returned None*"])


# -- options ----------------------------------------------------------------------


def test_options_parse_and_apply(pytester, inner) -> None:
    pytester.makepyfile(
        test_opts='''
from pathlib import Path


def test_opts(agent_vcr, request):
    assert agent_vcr.mode == "all"
    assert request.config.getoption("agent_vcr_fail_on") == "never"
    assert agent_vcr.golden_dir == Path.cwd() / "custom" / "golden" / "test_opts.py--test_opts"
    agent_vcr.run("t", lambda rec: {"ok": 1})
'''
    )
    inner("--vcr-dir=custom", "--vcr-record=all", "--vcr-fail-on=never").assert_outcomes(passed=1)
    assert len(list((pytester.path / "custom" / "golden").glob("*/*.vcr"))) == 1
    # Long-form alias (for when pytest-vcr owns --vcr-record).
    inner("--vcr-dir=custom", "--agent-vcr-record=all", "--vcr-fail-on=never").assert_outcomes(
        passed=1
    )

    bad = inner("--vcr-record=sometimes")
    assert bad.ret == pytest.ExitCode.USAGE_ERROR
    bad.stderr.fnmatch_lines(["*invalid choice: 'sometimes'*"])

    helptext = inner("--help")
    helptext.stdout.fnmatch_lines(
        ["*--vcr-dir=DIR*", "*--vcr-record={none,once,all}*", "*--vcr-fail-on={degraded,failed,never}*"]
    )
    markers = inner("--markers")
    markers.stdout.fnmatch_lines(["*@pytest.mark.agent_vcr(task=None, identity=None, record=None)*"])


def test_marker_is_registered_for_strict_markers(pytester, inner) -> None:
    pytester.makepyfile(
        test_strict='''
import pytest


@pytest.mark.agent_vcr(task="t")
def test_strict():
    pass
'''
    )
    inner("--strict-markers").assert_outcomes(passed=1)


# -- vcr_recorder ------------------------------------------------------------------


def test_vcr_recorder_fixture_saves_a_session_named_after_the_test(pytester, inner) -> None:
    pytester.makepyfile(
        test_rec='''
import pytest


class TestGroup:
    @pytest.mark.parametrize("n", [1])
    def test_rec(self, vcr_recorder, n):
        assert vcr_recorder.get_session().session_id == "test_rec.py--TestGroup--test_rec-1"
        vcr_recorder.record_step("step", {"n": n}, {"n": n + 1})
'''
    )
    inner().assert_outcomes(passed=1)
    path = pytester.path / ".vcr" / "pytest" / "sessions" / "test_rec.py--TestGroup--test_rec-1.vcr"
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert lines[0]["data"]["metadata"]["agent_vcr"]["test"] == "test_rec.py::TestGroup::test_rec[1]"
    assert lines[1]["data"]["node_name"] == "step"


def test_node_slug_is_file_safe_and_bounded() -> None:
    assert node_slug("tests/a.py::TestX::test_y[p-1/2]") == "tests-a.py--TestX--test_y-p-1-2"
    long = node_slug("t.py::test[" + "x" * 300 + "]")
    assert len(long) <= 120 and "/" not in long
    assert long != node_slug("t.py::test[" + "x" * 299 + "y]")


# -- inert when unused ------------------------------------------------------------


def test_plugin_is_inert_when_unused(pytester, inner) -> None:
    pytester.makepyfile(test_plain="def test_plain():\n    assert 1 + 1 == 2\n")
    result = inner()
    result.assert_outcomes(passed=1)
    assert "agent-vcr" not in result.stdout.str()
    assert not (pytester.path / ".vcr").exists()


def test_entry_point_is_declared() -> None:
    text = (REPO / "pyproject.toml").read_text()
    section = text.split("[project.entry-points.pytest11]", 1)[1].split("[", 1)[0]
    assert 'agent_vcr = "agent_vcr.pytest_plugin"' in section


def test_entry_point_loads_without_heavy_imports(pytester: pytest.Pytester, tmp_path) -> None:
    """Installed via its entry point, the plugin imports no other Agent VCR module."""
    dist = tmp_path / "site" / "agent_vcr_entrypoint_probe-0.0.dist-info"
    dist.mkdir(parents=True)
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: agent-vcr-entrypoint-probe\n"
                                   "Version: 0.0\n")
    (dist / "entry_points.txt").write_text("[pytest11]\nagent_vcr = agent_vcr.pytest_plugin\n")
    pytester.makeconftest(
        '''
import sys


def pytest_sessionfinish(session):
    mods = sorted(m for m in sys.modules if m == "agent_vcr" or m.startswith("agent_vcr."))
    open("mods.txt", "w").write(",".join(mods))
    loaded = session.config.pluginmanager.get_plugin("agent_vcr")
    open("plugin.txt", "w").write(getattr(loaded, "__name__", "missing"))
'''
    )
    pytester.makepyfile(test_plain="def test_plain():\n    pass\n")
    env = {k: v for k, v in os.environ.items() if k != "PYTEST_DISABLE_PLUGIN_AUTOLOAD"}
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), str(tmp_path / "site"), *sys.path]
    )
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
        cwd=pytester.path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert (pytester.path / "plugin.txt").read_text() == "agent_vcr.pytest_plugin"
    assert (pytester.path / "mods.txt").read_text() == "agent_vcr,agent_vcr.pytest_plugin"
