"""pytest plugin: record an agent run once, replay it forever, zero tokens in CI.

Registered through the ``pytest11`` entry point, so it loads in every pytest
run where ``ai-agent-vcr`` is installed. It stays inert until a test asks for
the ``agent_vcr`` or ``vcr_recorder`` fixture: at startup it only registers
options and a marker. Every Agent VCR module (and pydantic) is imported inside
the fixtures, which keeps ``pytest --help`` and unrelated suites fast.

Golden layout (commit ``golden/``)::

    <vcr-dir>/golden/<test-slug>/index.json        GoldenRunCache index
    <vcr-dir>/golden/<test-slug>/<fingerprint>.vcr one golden run per task + identity
    <vcr-dir>/sessions/<test-slug>.vcr             vcr_recorder output

Record modes follow vcrpy / pytest-recording: ``none`` replays only (a missing
golden fails the test), ``once`` records when there is no golden yet and
replays otherwise, ``all`` always runs the agent and re-records. A golden is
written at teardown, and only when the test passed.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import warnings
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from importlib.util import find_spec
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

import pytest

if TYPE_CHECKING:  # pragma: no cover - type checkers only; keeps startup light
    from agent_vcr.golden_cache import CostLedger, GoldenRunCache
    from agent_vcr.player import VCRPlayer
    from agent_vcr.recorder import VCRRecorder
    from agent_vcr.regression import CaseResult

RECORD_MODES = ("none", "once", "all")
FAIL_ON_CHOICES = ("degraded", "failed", "never")
DEFAULT_VCR_DIR = ".vcr/pytest"
DEFAULT_RECORD = "once"
DEFAULT_FAIL_ON = "degraded"
# Below this total, latency drift is timer noise, not a regression. Hard
# ceilings (max_latency_ms) and an explicit max_latency_delta still apply.
LATENCY_NOISE_FLOOR_MS = 50.0
RESULT_NODE = "agent_vcr:result"
_META_KEY = "agent_vcr"
_FIXTURES = ("agent_vcr", "vcr_recorder")
_MARKER_KWARGS = ("task", "identity", "record")

AgentFn = Callable[["VCRRecorder"], Any]
StepFn = Callable[[dict], Any]


class AgentVCRWarning(UserWarning):
    """A run drifted from its golden but ``fail_on`` did not fail the test."""


@dataclass
class _Settings:
    vcr_dir: Path
    record: str | None  # None: no global setting (command line, env, ini)
    fail_on: str
    record_flag: str
    notice: str = ""  # shown once when a fixture is used (pytest-vcr owns --vcr-record)
    root: Path | None = None


@dataclass
class _Stats:
    recorded: int = 0
    replayed: int = 0
    saved: int = 0
    not_saved: int = 0
    tokens_saved: int = 0
    cost_saved_usd: float = 0.0
    notice_shown: bool = False


# State lives in private attributes, not pytest's stash API (pytest >= 7), so
# that importing and registering the plugin can never fail on an older pytest.
_SETTINGS_ATTR = "_agent_vcr_settings"
_STATS_ATTR = "_agent_vcr_stats"
_REPORTS_ATTR = "_agent_vcr_reports"
_USED_ATTR = "_agent_vcr_used"
ENV_RECORD = "AGENT_VCR_RECORD"
INI_RECORD = "agent_vcr_record"


def _settings(config: Any) -> _Settings:
    settings = getattr(config, _SETTINGS_ATTR, None)
    if settings is None:  # pytest_configure did not run (should not happen)
        settings = _Settings(Path(DEFAULT_VCR_DIR).resolve(), None, DEFAULT_FAIL_ON, "--vcr-record")
        setattr(config, _SETTINGS_ATTR, settings)
    return settings  # type: ignore[no-any-return]


def _stats(config: Any) -> _Stats:
    stats = getattr(config, _STATS_ATTR, None)
    if stats is None:
        stats = _Stats()
        setattr(config, _STATS_ATTR, stats)
    return stats  # type: ignore[no-any-return]


# -- options, marker, hooks ---------------------------------------------------


def _pytest_vcr_installed() -> bool:
    try:
        return find_spec("pytest_vcr") is not None
    except (ImportError, ValueError):
        return False


def _record_flags() -> list[str]:
    """``--vcr-record`` unless pytest-vcr (which owns that flag) is installed."""
    if _pytest_vcr_installed():
        return ["--agent-vcr-record"]
    return ["--vcr-record", "--agent-vcr-record"]


def _record_value(value: Any, source: str) -> str | None:
    text = str(value or "").strip().lower()
    if not text:
        return None
    if text not in RECORD_MODES:
        raise pytest.UsageError(
            f"agent_vcr: {source} must be one of {', '.join(RECORD_MODES)}, got {value!r}"
        )
    return text


def _resolve_record(config: Any, record_flag: str) -> tuple[str | None, str]:
    """Global record mode: command line > $AGENT_VCR_RECORD > ini > nothing.

    Returns ``(mode, notice)``. When pytest-vcr owns ``--vcr-record`` and it was
    given, the value went to pytest-vcr: ``none`` is still honoured (it can only
    stop live calls), and the notice says how to set agent_vcr's mode.
    """
    for value, source in (
        (config.getoption("agent_vcr_record", None), record_flag),
        (os.environ.get(ENV_RECORD), f"${ENV_RECORD}"),
        (config.getini(INI_RECORD), f"the {INI_RECORD} ini option"),
    ):
        mode = _record_value(value, source)
        if mode is not None:
            return mode, ""
    foreign = config.getoption("vcr_record", None) if _pytest_vcr_installed() else None
    if not foreign:
        return None, ""
    notice = (
        f"agent_vcr: --vcr-record={foreign} was taken by pytest-vcr, which owns that flag. "
        f"Use --agent-vcr-record, ${ENV_RECORD} or the {INI_RECORD} ini option for agent_vcr. "
    )
    if foreign == "none":
        return "none", notice + "Treating it as none, so the agent is not called live."
    return None, notice + f"agent_vcr ignored it and used its default ({DEFAULT_RECORD})."


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("agent_vcr", "Agent VCR: record agent runs once, replay them")
    group.addoption(
        "--vcr-dir",
        dest="agent_vcr_dir",
        default=None,
        metavar="DIR",
        help=f"goldens and recordings, relative to rootdir (default: {DEFAULT_VCR_DIR})",
    )
    group.addoption(
        *_record_flags(),
        dest="agent_vcr_record",
        default=None,
        choices=RECORD_MODES,
        help="none: replay only, a missing golden fails; once: record if there is no golden "
        "yet, else replay; all: always run the agent and re-record (default: once)",
    )
    group.addoption(
        "--vcr-fail-on",
        dest="agent_vcr_fail_on",
        default=DEFAULT_FAIL_ON,
        choices=FAIL_ON_CHOICES,
        help="which assert_matches_golden verdicts fail a test (default: degraded)",
    )
    parser.addini(
        INI_RECORD,
        f"agent_vcr record mode (none|once|all) when neither --agent-vcr-record nor "
        f"${ENV_RECORD} is set",
        default="",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "agent_vcr(task=None, identity=None, record=None): defaults for the agent_vcr fixture "
        f"(record: none|once|all; --agent-vcr-record, ${ENV_RECORD} and the {INI_RECORD} "
        "ini option win)",
    )
    # Relative to rootdir, not the cwd: goldens must be found wherever pytest
    # is run from (and --vcr-dir often comes from addopts).
    # rootpath is pytest >= 6.1; older versions only have the py.path rootdir.
    rootpath = getattr(config, "rootpath", None) or Path(str(getattr(config, "rootdir", ".")))
    vcr_dir = Path(rootpath, config.getoption("agent_vcr_dir", None) or DEFAULT_VCR_DIR)
    record_flag = _record_flags()[0]
    record, notice = _resolve_record(config, record_flag)
    setattr(
        config,
        _SETTINGS_ATTR,
        _Settings(
            vcr_dir=vcr_dir,
            record=record,
            fail_on=config.getoption("agent_vcr_fail_on", DEFAULT_FAIL_ON) or DEFAULT_FAIL_ON,
            record_flag=record_flag,
            notice=notice,
            root=Path(rootpath),
        ),
    )
    setattr(config, _STATS_ATTR, _Stats())


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item: pytest.Item, call: pytest.CallInfo[None]) -> Iterator[None]:
    """Keep phase reports so teardown knows whether the test passed."""
    outcome = yield
    if getattr(item, _USED_ATTR, False) or any(
        name in getattr(item, "fixturenames", ()) for name in _FIXTURES
    ):
        report: pytest.TestReport = outcome.get_result()  # type: ignore[attr-defined]
        reports = getattr(item, _REPORTS_ATTR, None)
        if reports is None:
            reports = {}
            setattr(item, _REPORTS_ATTR, reports)
        reports[report.when] = report


def pytest_terminal_summary(terminalreporter: Any, config: pytest.Config) -> None:
    stats = getattr(config, _STATS_ATTR, None)
    if stats is None or not (stats.recorded or stats.replayed or stats.notice_shown):
        return
    parts = []
    if stats.replayed:
        parts.append(
            f"{stats.replayed} run(s) replayed (agent not called, {stats.tokens_saved:,} tokens / "
            f"${stats.cost_saved_usd:.4f} not spent)"
        )
    if stats.recorded:
        parts.append(f"{stats.recorded} run(s) recorded, {stats.saved} saved as golden")
    if stats.not_saved:
        parts.append(f"{stats.not_saved} not saved because the test failed")
    terminalreporter.write_sep("-", "agent-vcr")
    if parts:
        terminalreporter.write_line("agent-vcr: " + "; ".join(parts))
    if stats.notice_shown:
        terminalreporter.write_line(_settings(config).notice, yellow=True, bold=True)


# -- helpers --------------------------------------------------------------------


_SLUG_RE = re.compile(r"[^A-Za-z0-9._-]+")


_SLUG_MAX = 60


def node_slug(nodeid: str) -> str:
    """File-name-safe, unique form of a test node id.

    ``a/b.py::T::t[x]`` → ``b.py--T--t-x-<hash>``. The readable part is lossy
    (``[a/b]`` and ``[a-b]`` read the same) and is capped at 60 characters, so a
    hash of the exact node id is always appended: distinct tests never share a
    folder, even on case-insensitive filesystems, and paths stay short (Windows).
    """
    path, sep, rest = nodeid.partition("::")
    short = path.rsplit("/", 1)[-1] + sep + rest  # folders are in the hash, not the name
    readable = _SLUG_RE.sub("-", short.replace("::", "--")).strip("-.")
    readable = readable[:_SLUG_MAX].strip("-.") or "test"
    digest = hashlib.sha256(nodeid.encode("utf-8")).hexdigest()[:10]
    return f"{readable}-{digest}"


def _canonical(value: Any, where: str) -> Any:
    """A JSON value whose encoding never depends on hash seeds or object ids.

    Sets are sorted; anything without a stable JSON form raises ``TypeError``
    (``default=str`` would make set order, and so the golden key, vary per run).
    """
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if isinstance(key, str):
                name = key
            elif key is None or isinstance(key, (bool, int, float)):
                name = json.dumps(key)  # what json.dumps does with such keys
            else:
                raise TypeError(f"{where}: dict key {key!r} is not a string or number")
            out[name] = _canonical(item, f"{where}[{name!r}]")
        return out
    if isinstance(value, (list, tuple)):
        return [_canonical(item, f"{where}[{i}]") for i, item in enumerate(value)]
    if isinstance(value, (set, frozenset)):
        items = [_canonical(item, f"{where}{{...}}") for item in value]
        return sorted(items, key=lambda item: json.dumps(item, sort_keys=True))
    raise TypeError(
        f"{where} holds a {type(value).__name__}, which has no stable JSON form; use strings, "
        "numbers, lists, dicts or sets (for example a hash or str() of the value)"
    )


def _normalize(identity: dict[str, Any] | None) -> dict[str, Any]:
    if not identity:
        return {}
    return dict(_canonical(identity, "identity"))


def _changed_keys(old: dict[str, Any], new: dict[str, Any]) -> list[str]:
    return sorted(k for k in set(old) | set(new) if old.get(k) != new.get(k))


def _short(value: Any, limit: int = 70) -> str:
    text = repr(value)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _pct(new: float, old: float) -> str:
    return f"{(new - old) / abs(old) * 100:+.0f}%" if old else "was 0"


def _test_passed(item: pytest.Item) -> bool:
    reports = getattr(item, _REPORTS_ATTR, None) or {}
    call = reports.get("call")
    if call is None or not call.passed:
        return False
    # No setup report: the fixture was requested during the call
    # (request.getfixturevalue), so setup had already passed.
    setup = reports.get("setup")
    return setup is None or bool(setup.passed)


# -- results --------------------------------------------------------------------


@dataclass
class AgentRun:
    """What one ``agent_vcr.run()`` produced.

    ``output`` is always the *recorded* value (it went through the recording's
    serializer), so a test sees the same thing on the first run and on replay.
    """

    task: str
    identity: dict[str, Any]
    mode: str
    replayed: bool
    output: Any
    outputs: list[Any]
    recording: VCRPlayer
    fingerprint: str
    golden_path: Path | None = None
    ledger: CostLedger | None = None
    baseline: VCRPlayer | None = None
    baseline_note: str = ""
    _recorder: VCRRecorder | None = field(default=None, repr=False)

    @property
    def tokens(self) -> int:
        return int(self.recording.get_total_tokens())

    @property
    def cost_usd(self) -> float:
        return float(self.recording.get_total_cost())


class AgentVCR:
    """The ``agent_vcr`` fixture: run an agent through a recorder, or replay its golden."""

    def __init__(
        self,
        item: pytest.Item,
        settings: _Settings,
        stats: _Stats,
        tmp_factory: pytest.TempPathFactory,
    ) -> None:
        self._item = item
        self._settings = settings
        self._stats = stats
        self._tmp_factory = tmp_factory
        self._tmp: Path | None = None
        self._pending: list[AgentRun] = []  # live runs to save as goldens at teardown
        self.runs: list[AgentRun] = []
        self.nodeid = item.nodeid
        self.slug = node_slug(item.nodeid)
        self.golden_dir = settings.vcr_dir / "golden" / self.slug
        self._marker = self._read_marker()
        self.mode = self._resolve_mode()

    # -- public API -------------------------------------------------------------

    def run(
        self,
        task: str | AgentFn | Sequence[tuple[str, StepFn]] | None = None,
        agent: AgentFn | Sequence[tuple[str, StepFn]] | None = None,
        *,
        identity: dict[str, Any] | None = None,
        input: dict[str, Any] | None = None,
    ) -> AgentRun:
        """Run ``agent`` for ``task`` live (and record it), or replay its golden.

        ``agent`` is either a callable ``agent(recorder) -> result`` that
        records its own steps on the given ``VCRRecorder``, or a step pipeline
        ``[(name, step), ...]`` where ``step(state) -> state`` (or
        ``(state, usage)``; usage is a dict or ``FrameMetadata``) runs from
        ``input``. ``task`` and ``identity`` default to the ``agent_vcr`` marker.
        """
        if agent is None and task is not None and not isinstance(task, str):
            task, agent = None, task
        task = task if task is not None else self._marker.get("task")
        if not isinstance(task, str) or not task.strip():
            raise TypeError("agent_vcr.run() needs a task string (or @pytest.mark.agent_vcr(task=))")
        if agent is None:
            raise TypeError("agent_vcr.run() needs an agent callable or a list of (name, step)")
        if identity is None:
            identity = self._marker.get("identity")
        if identity is not None and not isinstance(identity, dict):
            raise TypeError("identity must be a dict (see GoldenRunCache.build_identity)")
        steps = self._as_steps(agent, input)
        identity = _normalize(identity)
        if steps is not None:
            # The pipeline input is part of what makes a golden valid.
            digest = hashlib.sha256(
                json.dumps(_canonical(input, "input"), sort_keys=True).encode()
            ).hexdigest()
            identity = {**identity, "input_sha256": digest[:16]}
        for prev in self.runs:
            if prev.task.strip().lower() == task.strip().lower() and prev.identity == identity:
                raise ValueError(f"agent_vcr.run() already ran task {task!r} in this test")

        golden_info, cache = self._lookup(task, identity)
        if golden_info is not None and cache is not None and self.mode != "all":
            result = self._replay(task, identity, golden_info, cache)
        elif self.mode == "none":
            pytest.fail(self._no_golden_message(task, identity, cache), pytrace=False)
        else:
            result = self._record(task, identity, agent, steps, input, golden_info, cache)
        self.runs.append(result)
        return result

    def assert_matches_golden(
        self,
        run: AgentRun | None = None,
        *,
        fail_on: str | None = None,
        expect: Any | None = None,
        **expectations: Any,
    ) -> CaseResult:
        """Fail the test when ``run`` regressed against its golden (Agent CI rules).

        Keyword arguments are ``agent_vcr.datasets.Expectation`` fields:
        budgets (``max_cost_usd``, ``max_latency_ms``, ``max_tokens``),
        ``require_no_errors``, ``expect_state``, ``forbid_tools``,
        ``require_checkpoints`` and the drift limits ``max_cost_delta``,
        ``max_latency_delta``, ``max_token_delta`` (fractions; ``None`` = off).
        """
        __tracebackhide__ = True
        from agent_vcr.datasets import Expectation, TestCase
        from agent_vcr.regression import CaseVerdict, evaluate_candidate

        if run is None:
            if not self.runs:
                raise RuntimeError("assert_matches_golden() needs a prior agent_vcr.run()")
            run = self.runs[-1]
        fail_on = fail_on or self._settings.fail_on
        if fail_on not in FAIL_ON_CHOICES:
            raise ValueError(f"fail_on must be one of {', '.join(FAIL_ON_CHOICES)}")
        unknown = sorted(set(expectations) - set(Expectation.model_fields))
        if unknown:
            raise TypeError(f"unknown expectation(s): {', '.join(unknown)}")
        base = expect if expect is not None else Expectation()
        exp = base.model_copy(update=expectations)
        if (
            "max_latency_delta" not in expectations
            and run.baseline is not None
            and run.baseline.get_total_latency() < LATENCY_NOISE_FLOOR_MS
        ):
            exp = exp.model_copy(update={"max_latency_delta": None})
        case = TestCase(
            name=self.nodeid,
            task=run.task,
            fingerprint=run.fingerprint or "unsaved",
            identity=run.identity,
            expect=exp,
        )
        result = evaluate_candidate(case, run.recording, run.baseline)
        failing = (result.verdict == CaseVerdict.FAIL and fail_on != "never") or (
            result.verdict == CaseVerdict.DEGRADED and fail_on == "degraded"
        )
        if result.verdict != CaseVerdict.PASS:
            message = self._regression_message(run, result, fail_on)
            if failing:
                raise AssertionError(message)
            warnings.warn(AgentVCRWarning(message), stacklevel=2)
        return result

    # -- record / replay -------------------------------------------------------

    def _replay(
        self, task: str, identity: dict[str, Any], info: dict[str, Any], cache: GoldenRunCache
    ) -> AgentRun:
        from agent_vcr.golden_cache import GhostReplayMode
        from agent_vcr.player import VCRPlayer
        from agent_vcr.recorder import VCRRecorder

        replay_rec = VCRRecorder(output_dir=str(self._tmpdir() / "replays"))
        outputs, ledger = cache.replay_with_mode(
            task, GhostReplayMode.FULL, identity=identity, recorder=replay_rec
        )
        golden = VCRPlayer.load(info["file"])
        self._stats.replayed += 1
        self._stats.tokens_saved += ledger.tokens_saved
        self._stats.cost_saved_usd += ledger.cost_saved_usd
        return AgentRun(
            task=task,
            identity=identity,
            mode=self.mode,
            replayed=True,
            output=_result_from(golden, outputs),
            outputs=outputs,
            recording=golden,
            fingerprint=str(_fingerprint_of(cache, info)),
            golden_path=Path(info["file"]),
            ledger=ledger,
        )

    def _record(
        self,
        task: str,
        identity: dict[str, Any],
        agent: Any,
        steps: list[tuple[str, StepFn]] | None,
        input: dict[str, Any] | None,
        golden_info: dict[str, Any] | None,
        cache: GoldenRunCache | None,
    ) -> AgentRun:
        from agent_vcr.models import FrameMetadata, StateSerializer, json_dumps_safe
        from agent_vcr.player import VCRPlayer
        from agent_vcr.recorder import VCRRecorder

        rec = VCRRecorder(output_dir=str(self._tmpdir() / "live"), auto_save=False)
        session = rec.start_session(
            self.slug, tags=["pytest"], metadata={_META_KEY: {"test": self.nodeid, "task": task}}
        )
        kind = "pipeline" if steps is not None else "agent"
        result_kind = "state"
        if steps is not None:
            state: dict[str, Any] = dict(input or {})
            for name, step in steps:
                started = time.perf_counter()
                returned = step(state)
                elapsed = (time.perf_counter() - started) * 1000
                out, meta = _split_step_result(name, returned)
                meta = FrameMetadata(**meta) if isinstance(meta, dict) else meta
                if meta is None:
                    meta = FrameMetadata()
                if not meta.latency_ms:
                    meta = meta.model_copy(update={"latency_ms": elapsed})
                rec.record_step(name, state, out, metadata=meta)
                state = out
        else:
            value = agent(rec)
            if value is not None:
                result_kind = "dict" if isinstance(value, dict) else "value"
                frame = rec.record_step(
                    RESULT_NODE,
                    {},
                    value if isinstance(value, dict) else {"result": value},
                    metadata=FrameMetadata(custom={"source": RESULT_NODE}),
                )
                frame.metadata.latency_ms = 0.0  # bookkeeping, not agent time
            else:
                result_kind = "none"
        session.metadata[_META_KEY].update({"kind": kind, "result": result_kind})
        frames = rec.get_frames()
        if not frames:
            pytest.fail(
                f"agent_vcr: the agent for task {task!r} recorded no steps and returned None, "
                "so there is nothing to replay. Record steps on the recorder it receives, or "
                "return its result.",
                pytrace=False,
            )
        # Round-trip like a replay from disk does, so record and replay match.
        outputs = [
            StateSerializer.deserialize(json.loads(json_dumps_safe(f.output_state)))
            for f in frames
        ]
        recording = VCRPlayer(session, frames)
        baseline, note = self._baseline_for(task, identity, golden_info, cache)
        run = AgentRun(
            task=task,
            identity=identity,
            mode=self.mode,
            replayed=False,
            output=_result_from(recording, outputs),
            outputs=outputs,
            recording=recording,
            fingerprint=str(_fingerprint_of(cache, golden_info)) if golden_info else "",
            golden_path=Path(golden_info["file"]) if golden_info else None,
            baseline=baseline,
            baseline_note=note,
            _recorder=rec,
        )
        self._pending.append(run)
        self._stats.recorded += 1
        return run

    def _finish(self) -> None:
        """Teardown: save fresh recordings as goldens, but only if the test passed."""
        if not self._pending:
            return
        if not _test_passed(self._item):
            self._stats.not_saved += len(self._pending)
            return
        from agent_vcr.golden_cache import GoldenRunCache

        cache = GoldenRunCache(str(self.golden_dir))
        for run in self._pending:
            assert run._recorder is not None
            run.fingerprint = cache.save_golden_run(
                run.task, run._recorder, tags=["pytest"], identity=run.identity
            )
            info = cache.get_golden_info(run.task, run.identity)
            run.golden_path = Path(info["file"]) if info else None
            self._stats.saved += 1

    # -- internals -----------------------------------------------------------------

    def _read_marker(self) -> dict[str, Any]:
        marker = self._item.get_closest_marker("agent_vcr")
        if marker is None:
            return {}
        kwargs = dict(marker.kwargs)
        if marker.args:
            kwargs.setdefault("task", marker.args[0])
        unknown = sorted(set(kwargs) - set(_MARKER_KWARGS))
        if unknown:
            raise TypeError(f"@pytest.mark.agent_vcr got unknown argument(s): {', '.join(unknown)}")
        record = kwargs.get("record")
        if record is not None and record not in RECORD_MODES:
            raise ValueError(f"@pytest.mark.agent_vcr(record=) must be one of {RECORD_MODES}")
        return kwargs

    def _resolve_mode(self) -> str:
        # Command line > $AGENT_VCR_RECORD > ini > marker > "once": a global
        # none (CI) always wins over a marker.
        return self._settings.record or self._marker.get("record") or DEFAULT_RECORD

    def _show(self, path: Path | None) -> str:
        """``path`` relative to rootdir when it is inside it (shorter messages)."""
        if path is None:
            return "-"
        try:
            return str(Path(path).relative_to(self._settings.root or Path.cwd()))
        except ValueError:
            return str(path)

    def _tmpdir(self) -> Path:
        if self._tmp is None:
            self._tmp = self._tmp_factory.mktemp("agent_vcr-" + self.slug[-40:].strip("-."))
        return self._tmp

    @staticmethod
    def _as_steps(agent: Any, input: dict[str, Any] | None) -> list[tuple[str, StepFn]] | None:
        if callable(agent):
            if input is not None:
                raise TypeError(
                    "input= is for step pipelines; bind an agent's arguments with a lambda"
                )
            return None
        if isinstance(agent, Sequence) and not isinstance(agent, (str, bytes)):
            steps = list(agent)
            for item in steps:
                if not (
                    isinstance(item, tuple)
                    and len(item) == 2
                    and isinstance(item[0], str)
                    and callable(item[1])
                ):
                    raise TypeError("a step pipeline is a list of (name, step_fn) tuples")
            if not steps:
                raise TypeError("a step pipeline needs at least one (name, step_fn)")
            if input is not None and not isinstance(input, dict):
                raise TypeError("input= must be a dict state")
            return steps
        raise TypeError("agent must be a callable or a list of (name, step_fn) tuples")

    def _lookup(
        self, task: str, identity: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, GoldenRunCache | None]:
        """Find the golden for task + identity without creating any directory."""
        if not (self.golden_dir / "index.json").is_file():
            return None, None
        from agent_vcr.golden_cache import GoldenRunCache

        cache = GoldenRunCache(str(self.golden_dir))
        info = cache.get_golden_info(task, identity)
        if info is not None and not Path(info["file"]).is_file():
            info = None  # index entry without its recording: treat as missing
        return info, cache

    def _same_task(self, task: str, cache: GoldenRunCache | None) -> list[dict[str, Any]]:
        if cache is None:
            return []
        wanted = task.strip().lower()
        runs = [
            r
            for r in cache.list_golden_runs()
            if str(r.get("task", "")).strip().lower() == wanted and Path(r["file"]).is_file()
        ]
        return sorted(runs, key=lambda r: str(r.get("created_at", "")))

    def _baseline_for(
        self,
        task: str,
        identity: dict[str, Any],
        golden_info: dict[str, Any] | None,
        cache: GoldenRunCache | None,
    ) -> tuple[VCRPlayer | None, str]:
        from agent_vcr.player import VCRPlayer

        if golden_info is not None:
            return VCRPlayer.load(golden_info["file"]), "golden"
        others = self._same_task(task, cache)
        if not others:
            return None, ""
        latest = others[-1]
        changed = _changed_keys(dict(latest.get("identity") or {}), identity)
        return (
            VCRPlayer.load(latest["file"]),
            f"previous golden for this task (identity changed: {', '.join(changed)})",
        )

    def _no_golden_message(
        self, task: str, identity: dict[str, Any], cache: GoldenRunCache | None
    ) -> str:
        flag = self._settings.record_flag
        lines = [
            f"agent_vcr: no golden recording for this test, and recording is off "
            f"({flag}=none), so the agent was not called.",
            f"  test:      {self.nodeid}",
            f"  task:      {task!r}",
            f"  identity:  {json.dumps(identity, sort_keys=True)}",
            f"  looked in: {self._show(self.golden_dir)}",
        ]
        others = self._same_task(task, cache)
        if others:
            changed = _changed_keys(dict(others[-1].get("identity") or {}), identity)
            lines.append(
                f"  A golden for this task exists with a different identity "
                f"(changed: {', '.join(changed)}). Re-record it for the new identity."
            )
        lines += [
            "Record it once where the agent can run, review it, and commit the folder:",
            f"  pytest {flag}=once '{self.nodeid}'",
        ]
        return "\n".join(lines)

    def _regression_message(self, run: AgentRun, result: CaseResult, fail_on: str) -> str:
        verdict = result.verdict.value.upper()
        lines = [
            f"agent_vcr: {verdict} vs golden (fail_on={fail_on})",
            f"  test:      {self.nodeid}",
            f"  task:      {run.task!r}",
        ]
        if run.replayed:
            lines.append(f"  golden:    {self._show(run.golden_path)} (replayed)")
        else:
            if run.golden_path is not None:
                lines.append(f"  golden:    {self._show(run.golden_path)}")
            lines.append(f"  candidate: {self._save_candidate(run)}")
        if run.baseline_note and run.baseline_note != "golden":
            lines.append(f"  baseline:  {run.baseline_note}")
        lines.append("  reasons:")
        lines += [f"    - {reason}" for reason in result.reasons]
        diff = result.diff
        if diff.has_baseline and run.baseline is not None:
            base, cand = run.baseline, run.recording
            rows = [
                ("frames", diff.baseline_frames, diff.candidate_frames, "{}"),
                ("tokens", base.get_total_tokens(), cand.get_total_tokens(), "{:,}"),
                ("cost", base.get_total_cost(), cand.get_total_cost(), "${:.4f}"),
                ("latency", base.get_total_latency(), cand.get_total_latency(), "{:.0f}ms"),
            ]
            lines.append("  golden -> this run:")
            for label, old, new, fmt in rows:
                change = "" if old == new else f"  ({_pct(float(new), float(old))})"
                lines.append(f"    {label:<8} {fmt.format(old)} -> {fmt.format(new)}{change}")
            for label, added, removed in (
                ("tools", diff.added_tools, diff.removed_tools),
                ("models", diff.added_models, diff.removed_models),
            ):
                if added or removed:
                    marks = [f"+{x}" for x in added] + [f"-{x}" for x in removed]
                    lines.append(f"    {label:<8} {' '.join(marks)}")
            if diff.missing_checkpoints:
                lines.append(f"    dropped checkpoints: {', '.join(diff.missing_checkpoints)}")
            state = diff.state_diff
            if not diff.states_match:
                lines.append("  final state:")
                for key, change in sorted(state.get("modified", {}).items()):
                    lines.append(
                        f"    ~ {key}: {_short(change['before'])} -> {_short(change['after'])}"
                    )
                for key, value in sorted(state.get("added", {}).items()):
                    lines.append(f"    + {key}: {_short(value)}")
                for key, value in sorted(state.get("removed", {}).items()):
                    lines.append(f"    - {key}: {_short(value)}")
        elif run.replayed:
            lines.append(
                "  (replayed from the golden: only your expectations were checked; "
                f"{self._settings.record_flag}=all compares a fresh run)"
            )
        if diff.has_baseline:
            lines += [
                "To accept this run as the new golden:",
                f"  pytest {self._settings.record_flag}=all --vcr-fail-on=never '{self.nodeid}'",
            ]
        return "\n".join(lines)

    def _save_candidate(self, run: AgentRun) -> Path:
        """Write a fresh run to the test's tmp dir so it can be opened with ``vcr``."""
        from agent_vcr.recorder import rewrite_session_file

        index = self.runs.index(run) if run in self.runs else len(self.runs)
        path = self._tmpdir() / f"candidate-{index}.vcr"
        rewrite_session_file(path, run.recording.session, run.recording.frames)
        return path


def _split_step_result(name: str, returned: Any) -> tuple[dict[str, Any], Any]:
    from agent_vcr.models import FrameMetadata

    meta: Any = None
    if (
        isinstance(returned, tuple)
        and len(returned) == 2
        and isinstance(returned[1], (dict, FrameMetadata))
    ):
        returned, meta = returned
    if not isinstance(returned, dict):
        raise TypeError(
            f"step {name!r} must return a dict state (or (state, usage)), "
            f"got {type(returned).__name__}"
        )
    return returned, meta


def _result_from(recording: VCRPlayer, outputs: list[Any]) -> Any:
    """The agent's return value as stored in the recording."""
    meta = recording.session.metadata.get(_META_KEY) or {}
    kind = meta.get("result", "state")
    if kind == "none" or not outputs:
        return None
    last = outputs[-1]
    if kind == "value" and isinstance(last, dict):
        return last.get("result")
    return last


def _fingerprint_of(cache: GoldenRunCache | None, info: dict[str, Any] | None) -> str:
    if cache is None or info is None:
        return ""
    for run in cache.list_golden_runs():
        if run.get("file") == info.get("file"):
            return str(run.get("fingerprint", ""))
    return ""


# -- fixtures -------------------------------------------------------------------


def _mark_used(request: pytest.FixtureRequest) -> None:
    """Flag the item so its reports are kept, also for request.getfixturevalue()."""
    setattr(request.node, _USED_ATTR, True)
    settings, stats = _settings(request.config), _stats(request.config)
    if settings.notice and not stats.notice_shown:
        stats.notice_shown = True
        warnings.warn(AgentVCRWarning(settings.notice), stacklevel=3)


@pytest.fixture
def agent_vcr(
    request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[AgentVCR]:
    """Record your agent once, replay it forever: ``agent_vcr.run(task, agent)``."""
    _mark_used(request)
    session = AgentVCR(
        request.node, _settings(request.config), _stats(request.config), tmp_path_factory
    )
    yield session
    session._finish()


@pytest.fixture
def vcr_recorder(request: pytest.FixtureRequest) -> Iterator[VCRRecorder]:
    """A started ``VCRRecorder`` session named after the test, saved at teardown."""
    from agent_vcr.recorder import VCRRecorder

    _mark_used(request)
    settings = _settings(request.config)
    rec = VCRRecorder(output_dir=str(settings.vcr_dir / "sessions"))
    rec.start_session(
        node_slug(request.node.nodeid),
        tags=["pytest"],
        metadata={_META_KEY: {"test": request.node.nodeid}},
    )
    yield rec
    rec.save()
