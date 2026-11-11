"""CrewAI integration tests.

* ``TestPydanticCrewSemantics`` runs everywhere: it uses small pydantic models that
  mimic how CrewAI (0.1xx-0.2xx) propagates callbacks at kickoff, so the
  save/chain/restore logic is covered without CrewAI installed.
* ``TestRealCrewAI`` / ``TestRealEventBus`` run against a real ``crewai``
  installation (skipped otherwise) with a fake LLM, so no network is needed.
"""

from __future__ import annotations

import asyncio
import gc
import os
import tempfile
import warnings
from typing import Any, Callable

import pytest
from pydantic import BaseModel, ConfigDict

from agent_vcr import VCRRecorder
from agent_vcr.integrations.crewai import VCRCrewAI, VCRCrewCallback
from agent_vcr.models import FrameType

# Keep CrewAI quiet/offline: no first-run trace prompt, no telemetry.
os.environ.setdefault("CREWAI_TESTING", "true")
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")


def make_recorder(name: str) -> VCRRecorder:
    recorder = VCRRecorder(output_dir=tempfile.mkdtemp(), auto_save=False)
    recorder.start_session(name)
    return recorder


# ---------------------------------------------------------------------------
# Pydantic fakes mimicking CrewAI callback propagation
# ---------------------------------------------------------------------------


class FakeAgent(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    role: str
    step_callback: Any = None


class FakeTaskOutput(BaseModel):
    description: str
    raw: str
    agent: str
    name: Any = None


class FakeTask(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    description: str
    agent: FakeAgent
    callback: Any = None
    name: Any = None

    def run(self, crew: FakeCrew) -> None:
        cb = self.agent.step_callback
        if cb:
            cb({"agent": self.agent.role, "input": {"t": self.description}, "output": {}})
        out = FakeTaskOutput(description=self.description, raw="ok", agent=self.agent.role)
        if self.callback:
            self.callback(out)
        if crew.task_callback and crew.task_callback != self.callback:
            crew.task_callback(out)


class FakeCrew(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    agents: list[FakeAgent]
    tasks: list[FakeTask]
    step_callback: Any = None
    task_callback: Any = None
    fail: bool = False

    def kickoff(self, inputs: dict[str, Any] | None = None) -> str:
        for task in self.tasks:  # Crew._set_tasks_callbacks
            if not task.callback:
                task.callback = self.task_callback
        for agent in self.agents:
            if not agent.step_callback:
                agent.step_callback = self.step_callback
        for task in self.tasks:
            task.run(self)
        if self.fail:
            raise RuntimeError("crew failed")
        return "done"


FakeTask.model_rebuild()


def make_fake_crew(**kwargs: Any) -> FakeCrew:
    a1, a2 = FakeAgent(role="r1"), FakeAgent(role="r2")
    t1 = FakeTask(description="first task " + "x" * 200, agent=a1)
    t2 = FakeTask(description="second task", agent=a2)
    return FakeCrew(agents=[a1, a2], tasks=[t1, t2], **kwargs)


def counter() -> tuple[list[Any], Callable[[Any], None]]:
    calls: list[Any] = []
    return calls, calls.append


class TestPydanticCrewSemantics:
    def test_pydantic_tasks_are_not_hashed(self) -> None:
        crew = make_fake_crew()
        rec = make_recorder("p1")
        VCRCrewAI(rec).kickoff(crew)  # used to raise TypeError: unhashable type
        names = [f.node_name for f in rec.frames]
        assert names.count("task:second task") == 1
        assert sum(n.startswith("task:first task") for n in names) == 1
        assert sum(n in ("r1", "r2") for n in names) == 2

    def test_long_description_gives_short_node_name(self) -> None:
        crew = make_fake_crew()
        rec = make_recorder("p2")
        VCRCrewAI(rec).kickoff(crew)
        task_frames = [f for f in rec.frames if f.node_name.startswith("task:first")]
        assert len(task_frames[0].node_name) <= len("task:") + 60
        assert task_frames[0].input_state["description"].endswith("x" * 200)

    def test_sequential_kickoffs_are_isolated_and_restored(self) -> None:
        crew = make_fake_crew()
        rec1, rec2 = make_recorder("a"), make_recorder("b")
        VCRCrewAI(rec1).kickoff(crew)
        n1 = len(rec1.frames)
        assert n1 == 4
        VCRCrewAI(rec2).kickoff(crew)
        assert len(rec1.frames) == n1
        assert len(rec2.frames) == 4
        crew.kickoff()  # plain kickoff (fake propagates crew callbacks -> None)
        assert len(rec1.frames) == n1 and len(rec2.frames) == 4
        assert crew.step_callback is None and crew.task_callback is None
        assert all(a.step_callback is None for a in crew.agents)

    def test_user_callbacks_chained_once(self) -> None:
        crew_steps, crew_step_cb = counter()
        crew_tasks, crew_task_cb = counter()
        own_steps, own_step_cb = counter()
        own_tasks, own_task_cb = counter()
        crew = make_fake_crew(step_callback=crew_step_cb, task_callback=crew_task_cb)
        crew.agents[0].step_callback = own_step_cb
        crew.tasks[0].callback = own_task_cb

        rec = make_recorder("c")
        VCRCrewAI(rec).kickoff(crew)

        assert len(own_steps) == 1  # agent r1's own callback
        assert len(crew_steps) == 1  # agent r2 inherits the crew one
        assert len(own_tasks) == 1
        assert len(crew_tasks) == 2  # crew.task_callback fires for every task
        assert len(rec.frames) == 4  # 2 steps + 2 tasks, no duplicates
        assert crew.agents[0].step_callback is own_step_cb
        assert crew.agents[1].step_callback is None
        assert crew.tasks[0].callback is own_task_cb
        assert crew.step_callback is crew_step_cb and crew.task_callback is crew_task_cb

    def test_restored_on_error(self) -> None:
        crew = make_fake_crew(fail=True)
        rec = make_recorder("e")
        with pytest.raises(RuntimeError, match="crew failed"):
            VCRCrewAI(rec).kickoff(crew)
        assert any(f.frame_type == FrameType.ERROR for f in rec.frames)
        assert crew.step_callback is None and crew.task_callback is None
        assert all(t.callback is None for t in crew.tasks)
        assert all(a.step_callback is None for a in crew.agents)

    def test_recorder_failure_does_not_break_crew(self) -> None:
        crew = make_fake_crew()
        rec = VCRRecorder(output_dir=tempfile.mkdtemp(), auto_save=False)  # no session
        assert VCRCrewAI(rec).kickoff(crew) == "done"

    def test_wrap_task_on_pydantic_task_uses_callback(self) -> None:
        crew = make_fake_crew()
        rec = make_recorder("w")
        VCRCrewAI(rec).wrap_task(crew.tasks[1])  # used to raise ValueError
        VCRCrewAI(rec).wrap_task(crew.tasks[1])  # idempotent
        crew.kickoff()
        assert [f.node_name for f in rec.frames] == ["task:second task"]

    def test_kickoff_async(self) -> None:
        class AsyncCrew(FakeCrew):
            async def kickoff_async(self, inputs: dict[str, Any] | None = None) -> str:
                return await asyncio.to_thread(self.kickoff, inputs)

        a = FakeAgent(role="r")
        crew = AsyncCrew(agents=[a], tasks=[FakeTask(description="t", agent=a)])
        rec = make_recorder("async")
        assert asyncio.run(VCRCrewAI(rec).kickoff_async(crew, timeout=30)) == "done"
        assert len(rec.frames) == 2
        assert crew.task_callback is None and a.step_callback is None

    def test_unattached_listener_warns(self) -> None:
        rec = make_recorder("warn")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            VCRCrewCallback(rec)
            gc.collect()
        assert any("never attached" in str(w.message) for w in caught)


# ---------------------------------------------------------------------------
# Real CrewAI
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def crewai_mod() -> Any:
    return pytest.importorskip("crewai")


@pytest.fixture(scope="module")
def fake_llm_cls(crewai_mod: Any) -> Any:
    base_llm = pytest.importorskip("crewai.llms.base_llm")

    from crewai.events import LLMCallCompletedEvent, crewai_event_bus
    from crewai.events.types.llm_events import LLMCallType

    class FakeLLM(base_llm.BaseLLM):  # type: ignore[misc,name-defined]
        """Scripted LLM; emits LLMCallCompletedEvent like crewai.LLM does."""

        def __init__(self, fail: bool = False, script: list[str] | None = None) -> None:
            super().__init__(model="fake-llm")
            self.fail = fail
            self.script = list(script or [])

        def call(self, messages: Any, *args: Any, **kwargs: Any) -> str:
            if self.fail:
                raise RuntimeError("llm exploded")
            answer = self.script.pop(0) if self.script else "Thought: done\nFinal Answer: 42"
            crewai_event_bus.emit(
                self,
                event=LLMCallCompletedEvent(
                    messages=messages,
                    response=answer,
                    call_type=LLMCallType.LLM_CALL,
                    model=self.model,
                ),
            )
            return answer

        def supports_function_calling(self) -> bool:
            return False

        def supports_stop_words(self) -> bool:
            return False

        def get_context_window_size(self) -> int:
            return 8192

    return FakeLLM


def real_crew(
    crewai_mod: Any,
    llm_cls: Any,
    n_tasks: int = 1,
    fail: bool = False,
    script: list[str] | None = None,
    tools: list[Any] | None = None,
) -> Any:
    agent = crewai_mod.Agent(
        role="researcher",
        goal="g",
        backstory="b",
        llm=llm_cls(fail=fail, script=script),
        tools=tools or [],
        verbose=False,
        max_retry_limit=0,
    )
    tasks = [
        crewai_mod.Task(description=f"say 42 #{i}", expected_output="a number", agent=agent)
        for i in range(n_tasks)
    ]
    return crewai_mod.Crew(agents=[agent], tasks=tasks, verbose=False)


def assert_clean(crew: Any) -> None:
    assert crew.step_callback is None
    assert crew.task_callback is None
    assert all(a.step_callback is None for a in crew.agents)
    assert all(t.callback is None for t in crew.tasks)


class TestRealCrewAI:
    def test_two_kickoffs_record_into_right_recorders(
        self, crewai_mod: Any, fake_llm_cls: Any
    ) -> None:
        crew = real_crew(crewai_mod, fake_llm_cls, n_tasks=2)
        rec1, rec2 = make_recorder("r1"), make_recorder("r2")

        VCRCrewAI(rec1).kickoff(crew)
        names1 = [f.node_name for f in rec1.frames]
        assert names1.count("agent_finish") == 2
        assert "task:say 42 #0" in names1 and "task:say 42 #1" in names1
        assert len(names1) == 4
        assert_clean(crew)

        VCRCrewAI(rec2).kickoff(crew)
        assert len(rec1.frames) == 4
        assert len(rec2.frames) == 4
        assert_clean(crew)

        crew.kickoff()  # plain kickoff must record nothing anywhere
        assert len(rec1.frames) == 4 and len(rec2.frames) == 4

        finish = next(f for f in rec1.frames if f.node_name == "agent_finish")
        assert finish.input_state["agent_role"] == "researcher"
        assert "42" in str(finish.output_state["output"])
        task_frame = next(f for f in rec1.frames if f.node_name == "task:say 42 #0")
        assert task_frame.input_state["description"] == "say 42 #0"

    def test_user_callbacks_still_fire(self, crewai_mod: Any, fake_llm_cls: Any) -> None:
        crew = real_crew(crewai_mod, fake_llm_cls)
        crew_steps, crew_step_cb = counter()
        crew_tasks, crew_task_cb = counter()
        own_tasks, own_task_cb = counter()
        crew.step_callback = crew_step_cb
        crew.task_callback = crew_task_cb
        crew.tasks[0].callback = own_task_cb

        rec = make_recorder("user_cb")
        VCRCrewAI(rec).kickoff(crew)

        assert len(crew_steps) == 1
        assert len(crew_tasks) == 1
        assert len(own_tasks) == 1
        assert [f.node_name for f in rec.frames] == ["agent_finish", "task:say 42 #0"]
        assert crew.step_callback is crew_step_cb
        assert crew.task_callback is crew_task_cb
        assert crew.tasks[0].callback is own_task_cb
        assert crew.agents[0].step_callback is None

    def test_agent_with_own_step_callback_is_recorded(
        self, crewai_mod: Any, fake_llm_cls: Any
    ) -> None:
        crew = real_crew(crewai_mod, fake_llm_cls)
        own_steps, own_step_cb = counter()
        crew.agents[0].step_callback = own_step_cb

        rec = make_recorder("own_step")
        VCRCrewAI(rec).kickoff(crew)

        assert len(own_steps) == 1
        assert "agent_finish" in [f.node_name for f in rec.frames]
        assert crew.agents[0].step_callback is own_step_cb

    def test_error_restores_callbacks(self, crewai_mod: Any, fake_llm_cls: Any) -> None:
        crew = real_crew(crewai_mod, fake_llm_cls, fail=True)
        rec = make_recorder("err")
        with pytest.raises(Exception, match="llm exploded"):
            VCRCrewAI(rec).kickoff(crew)
        assert any(f.frame_type == FrameType.ERROR for f in rec.frames)
        assert_clean(crew)

    def test_kickoff_async(self, crewai_mod: Any, fake_llm_cls: Any) -> None:
        crew = real_crew(crewai_mod, fake_llm_cls)
        rec = make_recorder("async")
        asyncio.run(VCRCrewAI(rec).kickoff_async(crew, timeout=120))
        assert [f.node_name for f in rec.frames] == ["agent_finish", "task:say 42 #0"]
        assert_clean(crew)

    def test_kickoff_for_each(self, crewai_mod: Any, fake_llm_cls: Any) -> None:
        crew = real_crew(crewai_mod, fake_llm_cls)
        rec = make_recorder("each")
        results = VCRCrewAI(rec).kickoff_for_each(crew, [{"x": "1"}, {"x": "2"}])
        assert len(results) == 2
        names = [f.node_name for f in rec.frames]
        assert names.count("agent_finish") == 2
        assert names.count("task:say 42 #0") == 2
        assert_clean(crew)

    def test_tool_step_recorded(self, crewai_mod: Any, fake_llm_cls: Any) -> None:
        from crewai.tools import tool

        @tool("lookup")
        def lookup(query: str) -> str:
            """Look something up."""
            return "the answer is 42"

        script = [
            'Thought: use the tool\nAction: lookup\nAction Input: {"query": "q"}',
            "Thought: done\nFinal Answer: 42",
        ]
        crew = real_crew(crewai_mod, fake_llm_cls, script=script, tools=[lookup])
        rec = make_recorder("tool")
        with VCRCrewCallback(rec, record_llm_calls=False):
            VCRCrewAI(rec).kickoff(crew)
        names = [f.node_name for f in rec.frames]
        assert "agent_action::lookup" in names
        assert "agent_finish" in names
        action = next(f for f in rec.frames if f.node_name == "agent_action::lookup")
        assert "42" in str(action.output_state)
        tool_frames = [f for f in rec.frames if f.frame_type == FrameType.TOOL_CALL]
        assert len(tool_frames) == 1
        assert_clean(crew)

    def test_wrap_task_real(self, crewai_mod: Any, fake_llm_cls: Any) -> None:
        crew = real_crew(crewai_mod, fake_llm_cls)
        rec = make_recorder("wrap")
        VCRCrewAI(rec).wrap_task(crew.tasks[0])  # used to raise ValueError
        crew.kickoff()
        assert [f.node_name for f in rec.frames] == ["task:say 42 #0"]


class TestRealEventBus:
    @staticmethod
    def _handler_count() -> int:
        from crewai.events import crewai_event_bus

        return sum(len(v) for v in crewai_event_bus._handlers.values())

    def test_listener_records_and_detaches(self, crewai_mod: Any, fake_llm_cls: Any) -> None:
        crew = real_crew(crewai_mod, fake_llm_cls)
        rec = make_recorder("bus")
        before = self._handler_count()

        with VCRCrewCallback(rec) as listener:
            assert listener.attached
            assert self._handler_count() > before
            crew.kickoff()

        assert self._handler_count() == before
        names = [f.node_name for f in rec.frames]
        assert "task:say 42 #0" in names
        assert "agent:researcher" in names
        assert "llm:fake-llm" in names
        task_frame = next(f for f in rec.frames if f.node_name == "task:say 42 #0")
        assert task_frame.input_state["agent_role"] == "researcher"
        assert task_frame.input_state["description"] == "say 42 #0"
        assert any(f.frame_type == FrameType.LLM_CALL for f in rec.frames)

        n = len(rec.frames)
        crew.kickoff()  # detached -> nothing recorded
        assert len(rec.frames) == n

    def test_listeners_are_scoped_to_their_recorder(
        self, crewai_mod: Any, fake_llm_cls: Any
    ) -> None:
        crew = real_crew(crewai_mod, fake_llm_cls)
        rec1, rec2 = make_recorder("l1"), make_recorder("l2")
        with VCRCrewCallback(rec1, record_llm_calls=False):
            crew.kickoff()
        with VCRCrewCallback(rec2, record_llm_calls=False):
            crew.kickoff()
        assert len(rec1.frames) == len(rec2.frames) == 2

    def test_crew_callbacks_misuse_is_loud(self, crewai_mod: Any, fake_llm_cls: Any) -> None:
        rec = make_recorder("misuse")
        agent = real_crew(crewai_mod, fake_llm_cls).agents[0]
        task = crewai_mod.Task(description="t", expected_output="x", agent=agent)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            crewai_mod.Crew(agents=[agent], tasks=[task], callbacks=[VCRCrewCallback(rec)])
            gc.collect()
        assert any("never attached" in str(w.message) for w in caught)
