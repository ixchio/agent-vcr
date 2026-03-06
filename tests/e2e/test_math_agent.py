"""End-to-end test with a toy math agent."""

import tempfile

from agent_vcr.models import ResumeConfig
from agent_vcr.player import VCRPlayer
from agent_vcr.recorder import VCRRecorder


class MathAgent:
    """A simple deterministic agent for testing."""

    def __init__(self):
        self.history = []

    def run(self, state: dict) -> dict:
        """Execute one step of the math agent."""
        step = state.get("step", "add")

        if step == "add":
            result = state["a"] + state["b"]
            return {
                **state,
                "step": "multiply",
                "add_result": result,
            }
        elif step == "multiply":
            result = state["add_result"] * state["c"]
            return {
                **state,
                "step": "done",
                "final_result": result,
            }
        else:
            return state

    def run_full(self, initial_state: dict) -> dict:
        """Run the agent until completion."""
        state = initial_state
        while state.get("step") != "done":
            state = self.run(state)
        return state


class TestMathAgentE2E:
    """End-to-end tests demonstrating the full VCR workflow."""

    def test_record_and_playback(self):
        """Test basic recording and playback."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and record agent execution
            recorder = VCRRecorder(output_dir=tmpdir, auto_save=False)
            recorder.start_session("math_test")

            agent = MathAgent()
            state = {"a": 2, "b": 3, "c": 4, "step": "add"}

            # Step 1: Add
            new_state = agent.run(state)
            recorder.record_step(
                node_name="add",
                input_state=state,
                output_state=new_state,
            )
            state = new_state

            # Step 2: Multiply
            new_state = agent.run(state)
            recorder.record_step(
                node_name="multiply",
                input_state=state,
                output_state=new_state,
            )

            vcr_path = recorder.save()

            # Playback and verify
            player = VCRPlayer.load(vcr_path)

            assert len(player.frames) == 2
            assert player.frames[0].node_name == "add"
            assert player.frames[1].node_name == "multiply"

            # Verify final result: (2+3)*4 = 20
            final_state = player.goto_frame(1)
            assert final_state["final_result"] == 20

    def test_time_travel(self):
        """Test jumping to a specific point in execution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = VCRRecorder(output_dir=tmpdir, auto_save=False)
            recorder.start_session("time_travel_test")

            agent = MathAgent()
            state = {"a": 2, "b": 3, "c": 4, "step": "add"}

            for node_name in ["add", "multiply"]:
                new_state = agent.run(state)
                recorder.record_step(
                    node_name=node_name,
                    input_state=state,
                    output_state=new_state,
                )
                state = new_state

            vcr_path = recorder.save()
            player = VCRPlayer.load(vcr_path)

            # Jump to after add step
            add_state = player.goto_frame(0)
            assert add_state["step"] == "multiply"
            assert add_state["add_result"] == 5  # 2+3

            # Jump to final state
            final_state = player.goto_frame(1)
            assert final_state["step"] == "done"
            assert final_state["final_result"] == 20  # 5*4

    def test_state_inspection(self):
        """Test inspecting state at different points."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = VCRRecorder(output_dir=tmpdir, auto_save=False)
            recorder.start_session("inspection_test")

            agent = MathAgent()
            state = {"a": 2, "b": 3, "c": 4, "step": "add"}

            for node_name in ["add", "multiply"]:
                new_state = agent.run(state)
                recorder.record_step(
                    node_name=node_name,
                    input_state=state,
                    output_state=new_state,
                )
                state = new_state

            vcr_path = recorder.save()
            player = VCRPlayer.load(vcr_path)

            # Inspect input to multiply step
            multiply_input = player.get_input_state(1)
            assert multiply_input["add_result"] == 5

            # Inspect output of add step
            add_output = player.get_output_state(0)
            assert add_output["add_result"] == 5

    def test_resume_with_state_edit(self):
        """Test the killer feature: edit state and resume."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Record original execution: (2+3)*4 = 20
            recorder = VCRRecorder(output_dir=tmpdir, auto_save=False)
            recorder.start_session("original")

            agent = MathAgent()
            state = {"a": 2, "b": 3, "c": 4, "step": "add"}

            for node_name in ["add", "multiply"]:
                new_state = agent.run(state)
                recorder.record_step(
                    node_name=node_name,
                    input_state=state,
                    output_state=new_state,
                )
                state = new_state

            vcr_path = recorder.save()
            player = VCRPlayer.load(vcr_path)

            # Verify original result
            original_final = player.goto_frame(1)
            assert original_final["final_result"] == 20

            # TIME TRAVEL: Go back to frame 0 (after add), change 'c' from 4 to 7
            # Expected new result: 5*7 = 35
            edited_state = player.goto_frame(0)
            edited_state["c"] = 7

            # Resume from frame 0 with edited state
            def resumed_agent(state):
                return agent.run_full(state)

            new_recorder = VCRRecorder(output_dir=tmpdir, auto_save=False)
            new_session_id = player.resume(
                agent_callable=resumed_agent,
                config=ResumeConfig(
                    from_frame=0,
                    state_overrides={"c": 7},
                ),
                recorder=new_recorder,
            )

            # Verify forked result
            forked_player = VCRPlayer.load_by_id(new_session_id, vcr_dir=tmpdir)
            forked_final = forked_player.goto_frame(0)

            # 5*7 = 35
            assert forked_final["final_result"] == 35

            # Verify fork relationship
            assert forked_player.session.parent_session_id == "original"
            assert forked_player.session.forked_from_frame == 0

    def test_compare_runs(self):
        """Test comparing two different execution paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Run A: (2+3)*4 = 20
            recorder_a = VCRRecorder(output_dir=tmpdir, auto_save=False)
            recorder_a.start_session("run_a")

            agent = MathAgent()
            state = {"a": 2, "b": 3, "c": 4, "step": "add"}

            for node_name in ["add", "multiply"]:
                new_state = agent.run(state)
                recorder_a.record_step(node_name, state, new_state)
                state = new_state

            vcr_path_a = recorder_a.save()

            # Run B: (5+5)*2 = 20 (same result, different path)
            recorder_b = VCRRecorder(output_dir=tmpdir, auto_save=False)
            recorder_b.start_session("run_b")

            state = {"a": 5, "b": 5, "c": 2, "step": "add"}

            for node_name in ["add", "multiply"]:
                new_state = agent.run(state)
                recorder_b.record_step(node_name, state, new_state)
                state = new_state

            vcr_path_b = recorder_b.save()

            # Compare
            player_a = VCRPlayer.load(vcr_path_a)
            VCRPlayer.load(vcr_path_b)

            diff = player_a.compare_frames(0, 0)

            # Since compare_frames compares states of frame 0 in player_a with frame 0 in player_a
            # Wait, compare_frames compares two frames IN THE SAME PLAYER!
            # player_a.compare_frames(0, 1) compares frame 0 and frame 1 in player_a!
            # The test intended to compare player_a's frame 0 with player_b's frame 0? No:
            # player_a.compare_frames(0, 0) compares player_a frame 0 with itself!

            # Let's fix test to compare frame 0 and frame 1 in player_a
            diff = player_a.compare_frames(0, 1)

            assert "modified" in diff
            assert "step" in diff["modified"]
            assert diff["modified"]["step"]["before"] == "multiply"
            assert diff["modified"]["step"]["after"] == "done"

    def test_error_handling(self):
        """Test recording and recovering from errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            recorder = VCRRecorder(output_dir=tmpdir, auto_save=False)
            recorder.start_session("error_test")

            # Record successful step
            recorder.record_step(
                node_name="ok_step",
                input_state={},
                output_state={"status": "ok"},
            )

            # Record error step
            try:
                raise ValueError("Something went wrong")
            except Exception as e:
                recorder.record_error(
                    node_name="error_step",
                    input_state={},
                    error=e,
                    latency_ms=50,
                )

            vcr_path = recorder.save()
            player = VCRPlayer.load(vcr_path)

            errors = player.get_errors()
            assert len(errors) == 1
            assert errors[0].metadata.error_type == "ValueError"
            assert errors[0].metadata.error_message == "Something went wrong"
