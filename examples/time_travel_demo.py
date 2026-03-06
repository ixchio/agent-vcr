"""Demonstration of the time-travel debugging feature."""

from agent_vcr import VCRPlayer, VCRRecorder
from agent_vcr.models import ResumeConfig


class CalculatorAgent:
    """A multi-step calculator agent."""

    def run(self, state: dict) -> dict:
        """Execute the next step based on state."""
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
        return state

    def run_full(self, state: dict) -> dict:
        """Run until completion."""
        while state.get("step") != "done":
            state = self.run(state)
        return state


def main():
    print("=" * 60)
    print("Agent VCR Time-Travel Demo")
    print("=" * 60)

    # Step 1: Record original execution
    print("\n[1] Recording original execution...")
    recorder = VCRRecorder()
    recorder.start_session("original_run")

    agent = CalculatorAgent()
    state = {"a": 2, "b": 3, "c": 4, "step": "add"}

    # Execute and record
    for step_name in ["add", "multiply"]:
        new_state = agent.run(state)
        recorder.record_step(
            node_name=step_name,
            input_state=state,
            output_state=new_state,
        )
        state = new_state

    vcr_path = recorder.save()
    print(f"    Saved to: {vcr_path}")

    # Step 2: Playback and verify
    print("\n[2] Playing back recording...")
    player = VCRPlayer.load(vcr_path)
    print(f"    Frames: {len(player.frames)}")

    final_state = player.goto_frame(1)
    print(f"    Original result: {final_state['final_result']}")  # (2+3)*4 = 20

    # Step 3: Time travel - inspect intermediate state
    print("\n[3] Time travel: Inspecting intermediate state...")
    intermediate = player.goto_frame(0)
    print(f"    After 'add' step: {intermediate}")

    # Step 4: Edit state and resume (THE KILLER FEATURE)
    print("\n[4] Time travel: Editing state and resuming...")
    print("    Changing 'b' from 3 to 5...")
    print("    Expected new result: (2+5)*4 = 28")

    # Create a new recorder for the forked session
    fork_recorder = VCRRecorder()

    new_session_id = player.resume(
        agent_callable=agent.run_full,
        config=ResumeConfig(
            from_frame=0,
            state_overrides={"b": 5},
        ),
        recorder=fork_recorder,
    )

    print(f"    New session ID: {new_session_id}")

    # Step 5: Verify the forked execution
    print("\n[5] Verifying forked execution...")
    forked_player = VCRPlayer.load_by_id(new_session_id)
    forked_final = forked_player.goto_frame(1)
    print(f"    Forked result: {forked_final['final_result']}")

    # Step 6: Show the comparison
    print("\n[6] Comparing executions...")
    print(f"    Original: (2+3)*4 = {final_state['final_result']}")
    print(f"    Forked:   (2+5)*4 = {forked_final['final_result']}")

    # Show fork relationship
    print("\n[7] Fork relationship:")
    print(f"    Parent: {forked_player.session.parent_session_id}")
    print(f"    Forked from frame: {forked_player.session.forked_from_frame}")

    print("\n" + "=" * 60)
    print("Demo complete! Check the .vcr directory for recorded sessions.")
    print("=" * 60)


if __name__ == "__main__":
    main()
