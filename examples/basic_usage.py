"""Basic usage example for Agent VCR."""

from agent_vcr import VCRPlayer, VCRRecorder


def simple_agent(state: dict) -> dict:
    """A simple agent that adds two numbers."""
    a = state.get("a", 0)
    b = state.get("b", 0)
    return {**state, "result": a + b}


def main():
    # Create a recorder
    recorder = VCRRecorder()

    # Start a session
    session = recorder.start_session("my_first_session")
    print(f"Started session: {session.session_id}")

    # Record some steps
    state = {"a": 5, "b": 3}

    result = simple_agent(state)
    frame = recorder.record_step(
        node_name="addition",
        input_state=state,
        output_state=result,
    )
    print(f"Recorded frame: {frame.frame_id}")

    # Save the session
    path = recorder.save()
    print(f"Saved to: {path}")

    # Load and playback
    player = VCRPlayer.load(path)
    print(f"\nLoaded session with {len(player.frames)} frames")

    # Time travel: go to frame 0
    state_at_frame = player.goto_frame(0)
    print(f"State at frame 0: {state_at_frame}")

    # Get statistics
    print("\nSession Statistics:")
    print(f"  Total latency: {player.get_total_latency():.2f}ms")
    print(f"  Nodes executed: {player.list_nodes()}")


if __name__ == "__main__":
    main()
