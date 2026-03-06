"""Async usage example for Agent VCR."""

import asyncio

from agent_vcr.async_recorder import AsyncVCRRecorder
from agent_vcr.async_player import AsyncVCRPlayer
from agent_vcr.models import FrameMetadata


async def simulate_agent(state: dict) -> dict:
    """Simulate an async agent processing step."""
    await asyncio.sleep(0.01)  # Simulate async work
    return {**state, "processed": True, "result": state.get("value", 0) * 2}


async def main():
    # Record an async agent execution
    recorder = AsyncVCRRecorder(output_dir="/tmp/async_vcr_demo", auto_save=False)
    session = await recorder.start_session(
        session_id="async-demo",
        metadata={"agent": "async_example"},
        tags=["demo", "async"],
    )

    print(f"📹 Recording session: {session.session_id}")

    # Step 1: Initialize
    state = {"value": 5, "step": 0}
    await recorder.record_step("init", {}, state)

    # Step 2: Process with simulated async agent
    result = await simulate_agent(state)
    await recorder.record_step(
        "process",
        state,
        result,
        metadata=FrameMetadata(latency_ms=10),
    )

    # Step 3: Record an LLM call
    await recorder.record_llm_call(
        model="gpt-4",
        messages=[{"role": "user", "content": "Summarize the result"}],
        response="The value was doubled to 10.",
        tokens_input=15,
        tokens_output=8,
        latency_ms=250,
    )

    # Step 4: Record a tool call
    await recorder.record_tool_call(
        tool_name="calculator",
        tool_input={"expression": "5 * 2"},
        tool_output=10,
        latency_ms=5,
    )

    # Save the session
    path = await recorder.save()
    print(f"💾 Saved to: {path}")

    # Load and replay
    player = await AsyncVCRPlayer.load(path)
    print(f"\n⏮️  Loaded session with {len(player.frames)} frames")

    # Navigate frames
    for i, frame in enumerate(player.frames):
        state = player.goto_frame(i)
        print(f"  Frame {i}: {frame.node_name} → {state}")

    # Time travel to a specific frame
    print(f"\n🕐 Nodes: {player.list_nodes()}")
    print(f"📊 Total latency: {player.get_total_latency():.1f}ms")
    print(f"🔤 Total tokens: {player.get_total_tokens()}")
    print(f"💰 Total cost: ${player.get_total_cost():.4f}")


if __name__ == "__main__":
    asyncio.run(main())
