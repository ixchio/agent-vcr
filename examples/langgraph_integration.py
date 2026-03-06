"""Example of LangGraph integration with Agent VCR."""

from typing import TypedDict

from agent_vcr import VCRRecorder
from agent_vcr.integrations.langgraph import VCRLangGraph


def main():
    """Demonstrate LangGraph auto-instrumentation."""
    try:
        from langgraph.graph import StateGraph
    except ImportError:
        print("This example requires langgraph. Install with: pip install langgraph")
        return

    print("=" * 60)
    print("Agent VCR + LangGraph Integration Demo")
    print("=" * 60)

    # Define state
    class AgentState(TypedDict):
        query: str
        plan: str
        code: str
        result: str

    # Define nodes
    def planner(state: AgentState) -> AgentState:
        """Plan the solution."""
        return {
            **state,
            "plan": f"Plan for: {state['query']}",
        }

    def coder(state: AgentState) -> AgentState:
        """Write code based on plan."""
        return {
            **state,
            "code": f"# Code for: {state['plan']}",
        }

    def tester(state: AgentState) -> AgentState:
        """Test the code."""
        return {
            **state,
            "result": "Tests passed!",
        }

    # Build graph
    print("\n[1] Building LangGraph...")
    graph = StateGraph(AgentState)
    graph.add_node("planner", planner)
    graph.add_node("coder", coder)
    graph.add_node("tester", tester)
    graph.add_edge("planner", "coder")
    graph.add_edge("coder", "tester")
    graph.set_entry_point("planner")
    graph.set_finish_point("tester")

    # Create VCR recorder
    print("[2] Setting up VCR recording...")
    recorder = VCRRecorder()
    recorder.start_session("langgraph_demo")

    # Wrap graph with VCR
    print("[3] Wrapping graph with VCR instrumentation...")
    vcr_wrapper = VCRLangGraph(recorder)
    instrumented_graph = vcr_wrapper.wrap_graph(graph)

    # Compile and run
    print("[4] Running instrumented graph...")
    compiled = instrumented_graph.compile()
    result = compiled.invoke({"query": "Build a todo app"})

    print(f"    Result: {result}")

    # Save recording
    vcr_path = recorder.save()
    print(f"\n[5] Recording saved to: {vcr_path}")

    # Playback
    print("\n[6] Playing back recording...")
    from agent_vcr import VCRPlayer

    player = VCRPlayer.load(vcr_path)
    print(f"    Frames recorded: {len(player.frames)}")
    print(f"    Nodes executed: {player.list_nodes()}")
    print(f"    Total latency: {player.get_total_latency():.2f}ms")

    # Show each frame
    print("\n[7] Execution timeline:")
    for i, frame in enumerate(player.frames):
        print(f"    Frame {i}: {frame.node_name} ({frame.metadata.latency_ms:.2f}ms)")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
