"""Terminal UI for Agent VCR - Replay agent execution locally."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rich.console import RenderableType
from rich.panel import Panel
from rich.syntax import Syntax
from rich.text import Text
from textual.app import App, ComposeResult
from textual.containers import Horizontal
from textual.reactive import reactive
from textual.widgets import Footer, Header, Static

from agent_vcr.async_player import AsyncVCRPlayer
from agent_vcr.models import Frame, FrameType


class FrameList(Static):
    """A widget displaying the list of frames in the timeline."""

    frames: reactive[list[Frame]] = reactive([])
    selected_index: reactive[int] = reactive(0)

    def render(self) -> RenderableType:
        if not self.frames:
            return Text("Loading frames...", style="dim")

        lines = []
        for i, frame in enumerate(self.frames):
            prefix = "▶ " if i == self.selected_index else "  "

            # Color code based on type
            color = "white"
            if frame.frame_type == FrameType.NODE_EXECUTION:
                color = "green"
            elif frame.frame_type == FrameType.TOOL_CALL:
                color = "yellow"
            elif frame.frame_type == FrameType.LLM_CALL:
                color = "blue"
            elif frame.frame_type == FrameType.ERROR:
                color = "red"

            node_name = frame.node_name
            latency = f"[{frame.metadata.latency_ms:.0f}ms]" if frame.metadata.latency_ms else ""

            line = Text(f"{prefix}[{i:03d}] ", style="dim")
            line.append(f"{node_name:<20}", style=color)
            if latency:
                line.append(f" {latency:>8}", style="dim " + color)

            lines.append(line)

        return Panel(
            Text("\n").join(lines),
            title="Timeline",
            border_style="blue",
        )


class StateViewer(Static):
    """A widget to display the selected frame's state or diff."""

    frame: reactive[Frame | None] = reactive(None)
    view_mode: reactive[str] = reactive("state")  # 'state', 'input', 'diff'

    def render(self) -> RenderableType:
        if not self.frame:
            return Panel("Select a frame", title="State", border_style="cyan")

        if self.view_mode == "state":
            content = json.dumps(self.frame.output_state, indent=2)
            title = "Output State"
        elif self.view_mode == "input":
            content = json.dumps(self.frame.input_state, indent=2)
            title = "Input State"
        elif self.view_mode == "diff":
            if self.frame.state_diff is not None:
                content = json.dumps(self.frame.state_diff, indent=2)
            else:
                content = "No diff available (First frame or diff_mode disabled)"
            title = "State Diff"
        else:
            content = "{}"
            title = "Unknown"

        syntax = Syntax(content, "json", theme="monokai", padding=1, word_wrap=True)
        return Panel(syntax, title=title, border_style="cyan")


class VCRApp(App):
    """The main VCR Terminal User Interface."""

    CSS = """
    #timeline-panel {
        width: 40;
        height: 100%;
        border-right: solid green;
    }
    #state-panel {
        width: 1fr;
        height: 100%;
    }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("up", "prev_frame", "Previous Frame"),
        ("k", "prev_frame", "Previous Frame"),
        ("down", "next_frame", "Next Frame"),
        ("j", "next_frame", "Next Frame"),
        ("1", "view_input", "Input State"),
        ("2", "view_output", "Output State"),
        ("3", "view_diff", "Diff"),
    ]

    def __init__(self, filepath: str):
        super().__init__()
        self.filepath = filepath
        self.player: AsyncVCRPlayer | None = None
        self.current_index = 0

    async def on_mount(self) -> None:
        """Load the VCR file when the app starts."""
        try:
            self.player = await AsyncVCRPlayer.load(self.filepath)
            self.title = f"Agent VCR - {self.player.session.session_id}"

            timeline = self.query_one(FrameList)
            timeline.frames = self.player.frames
            self.update_selection(0)
        except Exception as e:
            self.exit(f"Error loading VCR file: {e}")

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            yield FrameList(id="timeline-panel")
            yield StateViewer(id="state-panel")
        yield Footer()

    def update_selection(self, index: int) -> None:
        if not self.player or not self.player.frames:
            return

        index = max(0, min(index, len(self.player.frames) - 1))
        self.current_index = index

        timeline = self.query_one(FrameList)
        timeline.selected_index = index

        viewer = self.query_one(StateViewer)
        viewer.frame = self.player.frames[index]

    def action_prev_frame(self) -> None:
        """Go to the previous frame."""
        self.update_selection(self.current_index - 1)

    def action_next_frame(self) -> None:
        """Go to the next frame."""
        self.update_selection(self.current_index + 1)

    def action_view_input(self) -> None:
        self.query_one(StateViewer).view_mode = "input"

    def action_view_output(self) -> None:
        self.query_one(StateViewer).view_mode = "state"

    def action_view_diff(self) -> None:
        self.query_one(StateViewer).view_mode = "diff"


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent VCR Terminal UI")
    parser.add_argument("file", help="Path to the .vcr file to replay")
    args = parser.parse_args()

    filepath = Path(args.file)
    if not filepath.exists():
        print(f"Error: File not found: {filepath}", file=sys.stderr)
        sys.exit(1)

    app = VCRApp(str(filepath))
    app.run()


if __name__ == "__main__":
    main()
