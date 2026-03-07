"""Terminal UI for Agent VCR - Replay agent execution locally.

Features:
  - Arrow keys / j/k to step through frames
  - State diffs highlighted in green/red
  - Press 'e' to edit state inline
  - Press 'r' to resume from current/edited state
  - Press 's' to search/filter frames by name
"""

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
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.reactive import reactive
from textual.screen import ModalScreen
from textual.widgets import Footer, Header, Static, TextArea, Input, Button

from agent_vcr.async_player import AsyncVCRPlayer
from agent_vcr.models import Frame, FrameType


# ---------------------------------------------------------------------------
# Widgets
# ---------------------------------------------------------------------------


class FrameList(Static):
    """A widget displaying the list of frames in the timeline."""

    frames: reactive[list[Frame]] = reactive([])
    selected_index: reactive[int] = reactive(0)
    filter_text: reactive[str] = reactive("")

    @property
    def visible_frames(self) -> list[tuple[int, Frame]]:
        """Return (original_index, frame) pairs matching the filter."""
        if not self.filter_text:
            return list(enumerate(self.frames))
        needle = self.filter_text.lower()
        return [
            (i, f)
            for i, f in enumerate(self.frames)
            if needle in f.node_name.lower()
        ]

    def render(self) -> RenderableType:
        if not self.frames:
            return Text("Loading frames...", style="dim")

        visible = self.visible_frames
        if not visible:
            return Panel(
                Text("No frames match filter", style="dim italic"),
                title="Timeline",
                border_style="blue",
            )

        lines = []
        for orig_idx, frame in visible:
            prefix = "▶ " if orig_idx == self.selected_index else "  "

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
            latency = (
                f"[{frame.metadata.latency_ms:.0f}ms]"
                if frame.metadata.latency_ms
                else ""
            )

            line = Text(f"{prefix}[{orig_idx:03d}] ", style="dim")
            line.append(f"{node_name:<20}", style=color)
            if latency:
                line.append(f" {latency:>8}", style="dim " + color)

            lines.append(line)

        title = "Timeline"
        if self.filter_text:
            title += f"  🔍 {self.filter_text}"

        return Panel(
            Text("\n").join(lines),
            title=title,
            border_style="blue",
        )


class StateViewer(Static):
    """A widget to display the selected frame's state or diff."""

    frame: reactive[Frame | None] = reactive(None)
    prev_frame: reactive[Frame | None] = reactive(None)
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
            content = self._render_diff()
            title = "State Diff"
            # Return Rich Text for coloured diff
            return Panel(content, title=title, border_style="cyan")
        else:
            content = "{}"
            title = "Unknown"

        syntax = Syntax(content, "json", theme="monokai", padding=1, word_wrap=True)
        return Panel(syntax, title=title, border_style="cyan")

    def _render_diff(self) -> Text:
        """Render a coloured diff between previous and current frame."""
        cur = self.frame
        if cur is None:
            return Text("No frame selected", style="dim")

        # Use state_diff if available
        if cur.state_diff is not None:
            return self._colorize_diff_ops(cur.state_diff)

        # Otherwise compute diff from previous frame
        prev = self.prev_frame
        if prev is None:
            return Text("No previous frame (first frame)", style="dim")

        diff = self._compute_diff(prev.output_state, cur.output_state)
        return self._colorize_dict_diff(diff)

    def _compute_diff(self, before: dict, after: dict) -> dict:
        diff: dict = {"added": {}, "removed": {}, "modified": {}}
        all_keys = set(before.keys()) | set(after.keys())
        for key in sorted(all_keys):
            if key not in before:
                diff["added"][key] = after[key]
            elif key not in after:
                diff["removed"][key] = before[key]
            elif before[key] != after[key]:
                diff["modified"][key] = {"before": before[key], "after": after[key]}
        return diff

    def _colorize_dict_diff(self, diff: dict) -> Text:
        text = Text()
        for key, val in diff.get("added", {}).items():
            text.append(f"+ {key}: {json.dumps(val)}\n", style="bold green")
        for key, val in diff.get("removed", {}).items():
            text.append(f"- {key}: {json.dumps(val)}\n", style="bold red")
        for key, val in diff.get("modified", {}).items():
            text.append(f"~ {key}:\n", style="bold yellow")
            text.append(f"    before: {json.dumps(val['before'])}\n", style="red")
            text.append(f"    after:  {json.dumps(val['after'])}\n", style="green")
        if not text.plain.strip():
            text.append("No changes", style="dim")
        return text

    def _colorize_diff_ops(self, ops: list[dict]) -> Text:
        text = Text()
        for op in ops:
            kind = op.get("op", "?")
            path = op.get("path", "?")
            if kind == "add":
                text.append(f"+ {path}: {json.dumps(op.get('value'))}\n", style="bold green")
            elif kind == "remove":
                text.append(f"- {path}\n", style="bold red")
            elif kind == "replace":
                text.append(f"~ {path}:\n", style="bold yellow")
                text.append(f"    → {json.dumps(op.get('value'))}\n", style="green")
        if not text.plain.strip():
            text.append("No changes", style="dim")
        return text


# ---------------------------------------------------------------------------
# Modal screens
# ---------------------------------------------------------------------------


class EditStateScreen(ModalScreen[str | None]):
    """Modal screen for editing a frame's state JSON."""

    CSS = """
    EditStateScreen {
        align: center middle;
    }
    #edit-container {
        width: 80%;
        height: 80%;
        background: $surface;
        border: tall $primary;
        padding: 1;
    }
    #edit-title {
        text-align: center;
        margin-bottom: 1;
    }
    #edit-area {
        height: 1fr;
    }
    #edit-buttons {
        height: 3;
        align: center middle;
    }
    """

    def __init__(self, initial_json: str, frame_index: int) -> None:
        super().__init__()
        self._initial_json = initial_json
        self._frame_index = frame_index

    def compose(self) -> ComposeResult:
        with Vertical(id="edit-container"):
            yield Static(
                f"[bold cyan]Edit State — Frame {self._frame_index}[/]  "
                "[dim](Save: Ctrl+S  |  Cancel: Escape)[/]",
                id="edit-title",
            )
            yield TextArea(self._initial_json, language="json", id="edit-area")
            with Horizontal(id="edit-buttons"):
                yield Button("💾 Save", variant="success", id="save-btn")
                yield Button("Cancel", variant="error", id="cancel-btn")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "save-btn":
            area = self.query_one(TextArea)
            self.dismiss(area.text)
        else:
            self.dismiss(None)

    def key_escape(self) -> None:
        self.dismiss(None)


class SearchScreen(ModalScreen[str]):
    """Modal screen for searching/filtering frames."""

    CSS = """
    SearchScreen {
        align: center middle;
    }
    #search-container {
        width: 60%;
        height: auto;
        max-height: 10;
        background: $surface;
        border: tall $primary;
        padding: 1;
    }
    """

    def __init__(self, current_filter: str = "") -> None:
        super().__init__()
        self._current_filter = current_filter

    def compose(self) -> ComposeResult:
        with Vertical(id="search-container"):
            yield Static("[bold cyan]Search Frames[/]  [dim](Enter to apply, Esc to clear)[/]")
            yield Input(
                value=self._current_filter,
                placeholder="Filter by node name...",
                id="search-input",
            )

    def on_input_submitted(self, event: Input.Submitted) -> None:
        self.dismiss(event.value)

    def key_escape(self) -> None:
        self.dismiss("")


# ---------------------------------------------------------------------------
# Main App
# ---------------------------------------------------------------------------


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
    #status-bar {
        height: 1;
        dock: bottom;
        background: $primary-background;
        padding: 0 1;
    }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("up", "prev_frame", "Prev Frame", show=False),
        Binding("k", "prev_frame", "Prev Frame", show=False),
        Binding("down", "next_frame", "Next Frame", show=False),
        Binding("j", "next_frame", "Next Frame", show=False),
        Binding("1", "view_input", "Input"),
        Binding("2", "view_output", "Output"),
        Binding("3", "view_diff", "Diff"),
        Binding("e", "edit_state", "Edit"),
        Binding("r", "resume", "Resume"),
        Binding("s", "search", "Search"),
    ]

    def __init__(self, filepath: str):
        super().__init__()
        self.filepath = filepath
        self.player: AsyncVCRPlayer | None = None
        self.current_index = 0
        self._edited_state: dict | None = None

    async def on_mount(self) -> None:
        """Load the VCR file when the app starts."""
        try:
            self.player = await AsyncVCRPlayer.load(self.filepath)
            self.title = f"Agent VCR - {self.player.session.session_id}"

            timeline = self.query_one(FrameList)
            timeline.frames = self.player.frames
            self.update_selection(0)
            self._update_status()
        except Exception as e:
            self.exit(f"Error loading VCR file: {e}")

    def compose(self) -> ComposeResult:
        yield Header()
        with Horizontal():
            yield FrameList(id="timeline-panel")
            yield StateViewer(id="state-panel")
        yield Static("", id="status-bar")
        yield Footer()

    def update_selection(self, index: int) -> None:
        if not self.player or not self.player.frames:
            return

        index = max(0, min(index, len(self.player.frames) - 1))
        self.current_index = index
        self._edited_state = None  # reset on navigation

        timeline = self.query_one(FrameList)
        timeline.selected_index = index

        viewer = self.query_one(StateViewer)
        viewer.frame = self.player.frames[index]
        viewer.prev_frame = self.player.frames[index - 1] if index > 0 else None

        self._update_status()

    def _update_status(self) -> None:
        """Update the status bar with current position info."""
        status = self.query_one("#status-bar", Static)
        if self.player and self.player.frames:
            frame = self.player.frames[self.current_index]
            total = len(self.player.frames)
            edited = " [yellow]✏ EDITED[/]" if self._edited_state else ""
            status.update(
                f"[dim]Frame[/] [bold]{self.current_index + 1}/{total}[/]  "
                f"[dim]Node:[/] [cyan]{frame.node_name}[/]  "
                f"[dim]Type:[/] {frame.frame_type.value}  "
                f"[dim]Latency:[/] {frame.metadata.latency_ms:.0f}ms"
                f"{edited}"
            )

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------

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

    def action_edit_state(self) -> None:
        """Open inline editor for the current frame's output state."""
        if not self.player or not self.player.frames:
            return
        frame = self.player.frames[self.current_index]
        state = self._edited_state or frame.output_state
        initial_json = json.dumps(state, indent=2)

        def on_edit_result(result: str | None) -> None:
            if result is not None:
                try:
                    self._edited_state = json.loads(result)
                    self._update_status()
                    self.notify("State updated ✓", severity="information")
                except json.JSONDecodeError as e:
                    self.notify(f"Invalid JSON: {e}", severity="error")

        self.push_screen(
            EditStateScreen(initial_json, self.current_index),
            callback=on_edit_result,
        )

    def action_resume(self) -> None:
        """Resume execution from the current frame with optional state edits."""
        if not self.player:
            return

        state = self._edited_state or (
            self.player.frames[self.current_index].output_state
            if self.player.frames
            else {}
        )

        self.notify(
            f"Resume from frame {self.current_index} "
            f"({'edited state' if self._edited_state else 'original state'})\n"
            f"Use Python SDK:\n"
            f"  player.resume(agent_fn, ResumeConfig(from_frame={self.current_index}))",
            severity="information",
        )

    def action_search(self) -> None:
        """Open search/filter modal."""
        timeline = self.query_one(FrameList)

        def on_search_result(result: str) -> None:
            timeline.filter_text = result
            if result:
                self.notify(f"Filter: '{result}'", severity="information")
            else:
                self.notify("Filter cleared", severity="information")

        self.push_screen(
            SearchScreen(timeline.filter_text),
            callback=on_search_result,
        )


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
