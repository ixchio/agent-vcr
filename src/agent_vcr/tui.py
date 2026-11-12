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
import contextlib
import json
import os
import sys
import tempfile
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
from textual.widgets import Button, Footer, Header, Input, Static, TextArea

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
    loaded: reactive[bool] = reactive(False)

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
            if self.loaded:
                return Panel(
                    Text("This session has no frames", style="dim italic"),
                    title="Timeline",
                    border_style="blue",
                )
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
            # Return Rich Text for coloured diff
            return Panel(self._render_diff(), title="State Diff", border_style="cyan")
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


class EditStateScreen(ModalScreen):
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

    # priority: the focused TextArea must not swallow the save/cancel keys.
    BINDINGS = [
        Binding("ctrl+s", "save", "Save", priority=True),
        Binding("escape", "cancel", "Cancel", priority=True),
    ]

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
            self.action_save()
        else:
            self.action_cancel()

    def action_save(self) -> None:
        self.dismiss(self.query_one(TextArea).text)

    def action_cancel(self) -> None:
        self.dismiss(None)


class SearchScreen(ModalScreen):
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
        # Set when the file cannot be loaded; the app then exits with return
        # code 1 and callers (the ``vcr`` CLI) print this message.
        self.load_error: str | None = None

    async def on_mount(self) -> None:
        """Load the VCR file when the app starts."""
        try:
            self.player = await AsyncVCRPlayer.load(self.filepath)
            self.title = f"Agent VCR - {self.player.session.session_id}"

            timeline = self.query_one(FrameList)
            timeline.frames = self.player.frames
            timeline.loaded = True
            self.update_selection(0)
            self._update_status()
        except Exception as e:
            self.load_error = f"Error loading VCR file {self.filepath}: {e}"
            self.exit(self.load_error, return_code=1)

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
            edited = " [yellow]✏ EDITED[/]" if self._edited_state is not None else ""
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
        state = self._edited_state if self._edited_state is not None else frame.output_state
        initial_json = json.dumps(state, indent=2)

        def on_edit_result(result: str | None) -> None:
            if result is not None:
                try:
                    parsed = json.loads(result)
                except json.JSONDecodeError as e:
                    self.notify(f"Invalid JSON: {e}", severity="error")
                    return
                if not isinstance(parsed, dict):
                    self.notify("State must be a JSON object", severity="error")
                    return
                self._edited_state = parsed
                self._update_status()
                self.notify("State updated ✓", severity="information")

        self.push_screen(
            EditStateScreen(initial_json, self.current_index),
            callback=on_edit_result,
        )

    def action_resume(self) -> None:
        """Resume execution from the current frame with optional state edits.

        Exports the current (possibly edited) state to a JSON file so
        external scripts or the Python SDK can pick it up.
        """
        if not self.player:
            return
        if not self.player.frames:
            self.notify("This session has no frames to resume from", severity="warning")
            return

        frame = self.player.frames[self.current_index]
        edited = self._edited_state is not None
        state = self._edited_state if self._edited_state is not None else frame.output_state

        # Write the resume payload to a file so it's actually actionable
        resume_payload = {
            "session_id": self.player.session.session_id,
            "from_frame": self.current_index,
            "state": state,
            "edited": edited,
        }
        resume_path = Path(self.filepath).parent / "resume_state.json"
        try:
            _atomic_write_text(resume_path, json.dumps(resume_payload, indent=2, default=str))
            self.notify(
                f"Resume state saved to {resume_path}\n"
                f"Frame {self.current_index} | "
                f"{'edited' if edited else 'original'} state\n"
                f"Load with: json.load(open('{resume_path}'))",
                severity="information",
            )
        except Exception as e:
            self.notify(f"Failed to save resume state: {e}", severity="error")

    def action_search(self) -> None:
        """Open search/filter modal."""
        timeline = self.query_one(FrameList)

        def on_search_result(result: str | None) -> None:
            if result is None:
                return
            timeline.filter_text = result
            if result:
                self.notify(f"Filter: '{result}'", severity="information")
            else:
                self.notify("Filter cleared", severity="information")

        self.push_screen(
            SearchScreen(timeline.filter_text),
            callback=on_search_result,
        )


def _atomic_write_text(path: Path, text: str) -> None:
    """Write via a tmp sibling + ``os.replace`` so readers never see half a file."""
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(text)
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def run_app(filepath: str | Path) -> int:
    """Run the TUI on ``filepath``; return a process exit code.

    Load errors are printed to stderr and return 1 (the app itself only
    exits; it cannot print once the terminal is restored).
    """
    path = Path(filepath)
    if not path.exists():
        print(f"Error: File not found: {path}", file=sys.stderr)
        return 1
    if path.is_dir():
        print(f"Error: {path} is a directory, not a .vcr file", file=sys.stderr)
        return 1

    app = VCRApp(str(path))
    app.run()
    if app.load_error:
        print(app.load_error, file=sys.stderr)
        return 1
    return int(getattr(app, "return_code", 0) or 0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent VCR Terminal UI")
    parser.add_argument("file", help="Path to the .vcr file to replay")
    args = parser.parse_args()
    sys.exit(run_app(args.file))


if __name__ == "__main__":
    main()
