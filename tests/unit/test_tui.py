"""Tests for the Terminal User Interface (TUI)."""

from pathlib import Path

import pytest

from agent_vcr.tui import StateViewer, VCRApp

pytestmark = pytest.mark.asyncio


async def test_tui_loads_file(sample_vcr_file: Path):
    """Test that the TUI can load a valid VCR file and render properly."""

    app = VCRApp(str(sample_vcr_file))
    async with app.run_test() as pilot:
        # Give it a moment to load async file
        await pilot.pause(0.1)

        # Verify title updated to include session ID
        assert "Agent VCR -" in app.title

        # Verify the timeline has frames loaded
        assert len(app.player.frames) > 0

        # Verify keybindings exist
        binding_actions = [b.action for b in app.BINDINGS]
        assert "quit" in binding_actions

        # Verify we can interact with it
        await pilot.press("down")
        await pilot.pause(0.05)

        # The selected index should be 1 now
        assert app.current_index == 1

        # Change view mode
        await pilot.press("1")
        await pilot.pause(0.05)

        viewer = app.query_one(StateViewer)
        assert viewer.view_mode == "input"


async def test_tui_navigation(sample_vcr_file: Path):
    """Test frame navigation with arrow keys and j/k."""
    app = VCRApp(str(sample_vcr_file))
    async with app.run_test() as pilot:
        await pilot.pause(0.1)

        # Navigate forward
        await pilot.press("j")
        await pilot.pause(0.05)
        assert app.current_index == 1

        await pilot.press("j")
        await pilot.pause(0.05)
        assert app.current_index == 2

        # Navigate backward
        await pilot.press("k")
        await pilot.pause(0.05)
        assert app.current_index == 1


async def test_tui_view_modes(sample_vcr_file: Path):
    """Test switching between input, output, and diff views."""
    app = VCRApp(str(sample_vcr_file))
    async with app.run_test() as pilot:
        await pilot.pause(0.1)

        viewer = app.query_one(StateViewer)

        # Switch to input view
        await pilot.press("1")
        await pilot.pause(0.05)
        assert viewer.view_mode == "input"

        # Switch to output view
        await pilot.press("2")
        await pilot.pause(0.05)
        assert viewer.view_mode == "state"

        # Switch to diff view
        await pilot.press("3")
        await pilot.pause(0.05)
        assert viewer.view_mode == "diff"


async def test_tui_diff_coloring(sample_vcr_file: Path):
    """Test that diff view renders without errors."""
    app = VCRApp(str(sample_vcr_file))
    async with app.run_test() as pilot:
        await pilot.pause(0.1)

        # Go to second frame so there's a prev frame for diff
        await pilot.press("down")
        await pilot.pause(0.05)

        # Switch to diff view
        await pilot.press("3")
        await pilot.pause(0.05)

        viewer = app.query_one(StateViewer)
        assert viewer.view_mode == "diff"
        assert viewer.prev_frame is not None


async def test_tui_edit_binding_exists(sample_vcr_file: Path):
    """Test that the edit keybinding is registered."""
    app = VCRApp(str(sample_vcr_file))
    binding_actions = [b.action for b in app.BINDINGS]
    assert "edit_state" in binding_actions
    assert "resume" in binding_actions
    assert "search" in binding_actions


async def test_tui_status_bar(sample_vcr_file: Path):
    """Test that the status bar updates with frame info."""
    app = VCRApp(str(sample_vcr_file))
    async with app.run_test() as pilot:
        await pilot.pause(0.1)

        # Status bar should contain frame info
        from textual.widgets import Static
        status = app.query_one("#status-bar", Static)
        # Just verify it rendered without error
        assert status is not None
