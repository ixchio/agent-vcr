"""Tests for the Terminal User Interface (TUI)."""

from pathlib import Path

import pytest

from agent_vcr.tui import VCRApp

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
        assert "quit" in [b[1] for b in app.BINDINGS]

        # Verify we can interact with it
        await pilot.press("down")
        await pilot.pause(0.05)

        # The selected index should be 1 now
        assert app.current_index == 1

        # Change view mode
        await pilot.press("1")
        await pilot.pause(0.05)

        viewer = app.query_one("StateViewer")
        assert viewer.view_mode == "input"
