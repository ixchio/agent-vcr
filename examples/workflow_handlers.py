"""Handlers used by ``examples/hello_workflow.json``.

Run the example with:

    vcr workflow run --file examples/hello_workflow.json \
        --handlers examples/workflow_handlers.py
"""

from __future__ import annotations


def greet(name: str = "world") -> dict[str, str]:
    """Return a deterministic value that is safe to replay locally."""
    return {"message": f"Hello, {name}!"}


def uppercase(value: str = "") -> dict[str, str]:
    """Transform the preceding output to demonstrate output wiring."""
    return {"message": value.upper()}


TOOLS = {"greet": greet, "uppercase": uppercase}
