import sys
from unittest.mock import MagicMock

import pytest
from langchain_core.tools import Tool


def _fake_tool(name: str) -> Tool:
    """Create a fake LangChain Tool for testing."""
    def dummy_func():
        """Dummy tool function."""
        return ""

    tool = Tool.from_function(
        func=dummy_func,
        name=name,
        description=f"Fake tool {name}",
    )
    return tool


@pytest.fixture
def appointment_nodes(monkeypatch):
    """Import appointment_agent's node modules with Composio calls stubbed out.

    _tools.py calls ComposioToolSet.get_tools() at import time, which hits a
    real API. This fixture patches that method first, then does a fresh
    import so no real network call happens.
    """
    fakes = {
        "find_slots": _fake_tool("GOOGLECALENDAR_FIND_FREE_SLOTS"),
        "create_event": _fake_tool("GOOGLECALENDAR_CREATE_EVENT"),
        "gmail_draft": _fake_tool("GMAIL_CREATE_EMAIL_DRAFT"),
    }

    def fake_get_tools(self, actions=None, **kwargs):
        wanted = {getattr(a, "name", str(a)) for a in (actions or [])}
        return [t for t in fakes.values() if t.name in wanted or not wanted]

    # Create a fake ComposioToolSet class
    class FakeComposioToolSet:
        def __init__(self, **kw):
            pass

        def get_tools(self, actions=None, **kwargs):
            return fake_get_tools(self, actions, **kwargs)

    # Also create a fake Action enum
    class Action:
        pass

    Action.GOOGLECALENDAR_FIND_FREE_SLOTS = _fake_tool("GOOGLECALENDAR_FIND_FREE_SLOTS")
    Action.GOOGLECALENDAR_CREATE_EVENT = _fake_tool("GOOGLECALENDAR_CREATE_EVENT")
    Action.GMAIL_CREATE_EMAIL_DRAFT = _fake_tool("GMAIL_CREATE_EMAIL_DRAFT")

    # Patch the composio_langgraph module to add the ComposioToolSet class and Action enum
    import composio_langgraph
    composio_langgraph.ComposioToolSet = FakeComposioToolSet
    composio_langgraph.Action = Action

    mod_names = [
        "appointment_agent.nodes._tools",
        "appointment_agent.nodes.generate_response",
        "appointment_agent.graph",
    ]
    for name in mod_names:
        sys.modules.pop(name, None)

    import appointment_agent.nodes._tools as tools_mod
    import appointment_agent.nodes.generate_response as gen_mod

    yield {"tools": tools_mod, "generate_response": gen_mod, "fakes": fakes}

    for name in mod_names:
        sys.modules.pop(name, None)
