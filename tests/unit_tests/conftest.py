import sys
from unittest.mock import MagicMock

import pytest
from langchain_core.tools import BaseTool
from pydantic import BaseModel


class _EmptyArgsSchema(BaseModel):
    """Real pydantic model so ToolNode's schema introspection terminates.

    A MagicMock's own get_input_schema() would return another MagicMock,
    which get_all_basemodel_annotations() recurses into infinitely.
    """


def _fake_tool(name: str) -> MagicMock:
    """Create a fake LangChain Tool for testing.

    Returns a MagicMock with BaseTool spec so it passes isinstance(mock, BaseTool)
    checks in ToolNode while remaining fully mockable for Task 1's assertions.
    Pre-configures the attributes ToolNode.__init__ introspects (get_input_schema,
    func, coroutine) so construction doesn't recurse or blow up on mock internals.
    Note: .name attribute must be set explicitly after construction since
    MagicMock(name=...) sets the mock's repr name, not an actual .name attribute.
    """
    tool = MagicMock(spec=BaseTool)
    tool.name = name
    tool.description = f"Fake tool {name}"
    tool.args_schema = _EmptyArgsSchema
    tool.get_input_schema.return_value = _EmptyArgsSchema
    tool.func = None
    tool.coroutine = None
    tool.metadata = {}
    tool.tags = None
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
    monkeypatch.setattr(composio_langgraph, "ComposioToolSet", FakeComposioToolSet, raising=False)
    monkeypatch.setattr(composio_langgraph, "Action", Action, raising=False)

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
