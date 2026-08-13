from unittest.mock import MagicMock

from langchain_core.tools import BaseTool
from pydantic import BaseModel


class _EmptyArgsSchema(BaseModel):
    """Real pydantic model so ToolNode's schema introspection terminates."""


def _fake_tool(name: str) -> MagicMock:
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


def test_get_tools_called_once(monkeypatch):
    import sys

    calls = []

    def fake_get_tools(self, actions=None, **kwargs):
        calls.append(actions)
        return [_fake_tool(str(a)) for a in (actions or [])]

    class FakeComposioToolSet:
        def __init__(self, **kw):
            pass

        def get_tools(self, actions=None, **kwargs):
            return fake_get_tools(self, actions, **kwargs)

    class Action:
        pass

    Action.GOOGLECALENDAR_FIND_FREE_SLOTS = "GOOGLECALENDAR_FIND_FREE_SLOTS"
    Action.GOOGLECALENDAR_CREATE_EVENT = "GOOGLECALENDAR_CREATE_EVENT"
    Action.GMAIL_CREATE_EMAIL_DRAFT = "GMAIL_CREATE_EMAIL_DRAFT"

    import composio_langgraph
    monkeypatch.setattr(composio_langgraph, "ComposioToolSet", FakeComposioToolSet, raising=False)
    monkeypatch.setattr(composio_langgraph, "Action", Action, raising=False)

    sys.modules.pop("appointment_agent.nodes._tools", None)

    import appointment_agent.nodes._tools  # noqa: F401

    assert len(calls) == 1
