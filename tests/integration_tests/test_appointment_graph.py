import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph._internal._constants import CONF, CONFIG_KEY_RUNTIME
from langgraph.runtime import Runtime


@pytest.mark.asyncio
async def test_parallel_tool_calls_all_get_responses(appointment_nodes, monkeypatch):
    from appointment_agent.state import AppointmentAgentState

    fakes = appointment_nodes["fakes"]
    fakes["find_slots"].invoke.return_value = "slots: none"
    fakes["create_event"].invoke.return_value = "event created"
    # ToolNode.ainvoke runs tools via their async path (tool.ainvoke), passing
    # a ToolCall-shaped dict; a real BaseTool auto-wraps its result into a
    # ToolMessage in that case, but a mock doesn't, so the fakes must return
    # ToolMessages directly (matching what ToolNode._normalize_tool_response
    # requires) instead of raw strings.
    fakes["find_slots"].ainvoke.return_value = ToolMessage(
        content="slots: none",
        tool_call_id="call_1",
        name="GOOGLECALENDAR_FIND_FREE_SLOTS",
    )
    fakes["create_event"].ainvoke.return_value = ToolMessage(
        content="event created",
        tool_call_id="call_2",
        name="GOOGLECALENDAR_CREATE_EVENT",
    )

    import sys
    # appointment_agent/__init__.py and appointment_agent/nodes/__init__.py
    # both re-export names via `from ... import ...`; those bindings are
    # frozen at each package's own import time, so both packages (not just
    # .graph) must be popped too, or a stale reference from whichever test
    # ran first in this process would leak in here.
    sys.modules.pop("appointment_agent", None)
    sys.modules.pop("appointment_agent.nodes", None)
    sys.modules.pop("appointment_agent.graph", None)
    from appointment_agent import graph as graph_mod

    ai_message = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "GOOGLECALENDAR_FIND_FREE_SLOTS",
                "args": {},
                "id": "call_1",
            },
            {
                "name": "GOOGLECALENDAR_CREATE_EVENT",
                "args": {},
                "id": "call_2",
            },
        ],
    )
    state = AppointmentAgentState(
        messages=[HumanMessage(content="book it"), ai_message]
    )

    # ToolNode.ainvoke requires a Runtime object in config (langgraph 1.x) that
    # a compiled graph normally injects; supply a minimal one for this
    # standalone-node invocation.
    config = {CONF: {CONFIG_KEY_RUNTIME: Runtime()}}
    result = await graph_mod.schedule_tools_write_node.ainvoke(state, config)

    # Assert on content, not just presence of a message per tool_call_id:
    # ToolNode._validate_tool_call already returns a graceful error
    # ToolMessage (with the right tool_call_id) for any *unregistered* tool
    # name, so a mere tool_call_id-presence check would pass even on the
    # pre-fix node (which lacks the find_slots tool and would silently
    # return an "unknown tool" error for call_1 instead of routing to it).
    messages_by_id = {m.tool_call_id: m for m in result["messages"]}
    assert messages_by_id.keys() == {"call_1", "call_2"}
    assert messages_by_id["call_1"].content == "slots: none"
    assert messages_by_id["call_2"].content == "event created"
