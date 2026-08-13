from unittest.mock import AsyncMock, MagicMock
import sys

import pytest
from langchain_core.messages import AIMessage, HumanMessage


@pytest.mark.asyncio
async def test_custom_system_prompt_is_used(appointment_nodes, monkeypatch):
    # Access the module from sys.modules to get the actual module object
    gen_mod = sys.modules["appointment_agent.nodes.generate_response"]

    captured = {}

    async def fake_ainvoke(messages, config):
        captured["messages"] = messages
        return AIMessage(content="ok")

    # Create a mock with ainvoke method
    mock_model = MagicMock()
    mock_model.ainvoke = fake_ainvoke

    # Replace the entire model_with_tools object
    monkeypatch.setattr(gen_mod, "model_with_tools", mock_model)

    state = {"messages": [HumanMessage(content="hi")]}
    config = {"configurable": {"system_prompt": "You are Bob from Bob's Garage."}}

    await gen_mod.generate_response(state, config)

    system_content = captured["messages"][0]["content"]
    assert "Bob's Garage" in system_content
