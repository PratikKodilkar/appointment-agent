import sys

import pytest
from langsmith import unit

from react_agent.graph import react_graph

generate_response_mod = sys.modules["react_agent.nodes.generate_response"]


class _FakeModel:
    def bind_tools(self, tools):
        return self

    def get_num_tokens_from_messages(self, messages, tools=None):
        return sum(len(str(getattr(m, "content", m))) for m in messages)

    async def ainvoke(self, messages, config=None):
        from langchain_core.messages import AIMessage

        return AIMessage(content="hi there")


@pytest.mark.asyncio
@unit
async def test_react_agent_simple_passthrough(monkeypatch) -> None:
    monkeypatch.setattr(
        generate_response_mod, "load_chat_model", lambda name: _FakeModel()
    )

    res = await react_graph.ainvoke(
        {"messages": [("user", "hi?")]},
        {
            "configurable": {
                "system_prompt": "You are a helpful AI assistant.",
                "model": "groq/llama-3.2-1b-preview",
            }
        },
    )

    assert len(str(res["messages"][-1].content).lower()) > 0
