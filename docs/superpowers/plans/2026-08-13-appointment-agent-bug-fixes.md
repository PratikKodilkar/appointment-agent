# Appointment Agent Bug Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the 10 bugs found in the 2026-08-13 code review of `src/appointment_agent/`, in priority order, each with a regression test so it can't silently regress.

**Architecture:** No new components. One task (Task 1) removes the custom `find_slots` node/routing and merges its responsibility into the existing `ToolNode`, because that single change eliminates three of the ten bugs (unanswered parallel tool calls, unguarded `None` tool lookup, dead/shadowed `tools_condition` import) with less code than patching each one individually. It also folds in the redundant-`get_tools()`-call fix (bug 9), since the extra `schedule_tools_write` list becomes unused once the two tool sets merge. Every other task is a narrow, single-file fix.

**Tech Stack:** LangGraph (`langgraph.prebuilt.ToolNode`, `tools_condition`), LangChain Core (`trim_messages`), Composio (`composio_langgraph`), `langchain_google_genai`, `requests`, `pytest` + `pytest-asyncio`.

## Global Constraints

- No real network calls in the test suite. `composio_langgraph.ComposioToolSet.get_tools` and any `ChatGoogleGenerativeAI`/`requests` call reaching a live API must be mocked. (Project `CLAUDE.md`.)
- `appointment_agent/nodes/_tools.py` and `appointment_agent/nodes/generate_response.py` call `ComposioToolSet.get_tools(...)` and construct `ChatGoogleGenerativeAI(...)` **at import time**. Tests that need these modules must patch `ComposioToolSet.get_tools` *before* importing them, and must not rely on a prior, unpatched import already sitting in `sys.modules`.
- Run `python3 -m pytest tests/ -v` after every task; all tests (old and new) must pass before moving to the next task.
- Each task is scoped to be its own session/commit — do not start the next task until the current one's tests are green and committed.

## Priority Order & Rationale

| # | Task | Bugs fixed | Why this order |
|---|------|-----------|-----------------|
| 0 | Test infrastructure | — (enabler) | Tasks 1–4, 6 can't be tested without it |
| 1 | Unify tool routing into one `ToolNode` | #1 unanswered parallel tool_calls, #4 blocking sync call, #5 unguarded `None`, #6 dead import shadow, #9 redundant `get_tools()` | Only bug that can crash *any* conversation outright; highest blast radius |
| 2 | Wire `Configuration.system_prompt` through | #2 dead config field | Silently breaks an advertised customization feature |
| 3 | Confirm and fix Gemini model id | #7 model name mismatch | If the id is invalid, every single response fails — but needs a decision from you first (see task) |
| 4 | Add timeout to Bland.ai call | #3 hangs indefinitely | Real reliability risk, but only triggers when Bland.ai is slow/down |
| 5 | Constrain `trim_messages` boundaries | #8 mid-tool-call truncation | Only manifests in long conversations — lower likelihood |
| 6 | Fix `Dockerfile` `LANGSERVE_GRAPHS` | #10 wrong graph exposed | `compose.yaml` already overrides the container command to `langgraph dev`, which reads `langgraph.json` (already correct) — so this only matters for a deploy path that uses the image's default entrypoint. Lowest real-world impact of the ten. |

Bug numbering matches the original code-review findings you already have.

---

### Task 0: Test infrastructure for Composio/Gemini-touching modules

**Files:**
- Create: `tests/unit_tests/conftest.py`

**Interfaces:**
- Produces: `appointment_nodes` pytest fixture — yields a dict `{"tools": <_tools module>, "generate_response": <generate_response module>, "fakes": {"find_slots": MagicMock, "create_event": MagicMock, "gmail_draft": MagicMock}}` with `composio_langgraph.ComposioToolSet.get_tools` patched and the target modules freshly imported. Tasks 1, 2, 3, 6 depend on this fixture's exact shape.

- [ ] **Step 1: Write the fixture**

```python
# tests/unit_tests/conftest.py
import sys
from unittest.mock import MagicMock

import pytest


def _fake_tool(name: str) -> MagicMock:
    tool = MagicMock()
    tool.name = name
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

    monkeypatch.setattr(
        "composio_langgraph.ComposioToolSet.get_tools", fake_get_tools
    )
    monkeypatch.setattr(
        "composio_langgraph.ComposioToolSet.__init__", lambda self, **kw: None
    )

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
```

- [ ] **Step 2: Verify it imports cleanly without hitting the network**

Run: `python3 -m pytest tests/unit_tests/conftest.py --collect-only -q`
Expected: no errors (conftest has no tests of its own, this just checks it's syntactically valid and importable).

- [ ] **Step 3: Write a smoke test using the fixture**

```python
# tests/unit_tests/test_conftest_fixture.py
def test_appointment_nodes_fixture_avoids_network(appointment_nodes):
    tools_mod = appointment_nodes["tools"]
    names = {t.name for t in tools_mod.schedule_tools_set}
    assert names == {
        "GOOGLECALENDAR_FIND_FREE_SLOTS",
        "GOOGLECALENDAR_CREATE_EVENT",
        "GMAIL_CREATE_EMAIL_DRAFT",
    }
```

- [ ] **Step 4: Run it**

Run: `python3 -m pytest tests/unit_tests/test_conftest_fixture.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/unit_tests/conftest.py tests/unit_tests/test_conftest_fixture.py
git commit -m "test: add fixture to import appointment_agent nodes without live API calls"
```

---

### Task 1: Merge `find_slots` into a single `ToolNode`, delete the custom routing

**Files:**
- Modify: `src/appointment_agent/graph.py`
- Modify: `src/appointment_agent/nodes/_tools.py:33-41`
- Modify: `src/appointment_agent/nodes/__init__.py`
- Delete: `src/appointment_agent/nodes/find_slots.py`
- Test: `tests/unit_tests/test_tools.py`, `tests/integration_tests/test_appointment_graph.py`

**Interfaces:**
- Consumes: `appointment_nodes` fixture from Task 0 (`tests/unit_tests/conftest.py`)
- Produces: `schedule_tools_write_node` (existing name, kept for compatibility with `graph.py`) now built from `schedule_tools_set` (all three tools) instead of the separate `schedule_tools_write` list, which is deleted.

- [ ] **Step 1: Write failing test for bug #9 (redundant `get_tools` call)**

```python
# tests/unit_tests/test_tools.py
from unittest.mock import MagicMock


def test_get_tools_called_once(monkeypatch):
    import sys

    calls = []

    def fake_get_tools(self, actions=None, **kwargs):
        calls.append(actions)
        return [MagicMock(name=str(a)) for a in (actions or [])]

    monkeypatch.setattr(
        "composio_langgraph.ComposioToolSet.get_tools", fake_get_tools
    )
    monkeypatch.setattr(
        "composio_langgraph.ComposioToolSet.__init__", lambda self, **kw: None
    )
    sys.modules.pop("appointment_agent.nodes._tools", None)

    import appointment_agent.nodes._tools  # noqa: F401

    assert len(calls) == 1
```

- [ ] **Step 2: Run it, confirm it fails**

Run: `python3 -m pytest tests/unit_tests/test_tools.py::test_get_tools_called_once -v`
Expected: FAIL — `assert len(calls) == 1` fails because `calls == [actions1, actions2]` (two calls today).

- [ ] **Step 3: Write failing test for bug #1 (unanswered parallel tool_calls)**

```python
# tests/integration_tests/test_appointment_graph.py
import pytest
from langchain_core.messages import AIMessage, HumanMessage


@pytest.mark.asyncio
async def test_parallel_tool_calls_all_get_responses(appointment_nodes, monkeypatch):
    from appointment_agent.state import AppointmentAgentState

    fakes = appointment_nodes["fakes"]
    fakes["find_slots"].invoke.return_value = "slots: none"
    fakes["create_event"].invoke.return_value = "event created"

    import sys
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

    result = await graph_mod.schedule_tools_write_node.ainvoke(state)

    tool_call_ids = {m.tool_call_id for m in result["messages"]}
    assert tool_call_ids == {"call_1", "call_2"}
```

- [ ] **Step 4: Run it, confirm it fails**

Run: `python3 -m pytest tests/integration_tests/test_appointment_graph.py::test_parallel_tool_calls_all_get_responses -v`
Expected: FAIL — today `find_slots` (not `schedule_tools_write_node`) is the only node that would handle `call_1`, and it never touches `call_2`, so `tool_call_ids` only contains `{"call_1"}` when routed manually, or the test errors because `find_slots` isn't exercised by `schedule_tools_write_node` at all.

- [ ] **Step 5: Implement — collapse `_tools.py` to a single tool set**

```python
# src/appointment_agent/nodes/_tools.py (replace lines 24-41)
# Get the required tools
schedule_tools_set = composio_toolset.get_tools(
    actions=[
        Action.GOOGLECALENDAR_FIND_FREE_SLOTS,
        Action.GOOGLECALENDAR_CREATE_EVENT,
        Action.GMAIL_CREATE_EMAIL_DRAFT
    ]
)

schedule_tools_write_node = ToolNode(schedule_tools_set + [make_confirmation_call])
```

- [ ] **Step 6: Implement — delete the custom node and routing in `graph.py`**

```python
# src/appointment_agent/graph.py
"""This module defines the state graph for the react agent."""
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import tools_condition

from appointment_agent.configuration import Configuration
from appointment_agent.state import AppointmentAgentState
from appointment_agent.nodes import generate_response, schedule_tools_write_node

builder = StateGraph(AppointmentAgentState, config_schema=Configuration)

builder.add_node("agent", generate_response)
builder.add_node("tools", schedule_tools_write_node)

builder.add_edge(START, "agent")
builder.add_conditional_edges("agent", tools_condition, ["tools", END])
builder.add_edge("tools", "agent")

appointment_agent_graph = builder.compile()

appointment_agent_graph.name = "appointment_agent_graph"
```

- [ ] **Step 7: Implement — remove `find_slots` from the package**

Delete `src/appointment_agent/nodes/find_slots.py`.

```python
# src/appointment_agent/nodes/__init__.py
"""This module initializes the nodes for the react agent.

It imports the following nodes:
- `tools_node` from `appointment_agent.nodes._tools`
- `generate_response` from `appointment_agent.nodes.generate_response`

These nodes are included in the `__all__` list to specify the public API of this module.
"""

from appointment_agent.nodes._tools import schedule_tools_write_node
from appointment_agent.nodes.generate_response import generate_response

__all__ = ["schedule_tools_write_node", "generate_response"]
```

- [ ] **Step 8: Run both tests, confirm they pass**

Run: `python3 -m pytest tests/unit_tests/test_tools.py::test_get_tools_called_once tests/integration_tests/test_appointment_graph.py::test_parallel_tool_calls_all_get_responses -v`
Expected: PASS

- [ ] **Step 9: Run the full suite to check nothing else references `find_slots`**

Run: `python3 -m pytest tests/ -v`
Expected: all PASS. If anything still imports `appointment_agent.nodes.find_slots`, fix that import.

- [ ] **Step 10: Commit**

```bash
git add src/appointment_agent/graph.py src/appointment_agent/nodes/_tools.py \
        src/appointment_agent/nodes/__init__.py \
        tests/unit_tests/test_tools.py tests/integration_tests/test_appointment_graph.py
git rm src/appointment_agent/nodes/find_slots.py
git commit -m "fix: merge find_slots into single ToolNode to fix unanswered parallel tool calls"
```

---

### Task 2: Wire `Configuration.system_prompt` into `generate_response`

**Files:**
- Modify: `src/appointment_agent/nodes/generate_response.py:35-37`
- Test: `tests/unit_tests/test_generate_response.py`

**Interfaces:**
- Consumes: `appointment_nodes` fixture (Task 0); `Configuration.from_runnable_config` (existing, `src/appointment_agent/configuration.py:26-33`)
- Produces: no change to `generate_response`'s public signature.

- [ ] **Step 1: Write failing test**

```python
# tests/unit_tests/test_generate_response.py
from unittest.mock import AsyncMock

import pytest
from langchain_core.messages import AIMessage, HumanMessage


@pytest.mark.asyncio
async def test_custom_system_prompt_is_used(appointment_nodes, monkeypatch):
    gen_mod = appointment_nodes["generate_response"]

    captured = {}

    async def fake_ainvoke(messages, config):
        captured["messages"] = messages
        return AIMessage(content="ok")

    monkeypatch.setattr(gen_mod.model_with_tools, "ainvoke", fake_ainvoke)

    state = {"messages": [HumanMessage(content="hi")]}
    config = {"configurable": {"system_prompt": "You are Bob from Bob's Garage."}}

    await gen_mod.generate_response(state, config)

    system_content = captured["messages"][0]["content"]
    assert "Bob's Garage" in system_content
```

- [ ] **Step 2: Run it, confirm it fails**

Run: `python3 -m pytest tests/unit_tests/test_generate_response.py -v`
Expected: FAIL — `system_content` still contains the hardcoded `AGENT_SYSTEM` dental-clinic prompt, not "Bob's Garage".

- [ ] **Step 3: Implement**

```python
# src/appointment_agent/nodes/generate_response.py
from appointment_agent.configuration import Configuration
```

Add the import above near the other `appointment_agent` imports, then replace lines 35-37:

```python
    configuration = Configuration.from_runnable_config(config)
    today_datetime = datetime.datetime.now().isoformat()
    system_message = configuration.system_prompt.format(today_datetime=today_datetime)
```

- [ ] **Step 4: Run test, confirm it passes**

Run: `python3 -m pytest tests/unit_tests/test_generate_response.py -v`
Expected: PASS

- [ ] **Step 5: Run full suite**

Run: `python3 -m pytest tests/ -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/appointment_agent/nodes/generate_response.py tests/unit_tests/test_generate_response.py
git commit -m "fix: use Configuration.system_prompt instead of hardcoded AGENT_SYSTEM"
```

---

### Task 3: Align README with the intentional Gemini model id

**Decision (confirmed by user 2026-08-13):** `"gemini-3.5-flash-lite"` in `generate_response.py:16` is intentional — keep it as-is. The bug is that `README.md` documents the old `"Gemini-2.0-flash-exp"`, which is now stale. Fix the doc, not the code.

**Files:**
- Modify: `README.md`

**Interfaces:** none new — no code changes in this task.

- [ ] **Step 1: Find the stale reference**

Run: `grep -n "Gemini-2.0-flash-exp\|gemini-2.0" README.md`

- [ ] **Step 2: Update README.md to reference `gemini-3.5-flash-lite`**

Replace the stale `Gemini-2.0-flash-exp` mention(s) with `gemini-3.5-flash-lite` so the doc matches `src/appointment_agent/nodes/generate_response.py:16`.

- [ ] **Step 3: Smoke-check the model id is still accepted (confidence check, not a regression risk since code is unchanged)**

Run (requires a real `GOOGLE_API_KEY` in `.env`):

```bash
python3 -c "
from langchain_google_genai import ChatGoogleGenerativeAI
m = ChatGoogleGenerativeAI(model='gemini-3.5-flash-lite')
print(m.invoke('say hi in one word').content)
"
```

Expected: prints a short response, no `NotFound`/`InvalidArgument` error. If this fails, stop and flag it — it means the model id isn't actually reachable with the current API key/project, which contradicts the "intentional" decision above and needs to be revisited before committing.

- [ ] **Step 4: Run full suite**

Run: `python3 -m pytest tests/ -v`
Expected: all PASS (doc-only change, suite should be unaffected).

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: align README model reference with gemini-3.5-flash-lite"
```

---

### Task 4: Add a timeout to the Bland.ai confirmation call

**Files:**
- Modify: `src/appointment_agent/tools/make_confirmation_call.py:33`
- Test: `tests/unit_tests/test_make_confirmation_call.py`

**Interfaces:** none new — `make_confirmation_call(phone_number, instructions)` signature unchanged.

- [ ] **Step 1: Write failing test**

```python
# tests/unit_tests/test_make_confirmation_call.py
from unittest.mock import patch, MagicMock

from appointment_agent.tools.make_confirmation_call import make_confirmation_call


def test_post_call_has_timeout():
    fake_response = MagicMock()
    fake_response.json.return_value = {"status": "ok"}

    with patch("appointment_agent.tools.make_confirmation_call.requests.post", return_value=fake_response) as mock_post:
        make_confirmation_call("+15551234567", "confirm appointment")

    _, kwargs = mock_post.call_args
    assert "timeout" in kwargs
    assert kwargs["timeout"] is not None
```

- [ ] **Step 2: Run it, confirm it fails**

Run: `python3 -m pytest tests/unit_tests/test_make_confirmation_call.py -v`
Expected: FAIL — `assert "timeout" in kwargs` fails, no `timeout` kwarg passed today.

- [ ] **Step 3: Implement**

```python
# src/appointment_agent/tools/make_confirmation_call.py:33
    response = requests.post(url, json=payload, headers=headers, timeout=30)
```

- [ ] **Step 4: Run test, confirm it passes**

Run: `python3 -m pytest tests/unit_tests/test_make_confirmation_call.py -v`
Expected: PASS

- [ ] **Step 5: Run full suite**

Run: `python3 -m pytest tests/ -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/appointment_agent/tools/make_confirmation_call.py tests/unit_tests/test_make_confirmation_call.py
git commit -m "fix: add timeout to Bland.ai confirmation call to prevent indefinite hangs"
```

---

### Task 5: Prevent `trim_messages` from splitting a tool_call/tool_response pair

**Files:**
- Modify: `src/appointment_agent/nodes/generate_response.py:39-46`
- Test: `tests/unit_tests/test_generate_response.py`

**Interfaces:**
- Consumes: `appointment_nodes` fixture (Task 0)

- [ ] **Step 1: Write failing test**

```python
# tests/unit_tests/test_generate_response.py (append)
from langchain_core.messages import ToolMessage


@pytest.mark.asyncio
async def test_trim_messages_never_ends_mid_tool_call(appointment_nodes, monkeypatch):
    gen_mod = appointment_nodes["generate_response"]

    captured = {}

    async def fake_ainvoke(messages, config):
        captured["messages"] = messages
        return AIMessage(content="ok")

    monkeypatch.setattr(gen_mod.model_with_tools, "ainvoke", fake_ainvoke)
    monkeypatch.setattr(gen_mod, "trim_messages", lambda *a, **kw: kw)

    # Assert trim_messages is called with boundary constraints, not the raw
    # allow_partial=True/no-boundary config that lets it cut mid tool-call.
    state = {
        "messages": [
            HumanMessage(content="book a slot"),
            AIMessage(
                content="",
                tool_calls=[{"name": "x", "args": {}, "id": "1"}],
            ),
            ToolMessage(content="done", tool_call_id="1", name="x"),
        ]
    }
    await gen_mod.generate_response(state, {})

    call_kwargs = gen_mod.trim_messages.__wrapped__ if hasattr(gen_mod.trim_messages, "__wrapped__") else None
```

Simplify — the meaningful assertion is on the *call arguments* passed to `trim_messages`, so patch it directly and inspect the call:

```python
# tests/unit_tests/test_generate_response.py (append, replacing the test above)
from unittest.mock import MagicMock


@pytest.mark.asyncio
async def test_trim_messages_has_boundary_constraints(appointment_nodes, monkeypatch):
    gen_mod = appointment_nodes["generate_response"]

    async def fake_ainvoke(messages, config):
        return AIMessage(content="ok")

    monkeypatch.setattr(gen_mod.model_with_tools, "ainvoke", fake_ainvoke)

    real_trim_messages = gen_mod.trim_messages
    mock_trim = MagicMock(side_effect=real_trim_messages)
    monkeypatch.setattr(gen_mod, "trim_messages", mock_trim)

    state = {"messages": [HumanMessage(content="hi")]}
    await gen_mod.generate_response(state, {})

    _, kwargs = mock_trim.call_args
    assert kwargs.get("end_on") is not None
    assert kwargs.get("start_on") is not None
```

- [ ] **Step 2: Run it, confirm it fails**

Run: `python3 -m pytest tests/unit_tests/test_generate_response.py::test_trim_messages_has_boundary_constraints -v`
Expected: FAIL — `kwargs.get("end_on")` is `None` today (no `end_on`/`start_on` passed).

- [ ] **Step 3: Implement**

```python
# src/appointment_agent/nodes/generate_response.py:39-46
    trimmedStateMessages = trim_messages(
        state["messages"],
        max_tokens=60000,  # adjust for model's context window minus system & files message
        strategy="last",
        token_counter=model,
        include_system=False,  # Not needed since systemMessage is added separately
        allow_partial=False,
        start_on="human",
        end_on=("human", "tool"),
    )
```

- [ ] **Step 4: Run test, confirm it passes**

Run: `python3 -m pytest tests/unit_tests/test_generate_response.py::test_trim_messages_has_boundary_constraints -v`
Expected: PASS

- [ ] **Step 5: Run full suite**

Run: `python3 -m pytest tests/ -v`
Expected: all PASS

- [ ] **Step 6: Commit**

```bash
git add src/appointment_agent/nodes/generate_response.py tests/unit_tests/test_generate_response.py
git commit -m "fix: constrain trim_messages boundaries to avoid splitting tool_call/response pairs"
```

---

### Task 6: Fix `Dockerfile` `LANGSERVE_GRAPHS` to match `langgraph.json`

**Files:**
- Modify: `Dockerfile:9`

**Interfaces:** none — config-only change, not unit-testable via pytest.

- [ ] **Step 1: Implement**

```dockerfile
# Dockerfile:9
ENV LANGSERVE_GRAPHS='{"react_agent": "./src/react_agent/graph.py:react_graph", "appointment_agent": "./src/appointment_agent/graph.py:appointment_agent_graph"}'
```

- [ ] **Step 2: Verify manually**

Run: `docker compose build langgraph-api` (requires Docker running).
Expected: build succeeds.

Since `compose.yaml` already overrides the container `command` to `langgraph dev --port 8000` (which reads `langgraph.json`, already correct), this env var only matters for a deploy path using the image's default entrypoint. Confirm this is still worth carrying by checking whether any deploy target other than `compose.yaml` relies on the image default — if not, this step is a low-risk consistency fix either way.

- [ ] **Step 3: Commit**

```bash
git add Dockerfile
git commit -m "fix: expose appointment_agent graph in Dockerfile's LANGSERVE_GRAPHS"
```

---

## Self-Review Notes

- **Bug coverage:** all 10 original findings map to a task — #1/#4/#5/#6/#9 → Task 1, #2 → Task 2, #7 → Task 3, #3 → Task 4, #8 → Task 5, #10 → Task 6.
- **Non-bug findings from the review** (unrelated `pyproject.toml` template metadata, stale `CLAUDE.md` testing references to a nonexistent `OpenAICompatibleProvider`) are **not** included here — they weren't bugs, just noted drift. Flag separately if you want them cleaned up.
- **Task 3 is the one task with a real open question** (which model id is correct) — it can't be resolved from the code alone.
