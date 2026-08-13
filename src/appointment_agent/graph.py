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
