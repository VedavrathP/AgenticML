"""
LangGraph builder for the orchestrator-centered ML pipeline.

Topology: hub-and-spoke with the Orchestrator as the central node.

    ┌─────────────────────────────────────────────┐
    │              Orchestrator                    │
    │  (entry point — decides next agent or END)   │
    └──┬──┬──┬──┬──┬──┬──┬───────────────────────┘
       │  │  │  │  │  │  │
       ▼  ▼  ▼  ▼  ▼  ▼  ▼
     DP PP FE MS MT EV IV   ← specialised agents
       │  │  │  │  │  │  │
       └──┴──┴──┴──┴──┴──┘
              │
              ▼
         Orchestrator  (re-evaluates after each agent)
              │
              ▼
             END
"""

from typing import Literal

from langgraph.graph import StateGraph, END

from agenticml.state.workflow_state import WorkflowState
from agenticml.orchestrator.orchestrator_agent import run_orchestrator
from agenticml.agents import AGENT_REGISTRY


# ---------------------------------------------------------------------------
# Node wrappers — instantiate each agent and call run(state)
# ---------------------------------------------------------------------------

def _make_agent_node(agent_cls):
    """Return a node function that instantiates the agent and calls run."""
    def _node(state: WorkflowState) -> WorkflowState:
        agent = agent_cls()
        return agent.run(state)
    _node.__name__ = agent_cls.name + "_node"
    return _node


# ---------------------------------------------------------------------------
# Routing function
# ---------------------------------------------------------------------------

_VALID_AGENTS = set(AGENT_REGISTRY.keys())

def _route_from_orchestrator(
    state: WorkflowState,
) -> str:
    """Read state['next_agent'] and return the routing key."""
    next_agent = state.get("next_agent", "done")
    if next_agent in _VALID_AGENTS:
        return next_agent
    return "done"


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """Build the hub-and-spoke StateGraph."""
    graph = StateGraph(WorkflowState)

    # Central orchestrator node
    graph.add_node("orchestrator", run_orchestrator)

    # Specialised agent nodes
    for name, cls in AGENT_REGISTRY.items():
        graph.add_node(name, _make_agent_node(cls))

    # Entry point
    graph.set_entry_point("orchestrator")

    # Every agent returns to orchestrator
    for name in AGENT_REGISTRY:
        graph.add_edge(name, "orchestrator")

    # Orchestrator conditionally routes to an agent or END
    route_map = {name: name for name in AGENT_REGISTRY}
    route_map["done"] = END

    graph.add_conditional_edges(
        "orchestrator",
        _route_from_orchestrator,
        route_map,
    )

    return graph


def compile_graph():
    """Compile the graph, ready for execution."""
    return build_graph().compile()


_GRAPH_CONFIG = {"recursion_limit": 100}


def run_graph(initial_state: WorkflowState) -> WorkflowState:
    """Run the graph to completion and return the final state."""
    app = compile_graph()
    return app.invoke(initial_state, config=_GRAPH_CONFIG)


def run_graph_streaming(initial_state: WorkflowState):
    """Yield full accumulated state snapshots as the graph progresses.

    Uses ``stream_mode="values"`` so each yield is the complete
    WorkflowState after a node executes (not a partial update).
    """
    app = compile_graph()
    yield from app.stream(initial_state, stream_mode="values", config=_GRAPH_CONFIG)
