"""
LangGraph definition for the ML pipeline.

This module defines the StateGraph that orchestrates the entire ML workflow.
The graph controls flow, iteration, and conditional routing between agents.

Flow:
    Ingest → Planner → (Human Interaction if needed) → Profiler → Orchestrator 
    → Cleaner → Featurizer → Modeler → Evaluator → Critic → (loop) → Reporter → End
"""

from typing import Literal, Callable, Optional
from langgraph.graph import StateGraph, END

from agenticml.ml.state import PipelineState
from agenticml.ml.agents.planner import run_planner_agent, process_user_responses
from agenticml.ml.agents.orchestrator import run_orchestrator_agent
from agenticml.ml.agents.profiler import run_profiler_agent
from agenticml.ml.agents.cleaner import run_cleaner_agent
from agenticml.ml.agents.featurizer import run_featurizer_agent
from agenticml.ml.agents.modeler import run_modeler_agent
from agenticml.ml.agents.evaluator import run_evaluator_agent
from agenticml.ml.agents.critic import run_critic_agent
from agenticml.ml.agents.reporter import run_reporter_agent


# Global callback for human interaction (set by CLI)
_human_interaction_callback: Optional[Callable[[PipelineState], PipelineState]] = None


def set_human_interaction_callback(callback: Callable[[PipelineState], PipelineState]):
    """Set the callback function for human interaction."""
    global _human_interaction_callback
    _human_interaction_callback = callback


def get_human_interaction_callback() -> Optional[Callable[[PipelineState], PipelineState]]:
    """Get the current human interaction callback."""
    return _human_interaction_callback


# ============================================================================
# Node Functions
# ============================================================================

def ingest_node(state: PipelineState) -> PipelineState:
    """
    Initial node that validates input and prepares the pipeline.
    
    This is a simple pass-through that ensures state is properly initialized.
    """
    # Ensure required fields exist
    if "iteration" not in state:
        state["iteration"] = 0
    if "artifacts" not in state:
        state["artifacts"] = []
    if "decision_log" not in state:
        state["decision_log"] = []
    if "errors" not in state:
        state["errors"] = []
    if "clarification_questions" not in state:
        state["clarification_questions"] = []
    if "pending_questions" not in state:
        state["pending_questions"] = []
    if "user_responses" not in state:
        state["user_responses"] = {}
    if "planning_complete" not in state:
        state["planning_complete"] = False
    
    return state


def planner_node(state: PipelineState) -> PipelineState:
    """Run the planner agent to analyze data and generate clarification questions."""
    return run_planner_agent(state)


def human_interaction_node(state: PipelineState) -> PipelineState:
    """
    Handle human interaction for clarification questions.
    
    This node is called when the planner has questions that need user input.
    It uses the registered callback to get user responses.
    """
    pending_questions = state.get("pending_questions", [])
    
    if not pending_questions:
        # No questions to ask, mark as complete
        state["planning_complete"] = True
        return state
    
    # Get the human interaction callback
    callback = get_human_interaction_callback()
    
    if callback:
        # Use the callback to get user responses
        state = callback(state)
    else:
        # No callback registered - use defaults
        for q in pending_questions:
            q["status"] = "skipped"
            q["answer"] = q.get("default")
        state["pending_questions"] = []
        state["planning_complete"] = True
    
    # Process the responses
    if state.get("user_responses"):
        state = process_user_responses(state)
    
    return state


def profiler_node(state: PipelineState) -> PipelineState:
    """Run the profiler agent."""
    return run_profiler_agent(state)


def orchestrator_node(state: PipelineState) -> PipelineState:
    """Run the orchestrator agent."""
    return run_orchestrator_agent(state)


def cleaner_node(state: PipelineState) -> PipelineState:
    """Run the cleaner agent."""
    return run_cleaner_agent(state)


def featurizer_node(state: PipelineState) -> PipelineState:
    """Run the featurizer agent."""
    return run_featurizer_agent(state)


def modeler_node(state: PipelineState) -> PipelineState:
    """Run the modeler agent."""
    return run_modeler_agent(state)


def evaluator_node(state: PipelineState) -> PipelineState:
    """Run the evaluator agent."""
    return run_evaluator_agent(state)


def critic_node(state: PipelineState) -> PipelineState:
    """Run the critic agent."""
    return run_critic_agent(state)


def reporter_node(state: PipelineState) -> PipelineState:
    """Run the reporter agent."""
    return run_reporter_agent(state)


# ============================================================================
# Routing Functions
# ============================================================================

def route_after_planner(state: PipelineState) -> Literal["human_interaction", "profiler", "end"]:
    """
    Route after planning.
    
    If there are pending questions and interactive mode, go to human interaction.
    If planning failed, go to end.
    Otherwise, proceed to profiler.
    """
    stop_reason = state.get("stop_reason")
    
    if stop_reason in ["data_load_error", "no_target"]:
        return "end"
    
    # Check if we need human interaction
    pending_questions = state.get("pending_questions", [])
    interactive_mode = state.get("interactive_mode", False)
    
    if interactive_mode and pending_questions:
        return "human_interaction"
    
    return "profiler"


def route_after_human_interaction(state: PipelineState) -> Literal["profiler", "end"]:
    """
    Route after human interaction.
    
    If planning is complete, proceed to profiler.
    """
    if state.get("stop_reason"):
        return "end"
    
    return "profiler"


def route_after_profiler(state: PipelineState) -> Literal["orchestrator", "end"]:
    """
    Route after profiling.
    
    If profiling failed (no target, data load error), go to end.
    Otherwise, proceed to orchestrator.
    """
    stop_reason = state.get("stop_reason")
    
    if stop_reason in ["data_load_error", "no_target"]:
        return "end"
    
    return "orchestrator"


def route_after_critic(state: PipelineState) -> Literal["orchestrator", "reporter"]:
    """
    Route after critic review.
    
    If there are blocking issues and we haven't reached max iterations,
    go back to orchestrator for another iteration.
    Otherwise, proceed to reporter.
    """
    should_iterate = state.get("should_iterate", False)
    
    # Route based on critic's decision. Do NOT mutate state here —
    # conditional edge functions in LangGraph don't persist state changes.
    # The critic node already incremented iteration and set should_iterate.
    if should_iterate:
        return "orchestrator"
    
    return "reporter"


# ============================================================================
# Graph Builder
# ============================================================================

def build_ml_pipeline_graph() -> StateGraph:
    """
    Build the LangGraph for the ML pipeline.
    
    Flow:
        Ingest → Planner → (Human Interaction) → Profiler → Orchestrator 
        → Cleaner → Featurizer → Modeler → Evaluator → Critic 
        → (conditional) → Reporter → End
    
    The Planner can route to Human Interaction for clarification questions.
    The Critic can route back to Orchestrator for iteration.
    
    Returns:
        Compiled StateGraph
    """
    # Create the graph with PipelineState
    graph = StateGraph(PipelineState)
    
    # Add nodes
    graph.add_node("ingest", ingest_node)
    graph.add_node("planner", planner_node)
    graph.add_node("human_interaction", human_interaction_node)
    graph.add_node("profiler", profiler_node)
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("cleaner", cleaner_node)
    graph.add_node("featurizer", featurizer_node)
    graph.add_node("modeler", modeler_node)
    graph.add_node("evaluator", evaluator_node)
    graph.add_node("critic", critic_node)
    graph.add_node("reporter", reporter_node)
    
    # Set entry point
    graph.set_entry_point("ingest")
    
    # Add edges for the main flow
    graph.add_edge("ingest", "planner")
    
    # Conditional edge after planner
    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "human_interaction": "human_interaction",
            "profiler": "profiler",
            "end": END
        }
    )
    
    # Edge after human interaction
    graph.add_conditional_edges(
        "human_interaction",
        route_after_human_interaction,
        {
            "profiler": "profiler",
            "end": END
        }
    )
    
    # Conditional edge after profiler
    graph.add_conditional_edges(
        "profiler",
        route_after_profiler,
        {
            "orchestrator": "orchestrator",
            "end": END
        }
    )
    
    # Main pipeline flow
    graph.add_edge("orchestrator", "cleaner")
    graph.add_edge("cleaner", "featurizer")
    graph.add_edge("featurizer", "modeler")
    graph.add_edge("modeler", "evaluator")
    graph.add_edge("evaluator", "critic")
    
    # Conditional edge after critic (iteration or finish)
    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "orchestrator": "orchestrator",
            "reporter": "reporter"
        }
    )
    
    # Reporter goes to end
    graph.add_edge("reporter", END)
    
    return graph


def compile_pipeline():
    """
    Compile the ML pipeline graph.
    
    Returns:
        Compiled graph ready for execution
    """
    graph = build_ml_pipeline_graph()
    return graph.compile()


# ============================================================================
# Pipeline Execution
# ============================================================================

def run_pipeline(initial_state: PipelineState) -> PipelineState:
    """
    Run the complete ML pipeline.
    
    Args:
        initial_state: Initial pipeline state with configuration
    
    Returns:
        Final pipeline state with all results
    """
    # Compile the graph
    pipeline = compile_pipeline()
    
    # Run the pipeline
    final_state = pipeline.invoke(initial_state)
    
    return final_state


def run_pipeline_with_streaming(initial_state: PipelineState):
    """
    Run the pipeline with streaming output.
    
    Yields state updates as the pipeline progresses.
    
    Args:
        initial_state: Initial pipeline state
    
    Yields:
        State updates from each node
    """
    pipeline = compile_pipeline()
    
    for output in pipeline.stream(initial_state):
        yield output
