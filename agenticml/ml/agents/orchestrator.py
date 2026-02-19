"""
Orchestrator Agent

Responsible for:
- Owning the global plan
- Deciding next step based on state
- Controlling iteration loop
- Stopping when quality/budget satisfied
"""

import json
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage

from agenticml.ml.state import PipelineState, log_decision, add_error
from agenticml.ml.config import get_config
from agenticml.ml.tools.utils import safe_json_serialize
from agenticml.ml.tools.llm import invoke_llm
from agenticml.ml.tools.llm_factory import create_llm


ORCHESTRATOR_SYSTEM_PROMPT = """You are the Orchestrator Agent in an ML pipeline.

Your role is to:
1. Oversee the entire ML workflow
2. Make high-level decisions about pipeline direction
3. Determine when to stop iterating
4. Ensure quality standards are met

You receive the current pipeline state and must decide:
- Whether to continue or stop
- What the focus should be for the next iteration
- Any adjustments to the strategy

Respond in JSON format:
{
    "decision": {
        "action": "continue|stop",
        "rationale": "explanation",
        "focus_areas": ["area1", "area2"],
        "strategy_adjustments": ["adjustment1"]
    }
}
"""


def run_orchestrator_agent(state: PipelineState) -> PipelineState:
    """
    Run the orchestrator agent to control pipeline flow.
    
    This agent:
    1. Reviews current state
    2. Decides on next actions
    3. Updates iteration strategy
    
    Args:
        state: Current pipeline state
    
    Returns:
        Updated pipeline state
    """
    config = get_config()
    iteration = state.get("iteration", 0)
    
    # Log orchestrator activation
    log_decision(
        state, "orchestrator",
        f"Starting iteration {iteration + 1}",
        f"Max iterations: {state.get('max_iterations', config.max_iterations)}",
        {"iteration": iteration + 1}
    )
    
    # On first iteration, just set up and proceed
    if iteration == 0:
        log_decision(
            state, "orchestrator",
            "Initial pipeline setup",
            "Proceeding with standard workflow: Profile → Clean → Feature → Model → Evaluate → Critic"
        )
        return state
    
    # On subsequent iterations, analyze what needs to change
    critic_issues = state.get("critic_issues", [])
    blocking_issues = [i for i in critic_issues if i.get("severity") == "blocking"]
    
    # Get LLM guidance for iteration focus
    llm_decision = _get_llm_orchestration_decision(state, config, verbose=state.get("verbose", False))
    
    if llm_decision:
        focus_areas = llm_decision.get("focus_areas", [])
        strategy_adjustments = llm_decision.get("strategy_adjustments", [])
        
        if focus_areas:
            log_decision(
                state, "orchestrator",
                f"Iteration {iteration + 1} focus: {', '.join(focus_areas)}",
                llm_decision.get("rationale", ""),
                {"focus_areas": focus_areas, "adjustments": strategy_adjustments}
            )
        
        # Store iteration guidance in state
        state["iteration_guidance"] = {
            "focus_areas": focus_areas,
            "strategy_adjustments": strategy_adjustments,
            "blocking_issues": [i["description"] for i in blocking_issues]
        }
    else:
        # Default focus based on blocking issues
        focus_areas = list(set(i.get("affected_component", "unknown") for i in blocking_issues))
        
        log_decision(
            state, "orchestrator",
            f"Iteration {iteration + 1} focusing on: {', '.join(focus_areas) or 'general improvement'}",
            f"Addressing {len(blocking_issues)} blocking issues"
        )
        
        state["iteration_guidance"] = {
            "focus_areas": focus_areas,
            "blocking_issues": [i["description"] for i in blocking_issues]
        }
    
    return state


def _get_llm_orchestration_decision(state: PipelineState, config: Any, verbose: bool = False) -> dict:
    """Get LLM guidance for orchestration decisions."""
    if not config.llm_api_key:
        return {}
    
    try:
        llm = create_llm(config)
        
        # Summarize state
        summary = {
            "iteration": state.get("iteration", 0),
            "max_iterations": state.get("max_iterations", config.max_iterations),
            "problem_type": state.get("problem_type"),
            "best_model": state.get("best_model", {}).get("name"),
            "best_score": state.get("best_model", {}).get("primary_score"),
            "critic_issues": [
                {"severity": i.get("severity"), "category": i.get("category"), "description": i.get("description")}
                for i in state.get("critic_issues", [])
            ],
            "decision_log_summary": [
                {"agent": d.get("agent"), "decision": d.get("decision")}
                for d in state.get("decision_log", [])[-10:]  # Last 10 decisions
            ]
        }
        
        prompt = f"""Review the pipeline state and provide guidance for the next iteration.

Current State:
{json.dumps(summary, indent=2)}

Determine:
1. What areas need focus in this iteration
2. Any strategy adjustments needed
3. Whether the pipeline is on track"""

        messages = [
            SystemMessage(content=ORCHESTRATOR_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]
        
        response = invoke_llm(llm, messages, "Orchestrator", "Iteration guidance", verbose)
        
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        
        result = json.loads(content)
        return result.get("decision", result)
    
    except Exception:
        return {}


def should_continue_pipeline(state: PipelineState) -> bool:
    """
    Determine if the pipeline should continue to another iteration.
    
    Args:
        state: Current pipeline state
    
    Returns:
        True if should continue, False if should stop
    """
    config = get_config()
    
    # Check for stop conditions
    stop_reason = state.get("stop_reason")
    if stop_reason:
        return False
    
    # Check iteration limits
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", config.max_iterations)
    
    if iteration >= max_iterations:
        return False
    
    # Check for blocking issues (should iterate)
    has_blocking = state.get("has_blocking_issues", False)
    if has_blocking and iteration < max_iterations:
        return True
    
    # Check minimum iterations
    if iteration < config.min_iterations:
        return True
    
    return False


def get_next_step(state: PipelineState) -> str:
    """
    Determine the next step in the pipeline based on current state.
    
    This is used by the LangGraph routing logic.
    
    Args:
        state: Current pipeline state
    
    Returns:
        Name of the next node to execute
    """
    # Check for errors
    if state.get("stop_reason") in ["data_load_error", "no_target"]:
        return "end"
    
    # Check if we should iterate
    has_blocking = state.get("has_blocking_issues", False)
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", 5)
    
    if has_blocking and iteration < max_iterations:
        return "orchestrator"  # Go back for another iteration
    
    # Otherwise proceed to reporter
    return "reporter"
