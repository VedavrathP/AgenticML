"""
Orchestrator Agent — the single entry point for backend execution.

Receives the user's raw natural-language query + current WorkflowState,
uses a single LLM call to understand the user's goal and decide which
specialised agent to invoke next (or whether to finish).
"""

from langchain_core.messages import SystemMessage, HumanMessage

from agenticml.state.workflow_state import (
    WorkflowState,
    log_decision,
    record_execution,
)
from agenticml.services.llm_service import get_llm, invoke_llm_json
from agenticml.ml.config import get_config
from agenticml.ml.tools.utils import safe_json_serialize

FULL_PIPELINE_ORDER = [
    "dataset_profiling",
    "data_preprocessing",
    "feature_engineering",
    "model_selection",
    "model_training",
    "evaluation",
    "insight_visualization",
]

ORCHESTRATOR_SYSTEM_PROMPT = """\
You are the Orchestrator of an automated machine learning system.
Think like a human ML practitioner: train, check the score, decide whether
to keep experimenting or move on.

You receive the user's raw request and a summary of the current workflow state.
Your job is to **understand what the user wants** and decide **which agent to
invoke next**, or whether the workflow is **done**.

Available agents (use these exact names):
  dataset_profiling       – Load, explore, and summarise the dataset
  data_preprocessing      – Clean data (missing values, duplicates, outliers, correlated columns)
  feature_engineering     – Feature transforms, scaling, encoding, train/test split
  model_selection         – Choose which ML models to try
  model_training          – Train the selected models
  evaluation              – Evaluate models, compare metrics, pick the best
  insight_visualization   – Generate final report, plots, and insights

You must respond with a JSON object:

{
    "next_agent": "<agent name or 'done'>",
    "reasoning": "<why this agent is the right next step, referencing the user's request>",
    "user_goal_summary": "<1-sentence summary of what the user is asking for>"
}

Rules:
- Read the user's query carefully. Decide which part of the ML pipeline their
  request maps to.
- If the user asks to run everything, run the full pipeline, or gives a broad
  request like "build me a model", treat it as a full pipeline run and route
  to the first incomplete step.
- If the user's request requires a step whose prerequisites are missing (e.g.
  they ask to train but data hasn't been profiled/cleaned yet), route to the
  earliest missing prerequisite and explain in reasoning.
- The pipeline order is:
  dataset_profiling -> data_preprocessing -> feature_engineering ->
  model_selection -> model_training -> evaluation -> insight_visualization.
  Never skip a prerequisite.
- If all work required by the user's request is already done, return "done".
- Always ground your reasoning in the user's actual words.

Retrain / experimentation rules:
- Check "retrain_history" in the workflow state. It shows every retrain
  attempt with its score.
- If the last retrain attempt did NOT improve the score compared to the
  previous attempt, stop experimenting — route to "evaluation" (for a full
  evaluation) or "done".  Do NOT send the model back to model_training.
- If a model has been retrained 5 times (check the attempt count), stop
  experimenting regardless of whether the score improved.
- If the retrain DID improve the score AND the attempt count is below 5,
  you MAY route to model_training for one more attempt — but explain why
  in your reasoning.
- After model_training completes a retrain, the next logical step is
  "evaluation" (to get a full evaluation with plots), NOT model_training
  again, unless the score improved and you have a clear reason to try more.

General anti-loop rules:
- Check "last_completed_agent". If it already fulfilled the user's request,
  route to the logical next pipeline step or "done".
- Never route to the same agent more than twice in a row.
"""


def _build_state_summary(state: WorkflowState) -> dict:
    """Build a compact summary of what has been completed."""
    history = state.get("execution_history", [])
    completed = {e["agent"] for e in history if e.get("status") == "completed"}

    last_agent = None
    last_action = None
    non_orchestrator = [
        e for e in history
        if e.get("status") == "completed" and e.get("agent") != "orchestrator"
    ]
    if non_orchestrator:
        last_entry = non_orchestrator[-1]
        last_agent = last_entry["agent"]
        last_action = _summarize_last_action(last_agent, state)

    trained = state.get("trained_models", [])
    trained_summary = [
        {"name": m.get("name"), "params": m.get("config", {}).get("params", {})}
        for m in trained if m.get("success")
    ]

    eval_results = state.get("evaluation_results", [])
    eval_summary = [
        {
            "name": r.get("name"),
            "primary_metric": r.get("primary_metric"),
            "primary_score": r.get("primary_score"),
        }
        for r in eval_results if r.get("success")
    ]

    best_model_info = state.get("best_model") or {}
    retrain_history = state.get("retrain_history", [])

    summary: dict = {
        "has_dataset": bool(state.get("file_path")),
        "dataset_profiled": "dataset_profiling" in completed,
        "data_cleaned": "data_preprocessing" in completed,
        "features_engineered": "feature_engineering" in completed,
        "models_selected": "model_selection" in completed,
        "models_trained": "model_training" in completed,
        "models_evaluated": "evaluation" in completed,
        "insights_generated": "insight_visualization" in completed,
        "completed_agents": sorted(completed),
        "target": state.get("target"),
        "problem_type": state.get("problem_type"),
        "best_model": best_model_info.get("name"),
        "best_score": best_model_info.get("primary_score"),
        "n_trained_models": len(trained),
        "trained_models_summary": trained_summary,
        "evaluation_results_summary": eval_summary,
        "retrain_history": retrain_history,
        "n_errors": len(state.get("errors", [])),
    }

    if last_agent:
        summary["last_completed_agent"] = last_agent
    if last_action:
        summary["last_action_summary"] = last_action

    return summary


def _summarize_last_action(agent_name: str, state: WorkflowState) -> str:
    """Produce a human-readable summary of what the last agent did."""
    decision_log = state.get("decision_log", [])
    agent_decisions = [d for d in decision_log if d.get("agent") == agent_name]
    if not agent_decisions:
        return f"{agent_name} completed"

    last = agent_decisions[-1]
    return f"{last.get('decision', agent_name + ' completed')}"


def _next_pipeline_step(current_agent: str) -> str:
    """Return the agent that follows *current_agent* in the pipeline, or 'done'."""
    try:
        idx = FULL_PIPELINE_ORDER.index(current_agent)
        if idx + 1 < len(FULL_PIPELINE_ORDER):
            return FULL_PIPELINE_ORDER[idx + 1]
    except ValueError:
        pass
    return "done"


def _consecutive_runs(state: WorkflowState, agent_name: str) -> int:
    """Count how many times *agent_name* ran consecutively (most recent streak)."""
    history = state.get("execution_history", [])
    non_orch = [
        e["agent"] for e in history
        if e.get("status") == "completed" and e.get("agent") != "orchestrator"
    ]
    count = 0
    for name in reversed(non_orch):
        if name == agent_name:
            count += 1
        else:
            break
    return count


_MAX_CONSECUTIVE_SAME_AGENT = 5


def run_orchestrator(state: WorkflowState) -> WorkflowState:
    """
    Orchestrator node function called by the LangGraph.

    Single LLM call: reads the user's raw query + workflow state summary
    (including retrain_history with scores), understands the goal, and
    decides which agent to invoke next.

    Safety net: if the LLM routes to the same agent more than
    _MAX_CONSECUTIVE_SAME_AGENT times, force-advance to the next step.
    """
    config = get_config()
    verbose = state.get("verbose", False)
    user_query = state.get("user_query", "")
    state_summary = _build_state_summary(state)

    llm = get_llm(config)

    prompt = (
        f"User's request:\n\"{user_query}\"\n\n"
        f"Current workflow state:\n{safe_json_serialize(state_summary)}\n\n"
        "Understand what the user wants and decide which agent to invoke next "
        "(or 'done' if the request is already fulfilled)."
    )

    messages = [
        SystemMessage(content=ORCHESTRATOR_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    decision = invoke_llm_json(
        llm, messages,
        agent_name="Orchestrator",
        step_description="Understand query & route",
        verbose=verbose,
    )

    next_agent = decision.get("next_agent", "done")

    valid_agents = set(FULL_PIPELINE_ORDER) | {"done"}
    if next_agent not in valid_agents:
        next_agent = "done"

    # ── Safety net: cap consecutive runs of the same agent ─────────────
    if next_agent != "done":
        consecutive = _consecutive_runs(state, next_agent)
        if consecutive >= _MAX_CONSECUTIVE_SAME_AGENT:
            overridden_to = _next_pipeline_step(next_agent)
            log_decision(
                state, "orchestrator",
                f"Safety net: '{next_agent}' ran {consecutive} times consecutively "
                f"— forcing advance to '{overridden_to}'",
                f"Max consecutive limit ({_MAX_CONSECUTIVE_SAME_AGENT}) reached",
            )
            next_agent = overridden_to

    state["next_agent"] = next_agent
    state["current_phase"] = next_agent if next_agent != "done" else None

    goal_summary = decision.get("user_goal_summary", user_query)
    state["user_intent"] = {"goal": goal_summary}

    log_decision(
        state, "orchestrator",
        f"Routing to: {next_agent}",
        decision.get("reasoning", ""),
        decision,
    )

    record_execution(state, "orchestrator")
    return state
