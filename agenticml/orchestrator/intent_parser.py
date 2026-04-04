"""
Pure LLM-based intent parser.

Every user query is sent to the LLM with response_format=json_object.
There are NO keyword fallbacks — the LLM is the sole decision-maker.
"""

from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage

from agenticml.services.llm_service import get_llm, invoke_llm_json
from agenticml.ml.config import get_config

INTENT_SYSTEM_PROMPT = """You are an intent classifier for an automated machine learning system.

Given a user query and the current workflow state, classify the user's intent
into exactly one of these categories:

  profile          – Explore / describe / summarise the dataset (includes showing
                     null values, column types, distributions, etc.)
  preprocess       – Clean the data (handle missing values, duplicates, outliers)
  feature_engineer – Feature engineering, encoding, scaling, train/test split
  select_models    – Choose which ML models to try
  train            – Train the selected models
  evaluate         – Evaluate trained models, compare metrics, pick the best
  visualize        – Generate insights, plots, reports, or show results
  full_pipeline    – Run the complete end-to-end ML pipeline

You must respond with a JSON object with the following schema:

{
    "intent": "<one of the categories above>",
    "confidence": <float 0-1>,
    "focus": "<optional sub-focus, e.g. 'missing_values', 'metrics', or null>",
    "model_filter": ["<optional model names to restrict to, or null>"],
    "reasoning": "<one sentence explaining why you chose this intent>"
}

Rules:
- If the query is ambiguous, pick the most likely intent and explain in reasoning.
- If the user asks to "run everything" or similar, use "full_pipeline".
- "focus" should be null unless the query targets a specific aspect.
- "model_filter" should be null unless the user names specific models.
"""


def parse_intent(
    user_query: str,
    state_summary: dict,
    verbose: bool = False,
) -> dict:
    """
    Classify a user query into a structured intent using the LLM.

    Args:
        user_query: Raw natural-language query from the user.
        state_summary: Compact summary of the current WorkflowState
                       (which steps are done, what artifacts exist, etc.).
        verbose: Print prompt/response when True.

    Returns:
        Parsed intent dict with keys: intent, confidence, focus,
        model_filter, reasoning.
    """
    config = get_config()
    llm = get_llm(config)

    prompt = (
        f"User query: {user_query}\n\n"
        f"Current workflow state summary:\n{_format_state_summary(state_summary)}\n\n"
        "Classify the intent."
    )

    messages = [
        SystemMessage(content=INTENT_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    return invoke_llm_json(
        llm, messages,
        agent_name="IntentParser",
        step_description="Classify user intent",
        verbose=verbose,
    )


def _format_state_summary(summary: dict) -> str:
    """Produce a concise text representation of the state for the LLM."""
    lines = []
    for key in (
        "has_dataset", "dataset_profiled", "data_cleaned",
        "features_engineered", "models_selected", "models_trained",
        "models_evaluated", "insights_generated",
    ):
        val = summary.get(key)
        if val is not None:
            lines.append(f"- {key}: {val}")

    extra = summary.get("extra")
    if extra:
        lines.append(f"- extra context: {extra}")

    return "\n".join(lines) if lines else "No prior workflow state."
