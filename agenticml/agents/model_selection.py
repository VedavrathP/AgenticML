"""
Model Selection Agent.

Analyses data characteristics and asks the LLM to select 3-6 model
candidates.  Builds model configs from the selected names and ensures
the baseline model is first.  Does NOT train models -- that is the
responsibility of a separate training agent.

All LLM calls go through ``invoke_llm_json`` -- no free-text parsing or
silent fallbacks.
"""

import json
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage

from agenticml.agents.base_agent import BaseAgent
from agenticml.ml.config import (
    get_config,
    CLASSIFICATION_MODELS,
    REGRESSION_MODELS,
)
from agenticml.ml.tools.modeling import (
    get_model_candidates,
    suggest_models_for_data,
    get_baseline_model,
)
from agenticml.ml.tools.utils import safe_json_serialize
from agenticml.services.llm_service import get_llm, invoke_llm_json
from agenticml.state.workflow_state import (
    WorkflowState,
    log_decision,
    add_error,
    record_execution,
)


SYSTEM_PROMPT = """\
You are a Model Selection Agent in an ML pipeline.

You receive dataset characteristics and a list of available models.
Your job is to select 3-6 model candidates that are well-suited to the
data.

You must respond with a JSON object with exactly this schema:

{
    "selected_models": [
        {
            "name": "<ModelClassName>",
            "rationale": "<why this model is a good fit>",
            "params": {}
        }
    ],
    "overall_rationale": "<one-paragraph summary of selection strategy>"
}

Available models for classification:
  LogisticRegression (baseline), RandomForestClassifier,
  GradientBoostingClassifier, SVC, KNeighborsClassifier,
  DecisionTreeClassifier, XGBClassifier*, LGBMClassifier*
  (* = only if available)

Available models for regression:
  LinearRegression (baseline), Ridge, RandomForestRegressor,
  GradientBoostingRegressor, SVR, DecisionTreeRegressor,
  XGBRegressor*, LGBMRegressor*

Rules:
- Always include the baseline model.
- Select between 3 and 6 models total.
- Avoid SVM variants for datasets with >50 000 samples.
- Include at least one ensemble method.
- The "name" field MUST exactly match one of the class names listed above.
- The "params" field may override default hyper-parameters (or be empty {}).
"""


class ModelSelectionAgent(BaseAgent):
    """Select model candidates via LLM (does not train them)."""

    name = "model_selection"

    def run(self, state: WorkflowState) -> WorkflowState:
        config = get_config()
        problem_type = state["problem_type"]
        verbose = state.get("verbose", False)

        # ==================================================================
        # 1. Gather data characteristics
        # ==================================================================
        data_summary = state.get("data_summary", {})
        n_samples = state.get("n_train_samples", data_summary.get("n_rows", 0))
        n_features = state.get("n_features", data_summary.get("n_cols", 0))

        # Deterministic suggestions (context for LLM, not a fallback)
        base_candidates = suggest_models_for_data(
            profile=data_summary,
            problem_type=problem_type,
            max_models=config.max_models,
        )

        available_models = list(
            CLASSIFICATION_MODELS.keys()
            if problem_type == "classification"
            else REGRESSION_MODELS.keys()
        )

        # ==================================================================
        # 2. Ask LLM to select models
        # ==================================================================
        llm_decision = self._ask_llm(
            problem_type=problem_type,
            n_samples=n_samples,
            n_features=n_features,
            data_summary=data_summary,
            available_models=available_models,
            base_candidates=base_candidates,
            max_models=config.max_models,
            config=config,
            verbose=verbose,
        )

        selected_names = [
            m["name"] for m in llm_decision.get("selected_models", [])
        ]
        llm_params_map = {
            m["name"]: m.get("params", {})
            for m in llm_decision.get("selected_models", [])
        }

        # ==================================================================
        # 3. Build model configs & ensure baseline first
        # ==================================================================
        model_configs = _build_model_configs(
            selected_names, problem_type, llm_params_map
        )
        model_configs = _ensure_baseline_first(model_configs, problem_type)
        model_configs = model_configs[: config.max_models]

        state["selected_models"] = safe_json_serialize(model_configs)

        log_decision(
            state, self.name,
            f"Selected {len(model_configs)} model candidates",
            llm_decision.get("overall_rationale", ""),
            {"models": [m["name"] for m in model_configs]},
        )

        record_execution(state, self.name)
        return state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ask_llm(
        self,
        *,
        problem_type: str,
        n_samples: int,
        n_features: int,
        data_summary: dict,
        available_models: list[str],
        base_candidates: list[dict],
        max_models: int,
        config: Any,
        verbose: bool,
    ) -> dict:
        llm = get_llm(config)

        context = {
            "problem_type": problem_type,
            "n_samples": n_samples,
            "n_features": n_features,
            "n_numeric": len(data_summary.get("numeric_columns", [])),
            "n_categorical": len(data_summary.get("categorical_columns", [])),
            "available_models": available_models,
            "max_models": max_models,
        }

        prompt = (
            "Select models for this ML task.\n\n"
            f"Context:\n{json.dumps(context, indent=2)}\n\n"
            "Deterministic suggestions:\n"
            f"{json.dumps([{'name': m['name'], 'rationale': m.get('rationale', '')} for m in base_candidates], indent=2)}\n\n"
            "Requirements:\n"
            "1. Include the baseline model first.\n"
            f"2. Select 3-{max_models} models total.\n"
            f"3. Dataset has {n_samples} samples -- avoid SVM for >50k.\n"
            "4. Include at least one ensemble method.\n\n"
            "Respond with the JSON object described in the system prompt."
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        return invoke_llm_json(
            llm,
            messages,
            agent_name=self.name,
            step_description="Model selection",
            verbose=verbose,
        )


# ======================================================================
# Module-level helpers (ported from old modeler.py)
# ======================================================================

def _build_model_configs(
    model_names: list[str],
    problem_type: str,
    params_overrides: dict[str, dict] | None = None,
) -> list[dict]:
    """Build full model config dicts from a list of model class names."""
    models_dict = (
        CLASSIFICATION_MODELS
        if problem_type == "classification"
        else REGRESSION_MODELS
    )
    params_overrides = params_overrides or {}

    configs: list[dict] = []
    for name in model_names:
        if name not in models_dict:
            continue
        info = models_dict[name]
        merged_params = {**info["default_params"], **params_overrides.get(name, {})}
        configs.append({
            "name": name,
            "model_type": info["class"],
            "module": info["module"],
            "params": merged_params,
            "is_baseline": info.get("is_baseline", False),
            "complexity": info.get("complexity", "medium"),
        })
    return configs


def _ensure_baseline_first(
    model_configs: list[dict],
    problem_type: str,
) -> list[dict]:
    """Move the baseline model to position 0, adding it if absent."""
    baseline_idx = None
    for i, cfg in enumerate(model_configs):
        if cfg.get("is_baseline"):
            baseline_idx = i
            break

    if baseline_idx is not None and baseline_idx > 0:
        baseline = model_configs.pop(baseline_idx)
        model_configs.insert(0, baseline)
    elif baseline_idx is None:
        baseline = get_baseline_model(problem_type)
        model_configs.insert(0, baseline)

    return model_configs
