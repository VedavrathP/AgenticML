"""
ModelTrainingAgent – trains model candidates using per-algorithm tools.

The LLM decides **which** tool(s) to call and **what hyperparameters** to use.
When the user asks to retrain a specific model with different parameters, only
that model's tool is invoked — the rest of the already-trained models are kept.

Flow:
1. Load training data.
2. Build the tool schema for the current problem type.
3. Ask the LLM which tools to call (with params), honouring the user query,
   the ``model_filter`` from the intent parser, and the current state.
4. Execute only the requested tool calls.
5. Merge new results into ``state["trained_models"]``, replacing entries for
   models that were retrained and keeping the rest untouched.
"""

import os
import json
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage

from agenticml.agents.base_agent import BaseAgent
from agenticml.services.llm_service import get_llm, invoke_llm_json
from agenticml.state.workflow_state import (
    WorkflowState,
    log_decision,
    add_artifact,
    add_error,
    record_execution,
)
from agenticml.ml.config import get_config
from agenticml.ml.tools.data_io import load_dataframe
from agenticml.ml.tools.model_trainers import TOOL_REGISTRY, get_tool_schemas
from agenticml.ml.tools.modeling import get_model_info, get_baseline_model
from agenticml.ml.tools.evaluation import evaluate_model
from agenticml.services.artifact_service import save_model, load_model, save_json
from agenticml.ml.tools.utils import get_run_subdir, safe_json_serialize, format_duration


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

TOOL_CALL_SYSTEM_PROMPT = """\
You are the Model Training Agent in an automated ML pipeline.

You have access to per-algorithm training tools.  Each tool trains exactly
ONE model.  You must decide which tool(s) to call and with what
hyperparameters.

You must respond with a JSON object:

{{
    "tool_calls": [
        {{
            "tool_name": "<name from available tools>",
            "params": {{ <hyperparameters — omit any you want to keep at default> }}
        }}
    ],
    "rationale": "<why you chose these tools and parameters>"
}}

Available tools and their parameter schemas:
{tool_schemas}

Rules:
- If the user asked to retrain a SPECIFIC model (e.g. "try RandomForest with
  more trees"), call ONLY that model's tool with the requested changes.
- If the user asked for a full training run (or this is the first training),
  call the tools for ALL selected models.
- When the user hints at parameter changes ("more trees", "higher learning
  rate", "deeper trees"), translate those hints into concrete parameter values.
- Always include reasonable defaults for parameters you don't change.
- The "params" dict must only contain keys that exist in the tool's schema.
- Do NOT call tools for models that are not in the selected_models list unless
  the user explicitly asks for a new model.
- The "already_trained_models" list shows each model's previously used
  hyperparameters.  When the user asks to try different params, you MUST
  choose values that DIFFER from what was already tried.  Do not repeat the
  same hyperparameter combination.
"""

CLASS_WEIGHT_SYSTEM_PROMPT = (
    "You are a class-imbalance advisor for an automated ML pipeline. "
    "You must respond with a JSON object.\n\n"
    "Given the dataset profile and the list of models about to be trained, "
    "decide whether to apply class_weight='balanced' and to which models.\n\n"
    "Return exactly:\n"
    "{\n"
    '    "use_class_weight": true | false,\n'
    '    "rationale": "...",\n'
    '    "models_to_apply": ["ModelName", ...]\n'
    "}"
)


# ---------------------------------------------------------------------------
# Name ↔ tool mapping
# ---------------------------------------------------------------------------

_MODEL_NAME_TO_TOOL: dict[str, dict[str, str]] = {
    # classification
    "LogisticRegression":          {"tool": "train_logistic_regression",          "type": "classification"},
    "RandomForestClassifier":      {"tool": "train_random_forest_classifier",     "type": "classification"},
    "GradientBoostingClassifier":  {"tool": "train_gradient_boosting_classifier", "type": "classification"},
    "SVC":                         {"tool": "train_svc",                          "type": "classification"},
    "KNeighborsClassifier":        {"tool": "train_kneighbors_classifier",        "type": "classification"},
    "DecisionTreeClassifier":      {"tool": "train_decision_tree_classifier",     "type": "classification"},
    "XGBClassifier":               {"tool": "train_xgb_classifier",              "type": "classification"},
    "LGBMClassifier":              {"tool": "train_lgbm_classifier",             "type": "classification"},
    # regression
    "LinearRegression":            {"tool": "train_linear_regression",            "type": "regression"},
    "Ridge":                       {"tool": "train_ridge",                        "type": "regression"},
    "RandomForestRegressor":       {"tool": "train_random_forest_regressor",      "type": "regression"},
    "GradientBoostingRegressor":   {"tool": "train_gradient_boosting_regressor",  "type": "regression"},
    "SVR":                         {"tool": "train_svr",                          "type": "regression"},
    "DecisionTreeRegressor":       {"tool": "train_decision_tree_regressor",      "type": "regression"},
    "XGBRegressor":                {"tool": "train_xgb_regressor",               "type": "regression"},
    "LGBMRegressor":               {"tool": "train_lgbm_regressor",              "type": "regression"},
}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class ModelTrainingAgent(BaseAgent):
    """Trains model candidates using per-algorithm tools selected by the LLM."""

    name: str = "model_training"

    def run(self, state: WorkflowState) -> WorkflowState:
        config = get_config()
        run_dir = state["run_dir"]
        problem_type = state["problem_type"]
        target = state["target"]

        # ── 1. Load training data ─────────────────────────────────────
        train_path = state.get("train_data_path")
        if not train_path:
            add_error(state, self.name, "No training data path in state")
            record_execution(state, self.name, status="failed")
            return state

        try:
            train_df = load_dataframe(train_path)
            X_train = train_df.drop(columns=[target]).values
            y_train = train_df[target].values
        except Exception as exc:
            add_error(state, self.name, f"Failed to load training data: {exc}")
            record_execution(state, self.name, status="failed")
            return state

        n_samples, n_features = X_train.shape

        # ── 2. Retrieve selected models ───────────────────────────────
        model_configs: list[dict] = state.get("selected_models", [])
        if not model_configs:
            add_error(state, self.name, "No selected_models in state – run ModelSelectionAgent first")
            record_execution(state, self.name, status="failed")
            return state

        # ── 3. Build tool schemas for the LLM ─────────────────────────
        tool_schemas = get_tool_schemas(problem_type)

        # ── 4. Ask LLM which tools to call ────────────────────────────
        tool_calls = self._decide_tool_calls(
            state, config, model_configs, tool_schemas, n_samples, n_features,
        )

        if not tool_calls:
            tool_calls = self._fallback_all_models(model_configs, problem_type)

        # ── 5. Decide class-weight strategy (classification only) ─────
        cw_models: list[str] = []
        if problem_type == "classification" and not state.get("smote_applied", False):
            cw_decision = self._decide_class_weight(
                state, config, tool_calls, n_samples, n_features,
            )
            if cw_decision.get("use_class_weight", False):
                cw_models = cw_decision.get("models_to_apply", [])
            log_decision(
                state, self.name,
                "Class-weight decision",
                cw_decision.get("rationale", ""),
                {"use_class_weight": bool(cw_models), "models_to_apply": cw_models},
            )

        # ── 6. Execute tool calls ─────────────────────────────────────
        models_dir = get_run_subdir(run_dir, "models")
        new_results: list[dict] = []

        for tc in tool_calls:
            tool_name = tc["tool_name"]
            params = tc.get("params", {})

            fn = TOOL_REGISTRY.get(tool_name)
            if fn is None:
                log_decision(state, self.name, f"Unknown tool: {tool_name}", "Skipped")
                continue

            if cw_models:
                result_model_name = _tool_name_to_model_name(tool_name)
                if result_model_name in cw_models and "class_weight" not in params:
                    params["class_weight"] = "balanced"

            log_decision(
                state, self.name,
                f"Calling {tool_name}",
                f"Params: {json.dumps(params, default=str)}",
                {"tool_name": tool_name, "params": params},
            )

            try:
                result = fn(X_train, y_train, **params)

                model_name = result["name"]
                if result["success"]:
                    model_path = os.path.join(models_dir, f"{model_name}.joblib")
                    save_model(result["model"], model_path, metadata={
                        "name": model_name,
                        "params_used": result["params_used"],
                        "training_time": result["training_time"],
                    })

                    model_info = get_model_info(result["model"])

                    new_results.append({
                        "name": model_name,
                        "config": {
                            "name": model_name,
                            "params": result["params_used"],
                        },
                        "model_path": model_path,
                        "training_time": result["training_time"],
                        "is_baseline": _is_baseline(model_name, model_configs),
                        "success": True,
                        "model_info": model_info,
                    })

                    add_artifact(state, f"model_{model_name}", model_path, "model")
                    log_decision(
                        state, self.name,
                        f"Trained {model_name} in {format_duration(result['training_time'])}",
                        "Training successful",
                        {"training_time": result["training_time"], "params": result["params_used"]},
                    )
                else:
                    new_results.append({
                        "name": model_name,
                        "config": {"name": model_name, "params": result["params_used"]},
                        "success": False,
                        "error": result.get("error", "Unknown error"),
                    })
                    log_decision(state, self.name, f"Failed to train {model_name}", result.get("error", ""))

            except Exception as exc:
                log_decision(state, self.name, f"Error executing {tool_name}", str(exc))
                new_results.append({
                    "name": _tool_name_to_model_name(tool_name),
                    "success": False,
                    "error": str(exc),
                })

        # ── 7. Merge results: replace retrained, keep the rest ────────
        existing = state.get("trained_models", [])
        retrained_names = {r["name"] for r in new_results}
        merged = [m for m in existing if m["name"] not in retrained_names] + new_results
        state["trained_models"] = safe_json_serialize(merged)

        is_retrain = len(existing) > 0 and len(new_results) < len(model_configs)

        # ── 8. Inline quick-evaluation for retrained models ───────────
        if is_retrain:
            self._inline_evaluate_retrained(
                state, new_results, target, problem_type,
            )

        # ── 9. Persist training summary ───────────────────────────────
        successful = [m for m in merged if m.get("success")]
        total_time = sum(m.get("training_time", 0) for m in successful)

        training_summary = {
            "total_models": len(merged),
            "successful": len(successful),
            "failed": len(merged) - len(successful),
            "total_training_time": total_time,
            "newly_trained": [r["name"] for r in new_results],
            "kept_from_previous": [m["name"] for m in existing if m["name"] not in retrained_names],
            "is_retrain": is_retrain,
            "models": merged,
        }

        summary_path = os.path.join(models_dir, "training_summary.json")
        save_json(summary_path, training_summary)
        add_artifact(state, "training_summary", summary_path, "json")

        log_decision(
            state, self.name,
            f"Training complete: {len(new_results)} trained this round, {len(merged)} total",
            f"Total training time: {format_duration(total_time)}",
            training_summary,
        )

        record_execution(state, self.name)
        return state

    # ------------------------------------------------------------------
    # LLM call: decide which tools to call
    # ------------------------------------------------------------------
    def _decide_tool_calls(
        self,
        state: WorkflowState,
        config: Any,
        model_configs: list[dict],
        tool_schemas: list[dict],
        n_samples: int,
        n_features: int,
    ) -> list[dict]:
        llm = get_llm(config)

        user_query = state.get("user_query", "")
        user_intent = state.get("user_intent", {}) or {}
        model_filter = user_intent.get("model_filter") or []
        already_trained = [
            {
                "name": m.get("name"),
                "params": m.get("config", {}).get("params", {}),
            }
            for m in state.get("trained_models", [])
            if m.get("success")
        ]
        selected_model_names = [m["name"] for m in model_configs]

        context = json.dumps({
            "user_query": user_query,
            "model_filter": model_filter,
            "problem_type": state.get("problem_type"),
            "n_samples": n_samples,
            "n_features": n_features,
            "selected_models": selected_model_names,
            "already_trained_models": already_trained,
            "smote_applied": state.get("smote_applied", False),
        }, indent=2)

        schemas_text = json.dumps(tool_schemas, indent=2, default=str)

        prompt = (
            f"Context:\n{context}\n\n"
            "Decide which training tools to call and with what hyperparameters.\n\n"
            "If the user is asking to retrain a specific model with different "
            "parameters, call ONLY that model's tool.\n"
            "If this is a first-time full training run, call tools for all "
            "selected models.\n"
            "Translate any user hints about parameters into concrete values."
        )

        filled_system = TOOL_CALL_SYSTEM_PROMPT.format(tool_schemas=schemas_text)

        messages = [
            SystemMessage(content=filled_system),
            HumanMessage(content=prompt),
        ]

        decision = invoke_llm_json(
            llm, messages,
            agent_name=self.name,
            step_description="Decide tool calls",
            verbose=state.get("verbose", False),
        )

        log_decision(
            state, self.name,
            "LLM tool-call decision",
            decision.get("rationale", ""),
            decision,
        )

        return decision.get("tool_calls", [])

    # ------------------------------------------------------------------
    # LLM call: class-weight decision
    # ------------------------------------------------------------------
    def _decide_class_weight(
        self,
        state: WorkflowState,
        config: Any,
        tool_calls: list[dict],
        n_samples: int,
        n_features: int,
    ) -> dict:
        llm = get_llm(config)

        data_summary = state.get("data_summary", {})
        model_names = [_tool_name_to_model_name(tc["tool_name"]) for tc in tool_calls]

        prompt = json.dumps({
            "problem_type": state.get("problem_type"),
            "n_samples": n_samples,
            "n_features": n_features,
            "smote_applied": state.get("smote_applied", False),
            "data_summary_excerpt": {
                "n_rows": data_summary.get("n_rows"),
                "n_cols": data_summary.get("n_cols"),
            },
            "model_names": model_names,
        }, indent=2)

        messages = [
            SystemMessage(content=CLASS_WEIGHT_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        return invoke_llm_json(
            llm, messages,
            agent_name=self.name,
            step_description="class-weight decision",
            verbose=state.get("verbose", False),
        )

    # ------------------------------------------------------------------
    # Inline quick-evaluation after retrain
    # ------------------------------------------------------------------
    def _inline_evaluate_retrained(
        self,
        state: WorkflowState,
        new_results: list[dict],
        target: str,
        problem_type: str,
    ) -> None:
        """Score retrained models on test data and append to retrain_history."""
        test_path = state.get("test_data_path")
        if not test_path:
            return

        try:
            test_df = load_dataframe(test_path)
            X_test = test_df.drop(columns=[target]).values
            y_test = test_df[target].values
        except Exception:
            return

        metric = state.get("user_metric")
        retrain_history: list[dict] = list(state.get("retrain_history", []))

        for result in new_results:
            if not result.get("success") or not result.get("model_path"):
                continue

            model_name = result["name"]
            try:
                model = load_model(result["model_path"])
                eval_result = evaluate_model(model, X_test, y_test, problem_type, metric)
                metrics = eval_result.get("metrics", {})
                primary_metric = eval_result.get("primary_metric", metric or "")
                score = metrics.get(primary_metric, 0)

                prev_attempts = [
                    h for h in retrain_history if h.get("model") == model_name
                ]
                attempt = len(prev_attempts) + 1

                entry = {
                    "model": model_name,
                    "attempt": attempt,
                    "params": result.get("config", {}).get("params", {}),
                    "score": round(score, 6),
                    "metric": primary_metric,
                    "training_time": result.get("training_time", 0),
                }
                retrain_history.append(entry)

                prev_score = prev_attempts[-1]["score"] if prev_attempts else None
                if prev_score is not None:
                    delta = score - prev_score
                    direction = "improved" if delta > 0 else ("unchanged" if delta == 0 else "degraded")
                    log_decision(
                        state, self.name,
                        f"Retrain eval: {model_name} attempt {attempt} — "
                        f"{primary_metric}={score:.4f} ({direction}, delta={delta:+.4f})",
                        f"Previous: {prev_score:.4f}, Current: {score:.4f}",
                        entry,
                    )
                else:
                    log_decision(
                        state, self.name,
                        f"Retrain eval: {model_name} attempt {attempt} — "
                        f"{primary_metric}={score:.4f}",
                        "First retrain attempt evaluated",
                        entry,
                    )

            except Exception as exc:
                log_decision(
                    state, self.name,
                    f"Inline eval failed for {model_name}",
                    str(exc),
                )

        state["retrain_history"] = safe_json_serialize(retrain_history)

    # ------------------------------------------------------------------
    # Fallback: build tool calls for all selected models
    # ------------------------------------------------------------------
    def _fallback_all_models(
        self, model_configs: list[dict], problem_type: str,
    ) -> list[dict]:
        """If the LLM returned no tool_calls, build one per selected model."""
        calls: list[dict] = []
        for cfg in model_configs:
            info = _MODEL_NAME_TO_TOOL.get(cfg["name"])
            if info and info["type"] == problem_type:
                calls.append({
                    "tool_name": info["tool"],
                    "params": cfg.get("params", {}),
                })
        return calls


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def _tool_name_to_model_name(tool_name: str) -> str:
    """Reverse-lookup: tool function name -> sklearn class name."""
    for model_name, info in _MODEL_NAME_TO_TOOL.items():
        if info["tool"] == tool_name:
            return model_name
    return tool_name


def _is_baseline(model_name: str, model_configs: list[dict]) -> bool:
    for cfg in model_configs:
        if cfg["name"] == model_name:
            return cfg.get("is_baseline", False)
    return False
