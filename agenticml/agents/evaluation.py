"""
EvaluationAgent – evaluates trained models and runs quality checks.

Refactored from the old ``evaluator.py`` (model evaluation) and
``critic.py`` (deterministic quality checks).  All LLM interactions use
``invoke_llm_json`` and return structured JSON – no free-text parsing.
"""

import os
import json
from typing import Any

import numpy as np
from langchain_core.messages import SystemMessage, HumanMessage

from agenticml.agents.base_agent import BaseAgent
from agenticml.services.llm_service import get_llm, invoke_llm_json
from agenticml.state.workflow_state import (
    WorkflowState,
    IssueSeverity,
    log_decision,
    add_artifact,
    add_error,
    record_execution,
)
from agenticml.ml.config import get_config
from agenticml.ml.tools.data_io import load_dataframe, resolve_column_name
from agenticml.ml.tools.evaluation import (
    evaluate_model,
    cross_validate_model,
    generate_classification_plots,
    generate_regression_plots,
    generate_feature_importance_plot,
    generate_model_comparison_plot,
    select_best_model,
    is_metric_higher_better,
)
from agenticml.services.artifact_service import load_model, save_json
from agenticml.ml.tools.utils import get_run_subdir, safe_json_serialize


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

EVALUATION_SYSTEM_PROMPT = (
    "You are a Model Evaluation and Quality-Review Agent in an automated ML pipeline. "
    "You must respond with a JSON object.\n\n"
    "Given the evaluation results, deterministic quality-check issues, and "
    "pipeline context, produce a single JSON object with the following schema:\n"
    "{\n"
    '    "best_model": "model_name",\n'
    '    "selection_rationale": "...",\n'
    '    "runner_up": "model_name",\n'
    '    "concerns": ["..."],\n'
    '    "recommendations": ["..."],\n'
    '    "issues": [\n'
    "        {\n"
    '            "severity": "info|warn|blocking",\n'
    '            "category": "...",\n'
    '            "description": "...",\n'
    '            "recommendation": "...",\n'
    '            "affected_component": "..."\n'
    "        }\n"
    "    ],\n"
    '    "overall_assessment": "...",\n'
    '    "should_iterate": false\n'
    "}"
)


# ---------------------------------------------------------------------------
# Deterministic quality checks (preserved from old critic.py)
# ---------------------------------------------------------------------------

def _run_deterministic_checks(state: WorkflowState, config: Any) -> list[dict]:
    """Rule-based checks for suspicious scores, leakage, PII, etc."""
    issues: list[dict] = []

    evaluation_results = state.get("evaluation_results", [])

    # 1 – Suspiciously high scores
    for result in evaluation_results:
        if not result.get("success"):
            continue
        score = result.get("primary_score", 0)
        metric = result.get("primary_metric", "")
        if metric in ("accuracy", "f1", "precision", "recall", "roc_auc", "r2"):
            if score > config.suspicious_score_threshold:
                issues.append({
                    "severity": IssueSeverity.BLOCKING,
                    "category": "scores",
                    "description": (
                        f"Suspiciously high {metric} score ({score:.4f}) for "
                        f"{result['name']}. This often indicates data leakage."
                    ),
                    "recommendation": (
                        "Review feature engineering for target leakage. "
                        "Check if any features are derived from the target."
                    ),
                    "affected_component": "evaluation",
                })
            elif score < config.min_acceptable_score:
                issues.append({
                    "severity": IssueSeverity.WARN,
                    "category": "scores",
                    "description": (
                        f"Low {metric} score ({score:.4f}) for {result['name']}. "
                        "Model may not be useful."
                    ),
                    "recommendation": "Consider feature engineering, different models, or more data.",
                    "affected_component": "modeling",
                })

    # 2 – Unaddressed leakage warnings
    leakage_warnings = state.get("leakage_warnings", [])
    blocking_leakage = [w for w in leakage_warnings if w.get("severity") == "blocking"]
    if blocking_leakage:
        issues.append({
            "severity": IssueSeverity.BLOCKING,
            "category": "leakage",
            "description": (
                f"Found {len(blocking_leakage)} unaddressed data leakage risks: "
                f"{[w['column'] for w in blocking_leakage]}"
            ),
            "recommendation": "Remove or investigate columns with high target correlation.",
            "affected_component": "profiling",
        })

    # 3 – PII in features
    pii_warnings = state.get("pii_warnings", [])
    blocking_pii = [w for w in pii_warnings if w.get("severity") == "blocking"]
    if blocking_pii:
        issues.append({
            "severity": IssueSeverity.WARN,
            "category": "data_quality",
            "description": f"PII detected in {len(blocking_pii)} columns that may still be in features",
            "recommendation": "Ensure PII columns are removed or properly anonymized.",
            "affected_component": "cleaning",
        })

    # 4 – No successful models
    trained_models = state.get("trained_models", [])
    successful_models = [m for m in trained_models if m.get("success")]
    if not successful_models:
        issues.append({
            "severity": IssueSeverity.BLOCKING,
            "category": "modeling",
            "description": "No models were successfully trained",
            "recommendation": "Check data quality, feature engineering, and model configurations.",
            "affected_component": "modeling",
        })

    # 5 – Only baseline model succeeded
    if len(successful_models) == 1 and successful_models[0].get("is_baseline"):
        issues.append({
            "severity": IssueSeverity.WARN,
            "category": "modeling",
            "description": "Only the baseline model was successfully trained",
            "recommendation": "Investigate why other models failed. May indicate data issues.",
            "affected_component": "modeling",
        })

    # 6 – Large data loss during cleaning
    cleaning_report = state.get("cleaning_report", {})
    stats = cleaning_report.get("stats", {}) if isinstance(cleaning_report, dict) else {}
    if stats:
        rows_removed_pct = stats.get("rows_removed_pct", 0)
        if rows_removed_pct > 50:
            issues.append({
                "severity": IssueSeverity.WARN,
                "category": "data_quality",
                "description": f"Cleaning removed {rows_removed_pct:.1f}% of rows",
                "recommendation": "Review cleaning steps. Consider less aggressive cleaning.",
                "affected_component": "cleaning",
            })

    # 7 – CV scores with high variance
    for result in evaluation_results:
        cv_std = result.get("cv_std")
        cv_mean = result.get("cv_mean")
        if cv_std and cv_mean and cv_mean > 0:
            cv_ratio = cv_std / cv_mean
            if cv_ratio > 0.2:
                issues.append({
                    "severity": IssueSeverity.WARN,
                    "category": "scores",
                    "description": (
                        f"High CV variance for {result['name']} "
                        f"(std/mean = {cv_ratio:.2f})"
                    ),
                    "recommendation": "Model performance is unstable. Consider more data or simpler model.",
                    "affected_component": "evaluation",
                })

    return issues


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class EvaluationAgent(BaseAgent):
    """Evaluates trained models, selects the best, and runs quality checks."""

    name: str = "evaluation"

    # ------------------------------------------------------------------
    # public entry-point
    # ------------------------------------------------------------------
    def run(self, state: WorkflowState) -> WorkflowState:
        config = get_config()
        run_dir = state["run_dir"]
        problem_type = state["problem_type"]
        target = state["target"]
        metric = state.get("user_metric", config.get_default_metric(problem_type))

        # ── 1. Load data ─────────────────────────────────────────────
        test_path = state.get("test_data_path")
        train_path = state.get("train_data_path")
        use_cv = state.get("cv_splitter", False)

        if not test_path and not use_cv:
            add_error(state, self.name, "No test data path in state")
            record_execution(state, self.name, status="failed")
            return state

        try:
            train_df = load_dataframe(train_path)
            target = resolve_column_name(train_df, target)
            state["target"] = target
            X_train = train_df.drop(columns=[target]).values
            y_train = train_df[target].values

            if test_path:
                test_df = load_dataframe(test_path)
                X_test = test_df.drop(columns=[target]).values
                y_test = test_df[target].values
            else:
                X_test = None
                y_test = None
        except Exception as exc:
            add_error(state, self.name, f"Failed to load data: {exc}")
            record_execution(state, self.name, status="failed")
            return state

        log_decision(
            state, self.name,
            f"Evaluation mode: {'Cross-validation' if use_cv else 'Train/Test split'}",
            (
                f"Training samples: {len(X_train)}, "
                f"Test samples: {len(X_test) if X_test is not None else 'N/A (CV)'}"
            ),
        )

        # ── 2. Load trained models ───────────────────────────────────
        trained_models = state.get("trained_models", [])
        successful_models = [m for m in trained_models if m.get("success")]

        if not successful_models:
            add_error(state, self.name, "No successfully trained models to evaluate")
            record_execution(state, self.name, status="failed")
            return state

        # ── 3. Evaluate each model ───────────────────────────────────
        plots_dir = get_run_subdir(run_dir, "plots")
        metrics_dir = get_run_subdir(run_dir, "metrics")
        feature_names = state.get("feature_names", [])

        evaluation_results: list[dict] = []

        for model_info in successful_models:
            model_name = model_info["name"]
            model_path = model_info.get("model_path")
            if not model_path:
                continue

            try:
                model = load_model(model_path)
                eval_result = self._evaluate_single_model(
                    model, model_name, model_info,
                    X_train, y_train, X_test, y_test,
                    problem_type, metric, use_cv,
                )

                plots = self._generate_plots(
                    eval_result, model_name, model_info,
                    X_test, y_test, y_train, X_train,
                    problem_type, use_cv, plots_dir, feature_names,
                )

                result = {
                    "name": model_name,
                    "metrics": eval_result.get("metrics", {}),
                    "primary_metric": metric,
                    "primary_score": eval_result.get("metrics", {}).get(metric, 0),
                    "plots": plots,
                    "is_baseline": model_info.get("is_baseline", False),
                    "training_time": model_info.get("training_time", 0),
                    "success": True,
                }
                if "cv_scores" in eval_result:
                    result["cv_scores"] = eval_result["cv_scores"]
                    result["cv_mean"] = eval_result["cv_mean"]
                    result["cv_std"] = eval_result["cv_std"]

                evaluation_results.append(result)

                for plot_path in plots:
                    add_artifact(state, os.path.basename(plot_path), plot_path, "plot")

                log_decision(
                    state, self.name,
                    f"Evaluated {model_name}: {metric}={result['primary_score']:.4f}",
                    f"Metrics: {result['metrics']}",
                    result["metrics"],
                )

            except Exception as exc:
                evaluation_results.append({
                    "name": model_name,
                    "success": False,
                    "error": str(exc),
                })
                log_decision(state, self.name, f"Failed to evaluate {model_name}", str(exc))

        # ── 4. Comparison plot ───────────────────────────────────────
        successful_evals = [r for r in evaluation_results if r.get("success")]
        if len(successful_evals) > 1:
            comparison_plot = generate_model_comparison_plot(successful_evals, metric, plots_dir)
            add_artifact(state, "model_comparison", comparison_plot, "plot")

        # ── 5. Select best model (deterministic) ─────────────────────
        higher_is_better = is_metric_higher_better(metric)
        best_result = select_best_model(successful_evals, metric, higher_is_better)

        # Store evaluation_results in state before deterministic checks
        state["evaluation_results"] = safe_json_serialize(evaluation_results)

        # ── 6. Deterministic quality checks ──────────────────────────
        deterministic_issues = _run_deterministic_checks(state, config)

        # ── 7. LLM analysis (best-model reasoning + issue review) ────
        if best_result:
            llm_analysis = self._get_llm_analysis(
                state, config, successful_evals, deterministic_issues,
                metric, problem_type, best_result,
            )

            best_model_name = best_result["name"]
            best_model_info = next(
                (m for m in trained_models if m["name"] == best_model_name), {}
            )

            state["best_model"] = safe_json_serialize({
                "name": best_model_name,
                "path": best_model_info.get("model_path"),
                "metrics": best_result.get("metrics", {}),
                "primary_score": best_result.get("primary_score"),
                "is_baseline": best_result.get("is_baseline", False),
                "selection_rationale": llm_analysis.get("selection_rationale", "Best performance on primary metric"),
            })
            state["best_model_path"] = best_model_info.get("model_path")

            # Merge LLM-identified issues with deterministic ones
            all_issues = deterministic_issues.copy()
            for issue in llm_analysis.get("issues", []):
                if not any(
                    i["category"] == issue.get("category")
                    and i["description"][:50] == issue.get("description", "")[:50]
                    for i in all_issues
                ):
                    all_issues.append(issue)

            state["evaluation_issues"] = safe_json_serialize(all_issues)

            log_decision(
                state, self.name,
                f"Selected best model: {best_model_name}",
                llm_analysis.get("selection_rationale", f"Best {metric} score"),
                {"best_model": best_model_name, "score": best_result.get("primary_score")},
            )
        else:
            state["evaluation_issues"] = safe_json_serialize(deterministic_issues)

        # ── 8. Save evaluation summary ───────────────────────────────
        eval_summary = {
            "metric": metric,
            "higher_is_better": higher_is_better,
            "results": evaluation_results,
            "best_model": state.get("best_model"),
            "n_evaluated": len(successful_evals),
            "n_failed": len(evaluation_results) - len(successful_evals),
        }
        eval_path = os.path.join(metrics_dir, "evaluation_results.json")
        save_json(eval_path, eval_summary)
        add_artifact(state, "evaluation_results", eval_path, "json")

        log_decision(
            state, self.name,
            f"Completed evaluation of {len(successful_evals)} models",
            (
                f"Best: {best_result.get('name', 'N/A')} with "
                f"{metric}={best_result.get('primary_score', 0):.4f}"
                if best_result
                else "No successful evaluations"
            ),
            eval_summary,
        )

        record_execution(state, self.name)
        return state

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _evaluate_single_model(
        model: Any,
        model_name: str,
        model_info: dict,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray | None,
        y_test: np.ndarray | None,
        problem_type: str,
        metric: str,
        use_cv: bool,
    ) -> dict:
        if X_test is not None:
            return evaluate_model(model, X_test, y_test, problem_type=problem_type, metric=metric)

        from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict

        cv = (
            StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            if problem_type == "classification"
            else KFold(n_splits=5, shuffle=True, random_state=42)
        )

        cv_result = cross_validate_model(model, X_train, y_train, cv, metric)

        additional_metrics: dict[str, float] = {}
        y_pred_cv = None
        try:
            y_pred_cv = cross_val_predict(model, X_train, y_train, cv=cv)
            if problem_type == "classification":
                from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                additional_metrics = {
                    "accuracy": float(accuracy_score(y_train, y_pred_cv)),
                    "precision": float(precision_score(y_train, y_pred_cv, average="weighted", zero_division=0)),
                    "recall": float(recall_score(y_train, y_pred_cv, average="weighted", zero_division=0)),
                    "f1": float(f1_score(y_train, y_pred_cv, average="weighted", zero_division=0)),
                }
            else:
                from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                additional_metrics = {
                    "rmse": float(np.sqrt(mean_squared_error(y_train, y_pred_cv))),
                    "mae": float(mean_absolute_error(y_train, y_pred_cv)),
                    "r2": float(r2_score(y_train, y_pred_cv)),
                }
        except Exception:
            pass

        return {
            "metrics": {metric: cv_result["cv_mean"], **additional_metrics},
            "cv_scores": cv_result["cv_scores"],
            "cv_mean": cv_result["cv_mean"],
            "cv_std": cv_result["cv_std"],
            "predictions": y_pred_cv,
        }

    @staticmethod
    def _generate_plots(
        eval_result: dict,
        model_name: str,
        model_info: dict,
        X_test: np.ndarray | None,
        y_test: np.ndarray | None,
        y_train: np.ndarray,
        X_train: np.ndarray,
        problem_type: str,
        use_cv: bool,
        plots_dir: str,
        feature_names: list[str],
    ) -> list[str]:
        plots: list[str] = []
        y_pred = eval_result.get("predictions")
        y_proba = eval_result.get("probabilities")
        y_true = y_test if X_test is not None else y_train

        if y_pred is not None and len(y_pred) == len(y_true):
            if problem_type == "classification":
                plots.extend(
                    generate_classification_plots(y_true, y_pred, y_proba, plots_dir, model_name)
                )
            elif problem_type == "regression":
                plots.extend(
                    generate_regression_plots(y_true, y_pred, plots_dir, model_name)
                )

        importances = model_info.get("model_info", {}).get("feature_importances")
        if importances and feature_names and len(importances) == len(feature_names):
            plots.append(
                generate_feature_importance_plot(feature_names, importances, plots_dir, model_name)
            )

        return plots

    # ------------------------------------------------------------------
    # LLM call: evaluation analysis + issue review
    # ------------------------------------------------------------------
    def _get_llm_analysis(
        self,
        state: WorkflowState,
        config: Any,
        successful_evals: list[dict],
        deterministic_issues: list[dict],
        metric: str,
        problem_type: str,
        best_result: dict,
    ) -> dict:
        llm = get_llm(config)

        results_summary = []
        for r in successful_evals:
            results_summary.append({
                "name": r["name"],
                "primary_score": r.get("primary_score"),
                "metrics": r.get("metrics", {}),
                "is_baseline": r.get("is_baseline", False),
                "training_time": r.get("training_time", 0),
                "cv_mean": r.get("cv_mean"),
                "cv_std": r.get("cv_std"),
            })

        pipeline_summary = {
            "target": state.get("target"),
            "problem_type": problem_type,
            "n_rows": state.get("data_summary", {}).get("n_rows"),
            "cleaning_report": state.get("cleaning_report", {}),
            "preprocessing_plan": state.get("preprocessing_plan", {}),
            "pii_warnings_count": len(state.get("pii_warnings", [])),
            "leakage_warnings_count": len(state.get("leakage_warnings", [])),
        }

        prompt = json.dumps({
            "problem_type": problem_type,
            "primary_metric": metric,
            "evaluation_results": results_summary,
            "deterministic_best_model": best_result.get("name"),
            "deterministic_best_score": best_result.get("primary_score"),
            "deterministic_issues": deterministic_issues,
            "pipeline_summary": pipeline_summary,
        }, indent=2)

        messages = [
            SystemMessage(content=EVALUATION_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        return invoke_llm_json(
            llm, messages,
            agent_name=self.name,
            step_description="evaluation analysis and issue review",
            verbose=state.get("verbose", False),
        )
