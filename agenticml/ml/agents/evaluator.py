"""
Evaluator Agent

Responsible for:
- Choosing proper evaluation strategy
- Computing correct metrics for problem type
- Generating evaluation plots
- Selecting the best model
"""

import os
import json
from typing import Any

from agenticml.ml.tools.llm_factory import create_llm
from langchain_core.messages import SystemMessage, HumanMessage
import numpy as np
import pandas as pd

from agenticml.ml.state import PipelineState, log_decision, add_artifact, add_error
from agenticml.ml.config import get_config
from agenticml.ml.tools.data_io import load_dataframe
from agenticml.ml.tools.artifacts import load_model, save_json
from agenticml.ml.tools.evaluation import (
    evaluate_model,
    cross_validate_model,
    generate_classification_plots,
    generate_regression_plots,
    generate_feature_importance_plot,
    generate_model_comparison_plot,
    select_best_model,
    is_metric_higher_better
)
from agenticml.ml.tools.utils import get_run_subdir, safe_json_serialize
from agenticml.ml.tools.llm import invoke_llm


EVALUATOR_SYSTEM_PROMPT = """You are a Model Evaluation Agent in an ML pipeline.

Your role is to evaluate trained models and select the best one based on:
1. Primary metric performance
2. Cross-validation stability (if applicable)
3. Training time and complexity trade-offs
4. Interpretability requirements

Evaluation considerations:
- For classification: accuracy, precision, recall, F1, ROC-AUC
- For regression: RMSE, MAE, R², MAPE
- Consider overfitting (train vs test performance)
- Consider model complexity vs performance trade-off

Respond in JSON format:
{
    "evaluation_analysis": {
        "best_model": "model_name",
        "selection_rationale": "why this model is best",
        "runner_up": "second best model",
        "concerns": ["any concerns about the selection"],
        "recommendations": ["recommendations for improvement"]
    }
}
"""


def run_evaluator_agent(state: PipelineState) -> PipelineState:
    """
    Run the evaluator agent to evaluate trained models.
    
    This agent:
    1. Loads test data and trained models
    2. Evaluates each model
    3. Generates evaluation plots
    4. Selects the best model
    
    Args:
        state: Current pipeline state
    
    Returns:
        Updated pipeline state
    """
    config = get_config()
    run_dir = state["run_dir"]
    problem_type = state["problem_type"]
    target = state["target"]
    metric = state.get("user_metric", config.get_default_metric(problem_type))
    
    # =========================================================================
    # Step 1: Load test data
    # =========================================================================
    test_path = state.get("test_data_path")
    train_path = state.get("train_data_path")
    use_cv = state.get("cv_splitter", False)
    
    if not test_path and not use_cv:
        add_error(state, "evaluator", "No test data path in state")
        return state
    
    try:
        train_df = load_dataframe(train_path)
        X_train = train_df.drop(columns=[target]).values
        y_train = train_df[target].values
        
        if test_path:
            test_df = load_dataframe(test_path)
            X_test = test_df.drop(columns=[target]).values
            y_test = test_df[target].values
        else:
            # For CV, we'll evaluate using cross-validation on training data
            X_test = None
            y_test = None
    except Exception as e:
        add_error(state, "evaluator", f"Failed to load data: {str(e)}")
        return state
    
    log_decision(
        state, "evaluator",
        f"Evaluation mode: {'Cross-validation' if use_cv else 'Train/Test split'}",
        f"Training samples: {len(X_train)}, Test samples: {len(X_test) if X_test is not None else 'N/A (CV)'}"
    )
    
    # =========================================================================
    # Step 2: Load trained models
    # =========================================================================
    trained_models = state.get("trained_models", [])
    successful_models = [m for m in trained_models if m.get("success")]
    
    if not successful_models:
        add_error(state, "evaluator", "No successfully trained models to evaluate")
        return state
    
    # =========================================================================
    # Step 3: Evaluate each model
    # =========================================================================
    plots_dir = get_run_subdir(run_dir, "plots")
    metrics_dir = get_run_subdir(run_dir, "metrics")
    
    evaluation_results = []
    feature_names = state.get("feature_names", [])
    
    for model_info in successful_models:
        model_name = model_info["name"]
        model_path = model_info.get("model_path")
        
        if not model_path:
            continue
        
        try:
            model = load_model(model_path)
            
            # Evaluate on test set
            if X_test is not None:
                eval_result = evaluate_model(
                    model, X_test, y_test,
                    problem_type=problem_type,
                    metric=metric
                )
            else:
                # Use cross-validation
                from sklearn.model_selection import StratifiedKFold, KFold, cross_val_predict
                cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42) if problem_type == "classification" else KFold(n_splits=5, shuffle=True, random_state=42)
                
                # Get CV scores for primary metric
                cv_result = cross_validate_model(model, X_train, y_train, cv, metric)
                
                # Also get predictions via cross_val_predict for additional metrics
                try:
                    y_pred_cv = cross_val_predict(model, X_train, y_train, cv=cv)
                    
                    # Compute additional metrics from CV predictions
                    if problem_type == "classification":
                        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
                        additional_metrics = {
                            "accuracy": float(accuracy_score(y_train, y_pred_cv)),
                            "precision": float(precision_score(y_train, y_pred_cv, average="weighted", zero_division=0)),
                            "recall": float(recall_score(y_train, y_pred_cv, average="weighted", zero_division=0)),
                            "f1": float(f1_score(y_train, y_pred_cv, average="weighted", zero_division=0))
                        }
                    else:
                        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
                        additional_metrics = {
                            "rmse": float(np.sqrt(mean_squared_error(y_train, y_pred_cv))),
                            "mae": float(mean_absolute_error(y_train, y_pred_cv)),
                            "r2": float(r2_score(y_train, y_pred_cv))
                        }
                except Exception:
                    additional_metrics = {}
                
                eval_result = {
                    "metrics": {metric: cv_result["cv_mean"], **additional_metrics},
                    "cv_scores": cv_result["cv_scores"],
                    "cv_mean": cv_result["cv_mean"],
                    "cv_std": cv_result["cv_std"],
                    "predictions": y_pred_cv if 'y_pred_cv' in dir() else None
                }
            
            # Generate plots
            plots = []
            
            y_pred = eval_result.get("predictions")
            y_proba = eval_result.get("probabilities")
            
            # Use test data if available, otherwise use training data with CV predictions
            y_true_for_plots = y_test if X_test is not None else y_train
            
            if y_pred is not None and len(y_pred) == len(y_true_for_plots):
                if problem_type == "classification":
                    class_plots = generate_classification_plots(
                        y_true_for_plots, y_pred, y_proba,
                        plots_dir, model_name
                    )
                    plots.extend(class_plots)
                
                elif problem_type == "regression":
                    reg_plots = generate_regression_plots(
                        y_true_for_plots, y_pred,
                        plots_dir, model_name
                    )
                    plots.extend(reg_plots)
            
            # Feature importance plot
            model_details = model_info.get("model_info", {})
            importances = model_details.get("feature_importances")
            
            if importances and feature_names and len(importances) == len(feature_names):
                fi_plot = generate_feature_importance_plot(
                    feature_names, importances,
                    plots_dir, model_name
                )
                plots.append(fi_plot)
            
            # Build result
            result = {
                "name": model_name,
                "metrics": eval_result.get("metrics", {}),
                "primary_metric": metric,
                "primary_score": eval_result.get("metrics", {}).get(metric, 0),
                "plots": plots,
                "is_baseline": model_info.get("is_baseline", False),
                "training_time": model_info.get("training_time", 0),
                "success": True
            }
            
            if "cv_scores" in eval_result:
                result["cv_scores"] = eval_result["cv_scores"]
                result["cv_mean"] = eval_result["cv_mean"]
                result["cv_std"] = eval_result["cv_std"]
            
            evaluation_results.append(result)
            
            # Add plot artifacts
            for plot_path in plots:
                plot_name = os.path.basename(plot_path)
                add_artifact(state, plot_name, plot_path, "plot")
            
            log_decision(
                state, "evaluator",
                f"Evaluated {model_name}: {metric}={result['primary_score']:.4f}",
                f"Metrics: {result['metrics']}",
                result["metrics"]
            )
        
        except Exception as e:
            evaluation_results.append({
                "name": model_name,
                "success": False,
                "error": str(e)
            })
            log_decision(
                state, "evaluator",
                f"Failed to evaluate {model_name}",
                str(e)
            )
    
    # =========================================================================
    # Step 4: Generate comparison plot
    # =========================================================================
    successful_evals = [r for r in evaluation_results if r.get("success")]
    
    if len(successful_evals) > 1:
        comparison_plot = generate_model_comparison_plot(
            successful_evals, metric, plots_dir
        )
        add_artifact(state, "model_comparison", comparison_plot, "plot")
    
    # =========================================================================
    # Step 5: Select best model
    # =========================================================================
    higher_is_better = is_metric_higher_better(metric)
    best_result = select_best_model(successful_evals, metric, higher_is_better)
    
    if best_result:
        # Get LLM analysis
        llm_analysis = _get_llm_evaluation_analysis(
            evaluation_results=successful_evals,
            metric=metric,
            problem_type=problem_type,
            best_result=best_result,
            config=config,
            verbose=state.get("verbose", False)
        )
        
        best_model_name = best_result["name"]
        best_model_info = next(
            (m for m in trained_models if m["name"] == best_model_name),
            {}
        )
        
        state["best_model"] = safe_json_serialize({
            "name": best_model_name,
            "path": best_model_info.get("model_path"),
            "metrics": best_result.get("metrics", {}),
            "primary_score": best_result.get("primary_score"),
            "is_baseline": best_result.get("is_baseline", False),
            "selection_rationale": llm_analysis.get("selection_rationale", "Best performance on primary metric")
        })
        state["best_model_path"] = best_model_info.get("model_path")
        
        log_decision(
            state, "evaluator",
            f"Selected best model: {best_model_name}",
            llm_analysis.get("selection_rationale", f"Best {metric} score: {best_result.get('primary_score', 0):.4f}"),
            {"best_model": best_model_name, "score": best_result.get("primary_score")}
        )
    
    # =========================================================================
    # Step 6: Save evaluation results
    # =========================================================================
    state["evaluation_results"] = safe_json_serialize(evaluation_results)
    
    eval_summary = {
        "metric": metric,
        "higher_is_better": higher_is_better,
        "results": evaluation_results,
        "best_model": state.get("best_model"),
        "n_evaluated": len(successful_evals),
        "n_failed": len(evaluation_results) - len(successful_evals)
    }
    
    eval_path = os.path.join(metrics_dir, "evaluation_results.json")
    save_json(eval_path, eval_summary)
    add_artifact(state, "evaluation_results", eval_path, "json")
    
    log_decision(
        state, "evaluator",
        f"Completed evaluation of {len(successful_evals)} models",
        f"Best: {best_result.get('name', 'N/A')} with {metric}={best_result.get('primary_score', 0):.4f}" if best_result else "No successful evaluations",
        eval_summary
    )
    
    return state


def _get_llm_evaluation_analysis(
    evaluation_results: list,
    metric: str,
    problem_type: str,
    best_result: dict,
    config: Any,
    verbose: bool = False
) -> dict:
    """Use LLM to analyze evaluation results."""
    if not config.llm_api_key:
        return {"selection_rationale": f"Best {metric} score"}
    
    try:
        llm = create_llm(config)
        
        # Prepare results summary
        results_summary = []
        for r in evaluation_results:
            if r.get("success"):
                results_summary.append({
                    "name": r["name"],
                    "primary_score": r.get("primary_score"),
                    "metrics": r.get("metrics", {}),
                    "is_baseline": r.get("is_baseline", False),
                    "training_time": r.get("training_time", 0),
                    "cv_mean": r.get("cv_mean"),
                    "cv_std": r.get("cv_std")
                })
        
        prompt = f"""Analyze these model evaluation results and explain the best model selection.

Problem Type: {problem_type}
Primary Metric: {metric}

Results:
{json.dumps(results_summary, indent=2)}

Selected Best Model: {best_result.get('name')}
Best Score: {best_result.get('primary_score')}

Provide analysis including:
1. Why the selected model is best
2. Any concerns about the selection
3. Recommendations for improvement"""

        messages = [
            SystemMessage(content=EVALUATOR_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]
        
        response = invoke_llm(llm, messages, "Evaluator", "Evaluation analysis", verbose)
        
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
            result = json.loads(content)
            return result.get("evaluation_analysis", result)
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
            try:
                result = json.loads(content)
                return result.get("evaluation_analysis", result)
            except json.JSONDecodeError:
                pass
        
        # If not JSON, extract rationale from text
        return {"selection_rationale": content[:500]}
    
    except Exception:
        return {"selection_rationale": f"Best {metric} score: {best_result.get('primary_score', 0):.4f}"}
