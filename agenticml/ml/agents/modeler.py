"""
Modeling Agent

Responsible for:
- Choosing model candidates based on data characteristics
- Training baseline model first
- Training 3-6 model candidates
- Respecting runtime constraints
- Saving all trained models
"""

import os
import json
from typing import Any

from agenticml.ml.tools.llm_factory import create_llm
from langchain_core.messages import SystemMessage, HumanMessage
import numpy as np
import pandas as pd

from agenticml.ml.state import PipelineState, log_decision, add_artifact, add_error
from agenticml.ml.config import get_config, CLASSIFICATION_MODELS, REGRESSION_MODELS
from agenticml.ml.tools.data_io import load_dataframe
from agenticml.ml.tools.modeling import (
    get_model_candidates,
    create_model,
    train_model,
    train_models_batch,
    suggest_models_for_data,
    get_model_info,
    get_baseline_model
)
from agenticml.ml.tools.artifacts import save_model, save_json
from agenticml.ml.tools.utils import get_run_subdir, safe_json_serialize, format_duration
from agenticml.ml.tools.llm import invoke_llm


MODELER_SYSTEM_PROMPT = """You are a Model Selection Agent in an ML pipeline.

Your role is to select appropriate models for the dataset based on:
1. Problem type (classification/regression)
2. Dataset size and characteristics
3. Feature types (numeric, categorical)
4. Time constraints

Available models for classification:
- LogisticRegression (baseline, fast, interpretable)
- RandomForestClassifier (ensemble, handles mixed features)
- GradientBoostingClassifier (ensemble, good accuracy)
- SVC (good for small datasets)
- KNeighborsClassifier (simple, no assumptions)
- DecisionTreeClassifier (interpretable)
- XGBClassifier (if available, high performance)
- LGBMClassifier (if available, fast)

Available models for regression:
- LinearRegression (baseline, fast, interpretable)
- Ridge (regularized linear)
- RandomForestRegressor (ensemble)
- GradientBoostingRegressor (ensemble)
- SVR (good for small datasets)
- DecisionTreeRegressor (interpretable)
- XGBRegressor (if available)
- LGBMRegressor (if available)

Guidelines:
- Always include baseline model first
- Select 3-6 models total
- Consider dataset size (avoid SVM for large datasets)
- Consider interpretability needs

Respond in JSON format:
{
    "model_selection": {
        "models": [
            {
                "name": "ModelName",
                "rationale": "why this model"
            }
        ],
        "overall_rationale": "selection strategy explanation"
    }
}
"""


def run_modeler_agent(state: PipelineState) -> PipelineState:
    """
    Run the modeler agent to train models.
    
    This agent:
    1. Loads preprocessed training data
    2. Selects model candidates
    3. Applies class_weight if configured for imbalanced data
    4. Trains baseline model first
    5. Trains additional models
    6. Saves all trained models
    
    Args:
        state: Current pipeline state
    
    Returns:
        Updated pipeline state
    """
    config = get_config()
    run_dir = state["run_dir"]
    problem_type = state["problem_type"]
    target = state["target"]
    
    # Get planning decisions for class imbalance handling
    planning_decisions = state.get("planning_decisions", {})
    if isinstance(planning_decisions, str):
        planning_decisions = json.loads(planning_decisions) if planning_decisions else {}
    
    imbalance_strategy = planning_decisions.get("imbalance_strategy", "none")
    use_class_weight = (
        problem_type == "classification" and 
        imbalance_strategy == "class_weight" and
        not state.get("smote_applied", False)  # Don't use class_weight if SMOTE was applied
    )
    
    # =========================================================================
    # Step 1: Load training data
    # =========================================================================
    train_path = state.get("train_data_path")
    if not train_path:
        add_error(state, "modeler", "No training data path in state")
        return state
    
    try:
        train_df = load_dataframe(train_path)
        X_train = train_df.drop(columns=[target]).values
        y_train = train_df[target].values
    except Exception as e:
        add_error(state, "modeler", f"Failed to load training data: {str(e)}")
        return state
    
    n_samples, n_features = X_train.shape
    
    # =========================================================================
    # Step 2: Get model candidates
    # =========================================================================
    data_summary = state.get("data_summary", {})
    
    # Get base suggestions
    base_candidates = suggest_models_for_data(
        profile=data_summary,
        problem_type=problem_type,
        max_models=config.max_models
    )
    
    # Use LLM to refine selection
    llm_selection = _get_llm_model_selection(
        problem_type=problem_type,
        n_samples=n_samples,
        n_features=n_features,
        data_summary=data_summary,
        base_candidates=base_candidates,
        config=config,
        verbose=state.get("verbose", False)
    )
    
    # Build final model list
    if llm_selection and llm_selection.get("models"):
        selected_names = [m["name"] for m in llm_selection["models"]]
        model_configs = _build_model_configs(selected_names, problem_type)
    else:
        model_configs = base_candidates
    
    # Ensure baseline is first
    model_configs = _ensure_baseline_first(model_configs, problem_type)
    
    # Limit to max models
    model_configs = model_configs[:config.max_models]
    
    state["model_candidates"] = safe_json_serialize(model_configs)
    
    log_decision(
        state, "modeler",
        f"Selected {len(model_configs)} model candidates",
        llm_selection.get("overall_rationale", "") if llm_selection else "Using default selection",
        {"models": [m["name"] for m in model_configs]}
    )
    
    # =========================================================================
    # Step 3: Train models
    # =========================================================================
    models_dir = get_run_subdir(run_dir, "models")
    trained_models = []
    
    # Log class weight usage
    if use_class_weight:
        log_decision(
            state, "modeler",
            "Using class_weight='balanced' for all compatible models",
            "This helps handle class imbalance without resampling",
            {"imbalance_strategy": imbalance_strategy}
        )
    
    for i, model_config in enumerate(model_configs):
        model_name = model_config["name"]
        is_baseline = model_config.get("is_baseline", False)
        
        # Apply class_weight if configured and model supports it
        if use_class_weight:
            model_config = _apply_class_weight(model_config)
        
        log_decision(
            state, "modeler",
            f"Training model {i+1}/{len(model_configs)}: {model_name}",
            f"{'Baseline model' if is_baseline else 'Candidate model'}{'(with class_weight)' if use_class_weight else ''}",
            model_config
        )
        
        try:
            # Create and train model
            model = create_model(model_config)
            trained_model, train_info = train_model(
                model, X_train, y_train,
                timeout_seconds=config.model_timeout_seconds
            )
            
            if train_info["success"]:
                # Save model
                model_path = os.path.join(models_dir, f"{model_name}.joblib")
                save_model(trained_model, model_path, metadata={
                    "name": model_name,
                    "config": model_config,
                    "training_time": train_info["training_time"]
                })
                
                # Get model info
                model_info = get_model_info(trained_model)
                
                trained_models.append({
                    "name": model_name,
                    "config": model_config,
                    "model_path": model_path,
                    "training_time": train_info["training_time"],
                    "is_baseline": is_baseline,
                    "success": True,
                    "model_info": model_info
                })
                
                add_artifact(state, f"model_{model_name}", model_path, "model")
                
                log_decision(
                    state, "modeler",
                    f"Trained {model_name} in {format_duration(train_info['training_time'])}",
                    "Training successful",
                    {"training_time": train_info["training_time"]}
                )
            else:
                trained_models.append({
                    "name": model_name,
                    "config": model_config,
                    "success": False,
                    "error": train_info.get("error", "Unknown error")
                })
                
                log_decision(
                    state, "modeler",
                    f"Failed to train {model_name}",
                    train_info.get("error", "Unknown error")
                )
        
        except Exception as e:
            trained_models.append({
                "name": model_name,
                "config": model_config,
                "success": False,
                "error": str(e)
            })
            
            log_decision(
                state, "modeler",
                f"Error training {model_name}",
                str(e)
            )
    
    # =========================================================================
    # Step 4: Save training summary
    # =========================================================================
    state["trained_models"] = safe_json_serialize(trained_models)
    
    successful_models = [m for m in trained_models if m.get("success")]
    failed_models = [m for m in trained_models if not m.get("success")]
    
    total_training_time = sum(m.get("training_time", 0) for m in successful_models)
    
    training_summary = {
        "total_models": len(model_configs),
        "successful": len(successful_models),
        "failed": len(failed_models),
        "total_training_time": total_training_time,
        "models": trained_models
    }
    
    summary_path = os.path.join(models_dir, "training_summary.json")
    save_json(summary_path, training_summary)
    add_artifact(state, "training_summary", summary_path, "json")
    
    log_decision(
        state, "modeler",
        f"Completed model training: {len(successful_models)}/{len(model_configs)} successful",
        f"Total training time: {format_duration(total_training_time)}",
        training_summary
    )
    
    return state


def _get_llm_model_selection(
    problem_type: str,
    n_samples: int,
    n_features: int,
    data_summary: dict,
    base_candidates: list,
    config: Any,
    verbose: bool = False
) -> dict:
    """Use LLM to select models."""
    if not config.llm_api_key:
        return {}
    
    try:
        llm = create_llm(config)
        
        # Get available models
        available_models = list(CLASSIFICATION_MODELS.keys()) if problem_type == "classification" else list(REGRESSION_MODELS.keys())
        
        context = {
            "problem_type": problem_type,
            "n_samples": n_samples,
            "n_features": n_features,
            "n_numeric": len(data_summary.get("numeric_columns", [])),
            "n_categorical": len(data_summary.get("categorical_columns", [])),
            "available_models": available_models,
            "max_models": config.max_models
        }
        
        prompt = f"""Select models for this ML task.

Context:
{json.dumps(context, indent=2)}

Base Suggestions:
{json.dumps([{"name": m["name"], "rationale": m.get("rationale", "")} for m in base_candidates], indent=2)}

Requirements:
1. Include baseline model first (LogisticRegression for classification, LinearRegression for regression)
2. Select 3-6 models total
3. Consider dataset size ({n_samples} samples)
4. Avoid SVM for large datasets (>50k samples)
5. Include at least one ensemble method

Select the best models for this task."""

        messages = [
            SystemMessage(content=MODELER_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]
        
        response = invoke_llm(llm, messages, "Modeler", "Model selection", verbose)
        
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        
        result = json.loads(content)
        return result.get("model_selection", result)
    
    except Exception:
        return {}


def _build_model_configs(model_names: list, problem_type: str) -> list:
    """Build model configurations from names."""
    models_dict = CLASSIFICATION_MODELS if problem_type == "classification" else REGRESSION_MODELS
    
    configs = []
    for name in model_names:
        if name in models_dict:
            model_info = models_dict[name]
            configs.append({
                "name": name,
                "model_type": model_info["class"],
                "module": model_info["module"],
                "params": model_info["default_params"].copy(),
                "is_baseline": model_info.get("is_baseline", False),
                "complexity": model_info.get("complexity", "medium")
            })
    
    return configs


def _ensure_baseline_first(model_configs: list, problem_type: str) -> list:
    """Ensure baseline model is first in the list."""
    # Find baseline
    baseline_idx = None
    for i, config in enumerate(model_configs):
        if config.get("is_baseline"):
            baseline_idx = i
            break
    
    # If baseline found and not first, move it
    if baseline_idx is not None and baseline_idx > 0:
        baseline = model_configs.pop(baseline_idx)
        model_configs.insert(0, baseline)
    
    # If no baseline, add one
    if baseline_idx is None:
        baseline = get_baseline_model(problem_type)
        model_configs.insert(0, baseline)
    
    return model_configs


def _apply_class_weight(model_config: dict) -> dict:
    """
    Apply class_weight='balanced' to models that support it.
    
    Models that support class_weight:
    - LogisticRegression
    - RandomForestClassifier
    - DecisionTreeClassifier
    - SVC
    - GradientBoostingClassifier (via sample_weight, not directly)
    """
    config = model_config.copy()
    config["params"] = config.get("params", {}).copy()
    
    model_name = config.get("name", "")
    
    # Models that support class_weight parameter
    supports_class_weight = [
        "LogisticRegression",
        "RandomForestClassifier",
        "DecisionTreeClassifier",
        "SVC",
        "LinearSVC",
        "SGDClassifier",
        "ExtraTreesClassifier"
    ]
    
    if model_name in supports_class_weight:
        config["params"]["class_weight"] = "balanced"
    
    # XGBoost uses scale_pos_weight (handled differently)
    # GradientBoosting doesn't directly support class_weight
    
    return config
