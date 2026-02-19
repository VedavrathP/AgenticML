"""
Model training tools for the ML pipeline.

All functions are pure Python with NO LLM calls.
"""

import time
import importlib
from typing import Any, Optional
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from agenticml.ml.config import CLASSIFICATION_MODELS, REGRESSION_MODELS


def get_model_candidates(
    problem_type: str,
    max_models: int = 6,
    include_baseline: bool = True,
    data_size: int = 0,
    exclude_models: Optional[list[str]] = None
) -> list[dict]:
    """
    Get a list of model candidates for the given problem type.
    
    Args:
        problem_type: 'classification' or 'regression'
        max_models: Maximum number of models to return
        include_baseline: Whether to include baseline model first
        data_size: Number of samples (used to filter slow models)
        exclude_models: List of model names to exclude
    
    Returns:
        List of model candidate dicts
    """
    if problem_type == "classification":
        models = CLASSIFICATION_MODELS.copy()
    elif problem_type == "regression":
        models = REGRESSION_MODELS.copy()
    else:
        raise ValueError(f"Unknown problem type: {problem_type}")
    
    exclude_models = exclude_models or []
    
    candidates = []
    
    # Add baseline first if requested
    if include_baseline:
        for name, config in models.items():
            if config.get("is_baseline") and name not in exclude_models:
                candidates.append({
                    "name": name,
                    "model_type": config["class"],
                    "module": config["module"],
                    "params": config["default_params"].copy(),
                    "is_baseline": True,
                    "complexity": config.get("complexity", "low")
                })
                break
    
    # Add other models
    for name, config in models.items():
        if name in exclude_models:
            continue
        if config.get("is_baseline"):
            continue  # Already added
        
        # Skip high complexity models for large datasets
        if data_size > 100000 and config.get("complexity") == "high":
            continue
        
        candidates.append({
            "name": name,
            "model_type": config["class"],
            "module": config["module"],
            "params": config["default_params"].copy(),
            "is_baseline": False,
            "complexity": config.get("complexity", "medium")
        })
        
        if len(candidates) >= max_models:
            break
    
    return candidates


def create_model(model_config: dict) -> Any:
    """
    Create a model instance from a configuration dict.
    
    Args:
        model_config: Dict with 'module', 'model_type', and 'params'
    
    Returns:
        Instantiated model
    """
    module_name = model_config["module"]
    class_name = model_config["model_type"]
    params = model_config.get("params", {})
    
    # Import the module and get the class
    module = importlib.import_module(module_name)
    model_class = getattr(module, class_name)
    
    # Instantiate with parameters
    return model_class(**params)


def train_model(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    timeout_seconds: int = 300
) -> tuple[Any, dict]:
    """
    Train a single model.
    
    Args:
        model: The model instance
        X_train: Training features
        y_train: Training target
        timeout_seconds: Maximum training time
    
    Returns:
        Tuple of (trained model, training info dict)
    """
    info = {
        "success": False,
        "training_time": 0.0,
        "error": None
    }
    
    start_time = time.time()
    
    try:
        model.fit(X_train, y_train)
        info["success"] = True
    except Exception as e:
        info["error"] = str(e)
    
    info["training_time"] = time.time() - start_time
    
    return model, info


def train_models_batch(
    model_configs: list[dict],
    X_train: np.ndarray,
    y_train: np.ndarray,
    preprocessor: Optional[Any] = None,
    timeout_per_model: int = 300
) -> list[dict]:
    """
    Train multiple models.
    
    Args:
        model_configs: List of model configuration dicts
        X_train: Training features
        y_train: Training target
        preprocessor: Optional preprocessor to include in pipeline
        timeout_per_model: Maximum time per model
    
    Returns:
        List of training result dicts
    """
    results = []
    
    for config in model_configs:
        model_name = config["name"]
        
        result = {
            "name": model_name,
            "config": config,
            "model": None,
            "pipeline": None,
            "success": False,
            "training_time": 0.0,
            "error": None
        }
        
        try:
            # Create model instance
            model = create_model(config)
            
            # Create pipeline if preprocessor provided
            if preprocessor is not None:
                pipeline = Pipeline([
                    ("preprocessor", clone(preprocessor)),
                    ("model", model)
                ])
                trained, info = train_model(pipeline, X_train, y_train, timeout_per_model)
                result["pipeline"] = trained
            else:
                trained, info = train_model(model, X_train, y_train, timeout_per_model)
                result["model"] = trained
            
            result["success"] = info["success"]
            result["training_time"] = info["training_time"]
            result["error"] = info["error"]
            
        except Exception as e:
            result["error"] = str(e)
        
        results.append(result)
    
    return results


def get_model_from_result(result: dict) -> Any:
    """
    Get the trained model from a training result.
    
    Args:
        result: Training result dict
    
    Returns:
        The trained model or pipeline
    """
    if result.get("pipeline") is not None:
        return result["pipeline"]
    return result.get("model")


def suggest_models_for_data(
    profile: dict,
    problem_type: str,
    max_models: int = 6
) -> list[dict]:
    """
    Suggest models based on data characteristics.
    
    This is a deterministic suggestion. The agent will refine it.
    
    Args:
        profile: Data profile from profile_dataframe
        problem_type: 'classification' or 'regression'
        max_models: Maximum number of models
    
    Returns:
        List of suggested model configs with rationale
    """
    n_rows = profile.get("n_rows", 0)
    n_cols = profile.get("n_cols", 0)
    n_numeric = len(profile.get("numeric_columns", []))
    n_categorical = len(profile.get("categorical_columns", []))
    
    suggestions = []
    
    # Get base candidates
    candidates = get_model_candidates(
        problem_type=problem_type,
        max_models=max_models,
        include_baseline=True,
        data_size=n_rows
    )
    
    for candidate in candidates:
        rationale = ""
        
        if candidate["is_baseline"]:
            rationale = "Baseline model for comparison"
        elif candidate["complexity"] == "low":
            rationale = "Fast, interpretable model"
        elif candidate["complexity"] == "medium":
            if "Forest" in candidate["name"] or "Boosting" in candidate["name"]:
                rationale = "Ensemble method, good for mixed feature types"
        elif candidate["complexity"] == "high":
            if n_rows < 10000:
                rationale = "Complex model suitable for smaller dataset"
        
        candidate["rationale"] = rationale
        suggestions.append(candidate)
    
    return suggestions


def get_model_info(model: Any) -> dict:
    """
    Get information about a trained model.
    
    Args:
        model: Trained model or pipeline
    
    Returns:
        Dict with model information
    """
    info = {
        "type": type(model).__name__,
        "params": {},
        "n_features": None,
        "feature_importances": None
    }
    
    # Get the actual model if it's a pipeline
    actual_model = model
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        actual_model = model.named_steps["model"]
    
    # Get parameters
    if hasattr(actual_model, "get_params"):
        info["params"] = actual_model.get_params()
    
    # Get number of features
    if hasattr(actual_model, "n_features_in_"):
        info["n_features"] = actual_model.n_features_in_
    
    # Get feature importances if available
    if hasattr(actual_model, "feature_importances_"):
        info["feature_importances"] = actual_model.feature_importances_.tolist()
    elif hasattr(actual_model, "coef_"):
        coef = actual_model.coef_
        if coef.ndim > 1:
            coef = coef[0]  # For multi-class, take first class
        info["feature_importances"] = np.abs(coef).tolist()
    
    return info


def compare_models(results: list[dict], metric: str = "score") -> list[dict]:
    """
    Compare trained models and rank them.
    
    Args:
        results: List of training results with evaluation scores
        metric: Metric to compare on
    
    Returns:
        Sorted list of results (best first)
    """
    # Filter successful models
    successful = [r for r in results if r.get("success")]
    
    # Sort by metric (higher is better for most metrics)
    sorted_results = sorted(
        successful,
        key=lambda x: x.get("metrics", {}).get(metric, 0),
        reverse=True
    )
    
    # Add rank
    for i, result in enumerate(sorted_results):
        result["rank"] = i + 1
    
    return sorted_results


def get_baseline_model(problem_type: str) -> dict:
    """
    Get the baseline model configuration for a problem type.
    
    Args:
        problem_type: 'classification' or 'regression'
    
    Returns:
        Baseline model config dict
    """
    if problem_type == "classification":
        models = CLASSIFICATION_MODELS
    else:
        models = REGRESSION_MODELS
    
    for name, config in models.items():
        if config.get("is_baseline"):
            return {
                "name": name,
                "model_type": config["class"],
                "module": config["module"],
                "params": config["default_params"].copy(),
                "is_baseline": True
            }
    
    # Fallback
    first_name = list(models.keys())[0]
    first_config = models[first_name]
    return {
        "name": first_name,
        "model_type": first_config["class"],
        "module": first_config["module"],
        "params": first_config["default_params"].copy(),
        "is_baseline": True
    }
