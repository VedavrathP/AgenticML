"""
Model evaluation tools for the ML pipeline.

All functions are pure Python with NO LLM calls.
"""

import os
from typing import Any, Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
    roc_curve, precision_recall_curve
)
from sklearn.model_selection import cross_val_score

from agenticml.ml.tools.utils import ensure_dir_exists


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    problem_type: str,
    metric: Optional[str] = None
) -> dict:
    """
    Evaluate a trained model on test data.
    
    Args:
        model: Trained model or pipeline
        X_test: Test features
        y_test: Test target
        problem_type: 'classification' or 'regression'
        metric: Primary metric to compute (default based on problem type)
    
    Returns:
        Dict with evaluation results
    """
    results = {
        "metrics": {},
        "predictions": None,
        "probabilities": None
    }
    
    # Get predictions
    y_pred = model.predict(X_test)
    results["predictions"] = y_pred
    
    if problem_type == "classification":
        results["metrics"] = compute_classification_metrics(y_test, y_pred, model, X_test)
        
        # Get probabilities if available
        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_test)
                results["probabilities"] = y_proba
            except Exception:
                pass
    
    elif problem_type == "regression":
        results["metrics"] = compute_regression_metrics(y_test, y_pred)
    
    # Set primary metric
    if metric:
        results["primary_metric"] = metric
        results["primary_score"] = results["metrics"].get(metric, 0)
    else:
        if problem_type == "classification":
            results["primary_metric"] = "f1"
            results["primary_score"] = results["metrics"].get("f1", 0)
        else:
            results["primary_metric"] = "rmse"
            results["primary_score"] = results["metrics"].get("rmse", 0)
    
    return results


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model: Any = None,
    X: np.ndarray = None
) -> dict:
    """
    Compute classification metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        model: Optional model for probability-based metrics
        X: Optional features for probability-based metrics
    
    Returns:
        Dict of metric name to value
    """
    metrics = {}
    
    # Basic metrics
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    
    # Handle binary vs multiclass
    n_classes = len(np.unique(y_true))
    average = "binary" if n_classes == 2 else "weighted"
    
    metrics["precision"] = float(precision_score(y_true, y_pred, average=average, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, average=average, zero_division=0))
    metrics["f1"] = float(f1_score(y_true, y_pred, average=average, zero_division=0))
    
    # Probability-based metrics
    if model is not None and X is not None and hasattr(model, "predict_proba"):
        try:
            y_proba = model.predict_proba(X)
            
            if n_classes == 2:
                # Binary classification
                metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba[:, 1]))
                metrics["log_loss"] = float(log_loss(y_true, y_proba))
            else:
                # Multiclass
                try:
                    metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted"))
                except Exception:
                    pass
                metrics["log_loss"] = float(log_loss(y_true, y_proba))
        except Exception:
            pass
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    metrics["confusion_matrix"] = cm.tolist()
    
    return metrics


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> dict:
    """
    Compute regression metrics.
    
    Args:
        y_true: True values
        y_pred: Predicted values
    
    Returns:
        Dict of metric name to value
    """
    metrics = {}
    
    metrics["mse"] = float(mean_squared_error(y_true, y_pred))
    metrics["rmse"] = float(np.sqrt(metrics["mse"]))
    metrics["mae"] = float(mean_absolute_error(y_true, y_pred))
    metrics["r2"] = float(r2_score(y_true, y_pred))
    
    # MAPE (handle zeros)
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
        metrics["mape"] = float(mape)
    
    return metrics


def cross_validate_model(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    cv: Any,
    scoring: str = "f1"
) -> dict:
    """
    Perform cross-validation on a model.
    
    Args:
        model: Model to evaluate
        X: Features
        y: Target
        cv: Cross-validation splitter
        scoring: Scoring metric
    
    Returns:
        Dict with CV results
    """
    # Map metric names to sklearn scoring
    scoring_map = {
        "f1": "f1_weighted",
        "accuracy": "accuracy",
        "precision": "precision_weighted",
        "recall": "recall_weighted",
        "roc_auc": "roc_auc",
        "rmse": "neg_root_mean_squared_error",
        "mae": "neg_mean_absolute_error",
        "r2": "r2"
    }
    
    sklearn_scoring = scoring_map.get(scoring, scoring)
    
    try:
        scores = cross_val_score(model, X, y, cv=cv, scoring=sklearn_scoring)
        
        # Handle negative scores (sklearn convention for error metrics)
        if sklearn_scoring.startswith("neg_"):
            scores = -scores
        
        return {
            "cv_scores": scores.tolist(),
            "cv_mean": float(scores.mean()),
            "cv_std": float(scores.std()),
            "cv_folds": len(scores),
            "scoring": scoring
        }
    except Exception as e:
        return {
            "cv_scores": [],
            "cv_mean": 0.0,
            "cv_std": 0.0,
            "cv_folds": 0,
            "scoring": scoring,
            "error": str(e)
        }


def generate_classification_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    output_dir: str,
    model_name: str,
    class_names: Optional[list[str]] = None
) -> list[str]:
    """
    Generate classification evaluation plots.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)
        output_dir: Directory to save plots
        model_name: Name of the model (for filenames)
        class_names: Optional class names for labels
    
    Returns:
        List of saved plot paths
    """
    ensure_dir_exists(output_dir + "/")
    plots = []
    
    # Infer class names if not provided
    if class_names is None:
        unique_classes = sorted(np.unique(np.concatenate([y_true, y_pred])))
        class_names = [str(c) for c in unique_classes]
    
    # Confusion matrix
    cm_path = os.path.join(output_dir, f"{model_name}_confusion_matrix.png")
    fig, ax = plt.subplots(figsize=(8, 6))
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    fig.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    plots.append(cm_path)
    
    # ROC curve (binary classification only)
    n_classes = len(np.unique(y_true))
    if y_proba is not None and n_classes == 2:
        roc_path = os.path.join(output_dir, f"{model_name}_roc_curve.png")
        fig, ax = plt.subplots(figsize=(8, 6))
        
        fpr, tpr, _ = roc_curve(y_true, y_proba[:, 1])
        auc = roc_auc_score(y_true, y_proba[:, 1])
        
        ax.plot(fpr, tpr, label=f"ROC (AUC = {auc:.3f})")
        ax.plot([0, 1], [0, 1], "k--", label="Random")
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve - {model_name}")
        ax.legend()
        plt.tight_layout()
        fig.savefig(roc_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plots.append(roc_path)
        
        # Precision-Recall curve
        pr_path = os.path.join(output_dir, f"{model_name}_precision_recall.png")
        fig, ax = plt.subplots(figsize=(8, 6))
        
        precision, recall, _ = precision_recall_curve(y_true, y_proba[:, 1])
        
        ax.plot(recall, precision)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Precision-Recall Curve - {model_name}")
        plt.tight_layout()
        fig.savefig(pr_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        plots.append(pr_path)
    
    return plots


def generate_regression_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str,
    model_name: str
) -> list[str]:
    """
    Generate regression evaluation plots.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        output_dir: Directory to save plots
        model_name: Name of the model
    
    Returns:
        List of saved plot paths
    """
    ensure_dir_exists(output_dir + "/")
    plots = []
    
    # Actual vs Predicted
    scatter_path = os.path.join(output_dir, f"{model_name}_actual_vs_predicted.png")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.scatter(y_true, y_pred, alpha=0.5)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], "r--", label="Perfect prediction")
    
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(f"Actual vs Predicted - {model_name}")
    ax.legend()
    plt.tight_layout()
    fig.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    plots.append(scatter_path)
    
    # Residuals plot
    residuals_path = os.path.join(output_dir, f"{model_name}_residuals.png")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    residuals = y_true - y_pred
    ax.scatter(y_pred, residuals, alpha=0.5)
    ax.axhline(y=0, color="r", linestyle="--")
    
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residuals")
    ax.set_title(f"Residuals Plot - {model_name}")
    plt.tight_layout()
    fig.savefig(residuals_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    plots.append(residuals_path)
    
    # Residuals distribution
    dist_path = os.path.join(output_dir, f"{model_name}_residuals_dist.png")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    sns.histplot(residuals, kde=True, ax=ax)
    ax.set_xlabel("Residuals")
    ax.set_title(f"Residuals Distribution - {model_name}")
    plt.tight_layout()
    fig.savefig(dist_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    plots.append(dist_path)
    
    return plots


def generate_feature_importance_plot(
    feature_names: list[str],
    importances: list[float],
    output_dir: str,
    model_name: str,
    top_n: int = 20
) -> str:
    """
    Generate feature importance plot.
    
    Args:
        feature_names: List of feature names
        importances: List of importance values
        output_dir: Directory to save plot
        model_name: Name of the model
        top_n: Number of top features to show
    
    Returns:
        Path to saved plot
    """
    ensure_dir_exists(output_dir + "/")
    
    # Create DataFrame and sort
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })
    df = df.sort_values("importance", ascending=True).tail(top_n)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.3)))
    
    ax.barh(df["feature"], df["importance"])
    ax.set_xlabel("Importance")
    ax.set_title(f"Feature Importance - {model_name}")
    plt.tight_layout()
    
    path = os.path.join(output_dir, f"{model_name}_feature_importance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    return path


def generate_model_comparison_plot(
    results: list[dict],
    metric: str,
    output_dir: str
) -> str:
    """
    Generate a comparison plot of multiple models.
    
    Args:
        results: List of evaluation result dicts
        metric: Metric to compare
        output_dir: Directory to save plot
    
    Returns:
        Path to saved plot
    """
    ensure_dir_exists(output_dir + "/")
    
    # Extract data
    model_names = []
    scores = []
    
    for result in results:
        if result.get("success", True):
            model_names.append(result.get("name", "Unknown"))
            scores.append(result.get("metrics", {}).get(metric, 0))
    
    # Sort by score
    sorted_pairs = sorted(zip(model_names, scores), key=lambda x: x[1], reverse=True)
    model_names, scores = zip(*sorted_pairs) if sorted_pairs else ([], [])
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, max(6, len(model_names) * 0.5)))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(model_names)))
    ax.barh(model_names, scores, color=colors)
    ax.set_xlabel(metric.upper())
    ax.set_title(f"Model Comparison - {metric.upper()}")
    
    # Add value labels
    for i, (name, score) in enumerate(zip(model_names, scores)):
        ax.text(score, i, f" {score:.4f}", va="center")
    
    plt.tight_layout()
    
    path = os.path.join(output_dir, f"model_comparison_{metric}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    
    return path


def select_best_model(
    results: list[dict],
    metric: str,
    higher_is_better: bool = True
) -> dict:
    """
    Select the best model based on a metric.
    
    Args:
        results: List of evaluation result dicts
        metric: Metric to use for selection
        higher_is_better: Whether higher metric values are better
    
    Returns:
        The best result dict
    """
    # Filter successful results
    successful = [r for r in results if r.get("success", True)]
    
    if not successful:
        return {}
    
    # Sort by metric
    sorted_results = sorted(
        successful,
        key=lambda x: x.get("metrics", {}).get(metric, float("-inf") if higher_is_better else float("inf")),
        reverse=higher_is_better
    )
    
    return sorted_results[0]


def is_metric_higher_better(metric: str) -> bool:
    """
    Determine if higher values are better for a metric.
    
    Args:
        metric: Metric name
    
    Returns:
        True if higher is better
    """
    lower_is_better = ["rmse", "mse", "mae", "mape", "log_loss"]
    return metric.lower() not in lower_is_better
