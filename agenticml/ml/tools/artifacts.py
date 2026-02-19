"""
Artifact management tools for saving and loading pipeline artifacts.

All functions are pure Python with NO LLM calls.
"""

import os
import json
from datetime import datetime
from typing import Any, Optional
import joblib
import pandas as pd

from agenticml.ml.tools.utils import (
    ensure_dir_exists,
    safe_json_serialize,
    get_timestamp
)


def save_model(
    model: Any,
    path: str,
    metadata: Optional[dict] = None
) -> dict:
    """
    Save a trained model using joblib.
    
    Args:
        model: The model object to save
        path: Output path (should end in .joblib)
        metadata: Optional metadata to save alongside
    
    Returns:
        Dict with save information
    """
    ensure_dir_exists(path)
    
    # Save the model
    joblib.dump(model, path)
    
    # Save metadata if provided
    if metadata:
        meta_path = path.replace(".joblib", "_metadata.json")
        save_json(meta_path, metadata)
    
    return {
        "path": path,
        "file_size_bytes": os.path.getsize(path),
        "saved_at": get_timestamp()
    }


def load_model(path: str) -> Any:
    """
    Load a model from disk.
    
    Args:
        path: Path to the saved model
    
    Returns:
        The loaded model object
    """
    return joblib.load(path)


def save_json(path: str, data: Any, indent: int = 2) -> dict:
    """
    Save data to a JSON file.
    
    Args:
        path: Output path
        data: Data to save (will be serialized)
        indent: JSON indentation level
    
    Returns:
        Dict with save information
    """
    ensure_dir_exists(path)
    
    serialized = safe_json_serialize(data)
    
    with open(path, "w") as f:
        json.dump(serialized, f, indent=indent, default=str)
    
    return {
        "path": path,
        "file_size_bytes": os.path.getsize(path),
        "saved_at": get_timestamp()
    }


def load_json(path: str) -> Any:
    """
    Load data from a JSON file.
    
    Args:
        path: Path to the JSON file
    
    Returns:
        Loaded data
    """
    with open(path, "r") as f:
        return json.load(f)


def save_plot(
    fig: Any,
    path: str,
    dpi: int = 150,
    bbox_inches: str = "tight"
) -> dict:
    """
    Save a matplotlib figure to disk.
    
    Args:
        fig: Matplotlib figure object
        path: Output path (should end in .png or .pdf)
        dpi: Resolution for raster formats
        bbox_inches: Bounding box setting
    
    Returns:
        Dict with save information
    """
    ensure_dir_exists(path)
    
    fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
    
    return {
        "path": path,
        "file_size_bytes": os.path.getsize(path),
        "saved_at": get_timestamp()
    }


def save_dataframe_artifact(
    df: pd.DataFrame,
    path: str,
    index: bool = False
) -> dict:
    """
    Save a DataFrame as a CSV artifact.
    
    Args:
        df: DataFrame to save
        path: Output path
        index: Whether to include index
    
    Returns:
        Dict with save information
    """
    ensure_dir_exists(path)
    
    df.to_csv(path, index=index)
    
    return {
        "path": path,
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "file_size_bytes": os.path.getsize(path),
        "saved_at": get_timestamp()
    }


def create_run_manifest(
    run_dir: str,
    state: dict,
    config: dict
) -> dict:
    """
    Create a manifest file for a pipeline run.
    
    The manifest contains all metadata about the run for reproducibility.
    
    Args:
        run_dir: The run directory
        state: The final pipeline state
        config: The configuration used
    
    Returns:
        The manifest dict
    """
    manifest = {
        "run_id": state.get("run_id"),
        "created_at": get_timestamp(),
        "started_at": state.get("started_at"),
        "file_path": state.get("file_path"),
        "target": state.get("target"),
        "problem_type": state.get("problem_type"),
        "metric": state.get("user_metric"),
        "iterations": state.get("iteration", 0),
        "stop_reason": state.get("stop_reason"),
        "config": safe_json_serialize(config),
        "artifacts": state.get("artifacts", []),
        "best_model": state.get("best_model"),
        "evaluation_results": state.get("evaluation_results", []),
        "critic_issues": state.get("critic_issues", []),
        "decision_log": state.get("decision_log", []),
        "errors": state.get("errors", [])
    }
    
    manifest_path = os.path.join(run_dir, "run_manifest.json")
    save_json(manifest_path, manifest)
    
    return manifest


def save_preprocessing_pipeline(
    pipeline: Any,
    path: str,
    feature_names: Optional[list[str]] = None
) -> dict:
    """
    Save a sklearn preprocessing pipeline.
    
    Args:
        pipeline: The sklearn pipeline/transformer
        path: Output path
        feature_names: Optional list of feature names
    
    Returns:
        Dict with save information
    """
    ensure_dir_exists(path)
    
    # Save the pipeline
    joblib.dump(pipeline, path)
    
    # Save feature names if provided
    if feature_names:
        names_path = path.replace(".joblib", "_features.json")
        save_json(names_path, {"feature_names": feature_names})
    
    return {
        "path": path,
        "file_size_bytes": os.path.getsize(path),
        "saved_at": get_timestamp()
    }


def load_preprocessing_pipeline(path: str) -> Any:
    """
    Load a sklearn preprocessing pipeline.
    
    Args:
        path: Path to the saved pipeline
    
    Returns:
        The loaded pipeline
    """
    return joblib.load(path)


def save_report(
    content: str,
    path: str
) -> dict:
    """
    Save a markdown report.
    
    Args:
        content: The markdown content
        path: Output path
    
    Returns:
        Dict with save information
    """
    ensure_dir_exists(path)
    
    with open(path, "w") as f:
        f.write(content)
    
    return {
        "path": path,
        "file_size_bytes": os.path.getsize(path),
        "saved_at": get_timestamp()
    }


def list_artifacts(run_dir: str) -> list[dict]:
    """
    List all artifacts in a run directory.
    
    Args:
        run_dir: The run directory to scan
    
    Returns:
        List of artifact info dicts
    """
    artifacts = []
    
    for root, dirs, files in os.walk(run_dir):
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, run_dir)
            
            # Determine artifact type
            ext = os.path.splitext(file)[1].lower()
            if ext == ".joblib":
                artifact_type = "model"
            elif ext == ".json":
                artifact_type = "json"
            elif ext in [".png", ".pdf", ".jpg"]:
                artifact_type = "plot"
            elif ext == ".csv":
                artifact_type = "data"
            elif ext == ".md":
                artifact_type = "report"
            elif ext == ".log":
                artifact_type = "log"
            else:
                artifact_type = "other"
            
            artifacts.append({
                "name": file,
                "path": file_path,
                "relative_path": rel_path,
                "artifact_type": artifact_type,
                "file_size_bytes": os.path.getsize(file_path)
            })
    
    return artifacts


def cleanup_old_runs(
    runs_dir: str,
    keep_last_n: int = 10
) -> list[str]:
    """
    Clean up old run directories, keeping only the most recent ones.
    
    Args:
        runs_dir: The base runs directory
        keep_last_n: Number of recent runs to keep
    
    Returns:
        List of deleted run directories
    """
    import shutil
    
    if not os.path.exists(runs_dir):
        return []
    
    # Get all run directories with their modification times
    runs = []
    for name in os.listdir(runs_dir):
        path = os.path.join(runs_dir, name)
        if os.path.isdir(path):
            mtime = os.path.getmtime(path)
            runs.append((path, mtime))
    
    # Sort by modification time (newest first)
    runs.sort(key=lambda x: x[1], reverse=True)
    
    # Delete old runs
    deleted = []
    for path, _ in runs[keep_last_n:]:
        shutil.rmtree(path)
        deleted.append(path)
    
    return deleted
