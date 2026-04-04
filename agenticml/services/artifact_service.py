"""
Artifact management service.

Thin wrapper around the existing tools/artifacts.py functions so that the
new agent layer has a clean import path.  The underlying logic is preserved.
"""

import os
import json
from typing import Any, Optional

import joblib
import pandas as pd

from agenticml.ml.tools.utils import (
    ensure_dir_exists,
    safe_json_serialize,
    get_timestamp,
)


def save_model(model: Any, path: str, metadata: Optional[dict] = None) -> dict:
    ensure_dir_exists(path)
    joblib.dump(model, path)
    if metadata:
        meta_path = path.replace(".joblib", "_metadata.json")
        save_json(meta_path, metadata)
    return {"path": path, "file_size_bytes": os.path.getsize(path), "saved_at": get_timestamp()}


def load_model(path: str) -> Any:
    return joblib.load(path)


def save_json(path: str, data: Any, indent: int = 2) -> dict:
    ensure_dir_exists(path)
    serialized = safe_json_serialize(data)
    with open(path, "w") as f:
        json.dump(serialized, f, indent=indent, default=str)
    return {"path": path, "file_size_bytes": os.path.getsize(path), "saved_at": get_timestamp()}


def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def save_plot(fig: Any, path: str, dpi: int = 150, bbox_inches: str = "tight") -> dict:
    ensure_dir_exists(path)
    fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
    return {"path": path, "file_size_bytes": os.path.getsize(path), "saved_at": get_timestamp()}


def save_dataframe_artifact(df: pd.DataFrame, path: str, index: bool = False) -> dict:
    ensure_dir_exists(path)
    df.to_csv(path, index=index)
    return {
        "path": path,
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "file_size_bytes": os.path.getsize(path),
        "saved_at": get_timestamp(),
    }


def save_preprocessing_pipeline(pipeline: Any, path: str, feature_names: Optional[list[str]] = None) -> dict:
    ensure_dir_exists(path)
    joblib.dump(pipeline, path)
    if feature_names:
        names_path = path.replace(".joblib", "_features.json")
        save_json(names_path, {"feature_names": feature_names})
    return {"path": path, "file_size_bytes": os.path.getsize(path), "saved_at": get_timestamp()}


def load_preprocessing_pipeline(path: str) -> Any:
    return joblib.load(path)


def save_report(content: str, path: str) -> dict:
    ensure_dir_exists(path)
    with open(path, "w") as f:
        f.write(content)
    return {"path": path, "file_size_bytes": os.path.getsize(path), "saved_at": get_timestamp()}


def create_run_manifest(run_dir: str, state: dict, config: dict) -> dict:
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
        "decision_log": state.get("decision_log", []),
        "errors": state.get("errors", []),
    }
    manifest_path = os.path.join(run_dir, "run_manifest.json")
    save_json(manifest_path, manifest)
    return manifest


def list_artifacts(run_dir: str) -> list[dict]:
    artifacts = []
    for root, _dirs, files in os.walk(run_dir):
        for file in files:
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, run_dir)
            ext = os.path.splitext(file)[1].lower()
            type_map = {
                ".joblib": "model", ".json": "json", ".png": "plot",
                ".pdf": "plot", ".jpg": "plot", ".csv": "data",
                ".md": "report", ".log": "log",
            }
            artifacts.append({
                "name": file,
                "path": file_path,
                "relative_path": rel_path,
                "artifact_type": type_map.get(ext, "other"),
                "file_size_bytes": os.path.getsize(file_path),
            })
    return artifacts
