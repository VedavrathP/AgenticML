"""
Utility functions for the ML pipeline.

This module provides logging, directory management, and helper functions
used across the pipeline.
"""

import os
import uuid
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
import json


# ============================================================================
# Logging Setup
# ============================================================================

def setup_logging(
    run_id: str,
    run_dir: str,
    level: int = logging.INFO
) -> logging.Logger:
    """
    Set up logging for a pipeline run.
    
    Creates both console and file handlers for comprehensive logging.
    """
    logger = logging.getLogger(f"ml_pipeline.{run_id}")
    logger.setLevel(level)
    
    # Clear existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    log_file = os.path.join(run_dir, "pipeline.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)  # More verbose in file
    file_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    return logger


def get_logger(run_id: str) -> logging.Logger:
    """Get the logger for a specific run."""
    return logging.getLogger(f"ml_pipeline.{run_id}")


# ============================================================================
# Run Directory Management
# ============================================================================

def generate_run_id() -> str:
    """Generate a unique run ID."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:8]
    return f"{timestamp}_{short_uuid}"


def create_run_directory(base_dir: str, run_id: str) -> str:
    """
    Create the directory structure for a pipeline run.
    
    Structure:
        runs/<run_id>/
            raw/
            cleaned/
            features/
            models/
            plots/
            metrics/
    """
    run_dir = os.path.join(base_dir, run_id)
    
    subdirs = [
        "raw",
        "cleaned",
        "features",
        "models",
        "plots",
        "metrics"
    ]
    
    for subdir in subdirs:
        os.makedirs(os.path.join(run_dir, subdir), exist_ok=True)
    
    return run_dir


def get_run_subdir(run_dir: str, subdir: str) -> str:
    """Get the path to a subdirectory within the run directory."""
    path = os.path.join(run_dir, subdir)
    os.makedirs(path, exist_ok=True)
    return path


# ============================================================================
# File Path Helpers
# ============================================================================

def get_file_extension(file_path: str) -> str:
    """Get the lowercase file extension."""
    return Path(file_path).suffix.lower()


def is_csv_file(file_path: str) -> bool:
    """Check if a file is a CSV file."""
    return get_file_extension(file_path) == ".csv"


def is_excel_file(file_path: str) -> bool:
    """Check if a file is an Excel file."""
    return get_file_extension(file_path) in [".xlsx", ".xls"]


def ensure_dir_exists(file_path: str) -> None:
    """Ensure the directory for a file path exists."""
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)


# ============================================================================
# Timestamp Helpers
# ============================================================================

def get_timestamp() -> str:
    """Get current timestamp in ISO format."""
    return datetime.now().isoformat()


def get_timestamp_str() -> str:
    """Get current timestamp as a filename-safe string."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def format_duration(seconds: float) -> str:
    """Format a duration in seconds to a human-readable string."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h"


# ============================================================================
# JSON Helpers
# ============================================================================

def safe_json_serialize(obj: Any) -> Any:
    """
    Safely serialize an object to JSON-compatible format.
    
    Handles numpy arrays, pandas objects, and other non-serializable types.
    """
    import numpy as np
    import pandas as pd
    
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    elif isinstance(obj, dict):
        return {k: safe_json_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [safe_json_serialize(item) for item in obj]
    elif hasattr(obj, "__dict__"):
        return safe_json_serialize(obj.__dict__)
    else:
        return obj


def save_json_safe(path: str, data: Any, indent: int = 2) -> None:
    """Save data to JSON file with safe serialization."""
    ensure_dir_exists(path)
    serialized = safe_json_serialize(data)
    with open(path, "w") as f:
        json.dump(serialized, f, indent=indent, default=str)


def load_json_safe(path: str) -> Any:
    """Load data from JSON file."""
    with open(path, "r") as f:
        return json.load(f)


# ============================================================================
# Data Type Helpers
# ============================================================================

def infer_dtype_category(dtype_str: str) -> str:
    """
    Infer a high-level category from a pandas dtype string.
    
    Returns: 'numeric', 'categorical', 'datetime', 'text', or 'other'
    """
    dtype_lower = dtype_str.lower()
    
    if any(t in dtype_lower for t in ["int", "float", "numeric"]):
        return "numeric"
    elif any(t in dtype_lower for t in ["datetime", "timestamp", "date"]):
        return "datetime"
    elif any(t in dtype_lower for t in ["bool"]):
        return "categorical"
    elif any(t in dtype_lower for t in ["object", "string", "category"]):
        return "text_or_categorical"
    else:
        return "other"


def is_likely_id_column(column_name: str, cardinality: int, n_rows: int) -> bool:
    """
    Check if a column is likely an ID column.
    
    Heuristics:
    - Column name contains 'id', 'key', 'index'
    - Cardinality equals number of rows (all unique)
    """
    name_lower = column_name.lower()
    id_patterns = ["id", "_id", "key", "index", "uuid", "guid"]
    
    has_id_name = any(pattern in name_lower for pattern in id_patterns)
    is_all_unique = cardinality == n_rows
    
    return has_id_name or is_all_unique


def truncate_string(s: str, max_length: int = 100) -> str:
    """Truncate a string to a maximum length."""
    if len(s) <= max_length:
        return s
    return s[:max_length - 3] + "..."


# ============================================================================
# Validation Helpers
# ============================================================================

def validate_file_exists(file_path: str) -> tuple[bool, str]:
    """Validate that a file exists and is readable."""
    if not os.path.exists(file_path):
        return False, f"File not found: {file_path}"
    if not os.path.isfile(file_path):
        return False, f"Path is not a file: {file_path}"
    if not os.access(file_path, os.R_OK):
        return False, f"File is not readable: {file_path}"
    return True, ""


def validate_column_exists(columns: list[str], column_name: str) -> tuple[bool, str]:
    """Validate that a column exists in the dataset."""
    if column_name not in columns:
        return False, f"Column '{column_name}' not found. Available columns: {columns}"
    return True, ""


def validate_problem_type(problem_type: str) -> tuple[bool, str]:
    """Validate the problem type."""
    valid_types = ["classification", "regression"]
    if problem_type not in valid_types:
        return False, f"Invalid problem type '{problem_type}'. Must be one of: {valid_types}"
    return True, ""


# ============================================================================
# Memory and Performance Helpers
# ============================================================================

def estimate_memory_usage(n_rows: int, n_cols: int, avg_col_size: int = 8) -> str:
    """Estimate memory usage for a dataset."""
    bytes_estimate = n_rows * n_cols * avg_col_size
    
    if bytes_estimate < 1024:
        return f"{bytes_estimate} B"
    elif bytes_estimate < 1024 ** 2:
        return f"{bytes_estimate / 1024:.2f} KB"
    elif bytes_estimate < 1024 ** 3:
        return f"{bytes_estimate / (1024 ** 2):.2f} MB"
    else:
        return f"{bytes_estimate / (1024 ** 3):.2f} GB"
