"""
Data I/O tools for loading and saving datasets.

All functions are pure Python with NO LLM calls.
"""

import os
from typing import Optional, Tuple
import pandas as pd
import numpy as np

from agenticml.ml.tools.utils import (
    get_file_extension,
    is_csv_file,
    is_excel_file,
    ensure_dir_exists,
    validate_file_exists
)


def load_data(
    file_path: str,
    sheet_name: Optional[str] = None,
    nrows: Optional[int] = None
) -> Tuple[pd.DataFrame, dict]:
    """
    Load data from CSV or Excel file.
    
    Args:
        file_path: Path to the data file
        sheet_name: For Excel files, the sheet to load (default: first sheet)
        nrows: Number of rows to load (default: all)
    
    Returns:
        Tuple of (DataFrame, metadata dict)
    
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is not supported
    """
    # Validate file exists
    valid, error = validate_file_exists(file_path)
    if not valid:
        raise FileNotFoundError(error)
    
    ext = get_file_extension(file_path)
    metadata = {
        "file_path": file_path,
        "file_extension": ext,
        "file_size_bytes": os.path.getsize(file_path)
    }
    
    # Load based on file type
    if is_csv_file(file_path):
        df = _load_csv(file_path, nrows)
        metadata["file_type"] = "csv"
    elif is_excel_file(file_path):
        df = _load_excel(file_path, sheet_name, nrows)
        metadata["file_type"] = "excel"
        metadata["sheet_name"] = sheet_name
    else:
        raise ValueError(f"Unsupported file format: {ext}. Supported: .csv, .xlsx, .xls")
    
    metadata["n_rows"] = len(df)
    metadata["n_cols"] = len(df.columns)
    metadata["columns"] = df.columns.tolist()
    
    return df, metadata


def _load_csv(
    file_path: str,
    nrows: Optional[int] = None
) -> pd.DataFrame:
    """Load a CSV file with automatic encoding detection."""
    # Try common encodings
    encodings = ["utf-8", "latin-1", "cp1252", "iso-8859-1"]
    
    for encoding in encodings:
        try:
            df = pd.read_csv(
                file_path,
                encoding=encoding,
                nrows=nrows,
                low_memory=False
            )
            return df
        except UnicodeDecodeError:
            continue
    
    # If all encodings fail, try with errors='replace'
    return pd.read_csv(
        file_path,
        encoding="utf-8",
        errors="replace",
        nrows=nrows,
        low_memory=False
    )


def _load_excel(
    file_path: str,
    sheet_name: Optional[str] = None,
    nrows: Optional[int] = None
) -> pd.DataFrame:
    """Load an Excel file."""
    return pd.read_excel(
        file_path,
        sheet_name=sheet_name or 0,
        nrows=nrows
    )


def save_dataframe(
    df: pd.DataFrame,
    path: str,
    index: bool = False
) -> dict:
    """
    Save a DataFrame to CSV.
    
    Args:
        df: DataFrame to save
        path: Output path
        index: Whether to include the index
    
    Returns:
        Metadata about the saved file
    """
    ensure_dir_exists(path)
    df.to_csv(path, index=index)
    
    return {
        "path": path,
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "file_size_bytes": os.path.getsize(path)
    }


def load_dataframe(path: str) -> pd.DataFrame:
    """
    Load a DataFrame from CSV.
    
    Args:
        path: Path to the CSV file
    
    Returns:
        Loaded DataFrame
    """
    return pd.read_csv(path, low_memory=False)


def get_sample_data(
    df: pd.DataFrame,
    n_rows: int = 5,
    random: bool = False
) -> pd.DataFrame:
    """
    Get a sample of the data.
    
    Args:
        df: Source DataFrame
        n_rows: Number of rows to sample
        random: If True, sample randomly; if False, take first n rows
    
    Returns:
        Sample DataFrame
    """
    if random:
        return df.sample(n=min(n_rows, len(df)))
    return df.head(n_rows)


def resolve_column_name(df: pd.DataFrame, name: str) -> str:
    """
    Return the actual column label in ``df`` for ``name``.

    If ``name`` is not present (e.g. after ``rename_column`` or
    ``lowercase_column_names``), match case-insensitively so ``state['target']``
    stays aligned with the cleaned DataFrame.
    """
    if name in df.columns:
        return name
    lower = name.lower()
    for col in df.columns:
        if str(col).lower() == lower:
            return col
    return name


def split_features_target(
    df: pd.DataFrame,
    target_column: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Split DataFrame into features and target.
    
    Args:
        df: Source DataFrame
        target_column: Name of the target column
    
    Returns:
        Tuple of (features DataFrame, target Series)
    
    Raises:
        ValueError: If target column doesn't exist
    """
    target_column = resolve_column_name(df, target_column)
    if target_column not in df.columns:
        raise ValueError(f"Target column '{target_column}' not found in DataFrame")
    
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    return X, y


def combine_features_target(
    X: pd.DataFrame,
    y: pd.Series,
    target_name: str
) -> pd.DataFrame:
    """
    Combine features and target back into a single DataFrame.
    
    Args:
        X: Features DataFrame
        y: Target Series
        target_name: Name for the target column
    
    Returns:
        Combined DataFrame
    """
    df = X.copy()
    df[target_name] = y.values
    return df


def get_column_info(df: pd.DataFrame) -> dict:
    """
    Get detailed information about DataFrame columns.
    
    Returns a dict with column names as keys and info dicts as values.
    """
    info = {}
    
    for col in df.columns:
        col_data = df[col]
        info[col] = {
            "dtype": str(col_data.dtype),
            "n_unique": col_data.nunique(),
            "n_missing": col_data.isna().sum(),
            "missing_pct": (col_data.isna().sum() / len(df)) * 100,
            "sample_values": col_data.dropna().head(3).tolist()
        }
        
        # Add numeric stats if applicable
        if pd.api.types.is_numeric_dtype(col_data):
            info[col]["min"] = float(col_data.min()) if not col_data.isna().all() else None
            info[col]["max"] = float(col_data.max()) if not col_data.isna().all() else None
            info[col]["mean"] = float(col_data.mean()) if not col_data.isna().all() else None
            info[col]["std"] = float(col_data.std()) if not col_data.isna().all() else None
    
    return info


def detect_datetime_columns(df: pd.DataFrame) -> list[str]:
    """
    Detect columns that might be datetime.
    
    Checks both dtype and attempts to parse object columns.
    """
    datetime_cols = []
    
    for col in df.columns:
        # Already datetime
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            datetime_cols.append(col)
            continue
        
        # Try to parse object columns
        if df[col].dtype == "object":
            try:
                # Sample a few non-null values
                sample = df[col].dropna().head(100)
                if len(sample) > 0:
                    parsed = pd.to_datetime(sample, errors="coerce")
                    # If most values parse successfully, it's likely datetime
                    if parsed.notna().mean() > 0.8:
                        datetime_cols.append(col)
            except Exception:
                pass
    
    return datetime_cols


def infer_separator(file_path: str) -> str:
    """
    Infer the separator used in a CSV file.
    
    Args:
        file_path: Path to the CSV file
    
    Returns:
        The detected separator character
    """
    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
        first_line = f.readline()
    
    # Count potential separators
    separators = [",", ";", "\t", "|"]
    counts = {sep: first_line.count(sep) for sep in separators}
    
    # Return the most common one
    return max(counts, key=counts.get)
