"""
Data cleaning tools for the ML pipeline.

All functions are pure Python with NO LLM calls.
"""

from typing import Optional, Any
import pandas as pd
import numpy as np


def apply_cleaning(
    df: pd.DataFrame,
    plan: dict
) -> tuple[pd.DataFrame, dict]:
    """
    Apply a cleaning plan to a DataFrame.
    
    The plan is a dict with a 'steps' key containing a list of cleaning operations.
    Each step has an 'action' and optional 'column' and 'params'.
    
    Supported actions:
    - drop_column: Remove a column
    - drop_columns: Remove multiple columns
    - fill_missing: Fill missing values
    - remove_duplicates: Remove duplicate rows
    - drop_missing_rows: Drop rows with missing values
    - clip_outliers: Clip outlier values
    - convert_dtype: Convert column data type
    - rename_column: Rename a column
    - drop_constant_columns: Remove columns with single value
    - drop_high_missing: Drop columns with high missing percentage
    
    Args:
        df: The DataFrame to clean
        plan: Cleaning plan dict with 'steps' list
    
    Returns:
        Tuple of (cleaned DataFrame, report dict)
    """
    df_cleaned = df.copy()
    
    report = {
        "steps_executed": [],
        "rows_before": len(df),
        "cols_before": len(df.columns),
        "rows_after": 0,
        "cols_after": 0,
        "changes_summary": ""
    }
    
    steps = plan.get("steps", [])
    
    for step in steps:
        action = step.get("action")
        column = step.get("column")
        params = step.get("params", {})
        
        step_result = {
            "action": action,
            "column": column,
            "params": params,
            "success": False,
            "message": ""
        }
        
        try:
            if action == "drop_column":
                df_cleaned, msg = _drop_column(df_cleaned, column)
                step_result["message"] = msg
                step_result["success"] = True
            
            elif action == "drop_columns":
                columns = params.get("columns", [])
                df_cleaned, msg = _drop_columns(df_cleaned, columns)
                step_result["message"] = msg
                step_result["success"] = True
            
            elif action == "fill_missing":
                strategy = params.get("strategy", "mean")
                fill_value = params.get("value")
                df_cleaned, msg = _fill_missing(df_cleaned, column, strategy, fill_value)
                step_result["message"] = msg
                step_result["success"] = True
            
            elif action == "remove_duplicates":
                subset = params.get("subset")
                keep = params.get("keep", "first")
                df_cleaned, msg = _remove_duplicates(df_cleaned, subset, keep)
                step_result["message"] = msg
                step_result["success"] = True
            
            elif action == "drop_missing_rows":
                subset = params.get("subset")
                how = params.get("how", "any")
                thresh = params.get("thresh")
                df_cleaned, msg = _drop_missing_rows(df_cleaned, subset, how, thresh)
                step_result["message"] = msg
                step_result["success"] = True
            
            elif action == "clip_outliers":
                method = params.get("method", "iqr")
                threshold = params.get("threshold", 1.5)
                df_cleaned, msg = _clip_outliers(df_cleaned, column, method, threshold)
                step_result["message"] = msg
                step_result["success"] = True
            
            elif action == "convert_dtype":
                dtype = params.get("dtype")
                df_cleaned, msg = _convert_dtype(df_cleaned, column, dtype)
                step_result["message"] = msg
                step_result["success"] = True
            
            elif action == "rename_column":
                new_name = params.get("new_name")
                df_cleaned, msg = _rename_column(df_cleaned, column, new_name)
                step_result["message"] = msg
                step_result["success"] = True
            
            elif action == "drop_constant_columns":
                df_cleaned, msg = _drop_constant_columns(df_cleaned)
                step_result["message"] = msg
                step_result["success"] = True
            
            elif action == "drop_high_missing":
                threshold = params.get("threshold", 0.5)
                df_cleaned, msg = _drop_high_missing(df_cleaned, threshold)
                step_result["message"] = msg
                step_result["success"] = True
            
            elif action == "lowercase_column_names":
                df_cleaned, msg = _lowercase_column_names(df_cleaned)
                step_result["message"] = msg
                step_result["success"] = True
            
            elif action == "strip_whitespace":
                df_cleaned, msg = _strip_whitespace(df_cleaned, column)
                step_result["message"] = msg
                step_result["success"] = True
            
            else:
                step_result["message"] = f"Unknown action: {action}"
                step_result["success"] = False
        
        except Exception as e:
            step_result["message"] = f"Error: {str(e)}"
            step_result["success"] = False
        
        report["steps_executed"].append(step_result)
    
    report["rows_after"] = len(df_cleaned)
    report["cols_after"] = len(df_cleaned.columns)
    
    rows_removed = report["rows_before"] - report["rows_after"]
    cols_removed = report["cols_before"] - report["cols_after"]
    
    report["changes_summary"] = (
        f"Removed {rows_removed} rows ({rows_removed/report['rows_before']*100:.1f}%) "
        f"and {cols_removed} columns. "
        f"Final shape: {report['rows_after']} rows x {report['cols_after']} columns."
    )
    
    return df_cleaned, report


def _drop_column(df: pd.DataFrame, column: str) -> tuple[pd.DataFrame, str]:
    """Drop a single column."""
    if column not in df.columns:
        return df, f"Column '{column}' not found, skipping"
    
    df = df.drop(columns=[column])
    return df, f"Dropped column '{column}'"


def _drop_columns(df: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, str]:
    """Drop multiple columns."""
    existing = [c for c in columns if c in df.columns]
    missing = [c for c in columns if c not in df.columns]
    
    if existing:
        df = df.drop(columns=existing)
    
    msg = f"Dropped {len(existing)} columns"
    if missing:
        msg += f" (skipped {len(missing)} not found)"
    
    return df, msg


def _fill_missing(
    df: pd.DataFrame,
    column: Optional[str],
    strategy: str,
    fill_value: Any = None
) -> tuple[pd.DataFrame, str]:
    """Fill missing values in a column or all columns."""
    if column:
        columns = [column] if column in df.columns else []
    else:
        columns = df.columns.tolist()
    
    filled_count = 0
    
    for col in columns:
        n_missing = df[col].isna().sum()
        if n_missing == 0:
            continue
        
        if strategy == "value" and fill_value is not None:
            df[col] = df[col].fillna(fill_value)
        elif strategy == "mean" and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mean())
        elif strategy == "median" and pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        elif strategy == "mode":
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val[0])
        elif strategy == "ffill":
            df[col] = df[col].ffill()
        elif strategy == "bfill":
            df[col] = df[col].bfill()
        elif strategy == "zero":
            df[col] = df[col].fillna(0)
        elif strategy == "empty_string":
            df[col] = df[col].fillna("")
        else:
            continue
        
        filled_count += n_missing
    
    return df, f"Filled {filled_count} missing values using '{strategy}' strategy"


def _remove_duplicates(
    df: pd.DataFrame,
    subset: Optional[list[str]],
    keep: str
) -> tuple[pd.DataFrame, str]:
    """Remove duplicate rows."""
    n_before = len(df)
    df = df.drop_duplicates(subset=subset, keep=keep)
    n_removed = n_before - len(df)
    
    return df, f"Removed {n_removed} duplicate rows"


def _drop_missing_rows(
    df: pd.DataFrame,
    subset: Optional[list[str]],
    how: str,
    thresh: Optional[int]
) -> tuple[pd.DataFrame, str]:
    """Drop rows with missing values."""
    n_before = len(df)
    
    if thresh is not None:
        df = df.dropna(subset=subset, thresh=thresh)
    else:
        df = df.dropna(subset=subset, how=how)
    
    n_removed = n_before - len(df)
    
    return df, f"Dropped {n_removed} rows with missing values"


def _clip_outliers(
    df: pd.DataFrame,
    column: str,
    method: str,
    threshold: float
) -> tuple[pd.DataFrame, str]:
    """Clip outlier values in a column."""
    if column not in df.columns:
        return df, f"Column '{column}' not found"
    
    if not pd.api.types.is_numeric_dtype(df[column]):
        return df, f"Column '{column}' is not numeric, skipping"
    
    col_data = df[column].dropna()
    
    if method == "iqr":
        q1 = col_data.quantile(0.25)
        q3 = col_data.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - threshold * iqr
        upper = q3 + threshold * iqr
    elif method == "zscore":
        mean = col_data.mean()
        std = col_data.std()
        lower = mean - threshold * std
        upper = mean + threshold * std
    elif method == "percentile":
        lower = col_data.quantile(threshold / 100)
        upper = col_data.quantile(1 - threshold / 100)
    else:
        return df, f"Unknown method: {method}"
    
    n_clipped = ((df[column] < lower) | (df[column] > upper)).sum()
    df[column] = df[column].clip(lower=lower, upper=upper)
    
    return df, f"Clipped {n_clipped} outliers in '{column}' to [{lower:.2f}, {upper:.2f}]"


def _convert_dtype(
    df: pd.DataFrame,
    column: str,
    dtype: str
) -> tuple[pd.DataFrame, str]:
    """Convert column data type."""
    if column not in df.columns:
        return df, f"Column '{column}' not found"
    
    try:
        if dtype == "int":
            df[column] = pd.to_numeric(df[column], errors="coerce").astype("Int64")
        elif dtype == "float":
            df[column] = pd.to_numeric(df[column], errors="coerce")
        elif dtype == "str":
            df[column] = df[column].astype(str)
        elif dtype == "category":
            df[column] = df[column].astype("category")
        elif dtype == "datetime":
            df[column] = pd.to_datetime(df[column], errors="coerce")
        elif dtype == "bool":
            df[column] = df[column].astype(bool)
        else:
            df[column] = df[column].astype(dtype)
        
        return df, f"Converted '{column}' to {dtype}"
    except Exception as e:
        return df, f"Failed to convert '{column}' to {dtype}: {str(e)}"


def _rename_column(
    df: pd.DataFrame,
    old_name: str,
    new_name: str
) -> tuple[pd.DataFrame, str]:
    """Rename a column."""
    if old_name not in df.columns:
        return df, f"Column '{old_name}' not found"
    
    if new_name in df.columns:
        return df, f"Column '{new_name}' already exists"
    
    df = df.rename(columns={old_name: new_name})
    return df, f"Renamed '{old_name}' to '{new_name}'"


def _drop_constant_columns(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Drop columns with only one unique value."""
    constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
    
    if constant_cols:
        df = df.drop(columns=constant_cols)
        return df, f"Dropped {len(constant_cols)} constant columns: {constant_cols}"
    
    return df, "No constant columns found"


def _drop_high_missing(
    df: pd.DataFrame,
    threshold: float
) -> tuple[pd.DataFrame, str]:
    """Drop columns with missing percentage above threshold."""
    missing_pct = df.isna().mean()
    high_missing = missing_pct[missing_pct > threshold].index.tolist()
    
    if high_missing:
        df = df.drop(columns=high_missing)
        return df, f"Dropped {len(high_missing)} columns with >{threshold*100:.0f}% missing"
    
    return df, f"No columns with >{threshold*100:.0f}% missing values"


def _lowercase_column_names(df: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Convert all column names to lowercase."""
    df.columns = df.columns.str.lower()
    return df, "Converted all column names to lowercase"


def _strip_whitespace(
    df: pd.DataFrame,
    column: Optional[str]
) -> tuple[pd.DataFrame, str]:
    """Strip whitespace from string columns."""
    if column:
        columns = [column] if column in df.columns else []
    else:
        columns = df.select_dtypes(include=["object"]).columns.tolist()
    
    for col in columns:
        if df[col].dtype == "object":
            df[col] = df[col].str.strip()
    
    return df, f"Stripped whitespace from {len(columns)} columns"


def get_cleaning_stats(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame
) -> dict:
    """
    Get statistics comparing before and after cleaning.
    
    Args:
        df_before: DataFrame before cleaning
        df_after: DataFrame after cleaning
    
    Returns:
        Dict with comparison statistics
    """
    stats = {
        "rows_before": len(df_before),
        "rows_after": len(df_after),
        "rows_removed": len(df_before) - len(df_after),
        "rows_removed_pct": ((len(df_before) - len(df_after)) / len(df_before) * 100) if len(df_before) > 0 else 0,
        "cols_before": len(df_before.columns),
        "cols_after": len(df_after.columns),
        "cols_removed": len(df_before.columns) - len(df_after.columns),
        "missing_before": df_before.isna().sum().sum(),
        "missing_after": df_after.isna().sum().sum(),
        "memory_before_mb": df_before.memory_usage(deep=True).sum() / 1024 / 1024,
        "memory_after_mb": df_after.memory_usage(deep=True).sum() / 1024 / 1024
    }
    
    stats["missing_removed"] = stats["missing_before"] - stats["missing_after"]
    stats["memory_saved_mb"] = stats["memory_before_mb"] - stats["memory_after_mb"]
    
    return stats


def suggest_cleaning_steps(profile: dict) -> list[dict]:
    """
    Suggest cleaning steps based on a data profile.
    
    This is a deterministic suggestion based on profile statistics.
    The agent will use LLM to refine these suggestions.
    
    Args:
        profile: Profile dict from profile_dataframe
    
    Returns:
        List of suggested cleaning step dicts
    """
    suggestions = []
    
    # Suggest dropping constant columns
    if profile.get("constant_columns"):
        suggestions.append({
            "action": "drop_constant_columns",
            "rationale": f"Found {len(profile['constant_columns'])} constant columns with no variance"
        })
    
    # Suggest handling high missing columns
    for col, pct in profile.get("missing_percentages", {}).items():
        if pct > 50:
            suggestions.append({
                "action": "drop_column",
                "column": col,
                "rationale": f"Column has {pct:.1f}% missing values"
            })
        elif pct > 0:
            # Suggest fill strategy based on column type
            if col in profile.get("numeric_columns", []):
                suggestions.append({
                    "action": "fill_missing",
                    "column": col,
                    "params": {"strategy": "median"},
                    "rationale": f"Fill {pct:.1f}% missing with median (numeric column)"
                })
            else:
                suggestions.append({
                    "action": "fill_missing",
                    "column": col,
                    "params": {"strategy": "mode"},
                    "rationale": f"Fill {pct:.1f}% missing with mode (categorical column)"
                })
    
    # Suggest removing duplicates if any
    suggestions.append({
        "action": "remove_duplicates",
        "params": {"keep": "first"},
        "rationale": "Remove any duplicate rows"
    })
    
    return suggestions
