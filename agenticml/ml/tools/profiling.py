"""
Data profiling tools for understanding datasets.

All functions are pure Python with NO LLM calls.
"""

import re
from typing import Optional
import pandas as pd
import numpy as np

from agenticml.ml.config import PII_PATTERNS, PII_COLUMN_PATTERNS


def profile_dataframe(
    df: pd.DataFrame,
    target: Optional[str] = None
) -> dict:
    """
    Generate a comprehensive profile of a DataFrame.
    
    Args:
        df: The DataFrame to profile
        target: Optional target column name
    
    Returns:
        Dict containing profile information
    """
    profile = {
        "n_rows": len(df),
        "n_cols": len(df.columns),
        "columns": df.columns.tolist(),
        "dtypes": {col: str(df[col].dtype) for col in df.columns},
        "missing_counts": {},
        "missing_percentages": {},
        "cardinality": {},
        "numeric_columns": [],
        "categorical_columns": [],
        "datetime_columns": [],
        "text_columns": [],
        "constant_columns": [],
        "high_cardinality_columns": [],
        "sample_values": {},
        "numeric_stats": {},
        "memory_usage_bytes": df.memory_usage(deep=True).sum()
    }
    
    high_cardinality_threshold = 50
    
    for col in df.columns:
        col_data = df[col]
        
        # Missing values
        n_missing = col_data.isna().sum()
        profile["missing_counts"][col] = int(n_missing)
        profile["missing_percentages"][col] = round((n_missing / len(df)) * 100, 2)
        
        # Cardinality
        n_unique = col_data.nunique()
        profile["cardinality"][col] = int(n_unique)
        
        # Sample values (non-null)
        sample = col_data.dropna().head(5).tolist()
        profile["sample_values"][col] = [str(v)[:100] for v in sample]  # Truncate long values
        
        # Constant columns
        if n_unique <= 1:
            profile["constant_columns"].append(col)
        
        # High cardinality
        if n_unique > high_cardinality_threshold:
            profile["high_cardinality_columns"].append(col)
        
        # Categorize column type
        if pd.api.types.is_numeric_dtype(col_data):
            profile["numeric_columns"].append(col)
            
            # Numeric stats
            if not col_data.isna().all():
                profile["numeric_stats"][col] = {
                    "min": float(col_data.min()),
                    "max": float(col_data.max()),
                    "mean": float(col_data.mean()),
                    "std": float(col_data.std()) if len(col_data.dropna()) > 1 else 0.0,
                    "median": float(col_data.median()),
                    "q25": float(col_data.quantile(0.25)),
                    "q75": float(col_data.quantile(0.75))
                }
        
        elif pd.api.types.is_datetime64_any_dtype(col_data):
            profile["datetime_columns"].append(col)
        
        elif col_data.dtype == "object" or col_data.dtype.name == "category":
            # Check if it's likely text or categorical
            avg_len = col_data.dropna().astype(str).str.len().mean() if len(col_data.dropna()) > 0 else 0
            
            if avg_len > 50:  # Likely text
                profile["text_columns"].append(col)
            else:
                profile["categorical_columns"].append(col)
    
    # Add target info if provided
    if target and target in df.columns:
        profile["target"] = target
        profile["target_dtype"] = str(df[target].dtype)
        profile["target_cardinality"] = int(df[target].nunique())
        
        if pd.api.types.is_numeric_dtype(df[target]):
            profile["target_stats"] = profile["numeric_stats"].get(target, {})
    
    return profile


def detect_pii(
    df: pd.DataFrame,
    sample_size: int = 1000
) -> list[dict]:
    """
    Detect potential PII (Personally Identifiable Information) in a DataFrame.
    
    Uses both column name patterns and content patterns.
    
    Args:
        df: The DataFrame to check
        sample_size: Number of rows to sample for content checking
    
    Returns:
        List of PII warning dicts
    """
    warnings = []
    
    # Sample data for content checking
    if len(df) > sample_size:
        sample_df = df.sample(n=sample_size, random_state=42)
    else:
        sample_df = df
    
    for col in df.columns:
        col_lower = col.lower()
        
        # Check column name patterns
        for pattern in PII_COLUMN_PATTERNS:
            if pattern in col_lower:
                warnings.append({
                    "column": col,
                    "type": "column_name",
                    "pattern": pattern,
                    "description": f"Column name '{col}' suggests PII (matches pattern: {pattern})",
                    "severity": "warn"
                })
                break
        
        # Check content patterns for object columns
        if df[col].dtype == "object":
            col_sample = sample_df[col].dropna().astype(str)
            
            for pii_type, pii_info in PII_PATTERNS.items():
                pattern = pii_info["pattern"]
                description = pii_info["description"]
                
                # Check if any values match the pattern
                matches = col_sample.str.contains(pattern, regex=True, na=False)
                match_count = matches.sum()
                
                if match_count > 0:
                    match_pct = (match_count / len(col_sample)) * 100
                    warnings.append({
                        "column": col,
                        "type": "content_pattern",
                        "pattern": pii_type,
                        "description": f"Column '{col}' contains {description} ({match_count} matches, {match_pct:.1f}%)",
                        "severity": "warn" if match_pct < 50 else "blocking",
                        "match_count": int(match_count),
                        "match_percentage": round(match_pct, 2)
                    })
    
    return warnings


def detect_leakage_risks(
    df: pd.DataFrame,
    target: str,
    profile: dict
) -> list[dict]:
    """
    Detect potential data leakage risks.
    
    Checks for:
    - Columns highly correlated with target
    - Columns that might be derived from target
    - Future information leakage
    
    Args:
        df: The DataFrame
        target: Target column name
        profile: Profile dict from profile_dataframe
    
    Returns:
        List of leakage warning dicts
    """
    warnings = []
    
    if target not in df.columns:
        return warnings
    
    target_data = df[target]
    
    # Check for high correlation with target (numeric columns only)
    if pd.api.types.is_numeric_dtype(target_data):
        for col in profile["numeric_columns"]:
            if col == target:
                continue
            
            try:
                corr = df[col].corr(target_data)
                if abs(corr) > 0.95:
                    warnings.append({
                        "column": col,
                        "type": "high_correlation",
                        "description": f"Column '{col}' has very high correlation ({corr:.3f}) with target",
                        "severity": "blocking",
                        "correlation": round(corr, 4)
                    })
                elif abs(corr) > 0.85:
                    warnings.append({
                        "column": col,
                        "type": "high_correlation",
                        "description": f"Column '{col}' has high correlation ({corr:.3f}) with target",
                        "severity": "warn",
                        "correlation": round(corr, 4)
                    })
            except Exception:
                pass
    
    # Check for columns that might be derived from target
    target_name_lower = target.lower()
    suspicious_patterns = [
        f"{target_name_lower}_",
        f"_{target_name_lower}",
        f"{target_name_lower}2",
        "predicted", "prediction", "forecast", "estimate"
    ]
    
    for col in df.columns:
        if col == target:
            continue
        
        col_lower = col.lower()
        for pattern in suspicious_patterns:
            if pattern in col_lower:
                warnings.append({
                    "column": col,
                    "type": "derived_column",
                    "description": f"Column '{col}' might be derived from target (name pattern: {pattern})",
                    "severity": "warn"
                })
                break
    
    # Check for ID-like columns that shouldn't be used as features
    for col in df.columns:
        if col == target:
            continue
        
        cardinality = profile["cardinality"].get(col, 0)
        n_rows = profile["n_rows"]
        
        # If cardinality equals row count, it's likely an ID
        if cardinality == n_rows and cardinality > 10:
            col_lower = col.lower()
            if any(p in col_lower for p in ["id", "key", "index", "uuid"]):
                warnings.append({
                    "column": col,
                    "type": "id_column",
                    "description": f"Column '{col}' appears to be an ID column (all unique values)",
                    "severity": "info"
                })
    
    return warnings


def infer_problem_type(
    df: pd.DataFrame,
    target: str
) -> tuple[str, str]:
    """
    Infer the problem type (classification or regression) from the target column.
    
    Args:
        df: The DataFrame
        target: Target column name
    
    Returns:
        Tuple of (problem_type, rationale)
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found")
    
    target_data = df[target]
    n_unique = target_data.nunique()
    n_rows = len(df)
    dtype = target_data.dtype
    
    # Check if target is boolean
    if dtype == bool or set(target_data.dropna().unique()).issubset({0, 1, True, False}):
        return "classification", f"Target is binary (boolean or 0/1 values)"
    
    # Check if target is categorical/object
    if dtype == "object" or dtype.name == "category":
        return "classification", f"Target is categorical (dtype: {dtype})"
    
    # For numeric targets, use heuristics
    if pd.api.types.is_numeric_dtype(target_data):
        # If very few unique values relative to dataset size, likely classification
        unique_ratio = n_unique / n_rows
        
        if n_unique <= 2:
            return "classification", f"Target has only {n_unique} unique values (binary classification)"
        
        if n_unique <= 10 and unique_ratio < 0.01:
            return "classification", f"Target has {n_unique} unique values ({unique_ratio:.4f} ratio) - likely multi-class"
        
        if n_unique <= 20 and unique_ratio < 0.001:
            return "classification", f"Target has {n_unique} unique values with very low ratio - likely classification"
        
        # Check if values are integers
        if target_data.dropna().apply(lambda x: float(x).is_integer()).all():
            if n_unique <= 10:
                return "classification", f"Target has {n_unique} integer values - likely classification"
        
        # Default to regression for continuous numeric
        return "regression", f"Target is numeric with {n_unique} unique values ({unique_ratio:.4f} ratio)"
    
    # Default fallback
    return "classification", f"Unable to determine, defaulting to classification (dtype: {dtype})"


def infer_target_column(
    df: pd.DataFrame,
    profile: dict
) -> tuple[Optional[str], str]:
    """
    Attempt to infer the target column from the DataFrame.
    
    Uses heuristics like column names and position.
    
    Args:
        df: The DataFrame
        profile: Profile dict from profile_dataframe
    
    Returns:
        Tuple of (target_column or None, rationale)
    """
    columns = df.columns.tolist()
    
    # Common target column names
    target_patterns = [
        "target", "label", "class", "y", "outcome", "result",
        "is_", "has_", "will_", "did_",
        "price", "value", "amount", "count", "score"
    ]
    
    # Check for exact matches first
    for col in columns:
        col_lower = col.lower()
        if col_lower in ["target", "label", "class", "y", "outcome"]:
            return col, f"Column '{col}' matches common target name"
    
    # Check for pattern matches
    for col in columns:
        col_lower = col.lower()
        for pattern in target_patterns:
            if col_lower.startswith(pattern) or col_lower.endswith(pattern):
                return col, f"Column '{col}' matches target pattern '{pattern}'"
    
    # Last column heuristic (common convention)
    last_col = columns[-1]
    if last_col not in profile.get("constant_columns", []):
        return last_col, f"Using last column '{last_col}' as target (common convention)"
    
    return None, "Could not infer target column - please specify explicitly"


def get_column_correlations(
    df: pd.DataFrame,
    target: Optional[str] = None
) -> dict:
    """
    Compute correlations between numeric columns.
    
    Args:
        df: The DataFrame
        target: Optional target column to focus on
    
    Returns:
        Dict with correlation information
    """
    numeric_df = df.select_dtypes(include=[np.number])
    
    if len(numeric_df.columns) < 2:
        return {"correlations": {}, "target_correlations": {}}
    
    # Full correlation matrix
    corr_matrix = numeric_df.corr()
    
    result = {
        "correlations": {},
        "target_correlations": {},
        "high_correlations": []
    }
    
    # Find high correlations (excluding self-correlation)
    for i, col1 in enumerate(corr_matrix.columns):
        for j, col2 in enumerate(corr_matrix.columns):
            if i < j:  # Upper triangle only
                corr = corr_matrix.loc[col1, col2]
                if abs(corr) > 0.7:
                    result["high_correlations"].append({
                        "column1": col1,
                        "column2": col2,
                        "correlation": round(corr, 4)
                    })
    
    # Target correlations
    if target and target in numeric_df.columns:
        target_corrs = corr_matrix[target].drop(target).to_dict()
        result["target_correlations"] = {
            k: round(v, 4) for k, v in target_corrs.items()
        }
    
    return result


def detect_outliers(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
    method: str = "iqr",
    threshold: float = 1.5
) -> dict:
    """
    Detect outliers in numeric columns.
    
    Args:
        df: The DataFrame
        columns: Columns to check (default: all numeric)
        method: Detection method ('iqr' or 'zscore')
        threshold: Threshold for outlier detection
    
    Returns:
        Dict with outlier information per column
    """
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    outliers = {}
    
    for col in columns:
        if col not in df.columns:
            continue
        
        col_data = df[col].dropna()
        
        if len(col_data) == 0:
            continue
        
        if method == "iqr":
            q1 = col_data.quantile(0.25)
            q3 = col_data.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            
            outlier_mask = (col_data < lower_bound) | (col_data > upper_bound)
        
        elif method == "zscore":
            mean = col_data.mean()
            std = col_data.std()
            if std == 0:
                continue
            
            z_scores = (col_data - mean) / std
            outlier_mask = abs(z_scores) > threshold
        
        else:
            raise ValueError(f"Unknown method: {method}")
        
        n_outliers = outlier_mask.sum()
        
        if n_outliers > 0:
            outliers[col] = {
                "n_outliers": int(n_outliers),
                "outlier_percentage": round((n_outliers / len(col_data)) * 100, 2),
                "method": method,
                "threshold": threshold
            }
            
            if method == "iqr":
                outliers[col]["lower_bound"] = float(lower_bound)
                outliers[col]["upper_bound"] = float(upper_bound)
    
    return outliers


def get_value_counts(
    df: pd.DataFrame,
    column: str,
    top_n: int = 10
) -> dict:
    """
    Get value counts for a column.
    
    Args:
        df: The DataFrame
        column: Column name
        top_n: Number of top values to return
    
    Returns:
        Dict with value count information
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found")
    
    counts = df[column].value_counts()
    
    return {
        "column": column,
        "n_unique": int(df[column].nunique()),
        "top_values": {str(k): int(v) for k, v in counts.head(top_n).items()},
        "top_percentages": {
            str(k): round((v / len(df)) * 100, 2) 
            for k, v in counts.head(top_n).items()
        }
    }
