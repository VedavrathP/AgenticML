"""
Advanced Feature Engineering Tools

Provides sophisticated feature engineering capabilities:
- SMOTE and other resampling techniques for class imbalance
- Interaction features
- Polynomial features
- Binning/discretization
- Log/sqrt transformations
- Domain-specific feature creation
"""

from typing import Optional, Tuple, List, Dict, Any
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, KBinsDiscretizer
import warnings

# Try to import imbalanced-learn for SMOTE
try:
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
    from imblearn.under_sampling import RandomUnderSampler
    from imblearn.combine import SMOTETomek, SMOTEENN
    IMBLEARN_AVAILABLE = True
except ImportError:
    IMBLEARN_AVAILABLE = False
    warnings.warn("imbalanced-learn not installed. SMOTE and resampling features will be unavailable.")


def apply_smote(
    X: np.ndarray,
    y: np.ndarray,
    strategy: str = "smote",
    random_state: int = 42,
    k_neighbors: int = 5
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Apply SMOTE or other resampling techniques to handle class imbalance.
    
    Args:
        X: Feature matrix
        y: Target vector
        strategy: Resampling strategy - 'smote', 'adasyn', 'borderline', 'smote_tomek', 'smote_enn', 'undersample'
        random_state: Random seed
        k_neighbors: Number of neighbors for SMOTE
    
    Returns:
        Tuple of (resampled X, resampled y, info dict)
    """
    if not IMBLEARN_AVAILABLE:
        return X, y, {
            "applied": False,
            "error": "imbalanced-learn not installed",
            "original_shape": X.shape,
            "new_shape": X.shape
        }
    
    info = {
        "applied": True,
        "strategy": strategy,
        "original_shape": X.shape,
        "original_class_distribution": dict(pd.Series(y).value_counts())
    }
    
    try:
        # Adjust k_neighbors if needed
        min_class_count = min(pd.Series(y).value_counts())
        effective_k = min(k_neighbors, min_class_count - 1)
        if effective_k < 1:
            effective_k = 1
        
        if strategy == "smote":
            sampler = SMOTE(random_state=random_state, k_neighbors=effective_k)
        elif strategy == "adasyn":
            sampler = ADASYN(random_state=random_state, n_neighbors=effective_k)
        elif strategy == "borderline":
            sampler = BorderlineSMOTE(random_state=random_state, k_neighbors=effective_k)
        elif strategy == "smote_tomek":
            sampler = SMOTETomek(random_state=random_state)
        elif strategy == "smote_enn":
            sampler = SMOTEENN(random_state=random_state)
        elif strategy == "undersample":
            sampler = RandomUnderSampler(random_state=random_state)
        else:
            return X, y, {"applied": False, "error": f"Unknown strategy: {strategy}"}
        
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        
        info["new_shape"] = X_resampled.shape
        info["new_class_distribution"] = dict(pd.Series(y_resampled).value_counts())
        info["samples_added"] = X_resampled.shape[0] - X.shape[0]
        
        return X_resampled, y_resampled, info
    
    except Exception as e:
        return X, y, {
            "applied": False,
            "error": str(e),
            "original_shape": X.shape,
            "new_shape": X.shape
        }


def create_interaction_features(
    df: pd.DataFrame,
    column_pairs: List[Tuple[str, str]],
    operations: List[str] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create interaction features between pairs of columns.
    
    Args:
        df: Input DataFrame
        column_pairs: List of (col1, col2) tuples
        operations: List of operations to apply - 'multiply', 'divide', 'add', 'subtract'
    
    Returns:
        Tuple of (DataFrame with new features, list of new column names)
    """
    if operations is None:
        operations = ["multiply", "divide"]
    
    df = df.copy()
    new_columns = []
    
    for col1, col2 in column_pairs:
        if col1 not in df.columns or col2 not in df.columns:
            continue
        
        # Skip if not numeric
        if not pd.api.types.is_numeric_dtype(df[col1]) or not pd.api.types.is_numeric_dtype(df[col2]):
            continue
        
        if "multiply" in operations:
            new_col = f"{col1}_x_{col2}"
            df[new_col] = df[col1] * df[col2]
            new_columns.append(new_col)
        
        if "divide" in operations:
            # Avoid division by zero
            new_col = f"{col1}_div_{col2}"
            with np.errstate(divide='ignore', invalid='ignore'):
                df[new_col] = np.where(df[col2] != 0, df[col1] / df[col2], 0)
            new_columns.append(new_col)
        
        if "add" in operations:
            new_col = f"{col1}_plus_{col2}"
            df[new_col] = df[col1] + df[col2]
            new_columns.append(new_col)
        
        if "subtract" in operations:
            new_col = f"{col1}_minus_{col2}"
            df[new_col] = df[col1] - df[col2]
            new_columns.append(new_col)
    
    return df, new_columns


def create_polynomial_features(
    df: pd.DataFrame,
    columns: List[str],
    degree: int = 2,
    include_bias: bool = False,
    interaction_only: bool = False
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create polynomial features for specified columns.
    
    Args:
        df: Input DataFrame
        columns: Columns to create polynomial features for
        degree: Polynomial degree
        include_bias: Whether to include bias term
        interaction_only: If True, only interaction features (no powers)
    
    Returns:
        Tuple of (DataFrame with new features, list of new column names)
    """
    df = df.copy()
    new_columns = []
    
    # Filter to valid numeric columns
    valid_cols = [col for col in columns if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    
    if not valid_cols:
        return df, new_columns
    
    poly = PolynomialFeatures(
        degree=degree,
        include_bias=include_bias,
        interaction_only=interaction_only
    )
    
    X_poly = poly.fit_transform(df[valid_cols].fillna(0))
    feature_names = poly.get_feature_names_out(valid_cols)
    
    # Only add non-original columns
    for i, name in enumerate(feature_names):
        if name not in valid_cols and name != '1':  # Skip original and bias
            clean_name = name.replace(' ', '_').replace('^', '_pow_')
            df[clean_name] = X_poly[:, i]
            new_columns.append(clean_name)
    
    return df, new_columns


def apply_binning(
    df: pd.DataFrame,
    columns: List[str],
    n_bins: int = 5,
    strategy: str = "quantile",
    encode: str = "ordinal"
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply binning/discretization to continuous columns.
    
    Args:
        df: Input DataFrame
        columns: Columns to bin
        n_bins: Number of bins
        strategy: Binning strategy - 'uniform', 'quantile', 'kmeans'
        encode: Encoding - 'ordinal', 'onehot'
    
    Returns:
        Tuple of (DataFrame with binned features, list of new column names)
    """
    df = df.copy()
    new_columns = []
    
    for col in columns:
        if col not in df.columns:
            continue
        
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        
        try:
            discretizer = KBinsDiscretizer(
                n_bins=n_bins,
                encode=encode,
                strategy=strategy
            )
            
            col_values = df[col].fillna(df[col].median()).values.reshape(-1, 1)
            binned = discretizer.fit_transform(col_values)
            
            if encode == "ordinal":
                new_col = f"{col}_binned"
                df[new_col] = binned.flatten()
                new_columns.append(new_col)
            else:  # onehot
                for i in range(binned.shape[1]):
                    new_col = f"{col}_bin_{i}"
                    df[new_col] = binned[:, i]
                    new_columns.append(new_col)
        
        except Exception:
            # Skip if binning fails (e.g., not enough unique values)
            continue
    
    return df, new_columns


def apply_log_transform(
    df: pd.DataFrame,
    columns: List[str],
    handle_zeros: str = "add_one"
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply log transformation to skewed columns.
    
    Args:
        df: Input DataFrame
        columns: Columns to transform
        handle_zeros: How to handle zeros - 'add_one', 'replace_min', 'skip'
    
    Returns:
        Tuple of (DataFrame with transformed features, list of new column names)
    """
    df = df.copy()
    new_columns = []
    
    for col in columns:
        if col not in df.columns:
            continue
        
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        
        values = df[col].copy()
        
        # Handle zeros and negatives
        if handle_zeros == "add_one":
            values = values + 1
        elif handle_zeros == "replace_min":
            min_positive = values[values > 0].min() if (values > 0).any() else 1
            values = values.clip(lower=min_positive)
        elif handle_zeros == "skip":
            if (values <= 0).any():
                continue
        
        # Skip if any non-positive values remain
        if (values <= 0).any():
            continue
        
        new_col = f"{col}_log"
        df[new_col] = np.log(values)
        new_columns.append(new_col)
    
    return df, new_columns


def apply_sqrt_transform(
    df: pd.DataFrame,
    columns: List[str]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Apply square root transformation to columns.
    
    Args:
        df: Input DataFrame
        columns: Columns to transform
    
    Returns:
        Tuple of (DataFrame with transformed features, list of new column names)
    """
    df = df.copy()
    new_columns = []
    
    for col in columns:
        if col not in df.columns:
            continue
        
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        
        # Skip if negative values
        if (df[col] < 0).any():
            continue
        
        new_col = f"{col}_sqrt"
        df[new_col] = np.sqrt(df[col].fillna(0))
        new_columns.append(new_col)
    
    return df, new_columns


def create_ratio_features(
    df: pd.DataFrame,
    numerator_cols: List[str],
    denominator_cols: List[str]
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create ratio features between columns.
    
    Args:
        df: Input DataFrame
        numerator_cols: Columns to use as numerators
        denominator_cols: Columns to use as denominators
    
    Returns:
        Tuple of (DataFrame with ratio features, list of new column names)
    """
    df = df.copy()
    new_columns = []
    
    for num_col in numerator_cols:
        for denom_col in denominator_cols:
            if num_col == denom_col:
                continue
            
            if num_col not in df.columns or denom_col not in df.columns:
                continue
            
            if not pd.api.types.is_numeric_dtype(df[num_col]) or not pd.api.types.is_numeric_dtype(df[denom_col]):
                continue
            
            new_col = f"{num_col}_per_{denom_col}"
            with np.errstate(divide='ignore', invalid='ignore'):
                df[new_col] = np.where(df[denom_col] != 0, df[num_col] / df[denom_col], 0)
            new_columns.append(new_col)
    
    return df, new_columns


def create_aggregation_features(
    df: pd.DataFrame,
    group_col: str,
    agg_cols: List[str],
    agg_funcs: List[str] = None
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create aggregation features based on a grouping column.
    
    Args:
        df: Input DataFrame
        group_col: Column to group by
        agg_cols: Columns to aggregate
        agg_funcs: Aggregation functions - 'mean', 'std', 'min', 'max', 'count'
    
    Returns:
        Tuple of (DataFrame with aggregation features, list of new column names)
    """
    if agg_funcs is None:
        agg_funcs = ["mean", "std"]
    
    df = df.copy()
    new_columns = []
    
    if group_col not in df.columns:
        return df, new_columns
    
    for agg_col in agg_cols:
        if agg_col not in df.columns or agg_col == group_col:
            continue
        
        if not pd.api.types.is_numeric_dtype(df[agg_col]):
            continue
        
        for func in agg_funcs:
            new_col = f"{agg_col}_{func}_by_{group_col}"
            
            try:
                agg_values = df.groupby(group_col)[agg_col].transform(func)
                df[new_col] = agg_values
                new_columns.append(new_col)
            except Exception:
                continue
    
    return df, new_columns


def apply_feature_engineering_plan(
    df: pd.DataFrame,
    target: str,
    plan: dict
) -> Tuple[pd.DataFrame, List[str], dict]:
    """
    Apply a complete feature engineering plan to the DataFrame.
    
    Args:
        df: Input DataFrame
        target: Target column name (excluded from transformations)
        plan: Feature engineering plan dict with keys:
            - create_interactions: list of [col1, col2] pairs
            - create_polynomial: list of columns
            - apply_binning: list of {"column": col, "n_bins": n} dicts
            - apply_log_transform: list of columns
            - apply_sqrt_transform: list of columns
    
    Returns:
        Tuple of (transformed DataFrame, list of new columns, info dict)
    """
    df = df.copy()
    all_new_columns = []
    info = {
        "interactions_created": [],
        "polynomial_created": [],
        "binning_applied": [],
        "log_transformed": [],
        "sqrt_transformed": [],
        "total_new_features": 0
    }
    
    # Interaction features
    interactions = plan.get("create_interactions", [])
    if interactions:
        column_pairs = [(pair[0], pair[1]) for pair in interactions if len(pair) == 2]
        df, new_cols = create_interaction_features(df, column_pairs)
        all_new_columns.extend(new_cols)
        info["interactions_created"] = new_cols
    
    # Polynomial features
    poly_cols = plan.get("create_polynomial", [])
    if poly_cols:
        # Filter out target
        poly_cols = [c for c in poly_cols if c != target]
        df, new_cols = create_polynomial_features(df, poly_cols, degree=2)
        all_new_columns.extend(new_cols)
        info["polynomial_created"] = new_cols
    
    # Binning
    binning_specs = plan.get("apply_binning", [])
    if binning_specs:
        for spec in binning_specs:
            if isinstance(spec, dict):
                col = spec.get("column")
                n_bins = spec.get("n_bins", 5)
            else:
                col = spec
                n_bins = 5
            
            if col and col != target:
                df, new_cols = apply_binning(df, [col], n_bins=n_bins)
                all_new_columns.extend(new_cols)
                info["binning_applied"].extend(new_cols)
    
    # Log transform
    log_cols = plan.get("apply_log_transform", [])
    if log_cols:
        log_cols = [c for c in log_cols if c != target]
        df, new_cols = apply_log_transform(df, log_cols)
        all_new_columns.extend(new_cols)
        info["log_transformed"] = new_cols
    
    # Sqrt transform
    sqrt_cols = plan.get("apply_sqrt_transform", [])
    if sqrt_cols:
        sqrt_cols = [c for c in sqrt_cols if c != target]
        df, new_cols = apply_sqrt_transform(df, sqrt_cols)
        all_new_columns.extend(new_cols)
        info["sqrt_transformed"] = new_cols
    
    info["total_new_features"] = len(all_new_columns)
    
    return df, all_new_columns, info


def get_smote_availability() -> dict:
    """Check if SMOTE and related tools are available."""
    return {
        "available": IMBLEARN_AVAILABLE,
        "strategies": ["smote", "adasyn", "borderline", "smote_tomek", "smote_enn", "undersample"] if IMBLEARN_AVAILABLE else []
    }
