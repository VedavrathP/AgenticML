"""
Preprocessing tools for feature engineering and data splitting.

All functions are pure Python with NO LLM calls.
"""

from typing import Optional, Tuple, Any
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, TimeSeriesSplit
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def build_preprocess_pipeline(
    df: pd.DataFrame,
    target: str,
    plan: dict
) -> Tuple[ColumnTransformer, list[str]]:
    """
    Build a sklearn ColumnTransformer based on the preprocessing plan.
    
    IMPORTANT: This pipeline should only be fit on training data to prevent leakage.
    
    Args:
        df: The DataFrame (used to identify column types)
        target: Target column name (excluded from preprocessing)
        plan: Preprocessing plan dict with strategies
    
    Returns:
        Tuple of (ColumnTransformer, list of feature column names)
    """
    # Get feature columns (exclude target)
    feature_cols = [col for col in df.columns if col != target]
    
    # Identify column types
    numeric_cols = []
    categorical_cols = []
    
    for col in feature_cols:
        if col in plan.get("columns_to_drop", []):
            continue
        
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric_cols.append(col)
        elif df[col].dtype == "object" or df[col].dtype.name == "category":
            categorical_cols.append(col)
    
    # Build transformers
    transformers = []
    
    # Numeric transformer
    if numeric_cols:
        numeric_steps = []
        
        # Imputation
        numeric_steps.append(("imputer", SimpleImputer(strategy="median")))
        
        # Scaling
        numeric_strategy = plan.get("numeric_strategy", "standard")
        if numeric_strategy == "standard":
            numeric_steps.append(("scaler", StandardScaler()))
        elif numeric_strategy == "minmax":
            numeric_steps.append(("scaler", MinMaxScaler()))
        elif numeric_strategy == "robust":
            numeric_steps.append(("scaler", RobustScaler()))
        # "none" means no scaling
        
        numeric_transformer = Pipeline(steps=numeric_steps)
        transformers.append(("numeric", numeric_transformer, numeric_cols))
    
    # Categorical transformer
    if categorical_cols:
        categorical_steps = []
        
        # Imputation
        categorical_steps.append(("imputer", SimpleImputer(strategy="constant", fill_value="missing")))
        
        # Encoding
        categorical_strategy = plan.get("categorical_strategy", "onehot")
        handle_unknown = plan.get("handle_unknown", "ignore")
        
        if categorical_strategy == "onehot":
            categorical_steps.append((
                "encoder",
                OneHotEncoder(handle_unknown=handle_unknown, sparse_output=False)
            ))
        elif categorical_strategy == "ordinal":
            categorical_steps.append((
                "encoder",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            ))
        # "target" encoding would require y, handled separately
        
        categorical_transformer = Pipeline(steps=categorical_steps)
        transformers.append(("categorical", categorical_transformer, categorical_cols))
    
    # Create the column transformer
    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"  # Drop any columns not explicitly handled
    )
    
    # Return the preprocessor and the feature columns used
    used_cols = numeric_cols + categorical_cols
    
    return preprocessor, used_cols


def split_data(
    df: pd.DataFrame,
    target: str,
    plan: dict
) -> dict:
    """
    Split data according to the split plan.
    
    Args:
        df: The DataFrame
        target: Target column name
        plan: Split plan dict with strategy and parameters
    
    Returns:
        Dict with split data and metadata
    """
    X = df.drop(columns=[target])
    y = df[target]
    
    strategy = plan.get("strategy", "stratified")
    test_size = plan.get("test_size", 0.2)
    random_state = plan.get("random_state", 42)
    
    result = {
        "strategy": strategy,
        "test_size": test_size,
        "random_state": random_state
    }
    
    if strategy == "stratified":
        # Stratified split for classification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        result["X_train"] = X_train
        result["X_test"] = X_test
        result["y_train"] = y_train
        result["y_test"] = y_test
    
    elif strategy == "random":
        # Simple random split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state
        )
        result["X_train"] = X_train
        result["X_test"] = X_test
        result["y_train"] = y_train
        result["y_test"] = y_test
    
    elif strategy == "time_based":
        # Time-based split (no shuffle)
        time_column = plan.get("time_column")
        
        if time_column and time_column in df.columns:
            # Sort by time column
            df_sorted = df.sort_values(time_column)
            X = df_sorted.drop(columns=[target])
            y = df_sorted[target]
        
        split_idx = int(len(X) * (1 - test_size))
        
        result["X_train"] = X.iloc[:split_idx]
        result["X_test"] = X.iloc[split_idx:]
        result["y_train"] = y.iloc[:split_idx]
        result["y_test"] = y.iloc[split_idx:]
    
    elif strategy == "cv":
        # Cross-validation (return fold indices)
        cv_folds = plan.get("cv_folds", 5)
        
        # Determine if stratified CV is appropriate
        if y.dtype == "object" or y.nunique() < 20:
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        else:
            cv = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
        
        result["cv"] = cv
        result["X"] = X
        result["y"] = y
        result["cv_folds"] = cv_folds
    
    else:
        raise ValueError(f"Unknown split strategy: {strategy}")
    
    # Add metadata
    if "X_train" in result:
        result["train_size"] = len(result["X_train"])
        result["test_size_actual"] = len(result["X_test"])
        result["train_ratio"] = len(result["X_train"]) / len(X)
    
    return result


def fit_transform_pipeline(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Fit the preprocessor on training data and transform both train and test.
    
    CRITICAL: Fit only on training data to prevent data leakage.
    
    Args:
        preprocessor: The ColumnTransformer
        X_train: Training features
        X_test: Test features
    
    Returns:
        Tuple of (transformed X_train, transformed X_test, feature names)
    """
    # Fit on training data only
    X_train_transformed = preprocessor.fit_transform(X_train)
    
    # Transform test data
    X_test_transformed = preprocessor.transform(X_test)
    
    # Get feature names
    feature_names = get_feature_names(preprocessor)
    
    return X_train_transformed, X_test_transformed, feature_names


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """
    Get feature names from a fitted ColumnTransformer.
    
    Args:
        preprocessor: Fitted ColumnTransformer
    
    Returns:
        List of feature names
    """
    try:
        return preprocessor.get_feature_names_out().tolist()
    except AttributeError:
        # Fallback for older sklearn versions
        feature_names = []
        for name, transformer, columns in preprocessor.transformers_:
            if name == "remainder":
                continue
            
            if hasattr(transformer, "get_feature_names_out"):
                names = transformer.get_feature_names_out(columns)
                feature_names.extend(names)
            else:
                feature_names.extend(columns)
        
        return feature_names


def encode_target(
    y_train: pd.Series,
    y_test: pd.Series,
    problem_type: str
) -> Tuple[np.ndarray, np.ndarray, Optional[LabelEncoder]]:
    """
    Encode target variable if needed.
    
    Args:
        y_train: Training target
        y_test: Test target
        problem_type: 'classification' or 'regression'
    
    Returns:
        Tuple of (encoded y_train, encoded y_test, encoder or None)
    """
    if problem_type == "regression":
        return y_train.values, y_test.values, None
    
    # For classification, encode if not already numeric
    if y_train.dtype == "object" or y_train.dtype.name == "category":
        encoder = LabelEncoder()
        y_train_encoded = encoder.fit_transform(y_train)
        y_test_encoded = encoder.transform(y_test)
        return y_train_encoded, y_test_encoded, encoder
    
    return y_train.values, y_test.values, None


def create_datetime_features(
    df: pd.DataFrame,
    datetime_columns: list[str],
    features: list[str] = None
) -> pd.DataFrame:
    """
    Create features from datetime columns.
    
    Args:
        df: The DataFrame
        datetime_columns: List of datetime column names
        features: List of features to extract (default: year, month, day, dayofweek, hour)
    
    Returns:
        DataFrame with new datetime features
    """
    if features is None:
        features = ["year", "month", "day", "dayofweek", "hour"]
    
    df = df.copy()
    
    for col in datetime_columns:
        if col not in df.columns:
            continue
        
        # Convert to datetime if not already
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = pd.to_datetime(df[col], errors="coerce")
        
        dt = df[col].dt
        
        if "year" in features:
            df[f"{col}_year"] = dt.year
        if "month" in features:
            df[f"{col}_month"] = dt.month
        if "day" in features:
            df[f"{col}_day"] = dt.day
        if "dayofweek" in features:
            df[f"{col}_dayofweek"] = dt.dayofweek
        if "hour" in features and hasattr(dt, "hour"):
            df[f"{col}_hour"] = dt.hour
        if "quarter" in features:
            df[f"{col}_quarter"] = dt.quarter
        if "is_weekend" in features:
            df[f"{col}_is_weekend"] = dt.dayofweek.isin([5, 6]).astype(int)
        
        # Drop original datetime column
        df = df.drop(columns=[col])
    
    return df


def get_split_info(split_result: dict) -> dict:
    """
    Get summary information about a data split.
    
    Args:
        split_result: Result from split_data
    
    Returns:
        Dict with split summary
    """
    info = {
        "strategy": split_result.get("strategy"),
        "random_state": split_result.get("random_state")
    }
    
    if "X_train" in split_result:
        info["train_samples"] = len(split_result["X_train"])
        info["test_samples"] = len(split_result["X_test"])
        info["train_ratio"] = split_result.get("train_ratio", 0)
        info["n_features"] = split_result["X_train"].shape[1]
        
        # Target distribution
        y_train = split_result["y_train"]
        y_test = split_result["y_test"]
        
        if hasattr(y_train, "value_counts"):
            info["train_target_distribution"] = y_train.value_counts(normalize=True).to_dict()
            info["test_target_distribution"] = y_test.value_counts(normalize=True).to_dict()
    
    elif "cv" in split_result:
        info["cv_folds"] = split_result.get("cv_folds")
        info["total_samples"] = len(split_result["X"])
        info["n_features"] = split_result["X"].shape[1]
    
    return info


def suggest_preprocessing_plan(profile: dict, problem_type: str) -> dict:
    """
    Suggest a preprocessing plan based on data profile.
    
    This is a deterministic suggestion. The agent will refine it.
    
    Args:
        profile: Data profile from profile_dataframe
        problem_type: 'classification' or 'regression'
    
    Returns:
        Suggested preprocessing plan dict
    """
    plan = {
        "numeric_strategy": "standard",
        "categorical_strategy": "onehot",
        "handle_unknown": "ignore",
        "datetime_features": ["year", "month", "day", "dayofweek"],
        "columns_to_drop": [],
        "rationale": ""
    }
    
    # Check for high cardinality categoricals
    high_card_cats = [
        col for col in profile.get("categorical_columns", [])
        if col in profile.get("high_cardinality_columns", [])
    ]
    
    if high_card_cats:
        # Use ordinal encoding for high cardinality
        plan["categorical_strategy"] = "ordinal"
        plan["rationale"] += f"Using ordinal encoding due to high cardinality columns: {high_card_cats}. "
    
    # Check for outliers in numeric columns
    numeric_stats = profile.get("numeric_stats", {})
    has_outliers = False
    for col, stats in numeric_stats.items():
        if stats:
            iqr = stats.get("q75", 0) - stats.get("q25", 0)
            if iqr > 0:
                lower = stats.get("q25", 0) - 1.5 * iqr
                upper = stats.get("q75", 0) + 1.5 * iqr
                if stats.get("min", 0) < lower or stats.get("max", 0) > upper:
                    has_outliers = True
                    break
    
    if has_outliers:
        plan["numeric_strategy"] = "robust"
        plan["rationale"] += "Using robust scaling due to detected outliers. "
    
    # Suggest dropping constant columns
    if profile.get("constant_columns"):
        plan["columns_to_drop"].extend(profile["constant_columns"])
        plan["rationale"] += f"Dropping constant columns: {profile['constant_columns']}. "
    
    return plan


def suggest_split_plan(
    profile: dict,
    problem_type: str,
    has_datetime: bool = False
) -> dict:
    """
    Suggest a split plan based on data characteristics.
    
    Args:
        profile: Data profile
        problem_type: 'classification' or 'regression'
        has_datetime: Whether data has datetime columns
    
    Returns:
        Suggested split plan dict
    """
    n_rows = profile.get("n_rows", 0)
    
    plan = {
        "strategy": "random",
        "test_size": 0.2,
        "cv_folds": 5,
        "random_state": 42,
        "rationale": ""
    }
    
    # Use stratified split for classification
    if problem_type == "classification":
        plan["strategy"] = "stratified"
        plan["rationale"] = "Using stratified split to maintain class distribution. "
    
    # Use CV for small datasets
    if n_rows < 10000:
        plan["strategy"] = "cv"
        plan["rationale"] += f"Using cross-validation due to small dataset ({n_rows} rows). "
    
    # Use time-based split if datetime detected
    if has_datetime and profile.get("datetime_columns"):
        plan["strategy"] = "time_based"
        plan["time_column"] = profile["datetime_columns"][0]
        plan["rationale"] = f"Using time-based split due to datetime column. "
    
    return plan
