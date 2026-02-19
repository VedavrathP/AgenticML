"""
Feature/Preprocess Agent

Responsible for:
- Choosing encoding and scaling strategies
- Building sklearn ColumnTransformer
- Ensuring no data leakage (fit only on train)
- Handling datetime and text features
- Applying SMOTE or other resampling for class imbalance
- Creating advanced features (interactions, polynomial, binning)
"""

import os
import json
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage

from agenticml.ml.state import PipelineState, log_decision, add_artifact, add_error
from agenticml.ml.config import get_config
from agenticml.ml.tools.data_io import load_dataframe, save_dataframe
from agenticml.ml.tools.preprocessing import (
    build_preprocess_pipeline,
    split_data,
    fit_transform_pipeline,
    encode_target,
    create_datetime_features,
    suggest_preprocessing_plan,
    suggest_split_plan,
    get_split_info
)
from agenticml.ml.tools.profiling import profile_dataframe
from agenticml.ml.tools.artifacts import save_json, save_preprocessing_pipeline
from agenticml.ml.tools.utils import get_run_subdir, safe_json_serialize
from agenticml.ml.tools.feature_engineering import (
    apply_smote,
    apply_feature_engineering_plan,
    get_smote_availability
)
from agenticml.ml.tools.llm import invoke_llm
from agenticml.ml.tools.llm_factory import create_llm


FEATURIZER_SYSTEM_PROMPT = """You are a Feature Engineering Agent in an ML pipeline.

Your role is to decide on preprocessing strategies for the dataset:

1. Numeric Scaling:
   - standard: StandardScaler (mean=0, std=1) - good default
   - minmax: MinMaxScaler (0-1 range) - for bounded features
   - robust: RobustScaler - for data with outliers
   - none: No scaling

2. Categorical Encoding:
   - onehot: OneHotEncoder - for low cardinality
   - ordinal: OrdinalEncoder - for high cardinality or ordinal data

3. Split Strategy:
   - stratified: Maintain class distribution (classification)
   - random: Simple random split (regression)
   - time_based: For time series data
   - cv: Cross-validation for small datasets

Guidelines:
- Consider data characteristics (outliers, cardinality, size)
- Ensure preprocessing won't cause data leakage
- Be explicit about handling unknown categories

Respond in JSON format:
{
    "preprocessing_plan": {
        "numeric_strategy": "standard|minmax|robust|none",
        "categorical_strategy": "onehot|ordinal",
        "handle_unknown": "ignore|error",
        "datetime_features": ["year", "month", "day", "dayofweek"],
        "columns_to_drop": [],
        "rationale": "explanation"
    },
    "split_plan": {
        "strategy": "stratified|random|time_based|cv",
        "test_size": 0.2,
        "cv_folds": 5,
        "time_column": null,
        "rationale": "explanation"
    }
}
"""


def run_featurizer_agent(state: PipelineState) -> PipelineState:
    """
    Run the featurizer agent to preprocess the dataset.
    
    This agent:
    1. Analyzes cleaned data characteristics
    2. Applies advanced feature engineering (interactions, polynomial, binning)
    3. Decides on preprocessing strategies
    4. Splits data into train/test
    5. Builds and fits preprocessing pipeline
    6. Applies SMOTE if configured for class imbalance
    7. Transforms features
    
    Args:
        state: Current pipeline state
    
    Returns:
        Updated pipeline state
    """
    config = get_config()
    run_dir = state["run_dir"]
    target = state["target"]
    problem_type = state["problem_type"]
    
    # Get planning decisions for feature engineering and SMOTE
    planning_decisions = state.get("planning_decisions", {})
    if isinstance(planning_decisions, str):
        planning_decisions = json.loads(planning_decisions) if planning_decisions else {}
    
    # =========================================================================
    # Step 1: Load cleaned data
    # =========================================================================
    cleaned_data_path = state.get("cleaned_data_path")
    if not cleaned_data_path:
        add_error(state, "featurizer", "No cleaned data path in state")
        return state
    
    try:
        df = load_dataframe(cleaned_data_path)
    except Exception as e:
        add_error(state, "featurizer", f"Failed to load data: {str(e)}")
        return state
    
    # =========================================================================
    # Step 2: Profile the cleaned data
    # =========================================================================
    profile = profile_dataframe(df, target)
    has_datetime = len(profile.get("datetime_columns", [])) > 0
    
    # =========================================================================
    # Step 3: Apply advanced feature engineering from planning decisions
    # =========================================================================
    feature_eng_plan = planning_decisions.get("feature_engineering", {})
    if feature_eng_plan:
        df, new_features, fe_info = apply_feature_engineering_plan(df, target, feature_eng_plan)
        
        if new_features:
            state["new_features_created"] = new_features
            state["feature_engineering_applied"] = [fe_info]
            
            log_decision(
                state, "featurizer",
                f"Applied feature engineering: created {len(new_features)} new features",
                f"Interactions: {len(fe_info.get('interactions_created', []))}, "
                f"Polynomial: {len(fe_info.get('polynomial_created', []))}, "
                f"Binning: {len(fe_info.get('binning_applied', []))}",
                fe_info
            )
    
    # =========================================================================
    # Step 4: Get preprocessing suggestions
    # =========================================================================
    base_preprocess_plan = suggest_preprocessing_plan(profile, problem_type)
    base_split_plan = suggest_split_plan(profile, problem_type, has_datetime)
    
    # =========================================================================
    # Step 5: Use LLM to refine plans
    # =========================================================================
    llm_plans = _get_llm_preprocessing_plans(
        profile=profile,
        target=target,
        problem_type=problem_type,
        base_preprocess_plan=base_preprocess_plan,
        base_split_plan=base_split_plan,
        config=config,
        verbose=state.get("verbose", False)
    )
    
    # Use LLM plans if available, otherwise use base suggestions
    preprocessing_plan = llm_plans.get("preprocessing_plan", base_preprocess_plan)
    split_plan = llm_plans.get("split_plan", base_split_plan)
    
    state["preprocessing_plan"] = safe_json_serialize(preprocessing_plan)
    state["split_plan"] = safe_json_serialize(split_plan)
    
    log_decision(
        state, "featurizer",
        f"Created preprocessing plan: {preprocessing_plan.get('numeric_strategy')} scaling, "
        f"{preprocessing_plan.get('categorical_strategy')} encoding",
        preprocessing_plan.get("rationale", ""),
        preprocessing_plan
    )
    
    log_decision(
        state, "featurizer",
        f"Created split plan: {split_plan.get('strategy')} strategy",
        split_plan.get("rationale", ""),
        split_plan
    )
    
    # =========================================================================
    # Step 6: Handle datetime features if present
    # =========================================================================
    datetime_cols = profile.get("datetime_columns", [])
    if datetime_cols and preprocessing_plan.get("datetime_features"):
        df = create_datetime_features(
            df,
            datetime_cols,
            preprocessing_plan.get("datetime_features", ["year", "month", "day", "dayofweek"])
        )
        log_decision(
            state, "featurizer",
            f"Created datetime features from {len(datetime_cols)} columns",
            f"Extracted: {preprocessing_plan.get('datetime_features')}"
        )
    
    # =========================================================================
    # Step 7: Split the data
    # =========================================================================
    split_result = split_data(df, target, split_plan)
    split_info = get_split_info(split_result)
    
    # Handle CV vs train/test split
    if split_plan.get("strategy") == "cv":
        # For CV, we'll use the full data during training
        X_train = split_result["X"]
        y_train = split_result["y"]
        X_test = None
        y_test = None
        state["cv_splitter"] = True
    else:
        X_train = split_result["X_train"]
        X_test = split_result["X_test"]
        y_train = split_result["y_train"]
        y_test = split_result["y_test"]
        state["cv_splitter"] = False
    
    log_decision(
        state, "featurizer",
        f"Split data: {split_info.get('train_samples', len(X_train))} train samples",
        f"Strategy: {split_plan.get('strategy')}, "
        f"Test size: {split_info.get('test_samples', 0) if X_test is not None else 'N/A (CV)'}",
        split_info
    )
    
    # =========================================================================
    # Step 8: Build and fit preprocessing pipeline
    # =========================================================================
    preprocessor, feature_cols = build_preprocess_pipeline(df, target, preprocessing_plan)
    
    # Fit on training data only (CRITICAL for preventing leakage)
    if X_test is not None:
        X_train_transformed, X_test_transformed, feature_names = fit_transform_pipeline(
            preprocessor, X_train, X_test
        )
    else:
        # For CV, fit on all data (will be re-fit in each fold)
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = None
        try:
            feature_names = preprocessor.get_feature_names_out().tolist()
        except Exception:
            feature_names = feature_cols
    
    # Encode target if needed
    if y_test is not None:
        y_train_encoded, y_test_encoded, label_encoder = encode_target(
            y_train, y_test, problem_type
        )
    else:
        # CV mode: no test set, encode only training target
        y_train_encoded, _, label_encoder = encode_target(
            y_train, y_train, problem_type
        )
        y_test_encoded = None
    
    # =========================================================================
    # Step 9: Apply SMOTE if configured for class imbalance
    # =========================================================================
    imbalance_strategy = planning_decisions.get("imbalance_strategy", "none")
    user_constraints = state.get("user_constraints", {})
    
    # Check if SMOTE should be applied
    apply_resampling = (
        problem_type == "classification" and
        imbalance_strategy in ["smote", "adasyn", "undersample", "smote_tomek", "smote_enn"] and
        get_smote_availability()["available"]
    )
    
    # Also check user constraints
    if user_constraints.get("auto_smote"):
        apply_resampling = True
        imbalance_strategy = "smote"
    
    if apply_resampling:
        try:
            X_train_resampled, y_train_resampled, smote_info = apply_smote(
                X_train_transformed,
                y_train_encoded,
                strategy=imbalance_strategy,
                random_state=42
            )
            
            if smote_info.get("applied"):
                X_train_transformed = X_train_resampled
                y_train_encoded = y_train_resampled
                state["smote_applied"] = True
                
                log_decision(
                    state, "featurizer",
                    f"Applied {imbalance_strategy.upper()}: {smote_info.get('samples_added', 0)} samples added",
                    f"Original: {smote_info.get('original_shape', [0])[0]} → New: {smote_info.get('new_shape', [0])[0]} samples",
                    smote_info
                )
            else:
                log_decision(
                    state, "featurizer",
                    f"SMOTE not applied: {smote_info.get('error', 'unknown reason')}",
                    "Proceeding with original data"
                )
        except Exception as e:
            add_error(state, "featurizer", f"SMOTE failed: {str(e)}")
            log_decision(
                state, "featurizer",
                f"SMOTE failed: {str(e)}",
                "Proceeding with original data"
            )
    elif problem_type == "classification" and imbalance_strategy == "class_weight":
        # Note: class_weight will be handled by the modeler
        log_decision(
            state, "featurizer",
            "Class imbalance will be handled via class_weight in models",
            "No resampling applied"
        )
    
    # =========================================================================
    # Step 10: Save artifacts
    # =========================================================================
    features_dir = get_run_subdir(run_dir, "features")
    
    # Save preprocessor
    preprocessor_path = os.path.join(features_dir, "preprocessor.joblib")
    save_preprocessing_pipeline(preprocessor, preprocessor_path, feature_names)
    state["preprocessor_path"] = preprocessor_path
    add_artifact(state, "preprocessor", preprocessor_path, "model")
    
    # Save transformed data
    import pandas as pd
    import numpy as np
    
    train_df = pd.DataFrame(X_train_transformed, columns=feature_names)
    train_df[target] = y_train_encoded
    train_path = os.path.join(features_dir, "train_features.csv")
    save_dataframe(train_df, train_path)
    state["train_data_path"] = train_path
    add_artifact(state, "train_features", train_path, "data")
    
    if X_test_transformed is not None:
        test_df = pd.DataFrame(X_test_transformed, columns=feature_names)
        test_df[target] = y_test_encoded
        test_path = os.path.join(features_dir, "test_features.csv")
        save_dataframe(test_df, test_path)
        state["test_data_path"] = test_path
        add_artifact(state, "test_features", test_path, "data")
    
    # Save feature info
    feature_info = {
        "feature_names": feature_names,
        "n_features": len(feature_names),
        "original_columns": feature_cols,
        "preprocessing_plan": preprocessing_plan,
        "split_info": split_info
    }
    feature_info_path = os.path.join(features_dir, "feature_info.json")
    save_json(feature_info_path, feature_info)
    add_artifact(state, "feature_info", feature_info_path, "json")
    
    # Store in state for modeling
    state["feature_names"] = feature_names
    state["n_train_samples"] = len(X_train_transformed)
    state["n_test_samples"] = len(X_test_transformed) if X_test_transformed is not None else 0
    state["n_features"] = len(feature_names)
    
    log_decision(
        state, "featurizer",
        f"Completed feature engineering: {len(feature_names)} features",
        f"Original: {len(feature_cols)} columns → {len(feature_names)} features after encoding",
        feature_info
    )
    
    return state


def _get_llm_preprocessing_plans(
    profile: dict,
    target: str,
    problem_type: str,
    base_preprocess_plan: dict,
    base_split_plan: dict,
    config: Any,
    verbose: bool = False
) -> dict:
    """Use LLM to refine preprocessing and split plans."""
    if not config.llm_api_key:
        return {
            "preprocessing_plan": base_preprocess_plan,
            "split_plan": base_split_plan
        }
    
    try:
        llm = create_llm(config)
        
        context = {
            "n_rows": profile.get("n_rows"),
            "n_cols": profile.get("n_cols"),
            "target": target,
            "problem_type": problem_type,
            "numeric_columns": profile.get("numeric_columns"),
            "categorical_columns": profile.get("categorical_columns"),
            "datetime_columns": profile.get("datetime_columns"),
            "high_cardinality_columns": profile.get("high_cardinality_columns"),
            "has_outliers": any(
                stats.get("max", 0) > stats.get("q75", 0) + 1.5 * (stats.get("q75", 0) - stats.get("q25", 0))
                for stats in profile.get("numeric_stats", {}).values()
                if stats
            )
        }
        
        prompt = f"""Decide on preprocessing and split strategies for this dataset.

Dataset Context:
{json.dumps(context, indent=2)}

Base Preprocessing Suggestion:
{json.dumps(base_preprocess_plan, indent=2)}

Base Split Suggestion:
{json.dumps(base_split_plan, indent=2)}

Consider:
1. Use robust scaling if outliers detected
2. Use ordinal encoding for high cardinality categoricals
3. Use stratified split for classification
4. Use CV for small datasets (<10k rows)
5. Use time-based split if datetime columns present

Provide your refined plans with rationale."""

        messages = [
            SystemMessage(content=FEATURIZER_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]
        
        response = invoke_llm(llm, messages, "Featurizer", "Preprocessing and split plans", verbose)
        
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        
        return json.loads(content)
    
    except Exception as e:
        return {
            "preprocessing_plan": base_preprocess_plan,
            "split_plan": base_split_plan
        }
