"""
Feature Engineering Agent.

Loads cleaned data, asks the LLM for a preprocessing / split / feature-
engineering plan, applies feature transformations, splits data, builds and
fits the sklearn preprocessing pipeline (on train only!), optionally applies
SMOTE, and saves all artifacts.  All LLM calls go through
``invoke_llm_json`` -- no free-text parsing or silent fallbacks.
"""

import json
import os
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.messages import SystemMessage, HumanMessage

from agenticml.agents.base_agent import BaseAgent
from agenticml.ml.config import get_config
from agenticml.ml.tools.data_io import load_dataframe, save_dataframe, resolve_column_name
from agenticml.ml.tools.preprocessing import (
    build_preprocess_pipeline,
    split_data,
    fit_transform_pipeline,
    encode_target,
    create_datetime_features,
    get_split_info,
)
from agenticml.ml.tools.profiling import profile_dataframe
from agenticml.ml.tools.feature_engineering import (
    apply_smote,
    apply_feature_engineering_plan,
    get_smote_availability,
)
from agenticml.ml.tools.utils import get_run_subdir, safe_json_serialize
from agenticml.services.artifact_service import save_json, save_preprocessing_pipeline
from agenticml.services.llm_service import get_llm, invoke_llm_json
from agenticml.state.workflow_state import (
    WorkflowState,
    log_decision,
    add_artifact,
    add_error,
    record_execution,
)


SYSTEM_PROMPT = """\
You are a Feature Engineering Agent in an ML pipeline.

You receive a cleaned-data profile and must decide on:
1. Preprocessing strategy (scaling, encoding)
2. Train/test split strategy
3. Advanced feature engineering

You must respond with a JSON object with exactly this schema:

{
    "preprocessing_plan": {
        "numeric_strategy": "standard" | "minmax" | "robust" | "none",
        "categorical_strategy": "onehot" | "ordinal",
        "handle_unknown": "ignore" | "error",
        "datetime_features": [],
        "columns_to_drop": [],
        "rationale": "..."
    },
    "split_plan": {
        "strategy": "stratified" | "random" | "time_based" | "cv",
        "test_size": 0.2,
        "cv_folds": 5,
        "time_column": null,
        "rationale": "..."
    },
    "feature_engineering": {
        "interactions": [["col1", "col2"]],
        "polynomial_features": ["col"],
        "binning": [{"column": "col", "n_bins": 5}],
        "log_transform": ["col"],
        "sqrt_transform": ["col"],
        "ratios": [{"numerator": "col1", "denominator": "col2"}],
        "aggregations": [{"group_col": "cat_col", "agg_cols": ["num_col"], "agg_funcs": ["mean"]}],
        "rationale": "..."
    }
}

Feature engineering guidelines:
- **interactions**: Create multiplicative interaction features for column pairs \
that are likely to have a joint effect on the target.
- **polynomial_features**: Create squared terms for numeric columns with \
non-linear relationships to the target.
- **binning**: Discretise continuous columns into bins when the relationship \
with the target is step-wise rather than linear.
- **log_transform**: Apply log(1+x) to heavily right-skewed numeric columns \
(skewness > 2). Check the numeric_stats for skewness indicators.
- **sqrt_transform**: Apply sqrt to moderately skewed columns (1 < skewness <= 2).
- **ratios**: Create ratio features to capture per-unit relationships. \
This is especially valuable when highly correlated column pairs were dropped \
during cleaning -- instead of losing that information, create a ratio from \
the original pair (e.g. total_rooms / households = rooms_per_household). \
The context will list dropped columns and high-correlation pairs to guide you.
- **aggregations**: When a categorical column exists alongside numeric columns, \
create group-level statistics (mean, median, std) to capture cluster behaviour. \
Only use this when the categorical column has moderate cardinality (2-50 unique values).

General guidelines:
- Use robust scaling when outliers are detected.
- Use ordinal encoding for high-cardinality categoricals.
- Use stratified split for classification.
- Use CV for small datasets (<10 000 rows).
- Use time-based split when datetime columns are present and relevant.
- Prefer creating derived features (ratios, interactions) over simply dropping \
correlated columns -- preserve information rather than discarding it.
- Only include feature engineering operations you are confident will help; \
leave arrays empty if no operation of that type is warranted.
"""


class FeatureEngineeringAgent(BaseAgent):
    """Preprocess features, split data, and build the sklearn pipeline."""

    name = "feature_engineering"

    def run(self, state: WorkflowState) -> WorkflowState:
        config = get_config()
        run_dir = state["run_dir"]
        target = state["target"]
        problem_type = state["problem_type"]
        verbose = state.get("verbose", False)

        # ==================================================================
        # 1. Load cleaned data
        # ==================================================================
        cleaned_data_path = state.get("cleaned_data_path")
        if not cleaned_data_path:
            add_error(state, self.name, "No cleaned data path in state")
            record_execution(state, self.name, status="failed")
            return state

        try:
            df = load_dataframe(cleaned_data_path)
        except Exception as exc:
            add_error(state, self.name, f"Failed to load data: {exc}")
            record_execution(state, self.name, status="failed")
            return state

        # Sync target name in case cleaning renamed columns (e.g. lowercase)
        resolved_target = resolve_column_name(df, target)
        if resolved_target != target:
            log_decision(
                state, self.name,
                f"Resolved target column: '{target}' → '{resolved_target}'",
                "Column name changed by cleaning (e.g. lowercase); syncing state.",
                {"previous_target": target, "resolved_target": resolved_target},
            )
            state["target"] = resolved_target
            target = resolved_target

        # ==================================================================
        # 2. Profile cleaned data & enrich with skewness / target corr
        # ==================================================================
        profile = profile_dataframe(df, target)
        has_datetime = len(profile.get("datetime_columns", [])) > 0

        numeric_cols = profile.get("numeric_columns", [])
        for col in numeric_cols:
            if col in df.columns and col in profile.get("numeric_stats", {}):
                series = df[col].dropna()
                if len(series) > 2:
                    profile["numeric_stats"][col]["skew"] = float(series.skew())

        if target and target in df.columns and pd.api.types.is_numeric_dtype(df[target]):
            numeric_df = df[numeric_cols].select_dtypes(include=[np.number])
            if target in numeric_df.columns and len(numeric_df.columns) >= 2:
                corrs = numeric_df.corr()[target].drop(target, errors="ignore")
                for col, val in corrs.items():
                    if col in profile.get("numeric_stats", {}):
                        profile["numeric_stats"][col]["target_correlation"] = float(val)

        # ==================================================================
        # 3. Ask LLM for plans
        # ==================================================================
        high_corr = state.get("high_correlation_pairs", [])

        cleaning_report = state.get("cleaning_report", {})
        dropped_columns: list[str] = []
        for step in cleaning_report.get("steps_executed", []):
            if step.get("action") == "drop_column" and step.get("column"):
                dropped_columns.append(step["column"])

        llm_plans = self._ask_llm(
            profile=profile,
            target=target,
            problem_type=problem_type,
            config=config,
            verbose=verbose,
            high_correlation_pairs=high_corr,
            dropped_columns=dropped_columns,
        )

        preprocessing_plan = llm_plans["preprocessing_plan"]
        split_plan = llm_plans["split_plan"]
        fe_plan = llm_plans.get("feature_engineering", {})

        state["preprocessing_plan"] = safe_json_serialize(preprocessing_plan)
        state["split_plan"] = safe_json_serialize(split_plan)

        log_decision(
            state, self.name,
            (
                f"Preprocessing: {preprocessing_plan.get('numeric_strategy')} scaling, "
                f"{preprocessing_plan.get('categorical_strategy')} encoding"
            ),
            preprocessing_plan.get("rationale", ""),
            preprocessing_plan,
        )
        log_decision(
            state, self.name,
            f"Split strategy: {split_plan.get('strategy')}",
            split_plan.get("rationale", ""),
            split_plan,
        )

        # ==================================================================
        # 4. Apply advanced feature engineering from LLM plan
        # ==================================================================
        fe_tool_plan = self._translate_fe_plan(fe_plan)
        if fe_tool_plan:
            df, new_features, fe_info = apply_feature_engineering_plan(
                df, target, fe_tool_plan
            )
            if new_features:
                state["new_features_created"] = new_features
                state["feature_engineering_applied"] = [fe_info]
                log_decision(
                    state, self.name,
                    f"Created {len(new_features)} new features",
                    fe_plan.get("rationale", ""),
                    fe_info,
                )

        # ==================================================================
        # 5. Handle datetime features
        # ==================================================================
        datetime_cols = profile.get("datetime_columns", [])
        if datetime_cols and preprocessing_plan.get("datetime_features"):
            df = create_datetime_features(
                df,
                datetime_cols,
                preprocessing_plan.get("datetime_features",
                                       ["year", "month", "day", "dayofweek"]),
            )
            log_decision(
                state, self.name,
                f"Created datetime features from {len(datetime_cols)} columns",
                f"Extracted: {preprocessing_plan.get('datetime_features')}",
            )

        # ==================================================================
        # 6. Split data
        # ==================================================================
        split_result = split_data(df, target, split_plan)
        split_info = get_split_info(split_result)

        if split_plan.get("strategy") == "cv":
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
            state, self.name,
            f"Split data: {len(X_train)} train samples",
            (
                f"Strategy: {split_plan.get('strategy')}, "
                f"Test: {len(X_test) if X_test is not None else 'N/A (CV)'}"
            ),
            split_info,
        )

        # ==================================================================
        # 7. Build & fit preprocessing pipeline (fit on train only!)
        # ==================================================================
        preprocessor, feature_cols = build_preprocess_pipeline(
            df, target, preprocessing_plan
        )

        if X_test is not None:
            X_train_t, X_test_t, feature_names = fit_transform_pipeline(
                preprocessor, X_train, X_test
            )
        else:
            X_train_t = preprocessor.fit_transform(X_train)
            X_test_t = None
            try:
                feature_names = preprocessor.get_feature_names_out().tolist()
            except Exception:
                feature_names = feature_cols

        # Encode target
        if y_test is not None:
            y_train_enc, y_test_enc, _label_enc = encode_target(
                y_train, y_test, problem_type
            )
        else:
            y_train_enc, _, _label_enc = encode_target(
                y_train, y_train, problem_type
            )
            y_test_enc = None

        # ==================================================================
        # 8. Apply SMOTE if configured
        # ==================================================================
        user_constraints = state.get("user_constraints", {})
        smote_strategy = user_constraints.get("imbalance_strategy", "none")
        apply_resampling = (
            problem_type == "classification"
            and smote_strategy in (
                "smote", "adasyn", "undersample", "smote_tomek", "smote_enn"
            )
            and get_smote_availability()["available"]
        )

        if apply_resampling:
            X_resampled, y_resampled, smote_info = apply_smote(
                X_train_t, y_train_enc, strategy=smote_strategy, random_state=42
            )
            if smote_info.get("applied"):
                X_train_t = X_resampled
                y_train_enc = y_resampled
                state["smote_applied"] = True
                log_decision(
                    state, self.name,
                    (
                        f"Applied {smote_strategy.upper()}: "
                        f"{smote_info.get('samples_added', 0)} samples added"
                    ),
                    (
                        f"Original: {smote_info.get('original_shape', [0])[0]} -> "
                        f"New: {smote_info.get('new_shape', [0])[0]} samples"
                    ),
                    smote_info,
                )

        # ==================================================================
        # 9. Save artifacts
        # ==================================================================
        features_dir = get_run_subdir(run_dir, "features")

        preprocessor_path = os.path.join(features_dir, "preprocessor.joblib")
        save_preprocessing_pipeline(preprocessor, preprocessor_path, feature_names)
        state["preprocessor_path"] = preprocessor_path
        add_artifact(state, "preprocessor", preprocessor_path, "model")

        train_df = pd.DataFrame(X_train_t, columns=feature_names)
        train_df[target] = y_train_enc
        train_path = os.path.join(features_dir, "train_features.csv")
        save_dataframe(train_df, train_path)
        state["train_data_path"] = train_path
        add_artifact(state, "train_features", train_path, "data")

        if X_test_t is not None:
            test_df = pd.DataFrame(X_test_t, columns=feature_names)
            test_df[target] = y_test_enc
            test_path = os.path.join(features_dir, "test_features.csv")
            save_dataframe(test_df, test_path)
            state["test_data_path"] = test_path
            add_artifact(state, "test_features", test_path, "data")

        feature_info = {
            "feature_names": feature_names,
            "n_features": len(feature_names),
            "original_columns": feature_cols,
            "preprocessing_plan": preprocessing_plan,
            "split_info": split_info,
        }
        feature_info_path = os.path.join(features_dir, "feature_info.json")
        save_json(feature_info_path, feature_info)
        add_artifact(state, "feature_info", feature_info_path, "json")

        state["feature_names"] = feature_names
        state["n_train_samples"] = len(X_train_t)
        state["n_test_samples"] = len(X_test_t) if X_test_t is not None else 0
        state["n_features"] = len(feature_names)

        log_decision(
            state, self.name,
            f"Completed feature engineering: {len(feature_names)} features",
            (
                f"Original: {len(feature_cols)} columns -> "
                f"{len(feature_names)} features after encoding"
            ),
            feature_info,
        )

        record_execution(state, self.name)
        return state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _ask_llm(
        self,
        *,
        profile: dict,
        target: str,
        problem_type: str,
        config: Any,
        verbose: bool,
        high_correlation_pairs: list[dict] | None = None,
        dropped_columns: list[str] | None = None,
    ) -> dict:
        llm = get_llm(config)

        has_outliers = any(
            stats.get("max", 0)
            > stats.get("q75", 0) + 1.5 * (stats.get("q75", 0) - stats.get("q25", 0))
            for stats in profile.get("numeric_stats", {}).values()
            if stats
        )

        numeric_stats_summary = {}
        for col, stats in profile.get("numeric_stats", {}).items():
            if stats and col != target:
                numeric_stats_summary[col] = {
                    "mean": round(stats.get("mean", 0), 4),
                    "std": round(stats.get("std", 0), 4),
                    "min": round(stats.get("min", 0), 4),
                    "max": round(stats.get("max", 0), 4),
                    "skew": round(stats.get("skew", 0), 4) if stats.get("skew") is not None else None,
                }

        target_correlations = {}
        for col, stats in profile.get("numeric_stats", {}).items():
            if stats and col != target and "target_correlation" in stats:
                target_correlations[col] = round(stats["target_correlation"], 4)

        context: dict[str, Any] = {
            "n_rows": profile.get("n_rows"),
            "n_cols": profile.get("n_cols"),
            "target": target,
            "problem_type": problem_type,
            "numeric_columns": profile.get("numeric_columns"),
            "categorical_columns": profile.get("categorical_columns"),
            "datetime_columns": profile.get("datetime_columns"),
            "high_cardinality_columns": profile.get("high_cardinality_columns"),
            "constant_columns": profile.get("constant_columns"),
            "has_outliers": has_outliers,
            "numeric_stats": numeric_stats_summary,
        }

        if target_correlations:
            context["target_correlations"] = target_correlations
        if high_correlation_pairs:
            context["high_correlation_pairs"] = high_correlation_pairs
        if dropped_columns:
            context["columns_dropped_during_cleaning"] = dropped_columns

        prompt = (
            "Decide on preprocessing, split, and feature-engineering strategies "
            "for this dataset.\n\n"
            f"Dataset Context:\n{json.dumps(context, indent=2)}\n\n"
            "Respond with the JSON object described in the system prompt."
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        return invoke_llm_json(
            llm,
            messages,
            agent_name=self.name,
            step_description="Preprocessing & split plan",
            verbose=verbose,
        )

    @staticmethod
    def _translate_fe_plan(fe_plan: dict) -> dict:
        """Translate the LLM feature-engineering block into the format
        expected by ``apply_feature_engineering_plan``."""
        tool_plan: dict[str, Any] = {}

        interactions = fe_plan.get("interactions", [])
        if interactions:
            tool_plan["create_interactions"] = interactions

        poly = fe_plan.get("polynomial_features", [])
        if poly:
            tool_plan["create_polynomial"] = poly

        binning = fe_plan.get("binning", [])
        if binning:
            tool_plan["apply_binning"] = binning

        log_cols = fe_plan.get("log_transform", [])
        if log_cols:
            tool_plan["apply_log_transform"] = log_cols

        sqrt_cols = fe_plan.get("sqrt_transform", [])
        if sqrt_cols:
            tool_plan["apply_sqrt_transform"] = sqrt_cols

        ratios = fe_plan.get("ratios", [])
        if ratios:
            tool_plan["create_ratios"] = ratios

        aggregations = fe_plan.get("aggregations", [])
        if aggregations:
            tool_plan["create_aggregations"] = aggregations

        return tool_plan
