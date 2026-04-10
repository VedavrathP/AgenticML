"""
Data Preprocessing (Cleaning) Agent.

Loads raw data, asks the LLM for a structured cleaning plan, executes it
via deterministic tool functions, and saves the cleaned dataset.  All LLM
calls go through ``invoke_llm_json`` -- no free-text parsing or silent
fallbacks.
"""

import json
import os
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage

from agenticml.agents.base_agent import BaseAgent
from agenticml.ml.config import get_config
from agenticml.ml.tools.data_io import load_dataframe, save_dataframe, resolve_column_name
from agenticml.ml.tools.cleaning import (
    apply_cleaning,
    get_cleaning_stats,
    suggest_cleaning_steps,
)
from agenticml.ml.tools.utils import get_run_subdir, safe_json_serialize
from agenticml.services.artifact_service import save_json
from agenticml.services.llm_service import get_llm, invoke_llm_json
from agenticml.state.workflow_state import (
    WorkflowState,
    log_decision,
    add_artifact,
    add_error,
    record_execution,
)


SYSTEM_PROMPT = """\
You are a Data Cleaning Agent in an ML pipeline.

You receive a dataset profile, PII / leakage warnings, and deterministic
cleaning suggestions.  Your job is to produce a comprehensive cleaning plan.

You must respond with a JSON object with exactly this schema:

{
    "cleaning_steps": [
        {
            "action": "<action_name>",
            "column": "<column_name_or_null>",
            "params": {},
            "rationale": "<why this step>"
        }
    ],
    "overall_rationale": "<one-paragraph summary of cleaning strategy>"
}

Supported actions:
  drop_column, drop_columns, fill_missing, remove_duplicates,
  drop_missing_rows, clip_outliers, remove_outliers, convert_dtype,
  rename_column, drop_constant_columns, drop_high_missing,
  lowercase_column_names, strip_whitespace.

Notes on outlier handling:
  - remove_outliers: drops rows where the column value is an outlier.
    Use when outlier percentage is small (<=10%) so data loss is acceptable.
  - clip_outliers: clips extreme values to the boundary.
    Use when outlier percentage is large (>10%) to preserve data.
  Both accept params: {"method": "iqr"|"zscore", "threshold": 1.5}

Rules:
- NEVER drop or modify the target column.
- NEVER drop feature columns just because they are correlated with each other
  or with the target. High feature-target correlation is GOOD — it means the
  feature is predictive. High feature-feature correlation is handled later by
  the Feature Engineering Agent (ratios, interactions), NOT by dropping here.
- Handle outliers in feature columns using remove_outliers or clip_outliers
  based on the outlier summary provided.
- Be conservative -- preserve as much useful data as possible.
- Explain the rationale for every step.
"""


class DataPreprocessingAgent(BaseAgent):
    """Clean the raw dataset according to an LLM-generated plan."""

    name = "data_preprocessing"

    def run(self, state: WorkflowState) -> WorkflowState:
        config = get_config()
        run_dir = state["run_dir"]
        target = state["target"]
        verbose = state.get("verbose", False)

        # ==================================================================
        # 1. Load raw data
        # ==================================================================
        raw_data_path = state.get("raw_data_path")
        if not raw_data_path:
            add_error(state, self.name, "No raw data path in state")
            record_execution(state, self.name, status="failed")
            return state

        try:
            df = load_dataframe(raw_data_path)
        except Exception as exc:
            add_error(state, self.name, f"Failed to load data: {exc}")
            record_execution(state, self.name, status="failed")
            return state

        # ==================================================================
        # 2. Gather context for the LLM
        # ==================================================================
        profile = state.get("data_summary", {})
        pii_warnings = state.get("pii_warnings", [])
        leakage_warnings = state.get("leakage_warnings", [])
        outlier_summary = state.get("outlier_summary", {})
        high_correlation_pairs = state.get("high_correlation_pairs", [])
        base_suggestions = suggest_cleaning_steps(
            profile, outlier_summary, high_correlation_pairs,
        )

        # ==================================================================
        # 3. Ask LLM for cleaning plan
        # ==================================================================
        llm_plan = self._ask_llm(
            profile=profile,
            target=target,
            pii_warnings=pii_warnings,
            leakage_warnings=leakage_warnings,
            outlier_summary=outlier_summary,
            high_correlation_pairs=high_correlation_pairs,
            base_suggestions=base_suggestions,
            config=config,
            verbose=verbose,
        )

        cleaning_steps = llm_plan.get("cleaning_steps", [])
        cleaning_steps = _filter_target_operations(cleaning_steps, target)
        cleaning_steps = _guard_outlier_removal(cleaning_steps, df)

        cleaning_plan = {
            "steps": cleaning_steps,
            "rationale": llm_plan.get("overall_rationale", ""),
        }

        state["cleaning_plan"] = safe_json_serialize(cleaning_plan)

        log_decision(
            state, self.name,
            f"Created cleaning plan with {len(cleaning_steps)} steps",
            cleaning_plan.get("rationale", ""),
            {"n_steps": len(cleaning_steps)},
        )

        # ==================================================================
        # 4. Execute the cleaning plan
        # ==================================================================
        df_cleaned, cleaning_report = apply_cleaning(df, cleaning_plan)

        resolved_target = resolve_column_name(df_cleaned, target)
        if resolved_target != target:
            log_decision(
                state,
                self.name,
                f"Synced target column after cleaning: '{target}' → '{resolved_target}'",
                "Cleaning steps renamed columns (e.g. lowercase); state target updated to match.",
                {"previous_target": target, "resolved_target": resolved_target},
            )
            state["target"] = resolved_target
            target = resolved_target

        # ==================================================================
        # 5. Before / after statistics
        # ==================================================================
        cleaning_stats = get_cleaning_stats(df, df_cleaned)
        cleaning_report["stats"] = cleaning_stats
        state["cleaning_report"] = safe_json_serialize(cleaning_report)

        # ==================================================================
        # 6. Save cleaned data & report
        # ==================================================================
        cleaned_dir = get_run_subdir(run_dir, "cleaned")
        cleaned_path = os.path.join(cleaned_dir, "cleaned_data.csv")
        save_dataframe(df_cleaned, cleaned_path)
        state["cleaned_data_path"] = cleaned_path
        add_artifact(state, "cleaned_data", cleaned_path, "data")

        metrics_dir = get_run_subdir(run_dir, "metrics")
        report_path = os.path.join(metrics_dir, "cleaning_report.json")
        save_json(report_path, cleaning_report)
        add_artifact(state, "cleaning_report", report_path, "json")

        # ==================================================================
        # 7. Log summary
        # ==================================================================
        successful_steps = sum(
            1 for s in cleaning_report.get("steps_executed", []) if s.get("success")
        )
        log_decision(
            state, self.name,
            "Completed data cleaning",
            (
                f"Executed {successful_steps}/{len(cleaning_report.get('steps_executed', []))} steps. "
                f"Rows: {cleaning_stats['rows_before']} -> {cleaning_stats['rows_after']} "
                f"({cleaning_stats['rows_removed']} removed). "
                f"Cols: {cleaning_stats['cols_before']} -> {cleaning_stats['cols_after']}."
            ),
            cleaning_stats,
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
        pii_warnings: list,
        leakage_warnings: list,
        outlier_summary: dict,
        high_correlation_pairs: list,
        base_suggestions: list,
        config: Any,
        verbose: bool,
    ) -> dict:
        llm = get_llm(config)

        context = {
            "n_rows": profile.get("n_rows"),
            "n_cols": profile.get("n_cols"),
            "columns": profile.get("columns"),
            "missing_percentages": profile.get("missing_percentages"),
            "constant_columns": profile.get("constant_columns"),
            "high_cardinality_columns": profile.get("high_cardinality_columns"),
            "numeric_columns": profile.get("numeric_columns"),
            "categorical_columns": profile.get("categorical_columns"),
            "target": target,
            "pii_columns": [w["column"] for w in pii_warnings],
            "leakage_columns": [
                w["column"]
                for w in leakage_warnings
                if w.get("severity") == "blocking"
            ],
        }

        outlier_section = "None"
        if outlier_summary:
            outlier_section = json.dumps(outlier_summary, indent=2)

        corr_section = "None"
        if high_correlation_pairs:
            corr_section = json.dumps(high_correlation_pairs, indent=2)

        prompt = (
            "Create a data cleaning plan for this dataset.\n\n"
            f"Dataset Context:\n{json.dumps(context, indent=2)}\n\n"
            f"PII Warnings:\n{json.dumps(pii_warnings, indent=2) if pii_warnings else 'None'}\n\n"
            f"Leakage Warnings:\n{json.dumps(leakage_warnings, indent=2) if leakage_warnings else 'None'}\n\n"
            f"Outlier Summary (per column):\n{outlier_section}\n\n"
            f"Highly Correlated Feature Pairs (|r| >= 0.85):\n{corr_section}\n\n"
            f"Deterministic Suggestions:\n{json.dumps(base_suggestions, indent=2)}\n\n"
            "Requirements:\n"
            f"1. NEVER drop or modify the target column: {target}\n"
            "2. NEVER drop feature columns because they correlate with each other "
            "or with the target — high correlation with the target is DESIRABLE "
            "(it means the feature is predictive). Correlated feature pairs are "
            "handled by the Feature Engineering Agent, not here.\n"
            "3. Consider dropping columns with >50% missing values\n"
            "4. Drop constant columns\n"
            "5. Handle PII columns (drop if not needed for prediction)\n"
            "6. Address leakage risks (drop leaky columns)\n"
            "7. Fill missing values in important columns\n"
            "8. Remove duplicate rows\n"
            "9. Handle outliers CONSERVATIVELY:\n"
            "   - ONLY use remove_outliers when the outlier percentage is VERY small (<5%) "
            "AND the dataset has >5000 rows. For smaller datasets, prefer clip_outliers "
            "or NO outlier handling at all.\n"
            "   - NEVER apply remove_outliers to multiple columns — it compounds row loss "
            "dramatically (removing 10% per column across 5 columns can eliminate 40%+ of data).\n"
            "   - For classification tasks, extreme values in feature columns may correspond "
            "to rare but valid classes — removing them destroys minority class representation.\n"
            "   - When in doubt, DO NOT handle outliers. Prefer clip_outliers over remove_outliers.\n"
            "10. Be conservative -- preserve as much useful data as possible\n\n"
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
            step_description="Cleaning plan generation",
            verbose=verbose,
        )


# ======================================================================
# Module-level helper
# ======================================================================

def _guard_outlier_removal(steps: list[dict], df) -> list[dict]:
    """Convert remove_outliers → clip_outliers when row loss would exceed 15%.

    Applying remove_outliers to many columns compounds: removing 10% per column
    across 11 columns can eliminate 60%+ of rows, destroying minority classes.
    This guard estimates cumulative row loss and downgrades to clip_outliers
    when the plan is too aggressive.
    """
    import pandas as pd
    import numpy as np

    n_rows = len(df)
    remove_steps = [s for s in steps if s.get("action") == "remove_outliers"]

    # Estimate total rows that would be removed (union approximation)
    pct_removed = 0.0
    if remove_steps:
        outlier_mask = pd.Series([False] * n_rows, index=df.index)
        for step in remove_steps:
            col = step.get("column")
            if col and col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                params = step.get("params", {})
                method = params.get("method", "iqr")
                threshold = params.get("threshold", 1.5)
                series = df[col].dropna()
                if method == "iqr":
                    q1, q3 = series.quantile(0.25), series.quantile(0.75)
                    iqr = q3 - q1
                    mask = (df[col] < q1 - threshold * iqr) | (df[col] > q3 + threshold * iqr)
                else:
                    mean, std = series.mean(), series.std()
                    mask = (df[col] - mean).abs() > threshold * std
                outlier_mask = outlier_mask | mask.fillna(False)
        pct_removed = outlier_mask.sum() / n_rows

    if pct_removed > 0.15:
        # Downgrade all remove_outliers to clip_outliers
        guarded = []
        for step in steps:
            if step.get("action") == "remove_outliers":
                step = {**step, "action": "clip_outliers",
                        "rationale": (step.get("rationale", "") +
                                      f" [auto-downgraded: estimated {pct_removed:.0%} row loss exceeds 15% threshold]")}
            guarded.append(step)
        return guarded

    # Also guard clip_outliers when applied to many columns on small datasets —
    # clipping distorts feature distributions for minority classes
    clip_steps = [s for s in steps if s.get("action") == "clip_outliers"]
    if n_rows < 5000 and len(clip_steps) > 3:
        # Keep only the 3 most impactful clip steps (first 3 as LLM ordered them)
        clip_count = 0
        guarded = []
        for step in steps:
            if step.get("action") == "clip_outliers":
                clip_count += 1
                if clip_count > 3:
                    continue  # drop excess clip steps
            guarded.append(step)
        return guarded

    return steps


def _filter_target_operations(steps: list[dict], target: str) -> list[dict]:
    """Remove any cleaning operations that would drop or modify the target.

    Comparison is case-insensitive so that a lowercased target (e.g.
    'overall_impact') is still protected even when ``target`` was stored
    with its original casing ('Overall_Impact').
    """
    target_lower = target.lower()
    filtered: list[dict] = []

    for step in steps:
        action = step.get("action", "")
        column = step.get("column")
        params = step.get("params", {})

        if action == "drop_column" and column and column.lower() == target_lower:
            continue

        if action == "drop_columns":
            columns = params.get("columns", [])
            safe_cols = [c for c in columns if c.lower() != target_lower]
            if len(safe_cols) < len(columns):
                if not safe_cols:
                    continue
                step = {**step, "params": {**params, "columns": safe_cols}}

        if column and column.lower() == target_lower and action in (
            "clip_outliers", "remove_outliers", "convert_dtype"
        ):
            continue

        filtered.append(step)

    return filtered
