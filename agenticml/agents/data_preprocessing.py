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
            "2. Consider dropping columns with >50% missing values\n"
            "3. Drop constant columns\n"
            "4. Handle PII columns (drop if not needed for prediction)\n"
            "5. Address leakage risks (drop leaky columns)\n"
            "6. Fill missing values in important columns\n"
            "7. Remove duplicate rows\n"
            "8. Handle outliers: use remove_outliers (drops rows) when outlier "
            "percentage is small (<=10%); use clip_outliers when it is large (>10%)\n"
            "9. Drop one column from each highly correlated feature pair to "
            "reduce multicollinearity (prefer dropping the column with more "
            "missing values or less interpretability)\n"
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

def _filter_target_operations(steps: list[dict], target: str) -> list[dict]:
    """Remove any cleaning operations that would drop or modify the target."""
    filtered: list[dict] = []

    for step in steps:
        action = step.get("action", "")
        column = step.get("column")
        params = step.get("params", {})

        if action == "drop_column" and column == target:
            continue

        if action == "drop_columns":
            columns = params.get("columns", [])
            if target in columns:
                columns = [c for c in columns if c != target]
                if not columns:
                    continue
                step = {**step, "params": {**params, "columns": columns}}

        if column == target and action in ("clip_outliers", "remove_outliers", "convert_dtype"):
            continue

        filtered.append(step)

    return filtered
