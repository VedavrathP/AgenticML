"""
Data Cleaner Agent

Responsible for:
- Proposing cleaning steps with rationale
- Executing cleaning operations via tools
- Producing before/after statistics
"""

import os
import json
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage

from agenticml.ml.state import PipelineState, log_decision, add_artifact, add_error
from agenticml.ml.config import get_config
from agenticml.ml.tools.data_io import load_dataframe, save_dataframe, resolve_column_name
from agenticml.ml.tools.cleaning import apply_cleaning, get_cleaning_stats, suggest_cleaning_steps
from agenticml.ml.tools.artifacts import save_json
from agenticml.ml.tools.utils import get_run_subdir, safe_json_serialize
from agenticml.ml.tools.llm import invoke_llm
from agenticml.ml.tools.llm_factory import create_llm


CLEANER_SYSTEM_PROMPT = """You are a Data Cleaner Agent in an ML pipeline.

Your role is to propose and execute data cleaning operations to prepare data for modeling.

Consider these cleaning operations:
1. drop_column: Remove columns (ID columns, high missing, constant, leaky)
2. drop_columns: Remove multiple columns at once
3. fill_missing: Fill missing values (strategy: mean, median, mode, value, ffill, bfill)
4. remove_duplicates: Remove duplicate rows
5. drop_missing_rows: Drop rows with missing values
6. clip_outliers: Clip extreme values (method: iqr, zscore)
7. convert_dtype: Convert column data types
8. drop_constant_columns: Remove columns with single value
9. drop_high_missing: Drop columns with high missing percentage

Guidelines:
- Always preserve the target column
- Be conservative - don't remove too much data
- Explain rationale for each step
- Consider the impact on model training

Respond in JSON format:
{
    "cleaning_plan": {
        "steps": [
            {
                "action": "action_name",
                "column": "column_name or null",
                "params": {},
                "rationale": "why this step"
            }
        ],
        "overall_rationale": "explanation of cleaning strategy"
    }
}
"""


def run_cleaner_agent(state: PipelineState) -> PipelineState:
    """
    Run the cleaner agent to clean the dataset.
    
    This agent:
    1. Analyzes the data profile
    2. Proposes a cleaning plan using LLM
    3. Executes the cleaning plan
    4. Reports before/after statistics
    
    Args:
        state: Current pipeline state
    
    Returns:
        Updated pipeline state
    """
    config = get_config()
    run_dir = state["run_dir"]
    target = state["target"]
    
    # =========================================================================
    # Step 1: Load the raw data
    # =========================================================================
    raw_data_path = state.get("raw_data_path")
    if not raw_data_path:
        add_error(state, "cleaner", "No raw data path in state")
        return state
    
    try:
        df = load_dataframe(raw_data_path)
    except Exception as e:
        add_error(state, "cleaner", f"Failed to load data: {str(e)}")
        return state
    
    # =========================================================================
    # Step 2: Get data profile and generate cleaning suggestions
    # =========================================================================
    profile = state.get("data_summary", {})
    pii_warnings = state.get("pii_warnings", [])
    leakage_warnings = state.get("leakage_warnings", [])
    
    # Get deterministic suggestions
    base_suggestions = suggest_cleaning_steps(profile)
    
    # =========================================================================
    # Step 3: Use LLM to create comprehensive cleaning plan
    # =========================================================================
    cleaning_plan = _get_llm_cleaning_plan(
        profile=profile,
        target=target,
        pii_warnings=pii_warnings,
        leakage_warnings=leakage_warnings,
        base_suggestions=base_suggestions,
        config=config,
        verbose=state.get("verbose", False)
    )
    
    if not cleaning_plan or not cleaning_plan.get("steps"):
        # Fallback to deterministic suggestions
        cleaning_plan = {
            "steps": base_suggestions,
            "rationale": "Using default cleaning suggestions"
        }
    
    # Ensure target column is never dropped
    cleaning_plan["steps"] = _filter_target_operations(cleaning_plan["steps"], target)
    
    state["cleaning_plan"] = safe_json_serialize(cleaning_plan)
    
    log_decision(
        state, "cleaner",
        f"Created cleaning plan with {len(cleaning_plan['steps'])} steps",
        cleaning_plan.get("rationale", ""),
        {"n_steps": len(cleaning_plan["steps"])}
    )
    
    # =========================================================================
    # Step 4: Execute the cleaning plan
    # =========================================================================
    df_cleaned, cleaning_report = apply_cleaning(df, cleaning_plan)

    resolved_target = resolve_column_name(df_cleaned, target)
    if resolved_target != target:
        log_decision(
            state,
            "cleaner",
            f"Synced target column after cleaning: '{target}' → '{resolved_target}'",
            "Cleaning steps renamed columns (e.g. lowercase); state target updated to match.",
            {"previous_target": target, "resolved_target": resolved_target},
        )
        state["target"] = resolved_target
        target = resolved_target
    
    # =========================================================================
    # Step 5: Calculate before/after statistics
    # =========================================================================
    cleaning_stats = get_cleaning_stats(df, df_cleaned)
    cleaning_report["stats"] = cleaning_stats
    
    state["cleaning_report"] = safe_json_serialize(cleaning_report)
    
    # =========================================================================
    # Step 6: Save cleaned data
    # =========================================================================
    cleaned_dir = get_run_subdir(run_dir, "cleaned")
    cleaned_path = os.path.join(cleaned_dir, "cleaned_data.csv")
    save_dataframe(df_cleaned, cleaned_path)
    state["cleaned_data_path"] = cleaned_path
    add_artifact(state, "cleaned_data", cleaned_path, "data")
    
    # Save cleaning report
    metrics_dir = get_run_subdir(run_dir, "metrics")
    report_path = os.path.join(metrics_dir, "cleaning_report.json")
    save_json(report_path, cleaning_report)
    add_artifact(state, "cleaning_report", report_path, "json")
    
    # =========================================================================
    # Step 7: Log summary
    # =========================================================================
    successful_steps = sum(1 for s in cleaning_report["steps_executed"] if s.get("success"))
    
    log_decision(
        state, "cleaner",
        "Completed data cleaning",
        f"Executed {successful_steps}/{len(cleaning_report['steps_executed'])} steps. "
        f"Rows: {cleaning_stats['rows_before']} → {cleaning_stats['rows_after']} "
        f"({cleaning_stats['rows_removed']} removed). "
        f"Cols: {cleaning_stats['cols_before']} → {cleaning_stats['cols_after']}.",
        cleaning_stats
    )
    
    return state


def _get_llm_cleaning_plan(
    profile: dict,
    target: str,
    pii_warnings: list,
    leakage_warnings: list,
    base_suggestions: list,
    config: Any,
    verbose: bool = False
) -> dict:
    """Use LLM to create a comprehensive cleaning plan."""
    if not config.llm_api_key:
        return {"steps": base_suggestions, "rationale": "Using default suggestions (no API key)"}
    
    try:
        llm = create_llm(config)
        
        # Prepare context for LLM
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
            "leakage_columns": [w["column"] for w in leakage_warnings if w.get("severity") == "blocking"]
        }
        
        prompt = f"""Create a data cleaning plan for this dataset.

Dataset Context:
{json.dumps(context, indent=2)}

PII Warnings:
{json.dumps(pii_warnings, indent=2) if pii_warnings else "None"}

Leakage Warnings:
{json.dumps(leakage_warnings, indent=2) if leakage_warnings else "None"}

Base Suggestions:
{json.dumps(base_suggestions, indent=2)}

Requirements:
1. NEVER drop or modify the target column: {target}
2. Consider dropping columns with >50% missing values
3. Consider dropping constant columns (no variance)
4. Handle PII columns appropriately (drop if not needed)
5. Address leakage risks (drop leaky columns)
6. Fill missing values in important columns
7. Remove duplicate rows
8. Be conservative - preserve as much useful data as possible

Create a cleaning plan with specific steps and rationale."""

        messages = [
            SystemMessage(content=CLEANER_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]
        
        response = invoke_llm(llm, messages, "Cleaner", "Generating cleaning plan", verbose)
        
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        
        result = json.loads(content)
        
        # Extract the cleaning plan
        if "cleaning_plan" in result:
            return result["cleaning_plan"]
        return result
    
    except Exception as e:
        return {"steps": base_suggestions, "rationale": f"Fallback to defaults: {str(e)}"}


def _filter_target_operations(steps: list, target: str) -> list:
    """
    Filter out any operations that would affect the target column.

    Case-insensitive so that a lowercased target (e.g. 'overall_impact')
    is still protected even when ``target`` retains its original casing.
    """
    target_lower = target.lower()
    filtered = []

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
            "clip_outliers", "convert_dtype"
        ):
            continue

        filtered.append(step)

    return filtered
