"""
Data Profiler Agent

Responsible for:
- Loading and understanding the dataset
- Inferring target column if not provided
- Inferring problem type (classification/regression)
- Profiling schema, missingness, cardinality
- Detecting PII candidates and leakage risks
"""

import os
import json
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage

from agenticml.ml.state import PipelineState, log_decision, add_artifact, add_error
from agenticml.ml.config import get_config
from agenticml.ml.tools.data_io import load_data, save_dataframe
from agenticml.ml.tools.profiling import (
    profile_dataframe,
    detect_pii,
    detect_leakage_risks,
    infer_problem_type,
    infer_target_column
)
from agenticml.ml.tools.artifacts import save_json
from agenticml.ml.tools.utils import get_run_subdir, safe_json_serialize
from agenticml.ml.tools.llm import invoke_llm
from agenticml.ml.tools.llm_factory import create_llm


PROFILER_SYSTEM_PROMPT = """You are a Data Profiler Agent in an ML pipeline.

Your role is to analyze datasets and provide insights about:
1. Data quality and structure
2. Appropriate target column selection
3. Problem type determination (classification vs regression)
4. Potential data issues (PII, leakage risks)

You receive data profiles and must make decisions with clear rationale.
Always explain your reasoning for any inferences or recommendations.

Respond in JSON format with the following structure:
{
    "target_decision": {
        "column": "column_name or null",
        "confidence": "high/medium/low",
        "rationale": "explanation"
    },
    "problem_type_decision": {
        "type": "classification/regression",
        "confidence": "high/medium/low", 
        "rationale": "explanation"
    },
    "data_quality_assessment": {
        "overall_quality": "good/fair/poor",
        "key_issues": ["issue1", "issue2"],
        "recommendations": ["rec1", "rec2"]
    },
    "warnings": ["warning1", "warning2"]
}
"""


def run_profiler_agent(state: PipelineState) -> PipelineState:
    """
    Run the profiler agent to analyze the dataset.
    
    This agent:
    1. Loads the data (or uses data already loaded by planner)
    2. Profiles the dataset
    3. Validates target and problem type (may already be set by planner)
    4. Detects PII and leakage risks
    5. Updates state with findings
    
    Args:
        state: Current pipeline state
    
    Returns:
        Updated pipeline state
    """
    config = get_config()
    run_dir = state["run_dir"]
    file_path = state["file_path"]
    
    # =========================================================================
    # Step 1: Load the data (or use existing if planner already loaded it)
    # =========================================================================
    raw_data_path = state.get("raw_data_path")
    
    if raw_data_path and os.path.exists(raw_data_path):
        # Planner already loaded and saved the data
        try:
            from agenticml.ml.tools.data_io import load_dataframe
            df = load_dataframe(raw_data_path)
            load_metadata = {
                "n_rows": len(df),
                "n_cols": len(df.columns),
                "source": "planner_cache"
            }
            log_decision(
                state, "profiler",
                "Using data loaded by planner",
                f"Dataset: {load_metadata['n_rows']} rows x {load_metadata['n_cols']} columns",
                load_metadata
            )
        except Exception as e:
            add_error(state, "profiler", f"Failed to load cached data: {str(e)}")
            state["stop_reason"] = "data_load_error"
            return state
    else:
        # Load fresh
        try:
            df, load_metadata = load_data(file_path)
        except Exception as e:
            add_error(state, "profiler", f"Failed to load data: {str(e)}")
            state["stop_reason"] = "data_load_error"
            return state
        
        # Save raw data copy
        raw_dir = get_run_subdir(run_dir, "raw")
        raw_path = os.path.join(raw_dir, "raw_data.csv")
        save_dataframe(df, raw_path)
        state["raw_data_path"] = raw_path
        add_artifact(state, "raw_data", raw_path, "data")
        
        log_decision(
            state, "profiler",
            "Loaded dataset",
            f"Loaded {load_metadata['n_rows']} rows x {load_metadata['n_cols']} columns from {file_path}",
            load_metadata
        )
    
    # =========================================================================
    # Step 2: Profile the dataset
    # =========================================================================
    target = state.get("target")
    profile = profile_dataframe(df, target)
    
    # Save profile
    metrics_dir = get_run_subdir(run_dir, "metrics")
    profile_path = os.path.join(metrics_dir, "data_profile.json")
    save_json(profile_path, profile)
    add_artifact(state, "data_profile", profile_path, "json")
    
    # =========================================================================
    # Step 3: Detect PII
    # =========================================================================
    pii_warnings = []
    if config.pii_detection_enabled:
        pii_warnings = detect_pii(df)
        state["pii_warnings"] = pii_warnings
        
        if pii_warnings:
            log_decision(
                state, "profiler",
                "Detected potential PII",
                f"Found {len(pii_warnings)} potential PII issues",
                {"pii_warnings": pii_warnings}
            )
    
    # =========================================================================
    # Step 4: Infer target if not provided
    # =========================================================================
    if not target:
        # First try deterministic inference
        inferred_target, infer_rationale = infer_target_column(df, profile)
        
        # Use LLM to validate/refine the inference
        llm_decision = _get_llm_target_decision(profile, inferred_target, infer_rationale, config, verbose=state.get("verbose", False))
        
        if llm_decision and llm_decision.get("target_decision", {}).get("column"):
            target = llm_decision["target_decision"]["column"]
            confidence = llm_decision["target_decision"].get("confidence", "medium")
            rationale = llm_decision["target_decision"].get("rationale", infer_rationale)
        else:
            target = inferred_target
            confidence = "low"
            rationale = infer_rationale
        
        if target:
            state["target"] = target
            log_decision(
                state, "profiler",
                f"Inferred target column: {target}",
                rationale,
                {"confidence": confidence}
            )
        else:
            add_error(state, "profiler", "Could not infer target column")
            state["stop_reason"] = "no_target"
            return state
    
    # =========================================================================
    # Step 5: Infer problem type if not provided
    # =========================================================================
    problem_type = state.get("problem_type")
    
    if not problem_type:
        # Deterministic inference
        inferred_type, type_rationale = infer_problem_type(df, target)
        
        # Use LLM to validate
        llm_decision = _get_llm_problem_type_decision(profile, target, inferred_type, type_rationale, config, verbose=state.get("verbose", False))
        
        if llm_decision and llm_decision.get("problem_type_decision", {}).get("type"):
            problem_type = llm_decision["problem_type_decision"]["type"]
            rationale = llm_decision["problem_type_decision"].get("rationale", type_rationale)
        else:
            problem_type = inferred_type
            rationale = type_rationale
        
        state["problem_type"] = problem_type
        log_decision(
            state, "profiler",
            f"Inferred problem type: {problem_type}",
            rationale
        )
    
    # =========================================================================
    # Step 6: Detect leakage risks
    # =========================================================================
    leakage_warnings = detect_leakage_risks(df, target, profile)
    state["leakage_warnings"] = leakage_warnings
    
    if leakage_warnings:
        log_decision(
            state, "profiler",
            "Detected potential leakage risks",
            f"Found {len(leakage_warnings)} potential leakage issues",
            {"leakage_warnings": leakage_warnings}
        )
    
    # =========================================================================
    # Step 7: Update state with profile
    # =========================================================================
    state["data_summary"] = safe_json_serialize(profile)
    
    # Set default metric if not provided
    if not state.get("user_metric"):
        state["user_metric"] = config.get_default_metric(problem_type)
    
    log_decision(
        state, "profiler",
        "Completed data profiling",
        f"Dataset: {profile['n_rows']} rows, {profile['n_cols']} cols. "
        f"Target: {target} ({problem_type}). "
        f"Numeric: {len(profile['numeric_columns'])}, Categorical: {len(profile['categorical_columns'])}",
        {
            "n_rows": profile["n_rows"],
            "n_cols": profile["n_cols"],
            "n_numeric": len(profile["numeric_columns"]),
            "n_categorical": len(profile["categorical_columns"]),
            "n_missing_cols": sum(1 for v in profile["missing_percentages"].values() if v > 0)
        }
    )
    
    return state


def _get_llm_target_decision(
    profile: dict,
    inferred_target: str,
    infer_rationale: str,
    config: Any,
    verbose: bool = False
) -> dict:
    """Use LLM to validate/refine target column inference."""
    if not config.llm_api_key:
        return {}
    
    try:
        llm = create_llm(config)
        
        # Prepare profile summary for LLM
        profile_summary = {
            "columns": profile["columns"],
            "dtypes": profile["dtypes"],
            "cardinality": profile["cardinality"],
            "missing_percentages": profile["missing_percentages"],
            "sample_values": profile["sample_values"]
        }
        
        prompt = f"""Analyze this dataset profile and determine the most appropriate target column.

Dataset Profile:
{json.dumps(profile_summary, indent=2)}

Deterministic inference suggested: {inferred_target}
Rationale: {infer_rationale}

Consider:
1. Column names that suggest a target (label, target, class, outcome, etc.)
2. Data types appropriate for prediction
3. Cardinality (binary/multiclass for classification, continuous for regression)
4. Position (last column is often the target)

Respond with your decision in the specified JSON format."""

        messages = [
            SystemMessage(content=PROFILER_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]
        
        response = invoke_llm(llm, messages, "Profiler", "Target column decision", verbose)
        
        # Parse JSON response
        content = response.content
        # Extract JSON from response
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        
        return json.loads(content)
    
    except Exception:
        return {}


def _get_llm_problem_type_decision(
    profile: dict,
    target: str,
    inferred_type: str,
    type_rationale: str,
    config: Any,
    verbose: bool = False
) -> dict:
    """Use LLM to validate/refine problem type inference."""
    if not config.llm_api_key:
        return {}
    
    try:
        llm = create_llm(config)
        
        target_info = {
            "column": target,
            "dtype": profile["dtypes"].get(target),
            "cardinality": profile["cardinality"].get(target),
            "sample_values": profile["sample_values"].get(target, []),
            "stats": profile.get("numeric_stats", {}).get(target, {})
        }
        
        prompt = f"""Determine the problem type (classification or regression) for this target column.

Target Column Info:
{json.dumps(target_info, indent=2)}

Deterministic inference: {inferred_type}
Rationale: {type_rationale}

Consider:
1. Data type (categorical = classification, continuous = regression)
2. Cardinality (few unique values = classification)
3. Value distribution (integers 0-10 might be classification)
4. Column name hints

Respond with your decision in the specified JSON format."""

        messages = [
            SystemMessage(content=PROFILER_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]
        
        response = invoke_llm(llm, messages, "Profiler", "Problem type decision", verbose)
        
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        
        return json.loads(content)
    
    except Exception:
        return {}
