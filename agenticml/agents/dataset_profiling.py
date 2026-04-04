"""
Dataset Profiling Agent.

Loads the dataset, profiles schema / missingness / cardinality, detects PII
and leakage risks, then uses the LLM to infer (or validate) the target
column and problem type.  All LLM calls go through ``invoke_llm_json``
which guarantees a parsed JSON dict -- no free-text fallbacks.
"""

import json
import os
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage

from agenticml.agents.base_agent import BaseAgent
from agenticml.ml.config import get_config
from agenticml.ml.tools.data_io import load_data, save_dataframe, load_dataframe
from agenticml.ml.tools.profiling import (
    profile_dataframe,
    detect_pii,
    detect_leakage_risks,
    detect_outliers,
    get_column_correlations,
    infer_problem_type,
    infer_target_column,
)
from agenticml.ml.tools.eda_plots import generate_eda_plots
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
You are a Dataset Profiling Agent in an ML pipeline.

You receive a dataset profile together with a deterministic inference for the
target column and problem type.  Your job is to validate or override those
inferences based on the full profile context.

You must respond with a JSON object with exactly this schema:

{
    "target_column": "<column_name>",
    "problem_type": "classification" or "regression",
    "confidence": "high" | "medium" | "low",
    "reasoning": "<one-paragraph explanation>",
    "data_quality_assessment": "good" | "fair" | "poor",
    "warnings": ["<optional warning strings>"]
}

Rules:
- target_column MUST be one of the columns listed in the profile.
- problem_type MUST be exactly "classification" or "regression".
- Do NOT add any keys beyond those listed above.
"""


class DatasetProfilingAgent(BaseAgent):
    """Profile the dataset, infer target / problem type via LLM."""

    name = "dataset_profiling"

    def run(self, state: WorkflowState) -> WorkflowState:
        config = get_config()
        run_dir = state["run_dir"]
        file_path = state["file_path"]
        verbose = state.get("verbose", False)

        # ==================================================================
        # 1. Load data
        # ==================================================================
        df = self._load_dataset(state, file_path, run_dir, config)
        if df is None:
            record_execution(state, self.name, status="failed")
            return state

        # ==================================================================
        # 2. Profile
        # ==================================================================
        target = state.get("target")
        profile = profile_dataframe(df, target)

        metrics_dir = get_run_subdir(run_dir, "metrics")
        profile_path = os.path.join(metrics_dir, "data_profile.json")
        save_json(profile_path, profile)
        add_artifact(state, "data_profile", profile_path, "json")

        # ==================================================================
        # 3. PII detection
        # ==================================================================
        if config.pii_detection_enabled:
            pii_warnings = detect_pii(df)
            state["pii_warnings"] = pii_warnings
            if pii_warnings:
                log_decision(
                    state, self.name,
                    "Detected potential PII",
                    f"Found {len(pii_warnings)} potential PII issues",
                    {"pii_warnings": pii_warnings},
                )

        # ==================================================================
        # 4. Deterministic target / problem-type inference (context for LLM)
        # ==================================================================
        det_target, det_target_rationale = infer_target_column(df, profile)
        det_problem_type: str | None = None
        det_problem_rationale = ""
        if det_target:
            det_problem_type, det_problem_rationale = infer_problem_type(df, det_target)

        # ==================================================================
        # 5. LLM validation / refinement -- decision is FINAL
        # ==================================================================
        llm_decision = self._ask_llm(
            profile=profile,
            det_target=det_target,
            det_target_rationale=det_target_rationale,
            det_problem_type=det_problem_type,
            det_problem_rationale=det_problem_rationale,
            config=config,
            verbose=verbose,
        )

        target = llm_decision["target_column"]
        problem_type = llm_decision["problem_type"]

        state["target"] = target
        state["problem_type"] = problem_type

        log_decision(
            state, self.name,
            f"Inferred target={target}, problem_type={problem_type}",
            llm_decision.get("reasoning", ""),
            {
                "confidence": llm_decision.get("confidence"),
                "data_quality_assessment": llm_decision.get("data_quality_assessment"),
                "warnings": llm_decision.get("warnings", []),
            },
        )

        # ==================================================================
        # 6. Leakage detection (needs final target)
        # ==================================================================
        leakage_warnings = detect_leakage_risks(df, target, profile)
        state["leakage_warnings"] = leakage_warnings
        if leakage_warnings:
            log_decision(
                state, self.name,
                "Detected potential leakage risks",
                f"Found {len(leakage_warnings)} potential leakage issues",
                {"leakage_warnings": leakage_warnings},
            )

        # ==================================================================
        # 7. Store summary in state
        # ==================================================================
        state["data_summary"] = safe_json_serialize(profile)
        state["missing_value_summary"] = {
            col: pct
            for col, pct in profile.get("missing_percentages", {}).items()
            if pct > 0
        }

        if not state.get("user_metric"):
            state["user_metric"] = config.get_default_metric(problem_type)

        # ==================================================================
        # 8. Outlier detection
        # ==================================================================
        numeric_cols = [c for c in profile.get("numeric_columns", []) if c != target]
        outlier_info = detect_outliers(df, columns=numeric_cols)
        state["outlier_summary"] = safe_json_serialize(outlier_info)
        if outlier_info:
            log_decision(
                state, self.name,
                f"Detected outliers in {len(outlier_info)} columns",
                f"Columns with outliers: {list(outlier_info.keys())}",
                {"outlier_summary": outlier_info},
            )

        # ==================================================================
        # 9. High-correlation detection (feature-to-feature)
        # ==================================================================
        corr_info = get_column_correlations(df, target)
        high_corr_pairs = corr_info.get("high_correlations", [])
        # Keep only pairs where both columns are features (not target)
        high_corr_pairs = [
            p for p in high_corr_pairs
            if p["column1"] != target and p["column2"] != target
            and abs(p["correlation"]) >= 0.85
        ]
        state["high_correlation_pairs"] = safe_json_serialize(high_corr_pairs)
        if high_corr_pairs:
            log_decision(
                state, self.name,
                f"Detected {len(high_corr_pairs)} highly correlated feature pairs",
                ", ".join(
                    f"{p['column1']} <-> {p['column2']} ({p['correlation']:.2f})"
                    for p in high_corr_pairs[:5]
                ),
                {"high_correlation_pairs": high_corr_pairs},
            )

        # ==================================================================
        # 10. EDA plots
        # ==================================================================
        plots_dir = get_run_subdir(run_dir, "plots")
        eda_plot_paths = generate_eda_plots(
            df, target, problem_type, profile, plots_dir,
        )
        for plot_path in eda_plot_paths:
            add_artifact(state, os.path.basename(plot_path), plot_path, "plot")

        if eda_plot_paths:
            log_decision(
                state, self.name,
                f"Generated {len(eda_plot_paths)} EDA plots",
                f"Plots saved to {plots_dir}",
                {"eda_plots": [os.path.basename(p) for p in eda_plot_paths]},
            )

        log_decision(
            state, self.name,
            "Completed dataset profiling",
            (
                f"Dataset: {profile['n_rows']} rows, {profile['n_cols']} cols. "
                f"Target: {target} ({problem_type}). "
                f"Numeric: {len(profile['numeric_columns'])}, "
                f"Categorical: {len(profile['categorical_columns'])}"
            ),
            {
                "n_rows": profile["n_rows"],
                "n_cols": profile["n_cols"],
                "n_numeric": len(profile["numeric_columns"]),
                "n_categorical": len(profile["categorical_columns"]),
                "n_missing_cols": sum(
                    1 for v in profile["missing_percentages"].values() if v > 0
                ),
            },
        )

        record_execution(state, self.name)
        return state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_dataset(
        self,
        state: WorkflowState,
        file_path: str,
        run_dir: str,
        config: Any,
    ):
        """Return a DataFrame or ``None`` on failure (error added to state)."""
        raw_data_path = state.get("raw_data_path")

        if raw_data_path and os.path.exists(raw_data_path):
            try:
                return load_dataframe(raw_data_path)
            except Exception as exc:
                add_error(state, self.name, f"Failed to load cached data: {exc}")
                state["stop_reason"] = "data_load_error"
                return None

        try:
            df, load_metadata = load_data(file_path)
        except Exception as exc:
            add_error(state, self.name, f"Failed to load data: {exc}")
            state["stop_reason"] = "data_load_error"
            return None

        raw_dir = get_run_subdir(run_dir, "raw")
        raw_path = os.path.join(raw_dir, "raw_data.csv")
        save_dataframe(df, raw_path)
        state["raw_data_path"] = raw_path
        add_artifact(state, "raw_data", raw_path, "data")

        log_decision(
            state, self.name,
            "Loaded dataset",
            (
                f"Loaded {load_metadata['n_rows']} rows x "
                f"{load_metadata['n_cols']} columns from {file_path}"
            ),
            load_metadata,
        )
        return df

    def _ask_llm(
        self,
        *,
        profile: dict,
        det_target: str | None,
        det_target_rationale: str,
        det_problem_type: str | None,
        det_problem_rationale: str,
        config: Any,
        verbose: bool,
    ) -> dict:
        """Call the LLM to validate / refine target + problem-type inference."""
        llm = get_llm(config)

        profile_summary = {
            "columns": profile["columns"],
            "dtypes": profile["dtypes"],
            "cardinality": profile["cardinality"],
            "missing_percentages": profile["missing_percentages"],
            "sample_values": profile["sample_values"],
            "numeric_columns": profile["numeric_columns"],
            "categorical_columns": profile["categorical_columns"],
            "n_rows": profile["n_rows"],
            "n_cols": profile["n_cols"],
        }

        prompt = (
            "Analyze this dataset profile and determine the target column and "
            "problem type.\n\n"
            f"Dataset Profile:\n{json.dumps(profile_summary, indent=2)}\n\n"
            f"Deterministic target inference: {det_target}\n"
            f"Target rationale: {det_target_rationale}\n\n"
            f"Deterministic problem-type inference: {det_problem_type}\n"
            f"Problem-type rationale: {det_problem_rationale}\n\n"
            "Validate or override these inferences.  Respond with the JSON "
            "object described in the system prompt."
        )

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        return invoke_llm_json(
            llm,
            messages,
            agent_name=self.name,
            step_description="Target & problem-type inference",
            verbose=verbose,
        )
