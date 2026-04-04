"""
InsightVisualizationAgent – generates the final report and run manifest.

Refactored from the old ``reporter.py``.  All LLM interactions use
``invoke_llm_json`` and return structured JSON – no free-text parsing.
"""

import os
import json
from datetime import datetime
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage

from agenticml.agents.base_agent import BaseAgent
from agenticml.services.llm_service import get_llm, invoke_llm_json
from agenticml.state.workflow_state import (
    WorkflowState,
    log_decision,
    add_artifact,
    add_error,
    record_execution,
)
from agenticml.ml.config import get_config
from agenticml.services.artifact_service import save_report, create_run_manifest, save_json
from agenticml.ml.tools.utils import get_run_subdir, format_duration, safe_json_serialize


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

INSIGHT_SYSTEM_PROMPT = (
    "You are a Report-Generation Agent in an automated ML pipeline. "
    "You must respond with a JSON object.\n\n"
    "Given the full pipeline context, produce a JSON object with:\n"
    "{\n"
    '    "executive_summary": "2-3 sentence overview",\n'
    '    "key_findings": ["finding 1", "finding 2", ...],\n'
    '    "limitations": ["limitation 1", ...],\n'
    '    "recommendations": ["recommendation 1", ...]\n'
    "}"
)


# ---------------------------------------------------------------------------
# Report builder (preserved from old reporter.py)
# ---------------------------------------------------------------------------

def _generate_report(state: WorkflowState, llm_sections: dict) -> str:
    """Build the comprehensive markdown report."""
    report: list[str] = []

    # ── Header ────────────────────────────────────────────────────────
    report.append("# ML Pipeline Report")
    report.append(f"\n**Run ID:** {state.get('run_id')}")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**Status:** {state.get('stop_reason', 'completed')}")
    report.append("")

    # ── Executive Summary ─────────────────────────────────────────────
    report.append("## Executive Summary")
    report.append("")

    best_model = state.get("best_model", {})
    if best_model:
        report.append(f"**Best Model:** {best_model.get('name', 'N/A')}")
        report.append(f"**Primary Metric:** {state.get('user_metric', 'N/A')}")
        report.append(f"**Score:** {best_model.get('primary_score', 0):.4f}")

    if llm_sections.get("executive_summary"):
        report.append("")
        report.append(llm_sections["executive_summary"])

    report.append("")

    # ── Dataset Overview ──────────────────────────────────────────────
    report.append("## Dataset Overview")
    report.append("")

    data_summary = state.get("data_summary", {})
    n_rows = data_summary.get("n_rows", "N/A")
    n_rows_str = f"{n_rows:,}" if isinstance(n_rows, int) else str(n_rows)
    report.append(f"- **Source:** {state.get('file_path', 'N/A')}")
    report.append(f"- **Rows:** {n_rows_str}")
    report.append(f"- **Columns:** {data_summary.get('n_cols', 'N/A')}")
    report.append(f"- **Target:** {state.get('target', 'N/A')}")
    report.append(f"- **Problem Type:** {state.get('problem_type', 'N/A')}")
    report.append("")

    report.append("### Column Types")
    report.append("")
    report.append(f"- Numeric: {len(data_summary.get('numeric_columns', []))}")
    report.append(f"- Categorical: {len(data_summary.get('categorical_columns', []))}")
    report.append(f"- Datetime: {len(data_summary.get('datetime_columns', []))}")
    report.append("")

    # ── Data Quality ──────────────────────────────────────────────────
    report.append("## Data Quality")
    report.append("")

    missing = data_summary.get("missing_percentages", {})
    cols_with_missing = {k: v for k, v in missing.items() if v > 0}

    if cols_with_missing:
        report.append("### Missing Values")
        report.append("")
        report.append("| Column | Missing % |")
        report.append("|--------|-----------|")
        for col, pct in sorted(cols_with_missing.items(), key=lambda x: x[1], reverse=True)[:10]:
            report.append(f"| {col} | {pct:.1f}% |")
        report.append("")

    pii_warnings = state.get("pii_warnings", [])
    if pii_warnings:
        report.append("### PII Warnings")
        report.append("")
        for warning in pii_warnings[:5]:
            report.append(f"- **{warning.get('column')}**: {warning.get('description')}")
        report.append("")

    leakage_warnings = state.get("leakage_warnings", [])
    if leakage_warnings:
        report.append("### Leakage Risks")
        report.append("")
        for warning in leakage_warnings[:5]:
            report.append(f"- **{warning.get('column')}**: {warning.get('description')}")
        report.append("")

    # ── EDA Plots ─────────────────────────────────────────────────────
    eda_plots = [
        a for a in state.get("artifacts", [])
        if a.get("artifact_type") == "plot"
        and os.path.basename(a.get("path", "")).startswith("eda_")
    ]
    if eda_plots:
        report.append("## EDA Visualisations")
        report.append("")
        report.append(f"Generated {len(eda_plots)} exploratory plots during profiling:")
        report.append("")
        for artifact in eda_plots:
            name = os.path.basename(artifact["path"]).replace("eda_", "").replace(".png", "").replace("_", " ").title()
            report.append(f"- **{name}**: `{artifact['path']}`")
        report.append("")

    # ── Outlier Summary ──────────────────────────────────────────────
    outlier_summary = state.get("outlier_summary")
    if outlier_summary:
        report.append("## Outlier Summary")
        report.append("")
        report.append("| Column | Outliers | Outlier % | Method |")
        report.append("|--------|----------|-----------|--------|")
        for col, info in sorted(
            outlier_summary.items(),
            key=lambda x: x[1].get("outlier_percentage", 0),
            reverse=True,
        ):
            report.append(
                f"| {col} | {info.get('n_outliers', 0)} "
                f"| {info.get('outlier_percentage', 0):.1f}% "
                f"| {info.get('method', 'iqr')} |"
            )
        report.append("")

    # ── Preprocessing ─────────────────────────────────────────────────
    report.append("## Preprocessing")
    report.append("")

    cleaning_report = state.get("cleaning_report", {})
    if cleaning_report:
        stats = cleaning_report.get("stats", {}) if isinstance(cleaning_report, dict) else {}
        if stats:
            report.append("### Cleaning Summary")
            report.append("")
            report.append(
                f"- Rows: {stats.get('rows_before', 'N/A')} → "
                f"{stats.get('rows_after', 'N/A')} ({stats.get('rows_removed', 0)} removed)"
            )
            report.append(f"- Columns: {stats.get('cols_before', 'N/A')} → {stats.get('cols_after', 'N/A')}")
            report.append("")

    preprocessing_plan = state.get("preprocessing_plan", {})
    if preprocessing_plan:
        report.append("### Feature Engineering")
        report.append("")
        report.append(f"- Numeric scaling: {preprocessing_plan.get('numeric_strategy', 'N/A')}")
        report.append(f"- Categorical encoding: {preprocessing_plan.get('categorical_strategy', 'N/A')}")
        report.append("")

    split_plan = state.get("split_plan", {})
    if split_plan:
        report.append("### Data Split")
        report.append("")
        report.append(f"- Strategy: {split_plan.get('strategy', 'N/A')}")
        report.append(f"- Test size: {split_plan.get('test_size', 0.2) * 100:.0f}%")
        report.append("")

    # ── Model Results ─────────────────────────────────────────────────
    report.append("## Model Results")
    report.append("")

    evaluation_results = state.get("evaluation_results", [])
    successful_results = [r for r in evaluation_results if r.get("success")]

    if successful_results:
        metric = state.get("user_metric", "score")
        report.append(f"### Performance Comparison ({metric})")
        report.append("")
        report.append("| Model | Score | Training Time |")
        report.append("|-------|-------|---------------|")

        for result in sorted(successful_results, key=lambda x: x.get("primary_score", 0), reverse=True):
            score = result.get("primary_score", 0)
            time_str = format_duration(result.get("training_time", 0))
            baseline = " (baseline)" if result.get("is_baseline") else ""
            report.append(f"| {result['name']}{baseline} | {score:.4f} | {time_str} |")

        report.append("")

    # ── Best Model Details ────────────────────────────────────────────
    if best_model:
        report.append("### Best Model Details")
        report.append("")
        report.append(f"**{best_model.get('name')}**")
        report.append("")

        metrics = best_model.get("metrics", {})
        if metrics:
            report.append("Metrics:")
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    report.append(f"- {metric_name}: {value:.4f}")

        if best_model.get("selection_rationale"):
            report.append("")
            report.append(f"**Selection Rationale:** {best_model.get('selection_rationale')}")

        report.append("")

    # ── Issues and Warnings ───────────────────────────────────────────
    evaluation_issues = state.get("evaluation_issues", [])
    if evaluation_issues:
        report.append("## Issues and Warnings")
        report.append("")

        blocking = [i for i in evaluation_issues if i.get("severity") == "blocking"]
        warnings = [i for i in evaluation_issues if i.get("severity") == "warn"]
        info = [i for i in evaluation_issues if i.get("severity") == "info"]

        if blocking:
            report.append("### Blocking Issues")
            report.append("")
            for issue in blocking:
                report.append(f"- **{issue.get('category')}**: {issue.get('description')}")
                if issue.get("recommendation"):
                    report.append(f"  - *Recommendation:* {issue.get('recommendation')}")
            report.append("")

        if warnings:
            report.append("### Warnings")
            report.append("")
            for issue in warnings:
                report.append(f"- **{issue.get('category')}**: {issue.get('description')}")
            report.append("")

        if info:
            report.append("### Information")
            report.append("")
            for issue in info:
                report.append(f"- {issue.get('description')}")
            report.append("")

    # ── Key Findings (from LLM) ──────────────────────────────────────
    key_findings = llm_sections.get("key_findings", [])
    if key_findings:
        report.append("## Key Findings")
        report.append("")
        for finding in key_findings:
            report.append(f"- {finding}")
        report.append("")

    # ── Limitations ───────────────────────────────────────────────────
    report.append("## Limitations")
    report.append("")

    limitations = llm_sections.get("limitations", [])
    if limitations:
        for limitation in limitations:
            report.append(f"- {limitation}")
    else:
        report.append("- Model performance is based on the provided dataset and may not generalize to new data")
        report.append("- Feature importance is model-specific and may vary between algorithms")
        report.append("- Hyperparameters were not extensively tuned")

    report.append("")

    # ── Recommendations (from LLM) ───────────────────────────────────
    recommendations = llm_sections.get("recommendations", [])
    if recommendations:
        report.append("## Recommendations")
        report.append("")
        for rec in recommendations:
            report.append(f"- {rec}")
        report.append("")

    # ── Reproducibility ───────────────────────────────────────────────
    report.append("## Reproducibility")
    report.append("")
    report.append(f"- **Run ID:** {state.get('run_id')}")
    report.append("- **Random State:** 42")
    report.append(f"- **Iterations:** {state.get('iteration', 0) + 1}")
    report.append("")
    report.append("To reproduce this run, use the same input file and configuration.")
    report.append("")

    # ── Artifacts ─────────────────────────────────────────────────────
    artifacts = state.get("artifacts", [])
    if artifacts:
        report.append("## Artifacts")
        report.append("")
        report.append("| Name | Type | Path |")
        report.append("|------|------|------|")
        for artifact in artifacts:
            report.append(
                f"| {artifact.get('name')} | {artifact.get('artifact_type')} | {artifact.get('path')} |"
            )
        report.append("")

    return "\n".join(report)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class InsightVisualizationAgent(BaseAgent):
    """Generates the final report, insights, and run manifest."""

    name: str = "insight_visualization"

    # ------------------------------------------------------------------
    # public entry-point
    # ------------------------------------------------------------------
    def run(self, state: WorkflowState) -> WorkflowState:
        config = get_config()
        run_dir = state["run_dir"]

        # ── 1. LLM-generated insights ────────────────────────────────
        llm_sections = self._get_llm_insights(state, config)

        state["generated_insights"] = safe_json_serialize(llm_sections)

        log_decision(
            state, self.name,
            "Generated LLM insights",
            f"Sections: {list(llm_sections.keys())}",
            llm_sections,
        )

        # ── 2. Build markdown report ─────────────────────────────────
        report_content = _generate_report(state, llm_sections)

        report_path = os.path.join(run_dir, "report.md")
        save_report(report_content, report_path)
        add_artifact(state, "report", report_path, "report")

        # ── 3. Create run manifest ───────────────────────────────────
        manifest = create_run_manifest(run_dir, state, safe_json_serialize(config.__dict__))
        manifest_path = os.path.join(run_dir, "run_manifest.json")
        add_artifact(state, "run_manifest", manifest_path, "json")

        # ── 4. Collect generated plot paths ──────────────────────────
        generated_plots = [
            a["path"]
            for a in state.get("artifacts", [])
            if a.get("artifact_type") == "plot"
        ]
        state["generated_plots"] = generated_plots

        log_decision(
            state, self.name,
            "Generated final report and manifest",
            f"Report saved to {report_path}",
            {"report_path": report_path, "manifest_path": manifest_path},
        )

        # ── 5. Final summary log ─────────────────────────────────────
        best_model = state.get("best_model", {})
        best_score = best_model.get("primary_score", 0) if best_model else 0

        log_decision(
            state, self.name,
            f"Pipeline complete: {state.get('stop_reason', 'completed')}",
            f"Best model: {best_model.get('name', 'N/A')} with score {best_score:.4f}",
            {
                "run_id": state.get("run_id"),
                "iterations": state.get("iteration", 0) + 1,
                "best_model": best_model.get("name") if best_model else None,
                "best_score": best_score,
            },
        )

        record_execution(state, self.name)
        return state

    # ------------------------------------------------------------------
    # LLM call: executive summary, findings, limitations, recommendations
    # ------------------------------------------------------------------
    def _get_llm_insights(self, state: WorkflowState, config: Any) -> dict:
        llm = get_llm(config)

        best_model = state.get("best_model", {})

        summary_payload = json.dumps({
            "problem_type": state.get("problem_type"),
            "target": state.get("target"),
            "n_rows": state.get("data_summary", {}).get("n_rows"),
            "n_cols": state.get("data_summary", {}).get("n_cols"),
            "best_model": best_model.get("name") if best_model else None,
            "best_score": best_model.get("primary_score") if best_model else None,
            "metric": state.get("user_metric"),
            "evaluation_issues_count": len(state.get("evaluation_issues", [])),
            "iterations": state.get("iteration", 0) + 1,
            "evaluation_results": [
                {"name": r["name"], "score": r.get("primary_score")}
                for r in state.get("evaluation_results", [])
                if r.get("success")
            ],
        }, indent=2)

        messages = [
            SystemMessage(content=INSIGHT_SYSTEM_PROMPT),
            HumanMessage(content=summary_payload),
        ]

        return invoke_llm_json(
            llm, messages,
            agent_name=self.name,
            step_description="generate insights for final report",
            verbose=state.get("verbose", False),
        )
