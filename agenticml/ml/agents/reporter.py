"""
Reporter Agent

Responsible for:
- Generating comprehensive report.md
- Explaining dataset characteristics
- Documenting decisions made
- Summarizing models tried
- Explaining best model selection
- Noting limitations
- Providing reproducibility information
"""

import os
import json
from datetime import datetime
from typing import Any

from agenticml.ml.tools.llm_factory import create_llm
from langchain_core.messages import SystemMessage, HumanMessage

from agenticml.ml.state import PipelineState, log_decision, add_artifact
from agenticml.ml.config import get_config
from agenticml.ml.tools.artifacts import save_report, create_run_manifest, save_json
from agenticml.ml.tools.utils import get_run_subdir, format_duration
from agenticml.ml.tools.llm import invoke_llm


REPORTER_SYSTEM_PROMPT = """You are a Report Generation Agent in an ML pipeline.

Your role is to create a comprehensive, readable report that documents:
1. Dataset overview and characteristics
2. Data quality issues found
3. Preprocessing decisions and rationale
4. Models trained and their performance
5. Best model selection reasoning
6. Limitations and caveats
7. Recommendations for improvement

Write in clear, professional language suitable for both technical and non-technical stakeholders.
Use markdown formatting effectively.
"""


def run_reporter_agent(state: PipelineState) -> PipelineState:
    """
    Run the reporter agent to generate the final report.
    
    This agent:
    1. Gathers all pipeline information
    2. Generates a comprehensive markdown report
    3. Creates the run manifest
    4. Saves all final artifacts
    
    Args:
        state: Current pipeline state
    
    Returns:
        Updated pipeline state
    """
    config = get_config()
    run_dir = state["run_dir"]
    run_id = state["run_id"]
    
    # =========================================================================
    # Step 1: Generate report content
    # =========================================================================
    report_content = _generate_report(state, config, verbose=state.get("verbose", False))
    
    # =========================================================================
    # Step 2: Save report
    # =========================================================================
    report_path = os.path.join(run_dir, "report.md")
    save_report(report_content, report_path)
    add_artifact(state, "report", report_path, "report")
    
    # =========================================================================
    # Step 3: Create run manifest
    # =========================================================================
    manifest = create_run_manifest(run_dir, state, config.__dict__)
    
    log_decision(
        state, "reporter",
        "Generated final report and manifest",
        f"Report saved to {report_path}",
        {"report_path": report_path, "manifest_path": os.path.join(run_dir, "run_manifest.json")}
    )
    
    # =========================================================================
    # Step 4: Final summary
    # =========================================================================
    best_model = state.get("best_model", {})
    
    log_decision(
        state, "reporter",
        f"Pipeline complete: {state.get('stop_reason', 'completed')}",
        f"Best model: {best_model.get('name', 'N/A')} with score {best_model.get('primary_score', 0):.4f}",
        {
            "run_id": run_id,
            "iterations": state.get("iteration", 0) + 1,
            "best_model": best_model.get("name"),
            "best_score": best_model.get("primary_score")
        }
    )
    
    return state


def _generate_report(state: PipelineState, config: Any, verbose: bool = False) -> str:
    """Generate the markdown report content."""
    
    # Try to get LLM-enhanced report
    llm_sections = _get_llm_report_sections(state, config, verbose=verbose)
    
    # Build report
    report = []
    
    # Header
    report.append(f"# ML Pipeline Report")
    report.append(f"\n**Run ID:** {state.get('run_id')}")
    report.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"**Status:** {state.get('stop_reason', 'completed')}")
    report.append("")
    
    # Executive Summary
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
    
    # Dataset Overview
    report.append("## Dataset Overview")
    report.append("")
    
    data_summary = state.get("data_summary", {})
    report.append(f"- **Source:** {state.get('file_path', 'N/A')}")
    report.append(f"- **Rows:** {data_summary.get('n_rows', 'N/A'):,}")
    report.append(f"- **Columns:** {data_summary.get('n_cols', 'N/A')}")
    report.append(f"- **Target:** {state.get('target', 'N/A')}")
    report.append(f"- **Problem Type:** {state.get('problem_type', 'N/A')}")
    report.append("")
    
    # Column breakdown
    report.append("### Column Types")
    report.append("")
    report.append(f"- Numeric: {len(data_summary.get('numeric_columns', []))}")
    report.append(f"- Categorical: {len(data_summary.get('categorical_columns', []))}")
    report.append(f"- Datetime: {len(data_summary.get('datetime_columns', []))}")
    report.append("")
    
    # Data Quality
    report.append("## Data Quality")
    report.append("")
    
    # Missing values
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
    
    # PII warnings
    pii_warnings = state.get("pii_warnings", [])
    if pii_warnings:
        report.append("### PII Warnings")
        report.append("")
        for warning in pii_warnings[:5]:
            report.append(f"- **{warning.get('column')}**: {warning.get('description')}")
        report.append("")
    
    # Leakage warnings
    leakage_warnings = state.get("leakage_warnings", [])
    if leakage_warnings:
        report.append("### Leakage Risks")
        report.append("")
        for warning in leakage_warnings[:5]:
            report.append(f"- **{warning.get('column')}**: {warning.get('description')}")
        report.append("")
    
    # Preprocessing
    report.append("## Preprocessing")
    report.append("")
    
    cleaning_report = state.get("cleaning_report", {})
    if cleaning_report:
        stats = cleaning_report.get("stats", {})
        report.append("### Cleaning Summary")
        report.append("")
        report.append(f"- Rows: {stats.get('rows_before', 'N/A')} → {stats.get('rows_after', 'N/A')} ({stats.get('rows_removed', 0)} removed)")
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
    
    # Model Results
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
    
    # Best Model Details
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
    
    # Issues and Warnings
    critic_issues = state.get("critic_issues", [])
    if critic_issues:
        report.append("## Issues and Warnings")
        report.append("")
        
        blocking = [i for i in critic_issues if i.get("severity") == "blocking"]
        warnings = [i for i in critic_issues if i.get("severity") == "warn"]
        info = [i for i in critic_issues if i.get("severity") == "info"]
        
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
    
    # Limitations
    report.append("## Limitations")
    report.append("")
    
    if llm_sections.get("limitations"):
        report.append(llm_sections["limitations"])
    else:
        report.append("- Model performance is based on the provided dataset and may not generalize to new data")
        report.append("- Feature importance is model-specific and may vary between algorithms")
        report.append("- Hyperparameters were not extensively tuned")
    
    report.append("")
    
    # Reproducibility
    report.append("## Reproducibility")
    report.append("")
    report.append(f"- **Run ID:** {state.get('run_id')}")
    report.append(f"- **Random State:** 42")
    report.append(f"- **Iterations:** {state.get('iteration', 0) + 1}")
    report.append("")
    report.append("To reproduce this run, use the same input file and configuration.")
    report.append("")
    
    # Artifacts
    artifacts = state.get("artifacts", [])
    if artifacts:
        report.append("## Artifacts")
        report.append("")
        report.append("| Name | Type | Path |")
        report.append("|------|------|------|")
        for artifact in artifacts:
            report.append(f"| {artifact.get('name')} | {artifact.get('artifact_type')} | {artifact.get('path')} |")
        report.append("")
    
    return "\n".join(report)


def _get_llm_report_sections(state: PipelineState, config: Any, verbose: bool = False) -> dict:
    """Get LLM-generated report sections."""
    if not config.llm_api_key:
        return {}
    
    try:
        llm = create_llm(config)
        
        # Summarize state
        summary = {
            "problem_type": state.get("problem_type"),
            "target": state.get("target"),
            "n_rows": state.get("data_summary", {}).get("n_rows"),
            "best_model": state.get("best_model", {}).get("name"),
            "best_score": state.get("best_model", {}).get("primary_score"),
            "metric": state.get("user_metric"),
            "issues": len(state.get("critic_issues", [])),
            "iterations": state.get("iteration", 0) + 1
        }
        
        prompt = f"""Generate report sections for this ML pipeline run.

Summary:
{json.dumps(summary, indent=2)}

Generate:
1. A brief executive summary (2-3 sentences)
2. Key limitations to note (bullet points)

Keep it concise and professional."""

        messages = [
            SystemMessage(content=REPORTER_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]
        
        response = invoke_llm(llm, messages, "Reporter", "Report sections", verbose)
        content = response.content
        
        # Parse sections from response
        sections = {}
        
        if "executive summary" in content.lower():
            parts = content.split("limitations", 1)
            if len(parts) > 0:
                exec_part = parts[0]
                # Extract text after "executive summary"
                if ":" in exec_part:
                    sections["executive_summary"] = exec_part.split(":", 1)[1].strip()
        
        if "limitations" in content.lower():
            parts = content.lower().split("limitations", 1)
            if len(parts) > 1:
                sections["limitations"] = parts[1].strip()
        
        return sections
    
    except Exception:
        return {}
