"""
Critic Agent

Responsible for:
- Reviewing the entire pipeline for issues
- Detecting data leakage
- Identifying incorrect metrics usage
- Flagging suspiciously perfect scores
- Validating preprocessing steps
- Assigning severity levels (info/warn/blocking)
"""

import json
from typing import Any

from agenticml.ml.tools.llm_factory import create_llm
from langchain_core.messages import SystemMessage, HumanMessage

from agenticml.ml.state import PipelineState, log_decision, add_error, IssueSeverity
from agenticml.ml.config import get_config
from agenticml.ml.tools.utils import safe_json_serialize
from agenticml.ml.tools.llm import invoke_llm


CRITIC_SYSTEM_PROMPT = """You are a Critical Review Agent in an ML pipeline.

Your role is to review the entire ML pipeline and identify issues that could compromise model validity.

Issue Categories:
1. DATA LEAKAGE
   - Target information leaking into features
   - Future information in training data
   - Test data influencing preprocessing

2. METRIC ISSUES
   - Wrong metric for problem type
   - Misleading metric interpretation
   - Missing important metrics

3. PREPROCESSING ISSUES
   - Scaling/encoding applied before split
   - Missing value handling issues
   - Feature engineering problems

4. SUSPICIOUS SCORES
   - Perfect or near-perfect scores (>0.99)
   - Scores too good to be true
   - Large gap between train and test

5. DATA QUALITY
   - Insufficient data for model complexity
   - Class imbalance not addressed
   - Missing important features

Severity Levels:
- info: Informational, no action needed
- warn: Should be addressed but not blocking
- blocking: MUST be fixed before proceeding (forces another iteration)

Respond in JSON format:
{
    "review": {
        "issues": [
            {
                "severity": "info|warn|blocking",
                "category": "leakage|metrics|preprocessing|scores|data_quality",
                "description": "detailed description",
                "recommendation": "how to fix",
                "affected_component": "which part of pipeline"
            }
        ],
        "overall_assessment": "summary of pipeline quality",
        "should_iterate": true/false,
        "iteration_focus": "what to focus on if iterating"
    }
}
"""


def run_critic_agent(state: PipelineState) -> PipelineState:
    """
    Run the critic agent to review the pipeline.
    
    This agent:
    1. Reviews all pipeline decisions
    2. Checks for data leakage
    3. Validates metrics and scores
    4. Identifies issues with severity
    5. Decides if another iteration is needed
    
    Args:
        state: Current pipeline state
    
    Returns:
        Updated pipeline state
    """
    config = get_config()
    iteration = state.get("iteration", 0)
    max_iterations = state.get("max_iterations", config.max_iterations)
    
    # =========================================================================
    # Step 1: Gather pipeline information
    # =========================================================================
    pipeline_info = _gather_pipeline_info(state)
    
    # =========================================================================
    # Step 2: Run deterministic checks
    # =========================================================================
    deterministic_issues = _run_deterministic_checks(state, config)
    
    # =========================================================================
    # Step 3: Get LLM review
    # =========================================================================
    llm_review = _get_llm_review(pipeline_info, deterministic_issues, config, verbose=state.get("verbose", False))
    
    # =========================================================================
    # Step 4: Combine and deduplicate issues
    # =========================================================================
    all_issues = deterministic_issues.copy()
    
    if llm_review and llm_review.get("issues"):
        for issue in llm_review["issues"]:
            # Avoid duplicates
            if not any(
                i["category"] == issue.get("category") and 
                i["description"][:50] == issue.get("description", "")[:50]
                for i in all_issues
            ):
                all_issues.append(issue)
    
    state["critic_issues"] = safe_json_serialize(all_issues)
    
    # =========================================================================
    # Step 5: Determine if blocking issues exist
    # =========================================================================
    blocking_issues = [i for i in all_issues if i.get("severity") == IssueSeverity.BLOCKING]
    warning_issues = [i for i in all_issues if i.get("severity") == IssueSeverity.WARN]
    info_issues = [i for i in all_issues if i.get("severity") == IssueSeverity.INFO]
    
    has_blocking = len(blocking_issues) > 0
    state["has_blocking_issues"] = has_blocking
    
    # =========================================================================
    # Step 6: Log issues
    # =========================================================================
    for issue in all_issues:
        severity = issue.get("severity", "info")
        category = issue.get("category", "unknown")
        description = issue.get("description", "No description")
        
        log_decision(
            state, "critic",
            f"[{severity.upper()}] {category}: {description[:100]}",
            issue.get("recommendation", ""),
            issue
        )
    
    # =========================================================================
    # Step 7: Decide on iteration
    # =========================================================================
    should_iterate = False
    iteration_reason = ""
    
    if has_blocking and iteration < max_iterations - 1:
        should_iterate = True
        iteration_reason = f"Found {len(blocking_issues)} blocking issues"
    elif iteration < config.min_iterations - 1:
        should_iterate = True
        iteration_reason = f"Minimum iterations not reached ({iteration + 1}/{config.min_iterations})"
    
    if should_iterate:
        # Increment iteration here (in the node, where state persists).
        # The router (route_after_critic) only reads should_iterate to decide the route.
        state["iteration"] = iteration + 1
        state["should_iterate"] = True
        log_decision(
            state, "critic",
            f"Requesting iteration {iteration + 2}",
            iteration_reason,
            {
                "blocking_issues": len(blocking_issues),
                "warning_issues": len(warning_issues),
                "current_iteration": iteration + 1
            }
        )
    else:
        state["should_iterate"] = False
        if has_blocking:
            state["stop_reason"] = "max_iterations_with_issues"
        elif len(warning_issues) > 0:
            state["stop_reason"] = "completed_with_warnings"
        else:
            state["stop_reason"] = "completed_successfully"
        
        log_decision(
            state, "critic",
            f"Pipeline review complete: {state['stop_reason']}",
            f"Issues: {len(blocking_issues)} blocking, {len(warning_issues)} warnings, {len(info_issues)} info",
            {
                "blocking": len(blocking_issues),
                "warnings": len(warning_issues),
                "info": len(info_issues)
            }
        )
    
    return state


def _gather_pipeline_info(state: PipelineState) -> dict:
    """Gather all relevant pipeline information for review."""
    return {
        "data_summary": state.get("data_summary", {}),
        "target": state.get("target"),
        "problem_type": state.get("problem_type"),
        "cleaning_plan": state.get("cleaning_plan", {}),
        "cleaning_report": state.get("cleaning_report", {}),
        "preprocessing_plan": state.get("preprocessing_plan", {}),
        "split_plan": state.get("split_plan", {}),
        "model_candidates": state.get("model_candidates", []),
        "trained_models": state.get("trained_models", []),
        "evaluation_results": state.get("evaluation_results", []),
        "best_model": state.get("best_model", {}),
        "pii_warnings": state.get("pii_warnings", []),
        "leakage_warnings": state.get("leakage_warnings", []),
        "iteration": state.get("iteration", 0),
        "decision_log": state.get("decision_log", [])
    }


def _run_deterministic_checks(state: PipelineState, config: Any) -> list[dict]:
    """Run deterministic checks for common issues."""
    issues = []
    
    # Check 1: Suspiciously high scores
    evaluation_results = state.get("evaluation_results", [])
    for result in evaluation_results:
        if not result.get("success"):
            continue
        
        score = result.get("primary_score", 0)
        metric = result.get("primary_metric", "")
        
        # For metrics where higher is better
        if metric in ["accuracy", "f1", "precision", "recall", "roc_auc", "r2"]:
            if score > config.suspicious_score_threshold:
                issues.append({
                    "severity": IssueSeverity.BLOCKING,
                    "category": "scores",
                    "description": f"Suspiciously high {metric} score ({score:.4f}) for {result['name']}. This often indicates data leakage.",
                    "recommendation": "Review feature engineering for target leakage. Check if any features are derived from the target.",
                    "affected_component": "evaluation"
                })
            elif score < config.min_acceptable_score:
                issues.append({
                    "severity": IssueSeverity.WARN,
                    "category": "scores",
                    "description": f"Low {metric} score ({score:.4f}) for {result['name']}. Model may not be useful.",
                    "recommendation": "Consider feature engineering, different models, or more data.",
                    "affected_component": "modeling"
                })
    
    # Check 2: Unaddressed leakage warnings
    leakage_warnings = state.get("leakage_warnings", [])
    blocking_leakage = [w for w in leakage_warnings if w.get("severity") == "blocking"]
    
    if blocking_leakage:
        issues.append({
            "severity": IssueSeverity.BLOCKING,
            "category": "leakage",
            "description": f"Found {len(blocking_leakage)} unaddressed data leakage risks: {[w['column'] for w in blocking_leakage]}",
            "recommendation": "Remove or investigate columns with high target correlation.",
            "affected_component": "profiling"
        })
    
    # Check 3: PII in features
    pii_warnings = state.get("pii_warnings", [])
    blocking_pii = [w for w in pii_warnings if w.get("severity") == "blocking"]
    
    if blocking_pii:
        issues.append({
            "severity": IssueSeverity.WARN,
            "category": "data_quality",
            "description": f"PII detected in {len(blocking_pii)} columns that may still be in features",
            "recommendation": "Ensure PII columns are removed or properly anonymized.",
            "affected_component": "cleaning"
        })
    
    # Check 4: No successful models
    trained_models = state.get("trained_models", [])
    successful_models = [m for m in trained_models if m.get("success")]
    
    if not successful_models:
        issues.append({
            "severity": IssueSeverity.BLOCKING,
            "category": "modeling",
            "description": "No models were successfully trained",
            "recommendation": "Check data quality, feature engineering, and model configurations.",
            "affected_component": "modeling"
        })
    
    # Check 5: Only baseline model succeeded
    if len(successful_models) == 1 and successful_models[0].get("is_baseline"):
        issues.append({
            "severity": IssueSeverity.WARN,
            "category": "modeling",
            "description": "Only the baseline model was successfully trained",
            "recommendation": "Investigate why other models failed. May indicate data issues.",
            "affected_component": "modeling"
        })
    
    # Check 6: Large data loss during cleaning
    cleaning_report = state.get("cleaning_report", {})
    stats = cleaning_report.get("stats", {})
    
    if stats:
        rows_removed_pct = stats.get("rows_removed_pct", 0)
        if rows_removed_pct > 50:
            issues.append({
                "severity": IssueSeverity.WARN,
                "category": "data_quality",
                "description": f"Cleaning removed {rows_removed_pct:.1f}% of rows",
                "recommendation": "Review cleaning steps. Consider less aggressive cleaning.",
                "affected_component": "cleaning"
            })
    
    # Check 7: CV scores with high variance
    for result in evaluation_results:
        cv_std = result.get("cv_std")
        cv_mean = result.get("cv_mean")
        
        if cv_std and cv_mean and cv_mean > 0:
            cv_ratio = cv_std / cv_mean
            if cv_ratio > 0.2:
                issues.append({
                    "severity": IssueSeverity.WARN,
                    "category": "scores",
                    "description": f"High CV variance for {result['name']} (std/mean = {cv_ratio:.2f})",
                    "recommendation": "Model performance is unstable. Consider more data or simpler model.",
                    "affected_component": "evaluation"
                })
    
    return issues


def _get_llm_review(
    pipeline_info: dict,
    deterministic_issues: list,
    config: Any,
    verbose: bool = False
) -> dict:
    """Get LLM review of the pipeline."""
    if not config.llm_api_key:
        return {}
    
    try:
        llm = create_llm(config)
        
        # Summarize pipeline info
        summary = {
            "target": pipeline_info.get("target"),
            "problem_type": pipeline_info.get("problem_type"),
            "n_rows": pipeline_info.get("data_summary", {}).get("n_rows"),
            "n_features": len(pipeline_info.get("data_summary", {}).get("columns", [])),
            "cleaning_steps": len(pipeline_info.get("cleaning_plan", {}).get("steps", [])),
            "preprocessing": pipeline_info.get("preprocessing_plan", {}),
            "split_strategy": pipeline_info.get("split_plan", {}).get("strategy"),
            "models_trained": len([m for m in pipeline_info.get("trained_models", []) if m.get("success")]),
            "best_model": pipeline_info.get("best_model", {}).get("name"),
            "best_score": pipeline_info.get("best_model", {}).get("primary_score"),
            "evaluation_results": [
                {"name": r["name"], "score": r.get("primary_score")}
                for r in pipeline_info.get("evaluation_results", [])
                if r.get("success")
            ]
        }
        
        prompt = f"""Review this ML pipeline for issues.

Pipeline Summary:
{json.dumps(summary, indent=2)}

Already Identified Issues:
{json.dumps(deterministic_issues, indent=2)}

Review for:
1. Data leakage (target info in features, future info, test contamination)
2. Metric issues (wrong metric, misleading interpretation)
3. Preprocessing problems (leakage, missing handling)
4. Suspicious scores (too good, inconsistent)
5. Data quality concerns

Identify any additional issues not already found."""

        messages = [
            SystemMessage(content=CRITIC_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]
        
        response = invoke_llm(llm, messages, "Critic", "Pipeline review", verbose)
        
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        
        result = json.loads(content)
        return result.get("review", result)
    
    except Exception:
        return {}
