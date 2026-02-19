"""
Planning Agent

Responsible for:
- Initial data inspection and understanding
- Identifying key decisions that need user input
- Generating clarification questions (human-in-the-loop)
- Making planning decisions about:
  - Target column and problem type inference
  - Class imbalance detection and handling strategy (SMOTE, etc.)
  - Feature engineering opportunities
  - Data quality strategies
"""

import os
import json
import uuid
from typing import Any, Optional

from langchain_core.messages import SystemMessage, HumanMessage

from agenticml.ml.state import PipelineState, log_decision, add_artifact, add_error
from agenticml.ml.config import get_config
from agenticml.ml.tools.data_io import load_dataframe, save_dataframe
from agenticml.ml.tools.profiling import profile_dataframe, infer_problem_type
from agenticml.ml.tools.artifacts import save_json
from agenticml.ml.tools.utils import get_run_subdir, safe_json_serialize
from agenticml.ml.tools.llm import invoke_llm
from agenticml.ml.tools.llm_factory import create_llm


PLANNER_SYSTEM_PROMPT = """You are a Planning Agent in an ML pipeline. Your role is to:

1. INSPECT the data carefully and understand its characteristics
2. IDENTIFY key decisions that need to be made before proceeding
3. DECIDE which questions (if any) require human clarification
4. PROPOSE a comprehensive plan for the ML pipeline

## Key Areas to Analyze:

### Target & Problem Type
- Identify the most likely target column if not specified
- Determine if it's classification or regression
- Check for multi-class vs binary classification

### Class Imbalance (CRITICAL for Classification)
- Calculate class distribution
- If imbalanced (minority class < 20%), consider:
  - SMOTE (Synthetic Minority Oversampling)
  - ADASYN (Adaptive Synthetic Sampling)
  - Random undersampling
  - Class weights in models
  - Combination approaches

### Feature Engineering Opportunities
- Identify columns that could benefit from:
  - Polynomial features (numeric columns with non-linear relationships)
  - Interaction features (combinations of related features)
  - Binning/discretization (continuous to categorical)
  - Log/sqrt transformations (skewed distributions)
  - Domain-specific features

### Data Quality
- Missing value patterns (random vs systematic)
- Outlier detection and handling
- Duplicate detection
- Data type issues

## When to Ask Questions:

Ask the user ONLY for decisions that significantly impact results:
1. Class imbalance strategy (SMOTE vs original data) - IMPORTANT
2. Target column confirmation if ambiguous
3. Feature engineering preferences if domain knowledge helps
4. Handling of specific problematic columns

DO NOT ask about:
- Technical implementation details
- Standard preprocessing choices
- Model selection (that's for the Modeler)

## Response Format:

Respond in JSON:
{
    "data_analysis": {
        "n_rows": int,
        "n_cols": int,
        "target_analysis": {
            "inferred_target": "column_name",
            "confidence": "high|medium|low",
            "rationale": "why this column"
        },
        "problem_type_analysis": {
            "inferred_type": "classification|regression",
            "is_binary": true|false,
            "n_classes": int,
            "rationale": "why this type"
        },
        "class_imbalance": {
            "is_imbalanced": true|false,
            "imbalance_ratio": float,
            "minority_class": "class_name",
            "majority_class": "class_name",
            "class_distribution": {"class1": count, "class2": count}
        },
        "feature_opportunities": [
            {
                "type": "interaction|polynomial|binning|transformation",
                "columns": ["col1", "col2"],
                "rationale": "why this would help"
            }
        ],
        "data_quality_issues": [
            {
                "issue": "description",
                "severity": "high|medium|low",
                "affected_columns": ["col1"]
            }
        ]
    },
    "clarification_questions": [
        {
            "category": "class_imbalance|target|features|data_quality",
            "question": "The question to ask the user",
            "options": ["Option 1", "Option 2", "Option 3"],
            "default": "Option 1",
            "importance": "required|recommended|optional",
            "context": "Why this question matters"
        }
    ],
    "planning_decisions": {
        "target": "column_name",
        "problem_type": "classification|regression",
        "imbalance_strategy": "none|smote|adasyn|undersample|class_weight",
        "feature_engineering": {
            "create_interactions": [["col1", "col2"]],
            "create_polynomial": ["col1"],
            "apply_binning": [{"column": "col1", "n_bins": 5}],
            "apply_log_transform": ["col1"]
        },
        "missing_strategy": "drop|impute_mean|impute_median|impute_mode|knn",
        "outlier_strategy": "none|clip|remove|winsorize",
        "overall_rationale": "Summary of planning decisions"
    }
}
"""


def run_planner_agent(state: PipelineState) -> PipelineState:
    """
    Run the planning agent to analyze data and prepare clarification questions.
    
    This agent:
    1. Loads and inspects the raw data
    2. Analyzes target, problem type, class imbalance
    3. Identifies feature engineering opportunities
    4. Generates clarification questions if needed
    5. Creates initial planning decisions
    
    Args:
        state: Current pipeline state
    
    Returns:
        Updated pipeline state with planning decisions and questions
    """
    config = get_config()
    run_dir = state["run_dir"]
    file_path = state["file_path"]
    user_target = state.get("target")
    user_problem_type = state.get("problem_type")
    interactive_mode = state.get("interactive_mode", False)
    
    # =========================================================================
    # Step 1: Load and save raw data
    # =========================================================================
    try:
        df = load_dataframe(file_path)
    except Exception as e:
        add_error(state, "planner", f"Failed to load data: {str(e)}")
        state["stop_reason"] = "data_load_error"
        return state
    
    # Save raw data copy
    raw_dir = get_run_subdir(run_dir, "raw")
    raw_path = os.path.join(raw_dir, "raw_data.csv")
    save_dataframe(df, raw_path)
    state["raw_data_path"] = raw_path
    add_artifact(state, "raw_data", raw_path, "data")
    
    log_decision(
        state, "planner",
        f"Loaded data: {len(df)} rows, {len(df.columns)} columns",
        f"File: {os.path.basename(file_path)}"
    )
    
    # =========================================================================
    # Step 2: Basic profiling for planning
    # =========================================================================
    # Infer target if not provided
    target = user_target
    if not target:
        # Try to infer target from common patterns
        target = _infer_target_column(df)
    
    if not target:
        add_error(state, "planner", "Could not infer target column")
        state["stop_reason"] = "no_target"
        return state
    
    # Profile the data
    profile = profile_dataframe(df, target)
    
    # Infer problem type if not provided
    problem_type = user_problem_type
    if not problem_type:
        problem_type = infer_problem_type(df, target)
    
    # =========================================================================
    # Step 3: Analyze class imbalance (for classification)
    # =========================================================================
    class_imbalance_info = _analyze_class_imbalance(df, target, problem_type)
    
    # =========================================================================
    # Step 4: Identify feature engineering opportunities
    # =========================================================================
    feature_opportunities = _identify_feature_opportunities(df, target, profile)
    
    # =========================================================================
    # Step 5: Use LLM to create comprehensive plan and questions
    # =========================================================================
    llm_analysis = _get_llm_planning_analysis(
        df=df,
        target=target,
        problem_type=problem_type,
        profile=profile,
        class_imbalance_info=class_imbalance_info,
        feature_opportunities=feature_opportunities,
        user_target=user_target,
        user_problem_type=user_problem_type,
        interactive_mode=interactive_mode,
        config=config,
        verbose=state.get("verbose", False)
    )
    
    # =========================================================================
    # Step 6: Process LLM response and update state
    # =========================================================================
    
    # Update target and problem type from LLM analysis
    if not user_target and llm_analysis.get("planning_decisions", {}).get("target"):
        state["target"] = llm_analysis["planning_decisions"]["target"]
    else:
        state["target"] = target
    
    if not user_problem_type and llm_analysis.get("planning_decisions", {}).get("problem_type"):
        state["problem_type"] = llm_analysis["planning_decisions"]["problem_type"]
    else:
        state["problem_type"] = problem_type
    
    # Store planning decisions
    planning_decisions = llm_analysis.get("planning_decisions", {})
    planning_decisions["class_imbalance_info"] = class_imbalance_info
    state["planning_decisions"] = safe_json_serialize(planning_decisions)
    
    # Store clarification questions
    questions = llm_analysis.get("clarification_questions", [])
    
    # Add unique IDs to questions
    for q in questions:
        q["id"] = str(uuid.uuid4())[:8]
        q["status"] = "pending"
        q["answer"] = None
    
    state["clarification_questions"] = questions
    
    # If interactive mode, mark questions as pending
    if interactive_mode and questions:
        state["pending_questions"] = questions
        state["planning_complete"] = False
    else:
        # Auto-answer with defaults
        for q in questions:
            q["status"] = "skipped"
            q["answer"] = q.get("default")
        state["pending_questions"] = []
        state["planning_complete"] = True
        
        # Apply default decisions
        _apply_default_decisions(state, planning_decisions, class_imbalance_info)
    
    # Save planning analysis
    metrics_dir = get_run_subdir(run_dir, "metrics")
    planning_path = os.path.join(metrics_dir, "planning_analysis.json")
    save_json(planning_path, {
        "data_analysis": llm_analysis.get("data_analysis", {}),
        "clarification_questions": questions,
        "planning_decisions": planning_decisions
    })
    add_artifact(state, "planning_analysis", planning_path, "json")
    
    log_decision(
        state, "planner",
        f"Planning complete: {len(questions)} clarification questions generated",
        planning_decisions.get("overall_rationale", ""),
        {"questions": len(questions), "interactive": interactive_mode}
    )
    
    return state


def process_user_responses(state: PipelineState) -> PipelineState:
    """
    Process user responses to clarification questions and update planning decisions.
    
    This function is called after the user has answered the questions.
    
    Args:
        state: Pipeline state with user_responses populated
    
    Returns:
        Updated state with finalized planning decisions
    """
    user_responses = state.get("user_responses", {})
    questions = state.get("clarification_questions", [])
    planning_decisions = state.get("planning_decisions", {})
    
    if isinstance(planning_decisions, str):
        planning_decisions = json.loads(planning_decisions) if planning_decisions else {}
    
    # Update questions with answers
    for q in questions:
        q_id = q.get("id")
        if q_id in user_responses:
            q["answer"] = user_responses[q_id]
            q["status"] = "answered"
        elif q.get("status") == "pending":
            q["answer"] = q.get("default")
            q["status"] = "skipped"
    
    state["clarification_questions"] = questions
    
    # Apply user decisions
    for q in questions:
        answer = q.get("answer")
        category = q.get("category")
        
        if not answer:
            continue
        
        if category == "class_imbalance":
            # Parse imbalance strategy from answer
            answer_lower = answer.lower()
            if "smote" in answer_lower:
                planning_decisions["imbalance_strategy"] = "smote"
            elif "original" in answer_lower or "no" in answer_lower or "none" in answer_lower:
                planning_decisions["imbalance_strategy"] = "none"
            elif "undersample" in answer_lower:
                planning_decisions["imbalance_strategy"] = "undersample"
            elif "class weight" in answer_lower or "weight" in answer_lower:
                planning_decisions["imbalance_strategy"] = "class_weight"
        
        elif category == "target":
            # User confirmed or changed target
            if answer in state.get("data_summary", {}).get("columns", []):
                state["target"] = answer
                planning_decisions["target"] = answer
        
        elif category == "features":
            # User preferences for feature engineering
            planning_decisions["user_feature_preferences"] = answer
    
    state["planning_decisions"] = safe_json_serialize(planning_decisions)
    state["pending_questions"] = []
    state["planning_complete"] = True
    
    log_decision(
        state, "planner",
        f"Processed {len(user_responses)} user responses",
        f"Updated planning decisions based on user input"
    )
    
    return state


def _infer_target_column(df) -> Optional[str]:
    """Infer the most likely target column from common patterns."""
    # Common target column names
    target_patterns = [
        "target", "label", "class", "y", "output", "outcome",
        "survived", "churn", "fraud", "default", "price", "salary",
        "revenue", "sales", "conversion", "response"
    ]
    
    columns_lower = {col.lower(): col for col in df.columns}
    
    # Check for exact matches first
    for pattern in target_patterns:
        if pattern in columns_lower:
            return columns_lower[pattern]
    
    # Check for partial matches
    for pattern in target_patterns:
        for col_lower, col in columns_lower.items():
            if pattern in col_lower:
                return col
    
    # If no match, return the last column (common convention)
    return df.columns[-1]


def _analyze_class_imbalance(df, target: str, problem_type: str) -> dict:
    """Analyze class imbalance for classification problems."""
    result = {
        "is_imbalanced": False,
        "imbalance_ratio": 1.0,
        "minority_class": None,
        "majority_class": None,
        "class_distribution": {},
        "recommendation": "none"
    }
    
    if problem_type != "classification":
        return result
    
    if target not in df.columns:
        return result
    
    # Get class distribution
    class_counts = df[target].value_counts()
    result["class_distribution"] = class_counts.to_dict()
    
    if len(class_counts) < 2:
        return result
    
    # Calculate imbalance ratio
    majority_count = class_counts.iloc[0]
    minority_count = class_counts.iloc[-1]
    majority_class = class_counts.index[0]
    minority_class = class_counts.index[-1]
    
    imbalance_ratio = minority_count / majority_count
    
    result["imbalance_ratio"] = round(imbalance_ratio, 4)
    result["minority_class"] = str(minority_class)
    result["majority_class"] = str(majority_class)
    result["minority_percentage"] = round(minority_count / len(df) * 100, 2)
    
    # Determine if imbalanced (threshold: minority < 30% of majority)
    if imbalance_ratio < 0.3:
        result["is_imbalanced"] = True
        
        # Recommend strategy based on dataset size and imbalance severity
        n_minority = minority_count
        
        if imbalance_ratio < 0.1:
            # Severe imbalance
            if n_minority < 100:
                result["recommendation"] = "smote"
                result["rationale"] = "Severe imbalance with few minority samples - SMOTE recommended"
            else:
                result["recommendation"] = "smote_or_class_weight"
                result["rationale"] = "Severe imbalance - consider SMOTE or class weights"
        elif imbalance_ratio < 0.3:
            # Moderate imbalance
            if n_minority < 500:
                result["recommendation"] = "smote"
                result["rationale"] = "Moderate imbalance - SMOTE can help"
            else:
                result["recommendation"] = "class_weight"
                result["rationale"] = "Moderate imbalance with sufficient samples - class weights may suffice"
    
    return result


def _identify_feature_opportunities(df, target: str, profile: dict) -> list:
    """Identify potential feature engineering opportunities."""
    opportunities = []
    
    numeric_cols = profile.get("numeric_columns", [])
    categorical_cols = profile.get("categorical_columns", [])
    
    # Check for skewed numeric columns (log transform candidates)
    numeric_stats = profile.get("numeric_stats", {})
    for col in numeric_cols:
        if col == target:
            continue
        stats = numeric_stats.get(col, {})
        if stats:
            # Check skewness via mean vs median
            mean = stats.get("mean", 0)
            median = stats.get("median", 0)
            if mean > 0 and median > 0:
                skew_indicator = abs(mean - median) / max(mean, median)
                if skew_indicator > 0.2 and stats.get("min", 0) >= 0:
                    opportunities.append({
                        "type": "log_transform",
                        "columns": [col],
                        "rationale": f"Column '{col}' appears skewed (mean/median ratio suggests right skew)"
                    })
    
    # Check for interaction candidates (numeric columns that might interact)
    if len(numeric_cols) >= 2:
        # Suggest interactions between related-looking columns
        for i, col1 in enumerate(numeric_cols[:5]):  # Limit to first 5
            for col2 in numeric_cols[i+1:6]:
                if col1 == target or col2 == target:
                    continue
                # Check if columns might be related (similar names or scales)
                if _columns_might_interact(col1, col2, df):
                    opportunities.append({
                        "type": "interaction",
                        "columns": [col1, col2],
                        "rationale": f"Potential interaction between '{col1}' and '{col2}'"
                    })
    
    # Check for binning candidates (continuous with many unique values)
    for col in numeric_cols:
        if col == target:
            continue
        n_unique = df[col].nunique()
        if n_unique > 20 and n_unique < len(df) * 0.5:
            # Many unique values but not continuous - might benefit from binning
            opportunities.append({
                "type": "binning",
                "columns": [col],
                "rationale": f"Column '{col}' has {n_unique} unique values - binning might capture non-linear patterns"
            })
    
    # Check for polynomial candidates (numeric with potential non-linear relationship)
    for col in numeric_cols[:3]:  # Limit polynomial suggestions
        if col == target:
            continue
        opportunities.append({
            "type": "polynomial",
            "columns": [col],
            "rationale": f"Consider polynomial features for '{col}' to capture non-linear relationships"
        })
    
    return opportunities[:10]  # Limit total opportunities


def _columns_might_interact(col1: str, col2: str, df) -> bool:
    """Check if two columns might have meaningful interaction."""
    # Check name similarity
    col1_parts = set(col1.lower().replace("_", " ").split())
    col2_parts = set(col2.lower().replace("_", " ").split())
    
    if col1_parts & col2_parts:  # Common words
        return True
    
    # Check correlation (if both numeric)
    try:
        corr = abs(df[col1].corr(df[col2]))
        if 0.3 < corr < 0.8:  # Moderate correlation suggests potential interaction
            return True
    except:
        pass
    
    return False


def _get_llm_planning_analysis(
    df,
    target: str,
    problem_type: str,
    profile: dict,
    class_imbalance_info: dict,
    feature_opportunities: list,
    user_target: Optional[str],
    user_problem_type: Optional[str],
    interactive_mode: bool,
    config: Any,
    verbose: bool = False
) -> dict:
    """Use LLM to create comprehensive planning analysis."""
    
    # Default response if no API key
    default_response = {
        "data_analysis": {
            "n_rows": profile.get("n_rows"),
            "n_cols": profile.get("n_cols"),
            "target_analysis": {
                "inferred_target": target,
                "confidence": "high" if user_target else "medium",
                "rationale": "User specified" if user_target else "Inferred from common patterns"
            },
            "problem_type_analysis": {
                "inferred_type": problem_type,
                "is_binary": class_imbalance_info.get("class_distribution", {}) and len(class_imbalance_info.get("class_distribution", {})) == 2,
                "rationale": "User specified" if user_problem_type else "Inferred from target distribution"
            },
            "class_imbalance": class_imbalance_info,
            "feature_opportunities": feature_opportunities
        },
        "clarification_questions": [],
        "planning_decisions": {
            "target": target,
            "problem_type": problem_type,
            "imbalance_strategy": class_imbalance_info.get("recommendation", "none"),
            "feature_engineering": {},
            "missing_strategy": "impute_median",
            "outlier_strategy": "none",
            "overall_rationale": "Default planning decisions based on data analysis"
        }
    }
    
    # Add class imbalance question if detected and interactive
    if interactive_mode and class_imbalance_info.get("is_imbalanced"):
        minority_pct = class_imbalance_info.get("minority_percentage", 0)
        default_response["clarification_questions"].append({
            "category": "class_imbalance",
            "question": f"The dataset has class imbalance (minority class is {minority_pct}% of data). How would you like to handle this?",
            "options": [
                "Use SMOTE to generate synthetic minority samples",
                "Use original data with class weights in models",
                "Use random undersampling of majority class",
                "Keep original data without balancing"
            ],
            "default": "Use SMOTE to generate synthetic minority samples" if class_imbalance_info.get("recommendation") == "smote" else "Use original data with class weights in models",
            "importance": "recommended",
            "context": f"Class imbalance can lead to biased models. The minority class '{class_imbalance_info.get('minority_class')}' has only {minority_pct}% of samples."
        })
    
    if not config.llm_api_key:
        return default_response
    
    try:
        llm = create_llm(config)
        
        # Prepare context for LLM
        sample_data = df.head(5).to_dict()
        
        context = {
            "dataset_info": {
                "n_rows": len(df),
                "n_cols": len(df.columns),
                "columns": list(df.columns),
                "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
                "sample_values": {col: df[col].head(3).tolist() for col in df.columns[:10]}
            },
            "user_inputs": {
                "target": user_target,
                "problem_type": user_problem_type
            },
            "inferred": {
                "target": target,
                "problem_type": problem_type
            },
            "profile_summary": {
                "numeric_columns": profile.get("numeric_columns", []),
                "categorical_columns": profile.get("categorical_columns", []),
                "missing_summary": {k: v for k, v in profile.get("missing_percentages", {}).items() if v > 0},
                "high_cardinality": profile.get("high_cardinality_columns", [])
            },
            "class_imbalance": class_imbalance_info,
            "feature_opportunities": feature_opportunities,
            "interactive_mode": interactive_mode
        }
        
        prompt = f"""Analyze this dataset and create a comprehensive ML pipeline plan.

Dataset Context:
{json.dumps(context, indent=2, default=str)}

Instructions:
1. Analyze the data characteristics thoroughly
2. {"Generate clarification questions for important decisions (especially class imbalance handling)" if interactive_mode else "Do NOT generate clarification questions - make autonomous decisions"}
3. Create planning decisions with clear rationale
4. Focus especially on class imbalance - this is critical for model performance

{"IMPORTANT: Only ask questions that truly require human judgment. For class imbalance, always ask if detected." if interactive_mode else "IMPORTANT: Make all decisions autonomously based on best practices."}

Provide your analysis in the specified JSON format."""

        messages = [
            SystemMessage(content=PLANNER_SYSTEM_PROMPT),
            HumanMessage(content=prompt)
        ]
        
        response = invoke_llm(llm, messages, "Planner", "Comprehensive planning analysis", verbose)
        
        content = response.content
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0]
        elif "```" in content:
            content = content.split("```")[1].split("```")[0]
        
        result = json.loads(content)
        
        # Ensure required fields exist
        if "planning_decisions" not in result:
            result["planning_decisions"] = default_response["planning_decisions"]
        if "clarification_questions" not in result:
            result["clarification_questions"] = []
        
        return result
    
    except Exception as e:
        # Return default response on error
        default_response["planning_decisions"]["overall_rationale"] = f"Default decisions (LLM error: {str(e)[:50]})"
        return default_response


def _apply_default_decisions(state: PipelineState, planning_decisions: dict, class_imbalance_info: dict):
    """Apply default planning decisions when not in interactive mode."""
    # Store imbalance strategy based on analysis
    imbalance_strategy = planning_decisions.get("imbalance_strategy", "none")
    
    if class_imbalance_info.get("is_imbalanced"):
        # Default to class_weight for moderate imbalance, SMOTE for severe
        if class_imbalance_info.get("imbalance_ratio", 1) < 0.1:
            imbalance_strategy = "smote"
        else:
            imbalance_strategy = "class_weight"
    
    planning_decisions["imbalance_strategy"] = imbalance_strategy
    state["planning_decisions"] = safe_json_serialize(planning_decisions)
    
    log_decision(
        state, "planner",
        f"Applied default decisions: imbalance_strategy={imbalance_strategy}",
        "Auto-applied based on data analysis (non-interactive mode)"
    )
