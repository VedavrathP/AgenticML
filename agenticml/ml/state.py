"""
Shared state schema for the ML pipeline.

This module defines the PipelineState TypedDict that flows between all nodes
in the LangGraph. State is passed and mutated explicitly between nodes.
"""

from typing import TypedDict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum


# ============================================================================
# Issue Severity Levels
# ============================================================================

class IssueSeverity:
    """Severity levels for critic issues."""
    INFO = "info"           # Informational, no action needed
    WARN = "warn"           # Warning, should be addressed but not blocking
    BLOCKING = "blocking"   # Must be fixed before proceeding


# ============================================================================
# Human Interaction Status
# ============================================================================

class InteractionStatus(str, Enum):
    """Status of human-in-the-loop interaction."""
    PENDING = "pending"           # Waiting for user input
    ANSWERED = "answered"         # User has responded
    SKIPPED = "skipped"           # User chose to skip
    TIMEOUT = "timeout"           # User did not respond in time


# ============================================================================
# Data Classes for Structured State Components
# ============================================================================

@dataclass
class DataSummary:
    """Summary statistics and metadata about the dataset."""
    n_rows: int = 0
    n_cols: int = 0
    columns: list[str] = field(default_factory=list)
    dtypes: dict[str, str] = field(default_factory=dict)
    missing_counts: dict[str, int] = field(default_factory=dict)
    missing_percentages: dict[str, float] = field(default_factory=dict)
    cardinality: dict[str, int] = field(default_factory=dict)
    numeric_columns: list[str] = field(default_factory=list)
    categorical_columns: list[str] = field(default_factory=list)
    datetime_columns: list[str] = field(default_factory=list)
    text_columns: list[str] = field(default_factory=list)
    constant_columns: list[str] = field(default_factory=list)
    high_cardinality_columns: list[str] = field(default_factory=list)
    sample_values: dict[str, list] = field(default_factory=dict)
    numeric_stats: dict[str, dict] = field(default_factory=dict)  # min, max, mean, std, median


@dataclass
class CleaningStep:
    """A single cleaning operation."""
    action: str  # drop_column, fill_missing, remove_duplicates, etc.
    column: Optional[str] = None
    params: dict = field(default_factory=dict)
    rationale: str = ""


@dataclass
class CleaningPlan:
    """Plan for data cleaning operations."""
    steps: list[CleaningStep] = field(default_factory=list)
    rationale: str = ""


@dataclass
class CleaningReport:
    """Report of cleaning operations performed."""
    steps_executed: list[dict] = field(default_factory=list)
    rows_before: int = 0
    rows_after: int = 0
    cols_before: int = 0
    cols_after: int = 0
    changes_summary: str = ""


@dataclass
class PreprocessingPlan:
    """Plan for feature preprocessing."""
    numeric_strategy: str = "standard"  # standard, minmax, robust, none
    categorical_strategy: str = "onehot"  # onehot, ordinal, target
    handle_unknown: str = "ignore"  # ignore, error
    datetime_features: list[str] = field(default_factory=list)  # year, month, day, etc.
    text_strategy: str = "tfidf"  # tfidf, count, none
    columns_to_drop: list[str] = field(default_factory=list)
    rationale: str = ""


@dataclass
class SplitPlan:
    """Plan for train/test splitting."""
    strategy: str = "stratified"  # stratified, random, time_based, cv
    test_size: float = 0.2
    cv_folds: int = 5
    time_column: Optional[str] = None
    random_state: int = 42
    rationale: str = ""


@dataclass
class ModelCandidate:
    """A model candidate with its configuration."""
    name: str
    model_type: str  # e.g., "LogisticRegression", "RandomForest"
    params: dict = field(default_factory=dict)
    is_baseline: bool = False
    training_time: float = 0.0
    model_path: Optional[str] = None


@dataclass
class EvaluationResult:
    """Evaluation results for a single model."""
    model_name: str
    metrics: dict[str, float] = field(default_factory=dict)
    confusion_matrix: Optional[list[list[int]]] = None
    classification_report: Optional[dict] = None
    feature_importances: Optional[dict[str, float]] = None
    plots: list[str] = field(default_factory=list)  # paths to saved plots
    cv_scores: Optional[list[float]] = None
    cv_mean: Optional[float] = None
    cv_std: Optional[float] = None


@dataclass
class CriticIssue:
    """An issue identified by the critic agent."""
    severity: str  # info, warn, blocking
    category: str  # leakage, metrics, preprocessing, scores, etc.
    description: str
    recommendation: str = ""
    affected_component: str = ""  # which agent/step is affected


@dataclass
class Artifact:
    """Reference to a saved artifact."""
    name: str
    path: str
    artifact_type: str  # model, plot, data, json, report
    created_at: str = ""


@dataclass
class ClarificationQuestion:
    """A question that the Planning Agent wants to ask the user."""
    id: str
    category: str  # target, problem_type, class_imbalance, features, data_quality, etc.
    question: str
    options: list[str] = field(default_factory=list)  # Suggested options (can be empty for open-ended)
    default: Optional[str] = None  # Default value if user skips
    importance: str = "recommended"  # required, recommended, optional
    context: str = ""  # Additional context about why this question is asked
    answer: Optional[str] = None
    status: str = "pending"  # pending, answered, skipped


@dataclass 
class PlanningDecisions:
    """Decisions made during the planning phase."""
    # Target and problem type
    inferred_target: Optional[str] = None
    inferred_problem_type: Optional[str] = None
    target_rationale: str = ""
    
    # Class imbalance handling
    class_imbalance_detected: bool = False
    imbalance_ratio: Optional[float] = None
    imbalance_strategy: Optional[str] = None  # none, smote, undersample, class_weight, etc.
    imbalance_rationale: str = ""
    
    # Feature engineering plan
    feature_engineering_plan: dict = field(default_factory=dict)
    suggested_interactions: list[str] = field(default_factory=list)
    suggested_polynomial_features: list[str] = field(default_factory=list)
    suggested_binning: list[str] = field(default_factory=list)
    
    # Data quality decisions
    missing_value_strategy: Optional[str] = None  # drop, impute_mean, impute_median, impute_mode, knn
    outlier_strategy: Optional[str] = None  # none, clip, remove, winsorize
    
    # Overall rationale
    overall_rationale: str = ""


# ============================================================================
# Main Pipeline State
# ============================================================================

class PipelineState(TypedDict, total=False):
    """
    The shared state object that flows through the LangGraph pipeline.
    
    All fields are optional (total=False) to allow incremental updates.
    State is passed and mutated explicitly between nodes.
    """
    
    # Run metadata
    run_id: str
    file_path: str
    run_dir: str
    started_at: str
    
    # User inputs (may be inferred if not provided)
    target: Optional[str]
    problem_type: Optional[str]  # classification, regression
    user_metric: Optional[str]  # f1, roc_auc, rmse, etc.
    max_iterations: int
    
    # User constraints
    user_constraints: dict[str, Any]
    
    # Interactive mode flag
    interactive_mode: bool  # Whether to ask clarifying questions
    
    # Verbose mode flag
    verbose: bool  # Whether to print LLM prompts and responses
    
    # Planning phase
    planning_decisions: Optional[dict]  # Serialized PlanningDecisions
    clarification_questions: list[dict]  # List of serialized ClarificationQuestion
    pending_questions: list[dict]  # Questions waiting for user response
    planning_complete: bool  # Whether planning phase is done
    user_responses: dict[str, str]  # User responses to clarification questions
    
    # Data state (stored as paths, not actual dataframes)
    raw_data_path: Optional[str]
    cleaned_data_path: Optional[str]
    
    # Profiling results
    data_summary: Optional[dict]  # Serialized DataSummary
    pii_warnings: list[dict]  # List of PII detection warnings
    leakage_warnings: list[dict]  # Potential data leakage warnings
    
    # Cleaning
    cleaning_plan: Optional[dict]  # Serialized CleaningPlan
    cleaning_report: Optional[dict]  # Serialized CleaningReport
    
    # Preprocessing
    preprocessing_plan: Optional[dict]  # Serialized PreprocessingPlan
    preprocessor_path: Optional[str]  # Path to saved sklearn pipeline
    
    # Feature engineering (enhanced)
    feature_engineering_applied: list[dict]  # List of applied feature engineering steps
    smote_applied: bool  # Whether SMOTE was applied for class imbalance
    new_features_created: list[str]  # Names of newly created features
    
    # Splitting
    split_plan: Optional[dict]  # Serialized SplitPlan
    train_data_path: Optional[str]
    test_data_path: Optional[str]
    cv_splitter: bool  # True if using cross-validation instead of train/test split
    
    # Feature info
    feature_names: list[str]  # Names of features after preprocessing
    n_train_samples: int  # Number of training samples
    n_test_samples: int  # Number of test samples (0 if CV)
    n_features: int  # Number of features
    
    # Modeling
    model_candidates: list[dict]  # List of serialized ModelCandidate
    trained_models: list[dict]  # Models that have been trained
    
    # Evaluation
    evaluation_results: list[dict]  # List of serialized EvaluationResult
    best_model: Optional[dict]  # The selected best model
    best_model_path: Optional[str]
    
    # Critic review
    critic_issues: list[dict]  # List of serialized CriticIssue
    has_blocking_issues: bool
    should_iterate: bool  # Set by critic, consumed by route_after_critic
    
    # Iteration control
    iteration: int
    stop_reason: Optional[str]  # max_iterations, quality_satisfied, no_issues, etc.
    iteration_guidance: Optional[dict]  # Guidance for current iteration from orchestrator
    
    # Artifacts
    artifacts: list[dict]  # List of serialized Artifact
    
    # Agent decision logs (for auditability)
    decision_log: list[dict]  # Log of all agent decisions with rationale
    
    # Error tracking
    errors: list[dict]  # Any errors encountered during execution


# ============================================================================
# State Helper Functions
# ============================================================================

def create_initial_state(
    run_id: str,
    file_path: str,
    run_dir: str,
    target: Optional[str] = None,
    problem_type: Optional[str] = None,
    user_metric: Optional[str] = None,
    max_iterations: int = 5,
    user_constraints: Optional[dict] = None,
    interactive_mode: bool = False,
    verbose: bool = False
) -> PipelineState:
    """Create the initial state for a pipeline run."""
    return PipelineState(
        run_id=run_id,
        file_path=file_path,
        run_dir=run_dir,
        started_at=datetime.now().isoformat(),
        target=target,
        problem_type=problem_type,
        user_metric=user_metric,
        max_iterations=max_iterations,
        user_constraints=user_constraints or {},
        interactive_mode=interactive_mode,
        verbose=verbose,
        # Planning phase
        clarification_questions=[],
        pending_questions=[],
        planning_complete=False,
        user_responses={},
        # Feature engineering
        feature_engineering_applied=[],
        smote_applied=False,
        new_features_created=[],
        # Standard fields
        pii_warnings=[],
        leakage_warnings=[],
        model_candidates=[],
        trained_models=[],
        evaluation_results=[],
        critic_issues=[],
        has_blocking_issues=False,
        should_iterate=False,
        iteration=0,
        artifacts=[],
        decision_log=[],
        errors=[]
    )


def log_decision(
    state: PipelineState,
    agent: str,
    decision: str,
    rationale: str,
    details: Optional[dict] = None
) -> None:
    """Log an agent decision to the state for auditability."""
    if "decision_log" not in state:
        state["decision_log"] = []
    
    state["decision_log"].append({
        "timestamp": datetime.now().isoformat(),
        "agent": agent,
        "decision": decision,
        "rationale": rationale,
        "details": details or {},
        "iteration": state.get("iteration", 0)
    })


def add_artifact(
    state: PipelineState,
    name: str,
    path: str,
    artifact_type: str
) -> None:
    """Add an artifact reference to the state."""
    if "artifacts" not in state:
        state["artifacts"] = []
    
    state["artifacts"].append({
        "name": name,
        "path": path,
        "artifact_type": artifact_type,
        "created_at": datetime.now().isoformat()
    })


def add_error(
    state: PipelineState,
    agent: str,
    error: str,
    details: Optional[dict] = None
) -> None:
    """Log an error to the state."""
    if "errors" not in state:
        state["errors"] = []
    
    state["errors"].append({
        "timestamp": datetime.now().isoformat(),
        "agent": agent,
        "error": error,
        "details": details or {},
        "iteration": state.get("iteration", 0)
    })
