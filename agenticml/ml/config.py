"""
Configuration and settings for the ML pipeline.

This module contains all configurable parameters, defaults, and feature flags.
"""

import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
# Check multiple locations: project root, app directory, current directory
_possible_env_paths = [
    Path(__file__).parent.parent / ".env",  # Project root
    Path(__file__).parent / ".env",          # App directory
    Path.cwd() / ".env",                     # Current working directory
]

for _env_path in _possible_env_paths:
    if _env_path.exists():
        load_dotenv(_env_path)
        break
else:
    load_dotenv()  # Try default behavior


# ============================================================================
# Feature Flags for Optional Dependencies
# ============================================================================

def _check_optional_dependency(package: str) -> bool:
    """Check if an optional dependency is available."""
    try:
        __import__(package)
        return True
    except ImportError:
        return False


XGBOOST_AVAILABLE = _check_optional_dependency("xgboost")
LIGHTGBM_AVAILABLE = _check_optional_dependency("lightgbm")
CATBOOST_AVAILABLE = _check_optional_dependency("catboost")
SHAP_AVAILABLE = _check_optional_dependency("shap")


# ============================================================================
# Configuration Dataclass
# ============================================================================

@dataclass
class PipelineConfig:
    """Configuration for the ML pipeline."""
    
    # LLM settings (provider auto-detected from model name)
    llm_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    llm_model: str = "gpt-4o"
    llm_temperature: float = 0.1  # Low temperature for consistent reasoning
    
    # Pipeline settings
    max_iterations: int = 5
    min_iterations: int = 2  # At least 2 iterations unless trivial
    
    # Data settings
    max_rows_for_cv: int = 10000  # Use CV for datasets smaller than this
    high_cardinality_threshold: int = 50  # Columns with more unique values are high cardinality
    pii_detection_enabled: bool = True
    
    # Modeling settings
    max_models: int = 6  # Maximum number of model candidates
    model_timeout_seconds: int = 300  # 5 minutes per model
    train_baseline_first: bool = True
    
    # Evaluation settings
    default_test_size: float = 0.2
    default_cv_folds: int = 5
    random_state: int = 42
    
    # Metric defaults by problem type
    default_classification_metric: str = "f1"
    default_regression_metric: str = "rmse"
    
    # Supported metrics
    classification_metrics: list[str] = field(default_factory=lambda: [
        "accuracy", "precision", "recall", "f1", "roc_auc", "log_loss"
    ])
    regression_metrics: list[str] = field(default_factory=lambda: [
        "rmse", "mae", "mse", "r2", "mape"
    ])
    
    # Critic thresholds
    suspicious_score_threshold: float = 0.99  # Scores above this are suspicious
    min_acceptable_score: float = 0.5  # Scores below this trigger warnings
    
    # Artifact settings
    runs_dir: str = "runs"
    save_intermediate_data: bool = True
    
    # Feature flags
    use_xgboost: bool = XGBOOST_AVAILABLE
    use_lightgbm: bool = LIGHTGBM_AVAILABLE
    use_catboost: bool = CATBOOST_AVAILABLE
    use_shap: bool = SHAP_AVAILABLE
    
    def get_default_metric(self, problem_type: str) -> str:
        """Get the default metric for a problem type."""
        if problem_type == "classification":
            return self.default_classification_metric
        elif problem_type == "regression":
            return self.default_regression_metric
        else:
            raise ValueError(f"Unknown problem type: {problem_type}")
    
    def is_valid_metric(self, metric: str, problem_type: str) -> bool:
        """Check if a metric is valid for the given problem type."""
        if problem_type == "classification":
            return metric in self.classification_metrics
        elif problem_type == "regression":
            return metric in self.regression_metrics
        return False
    
    def validate(self) -> list[str]:
        """Validate the configuration and return any issues."""
        from agenticml.ml.tools.llm_factory import detect_provider, resolve_api_key

        issues = []

        try:
            provider = detect_provider(self.llm_model)
            api_key = resolve_api_key(provider, self.llm_api_key)
            if not api_key:
                from agenticml.ml.tools.llm_factory import _ENV_KEY_MAP
                env_var = _ENV_KEY_MAP.get(provider, "OPENAI_API_KEY")
                issues.append(f"{env_var} is not set (needed for model '{self.llm_model}')")
        except ValueError as e:
            issues.append(str(e))

        if self.max_iterations < self.min_iterations:
            issues.append(f"max_iterations ({self.max_iterations}) must be >= min_iterations ({self.min_iterations})")

        if self.default_test_size <= 0 or self.default_test_size >= 1:
            issues.append(f"default_test_size must be between 0 and 1, got {self.default_test_size}")

        return issues


# ============================================================================
# Model Configurations
# ============================================================================

# Default model configurations for classification
CLASSIFICATION_MODELS = {
    "LogisticRegression": {
        "module": "sklearn.linear_model",
        "class": "LogisticRegression",
        "default_params": {"max_iter": 1000, "random_state": 42},
        "is_baseline": True,
        "complexity": "low"
    },
    "RandomForestClassifier": {
        "module": "sklearn.ensemble",
        "class": "RandomForestClassifier",
        "default_params": {"n_estimators": 100, "random_state": 42, "n_jobs": -1},
        "is_baseline": False,
        "complexity": "medium"
    },
    "GradientBoostingClassifier": {
        "module": "sklearn.ensemble",
        "class": "GradientBoostingClassifier",
        "default_params": {"n_estimators": 100, "random_state": 42},
        "is_baseline": False,
        "complexity": "medium"
    },
    "SVC": {
        "module": "sklearn.svm",
        "class": "SVC",
        "default_params": {"probability": True, "random_state": 42},
        "is_baseline": False,
        "complexity": "high"
    },
    "KNeighborsClassifier": {
        "module": "sklearn.neighbors",
        "class": "KNeighborsClassifier",
        "default_params": {"n_neighbors": 5},
        "is_baseline": False,
        "complexity": "low"
    },
    "DecisionTreeClassifier": {
        "module": "sklearn.tree",
        "class": "DecisionTreeClassifier",
        "default_params": {"random_state": 42},
        "is_baseline": False,
        "complexity": "low"
    }
}

# Default model configurations for regression
REGRESSION_MODELS = {
    "LinearRegression": {
        "module": "sklearn.linear_model",
        "class": "LinearRegression",
        "default_params": {},
        "is_baseline": True,
        "complexity": "low"
    },
    "Ridge": {
        "module": "sklearn.linear_model",
        "class": "Ridge",
        "default_params": {"random_state": 42},
        "is_baseline": False,
        "complexity": "low"
    },
    "RandomForestRegressor": {
        "module": "sklearn.ensemble",
        "class": "RandomForestRegressor",
        "default_params": {"n_estimators": 100, "random_state": 42, "n_jobs": -1},
        "is_baseline": False,
        "complexity": "medium"
    },
    "GradientBoostingRegressor": {
        "module": "sklearn.ensemble",
        "class": "GradientBoostingRegressor",
        "default_params": {"n_estimators": 100, "random_state": 42},
        "is_baseline": False,
        "complexity": "medium"
    },
    "SVR": {
        "module": "sklearn.svm",
        "class": "SVR",
        "default_params": {},
        "is_baseline": False,
        "complexity": "high"
    },
    "DecisionTreeRegressor": {
        "module": "sklearn.tree",
        "class": "DecisionTreeRegressor",
        "default_params": {"random_state": 42},
        "is_baseline": False,
        "complexity": "low"
    }
}

# Add XGBoost models if available
if XGBOOST_AVAILABLE:
    CLASSIFICATION_MODELS["XGBClassifier"] = {
        "module": "xgboost",
        "class": "XGBClassifier",
        "default_params": {"n_estimators": 100, "random_state": 42, "use_label_encoder": False, "eval_metric": "logloss"},
        "is_baseline": False,
        "complexity": "medium"
    }
    REGRESSION_MODELS["XGBRegressor"] = {
        "module": "xgboost",
        "class": "XGBRegressor",
        "default_params": {"n_estimators": 100, "random_state": 42},
        "is_baseline": False,
        "complexity": "medium"
    }

# Add LightGBM models if available
if LIGHTGBM_AVAILABLE:
    CLASSIFICATION_MODELS["LGBMClassifier"] = {
        "module": "lightgbm",
        "class": "LGBMClassifier",
        "default_params": {"n_estimators": 100, "random_state": 42, "verbose": -1},
        "is_baseline": False,
        "complexity": "medium"
    }
    REGRESSION_MODELS["LGBMRegressor"] = {
        "module": "lightgbm",
        "class": "LGBMRegressor",
        "default_params": {"n_estimators": 100, "random_state": 42, "verbose": -1},
        "is_baseline": False,
        "complexity": "medium"
    }


# ============================================================================
# PII Detection Patterns
# ============================================================================

PII_PATTERNS = {
    "email": {
        "pattern": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "description": "Email addresses"
    },
    "ssn": {
        "pattern": r"\b\d{3}-\d{2}-\d{4}\b",
        "description": "Social Security Numbers (US)"
    },
    "phone_us": {
        "pattern": r"\b(?:\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
        "description": "Phone numbers (US format)"
    },
    "credit_card": {
        "pattern": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
        "description": "Credit card numbers"
    },
    "ip_address": {
        "pattern": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
        "description": "IP addresses"
    }
}

# Column name patterns that suggest PII
PII_COLUMN_PATTERNS = [
    "email", "mail", "phone", "tel", "mobile", "ssn", "social_security",
    "credit_card", "card_number", "passport", "license", "address",
    "street", "zip", "postal", "birth", "dob", "age", "salary", "income",
    "password", "secret", "token", "key", "name", "first_name", "last_name",
    "full_name", "surname", "ip_address", "ip"
]


# ============================================================================
# Global Config Instance
# ============================================================================

_config_instance: Optional[PipelineConfig] = None

def get_config() -> PipelineConfig:
    """Get the global configuration singleton instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = PipelineConfig()
    return _config_instance


def reset_config() -> None:
    """Reset the global config so the next ``get_config()`` creates a fresh instance."""
    global _config_instance
    _config_instance = None
