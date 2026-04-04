"""
Per-algorithm training tools.

Each function trains exactly ONE model with explicitly typed hyperparameters.
The ModelTrainingAgent's LLM decides which function to call and with what
parameters — so only the requested model is trained, not all candidates.

Every function returns a standardised result dict:
{
    "name": str,
    "success": bool,
    "model": <fitted sklearn estimator>,
    "training_time": float,
    "error": str | None,
    "params_used": dict,
}
"""

import time
from typing import Any, Optional

import numpy as np


def _train(model: Any, X: np.ndarray, y: np.ndarray, name: str, params: dict) -> dict:
    """Shared training harness used by every per-algorithm function."""
    result: dict[str, Any] = {
        "name": name,
        "success": False,
        "model": None,
        "training_time": 0.0,
        "error": None,
        "params_used": params,
    }
    start = time.time()
    try:
        model.fit(X, y)
        result["success"] = True
        result["model"] = model
    except Exception as exc:
        result["error"] = str(exc)
    result["training_time"] = time.time() - start
    return result


# ── Classification ──────────────────────────────────────────────────────────

def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_iter: int = 1000,
    C: float = 1.0,
    penalty: str = "l2",
    solver: str = "lbfgs",
    class_weight: Optional[str] = None,
    random_state: int = 42,
) -> dict:
    from sklearn.linear_model import LogisticRegression
    params = dict(max_iter=max_iter, C=C, penalty=penalty, solver=solver,
                  class_weight=class_weight, random_state=random_state)
    return _train(LogisticRegression(**params), X, y, "LogisticRegression", params)


def train_random_forest_classifier(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: Optional[str] = "sqrt",
    class_weight: Optional[str] = None,
    n_jobs: int = -1,
    random_state: int = 42,
) -> dict:
    from sklearn.ensemble import RandomForestClassifier
    params = dict(n_estimators=n_estimators, max_depth=max_depth,
                  min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                  max_features=max_features, class_weight=class_weight,
                  n_jobs=n_jobs, random_state=random_state)
    return _train(RandomForestClassifier(**params), X, y, "RandomForestClassifier", params)


def train_gradient_boosting_classifier(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    subsample: float = 1.0,
    random_state: int = 42,
) -> dict:
    from sklearn.ensemble import GradientBoostingClassifier
    params = dict(n_estimators=n_estimators, learning_rate=learning_rate,
                  max_depth=max_depth, min_samples_split=min_samples_split,
                  min_samples_leaf=min_samples_leaf, subsample=subsample,
                  random_state=random_state)
    return _train(GradientBoostingClassifier(**params), X, y, "GradientBoostingClassifier", params)


def train_svc(
    X: np.ndarray,
    y: np.ndarray,
    *,
    C: float = 1.0,
    kernel: str = "rbf",
    gamma: str = "scale",
    probability: bool = True,
    class_weight: Optional[str] = None,
    random_state: int = 42,
) -> dict:
    from sklearn.svm import SVC
    params = dict(C=C, kernel=kernel, gamma=gamma, probability=probability,
                  class_weight=class_weight, random_state=random_state)
    return _train(SVC(**params), X, y, "SVC", params)


def train_kneighbors_classifier(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_neighbors: int = 5,
    weights: str = "uniform",
    metric: str = "minkowski",
    n_jobs: int = -1,
) -> dict:
    from sklearn.neighbors import KNeighborsClassifier
    params = dict(n_neighbors=n_neighbors, weights=weights, metric=metric, n_jobs=n_jobs)
    return _train(KNeighborsClassifier(**params), X, y, "KNeighborsClassifier", params)


def train_decision_tree_classifier(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    class_weight: Optional[str] = None,
    random_state: int = 42,
) -> dict:
    from sklearn.tree import DecisionTreeClassifier
    params = dict(max_depth=max_depth, min_samples_split=min_samples_split,
                  min_samples_leaf=min_samples_leaf, class_weight=class_weight,
                  random_state=random_state)
    return _train(DecisionTreeClassifier(**params), X, y, "DecisionTreeClassifier", params)


def train_xgb_classifier(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 6,
    subsample: float = 1.0,
    colsample_bytree: float = 1.0,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    use_label_encoder: bool = False,
    eval_metric: str = "logloss",
    random_state: int = 42,
) -> dict:
    from xgboost import XGBClassifier
    params = dict(n_estimators=n_estimators, learning_rate=learning_rate,
                  max_depth=max_depth, subsample=subsample,
                  colsample_bytree=colsample_bytree, reg_alpha=reg_alpha,
                  reg_lambda=reg_lambda, use_label_encoder=use_label_encoder,
                  eval_metric=eval_metric, random_state=random_state)
    return _train(XGBClassifier(**params), X, y, "XGBClassifier", params)


def train_lgbm_classifier(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = -1,
    num_leaves: int = 31,
    subsample: float = 1.0,
    colsample_bytree: float = 1.0,
    reg_alpha: float = 0.0,
    reg_lambda: float = 0.0,
    verbose: int = -1,
    random_state: int = 42,
) -> dict:
    from lightgbm import LGBMClassifier
    params = dict(n_estimators=n_estimators, learning_rate=learning_rate,
                  max_depth=max_depth, num_leaves=num_leaves, subsample=subsample,
                  colsample_bytree=colsample_bytree, reg_alpha=reg_alpha,
                  reg_lambda=reg_lambda, verbose=verbose, random_state=random_state)
    return _train(LGBMClassifier(**params), X, y, "LGBMClassifier", params)


# ── Regression ──────────────────────────────────────────────────────────────

def train_linear_regression(
    X: np.ndarray,
    y: np.ndarray,
    *,
    fit_intercept: bool = True,
) -> dict:
    from sklearn.linear_model import LinearRegression
    params = dict(fit_intercept=fit_intercept)
    return _train(LinearRegression(**params), X, y, "LinearRegression", params)


def train_ridge(
    X: np.ndarray,
    y: np.ndarray,
    *,
    alpha: float = 1.0,
    fit_intercept: bool = True,
    random_state: int = 42,
) -> dict:
    from sklearn.linear_model import Ridge
    params = dict(alpha=alpha, fit_intercept=fit_intercept, random_state=random_state)
    return _train(Ridge(**params), X, y, "Ridge", params)


def train_random_forest_regressor(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    max_features: Optional[str] = "sqrt",
    n_jobs: int = -1,
    random_state: int = 42,
) -> dict:
    from sklearn.ensemble import RandomForestRegressor
    params = dict(n_estimators=n_estimators, max_depth=max_depth,
                  min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                  max_features=max_features, n_jobs=n_jobs, random_state=random_state)
    return _train(RandomForestRegressor(**params), X, y, "RandomForestRegressor", params)


def train_gradient_boosting_regressor(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 3,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    subsample: float = 1.0,
    random_state: int = 42,
) -> dict:
    from sklearn.ensemble import GradientBoostingRegressor
    params = dict(n_estimators=n_estimators, learning_rate=learning_rate,
                  max_depth=max_depth, min_samples_split=min_samples_split,
                  min_samples_leaf=min_samples_leaf, subsample=subsample,
                  random_state=random_state)
    return _train(GradientBoostingRegressor(**params), X, y, "GradientBoostingRegressor", params)


def train_svr(
    X: np.ndarray,
    y: np.ndarray,
    *,
    C: float = 1.0,
    kernel: str = "rbf",
    gamma: str = "scale",
    epsilon: float = 0.1,
) -> dict:
    from sklearn.svm import SVR
    params = dict(C=C, kernel=kernel, gamma=gamma, epsilon=epsilon)
    return _train(SVR(**params), X, y, "SVR", params)


def train_decision_tree_regressor(
    X: np.ndarray,
    y: np.ndarray,
    *,
    max_depth: Optional[int] = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    random_state: int = 42,
) -> dict:
    from sklearn.tree import DecisionTreeRegressor
    params = dict(max_depth=max_depth, min_samples_split=min_samples_split,
                  min_samples_leaf=min_samples_leaf, random_state=random_state)
    return _train(DecisionTreeRegressor(**params), X, y, "DecisionTreeRegressor", params)


def train_xgb_regressor(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = 6,
    subsample: float = 1.0,
    colsample_bytree: float = 1.0,
    reg_alpha: float = 0.0,
    reg_lambda: float = 1.0,
    random_state: int = 42,
) -> dict:
    from xgboost import XGBRegressor
    params = dict(n_estimators=n_estimators, learning_rate=learning_rate,
                  max_depth=max_depth, subsample=subsample,
                  colsample_bytree=colsample_bytree, reg_alpha=reg_alpha,
                  reg_lambda=reg_lambda, random_state=random_state)
    return _train(XGBRegressor(**params), X, y, "XGBRegressor", params)


def train_lgbm_regressor(
    X: np.ndarray,
    y: np.ndarray,
    *,
    n_estimators: int = 100,
    learning_rate: float = 0.1,
    max_depth: int = -1,
    num_leaves: int = 31,
    subsample: float = 1.0,
    colsample_bytree: float = 1.0,
    reg_alpha: float = 0.0,
    reg_lambda: float = 0.0,
    verbose: int = -1,
    random_state: int = 42,
) -> dict:
    from lightgbm import LGBMRegressor
    params = dict(n_estimators=n_estimators, learning_rate=learning_rate,
                  max_depth=max_depth, num_leaves=num_leaves, subsample=subsample,
                  colsample_bytree=colsample_bytree, reg_alpha=reg_alpha,
                  reg_lambda=reg_lambda, verbose=verbose, random_state=random_state)
    return _train(LGBMRegressor(**params), X, y, "LGBMRegressor", params)


# ── Registry ────────────────────────────────────────────────────────────────

TOOL_REGISTRY: dict[str, callable] = {
    # classification
    "train_logistic_regression": train_logistic_regression,
    "train_random_forest_classifier": train_random_forest_classifier,
    "train_gradient_boosting_classifier": train_gradient_boosting_classifier,
    "train_svc": train_svc,
    "train_kneighbors_classifier": train_kneighbors_classifier,
    "train_decision_tree_classifier": train_decision_tree_classifier,
    "train_xgb_classifier": train_xgb_classifier,
    "train_lgbm_classifier": train_lgbm_classifier,
    # regression
    "train_linear_regression": train_linear_regression,
    "train_ridge": train_ridge,
    "train_random_forest_regressor": train_random_forest_regressor,
    "train_gradient_boosting_regressor": train_gradient_boosting_regressor,
    "train_svr": train_svr,
    "train_decision_tree_regressor": train_decision_tree_regressor,
    "train_xgb_regressor": train_xgb_regressor,
    "train_lgbm_regressor": train_lgbm_regressor,
}


def get_tool_schemas(problem_type: str) -> list[dict]:
    """
    Return JSON-serialisable parameter schemas for every training tool
    relevant to the given problem type.  The LLM uses these schemas to
    decide which tool to call and with what arguments.
    """
    import inspect

    classification_tools = [
        "train_logistic_regression", "train_random_forest_classifier",
        "train_gradient_boosting_classifier", "train_svc",
        "train_kneighbors_classifier", "train_decision_tree_classifier",
        "train_xgb_classifier", "train_lgbm_classifier",
    ]
    regression_tools = [
        "train_linear_regression", "train_ridge",
        "train_random_forest_regressor", "train_gradient_boosting_regressor",
        "train_svr", "train_decision_tree_regressor",
        "train_xgb_regressor", "train_lgbm_regressor",
    ]

    tool_names = classification_tools if problem_type == "classification" else regression_tools
    schemas: list[dict] = []

    for name in tool_names:
        fn = TOOL_REGISTRY.get(name)
        if fn is None:
            continue
        sig = inspect.signature(fn)
        params_schema: dict[str, str] = {}
        defaults: dict[str, Any] = {}
        for pname, param in sig.parameters.items():
            if pname in ("X", "y"):
                continue
            annotation = param.annotation
            type_str = getattr(annotation, "__name__", str(annotation))
            params_schema[pname] = type_str
            if param.default is not inspect.Parameter.empty:
                defaults[pname] = param.default

        schemas.append({
            "tool_name": name,
            "parameters": params_schema,
            "defaults": defaults,
        })

    return schemas
