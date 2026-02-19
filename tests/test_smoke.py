"""
Smoke tests for the Multi-Agent ML Pipeline.

These tests verify that the pipeline runs end-to-end without errors
using the sklearn breast cancer dataset.
"""

import os
import tempfile
import shutil
import pytest
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_diabetes


class TestDataTools:
    """Test the data tools independently."""
    
    def test_load_csv(self, tmp_path):
        """Test loading a CSV file."""
        from agenticml.ml.tools.data_io import load_data, save_dataframe
        
        # Create test data
        df = pd.DataFrame({
            "a": [1, 2, 3],
            "b": ["x", "y", "z"],
            "target": [0, 1, 0]
        })
        
        csv_path = tmp_path / "test.csv"
        df.to_csv(csv_path, index=False)
        
        # Load and verify
        loaded_df, metadata = load_data(str(csv_path))
        
        assert len(loaded_df) == 3
        assert "a" in loaded_df.columns
        assert metadata["n_rows"] == 3
        assert metadata["n_cols"] == 3
    
    def test_profile_dataframe(self):
        """Test data profiling."""
        from agenticml.ml.tools.profiling import profile_dataframe
        
        df = pd.DataFrame({
            "numeric": [1.0, 2.0, 3.0, None],
            "categorical": ["a", "b", "a", "c"],
            "target": [0, 1, 0, 1]
        })
        
        profile = profile_dataframe(df, "target")
        
        assert profile["n_rows"] == 4
        assert profile["n_cols"] == 3
        assert "numeric" in profile["numeric_columns"]
        assert profile["missing_counts"]["numeric"] == 1
    
    def test_detect_pii(self):
        """Test PII detection."""
        from agenticml.ml.tools.profiling import detect_pii
        
        df = pd.DataFrame({
            "email": ["test@example.com", "user@test.org"],
            "name": ["John", "Jane"],
            "value": [1, 2]
        })
        
        warnings = detect_pii(df)
        
        # Should detect email column
        email_warnings = [w for w in warnings if w["column"] == "email"]
        assert len(email_warnings) > 0
    
    def test_apply_cleaning(self):
        """Test data cleaning."""
        from agenticml.ml.tools.cleaning import apply_cleaning
        
        df = pd.DataFrame({
            "a": [1, 2, 2, 3],
            "b": [None, None, None, None],
            "c": ["x", "y", "x", "z"]
        })
        
        plan = {
            "steps": [
                {"action": "drop_column", "column": "b"},
                {"action": "remove_duplicates", "params": {"keep": "first"}}
            ]
        }
        
        cleaned_df, report = apply_cleaning(df, plan)
        
        assert "b" not in cleaned_df.columns
        assert len(cleaned_df) == 3  # One duplicate removed
        assert report["cols_after"] == 2


class TestPreprocessing:
    """Test preprocessing tools."""
    
    def test_build_preprocess_pipeline(self):
        """Test building preprocessing pipeline."""
        from agenticml.ml.tools.preprocessing import build_preprocess_pipeline
        
        df = pd.DataFrame({
            "numeric": [1.0, 2.0, 3.0],
            "categorical": ["a", "b", "a"],
            "target": [0, 1, 0]
        })
        
        plan = {
            "numeric_strategy": "standard",
            "categorical_strategy": "onehot",
            "handle_unknown": "ignore"
        }
        
        preprocessor, feature_cols = build_preprocess_pipeline(df, "target", plan)
        
        assert preprocessor is not None
        assert "numeric" in feature_cols
        assert "categorical" in feature_cols
    
    def test_split_data(self):
        """Test data splitting."""
        from agenticml.ml.tools.preprocessing import split_data
        
        df = pd.DataFrame({
            "a": range(100),
            "b": range(100),
            "target": [0, 1] * 50
        })
        
        plan = {
            "strategy": "stratified",
            "test_size": 0.2,
            "random_state": 42
        }
        
        result = split_data(df, "target", plan)
        
        assert "X_train" in result
        assert "X_test" in result
        assert len(result["X_train"]) == 80
        assert len(result["X_test"]) == 20


class TestModeling:
    """Test modeling tools."""
    
    def test_get_model_candidates(self):
        """Test getting model candidates."""
        from agenticml.ml.tools.modeling import get_model_candidates
        
        candidates = get_model_candidates("classification", max_models=3)
        
        assert len(candidates) <= 3
        assert any(c["is_baseline"] for c in candidates)
    
    def test_create_and_train_model(self):
        """Test model creation and training."""
        from agenticml.ml.tools.modeling import create_model, train_model
        import numpy as np
        
        config = {
            "name": "LogisticRegression",
            "model_type": "LogisticRegression",
            "module": "sklearn.linear_model",
            "params": {"max_iter": 100}
        }
        
        model = create_model(config)
        
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        
        trained_model, info = train_model(model, X, y)
        
        assert info["success"]
        assert info["training_time"] > 0


class TestEvaluation:
    """Test evaluation tools."""
    
    def test_evaluate_classification_model(self):
        """Test classification model evaluation."""
        from agenticml.ml.tools.evaluation import evaluate_model
        from sklearn.linear_model import LogisticRegression
        import numpy as np
        
        X = np.random.randn(100, 5)
        y = np.random.randint(0, 2, 100)
        
        model = LogisticRegression(max_iter=100)
        model.fit(X, y)
        
        result = evaluate_model(model, X, y, "classification", "f1")
        
        assert "metrics" in result
        assert "f1" in result["metrics"]
        assert result["primary_metric"] == "f1"
    
    def test_evaluate_regression_model(self):
        """Test regression model evaluation."""
        from agenticml.ml.tools.evaluation import evaluate_model
        from sklearn.linear_model import LinearRegression
        import numpy as np
        
        X = np.random.randn(100, 5)
        y = np.random.randn(100)
        
        model = LinearRegression()
        model.fit(X, y)
        
        result = evaluate_model(model, X, y, "regression", "rmse")
        
        assert "metrics" in result
        assert "rmse" in result["metrics"]


class TestSmokeEndToEnd:
    """End-to-end smoke tests."""
    
    @pytest.fixture
    def breast_cancer_csv(self, tmp_path):
        """Create a CSV file from breast cancer dataset."""
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["target"] = data.target
        
        csv_path = tmp_path / "breast_cancer.csv"
        df.to_csv(csv_path, index=False)
        
        return str(csv_path)
    
    @pytest.fixture
    def diabetes_csv(self, tmp_path):
        """Create a CSV file from diabetes dataset."""
        data = load_diabetes()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df["target"] = data.target
        
        csv_path = tmp_path / "diabetes.csv"
        df.to_csv(csv_path, index=False)
        
        return str(csv_path)
    
    def test_tools_integration(self, breast_cancer_csv, tmp_path):
        """Test that all tools work together."""
        from agenticml.ml.tools.data_io import load_data
        from agenticml.ml.tools.profiling import profile_dataframe, infer_problem_type
        from agenticml.ml.tools.cleaning import apply_cleaning
        from agenticml.ml.tools.preprocessing import build_preprocess_pipeline, split_data
        from agenticml.ml.tools.modeling import get_model_candidates, create_model, train_model
        from agenticml.ml.tools.evaluation import evaluate_model
        
        # Load data
        df, _ = load_data(breast_cancer_csv)
        
        # Profile
        profile = profile_dataframe(df, "target")
        assert profile["n_rows"] > 0
        
        # Infer problem type
        problem_type, _ = infer_problem_type(df, "target")
        assert problem_type == "classification"
        
        # Clean (minimal)
        plan = {"steps": [{"action": "remove_duplicates", "params": {"keep": "first"}}]}
        df_cleaned, _ = apply_cleaning(df, plan)
        
        # Preprocess
        preprocess_plan = {
            "numeric_strategy": "standard",
            "categorical_strategy": "onehot",
            "handle_unknown": "ignore"
        }
        preprocessor, _ = build_preprocess_pipeline(df_cleaned, "target", preprocess_plan)
        
        # Split
        split_plan = {"strategy": "stratified", "test_size": 0.2, "random_state": 42}
        split_result = split_data(df_cleaned, "target", split_plan)
        
        X_train = split_result["X_train"]
        X_test = split_result["X_test"]
        y_train = split_result["y_train"]
        y_test = split_result["y_test"]
        
        # Fit preprocessor
        X_train_transformed = preprocessor.fit_transform(X_train)
        X_test_transformed = preprocessor.transform(X_test)
        
        # Get model
        candidates = get_model_candidates("classification", max_models=1)
        model = create_model(candidates[0])
        
        # Train
        trained_model, train_info = train_model(model, X_train_transformed, y_train.values)
        assert train_info["success"]
        
        # Evaluate
        eval_result = evaluate_model(trained_model, X_test_transformed, y_test.values, "classification", "f1")
        assert eval_result["metrics"]["f1"] > 0
    
    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set"
    )
    def test_full_pipeline_classification(self, breast_cancer_csv, tmp_path):
        """Test full pipeline on classification task."""
        from agenticml.ml.state import create_initial_state
        from agenticml.ml.graph import run_pipeline
        from agenticml.ml.tools.utils import generate_run_id, create_run_directory
        
        run_id = generate_run_id()
        run_dir = create_run_directory(str(tmp_path), run_id)
        
        initial_state = create_initial_state(
            run_id=run_id,
            file_path=breast_cancer_csv,
            run_dir=run_dir,
            target="target",
            problem_type="classification",
            max_iterations=2
        )
        
        final_state = run_pipeline(initial_state)
        
        # Verify artifacts exist
        assert os.path.exists(os.path.join(run_dir, "report.md"))
        assert os.path.exists(os.path.join(run_dir, "run_manifest.json"))
        assert os.path.exists(os.path.join(run_dir, "raw", "raw_data.csv"))
        assert os.path.exists(os.path.join(run_dir, "cleaned", "cleaned_data.csv"))
        
        # Verify state
        assert final_state.get("best_model") is not None
        assert final_state.get("stop_reason") is not None
    
    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set"
    )
    def test_full_pipeline_regression(self, diabetes_csv, tmp_path):
        """Test full pipeline on regression task."""
        from agenticml.ml.state import create_initial_state
        from agenticml.ml.graph import run_pipeline
        from agenticml.ml.tools.utils import generate_run_id, create_run_directory
        
        run_id = generate_run_id()
        run_dir = create_run_directory(str(tmp_path), run_id)
        
        initial_state = create_initial_state(
            run_id=run_id,
            file_path=diabetes_csv,
            run_dir=run_dir,
            target="target",
            problem_type="regression",
            max_iterations=2
        )
        
        final_state = run_pipeline(initial_state)
        
        # Verify artifacts exist
        assert os.path.exists(os.path.join(run_dir, "report.md"))
        assert final_state.get("best_model") is not None


def run_quick_smoke_test():
    """
    Run a quick smoke test without pytest.
    
    This can be run directly to verify basic functionality.
    """
    print("Running quick smoke test...")
    
    # Create test data
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df["target"] = data.target
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Save test data
        csv_path = os.path.join(tmp_dir, "test_data.csv")
        df.to_csv(csv_path, index=False)
        
        # Test data loading
        from agenticml.ml.tools.data_io import load_data
        loaded_df, metadata = load_data(csv_path)
        assert len(loaded_df) == len(df), "Data loading failed"
        print("✓ Data loading works")
        
        # Test profiling
        from agenticml.ml.tools.profiling import profile_dataframe
        profile = profile_dataframe(loaded_df, "target")
        assert profile["n_rows"] == len(df), "Profiling failed"
        print("✓ Data profiling works")
        
        # Test cleaning
        from agenticml.ml.tools.cleaning import apply_cleaning
        plan = {"steps": [{"action": "remove_duplicates", "params": {"keep": "first"}}]}
        cleaned_df, report = apply_cleaning(loaded_df, plan)
        assert len(cleaned_df) > 0, "Cleaning failed"
        print("✓ Data cleaning works")
        
        # Test preprocessing
        from agenticml.ml.tools.preprocessing import build_preprocess_pipeline, split_data
        preprocess_plan = {"numeric_strategy": "standard", "categorical_strategy": "onehot"}
        preprocessor, _ = build_preprocess_pipeline(cleaned_df, "target", preprocess_plan)
        assert preprocessor is not None, "Preprocessing failed"
        print("✓ Preprocessing works")
        
        # Test splitting
        split_plan = {"strategy": "stratified", "test_size": 0.2, "random_state": 42}
        split_result = split_data(cleaned_df, "target", split_plan)
        assert "X_train" in split_result, "Splitting failed"
        print("✓ Data splitting works")
        
        # Test modeling
        from agenticml.ml.tools.modeling import get_model_candidates, create_model, train_model
        candidates = get_model_candidates("classification", max_models=1)
        model = create_model(candidates[0])
        
        X_train = split_result["X_train"]
        y_train = split_result["y_train"]
        X_train_transformed = preprocessor.fit_transform(X_train)
        
        trained_model, info = train_model(model, X_train_transformed, y_train.values)
        assert info["success"], "Training failed"
        print("✓ Model training works")
        
        # Test evaluation
        from agenticml.ml.tools.evaluation import evaluate_model
        X_test = split_result["X_test"]
        y_test = split_result["y_test"]
        X_test_transformed = preprocessor.transform(X_test)
        
        result = evaluate_model(trained_model, X_test_transformed, y_test.values, "classification", "f1")
        assert result["metrics"]["f1"] > 0, "Evaluation failed"
        print("✓ Model evaluation works")
        
        print("\n✓ All smoke tests passed!")


if __name__ == "__main__":
    run_quick_smoke_test()
