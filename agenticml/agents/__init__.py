"""
Specialised analytical agents for the orchestrator-centered ML pipeline.

Each agent implements the BaseAgent interface: ``run(state) -> state``.
All LLM interactions return structured JSON objects -- no free-text parsing.
"""

from agenticml.agents.base_agent import BaseAgent
from agenticml.agents.dataset_profiling import DatasetProfilingAgent
from agenticml.agents.data_preprocessing import DataPreprocessingAgent
from agenticml.agents.feature_engineering import FeatureEngineeringAgent
from agenticml.agents.model_selection import ModelSelectionAgent
from agenticml.agents.model_training import ModelTrainingAgent
from agenticml.agents.evaluation import EvaluationAgent
from agenticml.agents.insight_visualization import InsightVisualizationAgent

AGENT_REGISTRY: dict[str, type[BaseAgent]] = {
    "dataset_profiling": DatasetProfilingAgent,
    "data_preprocessing": DataPreprocessingAgent,
    "feature_engineering": FeatureEngineeringAgent,
    "model_selection": ModelSelectionAgent,
    "model_training": ModelTrainingAgent,
    "evaluation": EvaluationAgent,
    "insight_visualization": InsightVisualizationAgent,
}

__all__ = [
    "BaseAgent",
    "AGENT_REGISTRY",
    "DatasetProfilingAgent",
    "DataPreprocessingAgent",
    "FeatureEngineeringAgent",
    "ModelSelectionAgent",
    "ModelTrainingAgent",
    "EvaluationAgent",
    "InsightVisualizationAgent",
]
