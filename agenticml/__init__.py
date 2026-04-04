"""
AgenticML — Orchestrator-Centered Multi-Agent ML System.

An agentic ML pipeline where a central Orchestrator dynamically routes
user requests to specialised analytical agents via LangGraph + LLMs.

Subpackages:
    ml       - Legacy tool functions and configuration
    state    - Shared WorkflowState
    agents   - Specialised analytical agents
    orchestrator - Central orchestrator and intent parsing
    graph    - LangGraph definition
    services - LLM and artifact services
"""

__version__ = "0.2.0"

from agenticml import ml  # noqa: F401 — backward compatibility
