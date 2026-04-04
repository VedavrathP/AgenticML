"""
Abstract base class for all specialised agents.

Every agent in the pipeline must subclass ``BaseAgent`` and implement
``run(state) -> state``.  The orchestrator calls agents through this
uniform interface.
"""

from abc import ABC, abstractmethod

from agenticml.state.workflow_state import WorkflowState


class BaseAgent(ABC):
    """Base class that all specialised agents must implement."""

    name: str = "base"

    @abstractmethod
    def run(self, state: WorkflowState) -> WorkflowState:
        """
        Execute the agent's logic and return the updated state.

        Each agent:
        1. Reads the fields it needs from *state*.
        2. Calls deterministic tool functions and/or the LLM (via
           ``invoke_llm_json``) to produce a structured JSON decision.
        3. Writes its results back into *state*.
        4. Returns the mutated *state*.
        """
        ...
