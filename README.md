# AgenticML

Multi-agent ML pipeline — automated machine learning powered by LangGraph and LLMs.

AgenticML assembles a team of specialised AI agents (Planner, Profiler, Cleaner, Featurizer, Modeler, Evaluator, Critic, Reporter) that collaborate iteratively to build, evaluate, and refine ML models on your data.

## Installation

```bash
pip install agentic-ml
```

**Optional providers** (only install the one you use):

```bash
pip install agentic-ml[anthropic]   # Claude models
pip install agentic-ml[google]      # Gemini models
pip install agentic-ml[boost]       # XGBoost + LightGBM
pip install agentic-ml[all]         # Everything
```

## Quick Start

### Python API

```python
from agenticml import ml

# Minimal — target and problem type are auto-detected
result = ml.run("data.csv")

# Explicit options
result = ml.run(
    "data.csv",
    target="price",
    problem_type="regression",
    metric="rmse",
    model="gpt-4o",          # or "claude-3-sonnet-20240229", "gemini-pro"
    api_key="sk-...",         # optional — falls back to env var
    verbose=True,             # print LLM prompts & responses
    max_iterations=3,
)
```

### CLI

```bash
# Uses OPENAI_API_KEY from environment
agenticml --file data.csv --target price --verbose

# Specify model and key
agenticml --file data.csv --model claude-3-sonnet-20240229 --api-key sk-ant-...

# All options
agenticml --file data.csv \
    --target label \
    --problem_type classification \
    --metric f1 \
    --model gpt-4o \
    --max_iterations 3 \
    --verbose \
    --stream
```

You can also run via module:

```bash
python -m agenticml --file data.csv
```

## LLM Provider Support

The provider is **auto-detected** from the model name:

| Model prefix | Provider | Env variable | Install extra |
|---|---|---|---|
| `gpt-*`, `o1*`, `o3*` | OpenAI | `OPENAI_API_KEY` | *(included)* |
| `claude-*` | Anthropic | `ANTHROPIC_API_KEY` | `pip install agentic-ml[anthropic]` |
| `gemini-*` | Google | `GOOGLE_API_KEY` | `pip install agentic-ml[google]` |

Pass the key directly or set the environment variable:

```bash
export OPENAI_API_KEY=sk-...
```

## Pipeline Architecture

```
Planner → Profiler → Cleaner → Featurizer → Modeler → Evaluator → Critic
                                                                      ↓
                                                              (blocking issues?)
                                                              ↓ yes        ↓ no
                                                          Orchestrator   Reporter
                                                              ↓
                                                          (next iteration)
```

Each run produces:
- `report.md` — human-readable summary
- `run_manifest.json` — full reproducibility metadata
- Trained models, evaluation plots, and intermediate data in the `runs/` directory

## Verbose Mode

Use `--verbose` (CLI) or `verbose=True` (Python) to see exactly what each agent sends to the LLM and what it gets back — useful for debugging and understanding pipeline decisions.

## License

MIT
