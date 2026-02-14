# Colonel Development Guide

This file provides context for AI coding agents working on the Colonel codebase.

## Project Overview

Colonel is an agentic CLI tool for GPU profiling and analysis. It wraps NVIDIA profiling tools (nsys, ncu), collects structured metrics, and uses an LLM-based analysis agent to identify bottlenecks and recommend optimizations.

## Architecture Principles

1. **PEAK-style modularity**: Core abstractions (ProfileContext, ProfileResult, BaseEvaluator, BaseTarget) are backend-agnostic. New profilers and targets are added by implementing ABCs.
2. **KForge-style analysis agent**: A dedicated agent interprets profiling data and returns structured recommendations. The agent is separate from the profiling pipeline.
3. **ParaCodex-style artifacts**: Structured markdown reports (profile_summary.md, bottlenecks.md, recommendations.md) are first-class outputs stored alongside raw data.
4. **Session-based workflows**: Every profiling run creates a checkpoint. Users can compare, revisit, and build on past runs.

## Key Data Flow

```
ProfileContext -> Executor -> Target (runs command) -> Evaluator (profiles) -> ProfileResult
ProfileResult -> AnalysisAgent -> recommendations
ProfileResult + AnalysisAgent output -> SessionManager (save checkpoint)
ProfileResult -> ReportGenerator -> markdown artifacts
```

## Package Structure

- `colonel/cli/` -- Typer CLI commands. Each file is one command group.
- `colonel/core/` -- Core data models and orchestration. ProfileContext is immutable; ProfileResult is mutable during construction.
- `colonel/evaluators/` -- Profiling tool wrappers. Each evaluator builds a command, parses output, and returns ProfileResult.
- `colonel/targets/` -- Execution backends. Local uses subprocess; SSH uses paramiko.
- `colonel/agent/` -- LLM analysis. provider.py abstracts the LLM; analyzer.py is the main agent; prompts.py has templates.
- `colonel/artifacts/` -- Jinja2-based report generation.
- `colonel/config/` -- Pydantic settings with env/file/default sources.
- `colonel/utils/` -- Parsers for profiler CSV output and Rich console formatting.

## Conventions

- Python 3.11+, type hints throughout
- Dataclasses for data models (not Pydantic models) -- keeps them lightweight
- All data models have `to_dict()` and `from_dict()` for JSON serialization
- Settings use pydantic-settings for env var / config file loading
- CLI uses Typer with Rich output
- Tests go in `tests/` directory

## Extension Points

- **New evaluator**: Create a file in `evaluators/`, implement `BaseEvaluator`, add to `registry.py`
- **New target**: Create a file in `targets/`, implement `BaseTarget`
- **New LLM provider**: Implement `BaseLLMProvider` in `agent/provider.py`
- **New report template**: Add `.md.j2` file to `artifacts/templates/`, add render function to `report.py`

## Common Tasks

- **Adding a new CLI command**: Create a file in `cli/`, register the Typer app in `cli/main.py`
- **Changing profiler command flags**: Edit `build_command()` in the evaluator
- **Changing analysis prompts**: Edit `agent/prompts.py`
- **Adding a config key**: Add a field to `ColonelSettings` in `config/settings.py`
