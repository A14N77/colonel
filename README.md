<p align="center">
  <img src="Colonel.png" alt="Colonel Logo" width="300">
</p>

<h1 align="center">Colonel</h1>
<p align="center">
  <strong>Agentic CLI for GPU Profiling and Analysis</strong>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11%2B-blue?logo=python&logoColor=white" alt="Python 3.11+"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License MIT"></a>
</p>

<p align="center">
  <a href="#installation">Installation</a> &bull;
  <a href="#quick-start">Quick Start</a> &bull;
  <a href="#commands">Commands</a> &bull;
  <a href="#architecture">Architecture</a> &bull;
  <a href="#contributing">Contributing</a>
</p>

---

Colonel is an open-source, AI-powered command-line tool for profiling GPU kernels and applications. It wraps NVIDIA profiling tools (Nsight Systems, Nsight Compute), collects structured metrics, and uses an LLM-based analysis agent to identify bottlenecks and recommend optimizations -- all from a single CLI.

**Key features:**

- **Profile** GPU applications locally or on remote machines via SSH
- **Analyze** profiling results with an AI agent (Anthropic Claude) that identifies bottlenecks and suggests optimizations
- **Compare** runs side-by-side to track performance changes
- **Session management** with checkpoints, so you never lose a profiling run
- **Artifact reports** -- structured markdown reports (summary, bottlenecks, recommendations) saved alongside raw data
- **Extensible** -- pluggable evaluators, targets, and LLM providers

## Installation

Requires Python 3.11+ and NVIDIA profiling tools (`nsys` and/or `ncu`) installed on the target machine.

```bash
pip install -e .
```

### Set up your API key

Colonel uses Anthropic Claude for AI-powered analysis. Set your API key:

```bash
colonel config set anthropic_api_key sk-ant-your-key-here
```

Or via environment variable:

```bash
export COLONEL_ANTHROPIC_API_KEY=sk-ant-your-key-here
```

### Verify installation

```bash
colonel --version
colonel profile detect
```

## Quick Start

### 1. Profile a GPU application

```bash
colonel profile run ./my_cuda_kernel --name baseline
```

This will:
1. Auto-detect the best profiler (nsys or ncu)
2. Run your application under the profiler
3. Parse and display metrics in a rich table
4. Run AI analysis on the results
5. Save everything as a session checkpoint

### 2. Analyze results with AI

```bash
# Analyze the most recent run
colonel analyze

# Deeper follow-up analysis
colonel analyze --deeper --question "Why is memory throughput so low?"

# Include source code for more specific recommendations
colonel analyze --with-source ./kernel.cu
```

### 3. Compare runs

```bash
# Profile after a change
colonel profile run ./my_cuda_kernel_v2 --name optimized

# Compare the two runs
colonel session compare <baseline-id> <optimized-id> --ai
```

### 4. Browse session history

```bash
colonel session list
colonel session show <session-id>
```

## Commands

### `colonel profile`

```
colonel profile run <command> [OPTIONS]

Options:
  --target, -t     Target: "local" or "ssh://user@host" (default: local)
  --evaluator, -e  Profiler: "nsys", "ncu", or "auto" (default: auto)
  --name, -n       Human-readable label for this run
  --no-analyze     Skip AI analysis after profiling
  --cwd, -C        Working directory for execution
```

```
colonel profile detect [--target TARGET]
```

### `colonel analyze`

```
colonel analyze [OPTIONS]

Options:
  --session, -s     Session ID to analyze (default: latest)
  --deeper, -d      Perform deeper follow-up analysis
  --with-source     Path to source code for context
  --question, -q    Specific question for the agent
```

### `colonel session`

```
colonel session list [--limit N]
colonel session show <session-id>
colonel session compare <id-a> <id-b> [--ai]
colonel session delete <session-id> [--force]
```

### `colonel config`

```
colonel config set <key> <value>
colonel config show
colonel config path
```

## Architecture

Colonel's architecture is modeled after research in LLM-assisted GPU optimization:

- **PEAK** (Microsoft Research) -- modular kernel context + evaluators + performance workflows
- **KForge** (Gimlet Labs / Stanford) -- dedicated performance analysis agent that interprets profiling data
- **ParaCodex** (Technion / Stanford) -- artifact-driven reasoning with structured reports

```
colonel/
  cli/                  # Typer CLI commands
    main.py             # Entry point
    profile_cmd.py      # `colonel profile`
    analyze_cmd.py      # `colonel analyze`
    session_cmd.py      # `colonel session`
    config_cmd.py       # `colonel config`
  core/                 # Core engine
    context.py          # ProfileContext (what to profile)
    result.py           # ProfileResult, Metric, KernelSummary, etc.
    executor.py         # Orchestrates targets + evaluators
    session.py          # Session manager with checkpoints
  evaluators/           # Profiling tool wrappers
    base.py             # BaseEvaluator ABC
    nsight_systems.py   # nsys wrapper
    nsight_compute.py   # ncu wrapper
    registry.py         # Auto-detection and selection
  targets/              # Execution backends
    base.py             # BaseTarget ABC
    local.py            # Local subprocess
    ssh.py              # Remote via SSH/paramiko
  agent/                # LLM-powered analysis
    analyzer.py         # AnalysisAgent
    prompts.py          # Prompt templates
    provider.py         # LLM provider interface (Anthropic)
  artifacts/            # Report generation
    report.py           # Jinja2-based markdown reports
    templates/          # Report templates
  config/               # Configuration
    settings.py         # Pydantic settings
  utils/                # Utilities
    parsers.py          # CSV/profiler output parsers
    rich_output.py      # Rich console formatting
```

### Data Flow

```
User runs: colonel profile run ./my_kernel
    |
    v
ProfileContext ----> Executor ----> Target (local/SSH)
                        |                  |
                        v                  v
                   Evaluator ---------> nsys/ncu
                        |
                        v
                   ProfileResult
                        |
              +---------+---------+
              |                   |
              v                   v
         AnalysisAgent      SessionManager
              |                   |
              v                   v
         Recommendations    Checkpoint saved
              |                   |
              v                   v
         Rich output        Artifact reports
         to terminal        (.colonel/sessions/)
```

## Configuration

Colonel looks for configuration in this order:

1. Environment variables prefixed with `COLONEL_`
2. `.colonel/config.json` in the current directory
3. `~/.colonel/config.json` in your home directory
4. Built-in defaults

| Key | Default | Description |
|-----|---------|-------------|
| `anthropic_api_key` | (empty) | Anthropic API key |
| `anthropic_model` | `claude-sonnet-4-20250514` | Claude model for analysis |
| `agent_max_tokens` | `4096` | Max tokens for agent responses |
| `default_evaluator` | `auto` | Default profiler (nsys/ncu/auto) |
| `default_target` | `local` | Default target |
| `nsys_path` | `nsys` | Path to nsys binary |
| `ncu_path` | `ncu` | Path to ncu binary |
| `sessions_dir` | `.colonel/sessions` | Session storage directory |

## Remote Profiling

Colonel supports profiling on remote GPU machines via SSH:

```bash
# Profile on a remote server
colonel profile run ./my_kernel --target ssh://user@gpu-server

# With a non-standard port
colonel profile run ./my_kernel --target ssh://user@gpu-server:2222
```

Requirements for the remote machine:
- SSH access with key-based authentication
- `nsys` or `ncu` installed and in PATH
- Your application already deployed

## Session Storage

All profiling data is stored in `.colonel/sessions/`:

```
.colonel/sessions/
  <session-id>/
    checkpoint.json       # Full checkpoint (context + result + analysis)
    profile_summary.md    # Human-readable summary
    bottlenecks.md        # Bottleneck analysis (if available)
    recommendations.md    # AI recommendations (if analyzed)
    raw_output.txt        # Raw profiler output
```

## Contributing

Contributions are welcome! Colonel is designed to be extensible:

- **New evaluators** -- Add support for AMD ROCm (`rocprof`), Apple Metal, etc. by implementing `BaseEvaluator`
- **New targets** -- Add cloud targets (AWS, GCP) by implementing `BaseTarget`
- **New LLM providers** -- Add OpenAI, local models, etc. by implementing `BaseLLMProvider`
- **Better prompts** -- Improve the analysis agent's prompts in `agent/prompts.py`

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run lints
ruff check colonel/
mypy colonel/

# Run tests
pytest
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

Colonel's architecture is inspired by recent research in LLM-assisted GPU optimization:

- [PEAK](https://arxiv.org/abs/2512.19018) -- Natural language transformations for GPU kernel optimization
- [KForge](https://arxiv.org/abs/2511.13274) -- Program synthesis for diverse AI hardware accelerators
- [ParaCodex](https://arxiv.org/abs/2601.04327) -- Profiling-guided autonomous parallel code generation
- [TritonForge](https://arxiv.org/abs/2512.09196) -- Profiling-guided Triton kernel optimization
