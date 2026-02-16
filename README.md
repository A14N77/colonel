<p align="center">
  <img src="Colonel.png" alt="Colonel Logo" width="300">
</p>

<h1 align="center">Colonel</h1>
<p align="center">
  <strong>Agentic CLI for GPU Profiling and Analysis</strong>
</p>

<p align="center">
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10%2B-blue?logo=python&logoColor=white" alt="Python 3.10+"></a>
  <a href="https://www.nvidia.com"><img src="https://img.shields.io/badge/NVIDIA-Nemotron-brightgreen?logo=nvidia&logoColor=white" alt="NVIDIA Nemotron"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-green" alt="License MIT"></a>
</p>

<p align="center">
  <a href="#getting-started">Getting Started</a> &bull;
  <a href="#workflow">Workflow</a> &bull;
  <a href="#commands">Commands</a> &bull;
  <a href="#ai-providers">AI Providers</a> &bull;
  <a href="#architecture">Architecture</a>
</p>

---

Colonel is an open-source, AI-powered command-line tool for profiling GPU kernels and applications. It wraps NVIDIA profiling tools (Nsight Systems, Nsight Compute), collects structured metrics, and uses an LLM analysis agent to identify bottlenecks and recommend optimizations -- all from a single CLI.

## Getting Started

### 1. Install Colonel

```bash
git clone https://github.com/colonel-gpu/colonel.git
cd colonel
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

### 2. Run the Setup Wizard

The setup wizard checks your environment, installs missing tools, configures
GPU profiling permissions, and sets up your AI provider.

```bash
colonel setup
```

The wizard walks through seven steps:

| Step | What it checks | Auto-fix? |
|------|---------------|-----------|
| 1. NVIDIA Driver | `nvidia-smi`, GPU model, driver version | -- |
| 2. CUDA Toolkit | `nvcc` on PATH or in `/usr/local/cuda-*` | Suggests PATH fix |
| 3. Nsight Systems | `nsys` in PATH or common install dirs | Offers `apt install` |
| 4. Nsight Compute | `ncu` in PATH or CUDA bin dirs | -- |
| 5. GPU Counters | `NVreg_RestrictProfilingToAdminUsers` setting | Live module reload or reboot |
| 6. AI Provider | API key for Anthropic, NVIDIA NIM, or HuggingFace | Interactive config |
| 7. Smoke Test | Profiles `gpu_smoke/matmul.py` with nsys | -- |

You can also run a non-interactive check anytime:

```bash
colonel setup --check
```

> **Note on GPU Counter Permissions:** Nsight Compute needs unrestricted access
> to hardware performance counters. If step 5 reports counters are restricted,
> the wizard will write a modprobe config and attempt a live nvidia module reload.
> If the reload fails (GPU in use by another process), a reboot is required.
> After reboot, run `colonel setup --check` to verify.

### 3. Verify Everything Works

```bash
colonel profile detect          # Should show: nsys, ncu
colonel run python gpu_smoke/matmul.py --name my-first-run
```

If everything is set up correctly, you'll see a kernel table, memory transfer
summary, and an AI-generated performance analysis.

## Workflow

Colonel follows a **profile → analyze → iterate** loop:

```
  colonel run ./my_app           Profile your application
        │
        ▼
  Kernel table + metrics         See GPU time, occupancy, throughput
  AI analysis + recommendations  Bottlenecks ranked by severity
  Session saved                  checkpoint.json + markdown reports
        │
        ▼
  colonel analyze --deeper       Ask follow-up questions
  colonel analyze --with-source  Include source code for specific fixes
        │
        ▼
  Make changes to your code
        │
        ▼
  colonel run ./my_app_v2        Re-profile after changes
        │
        ▼
  colonel session compare A B    Side-by-side comparison with AI
```

### Quick Example

```bash
# Profile a PyTorch training script with Nsight Systems
colonel run python train.py --name baseline --evaluator nsys

# Profile with Nsight Compute for detailed per-kernel metrics
colonel run python train.py --name baseline-ncu --evaluator ncu

# Analyze the latest run (or specify --session <id>)
colonel analyze

# Dig deeper into a specific issue
colonel analyze --deeper --question "Why is GEMM occupancy only 30%?"

# Include source code for code-specific recommendations
colonel analyze --with-source ./model.py

# After making changes, compare before and after
colonel run python train.py --name optimized
colonel session compare <baseline-id> <optimized-id> --ai
```

### Choosing a Profiler

| Profiler | Best for | Speed | Detail level |
|----------|----------|-------|-------------|
| **nsys** (Nsight Systems) | System-level overview, API overhead, memory transfers, timeline | Fast | Medium |
| **ncu** (Nsight Compute) | Per-kernel deep dive: occupancy, memory throughput, compute utilization, stalls | Slow (replays kernels) | High |

**Recommendation:** Start with `nsys` for a broad picture, then use `ncu` on
specific kernels that need investigation.

## Commands

### `colonel setup`

Interactive environment setup wizard.

```bash
colonel setup           # Full interactive wizard
colonel setup --check   # Non-interactive environment check
```

### `colonel run` (shortcut for `colonel profile run`)

Profile a GPU application and optionally run AI analysis.

```bash
colonel run <command> [args...] [OPTIONS]

Options:
  --target, -t     "local" or "ssh://user@host"    (default: local)
  --evaluator, -e  "nsys", "ncu", or "auto"        (default: auto)
  --name, -n       Human-readable label for this run
  --no-analyze     Skip AI analysis after profiling
  --cwd, -C        Working directory for execution
```

### `colonel profile`

```bash
colonel profile run <command> [OPTIONS]    # Same as colonel run
colonel profile detect [--target TARGET]   # List available profilers
```

### `colonel analyze`

Run AI-powered analysis on profiling results.

```bash
colonel analyze [OPTIONS]

Options:
  --session, -s     Session ID to analyze (default: latest)
  --deeper, -d      Follow-up analysis with more detail
  --with-source     Path to source code for context
  --question, -q    Specific question for the agent
```

### `colonel session`

Manage saved profiling sessions.

```bash
colonel session list [--limit N]
colonel session show <session-id>
colonel session compare <id-a> <id-b> [--ai]
colonel session delete <session-id> [--force]
```

### `colonel config`

View and update configuration.

```bash
colonel config show              # Show all settings
colonel config set <key> <value> # Set a config value
colonel config path              # Show config file location
```

## AI Providers

Colonel supports multiple LLM providers for the analysis agent. Configure
your preferred provider during `colonel setup` or manually:

### Anthropic Claude (default)

```bash
colonel config set llm_provider anthropic
colonel config set anthropic_api_key sk-ant-your-key
```

### NVIDIA Nemotron (via NIM API)

Uses NVIDIA's OpenAI-compatible endpoint at `build.nvidia.com`.

```bash
colonel config set llm_provider nvidia
colonel config set nvidia_api_key nvapi-your-key
```

Default model: `nvidia/llama-3.3-nemotron-super-49b-v1.5`

### HuggingFace Inference

```bash
colonel config set llm_provider huggingface
colonel config set huggingface_api_key hf_your-token
```

### Configuration Reference

| Key | Default | Description |
|-----|---------|-------------|
| `llm_provider` | `anthropic` | AI provider: `anthropic`, `nvidia`, `huggingface` |
| `anthropic_api_key` | | Anthropic API key |
| `anthropic_model` | `claude-sonnet-4-20250514` | Claude model for analysis |
| `nvidia_api_key` | | NVIDIA NIM API key |
| `nvidia_model` | `nvidia/llama-3.3-nemotron-super-49b-v1.5` | Nemotron model |
| `huggingface_api_key` | | HuggingFace API token |
| `huggingface_model` | `nvidia/llama-3.1-nemotron-70b-instruct` | HF model |
| `agent_max_tokens` | `4096` | Max tokens for agent responses |
| `default_evaluator` | `auto` | Default profiler (`nsys`/`ncu`/`auto`) |
| `default_target` | `local` | Default target |
| `sessions_dir` | `.colonel/sessions` | Session storage directory |

Configuration is loaded from (in priority order):
1. Environment variables prefixed with `COLONEL_`
2. `.colonel/config.json` in the current directory
3. `~/.colonel/config.json` in your home directory
4. Built-in defaults

## Session Storage

All profiling data is stored in `.colonel/sessions/`:

```
.colonel/sessions/
  <session-id>/
    checkpoint.json       # Full checkpoint (context + result + analysis)
    profile_summary.md    # Human-readable summary
    recommendations.md    # AI recommendations (if analyzed)
    raw_output.txt        # Raw profiler output
```

## Architecture

```
colonel/
  cli/                  # Typer CLI commands
    main.py             # Entry point
    profile_cmd.py      # colonel profile / colonel run
    analyze_cmd.py      # colonel analyze
    session_cmd.py      # colonel session
    config_cmd.py       # colonel config
    setup_cmd.py        # colonel setup
  core/                 # Core engine
    context.py          # ProfileContext (what to profile)
    result.py           # ProfileResult, Metric, KernelSummary
    executor.py         # Orchestrates targets + evaluators
    session.py          # Session manager with checkpoints
  evaluators/           # Profiling tool wrappers
    nsight_systems.py   # nsys wrapper
    nsight_compute.py   # ncu wrapper
    registry.py         # Auto-detection and selection
  targets/              # Execution backends
    local.py            # Local subprocess
    ssh.py              # Remote via SSH/paramiko
  agent/                # LLM-powered analysis
    analyzer.py         # AnalysisAgent
    prompts.py          # Prompt templates
    provider.py         # LLM providers (Anthropic, OpenAI-compatible)
  artifacts/            # Report generation (Jinja2 templates)
  config/               # Pydantic settings
  utils/                # Parsers and Rich console output
```

### Data Flow

```
colonel run ./my_kernel
    │
    ▼
ProfileContext ──▶ Executor ──▶ Target (local / SSH)
                      │                │
                      ▼                ▼
                 Evaluator ─────▶ nsys / ncu
                      │
                      ▼
                 ProfileResult
                      │
            ┌─────────┴─────────┐
            ▼                   ▼
      AnalysisAgent       SessionManager
            │                   │
            ▼                   ▼
      Recommendations    Checkpoint + reports
      to terminal        .colonel/sessions/
```

## Remote Profiling (SSH)

Colonel supports profiling on remote GPU machines via SSH:

```bash
colonel profile run ./my_kernel --target ssh://user@gpu-server
colonel profile run ./my_kernel --target ssh://user@gpu-server:2222
```

Requirements for the remote machine:
- SSH access with key-based authentication (via ssh-agent or key file)
- `nsys` or `ncu` installed and in PATH
- Your application already deployed

## Contributing

```bash
pip install -e ".[dev]"
ruff check colonel/
mypy colonel/
pytest
```

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

Colonel's architecture is inspired by research in LLM-assisted GPU optimization:

- [PEAK](https://arxiv.org/abs/2512.19018) -- Natural language transformations for GPU kernel optimization
- [KForge](https://arxiv.org/abs/2511.13274) -- Program synthesis for diverse AI hardware accelerators
- [ParaCodex](https://arxiv.org/abs/2601.04327) -- Profiling-guided autonomous parallel code generation
- [TritonForge](https://arxiv.org/abs/2512.09196) -- Profiling-guided Triton kernel optimization
