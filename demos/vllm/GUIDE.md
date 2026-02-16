# LLM Inference Profiling Demo — Colonel

Profile real LLM inference workloads on your A10G GPU with Colonel.

## Quick Start (5 minutes)

```bash
# 1. Install dependencies (in your colonel venv)
cd ~/colonel
source .venv/bin/activate
pip install transformers accelerate

# 2. Run your first profile (downloads TinyLlama on first run, ~2GB)
colonel run --name llm-first --evaluator nsys --no-analyze -- \
    python demos/vllm/01_single_inference.py
```

That's it. You should see kernel timings (cutlass GEMMs, flash attention, elementwise ops), memory transfers, and a session ID.

---

## Setup

### Prerequisites
- Colonel installed and `colonel setup` passing
- NVIDIA A10G (24GB) — all demos sized for this GPU
- PyTorch with CUDA already installed

### Install Dependencies
```bash
source .venv/bin/activate
pip install transformers accelerate
```

### Verify
```bash
python -c "from transformers import AutoModelForCausalLM; print('OK')"
```

### Models
The demos default to **TinyLlama-1.1B** (~3GB VRAM, downloads ~2GB on first run). HuggingFace downloads are cached in `~/.cache/huggingface/`.

For gated models (Llama-2), you need:
```bash
pip install huggingface_hub
huggingface-cli login
```

---

## CLI Syntax

Colonel options go BEFORE `--`, script args go AFTER:

```bash
colonel run --name <label> --evaluator <nsys|ncu> [--no-analyze] -- \
    python demos/vllm/<script>.py [--script-args]
```

---

## Demo Scenarios

### Demo 1: Single Inference (`01_single_inference.py`)

**What it does**: Loads TinyLlama, warms up, then generates 64 tokens from one prompt.

**What you learn**: The shape of a single LLM inference — prefill (one big compute pass) followed by decode (many small memory-bound passes).

```bash
# Profile with nsys (fast, system-level view)
colonel run --name llm-single-nsys --evaluator nsys -- \
    python demos/vllm/01_single_inference.py

# Profile with ncu (slow but detailed per-kernel metrics)
colonel run --name llm-single-ncu --evaluator ncu -- \
    python demos/vllm/04_ncu_deep_dive.py
```

**What Colonel shows**:
- `cutlass::Kernel2` and `ampere_fp16_s16816gemm` — the GEMM kernels that dominate compute (~70% of kernel time)
- `flash_fwd_kernel` — flash attention implementation (~4% of kernel time)
- `elementwise_kernel` — RMSNorm, SiLU activations, residual adds
- `reduce_kernel` — softmax and layer norm reductions
- Memory transfers: model weight loading (HtoD)

### Demo 2: Batch Size Sweep (`02_batch_sweep.py`)

**What it does**: Profiles the same model with different batch sizes. Each run is a separate Colonel session so you can compare.

```bash
# Run each batch size
colonel run --name llm-b1 --evaluator nsys --no-analyze -- \
    python demos/vllm/02_batch_sweep.py --batch-size 1

colonel run --name llm-b8 --evaluator nsys --no-analyze -- \
    python demos/vllm/02_batch_sweep.py --batch-size 8

colonel run --name llm-b16 --evaluator nsys --no-analyze -- \
    python demos/vllm/02_batch_sweep.py --batch-size 16

# Compare the runs
colonel session list
colonel session compare <b1-session-id> <b16-session-id> --ai
```

**What Colonel shows (real results from A10G)**:
- **batch=1**: Kernel time ~14,700 us, 88 GEMM invocations, small tile sizes (128x64)
- **batch=8**: Kernel time ~160,800 us (10.9x), 792 GEMM invocations, larger tiles (256x128 appear), attention switches from flash to FMHA cutlass
- Bigger batches = wider GEMM tiles = better GPU utilization
- Kernel launch overhead is amortized with larger batches

### Demo 3: Model Size Comparison (`03_model_compare.py`)

**What it does**: Profile different model sizes to see how kernel characteristics change.

```bash
# TinyLlama 1.1B (~3GB VRAM, fast)
colonel run --name llm-tinyllama --evaluator nsys -- \
    python demos/vllm/03_model_compare.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Mistral 7B (~14GB VRAM, open model, no login needed)
colonel run --name llm-mistral7b --evaluator nsys -- \
    python demos/vllm/03_model_compare.py --model mistralai/Mistral-7B-v0.1

# Compare
colonel session compare <tinyllama-id> <mistral-id> --ai
```

**What Colonel shows**:
- More layers = more kernel launches per decode step (22 for TinyLlama vs 32 for Mistral)
- Larger hidden dim = wider GEMM tiles, better GPU utilization
- 7B models saturate DRAM bandwidth during decode (~80-90% of peak)
- TinyLlama leaves GPU underutilized — room for larger batches

### Demo 4: NCU Deep-Dive (`04_ncu_deep_dive.py`)

**What it does**: Runs Nsight Compute with `--set full` on a minimal workload (8 tokens). This is the slowest but most detailed profile.

```bash
colonel run --name llm-ncu --evaluator ncu -- \
    python demos/vllm/04_ncu_deep_dive.py
```

**What Colonel shows**:
- Per-kernel occupancy breakdown
- Memory throughput vs compute throughput per kernel
- Register pressure (registers per thread)
- Shared memory usage
- L2 cache hit rates (attention kernels hit L2 heavily)
- GEMM vs attention: compute-bound vs memory-bound classification

---

## Complete Walkthrough

Run this sequence for the full demo experience:

```bash
cd ~/colonel
source .venv/bin/activate

# Step 1: Quick nsys profile to verify everything works
colonel run --name llm-warmup --evaluator nsys --no-analyze -- \
    python demos/vllm/01_single_inference.py --max-tokens 16

# Step 2: Full nsys profile with AI analysis
colonel run --name llm-single --evaluator nsys -- \
    python demos/vllm/01_single_inference.py

# Step 3: Batch sweep (the money demo — shows scaling)
colonel run --name llm-b1 --evaluator nsys --no-analyze -- \
    python demos/vllm/02_batch_sweep.py --batch-size 1
colonel run --name llm-b8 --evaluator nsys --no-analyze -- \
    python demos/vllm/02_batch_sweep.py --batch-size 8

# Step 4: Compare batch=1 vs batch=8
colonel session list
colonel session compare <b1-id> <b8-id> --ai

# Step 5: NCU deep-dive (takes a few minutes)
colonel run --name llm-ncu --evaluator ncu -- \
    python demos/vllm/04_ncu_deep_dive.py

# Step 6: Follow-up AI analysis
colonel analyze --session <ncu-session-id> --deeper \
    --question "Which kernels are memory-bound vs compute-bound?"
```

---

## Troubleshooting

### "OutOfMemoryError" or "CUDA out of memory"
Use a smaller model or reduce batch size. TinyLlama-1.1B uses only ~3GB.

### "model not found" or 403 from HuggingFace
For gated models (Llama-2), you need `huggingface-cli login` and approval. Use ungated models: `TinyLlama/TinyLlama-1.1B-Chat-v1.0` or `mistralai/Mistral-7B-v0.1`.

### NCU profiling hangs or is extremely slow
NCU replays each kernel multiple times. Use `04_ncu_deep_dive.py` which defaults to only 8 tokens.

### "No kernel data collected"
Make sure you're using the single-process demo scripts (not vLLM directly). The HuggingFace Transformers scripts run everything in one process so nsys/ncu can capture all kernels.

### Model download is slow
First run downloads the model weights. TinyLlama is ~2GB. Use `DEMO_MODEL` env var:
```bash
export DEMO_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

---

## Models That Fit on A10G (24GB)

| Model | Params | VRAM (FP16) | Download | Notes |
|-------|--------|-------------|----------|-------|
| TinyLlama-1.1B | 1.1B | ~3GB | ~2GB | Fast iteration, default |
| Qwen/Qwen2-1.5B | 1.5B | ~4GB | ~3GB | Open, no login needed |
| microsoft/phi-2 | 2.7B | ~6GB | ~5GB | Good mid-range |
| mistralai/Mistral-7B-v0.1 | 7B | ~14GB | ~13GB | Open, no login needed |
| meta-llama/Llama-2-7b-hf | 7B | ~14GB | ~13GB | Standard benchmark (gated) |

---

## Script Options

All scripts support these common flags:

| Flag | Default | Description |
|------|---------|-------------|
| `--model` | TinyLlama-1.1B | HuggingFace model ID |
| `--max-tokens` | varies | Max new tokens to generate |
| `--dtype` | float16 | Model precision (float16/bfloat16/float32) |
| `--batch-size` | 1 | Batch size (02, 03 scripts) |

---

## What Colonel Reveals About LLM Inference

After profiling, Colonel typically identifies:

1. **GEMM dominance** — cutlass/cuBLAS GEMMs (matrix multiply) account for 60-80% of kernel time. These are the linear layers (QKV projection, FFN up/down/gate).

2. **Attention scaling** — Flash attention kernels are fast for short sequences. At batch=8+, the attention implementation may switch (flash -> FMHA cutlass) for better throughput.

3. **Memory bandwidth during decode** — Single-token decode is fundamentally memory-bound (reading model weights for each token). Colonel shows high DRAM throughput but low compute utilization.

4. **Batch scaling** — batch=1 has 88 GEMM invocations; batch=8 has 792 (9x). But kernel times scale sub-linearly because wider tiles improve GPU utilization.

5. **Elementwise overhead** — RMSNorm, SiLU, residual connections collectively add up. They're individually fast (~2-3 us) but there are hundreds of them.
