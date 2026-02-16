"""Prompt templates for the analysis agent.

These prompts are designed to make the LLM act as an expert GPU performance
engineer, interpreting profiling data and producing actionable recommendations.
Inspired by the KForge performance analysis agent and ParaCodex artifact style.
"""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are Colonel, an expert GPU performance engineer and profiling analyst.

Your role is to analyze GPU profiling data and provide clear, actionable
performance recommendations. You have deep expertise in:

- NVIDIA GPU architectures (memory hierarchy, SM structure, warp scheduling)
- CUDA kernel optimization (occupancy, memory coalescing, bank conflicts)
- Triton kernel development (tiling, autotuning, fused operations)
- PyTorch custom ops and torch.compile optimization
- Profiling tools (Nsight Systems, Nsight Compute)
- Common GPU performance anti-patterns and their solutions

When analyzing profiling results, you should:

1. IDENTIFY the primary performance bottleneck (memory-bound, compute-bound,
   latency-bound, or transfer-bound)
2. RANK issues by severity and potential impact
3. EXPLAIN each issue in plain language with supporting metric values
4. RECOMMEND specific, actionable fixes with expected impact
5. When kernel fusion or optimization opportunities exist, provide concrete
   CUDA or Triton code snippets showing the optimized implementation
6. FLAG any anomalies or red flags in the data

Important context for profiling data:
- 0% occupancy in nsys output is normal — nsys does not measure occupancy.
  Only ncu (Nsight Compute) provides occupancy metrics. Do NOT flag 0%
  occupancy as a bug when the evaluator is nsys.
- Wall time >> GPU time is expected for scripts that include model loading,
  data preparation, and warmup. Focus on the kernel execution time, not
  the wall-to-GPU ratio for identifying bottlenecks.
- Elementwise kernels running separately (silu + multiply, norm + scale)
  are prime candidates for Triton kernel fusion.

Format your response in clear sections with markdown headers. Be concise
but thorough. Always cite specific metric values as evidence.

Do NOT hallucinate metrics that are not in the provided data. If data is
insufficient for a conclusion, say so explicitly.
"""

ANALYSIS_PROMPT_TEMPLATE = """\
Analyze the following GPU profiling results and provide a performance assessment.

## Hardware
{hardware_section}

## Profiling Summary (evaluator: {evaluator_name})
- Wall time: {wall_time_s:.3f}s
- GPU time: {gpu_time_us:.1f} us
  (kernel: {kernel_time_us:.1f} us, transfer: {transfer_time_us:.1f} us)
- API overhead: {api_time_us:.1f} us
- Unique kernels: {kernel_count}
- Total kernel invocations: {total_invocations}

## Kernel Details
{kernel_section}

## Memory Transfers
{transfer_section}

## Additional Metrics
{metrics_section}

{source_section}

Please provide:
1. **Executive Summary** -- One paragraph overview of the performance profile
2. **Bottleneck Analysis** -- Ranked list of identified bottlenecks with severity
3. **Optimization Recommendations** -- Specific, actionable optimization suggestions.
   For each recommendation that involves kernel-level changes, include a concrete
   code snippet (Triton kernel, CUDA kernel, or PyTorch optimization) showing the
   fix. For example, if you identify unfused elementwise operations, write the
   fused Triton kernel. If you identify a GEMM configuration issue, show the
   torch.compile or cuBLAS tuning approach.
4. **Next Steps** -- What additional profiling or analysis would be helpful
"""

DEEPER_ANALYSIS_PROMPT = """\
Based on the previous analysis below, provide a deeper investigation.

## Previous Analysis
{previous_analysis}

## Original Profiling Data
{original_data}

{additional_context}

Focus on:
1. Root cause analysis of the top bottleneck
2. Quantified estimates of potential improvement (e.g. "fusing these 3 kernels
   would save ~X us per iteration, reducing kernel time by Y%")
3. Concrete code implementations for the top optimizations:
   - Write complete Triton kernels for fusion opportunities
   - Show CUDA kernel code for custom optimizations
   - Provide torch.compile / PyTorch-level fixes where applicable
4. Trade-offs between different optimization strategies
"""

COMPARISON_PROMPT_TEMPLATE = """\
Compare the following two GPU profiling runs and analyze the performance differences.

## Run A: {name_a} ({timestamp_a})
{summary_a}

## Run B: {name_b} ({timestamp_b})
{summary_b}

Please provide:
1. **Change Summary** -- What improved, what regressed, what stayed the same
2. **Key Metric Deltas** -- Specific numeric comparisons
3. **Impact Assessment** -- Overall characterization of the changes
4. **Recommendations** -- What to try next based on the comparison
"""


def format_hardware_section(hardware_dict: dict) -> str:
    """Format hardware info into a readable string."""
    parts = []
    if hardware_dict.get("gpu_name"):
        parts.append(f"- GPU: {hardware_dict['gpu_name']}")
    if hardware_dict.get("gpu_memory_mb"):
        parts.append(f"- GPU Memory: {hardware_dict['gpu_memory_mb']} MB")
    if hardware_dict.get("compute_capability"):
        parts.append(f"- Compute Capability: {hardware_dict['compute_capability']}")
    if hardware_dict.get("driver_version"):
        parts.append(f"- Driver: {hardware_dict['driver_version']}")
    if hardware_dict.get("cuda_version"):
        parts.append(f"- CUDA: {hardware_dict['cuda_version']}")
    if hardware_dict.get("cpu_name"):
        parts.append(f"- CPU: {hardware_dict['cpu_name']}")
    if hardware_dict.get("system_memory_mb"):
        parts.append(f"- System RAM: {hardware_dict['system_memory_mb']} MB")
    return "\n".join(parts) if parts else "Not available"


def format_kernel_section(kernels: list[dict]) -> str:
    """Format kernel summaries into a readable table."""
    if not kernels:
        return "No kernel data available."

    lines = ["| Kernel | Duration (us) | Invocations | Avg (us) | Occupancy |"]
    lines.append("|--------|--------------|-------------|----------|-----------|")
    for k in kernels:
        name = k.get("name", "unknown")
        # Truncate long kernel names
        if len(name) > 50:
            name = name[:47] + "..."
        lines.append(
            f"| {name} | {k.get('duration_us', 0):.1f} | "
            f"{k.get('invocations', 0)} | {k.get('avg_duration_us', 0):.1f} | "
            f"{k.get('occupancy_pct', 0):.1f}% |"
        )
    return "\n".join(lines)


def format_transfer_section(transfers: list[dict]) -> str:
    """Format memory transfers into a readable table."""
    if not transfers:
        return "No memory transfer data available."

    lines = ["| Direction | Size | Duration (us) | Throughput (GB/s) |"]
    lines.append("|-----------|------|---------------|-------------------|")
    for t in transfers:
        size_mb = t.get("size_bytes", 0) / (1024 * 1024)
        lines.append(
            f"| {t.get('direction', '?')} | {size_mb:.2f} MB | "
            f"{t.get('duration_us', 0):.1f} | {t.get('throughput_gbps', 0):.2f} |"
        )
    return "\n".join(lines)


def format_metrics_section(metrics: list[dict]) -> str:
    """Format flat metrics into a readable list."""
    if not metrics:
        return "No additional metrics."

    lines = []
    for m in metrics:
        unit = m.get("unit", "")
        value = m.get("value", 0)
        if isinstance(value, float) and value == int(value):
            value = int(value)
        lines.append(f"- {m['name']}: {value} {unit}".strip())
    return "\n".join(lines)


def build_analysis_prompt(result_dict: dict, source_code: str = "") -> str:
    """Build the full analysis prompt from a ProfileResult dict.

    Args:
        result_dict: Serialized ProfileResult.
        source_code: Optional source code to include.

    Returns:
        Formatted prompt string.
    """
    hardware = result_dict.get("hardware", {})
    kernels = result_dict.get("kernels", [])
    transfers = result_dict.get("transfers", [])
    metrics = result_dict.get("metrics", [])

    source_section = ""
    if source_code:
        source_section = f"\n## Source Code\n```\n{source_code}\n```\n"

    return ANALYSIS_PROMPT_TEMPLATE.format(
        hardware_section=format_hardware_section(hardware),
        evaluator_name=result_dict.get("evaluator_name", "unknown"),
        wall_time_s=result_dict.get("wall_time_s", 0),
        gpu_time_us=result_dict.get("gpu_time_us", 0),
        kernel_time_us=result_dict.get("kernel_time_us", 0),
        transfer_time_us=result_dict.get("transfer_time_us", 0),
        api_time_us=result_dict.get("api_time_us", 0),
        kernel_count=len(kernels),
        total_invocations=sum(k.get("invocations", 0) for k in kernels),
        kernel_section=format_kernel_section(kernels),
        transfer_section=format_transfer_section(transfers),
        metrics_section=format_metrics_section(metrics),
        source_section=source_section,
    )
