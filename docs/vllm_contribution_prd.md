# PRD: vLLM Contribution via Colonel

**Status:** Draft v1
**Author:** Alan (+ Claude Code assist)
**Last updated:** 2026-04-21
**Scope owner:** Colonel maintainers
**Related artifacts:**
- Shadeform A4000 spike reports: `~/smoke/reports/{vllm_decode.nsys-rep, ncu_decode.ncu-rep, ncu_graph_smoke.ncu-rep}`
- Spike scripts: `~/smoke/{inproc_decode_rpc.py, graph_smoke.py, bench_decode.py}`
- Running progress log: `~/smoke/PROGRESS.md`

---

## 0. TL;DR

Colonel today ships a `demos/vllm/` directory that does not actually use
vLLM — every script there runs HuggingFace Transformers and says, in
effect, "trust us, it's the same kernels." That is a credibility and
scope problem: Colonel's strongest differentiator should be _agentic
profiling of the real serving stack_, not a proxy.

This PRD proposes a two-track contribution strategy:

- **Track A — Colonel learns to profile real vLLM.** Turn the Stage 0
  spike discoveries (V1 subprocess split, `collective_rpc` gating, NVTX
  flag, ncu replay mode) into a first-class, tested, documented vLLM
  adapter inside Colonel. Replace the HF-impersonator demos with real
  ones.
- **Track B — Use Colonel to find and fix a real vLLM upstream bug.**
  Once Track A lands, run Colonel against real vLLM on non-Hopper
  hardware, surface the `flash_fwd_splitkv_kernel` grid-utilization
  bottleneck (already isolated in the spike: 28 blocks on 48 SMs ≈ 0.58
  waves, scales _worse_ on H100), and land the fix upstream in
  `vllm-project/flash-attention`.

Track A is a Colonel feature. Track B is the hero demo for Track A, and
a real upstream contribution to vLLM. They should ship in that order.

---

## 1. Background

### 1.1 What Colonel is today

Colonel is an agentic CLI that wraps Nsight Systems (`nsys`) and Nsight
Compute (`ncu`), parses their output into a structured `ProfileResult`,
and feeds that result to an LLM analysis agent (Anthropic / NVIDIA NIM /
HF Inference) that returns bottleneck diagnosis and optimization
recommendations. It owns session state, SSH execution, and markdown
artifacts. Its architecture (`ProfileContext` → `Executor` → `Target` →
`Evaluator` → `ProfileResult` → `AnalysisAgent`) is deliberately
backend-agnostic.

Colonel's thesis — articulated in the README's "Acknowledgments" list
(PEAK, KForge, ParaCodex, TritonForge) — is that LLM-assisted kernel
profiling is a distinct, worthwhile tool category. The current
implementation delivers _general_ profiling with _general_ analysis
prompts.

### 1.2 What vLLM is

vLLM is the reference high-throughput LLM serving engine (PagedAttention,
continuous batching, CUDA graphs, FA2/FA3 backends, SM-specific kernels).
It is the de facto target for serving-side GPU optimization work. Any
"LLM-inference profiling" story that cannot point at real vLLM profiles
has a credibility gap.

### 1.3 The gap

Colonel's `demos/vllm/` directory does not contain a single vLLM import.
Every "vLLM demo" runs `transformers.AutoModelForCausalLM.generate()` in
a single process. The demo `GUIDE.md` explicitly instructs users:

> "Make sure you're using the single-process demo scripts (not vLLM
> directly). The HuggingFace Transformers scripts run everything in one
> process so nsys/ncu can capture all kernels."

This is an honest workaround, not a feature. It exists because real vLLM
V1 splits GPU execution into an `EngineCore` subprocess, and the default
`cudaProfilerStart/Stop` + `nsys launch` flow does not cross that
boundary cleanly. The Stage 0 spike (2026-04-21, on Shadeform A4000)
produced a working recipe; that recipe is not yet embedded in Colonel.

### 1.4 Why now

Three forces converge:

1. **Colonel is Alpha** (`Development Status :: 3 - Alpha` in
   `pyproject.toml`). Before the tool calcifies around the HF-demo
   workaround, it is cheap to fix the vLLM path properly.
2. **The spike retired all the technical risk.** Every unknown that
   could have killed this effort (can ncu profile CUDA-graph-captured
   decode? can `collective_rpc` gate profile ranges in the V1
   subprocess? does NVTX survive?) was answered on a real A4000 with
   real kernels in the reports listed above. The work is now plumbing,
   not research.
3. **vLLM's non-Hopper code path is under-attended.** FA3 requires SM90;
   FA4 requires SM100. Ampere (A100, A10G, A40, A4000, L4) falls back to
   FA2, and FA2's `num_splits > 1` is not exposed to Python at all —
   vLLM's `max_num_splits` plumbing is _unreachable_ on Ampere. The
   kernel-level data from the spike (see §4) shows `flash_fwd_splitkv`
   at 13.5% SM / 2.4% DRAM with a 28-block grid on 48 SMs. That is a
   real, measurable, upstream-fixable bottleneck on a very common class
   of production GPUs.

---

## 2. Problem Statement

**For users of Colonel who actually serve LLMs in production:** Colonel
cannot today profile the real serving path. Running `colonel run python
my_vllm_script.py` against a V1-mode vLLM job captures either no kernels
(because profile start/stop happens in the wrong process), all kernels
but no NVTX annotation (because `enable_layerwise_nvtx_tracing` defaults
to `False`), or a partial/slow capture (because ncu's default per-kernel
replay mode GPU-mem-save-restores between every launch). The workaround
is to profile HuggingFace Transformers and hope the kernel shapes
transfer.

**For the GPU-optimization community:** Ampere-class deployments of vLLM
(A100, A10G, L4, A40, RTX A-series) have a latent `flash_fwd_splitkv`
grid-size problem that is not visible from the Python API and has no
operator-surfaced knob. The FA2 interface inside vLLM literally raises
`NotImplementedError("FA2 does not support num_splits > 1")`. The
internal C++ heuristic in `vllm-project/flash-attention` picks a split
count that under-utilizes the SMs at small batch, and the problem grows
worse on larger-SM-count GPUs (H100: 132 SMs, H200: 132 SMs, B200: 208
SMs) — i.e. the problem scales with the hardware that matters.

**The meta-problem:** Colonel's README aspires to the PEAK/KForge/
ParaCodex/TritonForge lineage — profiling-guided, agentic kernel
optimization. But without a credible "real production workload" target,
Colonel's recommendations are untestable against a real contribution
loop. Track B closes that loop.

---

## 3. Goals and Non-Goals

### 3.1 Goals

**G1 (Track A):** `colonel run --flavor vllm -- python my_vllm_script.py`
produces a valid `ProfileResult` with populated kernel metrics on a
real vLLM V1 job, on Ampere or newer, under both `nsys` and `ncu`,
without the user writing any extra profiling glue.

**G2 (Track A):** A small, documented helper — `from colonel.profiling.vllm
import profile_region` — lets users bracket the decode loop they care
about, handling the `collective_rpc(cudaProfilerStart/Stop)` dance
transparently.

**G3 (Track A):** At least two of the four `demos/vllm/` scripts are
replaced with real vLLM workloads that run end-to-end under `colonel
run` on an A10G-class GPU.

**G4 (Track A):** `colonel setup` gains a vLLM-specific environment
check (verifies `VLLM_ALLOW_INSECURE_SERIALIZATION` guidance,
`RmProfilingAdminOnly` state, and vLLM import-ability).

**G5 (Track B):** An upstream PR against `vllm-project/flash-attention`
that improves the FA2 `get_num_splits_heuristic` for small-batch decode
on high-SM-count GPUs, with before/after ncu evidence captured _by
Colonel itself_ as the proof artifact.

**G6 (Track B):** A blog-post-grade writeup, co-located under
`docs/case_studies/flash_attn_splitkv.md`, demonstrating Colonel's
profile → analyze → fix → re-profile loop end-to-end on real vLLM.

### 3.2 Non-Goals

**N1:** Not contributing new attention kernels (Triton-level or CUDA-
level). This PRD is about identifying and fixing an existing-kernel
_launch_ problem, not authoring new kernels. Kernel authorship is a
Phase 3+ scope question.

**N2:** Not supporting vLLM V0 (the pre-V1 engine). V0 is deprecated
upstream; any profiling support we ship should target V1.

**N3:** Not shipping a full "LLM benchmark suite" (tokens/sec
leaderboards, cost-per-token dashboards). Colonel is a profiler, not a
benchmarking harness. If a user wants sustained throughput numbers,
`vllm bench` already exists.

**N4:** Not shipping an auto-patcher that rewrites user kernels. That is
the PEAK / KForge / TritonForge horizon; we cite it but do not attempt
it here. Colonel's current contract is analysis + recommendation, not
code generation.

**N5:** Not shipping Windows support for the vLLM adapter in this PRD.
Real vLLM does not run on Windows; the adapter follows.

### 3.3 Constraints

- **Upstream-friendliness.** All changes inside Colonel must work
  against _unmodified_ vLLM installs (pip-installed or source-built).
  We do not fork vLLM, patch it in-place, or require editable installs.
- **Backend-agnostic core.** The vLLM adapter must be implementable _on
  top of_ the existing `ProfileContext` / `Evaluator` abstractions. If
  it cannot, that is a signal the core needs a small, deliberate
  extension — not a signal to jam vLLM-specific knowledge into core.
- **No new heavy dependencies.** No `vllm` pin in `pyproject.toml`
  proper. The adapter imports `vllm` lazily, and `pip install colonel`
  continues to work without it.

---

## 4. Evidence: What the Spike Proved

These are the _verified_ outputs of the Stage 0 spike on Shadeform's
A4000 (48 SMs, GA104, CUDA 12.8, driver 570.195.03). They are the
empirical backbone for every design choice in §5 and §6.

### 4.1 The three gating questions

| # | Question | Answer | How verified |
|---|---|---|---|
| Q1 | Can ncu profile CUDA-graph-captured decode? | **Yes** | `graph_smoke.py` → 20/20 kernels × 7 replay passes; then end-to-end against real vLLM (Qwen2.5-0.5B) producing `ncu_decode.ncu-rep`. |
| Q2 | Do vLLM NVTX ranges survive into nsys? | **Not by default** — `ObservabilityConfig.enable_layerwise_nvtx_tracing` defaults to `False`; `nsys stats --report nvtx_sum` returned zero ranges. | Inspection of `vllm_decode.nsys-rep`. |
| Q3 | Where does Colonel become useful vs misleading on vLLM? | Three plumbing gotchas, not capability gaps. | See §4.3. |

### 4.2 Kernel-level SOL (re-extracted from `ncu_decode.ncu-rep` this session)

Qwen/Qwen2.5-0.5B, V1, decode, batch=4, max_tokens=32, A4000.

| Kernel                              | n | mean dur (µs) | SM %  | DRAM % | share |
|-------------------------------------|---|---------------|-------|--------|-------|
| `ampere_s16816gemm_bf16_64x128`     | 4 |          75.8 |  43.3 |   44.8 | 61.9% |
| `flash_fwd_splitkv_kernel<...>`     | 4 |          26.2 |  13.5 |    2.4 | 21.4% |
| `reshape_and_cache_flash_kernel`    | 4 |           8.6 |   0.5 |    1.1 |  7.0% |
| `triton_per_fused_rms_0`            | 4 |           5.7 |   0.4 |    2.2 |  4.6% |
| `triton_per_fused_rms_2`            | 3 |           6.0 |   0.5 |    2.9 |  3.7% |
| `triton_per_fused_rms_embed`        | 1 |           6.9 |   0.3 |    1.0 |  1.4% |

Wave occupancy on 48 SMs:

| Kernel                  | grid        | block | blocks | waves |
|-------------------------|-------------|-------|--------|-------|
| `flash_fwd_splitkv`     | (1, 2, 14)  | 128   |  28    | **0.58** |
| `ampere_gemm_64x128`    | (14, 1, 4)  | 128   |  56    | 1.17  |
| `reshape_and_cache`     | (8, 1, 1)   | 128   |   8    | 0.17  |
| `triton_rms_0`          | (8, 1, 1)   | 32    |   8    | 0.17  |

### 4.3 The three plumbing gotchas (design constraints for Track A)

**Gotcha 1 — V1 runs GPU in a subprocess.** `cudaProfilerStart/Stop` must
be routed through `llm.collective_rpc(fn)` where `fn(worker)` runs in
the `EngineCore` worker. Additionally, `VLLM_ALLOW_INSECURE_SERIALIZATION=1`
is required for arbitrary callables to pass through the RPC boundary.

**Gotcha 2 — ncu default replay is slow.** ncu's default
`--replay-mode kernel` does GPU-memory save/restore between each launch.
For small models this works; for production model sizes (7B+) it
massively distorts timing and sometimes stalls. `--replay-mode application`
replays the full process instead — much better signal on big models.

**Gotcha 3 — managed GPU clouds ship `RmProfilingAdminOnly=1`.** On
Shadeform (and likely most managed GPU providers), ncu requires sudo
until the nvidia driver is reloaded with
`NVreg_RestrictProfilingToAdminUsers=0`. Module reload may require a
reboot if `nvidia-persistenced` is active. Colonel's `setup` command
already detects this, but not in a vLLM-aware context.

### 4.4 Why `flash_fwd_splitkv` is the Track B target

**It is fixable at the right altitude.** The kernel itself is fine — it
is the _grid_ that's wrong. FA2's internal `get_num_splits_heuristic`
picks a split count that under-utilizes SMs at small batch. That is a
10–50-line heuristic change upstream in
`vllm-project/flash-attention`, not a kernel rewrite.

**It scales the wrong way with hardware.** 28 blocks on A4000's 48 SMs =
0.58 waves (SMs idle 42% of the kernel's lifetime). Same grid on H100
(132 SMs) = 28 / 132 = 0.21 waves (SMs idle _79%_). On B200 (208 SMs) =
0.13 waves. The problem compounds on exactly the hardware users care
about.

**SOL signature is unambiguous.** 13.5% SM / 2.4% DRAM is the
textbook fingerprint of grid-under-utilization: neither compute nor
memory is pressured; we are paying latency for unused parallel capacity.

**It is visible to Colonel's existing analysis.** The occupancy %,
memory %, compute % that `nsight_compute.py` already extracts are
exactly the signals an LLM agent needs to flag this kernel. No new
parsing required. This means Track B doubles as an _integration test_
for Colonel's existing analyzer.

**Competing targets and why they lose:**
- `ampere_gemm_64x128`: 62% of time but already at 43/45 SOL with ~1.2
  waves. It is close to doing its job; further wins need cuBLAS-level
  surgery, which is not an upstream-PR-size contribution.
- Triton-fused RMSNorm kernels (`triton_rms_*`): <1% SOL each but only
  8.3% of total time combined. The fix is torch.compile / Inductor
  fusion heuristics, not vLLM.
- `reshape_and_cache_flash`: 7% of time, 0.17 waves, launch-bound.
  Candidate for _fusion_ into the attention prologue. A fine Phase 3
  target, but (a) fusion is a new-kernel contribution (violates N1) and
  (b) the payoff is smaller than splitkv.

---

## 5. Design — Track A (Colonel ↔ vLLM)

### 5.1 Module layout

```
colonel/
  profiling/
    __init__.py
    vllm/
      __init__.py          # re-exports profile_region, install_hooks
      adapter.py           # vLLM-aware ProfileContext transform
      region.py            # profile_region() user-facing helper
      env.py               # env-var injection (VLLM_ALLOW_INSECURE_SERIALIZATION, etc.)
      nvtx.py              # enable layerwise NVTX via ObservabilityConfig
      rpc.py               # collective_rpc(start/stop) helpers
  evaluators/
    nsight_compute.py      # grows --replay-mode flag plumbing (small, non-breaking)
    nsight_systems.py      # grows --capture-range cudaProfilerApi awareness
  cli/
    profile_cmd.py         # --flavor {generic,vllm,...} option
```

A new top-level package `colonel.profiling.vllm` rather than mixing
vLLM-specific code into existing evaluators. Rationale: evaluators are
about _profilers_ (nsys, ncu); flavors are about _workloads_ (generic,
vLLM, later TRT-LLM, SGLang, …). Crossing those axes as two sub-packages
is cleaner than a combinatorial matrix of `vllm_ncu.py`, `vllm_nsys.py`.

### 5.2 `--flavor vllm` at the CLI layer

```bash
colonel run --flavor vllm --evaluator ncu -- python my_vllm_script.py
```

Concretely, `--flavor vllm` causes the CLI layer to:

1. **Inject env vars** into `ProfileContext.env`:
   - `VLLM_ALLOW_INSECURE_SERIALIZATION=1`
   - `VLLM_USE_V1=1` (explicit, to pin V1 semantics)
   - `VLLM_LOGGING_LEVEL=INFO` (so the engine's startup lines end up in session logs)
2. **Select ncu defaults** appropriate for V1:
   - `--replay-mode application`
   - `--target-processes all` (already default; kept for clarity)
   - `--import-source yes` (if sources are available, for later `--with-source` analysis)
3. **Select nsys defaults** appropriate for V1:
   - `--capture-range cudaProfilerApi --capture-range-end stop`
   - `--trace cuda,nvtx,osrt`
4. **Emit a preflight check** before launching: warn if the user's
   script does not import `colonel.profiling.vllm.profile_region`, since
   without the RPC gate ncu may capture zero kernels.

None of the above changes the CLI _schema_ observable to existing users;
`--flavor generic` (default) preserves today's behavior exactly.

### 5.3 `profile_region()` — the user-facing helper

```python
# User code — inside their vLLM script
from vllm import LLM, SamplingParams
from colonel.profiling.vllm import profile_region

llm = LLM(model="Qwen/Qwen2.5-7B", enforce_eager=False)
llm.generate(["warmup"] * 4, SamplingParams(max_tokens=8))  # warm CUDA graphs

with profile_region(llm):
    outs = llm.generate(prompts, SamplingParams(max_tokens=128))
```

Semantics:
- On `__enter__`: `llm.collective_rpc(_start_profiler)` where
  `_start_profiler(worker)` calls `torch.cuda.cudart().cudaProfilerStart()`.
- On `__exit__`: the symmetric stop.
- If `enforce_eager=True` (no CUDA graphs), the helper still works —
  ncu/nsys get a profile, just without graph-replay semantics.
- If the environment variable `COLONEL_DISABLE_PROFILE_REGION=1` is set,
  the helper becomes a no-op. Critical for letting _the same script_
  run in production uninstrumented and in Colonel instrumented.

Exactly this pattern is already proven in `~/smoke/inproc_decode_rpc.py`.
Track A lifts that 30-line proof-of-concept into a packaged helper.

### 5.4 ncu evaluator evolution

Today `nsight_compute.py:build_command()` is hard-coded:

```python
cmd_parts = [self._ncu_path, "--csv", "--log-file", "/dev/stdout",
             "--set", "full", "--target-processes", "all",
             ctx.full_command]
```

Changes:
- Accept `ctx.metadata["ncu_replay_mode"]` (default `"kernel"`; vLLM
  flavor sets `"application"`).
- Accept `ctx.metadata["ncu_capture_range"]` (default unset; vLLM flavor
  sets `"cudaProfilerApi"` so `profile_region()` actually gates capture).
- Accept `ctx.metadata["ncu_sections"]` (default `--set full`; advanced
  users can narrow to reduce replay cost).

No existing `colonel run` invocation changes behavior. Every new knob is
additive and reads from `ctx.metadata`, which already exists.

### 5.5 nsys evaluator evolution

Symmetric to §5.4 but for nsys. Specifically:
- `--capture-range cudaProfilerApi --capture-range-end stop` is set when
  `ctx.metadata["nsys_capture_range"] == "cudaProfilerApi"`.
- NVTX trace is opt-in via `ctx.metadata["nsys_enable_nvtx"]`; vLLM
  flavor sets this AND injects a hook (§5.6) to flip
  `enable_layerwise_nvtx_tracing` inside the worker.

### 5.6 NVTX enablement hook

`vllm.config.ObservabilityConfig.enable_layerwise_nvtx_tracing` is a
Python attribute; the straightforward mechanism is a monkey-patch at
import time _inside the user's script_, behind an opt-in:

```python
from colonel.profiling.vllm import enable_nvtx
enable_nvtx()   # call before constructing LLM(...)
```

We do _not_ modify vLLM's source. We simply set the config default via
its public API before `LLM(...)` is constructed. When the env var
`COLONEL_VLLM_ENABLE_NVTX=1` is set (which `--flavor vllm` does
automatically), `profile_region` calls `enable_nvtx()` as a safety net.

### 5.7 `colonel setup` additions

A new optional step between today's step 5 (GPU counters) and step 6 (AI
provider):

- Detect vLLM: `python -c "import vllm; print(vllm.__version__)"`
- If present, print the one-time env-var guidance (put these in your
  `.env` / service manifest):
  - `VLLM_ALLOW_INSECURE_SERIALIZATION=1`
  - `VLLM_USE_V1=1`
- If absent, skip silently. No forced install.

Rationale: Colonel's setup wizard is Colonel's credibility surface. A
user running `colonel setup` on a vLLM-serving host should be told, in
one paragraph, what they'll need. Today they are not.

### 5.8 Tests

Unit level (no GPU, fast):
- `profile_region()` context-manager API matches expectations on a
  mocked `LLM` with a fake `collective_rpc`.
- CLI flag `--flavor vllm` sets the expected env vars and `ncu/nsys`
  flags in the generated `ProfileContext`.

Integration level (requires GPU, gated by `pytest.mark.gpu`):
- End-to-end `colonel run --flavor vllm --evaluator nsys` on a
  10-token Qwen2.5-0.5B decode produces a `ProfileResult` with ≥5
  distinct kernels, including at least one `flash_fwd_*` and at least
  one `gemm_*`.
- Same with `--evaluator ncu` produces a `ProfileResult` with populated
  `occupancy_pct` and `compute_throughput_pct` on at least one kernel.
- Regression: `--flavor generic` on a plain matmul continues to produce
  identical output to today's `colonel run`.

### 5.9 Documentation

- Replace `demos/vllm/GUIDE.md`'s "Make sure you're using the single-
  process demo scripts (not vLLM directly)" paragraph with "To profile
  real vLLM V1, pass `--flavor vllm` and wrap the region you care about
  in `profile_region()`; see `demos/vllm/05_real_vllm_decode.py`."
- Add `docs/vllm_profiling.md` that explains the V1 subprocess model,
  the three gotchas from §4.3, and the one-liner mitigation for each.
- Update `README.md`'s "What Colonel Reveals About LLM Inference" to
  point to real vLLM output instead of HF-Transformers output.

---

## 6. Design — Track B (Colonel-found vLLM upstream fix)

### 6.1 Hypothesis (revised 2026-04-21 after upstream reconnaissance)

FA2's internal `get_num_splits_heuristic` (in the C++ source vendored at
`csrc/flash_attn/src/flash_fwd_kernel.h` et al. inside
`vllm-project/flash-attention`) picks `num_splits` using a formula that
assumes GPU SM counts from the Ampere-A100 era (108 SMs). On lower-SM
GPUs (A4000 / RTX-A / L4) the heuristic rarely over-allocates. On
higher-SM GPUs (H100 132 SMs, B200 208 SMs) the same heuristic
_systematically_ under-allocates blocks for small-batch decode, leaving
SMs idle.

**Upstream state (reconnaissance at PRD write time):**
- `vllm-project/flash-attention` **PR #72** (merged Jul 2025) added
  `num_splits` as an input arg to `flash_attn_varlen_func` at the
  Python binding layer.
- `vllm-project/flash-attention` **PR #110** (merged Dec 2025) extended
  that support into `mha_varlen_fwd` for FA2, validated on A100 with
  Qwen-3 32B, primarily for the batch-invariant use case (`num_splits=1`
  for reproducibility).
- **Net effect:** the Python-level pipe for passing `num_splits` into
  FA2 now exists. The `NotImplementedError("FA2 does not support
  num_splits > 1")` line in vLLM's own `flash_attn_interface.py`
  (verified on disk this session in the 0.19.1 install) is therefore
  either stale or specific to a path PR #110 did not touch — Phase 3
  must diff the vendored binding source against what vLLM is actually
  shipping.
- **No open issue in `vllm-project/vllm` mentions `flash_fwd_splitkv`
  or the `num_splits` heuristic at PRD write time** — Track B is not
  racing against an active contributor. (Verified via GitHub issue
  search 2026-04-21.)

**Revised hypothesis, after this reconnaissance:** the _plumbing_ to
pass a better `num_splits` is now present upstream; the _heuristic_
that runs when Python leaves `num_splits=0` is what still
under-utilizes SMs at small batch. Track B's contribution surface is
therefore:

- **(B-primary)** Improve `get_num_splits_heuristic` in the vendored
  C++ so the default (`num_splits=0` → "use the heuristic") picks a
  better value at small batch / high SM count.
- **(B-alt)** If the heuristic is architecturally hard to change,
  expose a Python/vLLM-side policy that detects the small-batch decode
  regime and passes an overriding `num_splits` at call time — riding
  on PR #110's plumbing.

Either path is an upstream contribution to `vllm-project/flash-attention`
(B-primary) or `vllm-project/vllm` (B-alt), not a private patch.

### 6.2 Evidence path

1. **Measure today.** Run Colonel with Track A on a real vLLM 7B decode
   at batch=1, on as many GPU tiers as we can access (at minimum: A4000
   local, A10G or A100 via Shadeform or Lambda).
2. **Extract the SOL signature.** Confirm `flash_fwd_splitkv` shows the
   "low SM%, very low DRAM%, low wave count" fingerprint across tiers,
   and that wave count _decreases_ with SM count for the same grid.
3. **Read the heuristic.** Clone `vllm-project/flash-attention` (the fork
   vLLM vendors — not upstream Dao-AILab), locate
   `get_num_splits_heuristic`, and map the formula to the observed grid.
4. **Propose a fix.** Draft a heuristic that is SM-count-aware for the
   decode (small-batch, single-query) regime specifically. The bwd path
   and prefill path should remain unchanged.
5. **Measure after.** Re-run Colonel with the patched wheel. Compare
   wave occupancy, per-kernel SOL, end-to-end tokens/sec, and
   time-per-decode-token. Accept only if end-to-end regresses on no
   tier.

### 6.3 Why this is upstreamable and not just an internal patch

- The fork `vllm-project/flash-attention` exists _precisely_ for changes
  that are vLLM-specific but not yet upstream-mergable to Dao-AILab
  flash-attention. A split-heuristic tune for the decode path is
  exactly in-scope.
- If the tuning turns out to generalize, it can be re-proposed against
  `Dao-AILab/flash-attention` in a follow-up PR. But the PR that ships
  first should be against the vLLM fork.

### 6.4 Acceptance criteria for the upstream PR

- Measured wave occupancy for `flash_fwd_splitkv` at batch=1 decode on
  7B, on ≥2 GPU tiers, increases from <1 to ≥1.5 wave-equivalents.
- Per-kernel duration for `flash_fwd_splitkv` decreases ≥20% (or
  decreases on some tiers without regressing on others).
- End-to-end tokens/sec on the standard vLLM-bench "decode" scenario
  does not regress on any tier we measure. Ideally improves 1–3%.
- No regression on the batched-prefill path, measured by re-running the
  existing flash-attention test suite.
- Colonel output (the "before" and "after" sessions) is the artifact
  attached to the PR as evidence.

### 6.5 What could kill this

- **The heuristic is already SM-aware and we just haven't read it
  carefully yet.** Likely outcome: the heuristic uses `num_sms` but
  caps somewhere (max `num_splits`), and we need a cap adjustment
  rather than a full rewrite. Still fixable, just smaller.
- **The actual bottleneck is elsewhere on production model sizes.**
  Qwen-0.5B's decode is unusually small-grid. On 7B the splitkv grid
  may already be fine, in which case we re-pick a different Track B
  target (likely `reshape_and_cache_flash` fusion, or an Inductor
  RMSNorm fusion). The Track A infrastructure makes re-picking a one-
  command iteration, not a re-spike.
- **FA3/FA4 path on Hopper/Blackwell is actually used in practice more
  than we think.** On hardware where FA3 is picked, the problem is
  different (num_splits is exposed and controllable via vLLM's
  `max_num_splits`). Our PR would still be valuable but narrower in
  user-count impact. This affects "how we sell it," not "should we do
  it."

---

## 7. Milestones

### Phase 1 — Track A MVP (target: ~1 week of focused work)

- [ ] `colonel.profiling.vllm.profile_region()` implemented and unit-
      tested against a mocked LLM.
- [ ] `--flavor vllm` wired through CLI → `ProfileContext.metadata` →
      evaluators; default knobs for ncu and nsys.
- [ ] `nsight_compute.py` + `nsight_systems.py` honor metadata knobs;
      `--flavor generic` behavior unchanged (regression test).
- [ ] One real vLLM demo (`demos/vllm/05_real_vllm_decode.py`) that
      runs under `colonel run --flavor vllm` end-to-end on the
      Shadeform A4000 box.
- [ ] `docs/vllm_profiling.md` written.
- [ ] `colonel setup` gains vLLM-detection step.

**Phase 1 exit signal:** `colonel run --flavor vllm --evaluator ncu --
python demos/vllm/05_real_vllm_decode.py` on a fresh box produces a
Colonel session whose `profile_summary.md` and AI analysis are
substantive (not stubs) and match the kernel-level data we already know
from the spike report.

### Phase 2 — Track A hardening (target: ~1 week after Phase 1)

- [ ] HF-impersonator demos (`01_..04_`) either retired or migrated to
      real vLLM; `GUIDE.md` rewritten.
- [ ] Integration tests gated by `pytest.mark.gpu` added; CI notes
      added to `README.md` about how to run them on a GPU box.
- [ ] Remote target (`--target ssh://...`) verified against a
      Shadeform/Lambda GPU box for the vLLM flavor end-to-end.
- [ ] Analyzer prompt (`prompts.py`) gains a small vLLM-aware section:
      when the kernel list contains `flash_fwd_*` or `reshape_and_cache_*`,
      the prompt mentions the V1 subprocess context so the LLM does not
      hallucinate single-process assumptions.

### Phase 3 — Track B (target: ~2–3 weeks after Phase 2)

- [ ] Run Phase 1/2 Colonel on at least two non-A4000 GPU tiers
      (targets: A10G, A100-40GB, ideally H100).
- [ ] Confirm `flash_fwd_splitkv` SOL signature at 7B batch=1 decode.
- [ ] Read and annotate `get_num_splits_heuristic` in
      `vllm-project/flash-attention`.
- [ ] Draft patch; validate locally with a rebuilt wheel.
- [ ] File PR against `vllm-project/flash-attention` with Colonel
      sessions as evidence.
- [ ] Write `docs/case_studies/flash_attn_splitkv.md` — the hero demo.

**Phase 3 exit signal:** PR merged or review-engaged upstream;
case-study doc published.

### Phase 4 — Optional horizons (scope decision after Phase 3)

- Multi-flavor infrastructure: same pattern for SGLang, TRT-LLM.
- Analyzer prompt enrichment: embed waves-per-SM math, SM-count-aware
  suggestions.
- Session-diff narration: when a user runs before/after on a known
  bottleneck (e.g. splitkv), have Colonel assert the improvement
  numerically rather than asking the LLM to judge.

---

## 8. Success Metrics

| Metric | Baseline | Target (Phase 1) | Target (Phase 3) |
|---|---|---|---|
| Lines of Colonel code in `demos/vllm/` that actually import `vllm` | **0** | ≥1 demo does | ≥3 demos do |
| Distinct kernels captured by `colonel run --flavor vllm --evaluator ncu` on a real 7B model | N/A (unsupported) | ≥8 | ≥8, including splitkv |
| Mean wave occupancy for `flash_fwd_splitkv` at batch=1, 7B, on a representative SM-count tier | ~0.2–0.6 | unchanged (measurement only) | **≥1.5** |
| Upstream PRs filed by the Colonel project against vLLM-adjacent repos | **0** | 0 | **≥1** |
| `colonel setup` detects vLLM and prints correct guidance | no | yes | yes |
| End-to-end iteration time "profile → change → reprofile" on a vLLM decode, timed from `colonel run` to `colonel run` | N/A | ≤5 min on A4000 | ≤5 min on A4000 |

---

## 9. Risks & Open Questions

### 9.1 Security

`VLLM_ALLOW_INSECURE_SERIALIZATION=1` is opt-in in vLLM for a reason —
it lets arbitrary Python callables traverse the RPC boundary. Colonel
setting it by default under `--flavor vllm` is a convenience that
trades off against isolation.

Mitigation: the flag is set _only_ for the Colonel-invoked process, via
`ProfileContext.env`. It does not propagate to the user's production
serving environment. Documentation makes this explicit. We also gate on
`COLONEL_REQUIRE_EXPLICIT_INSECURE_SERIALIZATION=1` to force the user
to set it themselves if they want even tighter hygiene.

### 9.2 vLLM API volatility

vLLM's V1 internals (`collective_rpc` signature,
`ObservabilityConfig.enable_layerwise_nvtx_tracing`, the exact
`EngineCore` process model) are not stable public API. They change
between minor versions.

Mitigation: pin tested vLLM versions in `docs/vllm_profiling.md`. Use
attribute-existence checks before setting config fields. Fail loudly
and with a pointer to this PRD if vLLM's surface has moved.

### 9.3 The PRD we thought existed doesn't

A prior session's memory referenced `docs/vllm_contribution_prd.md` as
if it were an authoritative document. It was not — it was notes from a
Mac that never made it into this repo. _This_ document is the first
real version and should be treated as the canonical plan going forward.
Memory has been updated to reflect that.

### 9.4 What we have not yet verified

- **`enable_layerwise_nvtx_tracing=True` actually produces NVTX ranges
  inside the EngineCore subprocess.** The spike established that
  leaving it `False` produces zero NVTX; it did not test flipping it.
  Phase 1 must verify this before we document it as the mitigation.
- **`--replay-mode application` on 7B-class models does not deadlock or
  OOM.** The spike tested only 0.5B. Phase 3 is the first place this is
  exercised on production sizes.
- **`profile_region()` semantics under nested use.** If a user wraps
  two generate calls in separate `with profile_region(llm)` blocks, do
  nsys/ncu handle the multiple start/stop cycles correctly? Known
  answer is "yes for nsys with `--capture-range cudaProfilerApi
  --capture-range-end stop:2`" but we should test.

### 9.5 Open questions

- Do we want a `--flavor` system at all, or would a simpler
  `colonel.profiling.vllm.profile_region()` + a documentation page be
  enough? Decision: yes, we want the flag, because the ncu/nsys default
  knobs (replay mode, capture range) are non-obvious and should be set
  centrally, not documented as a "don't forget to also pass…" footnote.
- Does the vLLM flavor belong in `colonel/profiling/` or
  `colonel/targets/`? Decision: profiling. A target is _where_ we run
  (local, ssh). A flavor is _what we run_ (generic process, vLLM engine).
- Should Colonel ship `vllm` as an optional extra (`pip install
  colonel[vllm]`)? Not in Phase 1. vLLM has heavy GPU/CUDA build
  requirements that we do not want to entangle with `pip install
  colonel`. Phase 4 revisit.

---

## 10. Prior Art (and how this PRD differs)

The README cites PEAK, KForge, ParaCodex, and TritonForge. Mapping to
this PRD:

- **PEAK** — natural-language → kernel transform. This PRD is _not_
  about transforms; it is about getting Colonel's profiling story to
  the point where a future PEAK-style layer could operate on real vLLM
  data. Track A is the prerequisite for a PEAK-inspired Phase 4.
- **KForge** — synthesis-based kernel generation. Same relationship:
  this PRD does not synthesize kernels (N1). But the eventual hero
  demo for a KForge-style feature would be "find the splitkv
  heuristic problem, synthesize a fix." Track B is a manual, human-
  authored version of exactly that loop — the blueprint for later
  automation.
- **ParaCodex** — profiling-guided code generation. Colonel's existing
  analyzer prompts already nod at this. Track A makes the "profiling"
  half of ParaCodex-style work actually trustworthy on real serving.
- **TritonForge** — profiling-guided Triton optimization. Colonel's
  current `demos/vllm/` leans heavily on Triton RMSNorm kernels
  appearing in the kernel list. Fusing those via Triton is a natural
  Phase 4 follow-on _after_ Track A lands.

---

## 11. Out of Scope

Reiterating from §3.2 in decision language, so future scope creep has a
clean answer:

- Authoring new attention / GEMM / fused kernels. (Future work.)
- Supporting vLLM V0. (Deprecated.)
- Building a throughput benchmark suite or leaderboard. (Use `vllm
  bench`.)
- Windows support for the vLLM flavor. (vLLM doesn't run there.)
- Any changes to vLLM's `main` repo directly in Phase 3 — our upstream
  PR goes to `vllm-project/flash-attention`. Changes to `vllm` proper
  are a separate conversation.
- Automatic PR generation ("Colonel opens the PR for you"). Maybe
  someday. Not this PRD.

---

## 12. Appendix A — Spike artifacts referenced

On the Shadeform A4000 box:

- `~/smoke/reports/vllm_decode.nsys-rep` — end-to-end nsys capture of
  40 decode requests against a running Qwen2.5-0.5B server.
- `~/smoke/reports/ncu_decode.ncu-rep` — ncu SOL section on 5 hot
  vLLM decode kernels, 4 replay passes each. _Every number in §4.2 was
  re-extracted from this file in the current session; it is authoritative._
- `~/smoke/reports/ncu_graph_smoke.ncu-rep` — ncu on a pure
  `torch.cuda.graph()` replay (Q1 answer).
- `~/smoke/inproc_decode_rpc.py` — the 30-line working pattern that
  §5.3 lifts into a packaged helper.

## 13. Appendix B — Concrete code sketches

### 13.1 `profile_region()` sketch

```python
# colonel/profiling/vllm/region.py
from contextlib import contextmanager

@contextmanager
def profile_region(llm, *, enable_nvtx: bool = True):
    """Bracket a vLLM generation call with cudaProfilerStart/Stop
    inside the EngineCore worker subprocess.

    Usage:
        with profile_region(llm):
            llm.generate(...)
    """
    if enable_nvtx:
        _ensure_nvtx_enabled()

    def _start(worker):
        import torch
        torch.cuda.cudart().cudaProfilerStart()

    def _stop(worker):
        import torch
        torch.cuda.cudart().cudaProfilerStop()

    llm.collective_rpc(_start)
    try:
        yield
    finally:
        llm.collective_rpc(_stop)
```

### 13.2 CLI wiring sketch

```python
# colonel/cli/profile_cmd.py (excerpt)
@app.command()
def run(
    command: list[str],
    flavor: str = typer.Option("generic", "--flavor"),
    evaluator: str = typer.Option("auto", "--evaluator", "-e"),
    # ...existing options...
):
    metadata = {}
    env = {}
    if flavor == "vllm":
        env.update({
            "VLLM_ALLOW_INSECURE_SERIALIZATION": "1",
            "VLLM_USE_V1": "1",
            "COLONEL_VLLM_ENABLE_NVTX": "1",
        })
        metadata.update({
            "ncu_replay_mode": "application",
            "ncu_capture_range": "cudaProfilerApi",
            "nsys_capture_range": "cudaProfilerApi",
            "nsys_enable_nvtx": True,
        })
    ctx = ProfileContext(
        command=command[0], args=command[1:],
        env=env, metadata=metadata, evaluator=evaluator,
        # ...
    )
    # ...hand off to executor as today
```

### 13.3 ncu evaluator extension sketch

```python
# colonel/evaluators/nsight_compute.py (excerpt)
def build_command(self, ctx: ProfileContext) -> str:
    parts = [self._ncu_path, "--csv", "--log-file", "/dev/stdout",
             "--set", ctx.metadata.get("ncu_sections", "full"),
             "--target-processes", "all"]
    replay = ctx.metadata.get("ncu_replay_mode")
    if replay:
        parts.extend(["--replay-mode", replay])
    capture = ctx.metadata.get("ncu_capture_range")
    if capture:
        parts.extend(["-c", capture])
    parts.append(ctx.full_command)
    return " ".join(parts)
```

These are sketches, not final code. They establish that the required
changes to `context.py`, `nsight_compute.py`, `nsight_systems.py`, and
`profile_cmd.py` are small, local, and backward-compatible.

---

## 14. Reconciliation with Open Upstream Work (added 2026-04-21)

This PRD was drafted before scanning the `vllm-project/vllm`
performance-label issue queue. This section reconciles the plan with
what is actually in flight upstream, so reviewers don't have to
re-check.

### 14.1 Upstream items that overlap with Track A

| # | Title | Overlap |
|---|---|---|
| [vllm #39603](https://github.com/vllm-project/vllm/issues/39603) | [Perf][Bug] "meet same thread as registerClient when start_profile" | Bug in vLLM's `/start_profile` API endpoint (`api_router.py`). Distinct from nsys/ncu profiling, but adjacent: Colonel's `profile_region()` deliberately sidesteps this path by using `collective_rpc(cudaProfilerStart/Stop)` inside the worker. Phase 2 stretch: if we fully understand the root cause, we file a separate small PR fixing the registerClient threading issue — a natural "first vLLM-proper PR from the Colonel project." |

No other open issue in the performance queue at PRD write time
directly overlaps with the Colonel profile-adapter mission.

### 14.2 Upstream items that overlap with Track B

| # | Title | Relationship |
|---|---|---|
| [flash-attention #72](https://github.com/vllm-project/flash-attention/pull/72) | "[Misc] Add num_splits input arg to flash_attn_varlen_func" | **Merged Jul 2025.** This is the Python-level plumbing for `num_splits`. Our Track B rides on this. |
| [flash-attention #110](https://github.com/vllm-project/flash-attention/pull/110) | "Add num_splits for mha_varlen_fwd FA2, support batch invariant" | **Merged Dec 2025.** Extended the FA2 path for batch-invariant mode on SM80. Our Track B either builds on this (if generalized) or complements it (if still batch-invariant-gated). |
| [flash-attention #124](https://github.com/vllm-project/flash-attention/pull/124) | "Combine kernel: increase pipeline depth from 4 to 8 stages" | Open Mar 2026. Adjacent kernel tuning — does not touch `num_splits`. Worth watching for conflicts during Phase 3. |
| [vllm #39924](https://github.com/vllm-project/vllm/pull/39924) | "[Attention] Add FLASH_ATTN_MLA_SPARSE backend" | DNM draft, MLA-specific — not the same code path as our splitkv target. |

### 14.3 Upstream items that **reinforce** our non-goals

| # | Title | Why this reinforces N1 |
|---|---|---|
| [vllm #39952](https://github.com/vllm-project/vllm/pull/39952) | "[Feature] Fused SiLU + Mul + per-token dynamic FP8 quantization (Triton)" | Active Triton-fusion PR. RMSNorm/SiLU fusion has upstream momentum — we stay out. |
| [vllm #39641](https://github.com/vllm-project/vllm/pull/39641) | "[Feature] silu block quant fusion Triton kernel" | Same; Triton fusion is active upstream. |
| [vllm #39897](https://github.com/vllm-project/vllm/pull/39897) | "[WIP][Kernel] Generalized LL GEMMs with PDL" | Active GEMM-kernel PR using Programmatic Dependent Launch. Reinforces our "don't go after the `ampere_gemm` kernel at the Colonel layer" call. |

### 14.4 Upstream items that are **unrelated** to this PRD

Listed here explicitly so no reviewer worries we missed them:

- [vllm #40181](https://github.com/vllm-project/vllm/pull/40181) — Gemma-4-MoE tuning on H200. MoE, not attention.
- [vllm #40119](https://github.com/vllm-project/vllm/pull/40119) — RISC-V RVV CPU attention kernels. CPU, not GPU.
- [vllm #40288, #40050, #39795](https://github.com/vllm-project/vllm/issues) — `vllm bench` tooling. Benchmarking, not profiling.
- vLLM IR track (#40167, #40135, #39453, etc.) — new IR representation for ops. Infrastructure-level; orthogonal.
- Spec-decode regressions (#39790, #39775) and MTP regressions (#39680) —
  Colonel could *characterize* these once Track A ships, but we are
  not picking them as our own target.
- Server variance (#40001) — tail-latency characterization task; same
  status as spec-decode regressions.

### 14.5 What the reconciliation changes in the plan

- §6.1 hypothesis is **revised** (above) to reflect that `num_splits`
  Python plumbing has landed; the contribution is about the heuristic
  or the small-batch-decode policy.
- Phase 2 gains an **optional stretch PR** against `vllm-project/vllm`
  itself (fixing the `/start_profile` threading bug #39603), alongside
  the main Track B PR against `vllm-project/flash-attention`.
- §3.2 non-goal N1 (no new kernels) is **reinforced**, because fusion
  and GEMM-kernel work are already live upstream.

### 14.6 Explicit honesty about grounding

The pre-v14 version of this PRD was drafted from:
- Re-extraction of `ncu_decode.ncu-rep` on disk (verified numbers).
- Reading Colonel's repo (`README.md`, `AGENTS.md`, evaluators,
  parsers, prompts).
- Reading the vLLM 0.19.1 install in `.venv` (the `NotImplementedError`
  grep hit, the FA-version selection logic, the `collective_rpc` call
  sites).
- My memory note from the prior-session spike.

It was **not** pre-grounded in the specific upstream issues shown to
me after draft (the PRs #40181 and #40119 in particular do not
overlap at all with the plan). §14 closes that gap.

## 15. Decision Record

| # | Decision | Rationale |
|---|---|---|
| D1 | Two-track plan (Track A before Track B) | Track A is the prerequisite for Track B being credible. Track B is the hero demo for Track A. Shipping them in the wrong order ships nothing that stands on its own. |
| D2 | New `colonel/profiling/vllm/` package, not modifications to evaluators | Evaluators are profilers; flavors are workloads. Don't cross those axes as combinatorial modules. |
| D3 | `--flavor` CLI flag, not auto-detection | Auto-detection ("if the script imports vllm, flip the flags") is unreliable and hides what Colonel is doing. Explicit > magical. |
| D4 | Phase 1 lands even without Track B | Colonel gains a real feature whether or not the upstream PR lands. Track B is the marketing; Track A is the product. |
| D5 | Target the vLLM fork of flash-attention, not Dao-AILab | The fork exists for vLLM-specific tunings; it is the right first home. Generalizing upstream to Dao-AILab is a post-merge follow-up. |
| D6 | Do not retire HF-impersonator demos in Phase 1 | Keep the fallback working until real-vLLM demos are stable. Retire in Phase 2. |
| D7 | Do not ship `vllm` as a pip extra in Phase 1 | vLLM's install graph is heavy; entangling it with `pip install colonel` hurts casual users. Phase 4 revisit. |
| D8 | Canonicalize this document as the PRD | Prior session's "memory PRD" was notes, not a doc. Treat this file as the authoritative plan. |
| D9 | Add §14 reconciliation rather than rewrite §6 wholesale | The draft's evidence-and-reasoning chain is sound; the upstream landscape is new context that tunes the contribution surface without invalidating the approach. An additive reconciliation section is the honest record. |
| D10 | Keep `get_num_splits_heuristic` as primary B-target despite PR #110 | PR #110 added a plumbing knob for the batch-invariant case (num_splits=1 for reproducibility). The _heuristic_ that runs when num_splits is left at 0 for general decode is untouched and is what the SOL data implicates. |
