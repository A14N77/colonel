"""Setup command: interactive wizard to configure Colonel and verify the environment."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time

import typer

from colonel.utils.rich_output import (
    console,
    print_error,
    print_header,
    print_info,
    print_success,
    print_warning,
)

setup_app = typer.Typer(invoke_without_command=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd: str, timeout: float = 15.0) -> tuple[bool, str]:
    """Run a shell command and return (success, combined_output)."""
    try:
        proc = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout,
        )
        return proc.returncode == 0, (proc.stdout + proc.stderr).strip()
    except subprocess.TimeoutExpired:
        return False, "(command timed out)"
    except FileNotFoundError:
        return False, ""


def _ask(prompt: str, default: str = "") -> str:
    """Prompt the user for input with an optional default."""
    suffix = f" [{default}]" if default else ""
    try:
        value = console.input(f"  [info]{prompt}{suffix}:[/info] ").strip()
    except (EOFError, KeyboardInterrupt):
        value = ""
    return value or default


def _ask_yes(prompt: str, default: bool = True) -> bool:
    hint = "Y/n" if default else "y/N"
    try:
        answer = console.input(f"  [info]{prompt} ({hint}):[/info] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = ""
    if not answer:
        return default
    return answer in ("y", "yes")


def _step(num: int, title: str) -> None:
    """Print a step header."""
    console.print(f"\n[bold]{num}. {title}[/bold]")


# ---------------------------------------------------------------------------
# Step 1: NVIDIA Driver & GPU
# ---------------------------------------------------------------------------

def _check_nvidia_driver() -> bool:
    _step(1, "NVIDIA Driver & GPU")

    ok, out = _run(
        "nvidia-smi --query-gpu=name,driver_version,memory.total "
        "--format=csv,noheader"
    )
    if ok and out:
        for line in out.splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                console.print(
                    f"  GPU: [metric]{parts[0]}[/metric]  "
                    f"Driver: [metric]{parts[1]}[/metric]  "
                    f"Memory: [metric]{parts[2]}[/metric]"
                )
        print_success("  NVIDIA driver detected.")
        return True

    print_error("  nvidia-smi not found or failed.")
    console.print(
        "  [dim]The NVIDIA driver is required. Install it with your package manager:[/dim]\n"
        "  [dim]  Ubuntu/Debian: sudo apt install nvidia-driver-550[/dim]\n"
        "  [dim]  Or see: https://docs.nvidia.com/cuda/cuda-installation-guide-linux/[/dim]"
    )
    return False


# ---------------------------------------------------------------------------
# Step 2: CUDA Toolkit
# ---------------------------------------------------------------------------

def _check_cuda() -> bool:
    _step(2, "CUDA Toolkit")

    ok, out = _run("nvcc --version")
    if ok:
        for line in out.splitlines():
            if "release" in line.lower():
                console.print(f"  {line.strip()}")
        print_success("  CUDA toolkit detected.")
        return True

    # nvcc not on PATH — check common install locations
    import glob as globmod

    cuda_homes = sorted(globmod.glob("/usr/local/cuda-*/bin/nvcc"))
    if cuda_homes:
        console.print(f"  [warning]nvcc found at {cuda_homes[-1]} but not on PATH.[/warning]")
        console.print(f"  [dim]Add to your shell profile:  export PATH={os.path.dirname(cuda_homes[-1])}:$PATH[/dim]")
        return True

    print_warning("  nvcc not found. CUDA toolkit recommended but not strictly required.")
    console.print(
        "  [dim]Install: sudo apt install nvidia-cuda-toolkit[/dim]\n"
        "  [dim]Or: https://developer.nvidia.com/cuda-downloads[/dim]"
    )
    return False


# ---------------------------------------------------------------------------
# Step 3: Nsight Systems (nsys)
# ---------------------------------------------------------------------------

def _check_nsys() -> tuple[bool, str]:
    _step(3, "Nsight Systems (nsys)")

    import glob as globmod

    # Search in order of preference
    candidates: list[str] = []

    # 1. On PATH
    on_path = shutil.which("nsys")
    if on_path:
        candidates.append(on_path)

    # 2. Common install locations
    search_globs = [
        "/usr/local/bin/nsys",
        "/opt/nvidia/nsight-systems/*/bin/nsys",
        "/opt/nvidia/nsight-systems/*/target/linux-x64/nsys",
        "/usr/lib/nsight-systems/*/bin/nsys",
    ]
    for pattern in search_globs:
        candidates.extend(sorted(globmod.glob(pattern)))

    # De-duplicate preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for c in candidates:
        real = os.path.realpath(c)
        if real not in seen:
            seen.add(real)
            unique.append(c)

    for path in unique:
        ok, out = _run(f'"{path}" --version')
        if ok:
            ver_lines = [l for l in out.splitlines() if "version" in l.lower()]
            console.print(f"  Found: [metric]{path}[/metric]")
            if ver_lines:
                console.print(f"  {ver_lines[0].strip()}")

            # Check if it has execute permission issues (common on apt installs)
            if not os.access(path, os.X_OK):
                console.print(f"  [warning]Missing execute permission on {path}[/warning]")
                if _ask_yes("  Fix permissions? (requires sudo)"):
                    parent = os.path.dirname(os.path.dirname(path))
                    _run(f"sudo chmod -R a+rx {parent}")
                    print_success("  Permissions fixed.")

            print_success("  nsys detected.")
            return True, path

    # Not found — offer to install
    print_error("  nsys not found.")
    console.print(
        "  [dim]nsys is required for system-level GPU profiling.[/dim]\n"
        "  [dim]Install options:[/dim]\n"
        "  [dim]  Ubuntu/Debian:  sudo apt install nsight-systems[/dim]\n"
        "  [dim]  From NVIDIA:    https://developer.nvidia.com/nsight-systems[/dim]\n"
        "  [dim]  CUDA toolkit:   nsys is often bundled with cuda-toolkit packages[/dim]"
    )

    if _ask_yes("  Attempt automatic install via apt?", default=False):
        console.print("  Installing nsight-systems...")
        ok_install, install_out = _run("sudo apt-get install -y nsight-systems 2>&1", timeout=120)
        if ok_install:
            # Re-check
            nsys_path = shutil.which("nsys")
            if nsys_path:
                print_success(f"  nsys installed at {nsys_path}")
                return True, nsys_path
            # Sometimes installed but needs PATH or chmod
            for pattern in search_globs:
                matches = globmod.glob(pattern)
                if matches:
                    path = matches[-1]
                    _run(f"sudo chmod -R a+rx {os.path.dirname(os.path.dirname(path))}")
                    print_success(f"  nsys installed at {path}")
                    return True, path
        print_warning("  apt install failed. Try installing manually.")
        if install_out:
            # Show last few lines of output for debugging
            for line in install_out.splitlines()[-3:]:
                console.print(f"  [dim]{line}[/dim]")

    return False, ""


# ---------------------------------------------------------------------------
# Step 4: Nsight Compute (ncu)
# ---------------------------------------------------------------------------

def _check_ncu() -> tuple[bool, str]:
    _step(4, "Nsight Compute (ncu)")

    import glob as globmod

    candidates: list[str] = []

    on_path = shutil.which("ncu")
    if on_path:
        candidates.append(on_path)

    for pattern in ["/usr/local/cuda*/bin/ncu", "/usr/local/cuda/bin/ncu"]:
        candidates.extend(sorted(globmod.glob(pattern)))

    seen: set[str] = set()
    for c in candidates:
        real = os.path.realpath(c)
        if real in seen:
            continue
        seen.add(real)

        ok, out = _run(f'"{c}" --version')
        if ok:
            ver_lines = [l for l in out.splitlines() if "Version" in l]
            console.print(f"  Found: [metric]{c}[/metric]")
            if ver_lines:
                console.print(f"  {ver_lines[0].strip()}")
            print_success("  ncu detected.")
            return True, c

    print_warning("  ncu not found (optional — provides detailed per-kernel metrics).")
    console.print(
        "  [dim]ncu is bundled with the CUDA toolkit. If you have CUDA installed,[/dim]\n"
        "  [dim]check that /usr/local/cuda/bin is on your PATH.[/dim]\n"
        "  [dim]Download: https://developer.nvidia.com/nsight-compute[/dim]"
    )
    return False, ""


# ---------------------------------------------------------------------------
# Step 5: GPU Performance Counter Permissions
# ---------------------------------------------------------------------------

def _check_ncu_permissions() -> bool:
    _step(5, "GPU Performance Counter Permissions")

    ok, out = _run("cat /proc/driver/nvidia/params")
    if not ok:
        print_warning("  Could not read nvidia params (driver may not be loaded).")
        return False

    if "RmProfilingAdminOnly: 0" in out:
        print_success("  GPU counters accessible to all users.")
        return True

    print_warning("  GPU performance counters are restricted to admin users.")
    console.print(
        "  [dim]ncu needs unrestricted access to GPU hardware counters.[/dim]\n"
        "  [dim]Without this, ncu profiling will fail with ERR_NVGPUCTRPERM.[/dim]"
    )

    if not _ask_yes("  Fix now? (requires sudo, may need reboot)"):
        console.print(
            "\n  [dim]To fix manually later:[/dim]\n"
            '  [dim]  echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" \\\n'
            "  [dim]    | sudo tee /etc/modprobe.d/ncu-perms.conf[/dim]\n"
            "  [dim]  sudo reboot[/dim]"
        )
        return False

    # Write modprobe config
    _run(
        "sudo sh -c '"
        'echo "options nvidia NVreg_RestrictProfilingToAdminUsers=0" '
        "> /etc/modprobe.d/ncu-perms.conf'"
    )
    print_info("  Written /etc/modprobe.d/ncu-perms.conf")

    # Attempt live reload
    console.print("  Attempting live nvidia module reload...")
    _run("sudo kill $(pgrep nvidia-persistenced) 2>/dev/null", timeout=5)
    _run("sudo kill $(pgrep nv-hostengine) 2>/dev/null", timeout=5)
    time.sleep(2)

    for mod in ["nvidia_uvm", "nvidia_modeset", "nvidia_fs", "gdrdrv", "nvidia"]:
        _run(f"sudo rmmod {mod} 2>/dev/null", timeout=5)

    ok_reload, _ = _run("sudo modprobe nvidia NVreg_RestrictProfilingToAdminUsers=0")
    if ok_reload:
        _run("sudo modprobe nvidia_uvm 2>/dev/null")
        _run("sudo modprobe nvidia_modeset 2>/dev/null")

        ok2, out2 = _run("cat /proc/driver/nvidia/params")
        if ok2 and "RmProfilingAdminOnly: 0" in out2:
            print_success("  GPU counters now accessible — no reboot needed!")
            return True

    print_warning("  Live reload failed (GPU may be in use by another process).")
    console.print(
        "  [dim]The modprobe config is saved. A reboot will apply it:[/dim]\n"
        "  [dim]  sudo reboot[/dim]\n"
        "  [dim]After reboot, run: colonel setup  (to verify)[/dim]"
    )
    return False


# ---------------------------------------------------------------------------
# Step 6: LLM Provider
# ---------------------------------------------------------------------------

def _setup_api_keys() -> None:
    from colonel.config.settings import get_settings, save_config

    _step(6, "LLM Provider (AI Analysis)")
    settings = get_settings()

    console.print("  Colonel uses an LLM to analyze profiling data and give recommendations.")
    console.print("  Supported providers:\n")
    console.print("    [metric]a[/metric]) Anthropic Claude  — recommended, highest quality")
    console.print("    [metric]b[/metric]) NVIDIA Nemotron   — via build.nvidia.com (NIM API)")
    console.print("    [metric]c[/metric]) HuggingFace       — inference providers")
    console.print("    [metric]d[/metric]) Skip for now\n")

    choice = _ask("  Choose provider", "a").strip().lower()

    if choice in ("a", "anthropic"):
        _configure_provider_key(
            settings,
            provider="anthropic",
            current_key=settings.anthropic_api_key,
            label="Anthropic",
            key_config="anthropic_api_key",
            signup_hint="https://console.anthropic.com/settings/keys",
        )

    elif choice in ("b", "nvidia", "nemotron"):
        _configure_provider_key(
            settings,
            provider="nvidia",
            current_key=settings.nvidia_api_key,
            label="NVIDIA NIM",
            key_config="nvidia_api_key",
            signup_hint="https://build.nvidia.com/settings/api-keys",
        )

    elif choice in ("c", "huggingface", "hf"):
        _configure_provider_key(
            settings,
            provider="huggingface",
            current_key=settings.huggingface_api_key,
            label="HuggingFace",
            key_config="huggingface_api_key",
            signup_hint="https://huggingface.co/settings/tokens",
        )

    else:
        print_info("  Skipped. Configure later: colonel config set anthropic_api_key <key>")


def _configure_provider_key(
    settings: object,
    *,
    provider: str,
    current_key: str,
    label: str,
    key_config: str,
    signup_hint: str,
) -> None:
    from colonel.config.settings import save_config

    masked = f"{current_key[:8]}...{current_key[-4:]}" if len(current_key) > 12 else "(not set)"
    console.print(f"  Current {label} key: [dim]{masked}[/dim]")

    need_key = not current_key
    if current_key:
        need_key = _ask_yes(f"  Update {label} API key?", default=False)

    if need_key:
        console.print(f"  [dim]Get your key at: {signup_hint}[/dim]")
        key = _ask(f"  {label} API key")
        if key:
            save_config(key_config, key)
            save_config("llm_provider", provider)
            print_success(f"  {label} key saved. Provider set to '{provider}'.")
            return
        print_warning("  No key entered.")
        return

    # Key exists and user doesn't want to update — just ensure provider is set
    save_config("llm_provider", provider)
    print_success(f"  Provider set to '{provider}'.")


# ---------------------------------------------------------------------------
# Step 7: Smoke Test
# ---------------------------------------------------------------------------

def _run_smoke_test(nsys_path: str, ncu_path: str) -> None:
    _step(7, "Smoke Test")

    if not _ask_yes("  Run a quick GPU profiling test?"):
        print_info("  Skipped. You can test manually: colonel run python gpu_smoke/matmul.py")
        return

    # Resolve Python with GPU support
    python = sys.executable
    # Check for PyTorch
    ok_torch, torch_out = _run(f'"{python}" -c "import torch; print(torch.cuda.is_available())"')
    if not ok_torch or "True" not in torch_out:
        print_warning("  PyTorch with CUDA not available in the current environment.")
        console.print(
            "  [dim]Install: pip install torch[/dim]\n"
            "  [dim]Then re-run: colonel setup[/dim]"
        )
        return

    # Find the smoke test script
    smoke_script = _find_smoke_script()
    if not smoke_script:
        print_warning("  Smoke test script not found (gpu_smoke/matmul.py).")
        console.print("  [dim]You can test manually: colonel run python your_script.py[/dim]")
        return

    evaluator = "nsys" if nsys_path else ("ncu" if ncu_path else "")
    if not evaluator:
        print_warning("  No profiler found — cannot run smoke test.")
        return

    console.print(f"  Running [metric]{smoke_script}[/metric] with {evaluator}...")

    from colonel.cli.profile_cmd import _run_profile

    try:
        _run_profile(
            command=python,
            args=[smoke_script],
            target="local",
            evaluator=evaluator,
            name="setup-smoke-test",
            no_analyze=True,
            working_dir=".",
        )
        print_success("  Smoke test passed! Profiling pipeline is working.")
    except SystemExit:
        print_warning("  Smoke test encountered an issue.")
        console.print(
            f"  [dim]Try manually: colonel run {python} {smoke_script} --evaluator {evaluator}[/dim]"
        )
    except Exception as exc:
        print_warning(f"  Smoke test error: {exc}")


def _find_smoke_script() -> str:
    """Locate gpu_smoke/matmul.py relative to cwd or the colonel package."""
    candidates = [
        os.path.join(os.getcwd(), "gpu_smoke", "matmul.py"),
        os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
                     "gpu_smoke", "matmul.py"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return ""


# ---------------------------------------------------------------------------
# Main setup command
# ---------------------------------------------------------------------------

@setup_app.callback(invoke_without_command=True)
def setup(
    check_only: bool = typer.Option(
        False, "--check", help="Only run checks, skip interactive prompts."
    ),
) -> None:
    """Interactive setup wizard for Colonel.

    Checks your GPU environment, configures profiling tools, sets up
    API keys for AI analysis, and verifies the pipeline with a smoke test.

    Examples:
        colonel setup           # Full interactive wizard
        colonel setup --check   # Non-interactive environment check
    """
    print_header("Colonel Setup")
    console.print("Checking your GPU profiling environment...\n")

    results: dict[str, bool] = {}

    # ── Step 1: Driver ──────────────────────────────────────────────
    results["driver"] = _check_nvidia_driver()
    if not results["driver"]:
        console.print()
        print_error("NVIDIA driver is required. Install it and re-run: colonel setup")
        raise typer.Exit(1)

    # ── Step 2: CUDA ────────────────────────────────────────────────
    results["cuda"] = _check_cuda()

    # ── Step 3: nsys ────────────────────────────────────────────────
    nsys_ok, nsys_path = _check_nsys()
    results["nsys"] = nsys_ok

    # ── Step 4: ncu ─────────────────────────────────────────────────
    ncu_ok, ncu_path = _check_ncu()
    results["ncu"] = ncu_ok

    # ── Step 5: Permissions ─────────────────────────────────────────
    if ncu_ok:
        if check_only:
            ok, out = _run("cat /proc/driver/nvidia/params")
            results["permissions"] = ok and "RmProfilingAdminOnly: 0" in out
            _step(5, "GPU Performance Counter Permissions")
            if results["permissions"]:
                print_success("  GPU counters accessible to all users.")
            else:
                print_warning("  GPU counters restricted. Run colonel setup (without --check) to fix.")
        else:
            results["permissions"] = _check_ncu_permissions()
    else:
        results["permissions"] = True  # N/A when ncu not present

    # ── Step 6: API keys ────────────────────────────────────────────
    if check_only:
        from colonel.config.settings import get_settings

        _step(6, "LLM Provider (AI Analysis)")
        settings = get_settings()
        provider = settings.llm_provider
        has_key = bool(getattr(settings, f"{provider}_api_key", "")) if provider != "openai" else bool(settings.openai_api_key)
        if has_key:
            print_success(f"  Provider: {provider} (key configured)")
        else:
            print_warning(f"  Provider: {provider} (no API key set)")
        results["api_key"] = has_key
    else:
        _setup_api_keys()
        results["api_key"] = True  # user was prompted

    # ── Step 7: Smoke test ──────────────────────────────────────────
    if not check_only:
        _run_smoke_test(nsys_path if nsys_ok else "", ncu_path if ncu_ok else "")

    # ── Summary ─────────────────────────────────────────────────────
    console.print("\n")
    console.print("[bold]Setup Summary[/bold]")
    console.print("─" * 44)

    def _icon(ok: bool) -> str:
        return "[success]  OK  [/success]" if ok else "[error]MISSING[/error]"

    console.print(f"  NVIDIA Driver      {_icon(results['driver'])}")
    console.print(f"  CUDA Toolkit       {_icon(results['cuda'])}")
    console.print(f"  Nsight Systems     {_icon(results['nsys'])}")
    console.print(f"  Nsight Compute     {_icon(results['ncu'])}")
    console.print(f"  GPU Counters       {_icon(results['permissions'])}")
    console.print(f"  API Key            {_icon(results.get('api_key', False))}")

    all_ok = all(results.values())
    console.print()
    if all_ok:
        print_success("Everything looks good! Get started:")
        console.print("  [dim]colonel run python your_script.py          # profile + analyze[/dim]")
        console.print("  [dim]colonel session list                       # view past runs[/dim]")
        console.print("  [dim]colonel analyze --session <id> --deeper    # dig deeper[/dim]")
    elif results["driver"] and results["nsys"]:
        print_warning("Some optional items need attention, but core profiling will work.")
        console.print("  [dim]Run: colonel setup  (to fix remaining items)[/dim]")
    else:
        print_error("Required tools are missing. Fix the items above and re-run: colonel setup")
