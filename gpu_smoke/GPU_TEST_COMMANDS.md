# Colonel GPU test – direct commands

Use these from the **colonel repo root** (or set `CWD` and use `--cwd`).

**Phase 2 requires an NVIDIA GPU and at least one of:** `nsys` (Nsight Systems) or `ncu` (Nsight Compute). There is **no pip package** for nsys/ncu—they are NVIDIA binaries installed via apt, the CUDA toolkit, or NVIDIA's .deb/.run installers. If `colonel profile detect` says "No GPU profiling tools found", install as below. Colonel looks for nsys in PATH and in common install locations (`/usr/lib/nsight-systems/bin/`, `/opt/nvidia/nsight-systems/*/bin/`, etc.).

**Troubleshooting: `undefined symbol: __libc_dlclose, version GLIBC_PRIVAT`**  
The Ubuntu apt package (nsight-systems 2021.x) is built against an older glibc. On newer systems you can get this symbol error when nsys injects into the target process. **Fix:** use a newer Nsight Systems from NVIDIA (Option B – CLI .deb or .run from [Get Started](https://developer.nvidia.com/nsight-systems/get-started)) so the binaries match your system's glibc. After installing the newer package, ensure `nsys` in your PATH points to it (or set `COLONEL_NSYS_PATH` to the new binary).

## Installing Nsight Systems from the terminal (SSH)

1. **Try apt first** (Ubuntu 22.04/24.04, needs sudo):
   ```bash
   sudo apt update
   sudo apt install -y nsight-systems
   colonel profile detect
   ```
   If apt says *"Package 'nsight-systems' has no installation candidate"* and lists versions like `nsight-systems-2025.5.2`, install one explicitly (newest is usually best):
   ```bash
   sudo apt install -y nsight-systems-2025.5.2
   colonel profile detect
   ```
   If that works, you're done.

2. **If nsight-systems isn't in apt, download the CLI .deb**  
   On the machine (x86_64 Linux), download and install the CLI-only package (smaller, no GUI). Filename: `NsightSystems-linux-cli-public-2026.1.1.204-3717666.deb`.

   If you already have the file on the machine:
   ```bash
   sudo dpkg -i /path/to/NsightSystems-linux-cli-public-2026.1.1.204-3717666.deb
   sudo apt-get install -f
   colonel profile detect
   ```

   If downloading (may require browser login for NVIDIA):
   ```bash
   cd /tmp
   wget "https://developer.nvidia.com/downloads/assets/tools/secure/nsight-systems/2026_1/NsightSystems-linux-cli-public-2026.1.1.204-3717666.deb" -O nsight-systems-cli.deb
   sudo dpkg -i nsight-systems-cli.deb
   sudo apt-get install -f
   colonel profile detect
   ```
   If wget returns 403 or an HTML page, download the .deb in a browser from Nsight Systems – Get Started, then copy to the machine (e.g. `scp`) and install as above.

3. **If CUDA is already installed but nsys isn't in PATH:**
   ```bash
   export PATH="/usr/local/cuda/bin:$PATH"
   colonel profile detect
   ```
   Add the same export to `~/.bashrc` if you want it in every session.

## Common environment issues

**nsys: Permission denied on `/opt/nvidia/nsight-systems`**  
The apt `nsight-systems-2025.x` package installs under `/opt/nvidia/nsight-systems/` with `0700 root:root` permissions. Non-root users can't run the binary even though `update-alternatives` creates `/usr/local/bin/nsys`. Fix:
```bash
sudo chmod 755 /opt/nvidia/nsight-systems
sudo chmod -R a+rX /opt/nvidia/nsight-systems/2025.5.2
```

**ncu: `ERR_NVGPUCTRPERM` (GPU performance counter access denied)**  
ncu needs access to hardware performance counters. On most systems this requires either root or a kernel module parameter change:
```bash
# Persistent fix (requires reboot):
echo 'options nvidia NVreg_RestrictProfilingToAdminUsers=0' | sudo tee /etc/modprobe.d/ncu-perms.conf
sudo reboot
```
Colonel will detect this error and show the fix instructions.

## Phase 1: Sanity (no workload)

```bash
cd /home/ubuntu/colonel
source .venv/bin/activate   # if using a venv
pip install -e .
pip install torch numpy    # for gpu_smoke/matmul.py (PyTorch needs numpy)

colonel profile detect
colonel config show
colonel session list
```

## Phase 2: Minimal GPU run

**NSys (recommended first):**
```bash
colonel run python gpu_smoke/matmul.py --name matmul-nsys --evaluator nsys --no-analyze
```

**NCU (same script):**
```bash
colonel run python gpu_smoke/matmul.py --name matmul-ncu --evaluator ncu --no-analyze
```

**Using a specific working dir:**
```bash
colonel run python matmul.py --name matmul-nsys --evaluator nsys --no-analyze --cwd /home/ubuntu/colonel/gpu_smoke
```

**After runs:**
```bash
colonel session list
colonel session show <id>
colonel session compare <id1> <id2>
```

**One-liner smoke:**
```bash
cd /home/ubuntu/colonel && colonel run python gpu_smoke/matmul.py --name smoke --evaluator nsys --no-analyze
```

All of this assumes an NVIDIA GPU, drivers, and nsys/ncu on PATH (or installed where Colonel looks). If PyTorch isn't desired, matmul.py will try CuPy when torch is not installed.
