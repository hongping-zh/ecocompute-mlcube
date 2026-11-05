# Quick Start Guide

Complete setup instructions for running EcoCompute energy measurements.

---

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Running your first measurement](#running-your-first-measurement)
- [Docker vs native mode](#docker-vs-native-mode)
- [Common configurations](#common-configurations)
- [Running on rented GPUs (AutoDL, vast.ai)](#running-on-rented-gpus)
- [Running from source](#running-from-source)
- [Understanding the output](#understanding-the-output)
- [Comparing with published data](#comparing-with-published-data)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

**Hardware:**
- NVIDIA GPU with NVML support (Turing, Ampere, Ada, Hopper, Blackwell architectures)
- Minimum 8 GB VRAM for 1.1B models, 24 GB for 7B models

**Software:**
- Docker (recommended) OR Python 3.9+ with CUDA toolkit
- ~10 GB disk space for container image
- ~2 GB for model download (first run only)

**Network:**
- Access to Hugging Face (for model download)
- Access to GitHub Container Registry (for Docker image)
- Optional: access to quantenergy.tech (for curve overlay)

---

## Installation

### Method 1: Quickstart script (recommended)

Download the script from an immutable release, verify checksum, then run:

```bash
# Download
curl -fsSLO https://github.com/hongping-zh/ecocompute-mlcube/releases/download/schema-1.3-r5/quickstart.sh
curl -fsSLO https://github.com/hongping-zh/ecocompute-mlcube/releases/download/schema-1.3-r5/SHA256SUMS

# Verify
sha256sum -c SHA256SUMS

# Preview what will happen (no execution)
bash quickstart.sh --dry-run

# Run
bash quickstart.sh
```

**Supply-chain properties:**
- Script and `SHA256SUMS` are release assets (immutable once published)
- Container image published with digest (no `:latest` drift)
- SBOM (CycloneDX) ships as release asset
- Container runs as non-root user (`ecocompute`)
- `ECOCOMPUTE_READONLY=1` adds `--read-only` + tmpfs `/tmp`

### Method 2: Docker directly

```bash
mkdir -p out
docker run --rm --gpus all --user "$(id -u):$(id -g)" \
  -v "$PWD/out:/workspace/outputs" \
  ghcr.io/hongping-zh/ecocompute-mlcube:latest energy_estimate \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --params_b 1.1 \
  --precision NF4 --gpu_arch auto --output_dir /workspace/outputs --share
```

### Method 3: MLCube CLI

```bash
pip install mlcube mlcube-docker
mlcube run --mlcube=. --task=energy_estimate --platform=docker
```

---

## Running your first measurement

The quickstart measures **NF4 and its own FP16 baseline** on TinyLlama-1.1B:

1. Loads the model (~2 GB download, cached for subsequent runs)
2. Runs FP16 baseline: 10 decode iterations after 2 warmups
3. Runs NF4 quantized: same protocol
4. Scores perplexity on fixed text (~10.6k words, vendored in `quality/`)
5. Writes `ecocompute-out/energy.json`
6. Prints a link to overlay your point on the published curve

**What you get:**
```json
{
  "total_energy_j": 12.34,
  "tokens": 2560,
  "energy_per_token_mj": 4.82,
  "avg_power_watts": 296,
  "throughput_tokens_per_s": 155.9,
  "vs_fp16_energy_pct": -61.9,
  "basis": "measured",
  "measurement_source": "direct-nvml",
  "quality": {
    "perplexity_fp16": 7.3260,
    "perplexity_quantized": 7.7364,
    "delta_vs_fp16_pct": 5.60
  }
}
```

**Time:**
- First run: ~57 minutes (downloads dominate: ~2 GB model + ~3 GB torch/CUDA wheels)
- Repeat runs: ~2 minutes (all cached)

---

## Docker vs native mode

The quickstart script auto-detects which mode to use:

| Mode | When | What it does |
|---|---|---|
| **docker** | Docker daemon responds | Pulls `ghcr.io/…/ecocompute-mlcube:latest`, runs with `--gpus all`; nothing installed on host |
| **native** | No usable Docker (rented GPU instances are themselves containers) | Clones repo, installs into venv, reuses host's CUDA PyTorch; keeps checkout/venv/model cache on data disk |

**Force a mode:**
```bash
ECOCOMPUTE_MODE=docker bash quickstart.sh
ECOCOMPUTE_MODE=native bash quickstart.sh
```

Both paths run the same `entrypoint.py`, same pins from `requirements.txt`, same 10 decode iterations after 2 warmups.

---

## Common configurations

### Change model and precision

```bash
ECOCOMPUTE_PRECISION=INT8 \
ECOCOMPUTE_MODEL=Qwen/Qwen2.5-3B ECOCOMPUTE_PARAMS_B=3 \
  bash quickstart.sh
```

### Prefetch prediction from website

```bash
ECOCOMPUTE_PREFETCH=1 bash quickstart.sh
```

Before measuring, asks the site estimator (`/v1/estimate`) for predicted energy delta vs FP16 and prints it, so you can eyeball prediction vs. your measured result. Offline/timeout → silently skipped.

### Skip perplexity probe

```bash
ECOCOMPUTE_QUALITY=0 bash quickstart.sh
```

### Enable power-trace sidecar (schema 1.3)

```bash
python3 entrypoint.py energy_estimate \
    --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --precision NF4 --gpu_arch auto --params_b 1.1 \
    --output_dir workspace/outputs --power_trace --share
```

Records raw NVML samples of the whole run (load, quantization, warmup, generation, quality probe) to `power_trace.csv`. The reported `total_energy_joules` is unchanged; the sidecar lets you re-integrate over a different window post hoc.

---

## Running on rented GPUs

### AutoDL

AutoDL instances are themselves containers and cannot nest Docker. Use native mode:

```bash
# Setup (creates venv, installs deps)
bash autodl/00_setup.sh

# Run measurement
bash quickstart.sh
```

**Python version:** AutoDL's default Python 3.8 is too old. Point `ECO_PYTHON` at a ≥3.9 interpreter:

```bash
rm -rf /root/autodl-tmp/ecocompute/venv
ECO_PYTHON=/root/miniconda3/envs/eco310/bin/python \
  SKIP_DOWNLOAD=1 bash autodl/00_setup.sh
```

**bitsandbytes CUDA mismatch:** If torch is from a newer CUDA line, the pinned bitsandbytes looks for a `libbitsandbytes_cudaXXX.so` it doesn't ship. Setup runs a real 4-bit kernel as preflight check, and on failure upgrades bitsandbytes (one small wheel from the index you're already using). If that doesn't work, it falls back to reinstalling torch from the cu121 index (~2.5 GB).

The upgrade leaves the published pins, so the report carries `software.differs_from_reference_pins` and `vs_fp16` is not directly comparable with image runs.

**Model download location:** Set `HF_HOME` to avoid filling the system disk:

```bash
export HF_HOME=/root/autodl-tmp/ecocompute/hf
```

### vast.ai

Same as AutoDL: use native mode, ensure Python ≥3.9, set `HF_HOME` to data disk.

---

## Running from source

No GPU needed for a smoke test — the container falls back to a dataset-derived reference report (clearly flagged, never a fabricated measurement).

```bash
git clone https://github.com/hongping-zh/ecocompute-mlcube.git
cd ecocompute-mlcube

# 1. Run the task directly (writes workspace/outputs/energy.json)
python3 entrypoint.py energy_estimate --dry_run \
    --parameters_file workspace/parameters/energy_params.yaml \
    --output_dir workspace/outputs
cat workspace/outputs/energy.json

# 2. Or via the official MLCube CLI + Docker (no GPU)
pip install mlcube mlcube-docker
mlcube run --mlcube=mlcube.cpu.yaml --task=energy_estimate --platform=docker

# 3. Real measurement on an NVIDIA GPU
mlcube run --mlcube=. --task=energy_estimate --platform=docker
```

**AMD GPUs (ROCm):** On-device power telemetry is NVML-based and therefore NVIDIA-only. On a ROCm host the entrypoint detects the AMD device, says so clearly, and reports energy values as `unavailable` rather than borrowing an NVIDIA fitted curve. Native ROCm power telemetry (rocm-smi / rocprof) is tracked in [#27](https://github.com/hongping-zh/ecocompute-mlcube/issues/27).

A ROCm build of PyTorch on NVIDIA hardware (the hip-nvcc case) is still measured: the entrypoint cross-checks the NVML device name against torch's device before declaring an AMD host.

---

## Understanding the output

### Key fields in `energy.json`

| Field | Meaning |
|---|---|
| `total_energy_j` | Total joules for the measurement window (decode iterations only) |
| `tokens` | Total tokens generated (2560 by default: 256 tokens × 10 iterations) |
| `energy_per_token_mj` | Millijoules per 1000 tokens |
| `avg_power_watts` | Average power during decode |
| `throughput_tokens_per_s` | Tokens per second |
| `vs_fp16_energy_pct` | Energy delta vs FP16 baseline (negative = quantization saves energy) |
| `basis` | `measured` / `interpolated` / `extrapolated` |
| `measurement_source` | `direct-nvml` / `ecocompute-dataset` (no-GPU fallback) |
| `quality.perplexity_fp16` | Perplexity of FP16 baseline |
| `quality.perplexity_quantized` | Perplexity of quantized model |
| `quality.delta_vs_fp16_pct` | Perplexity delta (positive = quantization hurts quality) |
| `software.matches_reference_pins` | Whether versions match the published pins |

### What "basis" means

- **`measured`**: Real hardware measurement via NVML
- **`interpolated`**: Derived from published dataset (no-GPU fallback)
- **`extrapolated`**: Outside the range of published data

Only `basis: measured` reports can be compared with the published crossover curves.

### What "measurement_source" means

- **`direct-nvml`**: On-device power telemetry
- **`ecocompute-dataset (no local GPU)`**: No NVIDIA GPU present, fell back to dataset
- **`ecocompute-dataset (on-device measurement failed)`**: GPU present but NVML failed

---

## Comparing with published data

### Online overlay

After a measured run, the quickstart prints a link like:

```
https://quantenergy.tech/?tab=run&overlay=<base64url>
```

Opening it restores your marker on the crossover curve entirely in the browser. **Nothing is uploaded and no server stores anything** — the overlay point is encoded in the URL itself.

### Regression check

Grade your measurement against the published fit:

```bash
python3 tools/check_regression.py ecocompute-out/energy.json
# Exit codes: 0 = within the band, 1 = outside it, 2 = not gradeable
```

The band is `k * resid_std` (default `k=2`), i.e. the fit's own residual dispersion; it is **not** a confidence interval.

**Deliberate limits:**
- Refuses to grade a report whose `basis` is not `measured` (checking an interpolated report against the curve is circular)
- Cannot run in CI (GitHub-hosted runners have no NVIDIA GPU)

A pass means "this build still measures what it used to", not "this confirms the curve": a single report is n=1, and a version mismatch with the published pins (printed as a caveat) accounts for part of any residual.

### Submit your replication

Publish your result (agreeing or disagreeing) at [quantenergy.tech/replications](https://quantenergy.tech/replications/#submit).

---

## Troubleshooting

### "No kernel image is available"

Your torch version doesn't have kernels for your GPU architecture. Common on RTX 5090 (Blackwell, sm_120) with older torch.

**Fix:** Install torch with CUDA 12.8+:

```bash
pip install torch --force-reinstall --index-url https://download.pytorch.org/whl/cu128
```

### "libbitsandbytes_cudaXXX.so not found"

bitsandbytes version doesn't match your torch's CUDA line.

**Fix:** Upgrade bitsandbytes:

```bash
pip install bitsandbytes --upgrade
```

The quickstart does this automatically as a preflight check.

### Model download fails or is slow

Hugging Face hub can be slow in some regions.

**Fix:** Set `HF_HOME` to a fast disk and use a mirror:

```bash
export HF_HOME=/path/to/fast/disk/hf
export HF_ENDPOINT=https://hf-mirror.com
```

### Out of memory (OOM)

7B models need ~24 GB VRAM in FP16, ~4 GB in NF4.

**Fix:** Use a smaller model or enable CPU offloading:

```bash
ECOCOMPUTE_MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0 bash quickstart.sh
```

### NVML power telemetry unavailable

Some consumer GPUs or vGPU configurations don't expose power reads.

**Fix:** The container automatically falls back to the dataset path with `basis != "measured"`. A single dropped power read never aborts a run (see `results.dropped_samples`).

### Report says "not directly comparable with image runs"

Your run used different package versions than the published pins (common on rented GPUs with lagging mirrors).

**What it means:** Your measurement is still real (`basis: measured`), but `vs_fp16_energy_pct` cannot be compared with the published curves without accounting for the version difference.

**What to do:** Check `software.differs_from_reference_pins` in the report to see which packages differ.

---

## Next steps

- **[TECHNICAL.md](TECHNICAL.md)** — JSON schema, validation, protocol conformance, reproducible builds
- **[docs/backends.md](docs/backends.md)** — How to add llama.cpp, vLLM, or TensorRT-LLM as inference engines
- **[quantenergy.tech](https://quantenergy.tech)** — Interactive crossover curves and estimator API
- **[Ask Watt](https://quantenergy.tech/watt/)** — Get a decision, not a measurement
