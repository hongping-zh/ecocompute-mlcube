#!/usr/bin/env bash
# ===========================================================================
# 00_setup.sh — one-time environment setup on an AutoDL RTX 4090D instance.
#
#   bash autodl/00_setup.sh              # set up env + pre-download models
#   SKIP_DOWNLOAD=1 bash autodl/00_setup.sh   # env only, download lazily later
#
# What it does:
#   1. enables AutoDL academic acceleration + the hf-mirror.com HF endpoint,
#   2. checks the NVIDIA driver is visible (nvidia-smi),
#   3. creates a venv on the data disk that REUSES AutoDL's preinstalled,
#      CUDA-matched PyTorch (via --system-site-packages) so we don't fight the
#      driver; installs torch from the cu121 wheel index only if none is found,
#   4. installs transformers / accelerate / bitsandbytes / nvidia-ml-py etc.,
#   5. (optional) pre-downloads every model in autodl/models.txt into the
#      HF cache on the data disk.
# ===========================================================================
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/env.sh"

eco_enable_turbo
eco_print_config

echo
echo "=== [1/5] NVIDIA driver ==================================================="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || nvidia-smi
else
  echo "!! nvidia-smi not found. On AutoDL this means the instance has no GPU"
  echo "!! attached, or the driver isn't loaded. A real measurement needs a GPU;"
  echo "!! you can still run everything in --dry_run mode (dataset reference)."
fi

echo
echo "=== [2/5] Python venv (data disk) ========================================"
PY="${ECO_PYTHON:-python3}"
if [ ! -d "$VENV_DIR" ]; then
  # --system-site-packages: reuse AutoDL's preinstalled CUDA PyTorch.
  "$PY" -m venv --system-site-packages "$VENV_DIR"
  echo "[setup] created venv at $VENV_DIR"
else
  echo "[setup] reusing existing venv at $VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools

echo
echo "=== [3/5] PyTorch (CUDA) ================================================="
if python -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  echo "[setup] usable CUDA PyTorch already present:"
  python -c "import torch; print('   torch', torch.__version__, 'cuda', torch.version.cuda, 'device', torch.cuda.get_device_name(0))"
else
  echo "[setup] no CUDA-enabled torch found — installing cu121 torch"
  # Only torch is needed (entrypoint.py doesn't use torchvision/torchaudio);
  # skipping them saves ~1GB of disk and download time.
  python -m pip install --index-url https://download.pytorch.org/whl/cu121 torch
fi

echo
echo "=== [4/5] Runtime deps ==================================================="
# transformers/accelerate/bitsandbytes track the installed torch. nvidia-ml-py
# gives NVML power sampling; hf_transfer speeds up mirror downloads.
python -m pip install \
  "transformers>=4.44" \
  "accelerate>=0.33" \
  "bitsandbytes>=0.43.3" \
  "nvidia-ml-py>=12.560" \
  "sentencepiece>=0.2.0" \
  "pyyaml>=6.0.1" \
  "hf_transfer>=0.1.6" \
  "huggingface_hub[cli]>=0.24"
echo "[setup] installed:"
python - <<'PY'
import importlib.metadata as m
for p in ("torch","transformers","accelerate","bitsandbytes","nvidia-ml-py","sentencepiece"):
    try: print(f"   {p:16s} {m.version(p)}")
    except Exception as e: print(f"   {p:16s} MISSING ({e})")
PY

echo
echo "=== [5/5] Pre-download models ==========================================="
if [ "${SKIP_DOWNLOAD:-0}" = "1" ]; then
  echo "[setup] SKIP_DOWNLOAD=1 — models will download on first use."
else
  while read -r model params _rest; do
    case "$model" in ""|\#*) continue;; esac
    echo "[setup] fetching $model -> $HF_HOME"
    huggingface-cli download "$model" \
      --exclude "*.pth" "*.onnx" "*.gguf" "*.msgpack" "*.h5" \
      >/dev/null 2>&1 && echo "   ok $model" \
      || echo "   !! download failed for $model (will retry lazily at run time)"
  done < "$MODELS_FILE"
fi

echo
echo "[setup] DONE. Next:  bash autodl/01_verify.sh"
