#!/usr/bin/env bash
# EcoCompute AutoDL quick smoke test — does my setup work?
#
# Usage (on an AutoDL instance with a GPU):
#   bash autodl/quick_test.sh
#
# Runs a minimal NF4 measurement on TinyLlama-1.1B in < 5 minutes.
# Prints pass/fail and basic energy numbers.
#
# This is NOT a full measurement — use `bash autodl/run_all.sh` for that.

set -euo pipefail

# --- AutoDL network turbo (if available) ---
if [ -f /etc/network_turbo ]; then
  source /etc/network_turbo 2>/dev/null || true
fi
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

echo "============================================================"
echo "EcoCompute AutoDL quick smoke test"
echo "============================================================"

# --- Check GPU ---
if ! command -v nvidia-smi &>/dev/null; then
  echo "FAIL: nvidia-smi not found — no NVIDIA GPU?"
  exit 1
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | xargs)
IDLE_POWER=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits | head -1 | xargs)
echo "GPU: $GPU_NAME"
echo "Idle power: ${IDLE_POWER} W"

# --- Check Python + torch ---
PYTHON="${ECO_PYTHON:-python3}"
if ! "$PYTHON" -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "FAIL: torch not installed or CUDA not available"
  echo "  Run: bash autodl/00_setup.sh  (first time only)"
  exit 1
fi

TORCH_VER=$("$PYTHON" -c "import torch; print(torch.__version__)")
CUDA_VER=$("$PYTHON" -c "import torch; print(torch.version.cuda)")
echo "torch: $TORCH_VER (CUDA $CUDA_VER)"

# --- Check venv ---
VENV_DIR="${ECOCOMPUTE_VENV:-/root/autodl-tmp/ecocompute/venv}"
if [ ! -d "$VENV_DIR" ]; then
  echo "FAIL: venv not found at $VENV_DIR"
  echo "  Run: bash autodl/00_setup.sh  (first time only)"
  exit 1
fi
source "$VENV_DIR/bin/activate"

# --- Check required packages ---
for pkg in transformers bitsandbytes pynvml; do
  if ! "$PYTHON" -c "import $pkg" 2>/dev/null; then
    echo "FAIL: $pkg not installed"
    echo "  Run: bash autodl/00_setup.sh  (first time only)"
    exit 1
  fi
done

# --- Run quick_test.py ---
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

echo ""
echo "Running quick_test.py..."
echo ""

cd "$REPO_DIR"
"$PYTHON" quick_test.py
EXIT_CODE=$?

if [ $EXIT_CODE -eq 0 ]; then
  echo ""
  echo "============================================================"
  echo "PASS — your AutoDL setup works for EcoCompute measurements"
  echo "============================================================"
  echo ""
  echo "Next steps:"
  echo "  - Full measurement sweep:  bash autodl/run_all.sh"
  echo "  - Or submit this result:   cat ecocompute-out/energy.json"
  echo "    → paste into https://github.com/hongping-zh/ecocompute-mlcube/issues/new"
else
  echo ""
  echo "============================================================"
  echo "FAIL — see error above"
  echo "============================================================"
fi

exit $EXIT_CODE
