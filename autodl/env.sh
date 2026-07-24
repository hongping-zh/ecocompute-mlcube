# shellcheck shell=bash
# ---------------------------------------------------------------------------
# EcoCompute MLCube — shared environment for AutoDL RTX 4090D runs.
#
# This file is *sourced* by the other autodl/*.sh scripts (and can be sourced
# by you interactively). It sets:
#   - repo / venv / cache / results paths (kept on the AutoDL data disk),
#   - the RTX 4090D architecture key expected by entrypoint.py (`ada`),
#   - a Hugging Face mirror + academic acceleration so weight downloads work
#     from inside China,
#   - sane sweep defaults (tokens / iterations / warmup / sample rate).
#
# Everything here can be overridden from the environment, e.g.
#   TOKENS=512 ITERATIONS=20 bash autodl/02_run_sweep.sh
# ---------------------------------------------------------------------------

# --- locate the repo root (this file lives in <repo>/autodl) ----------------
_ECO_ENV_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
export REPO_DIR="${REPO_DIR:-$(cd "$_ECO_ENV_DIR/.." && pwd)}"

# --- pick a base dir: AutoDL data disk if present, else the repo ------------
# On AutoDL the system disk (/root) is small; /root/autodl-tmp is the big data
# disk. Keep the venv, HF cache and results there so you don't fill the OS disk.
if [ -z "${ECO_BASE:-}" ]; then
  if [ -d /root/autodl-tmp ]; then
    ECO_BASE=/root/autodl-tmp/ecocompute
  else
    ECO_BASE="$REPO_DIR/.autodl"
  fi
fi
export ECO_BASE
mkdir -p "$ECO_BASE"

# --- keep pip cache + temp on the data disk ---------------------------------
# AutoDL's system disk (/root) is tiny and often nearly full; the default pip
# cache (~/.cache/pip) and TMPDIR live there and overflow it while installing
# large wheels (torch unpacks to several GB in TMPDIR). Force ALL of these onto
# the data disk. Note: forced (not `:-`) because AutoDL frequently pre-sets
# TMPDIR=/tmp on the system disk, which is exactly what overflows.
export PIP_CACHE_DIR="$ECO_BASE/pip-cache"
export TMPDIR="$ECO_BASE/tmp"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"

# --- python venv + hugging face cache + results ----------------------------
export VENV_DIR="${VENV_DIR:-$ECO_BASE/venv}"
export HF_HOME="${HF_HOME:-$ECO_BASE/hf}"                 # HF datasets/hub cache
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export RESULTS_DIR="${RESULTS_DIR:-$REPO_DIR/autodl/results}"
mkdir -p "$HF_HOME" "$RESULTS_DIR"

# --- Hugging Face access from China ----------------------------------------
# huggingface.co is often unreachable on AutoDL; hf-mirror.com is a full mirror.
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

# --- run configuration ------------------------------------------------------
# RTX 4090D is Ada Lovelace -> entrypoint.py arch key `ada`.
export GPU_ARCH="${GPU_ARCH:-ada}"
export GPU_LABEL="${GPU_LABEL:-RTX 4090D}"

# Measurement controls (match the published methodology: 10 Hz, 256 tok, 10 it).
export TOKENS="${TOKENS:-256}"
export ITERATIONS="${ITERATIONS:-10}"
export WARMUP="${WARMUP:-2}"
export SAMPLE_RATE_HZ="${SAMPLE_RATE_HZ:-10}"
export BATCH_SIZE="${BATCH_SIZE:-1}"

# Precisions to sweep for each model (space separated). INT8 on Ada is a *new*
# contribution vs the published dataset (which only has INT8 on A800).
export PRECISIONS="${PRECISIONS:-FP16 NF4 INT8}"

# Default model sweep matrix (used if autodl/models.txt is absent). Each line:
#   <hf_model_id> <params_b>
# Kept in one place so setup can pre-download and the sweep can iterate it.
export MODELS_FILE="${MODELS_FILE:-$REPO_DIR/autodl/models.txt}"

# --- academic acceleration (AutoDL) ----------------------------------------
# Speeds up github.com / huggingface / pytorch downloads. Safe no-op elsewhere.
eco_enable_turbo() {
  if [ -f /etc/network_turbo ]; then
    # shellcheck disable=SC1091
    source /etc/network_turbo || true
    echo "[env] AutoDL academic acceleration enabled (/etc/network_turbo)"
  fi
}

# --- activate the venv if it exists -----------------------------------------
eco_activate() {
  if [ -f "$VENV_DIR/bin/activate" ]; then
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
  fi
}

eco_print_config() {
  cat <<CFG
[env] REPO_DIR      = $REPO_DIR
[env] ECO_BASE      = $ECO_BASE
[env] VENV_DIR      = $VENV_DIR
[env] HF_HOME       = $HF_HOME
[env] PIP_CACHE_DIR = $PIP_CACHE_DIR
[env] TMPDIR        = $TMPDIR
[env] HF_ENDPOINT   = $HF_ENDPOINT
[env] RESULTS_DIR   = $RESULTS_DIR
[env] GPU_ARCH      = $GPU_ARCH ($GPU_LABEL)
[env] PRECISIONS    = $PRECISIONS
[env] TOKENS/ITER/WARMUP/HZ = $TOKENS/$ITERATIONS/$WARMUP/$SAMPLE_RATE_HZ
[env] MODELS_FILE   = $MODELS_FILE
CFG
}
