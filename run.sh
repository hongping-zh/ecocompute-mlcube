#!/usr/bin/env bash
# EcoCompute one-liner wrapper.
#
# Usage:
#   curl -sSL https://quantenergy.tech/run.sh | bash -s -- --model qwen2.5-3b --precision nf4
#
# This script is a thin parameter-mapping layer over the real quickstart.sh
# from the ecocompute-mlcube release. It:
#   1. Parses --model and --precision
#   2. Maps model shorthand to HuggingFace model ID + params_b
#   3. Downloads quickstart.sh + SHA256SUMS from a pinned release
#   4. Verifies the checksum
#   5. Executes quickstart.sh with the right environment variables
#
# The pinned release is immutable — once published, its assets never change.
# Run with --dry-run to see what would happen without executing anything.
set -euo pipefail

RELEASE="schema-1.3-r4"
BASE_URL="https://github.com/hongping-zh/ecocompute-mlcube/releases/download/${RELEASE}"

# ----------------------------------------------------------------- defaults --
MODEL_SHORT=""
PRECISION="NF4"
DRY_RUN=0

# --------------------------------------------------------------- model map --
# Maps shorthand names to "hf_id params_b". Add new models here.
declare -A MODEL_MAP=(
  ["tinyllama-1.1b"]="TinyLlama/TinyLlama-1.1B-Chat-v1.0 1.1"
  ["tinyllama"]="TinyLlama/TinyLlama-1.1B-Chat-v1.0 1.1"
  ["qwen2-1.5b"]="Qwen/Qwen2-1.5B-Instruct 1.5"
  ["qwen2.5-1.5b"]="Qwen/Qwen2.5-1.5B-Instruct 1.5"
  ["qwen2.5-3b"]="Qwen/Qwen2.5-3B-Instruct 3"
  ["qwen2.5-3b-instruct"]="Qwen/Qwen2.5-3B-Instruct 3"
  ["qwen2.5-7b"]="Qwen/Qwen2.5-7B-Instruct 7"
  ["phi-3-mini"]="microsoft/Phi-3-mini-4k-instruct 3.8"
  ["phi-3"]="microsoft/Phi-3-mini-4k-instruct 3.8"
  ["yi-1.5-6b"]="01-ai/Yi-1.5-6B-Chat 6"
  ["yi-6b"]="01-ai/Yi-1.5-6B-Chat 6"
  ["mistral-7b"]="mistralai/Mistral-7B-Instruct-v0.2 7"
  ["mistral"]="mistralai/Mistral-7B-Instruct-v0.2 7"
)

# ------------------------------------------------------------ argument parse --
show_help() {
  cat <<'EOF'
EcoCompute one-liner: measure LLM quantization energy on your GPU.

Usage:
  curl -sSL https://quantenergy.tech/run.sh | bash -s -- [OPTIONS]

Options:
  --model NAME        Model to measure (default: tinyllama-1.1b)
                      Known models: tinyllama-1.1b, qwen2.5-3b, qwen2.5-7b,
                      phi-3-mini, yi-1.5-6b, mistral-7b
                      Or pass a full HuggingFace ID with --model-id.
  --model-id ID       Full HuggingFace model ID (e.g. Qwen/Qwen2.5-3B-Instruct)
  --params-b N        Model size in billions (required with --model-id)
  --precision MODE    NF4 or INT8 (default: NF4; FP16 is always measured too)
  --dry-run           Print what would happen, execute nothing
  --help              Show this message

Examples:
  # Measure Qwen2.5-3B with NF4 (the crossover region)
  curl -sSL https://quantenergy.tech/run.sh | bash -s -- --model qwen2.5-3b --precision nf4

  # Measure a custom model
  curl -sSL https://quantenergy.tech/run.sh | bash -s -- \
    --model-id Qwen/Qwen2.5-7B-Instruct --params-b 7 --precision nf4

  # See what would happen without running
  curl -sSL https://quantenergy.tech/run.sh | bash -s -- --model qwen2.5-3b --dry-run

What it does:
  Downloads the ecocompute-mlcube container (or installs natively if Docker is
  unavailable), measures the energy of one (GPU x model x precision) config
  against its own FP16 baseline via NVML power sampling, and writes a
  schema-validated energy.json. The whole run takes ~2 minutes with warm caches.

  Output goes to ./ecocompute-out/energy.json. Nothing is uploaded anywhere.
EOF
  exit 0
}

MODEL_ID=""
PARAMS_B=""

while [ $# -gt 0 ]; do
  case "$1" in
    --model)
      [ $# -ge 2 ] || { echo "[error] --model requires a value" >&2; exit 1; }
      MODEL_SHORT="${2,,}"  # lowercase
      shift 2 ;;
    --model-id)
      [ $# -ge 2 ] || { echo "[error] --model-id requires a value" >&2; exit 1; }
      MODEL_ID="$2"
      shift 2 ;;
    --params-b)
      [ $# -ge 2 ] || { echo "[error] --params-b requires a value" >&2; exit 1; }
      PARAMS_B="$2"
      shift 2 ;;
    --precision)
      [ $# -ge 2 ] || { echo "[error] --precision requires a value" >&2; exit 1; }
      PRECISION="${2^^}"  # uppercase
      shift 2 ;;
    --dry-run|--dry_run)
      DRY_RUN=1
      shift ;;
    -h|--help)
      show_help ;;
    *)
      echo "[error] unknown option: $1 (use --help for usage)" >&2
      exit 1 ;;
  esac
done

# --------------------------------------------------------- resolve model id --
if [ -n "$MODEL_ID" ]; then
  # Explicit model ID provided
  [ -n "$PARAMS_B" ] || { echo "[error] --model-id requires --params-b" >&2; exit 1; }
elif [ -n "$MODEL_SHORT" ]; then
  # Look up shorthand
  if [ -n "${MODEL_MAP[$MODEL_SHORT]+x}" ]; then
    read -r MODEL_ID PARAMS_B <<< "${MODEL_MAP[$MODEL_SHORT]}"
  else
    # Maybe they passed a full HF ID directly via --model
    if [[ "$MODEL_SHORT" == *"/"* ]]; then
      echo "[error] --model '$MODEL_SHORT' looks like a full HuggingFace ID. Use --model-id instead, and provide --params-b." >&2
      exit 1
    fi
    echo "[error] unknown model '$MODEL_SHORT'. Known models:" >&2
    for k in "${!MODEL_MAP[@]}"; do echo "  $k" >&2; done | sort >&2
    echo "Or use --model-id with a full HuggingFace ID + --params-b." >&2
    exit 1
  fi
else
  # Default
  MODEL_ID="TinyLlama/TinyLlama-1.1B-Chat-v1.0"
  PARAMS_B="1.1"
fi

# Validate precision
case "$PRECISION" in
  NF4|INT8) ;;
  *)
    echo "[error] --precision must be NF4 or INT8 (got '$PRECISION')" >&2
    exit 1 ;;
esac

# ------------------------------------------------------------------ banner --
echo ""
echo "  EcoCompute energy measurement"
echo "  ─────────────────────────────"
echo "  Model:     $MODEL_ID"
echo "  Size:      ${PARAMS_B}B parameters"
echo "  Precision: $PRECISION (plus FP16 baseline)"
echo "  Release:   $RELEASE (pinned, immutable)"
echo ""

if [ "$DRY_RUN" = "1" ]; then
  echo "  [dry-run] Would download quickstart.sh from:"
  echo "    $BASE_URL/quickstart.sh"
  echo "  [dry-run] Would verify checksum against:"
  echo "    $BASE_URL/SHA256SUMS"
  echo "  [dry-run] Would execute:"
  echo "    ECOCOMPUTE_MODEL='$MODEL_ID' ECOCOMPUTE_PARAMS_B='$PARAMS_B' \\"
  echo "    ECOCOMPUTE_PRECISION='$PRECISION' bash quickstart.sh"
  echo ""
  echo "  [dry-run] Exiting without executing anything."
  exit 0
fi

# --------------------------------------------------------- download + verify --
TMPDIR=$(mktemp -d)
trap 'rm -rf "$TMPDIR"' EXIT

echo "  Downloading quickstart.sh from release $RELEASE ..."
curl -fsSLO --output-dir "$TMPDIR" "$BASE_URL/quickstart.sh"
curl -fsSLO --output-dir "$TMPDIR" "$BASE_URL/SHA256SUMS"

echo "  Verifying checksum ..."
(cd "$TMPDIR" && sha256sum -c SHA256SUMS) || {
  echo "[error] checksum verification failed — the download may be corrupted." >&2
  echo "[error] this should not happen; if it does, please report it." >&2
  exit 1
}

echo "  Checksum OK. Starting measurement ..."
echo ""

# -------------------------------------------------------------------- exec --
export ECOCOMPUTE_MODEL="$MODEL_ID"
export ECOCOMPUTE_PARAMS_B="$PARAMS_B"
export ECOCOMPUTE_PRECISION="$PRECISION"

exec bash "$TMPDIR/quickstart.sh"
