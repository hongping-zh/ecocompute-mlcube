#!/usr/bin/env bash
# EcoCompute energy MLCube - zero-configuration runner.
#
#   curl -fsSL https://raw.githubusercontent.com/hongping-zh/ecocompute-mlcube/main/quickstart.sh | bash
#
# One measurement of one (GPU x model x precision) configuration against its own
# FP16 baseline on the machine you run it on: samples GPU power with NVML, writes
# a schema-validated energy.json, and prints a link that overlays your point on
# https://quantenergy.tech (in-browser, no upload).
#
# Two ways to get there, picked automatically and no flags either way (the
# architecture is read from the NVML device name):
#   docker  - pull the prebuilt image and run it (nothing installed on the host);
#   native  - no Docker (AutoDL / vast.ai / other rented containers cannot nest
#             it): clone the repo and install into a venv on the data disk.
#
# Overrides:
#
#   ECOCOMPUTE_MODEL       HF model id      (default TinyLlama/TinyLlama-1.1B-Chat-v1.0)
#   ECOCOMPUTE_PARAMS_B    size in billions (default 1.1 - must match the model)
#   ECOCOMPUTE_PRECISION   NF4 | INT8       (default NF4; FP16 is always measured too)
#   ECOCOMPUTE_ITERATIONS  decode runs      (default 5; the published dataset used 10)
#   ECOCOMPUTE_IMAGE       image ref        (default ghcr.io/hongping-zh/ecocompute-mlcube:latest)
#   ECOCOMPUTE_OUT         output directory (default ./ecocompute-out)
#   ECOCOMPUTE_GPU_ARGS    docker GPU flags (default "--gpus all"; e.g. --gpus "device=1")
#   ECOCOMPUTE_NO_BUILD=1  fail instead of building locally when the pull fails
#   ECOCOMPUTE_MODE        docker | native  (default: docker if usable, else native)
#   ECOCOMPUTE_SRC         checkout dir for native mode (default: on the data disk)
set -euo pipefail

IMAGE="${ECOCOMPUTE_IMAGE:-ghcr.io/hongping-zh/ecocompute-mlcube:latest}"
MODEL="${ECOCOMPUTE_MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
PARAMS_B="${ECOCOMPUTE_PARAMS_B:-1.1}"
PRECISION="${ECOCOMPUTE_PRECISION:-NF4}"
ITERATIONS="${ECOCOMPUTE_ITERATIONS:-5}"
OUT="${ECOCOMPUTE_OUT:-$PWD/ecocompute-out}"
read -r -a GPU_ARGS <<< "${ECOCOMPUTE_GPU_ARGS---gpus all}"
CACHE="${ECOCOMPUTE_CACHE:-$HOME/.cache/ecocompute-hf}"
REPO_URL="https://github.com/hongping-zh/ecocompute-mlcube.git"

say()  { printf '\033[36m==>\033[0m %s\n' "$*"; }
warn() { printf '\033[33m[warn]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[31m[error]\033[0m %s\n' "$*" >&2; exit 1; }

summarize() {
  local report="$OUT/energy.json"
  [ -f "$report" ] || die "the run finished without writing $report."
  printf '\n'
  say "report: $report"
  python3 - "$report" <<'PY' 2>/dev/null || cat "$report"
import json, sys
r = json.load(open(sys.argv[1]))
res, sut, wl = r["results"], r["system_under_test"], r["workload"]
print("  GPU            : %s (%s)" % (sut["gpu"], sut["gpu_arch"]))
print("  Workload       : %s  %s  batch %s" % (wl["model_name"], wl["precision"], wl["batch_size"]))
print("  Energy/token   : %s mJ  (FP16 baseline %s mJ)"
      % (res.get("energy_per_token_mj"), res.get("fp16_energy_per_token_mj")))
print("  vs FP16        : %s %%" % res.get("vs_fp16_energy_pct"))
print("  Basis          : %s   (source: %s)" % (res.get("basis"), r.get("measurement_source")))
if res.get("basis") != "measured":
    print("\n  NOTE: this is NOT a measurement - GPU power telemetry was unreadable and\n"
          "        the run fell back to the published dataset. Fix the GPU access (see\n"
          "        the warnings above) and re-run before publishing it.")
PY
  if [ -f "$OUT/share_url.txt" ]; then
    printf '\n'
    say "overlay your point on the published curve (opens in your browser, nothing is uploaded):"
    cat "$OUT/share_url.txt"
  fi
  printf '\n'
  say "agree or disagree, publish it: https://quantenergy.tech/replications/#submit"
}

# ------------------------------------------------------------------ preflight --
if ! command -v nvidia-smi >/dev/null 2>&1; then
  warn "nvidia-smi not found. If there is no NVIDIA GPU here the run still completes, but the report will be dataset-derived (basis != measured) instead of a measurement."
fi

MODE="${ECOCOMPUTE_MODE:-}"
if [ -z "$MODE" ]; then
  if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
    MODE=docker
  else
    MODE=native
    say "no usable Docker here (rented GPU containers usually cannot nest it) - installing into a venv instead. Force with ECOCOMPUTE_MODE=docker|native."
  fi
fi

mkdir -p "$OUT"

# -------------------------------------------------------------- native mode --
if [ "$MODE" = native ]; then
  command -v git >/dev/null 2>&1 || die "native mode needs git (apt-get install -y git)."
  command -v python3 >/dev/null 2>&1 || die "native mode needs python3."

  # Rented GPU boxes have a tiny system disk; keep the checkout, venv and model
  # cache on the data disk when there is one (autodl/env.sh applies the same rule
  # to the venv/HF cache, plus the domestic mirrors that make downloads work).
  if [ -n "${ECOCOMPUTE_SRC:-}" ]; then SRC="$ECOCOMPUTE_SRC"
  elif [ -d /root/autodl-tmp ];  then SRC=/root/autodl-tmp/ecocompute/src
  else SRC="$HOME/.cache/ecocompute/src"
  fi

  if [ -d "$SRC/.git" ]; then
    say "updating $SRC"
    git -C "$SRC" pull --ff-only || warn "could not update the checkout; using it as-is."
  else
    say "cloning into $SRC"
    mkdir -p "$(dirname "$SRC")"
    git clone --depth 1 "$REPO_URL" "$SRC"
  fi

  set +u                       # the shared env file predates this script's set -u
  # shellcheck disable=SC1091
  source "$SRC/autodl/env.sh"
  set -u
  say "installing dependencies (venv: $VENV_DIR) - a few minutes the first time"
  SKIP_DOWNLOAD=1 bash "$SRC/autodl/00_setup.sh"
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"

  say "measuring $PRECISION vs FP16 on $MODEL ($ITERATIONS decode iterations each)"
  say "first run also downloads the model into $HF_HOME"
  python3 "$SRC/entrypoint.py" energy_estimate \
    --model "$MODEL" \
    --params_b "$PARAMS_B" \
    --precision "$PRECISION" \
    --gpu_arch auto \
    --iterations "$ITERATIONS" \
    --warmup 1 \
    --output_dir "$OUT" \
    --share
  summarize
  exit 0
fi

# --------------------------------------------------------------------- image --
mkdir -p "$CACHE"
if ! docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia; then
  warn "the NVIDIA container runtime is not registered with Docker; '--gpus all' may fail (the script then retries without it). Install the container toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
fi

say "pulling $IMAGE (~one-time download; later runs reuse it)"
if ! docker pull "$IMAGE"; then
  if docker image inspect "$IMAGE" >/dev/null 2>&1; then
    warn "pull failed (offline, or a local-only tag) - using the copy already on this host."
  elif [ "${ECOCOMPUTE_NO_BUILD:-0}" = "1" ]; then
    die "pull failed, no local copy of $IMAGE, and ECOCOMPUTE_NO_BUILD=1 is set."
  else
    command -v git >/dev/null 2>&1 || die "pull failed and git is unavailable to build locally."
    warn "pull failed - building the image from source instead (~10-20 min, mostly torch)."
    BUILD_DIR="$(mktemp -d)"
    trap 'rm -rf "$BUILD_DIR"' EXIT
    git clone --depth 1 "$REPO_URL" "$BUILD_DIR/src"
    IMAGE="ecocompute/mlcube-energy:local"
    docker build -t "$IMAGE" "$BUILD_DIR/src"
  fi
fi

# ----------------------------------------------------------------- measure --
say "measuring $PRECISION vs FP16 on $MODEL ($ITERATIONS decode iterations each)"
say "first run also downloads the model into $CACHE"
measure() {
  docker run --rm "$@" \
    --user "$(id -u):$(id -g)" \
    -e HOME=/workspace/models/.hf \
    -e HF_HOME=/workspace/models/.hf \
    -v "$OUT:/workspace/outputs" \
    -v "$CACHE:/workspace/models/.hf" \
    "$IMAGE" energy_estimate \
      --model "$MODEL" \
      --params_b "$PARAMS_B" \
      --precision "$PRECISION" \
      --gpu_arch auto \
      --iterations "$ITERATIONS" \
      --warmup 1 \
      --output_dir /workspace/outputs \
      --share
}

if ! measure "${GPU_ARGS[@]}"; then
  # Typically "could not select device driver with capabilities: [[gpu]]" - no
  # driver or no container toolkit. Retry without GPU access so the run still
  # produces a report and the pipeline is verified, but the report will carry
  # basis != "measured" and the summary below says so.
  if [ "${#GPU_ARGS[@]}" -gt 0 ]; then
    warn "docker could not start the container with GPU access. Retrying WITHOUT it: the pipeline gets verified, but the result will be dataset-derived, not a measurement."
    measure
  else
    die "the container failed to start; see the docker error above."
  fi
fi

summarize
