#!/usr/bin/env bash
# EcoCompute energy MLCube - zero-configuration runner.
#
#   curl -fsSL https://raw.githubusercontent.com/hongping-zh/ecocompute-mlcube/main/quickstart.sh | bash
#
# One measurement of one (GPU x model x precision) configuration against its own
# FP16 baseline on the machine you run it on: pulls the prebuilt image, samples
# GPU power with NVML, writes a schema-validated energy.json, and prints a link
# that overlays your point on https://quantenergy.tech (in-browser, no upload).
#
# Nothing is installed on the host beyond the Docker image, and no flags are
# required: the architecture is detected from the NVML device name. Overrides:
#
#   ECOCOMPUTE_MODEL       HF model id      (default TinyLlama/TinyLlama-1.1B-Chat-v1.0)
#   ECOCOMPUTE_PARAMS_B    size in billions (default 1.1 - must match the model)
#   ECOCOMPUTE_PRECISION   NF4 | INT8       (default NF4; FP16 is always measured too)
#   ECOCOMPUTE_ITERATIONS  decode runs      (default 5; the published dataset used 10)
#   ECOCOMPUTE_IMAGE       image ref        (default ghcr.io/hongping-zh/ecocompute-mlcube:latest)
#   ECOCOMPUTE_OUT         output directory (default ./ecocompute-out)
#   ECOCOMPUTE_GPU_ARGS    docker GPU flags (default "--gpus all"; e.g. --gpus "device=1")
#   ECOCOMPUTE_NO_BUILD=1  fail instead of building locally when the pull fails
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

# ------------------------------------------------------------------ preflight --
command -v docker >/dev/null 2>&1 || die \
  "docker not found. Install Docker Engine: https://docs.docker.com/engine/install/"
docker info >/dev/null 2>&1 || die \
  "the Docker daemon is not reachable as $(id -un). Start it, or add yourself to the 'docker' group (then re-login)."

if ! command -v nvidia-smi >/dev/null 2>&1; then
  warn "nvidia-smi not found on the host. If there is no NVIDIA GPU here the run still completes, but the report will be dataset-derived (basis != measured) instead of a measurement."
elif ! docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia; then
  warn "the NVIDIA container runtime is not registered with Docker; '--gpus all' will likely fail. Install the container toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
fi

mkdir -p "$OUT" "$CACHE"

# --------------------------------------------------------------------- image --
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
docker run --rm "${GPU_ARGS[@]}" \
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

REPORT="$OUT/energy.json"
[ -f "$REPORT" ] || die "the run finished without writing $REPORT."

# ------------------------------------------------------------------ summary --
printf '\n'
say "report: $REPORT"
python3 - "$REPORT" <<'PY' 2>/dev/null || cat "$REPORT"
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
    print("\n  NOTE: this is NOT a measurement - the container could not read GPU power\n"
          "        telemetry and fell back to the published dataset. Fix the GPU access\n"
          "        (see the warnings above) and re-run before publishing it.")
PY

if [ -f "$OUT/share_url.txt" ]; then
  printf '\n'
  say "overlay your point on the published curve (opens in your browser, nothing is uploaded):"
  cat "$OUT/share_url.txt"
fi
printf '\n'
say "agree or disagree, publish it: https://quantenergy.tech/replications/#submit"
