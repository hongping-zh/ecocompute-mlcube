#!/usr/bin/env bash
# EcoCompute energy MLCube - zero-configuration runner.
#
# Recommended way to run this (verify before you execute - a release is
# immutable and checksummed; `main` is neither):
#
#   curl -fsSLO https://github.com/hongping-zh/ecocompute-mlcube/releases/download/schema-1.3-r4/quickstart.sh
#   curl -fsSLO https://github.com/hongping-zh/ecocompute-mlcube/releases/download/schema-1.3-r4/SHA256SUMS
#   sha256sum -c SHA256SUMS
#   bash quickstart.sh
#
# `bash quickstart.sh --dry-run` prints the whole plan - the mode it would
# pick, every directory it would touch, every environment variable it reads,
# and the exact commands - and exits without doing anything.
#
# One measurement of one (GPU x model x precision) configuration against its own
# FP16 baseline on the machine you run it on: samples GPU power with NVML, writes
# a schema-validated energy.json, and prints a link that overlays your point on
# https://quantenergy.tech (in-browser, no upload).
#
# Two ways to get there, picked automatically and no flags either way (the
# architecture is read from the NVML device name):
#   docker  - pull the prebuilt image and run it (nothing installed on the
#             host; the container runs as your uid:gid, never root);
#   native  - no Docker (AutoDL / vast.ai / other rented containers cannot nest
#             it): clone the repo and install into a venv on the data disk.
#
# What this script changes on your machine, in full:
#   both modes   - creates $ECOCOMPUTE_OUT (default ./ecocompute-out) and
#                  writes the report and the overlay link into it. Nothing
#                  outside the directories listed here is written.
#   docker mode  - creates $ECOCOMPUTE_CACHE (default ~/.cache/ecocompute-hf)
#                  and bind-mounts exactly two paths into the container:
#                  $ECOCOMPUTE_OUT -> /workspace/outputs and
#                  $ECOCOMPUTE_CACHE -> /workspace/models/.hf. The model is
#                  downloaded into the cache. No host packages are installed.
#   native mode  - creates a checkout ($ECOCOMPUTE_SRC, else the data disk,
#                  else ~/.cache/ecocompute/src), a venv inside it, installs
#                  the pinned requirements, and downloads the model into an
#                  HF cache on the same disk. Re-running updates in place.
# It reads, and never writes, the ECOCOMPUTE_* variables listed below. It does
# not touch your shell profile, system packages or GPU drivers.
#
# Overrides:
#
#   ECOCOMPUTE_MODEL       HF model id      (default TinyLlama/TinyLlama-1.1B-Chat-v1.0)
#   ECOCOMPUTE_PARAMS_B    size in billions (default 1.1 - must match the model)
#   ECOCOMPUTE_PRECISION   NF4 | INT8       (default NF4; FP16 is always measured too)
#   ECOCOMPUTE_ITERATIONS  decode runs      (default 10, as in the published dataset)
#   ECOCOMPUTE_IMAGE       image ref        (default ghcr.io/hongping-zh/ecocompute-mlcube:latest;
#                          pin a digest from the release page for a reproducible image)
#   ECOCOMPUTE_OUT         output directory (default ./ecocompute-out)
#   ECOCOMPUTE_GPU_ARGS    docker GPU flags (default "--gpus all"; e.g. --gpus "device=1")
#   ECOCOMPUTE_NO_BUILD=1  fail instead of building locally when the pull fails
#   ECOCOMPUTE_MODE        docker | native  (default: docker if usable, else native)
#   ECOCOMPUTE_REF         git ref to clone (default: main). Pin a release tag
#                          (e.g. schema-1.3-r4) for a reproducible native run;
#                          existing checkouts at another ref are re-cloned.
#   ECOCOMPUTE_SRC         checkout dir for native mode (default: on the data disk)
#   ECOCOMPUTE_PREFETCH=1  ask the site for its prediction first and print it next
#                          to your measurement (one HTTPS GET; off by default)
#   ECOCOMPUTE_QUALITY=0   skip the perplexity probe (on by default; it runs after
#                          the power sampler stops, so it never enters the energy)
#   ECOCOMPUTE_READONLY=1  docker mode: add --read-only and a tmpfs /tmp, so the
#                          container's filesystem is immutable apart from the two
#                          mounts above (hardening; report a failure on your host)
#   ECOCOMPUTE_ALLOW_FALLBACK=1
#                          native mode: run even when the quantization backend is
#                          broken (the report will not be a measurement)
set -euo pipefail

# ----------------------------------------------------------------- arguments --
# The script is driven by environment variables; the only CLI flags it takes
# are --dry-run (print the plan, execute nothing) and --help.
DRY_RUN=0
for arg in "$@"; do
  case "$arg" in
    --dry-run|--dry_run) DRY_RUN=1 ;;
    -h|--help)
      # print the header comment (everything up to `set -euo pipefail`)
      hdr_end=$(awk 'NR>1 && /^set -euo/ {print NR-1; exit}' "$0")
      sed -n "2,${hdr_end}p" "$0" | sed 's/^# \{0,1\}//'
      exit 0 ;;
    *)
      printf '[error] unknown argument: %s (this script takes --dry-run only; configuration is via ECOCOMPUTE_* environment variables)\n' "$arg" >&2
      exit 1 ;;
  esac
done

IMAGE="${ECOCOMPUTE_IMAGE:-ghcr.io/hongping-zh/ecocompute-mlcube:latest}"
MODEL="${ECOCOMPUTE_MODEL:-TinyLlama/TinyLlama-1.1B-Chat-v1.0}"
PARAMS_B="${ECOCOMPUTE_PARAMS_B:-1.1}"
PRECISION="${ECOCOMPUTE_PRECISION:-NF4}"
ITERATIONS="${ECOCOMPUTE_ITERATIONS:-10}"
# --share only builds a local URL, so it is always on; --prefetch talks to the
# estimator API before measuring, so it stays opt-in.
EXTRA_ARGS=(--share)
if [ "${ECOCOMPUTE_PREFETCH:-0}" = "1" ]; then EXTRA_ARGS+=(--prefetch); fi
# The quality probe is on by default: it reuses the already-loaded model after the
# sampler stops, so it adds seconds and answers "did this quantization hurt?".
if [ "${ECOCOMPUTE_QUALITY:-1}" = "0" ]; then EXTRA_ARGS+=(--no_quality_probe); fi
OUT="${ECOCOMPUTE_OUT:-$PWD/ecocompute-out}"
read -r -a GPU_ARGS <<< "${ECOCOMPUTE_GPU_ARGS---gpus all}"
CACHE="${ECOCOMPUTE_CACHE:-$HOME/.cache/ecocompute-hf}"
REPO_URL="https://github.com/hongping-zh/ecocompute-mlcube.git"
REPO_REF="${ECOCOMPUTE_REF:-main}"

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
q = r.get("quality") or {}
if q.get("basis") == "measured":
    d = q.get("delta_vs_fp16_pct")
    print("  Quality (ppl)  : %s vs FP16 %s%s"
          % (q.get("value"), q.get("fp16_value", "-"),
             ("  (%+.3f%%)" % d) if d is not None else ""))
sw = r.get("software", {})
pkgs = sw.get("packages", {})
print("  Software       : python %s, torch %s, transformers %s, bitsandbytes %s"
      % (sw.get("python", "?"), pkgs.get("torch", "?"),
         pkgs.get("transformers", "?"), pkgs.get("bitsandbytes", "?")))
if sw.get("differs_from_reference_pins"):
    print("\n  NOTE: not the published pins (%s).\n"
          "        Still a real measurement, but quantization kernels change between\n"
          "        releases - don't compare vs_fp16 with image runs without saying so."
          % "; ".join(sw["differs_from_reference_pins"]))
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

# ------------------------------------------------------------------ dry run --
# Print the resolved plan and exit. Mode detection above is read-only
# (command -v / docker info), so a dry run never writes anything; everything
# below this block is where side effects start.
if [ "$DRY_RUN" = 1 ]; then
  say "dry run - nothing below is executed"
  printf '  mode            : %s\n' "$MODE"
  printf '  workload        : %s (%s B), %s vs FP16, %s iterations + 2 warmups\n' \
    "$MODEL" "$PARAMS_B" "$PRECISION" "$ITERATIONS"
  printf '  output dir      : %s  (created if missing; report + overlay link land here)\n' "$OUT"
  if [ "$MODE" = docker ]; then
    printf '  image           : %s\n' "$IMAGE"
    printf '  model cache     : %s  (created if missing)\n' "$CACHE"
    printf '  runs as         : uid %s, gid %s (never root; --user is always passed)\n' \
      "$(id -u)" "$(id -g)"
    printf '  host mounts     : %s -> /workspace/outputs\n' "$OUT"
    printf '                     %s -> /workspace/models/.hf\n' "$CACHE"
    printf '  readonly rootfs : %s\n' \
      "$([ "${ECOCOMPUTE_READONLY:-0}" = 1 ] && echo 'yes (--read-only + tmpfs /tmp)' || echo 'no (set ECOCOMPUTE_READONLY=1)')"
    printf '  commands        : docker pull %s\n' "$IMAGE"
    printf '                     docker run --rm --user %s:%s %s %s%s \\\n' \
      "$(id -u)" "$(id -g)" "${GPU_ARGS[*]}" \
      "$([ "${ECOCOMPUTE_READONLY:-0}" = 1 ] && echo '--read-only --tmpfs /tmp:rw,size=1g ' || echo '')" \
      "-v $OUT:/workspace/outputs -v $CACHE:/workspace/models/.hf ..."
    printf '                     ... energy_estimate --model %s --params_b %s --precision %s --gpu_arch auto --iterations %s --warmup 2 --output_dir /workspace/outputs %s\n' \
      "$MODEL" "$PARAMS_B" "$PRECISION" "$ITERATIONS" "${EXTRA_ARGS[*]}"
  else
    if   [ -n "${ECOCOMPUTE_SRC:-}" ]; then SRC="$ECOCOMPUTE_SRC"
    elif [ -d /root/autodl-tmp ];     then SRC=/root/autodl-tmp/ecocompute/src
    else SRC="$HOME/.cache/ecocompute/src"; fi
    printf '  checkout        : %s  (git clone --depth 1 --branch %s, or update if already there)\n' "$SRC" "$REPO_REF"
    printf '  venv + installs : inside the checkout; pinned requirements from that ref\n'
    printf '  model cache     : on the same disk (the setup script picks the HF cache dir)\n'
    printf '  commands        : git clone --depth 1 --branch %s https://github.com/hongping-zh/ecocompute-mlcube.git %s\n' "$REPO_REF" "$SRC"
    printf '                     bash %s/autodl/00_setup.sh   (creates the venv, installs the pins)\n' "$SRC"
    printf '                     %s/entrypoint.py energy_estimate --model %s --params_b %s --precision %s --gpu_arch auto --iterations %s --warmup 2 --output_dir %s %s\n' \
      "$SRC" "$MODEL" "$PARAMS_B" "$PRECISION" "$ITERATIONS" "$OUT" "${EXTRA_ARGS[*]}"
  fi
  printf '  env vars read   : ECOCOMPUTE_MODEL PARAMS_B PRECISION ITERATIONS IMAGE OUT GPU_ARGS\n'
  printf '                     MODE REF SRC CACHE PREFETCH QUALITY NO_BUILD READONLY ALLOW_FALLBACK\n'
  say "dry run - nothing was executed. Re-run without --dry-run to measure."
  exit 0
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

  if [ -d "$SRC/.git" ] && [ "$(git -C "$SRC" rev-parse --abbrev-ref HEAD 2>/dev/null || echo x)" = "${REPO_REF}" ]; then
    say "updating $SRC at $REPO_REF"
    git -C "$SRC" pull --ff-only || warn "could not update the checkout; using it as-is."
  elif [ -d "$SRC/.git" ]; then
    say "checkout at $SRC is not at ref '$REPO_REF' - recloning"
    rm -rf "$SRC"
    mkdir -p "$(dirname "$SRC")"
    git clone --depth 1 --branch "$REPO_REF" "$REPO_URL" "$SRC"
  else
    say "cloning into $SRC"
    mkdir -p "$(dirname "$SRC")"
    git clone --depth 1 --branch "$REPO_REF" "$REPO_URL" "$SRC"
  fi

  set +u                       # the shared env file predates this script's set -u
  # shellcheck disable=SC1091
  source "$SRC/autodl/env.sh"
  set -u
  # Say this before the install, not after: the pins need python >= 3.9 and an
  # existing venv can never change interpreter, so on AutoDL's 3.8 default the
  # run is already destined to be flagged non-comparable.
  if [ -x "$VENV_DIR/bin/python" ] \
     && ! "$VENV_DIR/bin/python" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else 1)'; then
    warn "$VENV_DIR is python $("$VENV_DIR/bin/python" -c 'import sys; print("%d.%d" % sys.version_info[:2])'), too old for the published pins (need >= 3.9)."
    warn "the run still measures, but the report will say it is not comparable with image runs. To use the pins:"
    warn "  rm -rf $VENV_DIR && ECO_PYTHON=/path/to/python3.10 SKIP_DOWNLOAD=1 bash $SRC/autodl/00_setup.sh"
  fi
  say "installing dependencies (venv: $VENV_DIR) - a few minutes the first time"
  SKIP_DOWNLOAD=1 bash "$SRC/autodl/00_setup.sh"
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"

  # Setup already tried to repair the quantization backend; if it could not, the
  # run would spend several minutes downloading a model only to fall back to a
  # dataset-derived report. Stop here instead - a volunteer's time is the scarce
  # resource, and a non-measurement helps nobody.
  if ! python3 - <<'PY'
import sys
try:
    import torch, bitsandbytes as bnb
    if not torch.cuda.is_available():
        sys.exit(0)          # no GPU here: entrypoint's own fallback path covers it
    bnb.functional.quantize_4bit(torch.randn(64, 64, device="cuda", dtype=torch.float16))
except Exception:
    sys.exit(1)
PY
  then
    warn "the quantization backend cannot run on this host (see the [4b/5] output above)."
    warn "measuring now would download the model and still produce basis != measured."
    if [ "${ECOCOMPUTE_ALLOW_FALLBACK:-0}" != "1" ]; then
      die "stopping. Re-run with ECOCOMPUTE_ALLOW_FALLBACK=1 to get the non-measured report anyway, or use the Docker image."
    fi
    warn "ECOCOMPUTE_ALLOW_FALLBACK=1 - continuing; do not publish the result as a measurement."
  fi

  say "measuring $PRECISION vs FP16 on $MODEL ($ITERATIONS decode iterations each)"
  say "first run also downloads the model into $HF_HOME"
  python3 "$SRC/entrypoint.py" energy_estimate \
    --model "$MODEL" \
    --params_b "$PARAMS_B" \
    --precision "$PRECISION" \
    --gpu_arch auto \
    --iterations "$ITERATIONS" \
    --warmup 2 \
    --output_dir "$OUT" \
    "${EXTRA_ARGS[@]}"
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
    git clone --depth 1 --branch "$REPO_REF" "$REPO_URL" "$BUILD_DIR/src"
    IMAGE="ecocompute/mlcube-energy:local"
    docker build -t "$IMAGE" "$BUILD_DIR/src"
  fi
fi

# ----------------------------------------------------------------- measure --
say "measuring $PRECISION vs FP16 on $MODEL ($ITERATIONS decode iterations each)"
say "first run also downloads the model into $CACHE"
measure() {
  # --user: the container never runs as root, whoever invokes the script.
  # --read-only + tmpfs /tmp (ECOCOMPUTE_READONLY=1): the container root
  # filesystem becomes immutable; the only writable paths are the two mounts
  # and /tmp. Opt-in because library write locations vary across hosts.
  local readonly_args=()
  if [ "${ECOCOMPUTE_READONLY:-0}" = "1" ]; then
    # shellcheck disable=SC2054  # the comma is docker's tmpfs option syntax
    readonly_args=(--read-only --tmpfs /tmp:rw,size=1g)
  fi
  docker run --rm "$@" \
    --user "$(id -u):$(id -g)" \
    "${readonly_args[@]}" \
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
      --warmup 2 \
      --output_dir /workspace/outputs \
      "${EXTRA_ARGS[@]}"
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
