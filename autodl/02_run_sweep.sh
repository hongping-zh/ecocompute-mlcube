#!/usr/bin/env bash
# ===========================================================================
# 02_run_sweep.sh — measure the full (model x precision) matrix on RTX 4090D.
#
#   bash autodl/02_run_sweep.sh
#
# For every model in autodl/models.txt and every precision in $PRECISIONS it
# calls entrypoint.py, which does a REAL NVML-sampled measurement when a GPU is
# present. One energy.json is written per config:
#
#   autodl/results/<model_slug>/<precision>/energy.json
#
# Non-FP16 runs also internally re-measure FP16 in the SAME run, so each report
# carries a self-consistent vs_fp16_energy_pct. Failures (e.g. OOM on a big
# model) are logged and skipped so the sweep keeps going.
#
# Override anything via env, e.g.:
#   PRECISIONS="FP16 NF4"  TOKENS=512  ITERATIONS=20  bash autodl/02_run_sweep.sh
#   MODELS_FILE=my_models.txt bash autodl/02_run_sweep.sh
#   DRY_RUN=1 bash autodl/02_run_sweep.sh      # no GPU: dataset reference path
# ===========================================================================
set -uo pipefail   # NB: no -e; a single failed config must not abort the sweep
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/env.sh"
eco_enable_turbo
eco_activate
eco_print_config

STAMP="$(date +%Y%m%d-%H%M%S)"
LOG="$RESULTS_DIR/sweep-$STAMP.log"
mkdir -p "$RESULTS_DIR"
echo "[sweep] logging to $LOG"

DRY_FLAG=""
[ "${DRY_RUN:-0}" = "1" ] && DRY_FLAG="--dry_run"
EXTRA=""
[ "${PREFETCH:-0}" = "1" ] && EXTRA="$EXTRA --prefetch"
[ "${SHARE:-0}" = "1" ] && EXTRA="$EXTRA --share"

n_ok=0; n_fail=0
{
  echo "=== EcoCompute RTX 4090D sweep $STAMP ==="
  eco_print_config

  while read -r model params _rest; do
    case "$model" in ""|\#*) continue;; esac
    slug="$(echo "$model" | tr '/ ' '__')"
    for prec in $PRECISIONS; do
      outdir="$RESULTS_DIR/$slug/$prec"
      mkdir -p "$outdir"
      echo
      echo ">>> $model  precision=$prec  params_b=$params  -> $outdir"
      # shellcheck disable=SC2086
      if python "$REPO_DIR/entrypoint.py" energy_estimate \
            --model "$model" --precision "$prec" --gpu_arch "$GPU_ARCH" \
            --params_b "$params" --batch_size "$BATCH_SIZE" \
            --tokens "$TOKENS" --iterations "$ITERATIONS" --warmup "$WARMUP" \
            --sample_rate_hz "$SAMPLE_RATE_HZ" --output_dir "$outdir" \
            $DRY_FLAG $EXTRA; then
        n_ok=$((n_ok+1))
      else
        n_fail=$((n_fail+1))
        echo "!!! FAILED: $model $prec (see above) — continuing"
      fi
    done
  done < "$MODELS_FILE"

  echo
  echo "=== sweep done: $n_ok ok, $n_fail failed ==="
} 2>&1 | tee "$LOG"

echo
echo "[sweep] aggregate the results with:  python autodl/03_aggregate.py"
