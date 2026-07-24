#!/usr/bin/env bash
# ===========================================================================
# run_all.sh — end-to-end on a fresh AutoDL RTX 4090D instance:
#   setup -> verify -> sweep -> aggregate.
#
#   bash autodl/run_all.sh
#
# Stops early if verification fails (unless FORCE=1). Honors the same env
# overrides as the individual scripts (PRECISIONS, TOKENS, MODELS_FILE, ...).
# ===========================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "########## [1/4] setup ##########"
bash "$HERE/00_setup.sh"

echo "########## [2/4] verify ##########"
if ! bash "$HERE/01_verify.sh"; then
  if [ "${FORCE:-0}" != "1" ]; then
    echo "Verification failed. Fix the issues above, or re-run with FORCE=1 to"
    echo "continue anyway (e.g. for a --dry_run reference sweep)." >&2
    exit 1
  fi
  echo "FORCE=1 set — continuing despite verification failure."
fi

echo "########## [3/4] sweep ##########"
bash "$HERE/02_run_sweep.sh"

echo "########## [4/4] aggregate ##########"
# shellcheck disable=SC1091
source "$HERE/env.sh"; eco_activate
python "$HERE/03_aggregate.py"

echo
echo "All done. Results under: $HERE/results"
