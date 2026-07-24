#!/usr/bin/env bash
# ===========================================================================
# 01_verify.sh — confirm this box can produce real NVML energy measurements.
# Wraps verify_env.py with the venv + AutoDL env activated.
#   bash autodl/01_verify.sh                # full check (incl. tiny GPU run)
#   bash autodl/01_verify.sh --no-measure   # skip the tiny GPU measurement
# ===========================================================================
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/env.sh"
eco_enable_turbo
eco_activate
exec python "$HERE/verify_env.py" "$@"
