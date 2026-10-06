#!/bin/bash
# Starts the AI Toolkit web UI on WSL2 at http://localhost:8675.
# Syncs dependencies first (no-op unless requirements changed), never runs
# `manager update`, so it never does a git pull on its own.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/scripts/common.sh"

require_wsl

if [[ ! -x "$VENV_PYTHON" ]]; then
    fail ".venv not found. Run mauro-files/install.sh first."
    exit 1
fi

cd "$REPO_ROOT"

python3 -m manager sync

echo "[startup] Pre-warming Triton CUDA utils..."
"$VENV_PYTHON" -c "
from triton.backends.nvidia.driver import CudaUtils
CudaUtils()
print('[startup] Triton pre-warm OK')
" || warn "Triton pre-warm failed. Kernels compiled by Triton may not work."

echo ""
echo -e "${BOLD}Open http://localhost:${UI_PORT} in your Windows browser once the build finishes.${NC}"
echo ""

exec python3 -m manager launch --no-browser
