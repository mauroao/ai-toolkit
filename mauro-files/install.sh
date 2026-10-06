#!/bin/bash
# First-time setup (or dependency re-sync) of AI Toolkit on WSL2.
# Wraps the repo's own installer (python3 -m manager), which creates .venv/
# with the right torch build for the detected driver, installs requirements,
# accelerators (flash-attn, NATTEN, torchcodec) and the UI dependencies.
#
# Usage: mauro-files/install.sh [--force]

set -euo pipefail
trap 'echo "ERROR: Command failed at line $LINENO with exit code $?" >&2; exit 1' ERR

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/scripts/common.sh"

require_wsl

MANAGER_ARGS=()
if [[ "${1:-}" == "--force" ]]; then
    MANAGER_ARGS+=(--force)
fi

echo ""
echo "============================================="
echo " AI Toolkit installer (WSL: ${WSL_DISTRO_NAME})"
echo "============================================="
echo ""

if ! nvidia-smi &>/dev/null; then
    fail "nvidia-smi is not working. Check the Windows NVIDIA driver."
    exit 1
fi
info "nvidia-smi: $(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader)"

if [[ ! -e "$WSL_LIB_DIR/libcuda.so.1" ]]; then
    fail "$WSL_LIB_DIR/libcuda.so.1 not found. GPU passthrough to WSL is broken."
    exit 1
fi
info "libcuda: $WSL_LIB_DIR/libcuda.so.1"

if ! python3 -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
    fail "python3 >= 3.10 is required (found $(python3 --version 2>&1))."
    exit 1
fi
info "python3: $(python3 --version 2>&1)"

if command -v node &>/dev/null && version_gte "$(node --version | tr -d v)" "20"; then
    info "node: $(node --version)"
else
    warn "Node.js >= 20 not found on PATH. The manager will download a portable copy into .node/"
fi

if command -v uv &>/dev/null; then
    info "uv: $(uv --version)"
else
    warn "uv not found on PATH. The manager will use venv + pip (slower)."
fi
echo ""

cd "$REPO_ROOT"

if [[ -x "$VENV_PYTHON" ]]; then
    echo ">>> .venv already exists — syncing dependencies..."
    python3 -m manager sync "${MANAGER_ARGS[@]}"
else
    echo ">>> Creating .venv and installing everything (this takes a while)..."
    python3 -m manager install "${MANAGER_ARGS[@]}"
fi

echo ""
echo ">>> Running environment check..."
"$SCRIPT_DIR/check.sh"
