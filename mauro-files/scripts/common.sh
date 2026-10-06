#!/bin/bash
# Shared helpers for mauro-files/*.sh — not meant to be run directly.
# Callers do: source "$SCRIPT_DIR/scripts/common.sh"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
NC='\033[0m'

MAURO_FILES_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(dirname "$MAURO_FILES_DIR")"
VENV_PYTHON="$REPO_ROOT/.venv/bin/python"
UI_PORT=8675
WSL_LIB_DIR=/usr/lib/wsl/lib

if [[ -n "${WSL_DISTRO_NAME:-}" ]]; then
    IS_WSL=true
else
    IS_WSL=false
fi

# WSL2: lets Triton find libcuda directly instead of relying on ldconfig,
# which can have timing issues right after WSL boots.
if $IS_WSL; then
    export TRITON_LIBCUDA_PATH="$WSL_LIB_DIR"
fi

require_wsl() {
    if ! $IS_WSL; then
        echo -e "${RED}[ERROR] This script is meant for WSL2 only.${NC}" >&2
        exit 1
    fi
}

info() { echo -e "${GREEN}[OK]${NC} $*"; }
warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
fail() { echo -e "${RED}[ERROR]${NC} $*" >&2; }

version_gte() {
    [ "$(printf '%s\n' "$1" "$2" | sort -V | head -n1)" = "$2" ]
}
