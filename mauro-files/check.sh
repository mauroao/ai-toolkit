#!/bin/bash
# Environment doctor for AI Toolkit on WSL2: driver/GPU passthrough, .venv,
# torch build, real GPU work, pinned memory and optional accelerators.
# Re-run after every Windows NVIDIA driver update.

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/scripts/common.sh"

NAMES=()
REQUIRED=()
FOUND=()
STATUS=()

add_row() {
    NAMES+=("$1")
    REQUIRED+=("$2")
    FOUND+=("$3")
    STATUS+=("$4")
}

check_wsl() {
    if [[ "$(uname -r)" == *microsoft* ]] && $IS_WSL; then
        add_row "WSL2 kernel" "microsoft" "$(uname -r | cut -d- -f1)" "OK"
    else
        add_row "WSL2 kernel" "microsoft" "$(uname -r)" "FAIL"
    fi
}

check_driver() {
    local smi_banner driver cuda_umd gpu
    smi_banner=$(nvidia-smi 2>/dev/null) || true
    if [[ -z "$smi_banner" ]]; then
        add_row "nvidia-smi" "working" "not working" "FAIL"
        add_row "Driver CUDA" ">= 13.0" "unknown" "FAIL"
        return
    fi
    driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)
    gpu=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n1 | sed 's/NVIDIA GeForce //')
    cuda_umd=$(echo "$smi_banner" | grep -oE 'CUDA( [A-Z]+)? Version: *[0-9]+\.[0-9]+' | grep -oE '[0-9]+\.[0-9]+$' | head -n1)

    add_row "nvidia-smi" "working" "driver $driver" "OK"
    add_row "GPU" "NVIDIA" "$gpu" "OK"
    if [[ -n "$cuda_umd" ]] && version_gte "$cuda_umd" "13.0"; then
        add_row "Driver CUDA" ">= 13.0" "$cuda_umd" "OK"
    else
        add_row "Driver CUDA" ">= 13.0" "${cuda_umd:-unknown}" "FAIL"
    fi
}

check_libcuda() {
    if [[ -e "$WSL_LIB_DIR/libcuda.so.1" ]]; then
        add_row "libcuda (WSL)" "present" "$WSL_LIB_DIR" "OK"
    else
        add_row "libcuda (WSL)" "present" "missing" "FAIL"
    fi
}

check_node() {
    local ver=""
    if [[ -x "$REPO_ROOT/.node/bin/node" ]]; then
        ver=$("$REPO_ROOT/.node/bin/node" --version 2>/dev/null | tr -d v)
    elif command -v node &>/dev/null; then
        ver=$(node --version 2>/dev/null | tr -d v)
    fi
    if [[ -z "$ver" ]]; then
        add_row "Node.js" ">= 20" "not found" "FAIL"
    elif version_gte "$ver" "20"; then
        add_row "Node.js" ">= 20" "$ver" "OK"
    else
        add_row "Node.js" ">= 20" "$ver" "FAIL"
    fi
}

check_port() {
    if ss -ltn 2>/dev/null | grep -q ":${UI_PORT} "; then
        add_row "UI port" "$UI_PORT" "in use (UI running?)" "WARN"
    else
        add_row "UI port" "$UI_PORT" "free" "OK"
    fi
}

check_venv() {
    if [[ ! -x "$VENV_PYTHON" ]]; then
        add_row ".venv" "present" "missing" "FAIL"
        return
    fi
    add_row ".venv" "present" "$("$VENV_PYTHON" --version 2>&1 | awk '{print $2}')" "OK"

    local rows
    rows=$("$VENV_PYTHON" - <<'PYEOF' 2>/dev/null
import importlib
import time


def row(name, required, found, status):
    print(f"{name}\t{required}\t{found}\t{status}")


try:
    import torch
except Exception as exc:
    row("torch", "2.13 cu13x", f"import failed: {type(exc).__name__}", "FAIL")
    raise SystemExit(0)

torch_ok = torch.__version__.startswith("2.13") and (torch.version.cuda or "").startswith("13")
row("torch", "2.13 cu13x", f"{torch.__version__}", "OK" if torch_ok else "FAIL")

if not torch.cuda.is_available():
    row("CUDA available", "True", "False", "FAIL")
    raise SystemExit(0)
row("CUDA available", "True", torch.cuda.get_device_name(0).replace("NVIDIA GeForce ", ""), "OK")

for dtype_name in ("float16", "bfloat16"):
    try:
        dtype = getattr(torch, dtype_name)
        a = torch.randn(4096, 4096, device="cuda", dtype=dtype)
        b = a @ a
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(10):
            b = a @ a
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        tflops = 10 * 2 * 4096**3 / elapsed / 1e12
        finite = bool(torch.isfinite(b).all())
        row(f"matmul {dtype_name}", "runs", f"{tflops:.1f} TFLOPS", "OK" if finite else "FAIL")
    except Exception as exc:
        row(f"matmul {dtype_name}", "runs", type(exc).__name__, "FAIL")

try:
    host = torch.randn(64, 1024, 1024, pin_memory=True)
    device = host.to("cuda", non_blocking=True)
    torch.cuda.synchronize()
    same = bool(torch.equal(device.cpu(), host))
    row("pinned memory", "works", "256 MB copy", "OK" if same else "FAIL")
except Exception as exc:
    row("pinned memory", "works", type(exc).__name__, "WARN")

free, total = torch.cuda.mem_get_info()
row("VRAM free", "info", f"{free / 2**30:.1f}/{total / 2**30:.1f} GiB", "OK")

for module, label in (
    ("triton", "triton"),
    ("flash_attn", "flash-attn"),
    ("natten", "natten"),
    ("torchcodec", "torchcodec"),
    ("bitsandbytes", "bitsandbytes"),
):
    try:
        mod = importlib.import_module(module)
        row(label, "optional", getattr(mod, "__version__", "installed"), "OK")
    except Exception as exc:
        row(label, "optional", f"{type(exc).__name__}", "WARN")
PYEOF
)
    if [[ -z "$rows" ]]; then
        add_row "torch probe" "runs" "crashed" "FAIL"
        return
    fi
    while IFS=$'\t' read -r name required found status; do
        add_row "$name" "$required" "$found" "$status"
    done <<< "$rows"
}

echo ">>> Checking AI Toolkit environment (this runs a short GPU test)..."
check_wsl
check_driver
check_libcuda
check_node
check_port
check_venv

echo ""
printf "${BOLD}======================================================================${NC}\n"
printf "${BOLD}  AI Toolkit Environment Check${NC}\n"
printf "${BOLD}======================================================================${NC}\n"
printf "${BOLD}  %-16s %-12s %-28s %s${NC}\n" "Component" "Required" "Found" "Status"
printf "  %-16s %-12s %-28s %s\n" "---------" "--------" "-----" "------"

any_fail=0
for i in "${!NAMES[@]}"; do
    case "${STATUS[$i]}" in
        OK) status_str="${GREEN}✓ OK${NC}" ;;
        WARN) status_str="${YELLOW}! WARN${NC}" ;;
        *) status_str="${RED}✗ FAIL${NC}"; any_fail=1 ;;
    esac
    printf "  %-16s %-12s %-28s ${status_str}\n" "${NAMES[$i]}" "${REQUIRED[$i]}" "${FOUND[$i]}"
done

printf "${BOLD}======================================================================${NC}\n"
echo ""

if [ "$any_fail" -eq 1 ]; then
    printf "${RED}${BOLD}Some checks failed. See table above.${NC}\n\n"
    exit 1
fi
printf "${GREEN}${BOLD}All required checks passed.${NC}\n\n"
exit 0
