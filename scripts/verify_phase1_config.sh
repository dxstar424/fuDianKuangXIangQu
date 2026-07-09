#!/bin/bash
# 静态校验 Phase 1 代码是否齐全（本地/CI 可跑，无需 DCU）
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ_DIR="$(dirname "$SCRIPT_DIR")"
FAIL=0

_check() {
    local desc="$1"
    shift
    if "$@"; then
        echo "[OK] $desc"
    else
        echo "[FAIL] $desc"
        FAIL=1
    fi
}

_check "launch.sh: gpu 0.94" \
    grep -q 'GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.94}"' "$PROJ_DIR/launch.sh"

_check "launch.sh: prefix caching" \
    grep -q 'enable-prefix-caching' "$PROJ_DIR/launch.sh"

_check "launch.sh: disable logs" \
    grep -q 'disable-log-requests' "$PROJ_DIR/launch.sh" \
    && grep -q 'disable-log-stats' "$PROJ_DIR/launch.sh"

_check "launch.sh: bf16 dtype" \
    grep -q 'bfloat16' "$PROJ_DIR/launch.sh"

_check "launch.sh: warmup" \
    grep -q 'warmup_server.py' "$PROJ_DIR/launch.sh"

_check "launch.sh: KV quant off" \
    grep -q 'FDU_ENABLE_KV_QUANT="${FDU_ENABLE_KV_QUANT:-0}"' "$PROJ_DIR/launch.sh"

_check "rocm_env.sh: HIP vars" \
    grep -q 'HSA_ENABLE_SDMA' "$PROJ_DIR/scripts/rocm_env.sh" \
    && grep -q 'PYTORCH_HIP_ALLOC_CONF' "$PROJ_DIR/scripts/rocm_env.sh"

_check "phase1_env.sh: Phase 2 hooks off" \
    grep -q 'FDU_KV_CACHE_STRATEGY="${FDU_KV_CACHE_STRATEGY:-none}"' "$PROJ_DIR/scripts/phase1_env.sh" \
    && grep -q 'FDU_ENABLE_GQA_OPT="${FDU_ENABLE_GQA_OPT:-0}"' "$PROJ_DIR/scripts/phase1_env.sh"

_check "vllm_env.py: logging level" \
    grep -q 'VLLM_LOGGING_LEVEL' "$PROJ_DIR/src/fdu_vllm/vllm_env.py"

_check "warmup_server.py: three tiers" \
    grep -q '4-8K' "$PROJ_DIR/scripts/warmup_server.py" \
    && grep -q '8-16K' "$PROJ_DIR/scripts/warmup_server.py" \
    && grep -q '16-32K' "$PROJ_DIR/scripts/warmup_server.py"

_check "hooks.py: phase 1 early return" \
    grep -q 'cfg.phase <= 1' "$PROJ_DIR/src/fdu_vllm/hooks.py"

_check "config.yaml: kv quant false" \
    grep -q 'enable_kv_quant: false' "$PROJ_DIR/config.yaml"

if [[ "$FAIL" -ne 0 ]]; then
    echo ""
    echo "Phase 1 config verification FAILED"
    exit 1
fi

echo ""
echo "Phase 1 config verification PASSED (1.1–1.7)"
