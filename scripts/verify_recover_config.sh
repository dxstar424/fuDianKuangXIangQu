#!/bin/bash
# 阶段1：校验 recover launch 默认（0.94 / eager / stock / warmup 关）
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJ="$(dirname "$SCRIPT_DIR")"
LAUNCH="$PROJ/launch.sh"

fail=0
_check() {
    local desc="$1" ok="$2"
    if [[ "$ok" == "1" ]]; then
        echo "[OK] $desc"
    else
        echo "[FAIL] $desc" >&2
        fail=1
    fi
}

[[ -f "$LAUNCH" ]] || { echo "missing launch.sh"; exit 1; }

grep -q 'GPU_MEMORY_UTILIZATION:-0.94' "$LAUNCH" && c1=1 || c1=0
_check "GPU_MEMORY_UTILIZATION default 0.94" "$c1"

grep -q 'ENFORCE_EAGER:-1' "$LAUNCH" && c2=1 || c2=0
_check "ENFORCE_EAGER default 1" "$c2"

grep -q 'USE_FDU_SERVER:-0' "$LAUNCH" && c3=1 || c3=0
_check "USE_FDU_SERVER default 0 (stock api_server)" "$c3"

grep -q 'DO_WARMUP:-0' "$LAUNCH" && c4=1 || c4=0
_check "DO_WARMUP default 0" "$c4"

grep -q 'FDU_ENABLE_KV_QUANT:-0' "$LAUNCH" && c5=1 || c5=0
_check "FDU_ENABLE_KV_QUANT default 0" "$c5"

# 禁止默认 0.95
if grep -qE 'GPU_MEMORY_UTILIZATION:-\s*0\.95' "$LAUNCH"; then
    _check "must NOT default gpu=0.95" 0
else
    _check "must NOT default gpu=0.95" 1
fi

if (( fail )); then
    echo "[verify_recover_config] FAILED — fix launch.sh before platform submit" >&2
    exit 1
fi
echo "[verify_recover_config] PASS — S1 recover defaults locked"
echo "Next: tag recover-pre-platform → SCNet 三档 ×10 → 平台提交 1 次（目标总分≥65）"
