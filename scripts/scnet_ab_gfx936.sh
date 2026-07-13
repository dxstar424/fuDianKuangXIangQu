#!/usr/bin/env bash
set -euo pipefail

PERSIST=/public/home/xdzs2026_c415
EXPERIMENT_ROOT=/public/home/xdzs2026_c415/experiments/gfx936_skinny
CONTROL_VENV=/public/home/xdzs2026_c415/venvs/vllm_baseline
CANDIDATE_VENV=/public/home/xdzs2026_c415/venvs/vllm_gfx936
RESULTS_ROOT=/public/home/xdzs2026_c415/results/gfx936_skinny
MODEL_ROOT=/public/home/xdzs2026_c415/Qwen3.5-27B
TESTDATA_ROOT=/public/home/xdzs2026_c415/testdata
SYSTEM_PYTHON="${SYSTEM_PYTHON:-python3}"
MAX_JOBS="${MAX_JOBS:-16}"
PORT="${PORT:-8001}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd -P)"
SOURCE_ROOT="$EXPERIMENT_ROOT/source"

export FDU_ENABLE=0
export VLLM_ROCM_USE_AITER=0
export VLLM_ROCM_USE_SKINNY_GEMM=1
unset HSA_OVERRIDE_GFX_VERSION ROCBLAS_LAYER PYTHONPATH

usage() {
    echo "usage: $0 {init|build-control|bench|build-candidate|start-control|start-candidate-stock|start-candidate|stop|probe|throughput|accuracy} [args]" >&2
    exit 2
}

require_file() {
    [[ -f "$1" ]] || { echo "missing required file: $1" >&2; exit 2; }
}

safe_clear_build_outputs() {
    [[ "$SOURCE_ROOT" == "$EXPERIMENT_ROOT"/* ]] || exit 2
    rm -rf "$SOURCE_ROOT/build" "$SOURCE_ROOT/dist"
    find "$SOURCE_ROOT" -maxdepth 1 -name '*.egg-info' -type d -exec rm -rf {} +
}

init() {
    mkdir -p "$EXPERIMENT_ROOT" "$RESULTS_ROOT" "$RESULTS_ROOT/wheels/control" \
        "$RESULTS_ROOT/wheels/candidate" "$RESULTS_ROOT/logs" "$RESULTS_ROOT/probes"
    "$SYSTEM_PYTHON" -m venv --system-site-packages "$CONTROL_VENV"
    "$SYSTEM_PYTHON" -m venv --system-site-packages "$CANDIDATE_VENV"
    if [[ "$REPO_ROOT" != "$SOURCE_ROOT" ]]; then
        mkdir -p "$SOURCE_ROOT"
        rsync -a --delete \
            --exclude .git --exclude build --exclude dist --exclude '*.egg-info' \
            "$REPO_ROOT/" "$SOURCE_ROOT/"
    fi
    echo "initialized isolated gfx936 experiment at $EXPERIMENT_ROOT"
}

build_wheel() {
    local role="$1" venv="$2" destination="$RESULTS_ROOT/wheels/$1"
    require_file "$SOURCE_ROOT/setup.py"
    if [[ "$role" == control ]]; then
        grep -Fq 'VALIDATED_GFX936_SHAPES: frozenset[SkinnyShape] = frozenset()' \
            "$SOURCE_ROOT/vllm/model_executor/layers/rocm_skinny_shapes.py" || {
                echo "control build requires an empty gfx936 whitelist" >&2
                exit 2
            }
    fi
    safe_clear_build_outputs
    (
        cd "$SOURCE_ROOT"
        PYTORCH_ROCM_ARCH=gfx936 MAX_JOBS="$MAX_JOBS" \
            "$venv/bin/python" setup.py bdist_wheel
    )
    local wheel
    wheel="$(find "$SOURCE_ROOT/dist" -maxdepth 1 -type f -name 'vllm-*.whl' -print -quit)"
    [[ -n "$wheel" ]] || { echo "wheel was not produced" >&2; exit 2; }
    mkdir -p "$destination"
    cp "$wheel" "$destination/"
    sha256sum "$destination/$(basename "$wheel")" >"$destination/SHA256SUMS"
    "$venv/bin/python" -m pip install --no-deps --force-reinstall "$wheel"
}

bench() {
    require_file "$MODEL_ROOT/config.json"
    "$CONTROL_VENV/bin/python" "$SOURCE_ROOT/scripts/preflight_rocm.py" \
        --expected-prefix "$CONTROL_VENV" --require-arch gfx936 --require-skinny
    (
        cd /tmp
        "$CONTROL_VENV/bin/python" "$SOURCE_ROOT/scripts/bench_gfx936_skinny.py" \
            --model-config "$MODEL_ROOT/config.json" \
            --output "$RESULTS_ROOT/microbench.json" \
            --write-whitelist "$SOURCE_ROOT/vllm/model_executor/layers/rocm_skinny_shapes.py"
    )
}

build_candidate() {
    (cd "$SOURCE_ROOT" && "$CANDIDATE_VENV/bin/python" -m unittest discover -s tests/fdu -p 'test_*.py')
    build_wheel candidate "$CANDIDATE_VENV"
}

stop_server() {
    local pid_file="$RESULTS_ROOT/server.pid" pid remaining
    [[ -f "$pid_file" ]] || return 0
    pid="$(tr -d '[:space:]' <"$pid_file")"
    [[ "$pid" =~ ^[0-9]+$ ]] || { echo "invalid recorded PID" >&2; exit 2; }
    if kill -0 "$pid" 2>/dev/null; then
        kill -TERM "$pid"
        remaining=30
        while kill -0 "$pid" 2>/dev/null && (( remaining > 0 )); do
            sleep 1
            ((remaining -= 1)) || true
        done
        if kill -0 "$pid" 2>/dev/null; then
            kill -KILL "$pid"
        fi
    fi
    rm -f "$pid_file" "$RESULTS_ROOT/server.json"
}

start_server() {
    local label="$1" venv="$2" force_stock="$3"
    stop_server
    "$venv/bin/python" "$SOURCE_ROOT/scripts/preflight_rocm.py" \
        --expected-prefix "$venv" --require-arch gfx936 --require-skinny
    local log="$RESULTS_ROOT/logs/$label.log"
    mkdir -p "$(dirname "$log")"
    (
        cd /tmp
        FDU_ENABLE=0 VLLM_ROCM_USE_AITER=0 VLLM_ROCM_USE_SKINNY_GEMM=1 \
        FDU_FORCE_STOCK_GEMM="$force_stock" \
        exec "$venv/bin/python" -m vllm.entrypoints.openai.api_server \
            --model "$MODEL_ROOT" --port "$PORT" --tensor-parallel-size 1 \
            --max-model-len 32768 --gpu-memory-utilization 0.94 \
            --dtype bfloat16 --trust-remote-code \
            --served-model-name Qwen3.5-27B --load-format auto \
            --no-enable-log-requests
    ) >"$log" 2>&1 &
    local pid=$!
    printf '%s\n' "$pid" >"$RESULTS_ROOT/server.pid"
    printf '{"label":"%s","port":%s,"pid":%s}\n' "$label" "$PORT" "$pid" \
        >"$RESULTS_ROOT/server.json"
    local waited=0
    while (( waited < 1200 )); do
        if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
            echo "$label healthy on port $PORT (pid $pid)"
            return 0
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            break
        fi
        sleep 2
        ((waited += 2)) || true
    done
    tail -n 200 "$log" >&2 || true
    stop_server
    return 2
}

probe() {
    local label="${1:?probe requires LABEL}"
    "$CANDIDATE_VENV/bin/python" "$SOURCE_ROOT/scripts/probe_gfx936.py" \
        --host 127.0.0.1 --port "$PORT" --model Qwen3.5-27B --label "$label" \
        --output "$RESULTS_ROOT/probes/$label.json"
}

fresh_eval_copy() {
    local scratch="$1"
    [[ "$scratch" == "$RESULTS_ROOT/eval_work/"* ]] || exit 2
    rm -rf "$scratch"
    mkdir -p "$scratch"
    rsync -a --exclude test --exclude accuracy_debug --exclude outputs \
        "$TESTDATA_ROOT/" "$scratch/"
}

throughput() {
    local tier="${1:?throughput requires TIER COUNT LABEL}"
    local count="${2:?throughput requires TIER COUNT LABEL}"
    local label="${3:?throughput requires TIER COUNT LABEL}"
    local scratch="$RESULTS_ROOT/eval_work/$label/throughput-$tier"
    local destination="$RESULTS_ROOT/throughput/$label"
    fresh_eval_copy "$scratch"
    mkdir -p "$destination"
    (cd "$scratch" && ./run_throughput.sh "$tier" "$count") \
        2>&1 | tee "$destination/$tier.log"
    cp "$scratch/test/${tier}_throughput/result.json" "$destination/$tier.json"
}

accuracy() {
    local task="${1:?accuracy requires TASK COUNT LABEL}"
    local count="${2:?accuracy requires TASK COUNT LABEL}"
    local label="${3:?accuracy requires TASK COUNT LABEL}"
    local scratch="$RESULTS_ROOT/eval_work/$label/accuracy-$task"
    local destination="$RESULTS_ROOT/accuracy/$label/$task"
    fresh_eval_copy "$scratch"
    mkdir -p "$destination"
    (cd "$scratch" && ./run_accuracy.sh "$task" "$count") \
        2>&1 | tee "$destination/run.log"
    if [[ -d "$scratch/test" ]]; then
        mkdir -p "$destination/test"
        cp -a "$scratch/test/." "$destination/test/"
    fi
    if [[ -d "$scratch/outputs" ]]; then cp -a "$scratch/outputs/." "$destination/"; fi
    if [[ -d "$scratch/accuracy_debug" ]]; then cp -a "$scratch/accuracy_debug/." "$destination/"; fi
}

mode="${1:-}"
shift || true
case "$mode" in
    init) init "$@" ;;
    build-control) build_wheel control "$CONTROL_VENV" ;;
    bench) bench "$@" ;;
    build-candidate) build_candidate "$@" ;;
    start-control) start_server control "$CONTROL_VENV" 1 ;;
    start-candidate-stock) start_server candidate-stock "$CANDIDATE_VENV" 1 ;;
    start-candidate) start_server candidate "$CANDIDATE_VENV" 0 ;;
    stop) stop_server ;;
    probe) probe "$@" ;;
    throughput) throughput "$@" ;;
    accuracy) accuracy "$@" ;;
    *) usage ;;
esac
