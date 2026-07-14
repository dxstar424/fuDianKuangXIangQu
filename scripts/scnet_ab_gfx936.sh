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
    echo "usage: $0 {init|build-control|bench|build-candidate|sync-candidate-python|quant-bench-w8|quant-bench-hybrid|start-control|start-candidate-stock|start-candidate|start-candidate-off|start-candidate-w8|start-candidate-hybrid|stop|probe|probe-candidate-off|probe-candidate-w8|probe-candidate-hybrid|throughput|accuracy} [args]" >&2
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
    echo "[gfx936:init] creating control venv: $CONTROL_VENV"
    "$SYSTEM_PYTHON" -m venv --without-pip --system-site-packages "$CONTROL_VENV"
    "$CONTROL_VENV/bin/python" -c \
        'import pip, torch; print("[gfx936:init] control", torch.__version__, torch.version.hip)'
    echo "[gfx936:init] creating candidate venv: $CANDIDATE_VENV"
    "$SYSTEM_PYTHON" -m venv --without-pip --system-site-packages "$CANDIDATE_VENV"
    "$CANDIDATE_VENV/bin/python" -c \
        'import pip, torch; print("[gfx936:init] candidate", torch.__version__, torch.version.hip)'
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
        "$CONTROL_VENV/bin/python" -u "$SOURCE_ROOT/scripts/bench_gfx936_skinny.py" \
            --model-config "$MODEL_ROOT/config.json" \
            --output "$RESULTS_ROOT/microbench.json" \
            --write-whitelist "$SOURCE_ROOT/vllm/model_executor/layers/rocm_skinny_shapes.py"
    )
}

build_candidate() {
    (cd "$SOURCE_ROOT" && "$CANDIDATE_VENV/bin/python" -m unittest discover -s tests/fdu -p 'test_*.py')
    build_wheel candidate "$CANDIDATE_VENV"
}

sync_candidate_python() {
    local package_root relative source destination
    local -a copied=()
    package_root="$("$CANDIDATE_VENV/bin/python" -c \
        'import pathlib,vllm; print(pathlib.Path(vllm.__file__).resolve().parent)')"
    [[ -d "$package_root" ]] || {
        echo "candidate package root is not a directory: $package_root" >&2
        exit 2
    }
    local -a files=(
        envs.py
        model_executor/layers/gfx936_online_quant.py
        model_executor/layers/linear.py
        model_executor/layers/utils.py
        model_executor/model_loader/utils.py
        model_executor/layers/rocm_skinny_policy.py
        model_executor/layers/rocm_skinny_shapes.py
    )
    for relative in "${files[@]}"; do
        source="$SOURCE_ROOT/vllm/$relative"
        destination="$package_root/$relative"
        require_file "$source"
        require_file "$destination"
        cp "$source" "$destination"
        copied+=("$destination")
    done
    "$CANDIDATE_VENV/bin/python" -m py_compile "${copied[@]}"
    if ! "$CANDIDATE_VENV/bin/python" -c \
        'import vllm._custom_ops as ops; raise SystemExit(0 if hasattr(ops, "LLMM1") else 1)'
    then
        echo "candidate extension is missing LLMM1; run build-candidate once: $0 build-candidate" >&2
        exit 2
    fi
}

quant_bench() {
    local quant_mode="$1" output log library
    case "$quant_mode" in
        w8)
            output=/tmp/fdu_gfx936_quant_w8.json
            log=/tmp/fdu_gfx936_quant_w8.log
            ;;
        hybrid_w4)
            output=/tmp/fdu_gfx936_quant_hybrid_w4.json
            log=/tmp/fdu_gfx936_quant_hybrid_w4.log
            ;;
        *) echo "unsupported quant benchmark mode: $quant_mode" >&2; exit 2 ;;
    esac
    require_file "$SOURCE_ROOT/csrc/fdu/gfx936_quant_gemv.hip"
    require_file "$SOURCE_ROOT/scripts/build_gfx936_quant_jit.py"
    require_file "$SOURCE_ROOT/scripts/preflight_gfx936_quant.py"
    require_file "$SOURCE_ROOT/scripts/bench_gfx936_quant.py"
    library="$("$CANDIDATE_VENV/bin/python" \
        "$SOURCE_ROOT/scripts/build_gfx936_quant_jit.py" \
        --source "$SOURCE_ROOT/csrc/fdu/gfx936_quant_gemv.hip" \
        --arch gfx936 --timeout 45)"
    (
        cd /tmp
        FDU_GFX936_QUANT_SO="$library" \
            "$CANDIDATE_VENV/bin/python" \
            "$SOURCE_ROOT/scripts/preflight_gfx936_quant.py" \
            --library "$library" --mode "$quant_mode" --smoke
        FDU_GFX936_QUANT_SO="$library" \
            "$CANDIDATE_VENV/bin/python" -u \
            "$SOURCE_ROOT/scripts/bench_gfx936_quant.py" \
            --mode "$quant_mode" --library "$library" --output "$output" \
            --warmup 2 --repetitions 8
    ) 2>&1 | tee "$log"
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
    local label="$1" venv="$2" force_stock="$3" quant_mode="$4" log
    stop_server
    require_file "$SOURCE_ROOT/launch.sh"
    grep -Fq -- '--dtype bfloat16' "$SOURCE_ROOT/launch.sh" || {
        echo "launch.sh no longer pins --dtype bfloat16" >&2
        return 2
    }
    "$venv/bin/python" "$SOURCE_ROOT/scripts/preflight_rocm.py" \
        --expected-prefix "$venv" --require-arch gfx936 --require-skinny
    case "$label:$quant_mode" in
        control:off) log=/tmp/fdu_gfx936_control.log ;;
        *:off) log=/tmp/fdu_gfx936_off.log ;;
        *:w8) log=/tmp/fdu_gfx936_w8.log ;;
        *:hybrid_w4) log=/tmp/fdu_gfx936_hybrid_w4.log ;;
        *) echo "unsupported server quant mode: $quant_mode" >&2; return 2 ;;
    esac
    (
        echo "[gfx936:start] quant_mode=$quant_mode"
        MODEL_PATH="$MODEL_ROOT" PORT="$PORT" PYTHON_BIN="$venv/bin/python" \
        FDU_ENABLE=0 VLLM_ROCM_USE_AITER=0 VLLM_ROCM_USE_SKINNY_GEMM=1 \
        FDU_FORCE_STOCK_GEMM="$force_stock" FDU_GFX936_QUANT_MODE="$quant_mode" \
        exec bash "$SOURCE_ROOT/launch.sh"
    ) >"$log" 2>&1 &
    local pid=$!
    printf '%s\n' "$pid" >"$RESULTS_ROOT/server.pid"
    printf '{"label":"%s","quant_mode":"%s","log":"%s","port":%s,"pid":%s}\n' \
        "$label" "$quant_mode" "$log" "$PORT" "$pid" \
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

probe_candidate_mode() {
    local label="$1" quant_mode="$2" log="$3"
    local output="$RESULTS_ROOT/probes/$label.json"
    curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null
    require_file "$log"
    mkdir -p "$RESULTS_ROOT/probes"
    "$CANDIDATE_VENV/bin/python" - "$PORT" "$output" <<'PY'
import json
import pathlib
import sys
import urllib.request

port = int(sys.argv[1])
output = pathlib.Path(sys.argv[2])
payload = {
    "model": "Qwen3.5-27B",
    "messages": [{"role": "user", "content": "Return OK exactly."}],
    "max_tokens": 16,
    "stream": False,
}
request = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)
with urllib.request.urlopen(request, timeout=300) as response:
    result = json.loads(response.read().decode("utf-8"))
choices = result.get("choices")
if not isinstance(choices, list) or not choices:
    raise SystemExit("probe response has no choices")
choice = choices[0]
message = choice.get("message") if isinstance(choice, dict) else None
content = message.get("content") if isinstance(message, dict) else None
if not isinstance(content, str) or not content.strip():
    raise SystemExit("probe response has no generated content")
output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
PY
    if ! grep -Fqx "[gfx936:start] quant_mode=$quant_mode" "$log"; then
        echo "server log does not declare requested quant_mode=$quant_mode" >&2
        return 2
    fi
    if grep -Eo 'quant_mode=(off|w8|hybrid_w4)' "$log" \
        | grep -Fvx "quant_mode=$quant_mode" >/dev/null
    then
        echo "server log declares another quant mode" >&2
        return 2
    fi
    if grep -Eiq \
        'Traceback|non-finite admission|OOM|out of memory|keeping BF16 path|nrmse=(nan|[+-]?inf)|cosine=(nan|[+-]?inf)|speedup=(nan|[+-]?inf)' \
        "$log"
    then
        echo "server log contains a fatal quantization marker" >&2
        return 2
    fi
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
    sync-candidate-python) sync_candidate_python "$@" ;;
    quant-bench-w8) quant_bench w8 ;;
    quant-bench-hybrid) quant_bench hybrid_w4 ;;
    start-control) start_server control "$CONTROL_VENV" 1 off ;;
    start-candidate-stock) start_server candidate-stock "$CANDIDATE_VENV" 1 off ;;
    start-candidate) start_server candidate "$CANDIDATE_VENV" 0 off ;;
    start-candidate-off) start_server candidate-off "$CANDIDATE_VENV" 0 off ;;
    start-candidate-w8) start_server candidate-w8 "$CANDIDATE_VENV" 0 w8 ;;
    start-candidate-hybrid) start_server candidate-hybrid "$CANDIDATE_VENV" 0 hybrid_w4 ;;
    stop) stop_server ;;
    probe) probe "$@" ;;
    probe-candidate-off) probe_candidate_mode candidate-off off /tmp/fdu_gfx936_off.log ;;
    probe-candidate-w8) probe_candidate_mode candidate-w8 w8 /tmp/fdu_gfx936_w8.log ;;
    probe-candidate-hybrid) probe_candidate_mode candidate-hybrid hybrid_w4 /tmp/fdu_gfx936_hybrid_w4.log ;;
    throughput) throughput "$@" ;;
    accuracy) accuracy "$@" ;;
    *) usage ;;
esac
