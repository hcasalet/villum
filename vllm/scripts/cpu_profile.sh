#!/usr/bin/env bash
# Profile one model's decode on the CPU decode node: where does the step time go?
# Same CPU settings as the disagg decode leg (cores, serving flags) minus the connector.
#
#   bash cpu_profile.sh [model]          # default unsloth/gpt-oss-20b
#
# Starts the server with the torch profiler enabled, sends 2 short requests with
# `vllm bench serve --profile` (which calls /start_profile and /stop_profile), stops the
# server, then prints a self-time breakdown (tools/profile_summary.py). Output under $OUT.
set -euo pipefail
WORK=${WORK:-$HOME/work}
export HF_HOME=${HF_HOME:-$WORK/hf}
CPU_VENV=${CPU_VENV:-$WORK/villum-cpu/.venv}
REPO=${REPO:-$WORK/villum-cpu}
MODEL=${1:-unsloth/gpt-oss-20b}
SLUG=$(echo "$MODEL" | tr '/' '_')
OUT=${OUT:-$REPO/experiments/jetstream2/cpu_profile/$SLUG}
PORT=${PORT:-8601}
DECODE_RESERVE=${DECODE_RESERVE:-2}
COMMON="--block-size 128 --max-model-len 4096 --tensor-parallel-size 1 --dtype bfloat16 --disable-hybrid-kv-cache-manager"
phys_cores() { lscpu -p=CPU,CORE | grep -v '^#' | awk -F, '!seen[$2]++ {print $1}'; }
CORES=$(phys_cores | tail -n +$((DECODE_RESERVE + 1)) | paste -sd, -)
mkdir -p "$OUT/trace"
# 5 profiled engine iterations (the profiler's default active window): at ~2.5 s/step for
# gpt-oss that is ~12 s of decode, plenty to attribute time, and a trace of modest size.
PROF='{"profiler":"torch","torch_profiler_dir":"'"$OUT/trace"'","torch_profiler_with_stack":false,"torch_profiler_record_shapes":true}'

echo "=== profiling $MODEL on cores $CORES -> $OUT"
( source "$CPU_VENV/bin/activate"; cd "$REPO"
  env CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 VLLM_CPU_KVCACHE_SPACE=${VLLM_CPU_KVCACHE_SPACE:-40} \
    VLLM_CPU_OMP_THREADS_BIND="$CORES" \
    vllm serve "$MODEL" --host 127.0.0.1 --port "$PORT" $COMMON --profiler-config "$PROF" \
    > "$OUT/serve.log" 2>&1 ) &
pid=$!
printf "waiting for health "
until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; do
  kill -0 $pid 2>/dev/null || { echo; echo "!! server died; tail of $OUT/serve.log:"; tail -n 40 "$OUT/serve.log"; exit 1; }
  sleep 5; printf .
done; echo " ok"
( source "$CPU_VENV/bin/activate"
  vllm bench serve --backend vllm --base-url "http://127.0.0.1:$PORT" --model "$MODEL" \
    --dataset-name random --random-input-len 128 --random-output-len 16 --random-range-ratio 0.0 \
    --num-prompts 2 --max-concurrency 1 --temperature 0 --ignore-eos --profile \
    --save-result --result-dir "$OUT" --result-filename bench.json ) 2>&1 | tee "$OUT/bench.log"
sleep 10   # let the profiler finish writing
pkill -f "vllm serve $MODEL --host 127.0.0.1 --port $PORT" || true
for _ in $(seq 30); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || break; sleep 2; done
ls -la "$OUT/trace"
"$CPU_VENV/bin/python" "$REPO/tools/profile_summary.py" "$OUT/trace" --top 25 | tee "$OUT/summary.txt"
echo ">>> summary: $OUT/summary.txt"
