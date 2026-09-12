#!/usr/bin/env bash
# Bring up / tear down GPU-prefill + CPU-decode disaggregation on ONE Jetstream2 g5.xl.
#   bash run_disagg_1vm.sh up        # start prefill, decode, proxy (backgrounded, logs in ~/work/logs/<ts>)
#   bash run_disagg_1vm.sh check     # quick completion through the proxy
#   bash run_disagg_1vm.sh refs      # same prompt straight to prefill-only and decode-only servers (no disagg) vs proxy
#   bash run_disagg_1vm.sh accuracy  # test_disagg_accuracy.py baseline+disagg comparison
#   bash run_disagg_1vm.sh down      # kill everything
# Same flags as experiments/functional_config.md, hosts collapsed to localhost.
set -euo pipefail
WORK=${WORK:-$HOME/work}
export HF_HOME=${HF_HOME:-$WORK/hf}
export HF_HUB_OFFLINE=1   # model is already in $HF_HOME; stops the API server hitting HF (429s) at startup
MODEL=${MODEL:-unsloth/gpt-oss-20b}
PREFILL_PORT=8401; DECODE_PORT=8402; PROXY_PORT=8192; BASELINE_PORT=8500
CUDA_VENV=$WORK/villum-cuda/.venv; CPU_VENV=$WORK/villum-cpu/.venv
LOGROOT=$WORK/logs; mkdir -p "$LOGROOT"
COMMON="--block-size 128 --max-model-len 4096 --tensor-parallel-size 1 --dtype bfloat16 --disable-hybrid-kv-cache-manager"
EXTRA=${EXTRA:-}   # e.g. EXTRA=--enforce-eager for the accuracy run
# Where prefill exposes its KV to NIXL. A CUDA-less decoder cannot read VRAM over shm/tcp
# (UCX: "cannot find remote protocol for: get into host memory from cuda"); that needs RDMA +
# GPUDirect (IB on ACES). With no RDMA (this VM, TCP clusters) prefill must stage KV into a
# pinned host buffer so both ends are DRAM.  cpu = host buffer (default), cuda = direct VRAM.
PREFILL_KV_BUFFER=${PREFILL_KV_BUFFER:-cpu}
# VLLM_USE_FLASHINFER_SAMPLER=0: the JS2 image has no CUDA toolkit (no nvcc, no /usr/local/cuda), and
# FlashInfer JIT-compiles its top-k/top-p sampler at first use -> "Could not find nvcc". The torch
# sampler is used instead; install cuda-toolkit + set CUDA_HOME if some other FlashInfer JIT path appears.

# --- CPU core plan: one logical CPU per physical core; first 2 cores reserved for prefill/proxy/OS
phys_cores() { lscpu -p=CPU,CORE | grep -v '^#' | awk -F, '!seen[$2]++ {print $1}'; }
DECODE_CORES=$(phys_cores | tail -n +3 | paste -sd, -)
# prefill/proxy get the 2 reserved physical cores plus every hyperthread sibling (decode never uses those)
PREFILL_CORES=$( { phys_cores | head -2; comm -23 <(lscpu -p=CPU | grep -v '^#' | sort) <(phys_cores | sort); } | sort -n | paste -sd, -)

up() {
  # refuse to start on top of stale servers from a previous `up` (decode would die with EADDRINUSE
  # and check would talk to the old one)
  for p in $PREFILL_PORT $DECODE_PORT $PROXY_PORT; do
    if curl -s -m 2 -o /dev/null http://127.0.0.1:$p/ 2>/dev/null || ss -ltn 2>/dev/null | grep -q ":$p "; then
      echo "!! port $p is already in use:"; pgrep -fa "vllm serve|toy_proxy" || ss -ltnp | grep ":$p "
      echo "   run: bash $0 down"; exit 1
    fi
  done
  LOG=$LOGROOT/$(date +%Y%m%d_%H%M); mkdir -p "$LOG"; ln -sfn "$LOG" $LOGROOT/latest
  echo "logs: $LOG   decode cores: $DECODE_CORES   prefill cores: $PREFILL_CORES   prefill kv_buffer_device: $PREFILL_KV_BUFFER"

  # prefill (GPU)
  ( source $CUDA_VENV/bin/activate; cd $WORK/villum-cuda
    # No system CUDA toolkit on the JS2 image: UCX's CUDA transport (bundled in the nixl wheel) dlopens
    # libcudart/libnvrtc etc. at runtime, so expose the pip-installed nvidia/* libs torch already ships.
    # Without this UCX logs "UCX CUDA support was not found" and VRAM registerMem fails (NIXL_ERR_BACKEND).
    NVLIBS=$(find $CUDA_VENV/lib/python3.12/site-packages/nvidia -type d -name lib 2>/dev/null | paste -sd: -)
    export LD_LIBRARY_PATH="${NVLIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    CUDA_VISIBLE_DEVICES=0 VLLM_KV_CACHE_LAYOUT=HND VLLM_USE_FLASHINFER_SAMPLER=0 \
    VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1 VLLM_NIXL_SIDE_CHANNEL_PORT=5559 \
    UCX_TLS=shm,tcp,cuda_copy,cuda_ipc VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO} \
    taskset -c "$PREFILL_CORES" \
    vllm serve $MODEL --host 127.0.0.1 --port $PREFILL_PORT --gpu-memory-utilization 0.7 $COMMON $EXTRA \
      --kv-transfer-config "{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_producer\",\"kv_buffer_device\":\"$PREFILL_KV_BUFFER\",\"kv_load_failure_policy\":\"fail\",\"kv_connector_extra_config\":{\"heterogeneous_hardware_disagg\":true}}" \
    > $LOG/prefill.log 2>&1 ) &
  PREFILL_PID=$!

  # decode (CPU, AMX)
  ( source $CPU_VENV/bin/activate; cd $WORK/villum-cpu
    CUDA_VISIBLE_DEVICES="" VLLM_CPU_KVCACHE_SPACE=${VLLM_CPU_KVCACHE_SPACE:-80} \
    VLLM_CPU_OMP_THREADS_BIND="$DECODE_CORES" VLLM_KV_CACHE_LAYOUT=HND \
    VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1 VLLM_NIXL_SIDE_CHANNEL_PORT=5659 \
    UCX_TLS=shm,tcp VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO} \
    vllm serve $MODEL --host 127.0.0.1 --port $DECODE_PORT $COMMON $EXTRA \
      --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_buffer_device":"cpu","kv_load_failure_policy":"fail","kv_connector_extra_config":{"heterogeneous_hardware_disagg":true}}' \
    > $LOG/decode.log 2>&1 ) &
  DECODE_PID=$!

  # wait for both /health; abort (with the log tail) if either server process dies first
  for pair in "$PREFILL_PORT:$PREFILL_PID:prefill" "$DECODE_PORT:$DECODE_PID:decode"; do
    IFS=: read -r port pid name <<<"$pair"
    printf "waiting for :$port ($name) "
    until curl -sf http://127.0.0.1:$port/health >/dev/null; do
      if ! kill -0 $pid 2>/dev/null; then
        echo; echo "!! $name exited before becoming healthy; last 40 lines of $LOG/$name.log:"; tail -n 40 $LOG/$name.log; exit 1
      fi
      sleep 10; printf .
    done; echo " ok"
  done

  ( source $CUDA_VENV/bin/activate; cd $WORK/villum-cuda
    python tests/v1/kv_connector/nixl_integration/toy_proxy_server.py --host 127.0.0.1 --port $PROXY_PORT \
      --prefiller-hosts 127.0.0.1 --prefiller-ports $PREFILL_PORT --decoder-hosts 127.0.0.1 --decoder-ports $DECODE_PORT \
    > $LOG/proxy.log 2>&1 ) &
  PROXY_PID=$!
  printf "waiting for :$PROXY_PORT (proxy) "
  until curl -sf http://127.0.0.1:$PROXY_PORT/healthcheck >/dev/null; do
    if ! kill -0 $PROXY_PID 2>/dev/null; then
      echo; echo "!! proxy exited; $LOG/proxy.log:"; tail -n 40 $LOG/proxy.log; exit 1
    fi
    sleep 2; printf .
  done; echo " ok"
  echo "UP: proxy http://127.0.0.1:$PROXY_PORT  (tail -f $LOG/*.log)"
}

check() {
  set +e   # report, don't abort, on failures here
  echo "--- proxy healthcheck:"; curl -sS -m 5 http://127.0.0.1:$PROXY_PORT/healthcheck; echo " (exit $?)"
  echo "--- completion via proxy:"
  curl -sS -m 300 http://127.0.0.1:$PROXY_PORT/v1/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"Disaggregated prefill on H100, decode on Sapphire Rapids. In one sentence, this means\",\"max_tokens\":48,\"temperature\":0}" \
    | jq . || echo "!! completion request failed (see $LOGROOT/latest/proxy.log)"
  echo "--- long-prompt completion via proxy (~500 tokens = several 128-token KV blocks, exercises the READ path):"
  LONG=$(python3 -c "print(' '.join(f'Sentence number {i} of a long prompt about disaggregated inference.' for i in range(60)))")
  curl -sS -m 600 http://127.0.0.1:$PROXY_PORT/v1/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"$LONG Summarize the above in one sentence:\",\"max_tokens\":32,\"temperature\":0}" \
    | jq '{text: .choices[0].text, usage}' || echo "!! long completion request failed"
  echo "--- NIXL hash agreement:"; grep -h "NIXL compatibility factors" $LOGROOT/latest/prefill.log $LOGROOT/latest/decode.log || true
  echo "--- errors:"; grep -hiE "error|traceback" $LOGROOT/latest/*.log | tail -20 || true
}

accuracy() {
  # baseline: monolithic GPU server, then compare. Both need --enforce-eager (see experiments/PLAN.md).
  source $CUDA_VENV/bin/activate; cd $WORK/villum-cuda
  CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 vllm serve $MODEL --host 127.0.0.1 --port $BASELINE_PORT --gpu-memory-utilization 0.25 $COMMON --enforce-eager \
    > $LOGROOT/latest/baseline.log 2>&1 &
  until curl -sf http://127.0.0.1:$BASELINE_PORT/health >/dev/null; do sleep 10; done
  python tests/v1/kv_connector/nixl_integration/test_disagg_accuracy.py --service_url http://127.0.0.1:$BASELINE_PORT --model_name $MODEL
  python tests/v1/kv_connector/nixl_integration/test_disagg_accuracy.py --service_url http://127.0.0.1:$PROXY_PORT   --model_name $MODEL --mode disagg
  pkill -f "port $BASELINE_PORT" || true
}

refs() {
  # Reference outputs WITHOUT disaggregation: same prompt straight to each server (each does its
  # own prefill+decode). Compare with `check`: garbage here too => model/backend problem on that
  # side; good here but garbage via the proxy => KV transfer/layout problem.
  set +e
  PROMPT="Disaggregated prefill on H100, decode on Sapphire Rapids. In one sentence, this means"
  for pair in "$PREFILL_PORT:GPU-only (prefill server)" "$DECODE_PORT:CPU-only (decode server)"; do
    IFS=: read -r port name <<<"$pair"
    echo "--- $name, port $port:"
    curl -sS -m 600 http://127.0.0.1:$port/v1/completions -H 'Content-Type: application/json' \
      -d "{\"model\":\"$MODEL\",\"prompt\":\"$PROMPT\",\"max_tokens\":48,\"temperature\":0}" \
      | jq -r '.choices[0].text' || echo "!! request failed"
  done
  echo "--- disaggregated (via proxy), port $PROXY_PORT:"
  curl -sS -m 600 http://127.0.0.1:$PROXY_PORT/v1/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"$PROMPT\",\"max_tokens\":48,\"temperature\":0}" \
    | jq -r '.choices[0].text' || echo "!! request failed"
}

down() { pkill -f "vllm serve" || true; pkill -f toy_proxy_server.py || true; sleep 2; pgrep -fa "vllm|toy_proxy" || echo "all down"; }

case "${1:-}" in up) up;; check) check;; refs) refs;; accuracy) accuracy;; down) down;; *) sed -n 2,8p "$0";; esac
