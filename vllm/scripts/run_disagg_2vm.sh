#!/usr/bin/env bash
# GPU-prefill / CPU-decode disaggregation across TWO Jetstream2 VMs.
#
#   prefill node : goldenberry   A100 + AMD EPYC Milan   (ROLE=cuda setup_node.sh)
#   decode  node : disagg-inf-1  Xeon 8468 w/ AMX        (CPU venv from 2026-09-13; H100 unused)
#
# Both nodes need PREFILL_HOST and DECODE_HOST set to the nodes' PRIVATE, directly routable
# addresses -- not floating/public ones; see "Why private IPs" below. On each node:
#   cp vllm/scripts/disagg_hosts.env.example vllm/scripts/disagg_hosts.env  # fill it in once
#   source vllm/scripts/disagg_hosts.env
# or pass the two variables inline.
#
#   # on goldenberry
#   bash run_disagg_2vm.sh probe      # side-channel reachability, before anything heavy
#   bash run_disagg_2vm.sh prefill    # A100 prefill server, backgrounded
#   bash run_disagg_2vm.sh proxy      # disagg proxy on :8192 (co-located with prefill)
#   bash run_disagg_2vm.sh baseline   # arm A: monolithic GPU-only server on :8500
#   bash run_disagg_2vm.sh refs       # 3-way output comparison
#   bash run_disagg_2vm.sh check      # healthcheck + short + long prompt + error grep
#
#   # on disagg-inf-1
#   bash run_disagg_2vm.sh decode     # CPU decode server, backgrounded
#
#   # either node
#   bash run_disagg_2vm.sh down
#
# ---------------------------------------------------------------------------------------
# Why private IPs (Issue 3, still unresolved): UCX picks the address it advertises for the
# data path by enumerating its own NICs. It has no idea a floating IP exists, so across a
# NAT boundary it hands the peer an unreachable address and the transfer dies. Both VMs on
# the same Jetstream2 project network see each other's 10.x addresses directly, which side-
# steps the whole problem. VLLM_NIXL_SIDE_CHANNEL_ADVERTISE_HOST (Issue 1) covers the ZMQ
# handshake only; it cannot fix UCX's own advertisement.
#
# Why PREFILL_KV_BUFFER is pinned to cpu (Issue 8): a decoder without CUDA cannot read the
# prefiller's VRAM over shm/tcp. That needs RDMA + GPUDirect, which Jetstream2 VMs do not
# have. Prefill stages KV into a pinned host buffer so both ends of the transfer are DRAM;
# the cost is one VRAM->host copy per request, charged to TTFT. On two nodes there is no
# DECODE_SEE_GPU escape hatch -- cuda_ipc is intra-node only. Do not "fix" this by setting
# kv_buffer_device=cuda; it will fail at the first request.
# ---------------------------------------------------------------------------------------
set -euo pipefail

WORK=${WORK:-$HOME/work}
export HF_HOME=${HF_HOME:-$WORK/hf}
export HF_HUB_OFFLINE=1
MODEL=${MODEL:-unsloth/gpt-oss-20b}
PREFILL_PORT=8401; DECODE_PORT=8402; PROXY_PORT=8192; BASELINE_PORT=8500
PREFILL_SC_PORT=5559; DECODE_SC_PORT=5659
CUDA_VENV=$WORK/villum-cuda/.venv; CPU_VENV=$WORK/villum-cpu/.venv
LOGROOT=$WORK/logs; mkdir -p "$LOGROOT"
COMMON="--block-size 128 --max-model-len 4096 --tensor-parallel-size 1 --dtype bfloat16 --disable-hybrid-kv-cache-manager"
EXTRA=${EXTRA:-}
PREFILL_KV_BUFFER=cpu   # see Issue 8 above; not a knob on two nodes
# MUST match PREFILL_GPU_UTIL. Every benchmark arm has to use identical execution flags; a
# gpu-memory-utilization mismatch between the monolithic baseline and the disagg legs silently
# gives them different KV cache budgets, which has corrupted a previous round of results here.
BASELINE_GPU_UTIL=${BASELINE_GPU_UTIL:-0.7}
PREFILL_GPU_UTIL=${PREFILL_GPU_UTIL:-0.7}

# Usage with no arguments must work before the host variables are set, so dispatch help first.
case "${1:-}" in
  ""|-h|--help|help) sed -n '2,35p' "$0"; exit 0 ;;
esac

# No apostrophes in these messages: bash processes quotes inside ${VAR:?word}, so a lone `'`
# opens a single-quoted string that swallows the closing brace and everything after it.
: "${PREFILL_HOST:?set PREFILL_HOST to the prefill node private 10.x address}"
: "${DECODE_HOST:?set DECODE_HOST to the decode node private 10.x address}"

# UCX must be pinned to the interface that actually routes to the peer (Issue 4). This is not
# a precaution, it is load-bearing on these two nodes: BOTH of them run docker and therefore both
# have 172.17.0.1 (docker0), 172.18.0.1 and 172.19.0.1 (bridges) -- the SAME addresses on each
# box. Left to enumerate freely, UCX can advertise 172.17.0.1, the peer dials it, and reaches its
# own docker bridge rather than the remote node. That is almost certainly the mechanism behind
# Issue 4's "intermittent connection failures". enp1s0 (10.3.200.0/24) is the only interface that
# actually routes between goldenberry and disagg-inf-1.
nic_to() { ip -o route get "$1" 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="dev") print $(i+1)}' | head -1; }

# --- decode core plan. Unlike the 1-VM script there is no prefill server competing for cores,
# so OMP gets every physical core except DECODE_RESERVE left for the API server, tokenisation,
# sampling and the UCX progress thread. NOTE: the 2026-09-13 single-VM numbers were taken with
# 16 OMP cores (PREFILL_NCORES=4). Set DECODE_RESERVE=4 to reproduce that configuration.
DECODE_RESERVE=${DECODE_RESERVE:-2}
phys_cores() { lscpu -p=CPU,CORE | grep -v '^#' | awk -F, '!seen[$2]++ {print $1}'; }

banner() { echo "=== $1 | prefill=$PREFILL_HOST decode=$DECODE_HOST | logs: $2"; }
newlog()  { L=$LOGROOT/$(date +%Y%m%d_%H%M); mkdir -p "$L"; ln -sfn "$L" $LOGROOT/latest; echo "$L"; }

probe() {
  local mine me peer myrole peerrole open=0 total=0
  mine=$(ip -4 -o addr show scope global | awk '{print $4}' | cut -d/ -f1 | tr '\n' ' ')
  case " $mine " in
    *" $PREFILL_HOST "*) me=$PREFILL_HOST; peer=$DECODE_HOST;  myrole="prefill node"; peerrole="decode node";;
    *" $DECODE_HOST "*)  me=$DECODE_HOST;  peer=$PREFILL_HOST; myrole="decode node";  peerrole="prefill node";;
    *) echo "!! this host has neither PREFILL_HOST ($PREFILL_HOST) nor DECODE_HOST ($DECODE_HOST)"
       echo "   local: $mine"; echo "   the DHCP leases probably moved -- update disagg_hosts.env"; return 1;;
  esac
  echo "this node : $me ($myrole)   $(hostname)"
  echo "peer      : $peer ($peerrole), routed via $(nic_to $peer)"
  echo "ping      : $(ping -c1 -W2 $peer >/dev/null 2>&1 && echo ok || echo UNREACHABLE)"
  echo
  echo "Local interfaces (note both nodes share 172.1x docker bridge addresses -- this is why"
  echo "UCX_NET_DEVICES must be pinned to $(nic_to $peer); see the Issue 4 note at the top):"
  ip -4 -o addr show scope global | awk '{print "    ", $2, $4}'
  echo
  echo "Reachability of the PEER only (self-ports are not tested; they prove nothing)."
  echo "Run \`bash $0 listen\` on $peer first, or every line below will read closed."
  for prt in $PREFILL_SC_PORT $DECODE_SC_PORT $PREFILL_PORT $DECODE_PORT 45000; do
    total=$((total + 1))
    if timeout 3 bash -c "</dev/tcp/$peer/$prt" 2>/dev/null; then
      open=$((open + 1)); echo "    $peer:$prt OPEN"
    else
      echo "    $peer:$prt closed/filtered"
    fi
  done
  echo
  if [ "$open" = "$total" ]; then
    echo "VERDICT: $me -> $peer is clear on all $total ports, including the ephemeral-range"
    echo "         probe :45000. This direction needs no security group change."
    echo "         Security groups are DIRECTIONAL and NIXL dials both ways (decode pulls from"
    echo "         prefill's side channel), so now reverse it: run \`listen\` here and \`probe\`"
    echo "         on $peer."
  elif [ "$open" = 0 ]; then
    echo "VERDICT: nothing reachable on $peer. Either \`listen\` is not running there right now,"
    echo "         or the security group blocks $me -> $peer entirely."
  else
    echo "VERDICT: $open/$total open -- partial. If \`listen\` was running for all of them, the"
    echo "         security group has per-port exceptions rather than a range. UCX needs a range:"
    echo "         ingress TCP 1-65535 from 10.3.200.0/24."
  fi
}

listen() {
  # Hold the ports open so the peer's `probe` measures reachability rather than absence.
  # bind+listen is enough: the kernel completes the handshake from the backlog, so a probe
  # connect() succeeds without this process ever calling accept().
  local secs=${1:-90}
  python3 - "$secs" <<'LISTEN_PY'
import socket, sys, time
# the two side-channel ports, the two API ports, and one ephemeral-range port to verify the
# security group rule is a range and not a handful of per-port exceptions.
PORTS = [5559, 5659, 8401, 8402, 45000]
# hold the SOCKET OBJECTS, not just the port numbers: dropping the last reference closes
# the socket, which silently unbinds every port but the last one.
held = []
for p in PORTS:
    sk = socket.socket()
    sk.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        sk.bind(("0.0.0.0", p)); sk.listen(16); held.append((p, sk))
    except OSError as e:
        print(f"    :{p} NOT bound ({e.strerror}) - a real server may already own it")
secs = int(sys.argv[1])
print("listening on " + ",".join(str(p) for p, _ in held) + f" for {secs}s; run `probe` on the other node now")
time.sleep(secs)
print("done")
LISTEN_PY
}

prefill() {
  LOG=$(newlog); banner "prefill (A100, kv_buffer_device=cpu)" "$LOG"
  NIC=$(nic_to $DECODE_HOST); echo "    UCX pinned to $NIC"
  ( source $CUDA_VENV/bin/activate; cd $WORK/villum-cuda
    # No system CUDA toolkit on the JS2 image: UCX's CUDA transport dlopens libcudart/libnvrtc
    # at runtime, so expose the pip-installed nvidia/* libs torch already ships.
    NVLIBS=$(find $CUDA_VENV/lib/python3.12/site-packages/nvidia -type d -name lib 2>/dev/null | paste -sd: -)
    export LD_LIBRARY_PATH="${NVLIBS}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    CUDA_VISIBLE_DEVICES=0 VLLM_KV_CACHE_LAYOUT=HND VLLM_USE_FLASHINFER_SAMPLER=0 \
    VLLM_NIXL_SIDE_CHANNEL_HOST=$PREFILL_HOST \
    VLLM_NIXL_SIDE_CHANNEL_ADVERTISE_HOST=$PREFILL_HOST \
    VLLM_NIXL_SIDE_CHANNEL_PORT=$PREFILL_SC_PORT \
    UCX_TLS=shm,tcp,cuda_copy,cuda_ipc UCX_NET_DEVICES=$NIC \
    VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO} \
    vllm serve $MODEL --host $PREFILL_HOST --port $PREFILL_PORT --gpu-memory-utilization $PREFILL_GPU_UTIL $COMMON $EXTRA \
      --kv-transfer-config "{\"kv_connector\":\"NixlConnector\",\"kv_role\":\"kv_producer\",\"kv_buffer_device\":\"$PREFILL_KV_BUFFER\",\"kv_load_failure_policy\":\"fail\",\"kv_connector_extra_config\":{\"heterogeneous_hardware_disagg\":true}}" \
    > $LOG/prefill.log 2>&1 ) &
  wait_health $! $PREFILL_HOST $PREFILL_PORT prefill "$LOG"
}

decode() {
  LOG=$(newlog); banner "decode (CPU/AMX, GPU hidden)" "$LOG"
  NIC=$(nic_to $PREFILL_HOST)
  CORES=$(phys_cores | tail -n +$((DECODE_RESERVE + 1)) | paste -sd, -)
  echo "    UCX pinned to $NIC | OMP cores: $CORES (reserved $DECODE_RESERVE for API/OS)"
  ( source $CPU_VENV/bin/activate; cd $WORK/villum-cpu
    # CUDA_VISIBLE_DEVICES= even though this box has an H100: we are measuring a CPU-only
    # decoder. Its UCX therefore needs no cuda transports (prefill stages to DRAM anyway).
    env CUDA_VISIBLE_DEVICES= VLLM_CPU_KVCACHE_SPACE=${VLLM_CPU_KVCACHE_SPACE:-80} \
    VLLM_CPU_OMP_THREADS_BIND="$CORES" VLLM_KV_CACHE_LAYOUT=HND \
    VLLM_NIXL_SIDE_CHANNEL_HOST=$DECODE_HOST \
    VLLM_NIXL_SIDE_CHANNEL_ADVERTISE_HOST=$DECODE_HOST \
    VLLM_NIXL_SIDE_CHANNEL_PORT=$DECODE_SC_PORT \
    UCX_TLS=shm,tcp UCX_NET_DEVICES=$NIC \
    VLLM_LOGGING_LEVEL=${VLLM_LOGGING_LEVEL:-INFO} \
    vllm serve $MODEL --host $DECODE_HOST --port $DECODE_PORT $COMMON $EXTRA \
      --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_consumer","kv_buffer_device":"cpu","kv_load_failure_policy":"fail","kv_connector_extra_config":{"heterogeneous_hardware_disagg":true}}' \
    > $LOG/decode.log 2>&1 ) &
  wait_health $! $DECODE_HOST $DECODE_PORT decode "$LOG"
}

wait_health() {
  local pid=$1 host=$2 port=$3 name=$4 log=$5
  printf "waiting for $host:$port ($name) "
  until curl -sf http://$host:$port/health >/dev/null; do
    if ! kill -0 $pid 2>/dev/null; then
      echo; echo "!! $name exited before becoming healthy; last 40 lines of $log/$name.log:"
      tail -n 40 $log/$name.log; exit 1
    fi
    sleep 10; printf .
  done; echo " ok"
}

proxy() {
  LOG=$LOGROOT/latest
  # co-located with prefill so the decode node's 20 Intel cores stay entirely with OMP.
  ( source $CUDA_VENV/bin/activate; cd $WORK/villum-cuda
    python tests/v1/kv_connector/nixl_integration/toy_proxy_server.py --host 127.0.0.1 --port $PROXY_PORT \
      --prefiller-hosts $PREFILL_HOST --prefiller-ports $PREFILL_PORT \
      --decoder-hosts $DECODE_HOST --decoder-ports $DECODE_PORT \
    > $LOG/proxy.log 2>&1 ) &
  local pid=$!
  printf "waiting for :$PROXY_PORT (proxy) "
  until curl -sf http://127.0.0.1:$PROXY_PORT/healthcheck >/dev/null; do
    if ! kill -0 $pid 2>/dev/null; then echo; tail -n 40 $LOG/proxy.log; exit 1; fi
    sleep 2; printf .
  done; echo " ok"
  echo "UP: proxy http://127.0.0.1:$PROXY_PORT"
}

baseline() {
  # Arm A: monolithic GPU-only on the A100, the thing CPU decode has to justify itself against.
  # NOTE this is an A100; the 2026-09-13 single-VM baseline was an H100. Not comparable across weeks.
  # Needs the A100 to itself: at the mandated 0.7 util (PLAN.md critical rule) it cannot coexist
  # with the prefill server, so run `down` on this node first.
  LOG=$(newlog)
  ( source $CUDA_VENV/bin/activate; cd $WORK/villum-cuda
    CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 \
    vllm serve $MODEL --host 127.0.0.1 --port $BASELINE_PORT \
      --gpu-memory-utilization $BASELINE_GPU_UTIL $COMMON $EXTRA > $LOG/baseline.log 2>&1 ) &
  wait_health $! 127.0.0.1 $BASELINE_PORT baseline "$LOG"
}

refs() {
  set +e
  PROMPT="Disaggregated prefill on A100, decode on Sapphire Rapids. In one sentence, this means"
  for triple in "$PREFILL_HOST:$PREFILL_PORT:GPU-only (prefill server)" \
                "$DECODE_HOST:$DECODE_PORT:CPU-only (decode server)" \
                "127.0.0.1:$PROXY_PORT:disaggregated (via proxy)"; do
    IFS=: read -r h p name <<<"$triple"
    echo "--- $name, $h:$p:"
    curl -sS -m 600 http://$h:$p/v1/completions -H 'Content-Type: application/json' \
      -d "{\"model\":\"$MODEL\",\"prompt\":\"$PROMPT\",\"max_tokens\":48,\"temperature\":0}" \
      | jq -r '.choices[0].text' || echo "!! request failed"
  done
  echo
  echo "All three must match. If the first two are fine and the proxy output is fluent but"
  echo "incoherent, suspect KV pairing (Issue 9) -- dump with VLLM_NIXL_DEBUG_DUMP and diff"
  echo "with tests/v1/kv_connector/nixl_integration/compare_kv_dump.py."
}

check() {
  set +e
  echo "--- proxy healthcheck:"; curl -sS -m 5 http://127.0.0.1:$PROXY_PORT/healthcheck; echo " (exit $?)"
  echo "--- completion via proxy:"
  curl -sS -m 300 http://127.0.0.1:$PROXY_PORT/v1/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"Disaggregated prefill on A100, decode on Sapphire Rapids. In one sentence, this means\",\"max_tokens\":48,\"temperature\":0}" \
    | jq . || echo "!! completion failed (see $LOGROOT/latest/proxy.log)"
  echo "--- long prompt (~500 tokens, multi-block READ path):"
  LONG=$(python3 -c "print(' '.join(f'Sentence number {i} of a long prompt about disaggregated inference.' for i in range(60)))")
  curl -sS -m 600 http://127.0.0.1:$PROXY_PORT/v1/completions -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"prompt\":\"$LONG Summarize the above in one sentence:\",\"max_tokens\":32,\"temperature\":0}" \
    | jq '{text: .choices[0].text, usage}' || echo "!! long completion failed"
  echo "--- memory types advertised (expect both DRAM with kv_buffer_device=cpu):"
  grep -h "registers KV in" $LOGROOT/latest/*.log || echo "    (none logged)"
  echo "--- errors:"; grep -hiE "error|traceback|NIXL_ERR" $LOGROOT/latest/*.log | tail -20 || true
}

down() {
  pkill -f "vllm serve" || true; pkill -f toy_proxy_server.py || true; sleep 2
  pgrep -fa "vllm|toy_proxy" || echo "all down"
}

case "${1:-}" in
  probe) probe;; listen) listen "${2:-90}";; prefill) prefill;; decode) decode;; proxy) proxy;;
  baseline) baseline;; refs) refs;; check) check;; down) down;;
  *) sed -n '2,35p' "$0";;
esac
