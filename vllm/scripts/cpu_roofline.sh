#!/usr/bin/env bash
# CPU decode roofline: is CPU decode slow because of the hardware or because of vLLM's
# software path? Run on the DECODE node (disagg-inf-1). No GPU, no disaggregation.
#
#   bash cpu_roofline.sh env        # record hardware/software context + power-counter check
#   bash cpu_roofline.sh stream     # memory bandwidth B (STREAM triad) on the decode cores
#   bash cpu_roofline.sh download   # fetch the Qwen ladder (checks disk space first)
#   bash cpu_roofline.sh predict    # write predicted tok/s = B / W  -- BEFORE any run
#   bash cpu_roofline.sh run-all    # serve each model on CPU, sweep c=1 and c=16
#   bash cpu_roofline.sh summarize  # measured vs predicted -> summary.csv / summary.md
#   bash cpu_roofline.sh all        # env, stream, download, predict, run-all, summarize
#
#   bash cpu_roofline.sh run <hf-model-id>    # one model only (e.g. gpt-oss for reference)
#
# Knobs (env vars): MODELS, OUT, DECODE_RESERVE, INPUT_LEN, OUTPUT_LEN, PORT.
# MODELS defaults to a Qwen3 dense ladder; check the latest small dense Qwen release and
# pin the exact repo ids before running. Outputs land in $OUT (default below); copy that
# directory back to your machine afterwards.
#
# Why the numbers mean something: each run uses the SAME CPU settings as the disagg decode
# leg in run_disagg_2vm.sh (same OMP core list, same serving flags minus the KV connector),
# and STREAM runs on exactly those cores. "Effective bandwidth" of a run is
# W / median TPOT at concurrency 1, where W = bytes of weight files on disk -- the same
# upper-bound method used for gpt-oss in the journal (13 GB / 2.52 s ~= 5.2 GB/s).
set -euo pipefail

WORK=${WORK:-$HOME/work}
export HF_HOME=${HF_HOME:-$WORK/hf}
CPU_VENV=${CPU_VENV:-$WORK/villum-cpu/.venv}
REPO=${REPO:-$WORK/villum-cpu}
OUT=${OUT:-$REPO/experiments/jetstream2/cpu_roofline}
MODELS=${MODELS:-"Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B Qwen/Qwen3-4B"}
PORT=${PORT:-8600}
INPUT_LEN=${INPUT_LEN:-128}
OUTPUT_LEN=${OUTPUT_LEN:-64}
DECODE_RESERVE=${DECODE_RESERVE:-2}      # must match run_disagg_2vm.sh
COMMON="--block-size 128 --max-model-len 4096 --tensor-parallel-size 1 --dtype bfloat16 --disable-hybrid-kv-cache-manager"
SWEEP=$REPO/tests/v1/kv_connector/nixl_integration/run_perf_sweep.sh
mkdir -p "$OUT"

phys_cores() { lscpu -p=CPU,CORE | grep -v '^#' | awk -F, '!seen[$2]++ {print $1}'; }
CORES=$(phys_cores | tail -n +$((DECODE_RESERVE + 1)) | paste -sd, -)
NCORES=$(echo "$CORES" | tr ',' '\n' | wc -l)
slug() { echo "$1" | tr '/' '_'; }

env_info() {
  { echo "date: $(date -Is)   host: $(hostname)"
    echo "decode cores ($NCORES): $CORES"
    echo "--- git"; (cd "$REPO" && git log -1 --format='%h %s' && git status --short | head)
    echo "--- vllm"; ("$CPU_VENV/bin/python" -c "import vllm; print(vllm.__version__)" 2>/dev/null || echo "?")
    echo "--- lscpu"; lscpu
    echo "--- numa"; (numactl -H 2>/dev/null || echo "numactl not installed")
    echo "--- memory"; free -g
    echo "--- disk"; df -h "$HF_HOME" "$HOME" 2>/dev/null | sort -u
    echo "--- power counters (perf)"; (perf list 2>/dev/null | grep -i "power/" || echo "none visible to perf")
    echo "--- powercap sysfs"; (ls /sys/class/powercap 2>/dev/null || echo "no /sys/class/powercap")
  } | tee "$OUT/env.txt"
}

stream() {
  local src=$WORK/STREAM
  if [ ! -f "$src/stream.c" ]; then
    git clone --depth 1 https://github.com/jeffhammond/STREAM "$src" 2>/dev/null || {
      mkdir -p "$src"; curl -fsSL https://www.cs.virginia.edu/stream/FTP/Code/stream.c -o "$src/stream.c"; }
  fi
  # 3 arrays x 200M doubles = 4.8 GB: far beyond any cache, well within RAM. Static arrays
  # over 2 GB need -mcmodel=medium or the link fails with "relocation truncated".
  gcc -O3 -march=native -fopenmp -mcmodel=medium -DSTREAM_ARRAY_SIZE=200000000 -DNTIMES=20 \
      "$src/stream.c" -o "$src/stream_bin"
  local places; places=$(echo "$CORES" | sed 's/\([0-9]\+\)/{\1}/g')
  echo "STREAM on $NCORES cores: $CORES" | tee "$OUT/stream.txt"
  OMP_NUM_THREADS=$NCORES OMP_PLACES="$places" OMP_PROC_BIND=close "$src/stream_bin" \
    | tee -a "$OUT/stream.txt"
  local triad; triad=$(awk '/^Triad:/ {print $2}' "$OUT/stream.txt")
  echo "$triad" > "$OUT/stream_triad_MBps.txt"
  echo ">>> B (triad) = $(awk -v t="$triad" 'BEGIN{printf "%.1f", t/1000}') GB/s"
}

download() {
  "$CPU_VENV/bin/python" - $MODELS <<'PY'
import shutil, sys, os
from huggingface_hub import HfApi, snapshot_download
api = HfApi()
need = 0
for repo in sys.argv[1:]:
    info = api.model_info(repo, files_metadata=True)
    sz = sum(s.size or 0 for s in info.siblings if s.rfilename.endswith(".safetensors"))
    print(f"{repo}: {sz/1e9:.1f} GB of safetensors")
    need += sz
free = shutil.disk_usage(os.environ["HF_HOME"]).free
print(f"need ~{need/1e9:.1f} GB, free {free/1e9:.1f} GB on HF_HOME")
if need + 5e9 > free:
    sys.exit("!! not enough disk (keeping a 5 GB margin) - trim MODELS or free space first")
for repo in sys.argv[1:]:
    # Whole repo except large alternate formats (journal Issue 11). Not an allow-list: a
    # partial snapshot makes newer huggingface_hub refuse it offline as "incomplete".
    snapshot_download(repo, ignore_patterns=["original/*", "metal/*", "onnx/*", "*.gguf",
                                             "*.bin", "*.pth", "*.pt", "*.h5", "*.msgpack"])
    print("downloaded", repo)
PY
}

weights_bytes() {   # bytes of .safetensors in the local snapshot of $1
  # Read the cache layout directly (hub/models--Org--Name/refs/main -> snapshots/<sha>) rather
  # than calling snapshot_download(local_files_only=True): newer huggingface_hub refuses a
  # snapshot it considers incomplete (e.g. README/LICENSE skipped), even though every weight
  # file is present. getsize() follows the snapshot symlinks to the real blobs.
  "$CPU_VENV/bin/python" - "$1" <<'PY'
import os, sys
repo = sys.argv[1]
root = os.path.join(os.environ["HF_HOME"], "hub", "models--" + repo.replace("/", "--"))
try:
    sha = open(os.path.join(root, "refs", "main")).read().strip()
except FileNotFoundError:
    sys.exit(f"!! {repo} is not in the HF cache under {root} - run 'download' first")
snap = os.path.join(root, "snapshots", sha)
files = [f for f in os.listdir(snap) if f.endswith(".safetensors")]
if not files:
    sys.exit(f"!! no .safetensors files in {snap}")
print(sum(os.path.getsize(os.path.join(snap, f)) for f in files))
PY
}

predict() {
  [ -s "$OUT/stream_triad_MBps.txt" ] || { echo "run 'stream' first"; exit 1; }
  local B; B=$(awk '{printf "%.3f", $1/1000}' "$OUT/stream_triad_MBps.txt")   # GB/s
  local f=$OUT/predictions.csv
  [ -e "$f" ] && { echo "$f exists - predictions are written once, before runs. Delete it to redo."; exit 1; }
  local tmp=$f.partial
  echo "model,weights_GB,B_GBps,pred_tok_s_c1,pred_tpot_ms_c1,written_at" > "$tmp"
  for m in $MODELS; do
    local W; W=$(weights_bytes "$m")
    awk -v m="$m" -v w="$W" -v b="$B" -v t="$(date -Is)" 'BEGIN{
      gb=w/1e9; tps=b/gb; printf "%s,%.2f,%.1f,%.2f,%.1f,%s\n", m, gb, b, tps, 1000/tps, t}' >> "$tmp"
  done
  mv "$tmp" "$f"   # only a complete prediction set ever appears under the real name
  column -s, -t "$f"
  echo ">>> Predictions written. Commit or copy $f now, before running anything."
}

run_one() {
  local m=$1 s; s=$(slug "$m")
  local log=$OUT/serve_$s.log
  echo "=== $m on CPU cores $CORES (port $PORT)"
  ( source "$CPU_VENV/bin/activate"; cd "$REPO"
    env CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 VLLM_CPU_KVCACHE_SPACE=${VLLM_CPU_KVCACHE_SPACE:-40} \
      VLLM_CPU_OMP_THREADS_BIND="$CORES" \
      vllm serve "$m" --host 127.0.0.1 --port "$PORT" $COMMON > "$log" 2>&1 ) &
  local pid=$!
  printf "waiting for health "
  until curl -sf "http://127.0.0.1:$PORT/health" >/dev/null; do
    kill -0 $pid 2>/dev/null || { echo; echo "!! server died; tail of $log:"; tail -n 40 "$log"; return 1; }
    sleep 5; printf .
  done; echo " ok"
  ( source "$CPU_VENV/bin/activate"
    bash "$SWEEP" --base-url "http://127.0.0.1:$PORT" --model "$m" --label "$s" --out-dir "$OUT" \
      --health-path /health --dataset-name random --input-len "$INPUT_LEN" --output-len "$OUTPUT_LEN" \
      --concurrency-list "1 16" --num-prompts "16 64" \
      --extra-args "--temperature 0 --ignore-eos --num-warmups 2" ) || echo "!! sweep failed for $m"
  pkill -f "vllm serve $m --host 127.0.0.1 --port $PORT" || true
  for _ in $(seq 30); do curl -sf "http://127.0.0.1:$PORT/health" >/dev/null || break; sleep 2; done
  sleep 5   # let the engine core release its memory before the next model loads
}

summarize() {
  "$CPU_VENV/bin/python" - "$OUT" <<'PY'
import csv, glob, json, os, sys
out = sys.argv[1]
pred = {r["model"]: r for r in csv.DictReader(open(os.path.join(out, "predictions.csv")))}
B = float(open(os.path.join(out, "stream_triad_MBps.txt")).read()) / 1000
rows = []
for m, p in pred.items():
    s = m.replace("/", "_")
    def load(c):
        f = os.path.join(out, f"{s}_c{c}.json")
        return json.load(open(f)) if os.path.exists(f) else None
    r1, r16 = load(1), load(16)
    if not r1:
        rows.append({"model": m, "note": "no c=1 result"}); continue
    W = float(p["weights_GB"]); tpot = r1["median_tpot_ms"]
    eff = W / (tpot / 1000)
    rows.append({
        "model": m, "weights_GB": f"{W:.2f}",
        "pred_tok_s_c1": p["pred_tok_s_c1"], "meas_tok_s_c1": f"{1000/tpot:.2f}",
        "meas_over_pred": f"{(1000/tpot)/float(p['pred_tok_s_c1']):.2f}",
        "median_tpot_ms_c1": f"{tpot:.1f}", "eff_GBps": f"{eff:.1f}", "frac_of_B": f"{eff/B:.2f}",
        "agg_tok_s_c16": f"{r16['output_throughput']:.1f}" if r16 else "",
        "scaling_c16_vs_c1": f"{r16['output_throughput']/r1['output_throughput']:.1f}x" if r16 else "",
        "failed": (r1.get("failed", 0) + (r16 or {}).get("failed", 0)),
    })
# journal reference point, gpt-oss-20b direct to the CPU decode server (2026-09-20)
rows.append({"model": "gpt-oss-20b (journal, MXFP4 MoE)", "weights_GB": "13.0",
             "meas_tok_s_c1": f"{1/2.526:.2f}", "median_tpot_ms_c1": "2526.0",
             "eff_GBps": f"{13.0/2.526:.1f}", "frac_of_B": f"{(13.0/2.526)/B:.2f}",
             "note": "upper bound: assumes every weight byte read per step"})
cols = ["model","weights_GB","pred_tok_s_c1","meas_tok_s_c1","meas_over_pred","median_tpot_ms_c1",
        "eff_GBps","frac_of_B","agg_tok_s_c16","scaling_c16_vs_c1","failed","note"]
with open(os.path.join(out, "summary.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); [w.writerow(r) for r in rows]
with open(os.path.join(out, "summary.md"), "w") as f:
    f.write(f"STREAM triad B = {B:.1f} GB/s\n\n| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
    for r in rows: f.write("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n")
# Fit  TPOT(c=1) = t0 + W / B_eff  across the dense models (least squares, needs >= 2).
# t0  = fixed per-step cost that does not scale with model size (scheduler, sampling, API,
#       Python): software overhead.  B_eff = how fast weight streaming actually runs.
pts = [(float(r["weights_GB"]), float(r["median_tpot_ms_c1"]) / 1000)
       for r in rows if r.get("pred_tok_s_c1") and r.get("median_tpot_ms_c1")]
with open(os.path.join(out, "summary.md"), "a") as f:
    if len(pts) >= 2:
        n = len(pts); mx = sum(x for x, _ in pts) / n; my = sum(y for _, y in pts) / n
        sxx = sum((x - mx) ** 2 for x, _ in pts)
        slope = sum((x - mx) * (y - my) for x, y in pts) / sxx if sxx else float("nan")
        t0 = my - slope * mx
        ss_res = sum((y - (t0 + slope * x)) ** 2 for x, y in pts)
        ss_tot = sum((y - my) ** 2 for _, y in pts)
        r2 = 1 - ss_res / ss_tot if ss_tot else float("nan")
        beff = 1 / slope if slope > 0 else float("nan")
        f.write(f"\nFit over {n} dense models: TPOT = t0 + W / B_eff\n"
                f"  t0 (fixed per-step overhead) = {t0*1000:.1f} ms\n"
                f"  B_eff (weight streaming)     = {beff:.1f} GB/s  = {beff/B:.2f} of STREAM\n"
                f"  R^2 = {r2:.3f}" + ("   (only 2 points: exact fit, R^2 meaningless)" if n == 2 else "") + "\n")
    else:
        f.write("\nFit skipped: need >= 2 dense models with c=1 results.\n")
print(open(os.path.join(out, "summary.md")).read())
PY
}

case "${1:-}" in
  env) env_info;; stream) stream;; download) download;; predict) predict;;
  run) run_one "${2:?model id}";;
  run-all) for m in $MODELS; do run_one "$m"; done;;
  summarize) summarize;;
  all) env_info; stream; download; predict; for m in $MODELS; do run_one "$m"; done; summarize;;
  *) sed -n '2,26p' "$0";;
esac
