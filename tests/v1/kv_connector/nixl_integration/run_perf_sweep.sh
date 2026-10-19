#!/bin/bash
# Sweep vllm bench serve over a list of concurrency levels against an
# already-running server/proxy (single instance, round-robin replica
# proxy, or NIXL disagg proxy), and combine the results into a CSV
# matching the column layout used across experiments/*/*.csv.
#
# This is the perf counterpart of test_disagg_accuracy.py: that script
# checks correctness against a running P/D setup, this one measures
# performance against any running endpoint (single GPU, replicas, or
# disagg proxy) with the same benchmark methodology.
#
# Example (single GPU baseline):
#   ./run_perf_sweep.sh --base-url http://localhost:8500 \
#       --model unsloth/gpt-oss-20b --label singleGPU \
#       --out-dir ../../../../experiments/singleGPU/gpt-oss-20b
#
# Example (disagg proxy):
#   ./run_perf_sweep.sh --base-url http://localhost:8192 \
#       --model unsloth/gpt-oss-20b --label disagg \
#       --out-dir ../../../../experiments/disagg/gpt-oss-20b
#
# Example (round-robin replica proxy):
#   ./run_perf_sweep.sh --base-url http://localhost:8300 \
#       --model unsloth/gpt-oss-20b --label replicas \
#       --out-dir ../../../../experiments/replicas/gpt-oss-20b
#
# Example (fixed-length synthetic prompts, for input-length sweeps). One run per
# input length; put the length in the label so results do not overwrite each other:
#   for ISL in 512 2048 8192 16384; do
#     ./run_perf_sweep.sh --base-url http://localhost:8192 \
#         --model Qwen/Qwen3-1.7B --label disagg_isl${ISL} \
#         --dataset-name random --input-len $ISL --output-len 32 \
#         --concurrency-list "1 16" --num-prompts "16 64" \
#         --extra-args "--temperature 0 --ignore-eos --num-warmups 4" \
#         --out-dir ../../../../experiments/jetstream2/isl_sweep
#   done
# The servers' --max-model-len must be >= input-len + output-len (+ a few tokens of
# tokenizer special tokens), or long points are rejected.

set -xe

BASE_URL=""
MODEL=""
LABEL=""
OUT_DIR=""
DATASET_NAME="sharegpt"    # sharegpt | random
DATASET_PATH="./ShareGPT_V3_unfiltered_cleaned_split.json"   # sharegpt only
INPUT_LEN=""    # random only: tokens per prompt (fixed; range ratio 0)
OUTPUT_LEN=""   # random only: tokens generated per request
NUM_PROMPTS_LIST=(1024)   # one value = same for every point; or one per concurrency level
CONCURRENCY_LIST=(1 2 4 8 16 32 64 96 128 256)
HEALTH_PATH="/healthcheck"  # /health for a bare (non-proxied) vllm serve
EXTRA_ARGS=()   # extra flags passed straight through to `vllm bench serve`

while [[ $# -gt 0 ]]; do
  case $1 in
    --base-url)
      BASE_URL="$2"; shift 2 ;;
    --model)
      MODEL="$2"; shift 2 ;;
    --label)
      LABEL="$2"; shift 2 ;;
    --out-dir)
      OUT_DIR="$2"; shift 2 ;;
    --dataset-name)
      DATASET_NAME="$2"; shift 2 ;;
    --dataset-path)
      DATASET_PATH="$2"; shift 2 ;;
    --input-len)
      INPUT_LEN="$2"; shift 2 ;;
    --output-len)
      OUTPUT_LEN="$2"; shift 2 ;;
    --num-prompts)
      # Either a single value applied to every concurrency point, or a
      # space-separated list matched positionally to --concurrency-list.
      # A per-point count is needed when the arms differ by orders of magnitude
      # in speed: a slow leg cannot afford many prompts at concurrency 1, while
      # a high concurrency point needs at least as many prompts as its batch
      # size or it never actually reaches that concurrency.
      read -r -a NUM_PROMPTS_LIST <<< "$2"
      shift 2 ;;
    --concurrency-list)
      # space-separated string, e.g. --concurrency-list "1 2 4 8"
      read -r -a CONCURRENCY_LIST <<< "$2"
      shift 2 ;;
    --health-path)
      HEALTH_PATH="$2"; shift 2 ;;
    --extra-args)
      # space-separated string forwarded verbatim to `vllm bench serve`, e.g.
      #   --extra-args "--temperature 0 --sharegpt-output-len 128"
      # Needed because the defaults are not comparable across arms: the bench
      # client no longer forces greedy sampling, and with sharegpt output
      # lengths coming from the dataset each arm averages over a different
      # length distribution (and a CPU decode leg can run for hours).
      read -r -a EXTRA_ARGS <<< "$2"
      shift 2 ;;
    *)
      echo "Unknown option $1"
      echo "Usage: $0 --base-url <url> --model <name> --label <label> --out-dir <dir> [--dataset-name sharegpt|random] [--dataset-path <path>] [--input-len <n>] [--output-len <n>] [--num-prompts <n>] [--concurrency-list \"1 2 4 ...\"] [--health-path </health|/healthcheck>] [--extra-args \"--flag val ...\"]"
      exit 1 ;;
  esac
done

if [[ -z "$BASE_URL" || -z "$MODEL" || -z "$LABEL" || -z "$OUT_DIR" ]]; then
  echo "Missing required argument. --base-url, --model, --label, and --out-dir are all required."
  exit 1
fi

case "$DATASET_NAME" in
  sharegpt)
    if [[ -n "$INPUT_LEN" || -n "$OUTPUT_LEN" ]]; then
      echo "--input-len/--output-len apply only to --dataset-name random" >&2
      exit 1
    fi
    DATASET_ARGS=(--dataset-name sharegpt --dataset-path "$DATASET_PATH") ;;
  random)
    if [[ -z "$INPUT_LEN" || -z "$OUTPUT_LEN" ]]; then
      echo "--dataset-name random needs both --input-len and --output-len" >&2
      exit 1
    fi
    # Range ratio 0 = every prompt exactly INPUT_LEN tokens, so a point measures one length,
    # not an average over a distribution. Pass --ignore-eos in --extra-args to force OUTPUT_LEN.
    DATASET_ARGS=(--dataset-name random --random-input-len "$INPUT_LEN"
                  --random-output-len "$OUTPUT_LEN" --random-range-ratio 0.0) ;;
  *)
    echo "--dataset-name must be sharegpt or random" >&2
    exit 1 ;;
esac

mkdir -p "$OUT_DIR"
RES_LOG="${OUT_DIR}/${LABEL}.res"
: > "$RES_LOG"  # truncate/create
{
  echo "label=${LABEL}  base_url=${BASE_URL}  model=${MODEL}"
  echo "num_prompts=[${NUM_PROMPTS_LIST[*]}]  concurrency=[${CONCURRENCY_LIST[*]}]"
  echo "dataset_args=[${DATASET_ARGS[*]}]"
  echo "extra_args=[${EXTRA_ARGS[*]}]"
} | tee -a "$RES_LOG"

# Confirm the endpoint is actually up before burning hours on a sweep.
if ! curl -s -o /dev/null -w "%{http_code}" "${BASE_URL}${HEALTH_PATH}" | grep -q "200"; then
  echo "Endpoint ${BASE_URL}${HEALTH_PATH} is not responding with 200 - aborting."
  exit 1
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"

if [[ ${#NUM_PROMPTS_LIST[@]} -ne 1 && ${#NUM_PROMPTS_LIST[@]} -ne ${#CONCURRENCY_LIST[@]} ]]; then
  echo "--num-prompts must have 1 value or exactly as many as --concurrency-list" >&2
  exit 1
fi

for IDX in "${!CONCURRENCY_LIST[@]}"; do
  CONCURRENCY="${CONCURRENCY_LIST[$IDX]}"
  if [[ ${#NUM_PROMPTS_LIST[@]} -eq 1 ]]; then
    NUM_PROMPTS="${NUM_PROMPTS_LIST[0]}"
  else
    NUM_PROMPTS="${NUM_PROMPTS_LIST[$IDX]}"
  fi
  if [[ "$NUM_PROMPTS" -lt "$CONCURRENCY" ]]; then
    echo "WARNING: concurrency=${CONCURRENCY} with only ${NUM_PROMPTS} prompts - the run will never reach that concurrency" | tee -a "$RES_LOG"
  fi
  echo "Running sweep point: concurrency=${CONCURRENCY} num_prompts=${NUM_PROMPTS}" | tee -a "$RES_LOG"

  vllm bench serve \
    --backend vllm \
    --base-url "$BASE_URL" \
    --model "$MODEL" \
    "${DATASET_ARGS[@]}" \
    --num-prompts "$NUM_PROMPTS" \
    --request-rate inf \
    --max-concurrency "$CONCURRENCY" \
    --save-result \
    --result-dir "$OUT_DIR" \
    --result-filename "${LABEL}_c${CONCURRENCY}.json" \
    "${EXTRA_ARGS[@]}" \
    2>&1 | tee -a "$RES_LOG"
done

python3 "${SCRIPT_DIR}/perf_result_to_csv.py" \
  --results-dir "$OUT_DIR" \
  --glob "${LABEL}_c*.json" \
  --out "${OUT_DIR}/${LABEL}.csv"

echo "Sweep complete. Log: ${RES_LOG}  CSV: ${OUT_DIR}/${LABEL}.csv"
