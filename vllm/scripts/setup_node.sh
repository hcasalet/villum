#!/usr/bin/env bash
# Set up ONE Jetstream2 node for GPU-prefill / CPU-decode disaggregation with the villum fork.
#
#   ROLE=cuda bash setup_node.sh    # prefill node: CUDA venv only (goldenberry: A100 + EPYC Milan)
#   ROLE=cpu  bash setup_node.sh    # decode node:  CPU venv only  (needs AMX for gpt-oss-20b MXFP4)
#   ROLE=both bash setup_node.sh    # both venvs on one host (single-VM disagg)
#
# Run as exouser:  ROLE=cuda bash setup_node.sh 2>&1 | tee ~/setup_node.log
# Re-runnable; each step is skipped if already done.
#
# Supersedes the earlier single-purpose setup scripts. Key points:
#   * ROLE switch: a prefill-only node skips the 20-40 min CPU source build entirely.
#   * hf download --exclude "original/*" "metal/*": the unsloth repo ships a 13 GB Apple Metal
#     build that vLLM never loads (it reads model.safetensors.index.json + the 3 MXFP4 shards).
#     Downloading it can fill a small (~60 GB) root volume.
#   * uv cache is cleaned at the end (~7 GB; only needed while building).
#   * AMX check is fatal for ROLE=cpu instead of a warning, and skipped for ROLE=cuda.
set -euo pipefail

ROLE=${ROLE:-both}
case "$ROLE" in cuda|cpu|both) ;; *) echo "ROLE must be cuda, cpu or both"; exit 1;; esac
REPO=${REPO:-https://github.com/hcasalet/villum.git}
BRANCH=${BRANCH:-main}
WORK=${WORK:-$HOME/work}
export HF_HOME=${HF_HOME:-$WORK/hf}
export UV_CACHE_DIR=$WORK/.uv-cache
mkdir -p "$WORK" "$HF_HOME"

log(){ printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------- 0. hardware sanity
log "Hardware check (ROLE=$ROLE)"
lscpu | grep -E 'Model name|^CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket'
# `|| true` is load-bearing: grep exits 1 when it matches nothing, and under `set -e` +
# `pipefail` that kills the script. On AMD EPYC Milan (Zen 3) NONE of these flags exist, so
# setup_g5.sh's version of this line would abort a prefill-node setup before step 1.
FLAGS=$(lscpu | grep -oE 'amx_bf16|amx_tile|avx512_bf16|avx512f' | sort -u | tr '\n' ' ') || true
echo "ISA flags: ${FLAGS:-none (expected on AMD EPYC; irrelevant for a prefill node)}"
if [ "$ROLE" != cuda ]; then
  # The decode node runs gpt-oss-20b MXFP4 on CPU. Without amx_bf16 the AMX kernels are not
  # built by -march=native and the MXFP4 CPU path is not expected to work. Fail early rather
  # than after a 40 minute build.
  case "$FLAGS" in *amx_bf16*) ;; *)
    echo "FATAL: amx_bf16 not exposed to the guest; this node cannot serve as the decode node."
    echo "       (all validated CPU-decode results come from the Xeon 8468 on disagg-inf-1)"
    exit 1;; esac
fi
if [ "$ROLE" != cpu ]; then
  nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
  DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | cut -d. -f1)
  [ "$DRV" -ge 575 ] || { echo "FATAL: driver $DRV < 575; torch 2.13 cu129 wheels need >= 575 (cu130 needs >= 580)."; exit 1; }
fi
df -h / | tail -1
FREE_GB=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
NEED=$([ "$ROLE" = both ] && echo 30 || echo 25)
[ "$FREE_GB" -ge "$NEED" ] || { echo "FATAL: only ${FREE_GB}G free on /; need ~${NEED}G (13G model + venv + build)."; exit 1; }

# ---------------------------------------------------------------- 1. OS packages
log "OS packages"
sudo apt-get update -qq
sudo apt-get install -y -qq build-essential libnuma-dev git jq curl ccache cmake ninja-build python3-dev
if [ "$ROLE" != cuda ]; then
  if sudo apt-get install -y -qq gcc-13 g++-13 2>/dev/null; then GCCV=13; else sudo apt-get install -y -qq gcc-12 g++-12; GCCV=12; fi
  echo "using gcc-$GCCV"; gcc-$GCCV --version | head -1
fi

# ---------------------------------------------------------------- 2. uv
if ! command -v uv >/dev/null; then
  log "Installing uv"; curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# ---------------------------------------------------------------- 3. clones
CLONES=""
[ "$ROLE" != cpu  ] && CLONES="$CLONES cuda"
[ "$ROLE" != cuda ] && CLONES="$CLONES cpu"
for v in $CLONES; do
  if [ ! -d "$WORK/villum-$v/.git" ]; then
    log "Cloning villum-$v"; git clone --branch "$BRANCH" "$REPO" "$WORK/villum-$v"
  else
    log "villum-$v exists; fast-forwarding"; git -C "$WORK/villum-$v" pull --ff-only
  fi
  # the Issue 7 / Issue 9 connector fixes must be present or disagg silently corrupts output
  git -C "$WORK/villum-$v" log --oneline -1
done

# ---------------------------------------------------------------- 4. CUDA venv (prefill)
if [ "$ROLE" != cpu ] && [ ! -f "$WORK/villum-cuda/.venv/.done" ]; then
  log "CUDA venv (precompiled vLLM wheel; fork changes are Python-only)"
  cd "$WORK/villum-cuda"
  uv venv --python 3.12 --seed .venv
  source .venv/bin/activate
  uv pip install "torch==2.13.0" "torchvision==0.28.0" "torchaudio==2.11.0" --torch-backend=auto
  CUMAJ=$(python -c "import torch; print(torch.version.cuda.split('.')[0])")
  uv pip uninstall nixl nixl-cu12 nixl-cu13 2>/dev/null || true
  uv pip install "nixl[cu${CUMAJ}]"
  git checkout -- pyproject.toml requirements/ 2>/dev/null || true
  python use_existing_torch.py
  uv pip install -r requirements/build/cuda.txt
  VLLM_USE_PRECOMPILED=1 uv pip install --no-build-isolation -e .
  python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
  touch .venv/.done; deactivate
fi

# ---------------------------------------------------------------- 5. CPU venv (decode)
if [ "$ROLE" != cuda ] && [ ! -f "$WORK/villum-cpu/.venv/.done" ]; then
  log "CPU venv (source build; ~20-40 min)"
  cd "$WORK/villum-cpu"
  uv venv --python 3.12 --seed .venv
  source .venv/bin/activate
  uv pip install -r requirements/build/cpu.txt --torch-backend cpu --index-strategy unsafe-best-match
  uv pip install -r requirements/cpu.txt       --torch-backend cpu --index-strategy unsafe-best-match
  uv pip install nixl
  export CC=gcc-$GCCV CXX=g++-$GCCV MAX_JOBS=${MAX_JOBS:-16}
  VLLM_TARGET_DEVICE=cpu uv pip install -e . --no-build-isolation -v 2>&1 | tee "$WORK/cpu_build.log" | grep -E "error:|Building|Successfully|AMX|AVX512" || true
  python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)"
  python - <<'PY'
import torch, vllm._C  # noqa
print("has convert_weight_packed:", hasattr(torch.ops._C, "convert_weight_packed"))
PY
  touch .venv/.done; deactivate
fi

# ---------------------------------------------------------------- 6. model
log "Model download (needs HF_TOKEN if gated)"
ANYVENV=$([ "$ROLE" = cpu ] && echo "$WORK/villum-cpu/.venv" || echo "$WORK/villum-cuda/.venv")
source "$ANYVENV/bin/activate"
uv pip install -q huggingface_hub
# Use the Python API, NOT `hf download ... --exclude "a" "b"`: the CLI treats bare arguments
# after the repo id as explicit FILENAMES, so the second pattern becomes a file to fetch and
# --exclude is silently dropped ("Ignoring `--exclude` since filenames have been explicitly
# set"), then 404s on metal/%2A. `huggingface-cli` is no longer a working fallback either.
# What we skip: original/ (raw OpenAI checkpoint metadata, 24 KB) and metal/ (a 13 GB Apple
# Metal build). vLLM loads neither -- it reads model.safetensors.index.json plus the three
# MXFP4 shards; pulling the whole repo roughly doubles the on-disk model size for nothing.
python -c "from huggingface_hub import snapshot_download as d; print('snapshot:', d('unsloth/gpt-oss-20b', ignore_patterns=['original/*','metal/*']))"
deactivate
du -sh "$HF_HOME/hub"

# ---------------------------------------------------------------- 7. reclaim build cache
log "Cleaning uv cache (~7 GB; venvs are already built)"
rm -rf "$UV_CACHE_DIR"
df -h / | tail -1

log "DONE (ROLE=$ROLE)."
echo "Next, once per node:"
echo "  cp $WORK/villum-cuda/vllm/scripts/disagg_hosts.env.example <clone>/vllm/scripts/disagg_hosts.env"
echo "  # fill in PREFILL_HOST / DECODE_HOST, then: source <clone>/vllm/scripts/disagg_hosts.env"
echo "  bash <clone>/vllm/scripts/run_disagg_2vm.sh probe"
