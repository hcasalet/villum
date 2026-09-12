#!/usr/bin/env bash
# One-time setup of a GPU-CPU machine (1x H100, Xeon 8468 w/ AMX, 20 vCPU, 240 GB)
# for GPU-prefill / CPU-decode disaggregation with the villum fork.
#
# How to run:   bash setup_gpu_cpu.sh 2>&1 | tee ~/setup_gpu_cpu.log
# Re-runnable; each step is skipped if already done.
#
# Layout it creates:
#   ~/work/villum-cuda   clone + .venv  (CUDA build, precompiled wheel)   -> prefill
#   ~/work/villum-cpu    clone + .venv  (CPU source build, AMX kernels)   -> decode
# Two clones because an *editable* install drops the compiled _C.*.so into the
# source tree, so one checkout cannot hold both a CUDA and a CPU build.
set -euo pipefail

REPO=${REPO:-https://github.com/hcasalet/villum.git}   # or git@github.com:hcasalet/villum.git if the VM has a deploy key
BRANCH=${BRANCH:-main}
WORK=${WORK:-$HOME/work}
export HF_HOME=${HF_HOME:-$WORK/hf}
export UV_CACHE_DIR=$WORK/.uv-cache
mkdir -p "$WORK" "$HF_HOME"

log(){ printf '\n\033[1;36m==> %s\033[0m\n' "$*"; }

# ---------------------------------------------------------------- 0. hardware sanity
log "Hardware check"
lscpu | grep -E 'Model name|^CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket'
FLAGS=$(lscpu | grep -oE 'amx_bf16|amx_tile|avx512_bf16|avx512f' | sort -u | tr '\n' ' ')
echo "ISA flags: $FLAGS"
case "$FLAGS" in *amx_bf16*) ;; *) echo "WARNING: amx_bf16 not exposed to the guest; gpt-oss-20b MXFP4 CPU path may fail" ;; esac
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
DRV=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | cut -d. -f1)
[ "$DRV" -ge 575 ] || echo "WARNING: driver $DRV < 575; torch 2.13 cu129 wheels need >= 575 (cu130 needs >= 580). Upgrade the driver or ask JS2."
df -h / | tail -1

# ---------------------------------------------------------------- 1. OS packages
log "OS packages"
sudo apt-get update -qq
sudo apt-get install -y -qq build-essential libnuma-dev git jq curl ccache cmake ninja-build python3-dev
# vLLM CPU build wants gcc >= 12.3; gcc-13 exists on 24.04, only gcc-12 on 22.04
if sudo apt-get install -y -qq gcc-13 g++-13 2>/dev/null; then GCCV=13; else sudo apt-get install -y -qq gcc-12 g++-12; GCCV=12; fi
echo "using gcc-$GCCV"; gcc-$GCCV --version | head -1

# ---------------------------------------------------------------- 2. uv
if ! command -v uv >/dev/null; then
  log "Installing uv"; curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"

# ---------------------------------------------------------------- 3. clones
for v in cuda cpu; do
  if [ ! -d "$WORK/villum-$v/.git" ]; then
    log "Cloning villum-$v"; git clone --branch "$BRANCH" "$REPO" "$WORK/villum-$v"
  fi
done

# ---------------------------------------------------------------- 4. CUDA venv (prefill)
if [ ! -f "$WORK/villum-cuda/.venv/.done" ]; then
  log "CUDA venv (precompiled vLLM wheel; fork changes are Python-only)"
  cd "$WORK/villum-cuda"
  uv venv --python 3.12 --seed .venv
  source .venv/bin/activate
  # MUST match requirements/cuda.txt (torch==2.13.0): the precompiled vLLM wheel's _C.so is built
  # against it. --torch-backend=auto picks cu129/cu130 from the installed driver; setup.py then
  # picks the matching wheels.vllm.ai variant from torch.version.cuda.
  uv pip install "torch==2.13.0" "torchvision==0.28.0" "torchaudio==2.11.0" --torch-backend=auto
  # install the NIXL wheel matching torch's CUDA major (cu12 vs cu13); never both, they each bundle a UCX
  CUMAJ=$(python -c "import torch; print(torch.version.cuda.split('.')[0])")
  uv pip uninstall nixl nixl-cu12 nixl-cu13 2>/dev/null || true
  uv pip install "nixl[cu${CUMAJ}]"
  # use_existing_torch.py strips torch from pyproject's build requirements, so the build MUST
  # run without isolation (setup.py imports torch); requirements/build/cuda.txt supplies the rest.
  git checkout -- pyproject.toml requirements/ 2>/dev/null || true   # idempotent re-runs
  python use_existing_torch.py
  uv pip install -r requirements/build/cuda.txt
  VLLM_USE_PRECOMPILED=1 uv pip install --no-build-isolation -e .
  python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
  touch .venv/.done; deactivate
fi

# ---------------------------------------------------------------- 5. CPU venv (decode) - source build with AMX
if [ ! -f "$WORK/villum-cpu/.venv/.done" ]; then
  log "CPU venv (source build; ~20-40 min on 20 vCPUs)"
  cd "$WORK/villum-cpu"
  uv venv --python 3.12 --seed .venv
  source .venv/bin/activate
  uv pip install -r requirements/build/cpu.txt --torch-backend cpu --index-strategy unsafe-best-match
  uv pip install -r requirements/cpu.txt       --torch-backend cpu --index-strategy unsafe-best-match
  uv pip install nixl
  export CC=gcc-$GCCV CXX=g++-$GCCV MAX_JOBS=${MAX_JOBS:-16}
  # No VLLM_CPU_X86 here: -march=native on this Sapphire Rapids host bakes in AMX/AVX512-BF16.
  VLLM_TARGET_DEVICE=cpu uv pip install -e . --no-build-isolation -v 2>&1 | tee "$WORK/cpu_build.log" | grep -E "error:|Building|Successfully|AMX|AVX512" || true
  python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)"
  python - <<'PY'
import torch, vllm._C  # noqa
print("has convert_weight_packed:", hasattr(torch.ops._C, "convert_weight_packed"))
PY
  IOMP=$(find .venv -name 'libiomp5.so' | head -1); echo "libiomp5: ${IOMP:-not found (fine for source builds)}"
  touch .venv/.done; deactivate
fi

# ---------------------------------------------------------------- 6. model
log "Model download (needs HF_TOKEN if gated)"
source "$WORK/villum-cuda/.venv/bin/activate"
uv pip install -q huggingface_hub
hf download unsloth/gpt-oss-20b >/dev/null 2>&1 || huggingface-cli download unsloth/gpt-oss-20b
deactivate
du -sh "$HF_HOME/hub"

log "DONE. Next: bash run_disagg_1vm.sh up"
