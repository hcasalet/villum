#!/usr/bin/env bash
# Workaround for the CPU decode slowdown seen with the NIXL connector configured (journal, 2026-10-19):
# the worker's MAIN thread - an OpenMP participant - ends up pinned to the LAST OMP core, doubling up with an OMP
# thread there, while connector/housekeeping threads (UCX "async", a busy-polling thread, gloo, ZMQ) sit pinned on
# the FIRST OMP core. Run AFTER the decode server is healthy:
#
#   bash repin_decode_threads.sh [--main-only] [--dry-run]
#
#   default      main thread -> first OMP core; every other non-OMP worker thread -> the reserved cores
#   --main-only  only the main thread (reproduces the 2026-10-19 single-move test)
# OMP core list = OMP_CORES if set, else computed like run_disagg_2vm.sh from DECODE_RESERVE (default 2).
set -euo pipefail
MAIN_ONLY=0; DRY=0
for a in "$@"; do case $a in --main-only) MAIN_ONLY=1;; --dry-run) DRY=1;; *) echo "unknown $a"; exit 1;; esac; done

W=$(pgrep -f "VLLM::Worker" | head -1) || { echo "no VLLM::Worker process"; exit 1; }
# The OMP core list: OMP_CORES if given, else computed exactly as run_disagg_2vm.sh does (physical cores minus
# the first DECODE_RESERVE). Not read from /proc/$W/environ: vLLM's process renaming (VLLM::Worker) overwrites
# the memory that file shows, so the variable is invisible there even though it is set.
DECODE_RESERVE=${DECODE_RESERVE:-2}
phys_cores() { lscpu -p=CPU,CORE | grep -v '^#' | awk -F, '!seen[$2]++ {print $1}'; }
BIND=${OMP_CORES:-$(phys_cores | tail -n +$((DECODE_RESERVE + 1)) | paste -sd, -)}
[ -n "$BIND" ] || { echo "could not determine the OMP core list"; exit 1; }
# Sanity check against the live process: the OMP team threads are pinned 1:1 to these cores.
pinned=$(for t in /proc/$W/task/*; do awk '/Cpus_allowed_list/{print $2}' $t/status; done | grep -cxF -f <(tr ',' '\n' <<<"$BIND") || true)
echo "threads pinned to a single core of the OMP list: $pinned (expect >= $(tr ',' '\n' <<<"$BIND" | wc -l))"
FIRST=${BIND%%,*}
NCPU=$(nproc --all)
# reserved = every core not in the OMP list
RESERVED=$(comm -23 <(seq 0 $((NCPU-1)) | sort) <(tr ',' '\n' <<<"$BIND" | sort) | sort -n | paste -sd, -)
[ -n "$RESERVED" ] || { echo "no reserved cores outside $BIND"; exit 1; }
echo "worker pid $W | OMP cores $BIND | first OMP core $FIRST | reserved $RESERVED"

run() { echo "  $*"; [ $DRY = 1 ] || "$@" >/dev/null; }
echo "main thread $W: $(awk '/Cpus_allowed_list/{print $2}' /proc/$W/task/$W/status) -> $FIRST"
run taskset -p -c "$FIRST" "$W"
[ $MAIN_ONLY = 1 ] && exit 0

# OMP team threads are the ones pinned to a single core in the OMP list other than FIRST; leave them alone.
moved=0
for t in /proc/$W/task/*; do
  tid=${t##*/}; [ "$tid" = "$W" ] && continue
  allowed=$(awk '/Cpus_allowed_list/{print $2}' $t/status)
  if [ "$allowed" = "$FIRST" ]; then            # housekeeping threads that inherited the first OMP core
    run taskset -p -c "$RESERVED" "$tid"; moved=$((moved+1))
  fi
done
echo "moved $moved non-OMP threads off core $FIRST to $RESERVED"
