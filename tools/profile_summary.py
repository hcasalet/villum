#!/usr/bin/env python3
"""Summarize torch profiler traces (chrome-trace .json / .json.gz) by SELF time.

Self time = an op's duration minus the time spent in ops nested inside it on the same
thread, so nested ops are not double counted. Prints the top ops and a breakdown by
category (MoE / attention / dense GEMM / dequant / other) for each trace file.

    python tools/profile_summary.py <trace_dir_or_file> [--top 25]
"""
import argparse, glob, gzip, json, os, re, sys
from collections import defaultdict

CATEGORIES = [  # first match wins
    ("moe",       re.compile(r"moe|expert|topk|router|grouped", re.I)),
    ("dequant",   re.compile(r"mxfp4|fp4|dequant|unpack", re.I)),
    ("attention", re.compile(r"attn|attention|paged|sdpa|flash|softmax", re.I)),
    ("gemm",      re.compile(r"linear|addmm|\bmm\b|matmul|gemm|onednn|dnnl|bmm", re.I)),
    ("norm/act",  re.compile(r"norm|silu|gelu|swiglu|act_and_mul|rotary|rope", re.I)),
    ("copy/mem",  re.compile(r"copy|to\b|cat|index|fill|zero|empty|contiguous|clone", re.I)),
]

def category(name):
    for cat, rx in CATEGORIES:
        if rx.search(name):
            return cat
    return "other"

def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        d = json.load(f)
    return d.get("traceEvents", d if isinstance(d, list) else [])

def self_times(events):
    by_thread = defaultdict(list)
    for e in events:
        if e.get("ph") == "X" and e.get("cat") in ("cpu_op", "user_annotation", "kernel", "gpu_user_annotation") \
                and "dur" in e:
            by_thread[(e.get("pid"), e.get("tid"))].append(e)
    agg = defaultdict(lambda: [0.0, 0])           # name -> [self_us, count]
    thread_total = {}
    for key, evs in by_thread.items():
        evs.sort(key=lambda e: (e["ts"], -e["dur"]))
        stack, total = [], 0.0                   # stack of [end, name, child_us]
        def pop_until(ts):
            while stack and stack[-1][0] <= ts:
                end, name, child, dur = stack.pop()
                agg[name][0] += max(dur - child, 0.0); agg[name][1] += 1
                if stack: stack[-1][2] += dur
        for e in evs:
            pop_until(e["ts"])
            if not stack: total += e["dur"]
            stack.append([e["ts"] + e["dur"], e["name"], 0.0, e["dur"]])
        pop_until(float("inf"))
        thread_total[key] = total
    return agg, thread_total

def report(path, top):
    agg, tt = self_times(load(path))
    if not agg:
        print(f"== {os.path.basename(path)}: no cpu_op events"); return
    total = sum(v[0] for v in agg.values())
    print(f"\n== {os.path.basename(path)}  (self time over {len(tt)} threads: {total/1e6:.2f} s)")
    cats = defaultdict(float)
    for n, (us, _) in agg.items():
        cats[category(n)] += us
    print("  by category:")
    for c, us in sorted(cats.items(), key=lambda kv: -kv[1]):
        print(f"    {c:<10} {us/1e6:9.3f} s  {100*us/total:5.1f}%")
    print(f"  top {top} ops by self time:")
    for n, (us, cnt) in sorted(agg.items(), key=lambda kv: -kv[1][0])[:top]:
        print(f"    {100*us/total:5.1f}%  {us/1e6:8.3f} s  x{cnt:<6} [{category(n)}] {n[:90]}")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("path"); ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()
    files = [a.path] if os.path.isfile(a.path) else sorted(
        glob.glob(os.path.join(a.path, "**", "*.json*"), recursive=True))
    if not files: sys.exit(f"no trace files under {a.path}")
    for f in files:
        report(f, a.top)

if __name__ == "__main__":
    main()
