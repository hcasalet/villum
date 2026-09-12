# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diff the KV blocks a NIXL prefiller sent against what the decoder received
and post-processed (heterogeneous GPU-prefill / CPU-decode debugging aid).

Start both servers with VLLM_NIXL_DEBUG_DUMP=<dir> (see
NixlConnectorWorker._debug_dump_kv), send one request through the proxy, then
run this in the decoder's environment:
    python compare_kv_dump.py <dir>
It reports, per layer: transfer fidelity (raw bytes, incl. layer-order
permutations), post-process fidelity (pack_kv_cache), and a sanity look at
the sent content.
"""
import glob
import os
import sys

import torch
from vllm.platforms import current_platform

d = sys.argv[1] if len(sys.argv) > 1 else "/tmp/nixl_dump"
def latest(tag):
    fs = sorted(glob.glob(os.path.join(d, f"{tag}_*.pt")))
    assert fs, f"no {tag}_*.pt in {d}"
    return torch.load(fs[-1])

P, R, K = latest("prefill"), latest("decode_raw"), latest("decode_packed")
print("prefill blocks", P["block_ids"], "| decode blocks", R["block_ids"])
pl, rl, kl = list(P["layers"]), list(R["layers"]), list(K["layers"])
print(f"layers: prefill {len(pl)}, decode {len(rl)}; same names+order: {pl == rl}")
if pl != rl:
    print("  prefill order:", pl[:4], "...\n  decode  order:", rl[:4], "...")

def close(a, b):
    return torch.equal(a, b), (a.float() - b.float()).abs().max().item()

print("\n[1] raw bytes: decode_raw[L] == prefill[L] ?  (transfer fidelity)")
bad = 0
for i, L in enumerate(rl):
    a, b = R["layers"][L], P["layers"][pl[i]]
    if a.shape != b.shape:
        print(f"  layer {i} shape mismatch: decode {tuple(a.shape)} vs prefill {tuple(b.shape)}"); bad += 1; continue
    eq, mx = close(a, b)
    if not eq:
        bad += 1
        # does it match some OTHER prefill layer? (layer permutation)
        other = [j for j, Lp in enumerate(pl) if torch.equal(a, P["layers"][Lp])]
        print(f"  layer {i} ({L}): DIFFERS max|d|={mx:.3g}; equals prefill layer(s) {other}")
print("  all raw layers identical" if bad == 0 else f"  {bad} layers differ")

print("\n[2] packed: decode_packed[L] == pack(prefill[L]) ?  (post-process fidelity)")
bad = 0
for i, L in enumerate(kl):
    src = P["layers"][pl[i]].clone()          # (nblk, H, B, 2D) as sent
    n = src.shape[0]
    buf = torch.zeros((n,) + tuple(src.shape[1:]), dtype=src.dtype)
    buf[:] = src
    current_platform.pack_kv_cache(kv_cache=buf, indices=torch.arange(n))
    eq, mx = close(K["layers"][L], buf)
    if not eq:
        bad += 1
        print(f"  layer {i} ({L}): DIFFERS max|d|={mx:.3g}")
print("  all packed layers as expected" if bad == 0 else f"  {bad} layers differ")

print("\n[3] sanity on prefill content (block 0 of first layer): is it all zeros / how many nonzero rows?")
b0 = P["layers"][pl[0]][0].float()            # (H, B, 2D)
D = b0.shape[-1] // 2
tok_nonzero = (b0.abs().sum(dim=(0, 2)) > 0)  # per token
print(f"  tokens with nonzero KV: {int(tok_nonzero.sum())}/{b0.shape[1]}  "
      f"K|V split check: mean|K|={b0[..., :D].abs().mean():.4f} mean|V|={b0[..., D:].abs().mean():.4f}")
