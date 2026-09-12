# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Heterogeneous NIXL post-process on the CPU_ATTN side.

When a CPU decoder receives KV from a GPU prefiller, the blocks arrive in the
GPU's logical layout - (H, block_size, 2*head_size) per block with [K|V] packed
per token (FlashAttention HND / host xfer buffer) - and
`CpuPlatform.pack_kv_cache` must rewrite them into CPU_ATTN's physical layout
(ISA-specific, e.g. AMX tile packing). This checks that the rewrite is
bit-identical to what the backend's own `cpu_attn_reshape_and_cache` produces.
"""

import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cpu(), reason="requires the vLLM CPU build (_C CPU ops)"
)


@pytest.mark.parametrize("head_size", [64, 128])
@pytest.mark.parametrize("block_size", [32, 128])
def test_pack_kv_cache_matches_backend_layout(head_size: int, block_size: int):
    from vllm._custom_ops import cpu_attn_reshape_and_cache
    from vllm.v1.attention.backends.cpu_attn import _get_attn_isa

    num_blocks, num_kv_heads = 6, 8
    dtype = torch.bfloat16
    isa = _get_attn_isa(dtype, block_size, head_size)

    indices = torch.tensor([1, 3, 4], dtype=torch.long)
    num_tokens = indices.numel() * block_size
    torch.manual_seed(0)
    key = torch.randn(num_tokens, num_kv_heads, head_size, dtype=dtype)
    value = torch.randn(num_tokens, num_kv_heads, head_size, dtype=dtype)
    slot_mapping = (
        torch.arange(block_size).reshape(1, block_size)
        + indices.reshape(-1, 1) * block_size
    ).flatten()

    # Ground truth: the backend's own cache write.
    ref = torch.zeros(num_blocks, num_kv_heads, block_size, 2 * head_size, dtype=dtype)
    key_cache, value_cache = ref.view(
        num_blocks, num_kv_heads, 2 * block_size, head_size
    ).chunk(2, dim=2)
    cpu_attn_reshape_and_cache(key, value, key_cache, value_cache, slot_mapping, isa)

    # Candidate: bytes as sent by a GPU prefiller, then the receive post-process.
    recv = torch.zeros_like(ref)
    packed = torch.cat([key, value], dim=-1)  # (T, H, 2D): [K|V] per token
    packed = packed.view(indices.numel(), block_size, num_kv_heads, 2 * head_size)
    recv[indices] = packed.permute(0, 2, 1, 3)  # (nblk, H, B, 2D) == HND
    current_platform.pack_kv_cache(kv_cache=recv, indices=indices)

    assert torch.equal(recv, ref)
    untouched = [i for i in range(num_blocks) if i not in indices.tolist()]
    assert bool((recv[untouched] == 0).all())
