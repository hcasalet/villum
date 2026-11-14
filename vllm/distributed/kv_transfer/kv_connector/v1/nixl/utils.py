# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared constants, lazy imports and helpers for the NIXL connector."""

import contextlib
from collections.abc import Iterator
from typing import Any

import regex as re
import torch
import zmq

from vllm.platforms import current_platform
from vllm.utils.network_utils import make_zmq_socket
from vllm.v1.kv_cache_interface import KVCacheSpec, UniformTypeKVCacheSpecs

# Supported platforms and types of kv transfer buffer.
# {device: tuple of supported kv buffer types}
_NIXL_SUPPORTED_DEVICE = {
    "cuda": (
        "cuda",
        "cpu",
    ),
    "tpu": ("cpu",),
    "xpu": (
        "cpu",
        "xpu",
    ),
    "cpu": ("cpu",),
}
# support for oot platform by providing mapping in current_platform
_NIXL_SUPPORTED_DEVICE.update(current_platform.get_nixl_supported_devices())


# TODO: merge with vllm.utils.network_utils.zmq_socket_ctx
@contextlib.contextmanager
def zmq_ctx(socket_type: Any, addr: str) -> Iterator[zmq.Socket]:
    """Context manager for a ZMQ socket"""

    if socket_type not in (zmq.ROUTER, zmq.REQ):
        raise ValueError(f"Unexpected socket type: {socket_type}")

    ctx: zmq.Context | None = None
    try:
        ctx = zmq.Context()  # type: ignore[attr-defined]
        yield make_zmq_socket(
            ctx=ctx, path=addr, socket_type=socket_type, bind=socket_type == zmq.ROUTER
        )
    finally:
        if ctx is not None:
            ctx.destroy(linger=0)


def get_representative_spec_type(spec: KVCacheSpec) -> type[KVCacheSpec]:
    if isinstance(spec, UniformTypeKVCacheSpecs):
        # All inner specs are the same type; pick any.
        inner = next(iter(spec.kv_cache_specs.values()))
        return type(inner)
    return type(spec)


# Trailing 8-hex randomization suffix appended by
# ``input_processor.assign_request_id`` as ``-{random_uuid():.8}``.
_RANDOM_SUFFIX_RE = re.compile(r"-[0-9a-f]{8}$", re.IGNORECASE)


def get_base_request_id(request_id: str) -> str:
    """Strip the per-request ``-<8 hex>`` randomization suffix, if present."""
    return _RANDOM_SUFFIX_RE.sub("", request_id)


def unpack_cpu_attn_amx_kv(kv_cache: torch.Tensor, block_ids: list[int]) -> None:
    """Rewrite blocks read from a CPU_ATTN (AMX, bf16) peer into the fused layout.

    Inverse of ``CpuPlatform.pack_kv_cache``: a CPU decoder keeps each
    (block, head) as an AMX-friendly K region followed by a V region (see
    ``csrc/cpu/cpu_attn_amx.hpp``), while other backends keep (block_size, 2 * D)
    with K|V fused per token. Blocks read back from such a peer are converted in
    place. The peer's ISA is not part of the handshake, so AMX is assumed.

    Args:
        kv_cache: Cache or host buffer of shape (num_blocks, H, block_size, 2 * D).
        block_ids: Blocks of ``kv_cache`` that hold CPU-packed data.
    """
    if kv_cache.dim() != 4 or not block_ids:
        return
    _, num_heads, block_size, fused = kv_cache.shape
    head_size = fused // 2
    if (
        kv_cache.dtype != torch.bfloat16
        or block_size % 32 != 0
        or head_size % 32 != 0
    ):
        raise NotImplementedError(
            "Reading KV back from a CPU_ATTN peer needs bfloat16, block_size and "
            f"head_size divisible by 32; got {kv_cache.dtype}, {block_size}, "
            f"{head_size}."
        )
    idx = torch.as_tensor(block_ids, device=kv_cache.device, dtype=torch.long)
    n = idx.numel()
    packed = kv_cache.index_select(0, idx).reshape(
        n, num_heads, 2, block_size * head_size
    )
    key = (
        packed[:, :, 0]
        .reshape(n, num_heads, block_size // 16, head_size // 2, 16, 2)
        .permute(0, 1, 2, 4, 3, 5)
        .reshape(n, num_heads, block_size, head_size)
    )
    value = (
        packed[:, :, 1]
        .reshape(n, num_heads, head_size // 16, block_size // 2, 16, 2)
        .permute(0, 1, 3, 5, 2, 4)
        .reshape(n, num_heads, block_size, head_size)
    )
    kv_cache.index_copy_(0, idx, torch.cat((key, value), dim=-1))
