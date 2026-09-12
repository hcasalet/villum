# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Metadata dataclasses and helpers for the NIXL connector."""

import re
from dataclasses import dataclass, field
from typing import Any

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.utils import BlockIds, EngineId
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorHandshakeMetadata,
    KVConnectorMetadata,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

TransferHandle = int
ReqId = str

GET_META_MSG = b"get_meta_msg"

# Push-mode (WRITE-based) registration notification.
# Sent worker-to-worker over NIXL: D worker -> P worker, encoded as
# PUSH_REG_NOTIF_PREFIX + msgpack(registration_data).
PUSH_REG_NOTIF_PREFIX = b"PUSH_REG:"
#
# NIXL Connector Version
#
# Increment this version whenever there is an incompatible change to:
#   - NixlAgentMetadata schema
#   - kv_transfer_params schema or semantics
#   - NIXL transfer protocol or wire format
#   - KV cache memory layout or block organization
#   - Any other change that breaks P/D interoperability
#
# Version History:
#   1: Initial version with compatibility checking
#   2: Add remote_request_id to kv_transfer_params
#   3: Add physical_blocks_per_logical_kv_block to NixlAgentMetadata
#   4: Add KV block lease renewal through heartbeats
#   5: Add remote_blocks_expiry_time to kv_transfer_params + handshake
#      clock-sync timestamp
#   6: Validate EAGLE/MTP speculative configuration compatibility
#
NIXL_CONNECTOR_VERSION: int = 6


@dataclass
class NixlAgentMetadata:
    engine_id: str
    agent_metadata: bytes
    kv_caches_base_addr: list[int]
    device_id: int
    num_blocks: int
    block_lens: list[int]
    kv_cache_layout: str
    block_size: int
    ssm_sizes: tuple[int, int]
    attn_backend_name: str
    physical_blocks_per_logical_kv_block: int
    # NIXL memory type ("VRAM"/"DRAM") the advertised kv_caches_base_addr live in.
    # A peer must build transfer descriptors for *our* regions with *our* type;
    # with heterogeneous hardware (GPU prefill, CPU decode) it differs from its own.
    # Defaults to "" for wire-compat with peers that predate this field, in which
    # case the reader falls back to its own memory type (homogeneous assumption).
    nixl_memory_type: str = ""
    # Layer names in the order the KV regions (kv_caches_base_addr/block_lens)
    # were registered. Regions are matched by position, so both sides must
    # register in the same order; see NixlConnectorWorker._canonical_kv_caches.
    # Empty for peers predating this field or for packed single-region caches.
    layer_names: list[str] = field(default_factory=list)


@dataclass
class NixlHandshakePayload(KVConnectorHandshakeMetadata):
    """
    Wrapper for NIXL handshake sent over the wire.

    Enables two-phase decoding for graceful compatibility checking:
    1. Decode NixlHandshakePayload to get compatibility_hash
    2. Compute local hash and compare
    3. Only if hashes match, decode agent_metadata_bytes

    This prevents decoder errors when NixlAgentMetadata schema is
    incompatible, allowing graceful failure with clear error message.
    """

    compatibility_hash: str
    agent_metadata_bytes: bytes  # NixlAgentMetadata encoded


def _get_speculative_compatibility_factors(
    vllm_config: VllmConfig,
) -> dict[str, Any] | None:
    """Return NIXL compatibility factors for hidden-state-based speculators."""
    speculative_config = vllm_config.speculative_config
    if speculative_config is None or not speculative_config.use_eagle():
        return None

    draft_model_config = speculative_config.draft_model_config
    assert draft_model_config is not None
    auxiliary_layer_ids = getattr(
        draft_model_config.hf_config,
        "eagle_aux_hidden_state_layer_ids",
        None,
    )

    # kv_cache_dtype is a user override that defaults to None, meaning "inherit
    # the target's --kv-cache-dtype". Resolve it to the effective value so an
    # explicit setting on one side and inheritance on the other (same effective
    # dtype) don't spuriously mismatch.
    kv_cache_dtype = (
        speculative_config.kv_cache_dtype or vllm_config.cache_config.cache_dtype
    )

    # Note: the draft attention_backend is intentionally not hashed. Its only
    # transfer-relevant effect is the KV block layout/size, which is validated
    # per region at runtime in _validate_remote_agent_handshake. The connector
    # only sees the raw override here (usually None = auto-select), never the
    # resolved backend, so hashing it would cause false mismatches without
    # catching anything the runtime layout check misses.
    return {
        "method": speculative_config.method,
        "model": draft_model_config.model,
        "revision": draft_model_config.revision,
        "code_revision": draft_model_config.code_revision,
        "parallel_drafting": speculative_config.parallel_drafting,
        "kv_cache_dtype": str(kv_cache_dtype),
        "auxiliary_layer_ids": (
            tuple(auxiliary_layer_ids) if auxiliary_layer_ids is not None else None
        ),
    }


# Matches the setuptools_scm local-version segment appended after the git
# commit hash (e.g. ".precompiled" in "0.1.dev19795+g6d8600b52.precompiled"
# vs ".cpu" in "0.1.dev19795+g6d8600b52.cpu"), so GPU and CPU wheel builds
# of the identical commit normalize to the same string.
_BUILD_VARIANT_RE = re.compile(r"(\+g[0-9a-f]+)(\.\w+)*$")


def _normalize_vllm_version(version: str) -> str:
    """Strip build-variant local-version suffixes (e.g. '.precompiled' vs
    '.cpu') from a setuptools_scm version string, while still keeping the
    git commit hash so a genuinely different commit still mismatches."""
    return _BUILD_VARIANT_RE.sub(r"\1", version)


def compute_nixl_compatibility_hash(
    vllm_config: VllmConfig, attn_backend_name: str, cross_layers_blocks: bool
) -> str:
    """
    Compute compatibility hash for NIXL KV transfer.

    Hash only the factors that affect whether two NIXL instances can
    successfully transfer KV cache data.

    Factors included:
    - vLLM version and NIXL connector version
    - Model architecture (name, dtype, KV heads, layers)
    - KV cache format (dtype, sliding window)
    - Attention backend
    - EAGLE/MTP configuration that affects transferred state

    Note: Factors like tensor_parallel_size, block_size, and kv_cache_layout
    are validated at runtime in _validate_remote_agent_handshake and are not
    included in this hash to support heterogeneous deployments.

    When kv_transfer_config.kv_connector_extra_config["heterogeneous_hardware_disagg"]
    is set to true (must be set on BOTH the producer and consumer legs, since
    each side computes and compares its own hash independently), two
    additional factors are normalized before hashing rather than compared
    as-is:
    - vllm_version: the setuptools_scm build-variant suffix (e.g.
      ".precompiled" vs ".cpu") is stripped, since GPU and CPU wheels built
      from the identical commit otherwise report different version strings.
    - attn_backend_name: excluded from the hash entirely, since a CPU
      decode leg pairing with a GPU prefill leg is expected to report a
      different attention backend. This is already accommodated at runtime
      (see the `enable_heterogeneous_attn_post_process` handling for
      `CPU_ATTN` in base_worker.py's handshake validation), so the hash
      rejecting it earlier was blocking a case vLLM otherwise supports.
    Every other factor (model, dtype, kv head count, cache dtype,
    speculative config, etc.) is still hashed and compared strictly either
    way, so this flag narrows the check rather than disabling it the way
    `enforce_handshake_compat=false` does.

    Note - the set of factors are likely to evolve significantly over
    time to be more or less permissive.

    Returns:
        SHA-256 hex digest
    """
    from vllm import __version__ as vllm_version
    from vllm.config.utils import hash_factors

    model_config = vllm_config.model_config
    cache_config = vllm_config.cache_config
    is_hma_enabled = not vllm_config.scheduler_config.disable_hybrid_kv_cache_manager

    heterogeneous_hw = bool(
        vllm_config.kv_transfer_config
        and vllm_config.kv_transfer_config.get_from_extra_config(
            "heterogeneous_hardware_disagg", False
        )
    )
    hashed_vllm_version = vllm_version
    hashed_attn_backend_name = attn_backend_name
    if heterogeneous_hw:
        hashed_vllm_version = _normalize_vllm_version(vllm_version)
        hashed_attn_backend_name = "heterogeneous"

    factors = {
        # Version compatibility
        "vllm_version": hashed_vllm_version,
        "nixl_connector_version": NIXL_CONNECTOR_VERSION,
        # Model architecture - affects KV cache shape
        "model": model_config.model,
        "dtype": str(model_config.dtype),
        "num_kv_heads": model_config.get_total_num_kv_heads(),
        "head_size": model_config.get_head_size(),
        "num_hidden_layers": model_config.get_total_num_hidden_layers(),
        # Attention backend and KV cache dtype affect memory layout
        "attn_backend_name": hashed_attn_backend_name,
        "cache_dtype": str(cache_config.cache_dtype),
        "cross_layers_blocks": cross_layers_blocks,
        "is_hma_enabled": is_hma_enabled,
        "speculative_config": _get_speculative_compatibility_factors(vllm_config),
    }

    compat_hash = hash_factors(factors)
    logger.debug(
        "NIXL compatibility hash: %s (model=%s, dtype=%s, num_kv_heads=%d, "
        "cache_dtype=%s, attn_backend=%s)",
        compat_hash,
        factors["model"],
        factors["dtype"],
        factors["num_kv_heads"],
        factors["cache_dtype"],
        attn_backend_name,
    )
    if heterogeneous_hw:
        # Full factor dict, in case a mismatch persists after normalization
        # (e.g. cross_layers_blocks differing due to a backend-specific
        # block-layout heuristic) -- diff this against the remote side's
        # log line to find the culprit directly instead of guessing.
        logger.debug(
            "NIXL compatibility factors (heterogeneous_hardware_disagg=true): %s",
            factors,
        )
    return compat_hash


@dataclass
class HeartbeatInfo:
    """Heartbeat data for a single remote engine, sent from D worker to P."""

    req_ids: set[ReqId]
    host: str
    port: int
    tp_size: int
    pp_size: int = 1


@dataclass
class RemoteMeta:
    block_ids: BlockIds
    host: str
    port: int
    engine_id: str
    request_id: str
    blocks_expiry_time: float | None = None


@dataclass
class ReqMeta:
    local_block_ids: BlockIds
    # To be used when logical block size does not match the kernel block size
    local_physical_block_ids: BlockIds
    tp_size: int
    remote: RemoteMeta | None = None
    # Remote block size, discovered during NIXL handshake (push mode).
    remote_block_size: int | None = None
    # Remote producer pipeline-parallel size (push mode, D side).
    pp_size: int = 1


class NixlConnectorMetadata(KVConnectorMetadata):
    def __init__(self):
        self.reqs_to_recv: dict[ReqId, ReqMeta] = {}
        self.reqs_to_save: dict[ReqId, ReqMeta] = {}
        self.reqs_to_send: dict[ReqId, float] = {}
        # The scheduler process's time.perf_counter() when this metadata was
        # built. reqs_to_send deadlines are stamped with the scheduler's
        # clock, which is NOT comparable across processes (perf_counter is
        # process/boot-local): workers must rebase the remaining TTL onto
        # their own clock via this reference. 0.0 = unset (legacy metadata).
        self.scheduler_clock: float = 0.0
        self.reqs_in_batch: set[ReqId] = set()
        self.reqs_not_processed: set[ReqId] = set()
        # Heartbeat data grouped by remote engine, sent by D worker to P.
        self.heartbeat_by_engine: dict[EngineId, HeartbeatInfo] = {}
        # Push mode (D side): registration data the D worker should send to
        # P workers via NIXL notification on this step.
        self.push_registrations: dict[ReqId, dict[str, Any]] = {}
        # Push mode (P side): newly finished request blocks to be matched
        # against pending D registrations on the P worker.
        self.push_finished_blocks: dict[ReqId, BlockIds] = {}

    def _add_new_req(
        self,
        local_block_ids: BlockIds,
        kv_transfer_params: dict[str, Any],
    ) -> ReqMeta:
        return ReqMeta(
            local_block_ids=local_block_ids,
            local_physical_block_ids=local_block_ids,
            # P workers don't need to receive tp_size from proxy here.
            tp_size=kv_transfer_params.get("tp_size", 1),
            remote_block_size=kv_transfer_params.get("remote_block_size"),
            pp_size=kv_transfer_params.get("pp_size", 1),
        )

    def add_new_req_to_save(
        self,
        request_id: ReqId,
        local_block_ids: BlockIds,
        kv_transfer_params: dict[str, Any],
    ):
        self.reqs_to_save[request_id] = self._add_new_req(
            local_block_ids, kv_transfer_params
        )

    def add_new_req_to_recv(
        self,
        request_id: ReqId,
        local_block_ids: BlockIds,
        kv_transfer_params: dict[str, Any],
    ):
        req = self._add_new_req(local_block_ids, kv_transfer_params)
        req.remote = RemoteMeta(
            block_ids=kv_transfer_params["remote_block_ids"],
            engine_id=kv_transfer_params["remote_engine_id"],
            request_id=kv_transfer_params["remote_request_id"],
            host=kv_transfer_params["remote_host"],
            port=kv_transfer_params["remote_port"],
            blocks_expiry_time=kv_transfer_params.get("remote_blocks_expiry_time"),
        )
        self.reqs_to_recv[request_id] = req
