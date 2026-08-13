# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
import torch.nn.functional as F
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)

logger = init_logger(__name__)

try:
    from sageattn3 import sageattn3_blackwell  # noqa: F401
except ImportError:
    logger.warning(
        "SageAttention3Backend is not available. Install `sageattn3` from "
        "https://github.com/thu-ml/SageAttention/tree/main/sageattention3_blackwell"
    )
    raise ImportError


# Wrapping sageattn3_blackwell as a torch.library custom op keeps it opaque to
# torch.compile. Otherwise Dynamo graph-breaks on the raw pybind11 kernel and
# Inductor fails scheduling with KeyError: 'op5'. The hasattr guard keeps this
# idempotent across test re-imports that pop the module from sys.modules.
if not hasattr(torch.ops.vllm_omni, "sageattn3_blackwell"):

    @torch.library.custom_op("vllm_omni::sageattn3_blackwell", mutates_args=())
    def _sageattn3_blackwell_op(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        is_causal: bool,
    ) -> torch.Tensor:
        from sageattn3 import sageattn3_blackwell as _kernel

        return _kernel(query, key, value, is_causal=is_causal)

    @_sageattn3_blackwell_op.register_fake
    def _(query, key, value, is_causal):
        return torch.empty_like(query)


_sageattn3_blackwell_op = torch.ops.vllm_omni.sageattn3_blackwell

_PACKED_PREFIX_KEYS = (
    "cu_seqlens_q",
    "cu_seqlens_k",
    "max_seqlen_q",
    "max_seqlen_k",
    "valid_kv_length",
)


class SageAttention3Backend(AttentionBackend):
    accept_output_buffer: bool = True
    supports_packed_prefix_slicing: bool = True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 128, 256]

    @staticmethod
    def get_name() -> str:
        return "SAGE_ATTN_3"

    @staticmethod
    def get_impl_cls() -> type["SageAttention3Impl"]:
        return SageAttention3Impl


class SageAttention3Impl(AttentionImpl):
    _warned_gqa_fallback_global: bool = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        **extra_impl_args,
    ) -> None:
        self.causal = causal
        self.softmax_scale = softmax_scale
        self.dropout = extra_impl_args.get("dropout_p", 0.0)

    @staticmethod
    def _packed_prefix_length(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
    ) -> int | None:
        if attn_metadata is None:
            return None
        if attn_metadata.attn_mask is not None:
            raise ValueError("SAGE_ATTN_3 does not support arbitrary attention masks")

        extra = attn_metadata.extra
        present = [name for name in _PACKED_PREFIX_KEYS if name in extra]
        if not present:
            return None
        if len(present) != len(_PACKED_PREFIX_KEYS):
            missing = sorted(set(_PACKED_PREFIX_KEYS) - set(present))
            raise ValueError(f"Incomplete packed SAGE_ATTN_3 metadata; missing {missing}")
        if any(tensor.ndim != 4 for tensor in (query, key, value)):
            raise ValueError("Packed SAGE_ATTN_3 requires 4D Q/K/V tensors")
        if query.shape[:2] != key.shape[:2] or key.shape[:2] != value.shape[:2]:
            raise ValueError("Packed SAGE_ATTN_3 requires matching Q/K/V batch and sequence dimensions")

        for name, device in (
            ("cu_seqlens_q", query.device),
            ("cu_seqlens_k", key.device),
        ):
            cu_seqlens = extra[name]
            if not isinstance(cu_seqlens, torch.Tensor):
                raise ValueError(f"{name} must be a torch.Tensor")
            if cu_seqlens.ndim != 1 or cu_seqlens.numel() != 3:
                raise ValueError(f"{name} must contain three boundaries")
            if cu_seqlens.dtype != torch.int32:
                raise ValueError(f"{name} must use torch.int32")
            if cu_seqlens.device != device:
                raise ValueError(f"{name} must be on {device}")

        lengths = tuple(extra[name] for name in _PACKED_PREFIX_KEYS[2:])
        if any(type(length) is not int for length in lengths):
            raise ValueError("Packed SAGE_ATTN_3 lengths must be plain integers")
        max_q, max_k, valid_length = lengths
        if max_q != max_k or max_q != valid_length:
            raise ValueError("Packed SAGE_ATTN_3 requires equal max and valid lengths")
        if not 0 < valid_length <= query.shape[1]:
            raise ValueError("valid_kv_length must be within the physical sequence")
        return valid_length

    @staticmethod
    def _restore_physical_length(
        output: torch.Tensor,
        physical_length: int,
    ) -> torch.Tensor:
        padding = physical_length - output.shape[1]
        if padding == 0:
            return output
        suffix = output.new_zeros(
            output.shape[0],
            padding,
            output.shape[2],
            output.shape[3],
        )
        return torch.cat((output, suffix), dim=1)

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        physical_length = query.shape[1]
        valid_length = self._packed_prefix_length(
            query,
            key,
            value,
            attn_metadata,
        )
        if valid_length is not None:
            query = query[:, :valid_length]
            key = key[:, :valid_length]
            value = value[:, :valid_length]

        query = query.transpose(1, 2).contiguous()
        key = key.transpose(1, 2).contiguous()
        value = value.transpose(1, 2).contiguous()

        if key.shape[1] != query.shape[1]:
            if query.shape[1] % key.shape[1] != 0:
                raise ValueError(
                    "GQA/MQA requires query heads to be a multiple of KV heads, "
                    f"got q_heads={query.shape[1]} and kv_heads={key.shape[1]}"
                )
            if not type(self)._warned_gqa_fallback_global:
                logger.warning("SageAttention3 does not support GQA/MQA (Hq != Hkv); falling back to torch SDPA.")
                type(self)._warned_gqa_fallback_global = True
            output = F.scaled_dot_product_attention(
                query,
                key,
                value,
                is_causal=self.causal,
                dropout_p=self.dropout,
                scale=self.softmax_scale,
                enable_gqa=True,
            )
        else:
            output = _sageattn3_blackwell_op(query, key, value, self.causal)

        output = output.transpose(1, 2).contiguous()
        return self._restore_physical_length(output, physical_length)
