# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
import sys
import types

import pytest
import torch
from vllm.platforms import current_platform

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


SAGE_ATTN3_MODULE = "vllm_omni.diffusion.attention.backends.sage_attn3"


def load_sage_attn3_module(monkeypatch: pytest.MonkeyPatch, kernel_impl):
    fake_module = types.ModuleType("sageattn3")
    fake_module.sageattn3_blackwell = kernel_impl
    monkeypatch.setitem(sys.modules, "sageattn3", fake_module)
    sys.modules.pop(SAGE_ATTN3_MODULE, None)
    return importlib.import_module(SAGE_ATTN3_MODULE)


def packed_metadata(
    backend_module,
    *,
    valid_length: int,
    physical_length: int,
    dtype: torch.dtype = torch.int32,
):
    cu_seqlens = torch.tensor(
        [0, valid_length, physical_length],
        dtype=dtype,
    )
    return backend_module.AttentionMetadata(
        extra={
            "cu_seqlens_q": cu_seqlens,
            "cu_seqlens_k": cu_seqlens,
            "max_seqlen_q": valid_length,
            "max_seqlen_k": valid_length,
            "valid_kv_length": valid_length,
        }
    )


def make_impl(backend_module):
    return backend_module.SageAttention3Impl(
        num_heads=4,
        head_size=64,
        softmax_scale=1.0 / 8.0,
        causal=False,
    )


def validation_backend(monkeypatch: pytest.MonkeyPatch):
    def unexpected_kernel(*args, **kwargs):
        raise AssertionError("invalid metadata must fail before kernel dispatch")

    backend_module = load_sage_attn3_module(monkeypatch, unexpected_kernel)
    return backend_module, make_impl(backend_module)


def test_sage_attn3_forward_uses_blackwell_layout(monkeypatch: pytest.MonkeyPatch):
    calls = {}

    def fake_kernel(query, key, value, is_causal=False):
        calls["query_shape"] = query.shape
        calls["is_causal"] = is_causal
        return query + key + value

    backend_module = load_sage_attn3_module(monkeypatch, fake_kernel)
    impl = make_impl(backend_module)

    query = torch.randn(2, 8, 4, 64)
    key = torch.randn(2, 8, 4, 64)
    value = torch.randn(2, 8, 4, 64)

    output = impl.forward_cuda(query, key, value)

    assert calls["query_shape"] == (2, 4, 8, 64)
    assert calls["is_causal"] is False
    expected = (query.transpose(1, 2) + key.transpose(1, 2) + value.transpose(1, 2)).transpose(1, 2)
    assert torch.allclose(output, expected)


def test_sage_attn3_packed_prefix_excludes_and_restores_padding(
    monkeypatch: pytest.MonkeyPatch,
):
    calls = {}

    def fake_kernel(query, key, value, is_causal=False):
        calls["shapes"] = (query.shape, key.shape, value.shape)
        return query + key + value

    backend_module = load_sage_attn3_module(monkeypatch, fake_kernel)
    assert backend_module.SageAttention3Backend.supports_packed_prefix_slicing
    impl = make_impl(backend_module)
    query = torch.randn(2, 8, 4, 64)
    key = torch.randn_like(query)
    value = torch.randn_like(query)
    metadata = packed_metadata(
        backend_module,
        valid_length=5,
        physical_length=8,
    )

    output = impl.forward_cuda(query, key, value, metadata)

    assert calls["shapes"] == 3 * ((2, 4, 5, 64),)
    torch.testing.assert_close(
        output[:, :5],
        query[:, :5] + key[:, :5] + value[:, :5],
    )
    assert output.shape == query.shape
    assert torch.count_nonzero(output[:, 5:]) == 0


def test_sage_attn3_packed_gqa_fallback_excludes_padding(
    monkeypatch: pytest.MonkeyPatch,
):
    def fake_kernel(*args, **kwargs):
        raise AssertionError("packed GQA must use exact SDPA")

    backend_module = load_sage_attn3_module(monkeypatch, fake_kernel)
    calls = {}

    def fake_sdpa(query, key, value, **kwargs):
        calls["shapes"] = (query.shape, key.shape, value.shape)
        calls["enable_gqa"] = kwargs["enable_gqa"]
        return query + 1

    monkeypatch.setattr(
        backend_module.F,
        "scaled_dot_product_attention",
        fake_sdpa,
    )
    impl = make_impl(backend_module)
    query = torch.randn(2, 8, 4, 64)
    key = torch.randn(2, 8, 2, 64)
    value = torch.randn_like(key)

    output = impl.forward_cuda(
        query,
        key,
        value,
        packed_metadata(
            backend_module,
            valid_length=5,
            physical_length=8,
        ),
    )

    assert calls["shapes"] == (
        (2, 4, 5, 64),
        (2, 2, 5, 64),
        (2, 2, 5, 64),
    )
    assert calls["enable_gqa"] is True
    assert output.shape == query.shape
    assert torch.count_nonzero(output[:, 5:]) == 0


@pytest.mark.parametrize(
    ("remove", "updates", "message"),
    [
        (
            "cu_seqlens_k",
            {},
            "Incomplete packed SAGE_ATTN_3 metadata",
        ),
        (None, {"cu_seqlens_q": [0, 5, 8]}, "must be a torch.Tensor"),
        (
            None,
            {"cu_seqlens_q": torch.tensor([0, 5], dtype=torch.int32)},
            "must contain three boundaries",
        ),
        (
            None,
            {"cu_seqlens_q": torch.tensor([0, 5, 8], dtype=torch.int64)},
            "must use torch.int32",
        ),
        (None, {"max_seqlen_q": True}, "lengths must be plain integers"),
        (None, {"max_seqlen_k": 4}, "equal max and valid lengths"),
        (
            None,
            {
                "cu_seqlens_q": torch.tensor([0, 9, 8], dtype=torch.int32),
                "cu_seqlens_k": torch.tensor([0, 9, 8], dtype=torch.int32),
                "max_seqlen_q": 9,
                "max_seqlen_k": 9,
                "valid_kv_length": 9,
            },
            "within the physical sequence",
        ),
        (
            None,
            {
                "cu_seqlens_q": torch.empty(
                    3,
                    dtype=torch.int32,
                    device="meta",
                )
            },
            "cu_seqlens_q must be on cpu",
        ),
    ],
    ids=(
        "incomplete",
        "non-tensor-cu-seqlens",
        "cu-seqlens-count",
        "cu-seqlens-dtype",
        "length-type",
        "max-mismatch",
        "length-exceeds-physical",
        "cu-seqlens-device",
    ),
)
def test_sage_attn3_rejects_malformed_packed_metadata(
    monkeypatch: pytest.MonkeyPatch,
    remove: str | None,
    updates: dict[str, object],
    message: str,
):
    backend_module, impl = validation_backend(monkeypatch)
    metadata = packed_metadata(
        backend_module,
        valid_length=5,
        physical_length=8,
    )
    if remove is not None:
        metadata.extra.pop(remove)
    metadata.extra.update(updates)
    q = torch.randn(1, 8, 4, 64)

    with pytest.raises(ValueError, match=message):
        impl.forward_cuda(q, q, q, metadata)


@pytest.mark.parametrize(
    ("query_shape", "key_shape", "value_shape", "message"),
    [
        ((1, 8, 64), (1, 8, 64), (1, 8, 64), "4D Q/K/V tensors"),
        (
            (1, 8, 4, 64),
            (1, 7, 4, 64),
            (1, 7, 4, 64),
            "matching Q/K/V batch and sequence dimensions",
        ),
    ],
)
def test_sage_attn3_rejects_invalid_packed_qkv_shapes(
    monkeypatch: pytest.MonkeyPatch,
    query_shape: tuple[int, ...],
    key_shape: tuple[int, ...],
    value_shape: tuple[int, ...],
    message: str,
):
    backend_module, impl = validation_backend(monkeypatch)
    metadata = packed_metadata(
        backend_module,
        valid_length=5,
        physical_length=8,
    )

    with pytest.raises(ValueError, match=message):
        impl.forward_cuda(
            torch.randn(query_shape),
            torch.randn(key_shape),
            torch.randn(value_shape),
            metadata,
        )


@pytest.mark.parametrize("packed", [False, True])
def test_sage_attn3_rejects_attention_masks(
    monkeypatch: pytest.MonkeyPatch,
    packed: bool,
):
    backend_module, impl = validation_backend(monkeypatch)
    metadata = (
        packed_metadata(
            backend_module,
            valid_length=5,
            physical_length=8,
        )
        if packed
        else backend_module.AttentionMetadata()
    )
    metadata.attn_mask = torch.ones(1, 8, dtype=torch.bool)
    q = torch.randn(1, 8, 4, 64)

    with pytest.raises(ValueError, match="does not support arbitrary attention masks"):
        impl.forward_cuda(q, q, q, metadata)


def test_sage_attn3_falls_back_to_sdpa_for_gqa(monkeypatch: pytest.MonkeyPatch):
    def fake_kernel(*args, **kwargs):
        raise AssertionError("sageattn3_blackwell should not be used for GQA")

    backend_module = load_sage_attn3_module(monkeypatch, fake_kernel)
    sdpa_calls = {}

    def fake_sdpa(query, key, value, **kwargs):
        sdpa_calls["query_shape"] = query.shape
        sdpa_calls["key_shape"] = key.shape
        sdpa_calls["enable_gqa"] = kwargs["enable_gqa"]
        return query + 1

    monkeypatch.setattr(backend_module.F, "scaled_dot_product_attention", fake_sdpa)

    impl = make_impl(backend_module)

    query = torch.randn(2, 8, 4, 64)
    key = torch.randn(2, 8, 2, 64)
    value = torch.randn(2, 8, 2, 64)

    output = impl.forward_cuda(query, key, value)

    assert sdpa_calls["query_shape"] == (2, 4, 8, 64)
    assert sdpa_calls["key_shape"] == (2, 2, 8, 64)
    assert sdpa_calls["enable_gqa"] is True
    expected = (query.permute(0, 2, 1, 3) + 1).permute(0, 2, 1, 3)
    assert torch.allclose(output, expected)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="sage_attn3 tests require CUDA platform")
def test_cuda_platform_selects_sage_attn3_alias(monkeypatch: pytest.MonkeyPatch):
    from vllm.platforms.interface import DeviceCapability

    from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum
    from vllm_omni.diffusion.envs import PACKAGES_CHECKER
    from vllm_omni.platforms.cuda import platform as cuda_platform_module
    from vllm_omni.platforms.cuda.platform import CudaOmniPlatform

    original_import_module = importlib.import_module

    monkeypatch.setattr(
        CudaOmniPlatform,
        "get_device_capability",
        classmethod(lambda cls, device_id=0: DeviceCapability(10, 0)),
    )
    monkeypatch.setattr(PACKAGES_CHECKER, "get_packages_info", lambda: {"has_flash_attn": False})
    monkeypatch.setattr(
        cuda_platform_module.importlib,
        "import_module",
        lambda module_name: object() if module_name == "sageattn3" else original_import_module(module_name),
    )

    backend_path = CudaOmniPlatform.get_diffusion_attn_backend_cls("SAGE_ATTN_3", head_size=64)

    assert backend_path == DiffusionAttentionBackendEnum.SAGE_ATTN_3.get_path()


@pytest.mark.skipif(not current_platform.is_cuda(), reason="sage_attn3 tests require CUDA platform")
def test_cuda_platform_falls_back_when_sage_attn3_gpu_is_unsupported(monkeypatch: pytest.MonkeyPatch):
    from vllm.platforms.interface import DeviceCapability

    from vllm_omni.diffusion.attention.backends.registry import DiffusionAttentionBackendEnum
    from vllm_omni.diffusion.envs import PACKAGES_CHECKER
    from vllm_omni.platforms.cuda.platform import CudaOmniPlatform

    monkeypatch.setattr(
        CudaOmniPlatform,
        "get_device_capability",
        classmethod(lambda cls, device_id=0: DeviceCapability(9, 0)),
    )
    monkeypatch.setattr(PACKAGES_CHECKER, "get_packages_info", lambda: {"has_flash_attn": False})

    backend_path = CudaOmniPlatform.get_diffusion_attn_backend_cls("SAGE_ATTN_3", head_size=64)

    assert backend_path == DiffusionAttentionBackendEnum.TORCH_SDPA.get_path()
