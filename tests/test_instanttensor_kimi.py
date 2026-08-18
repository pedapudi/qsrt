from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from instanttensor import Backend
from safetensors.torch import save_file

from qsrt.instanttensor_kimi import (
    InstantTensorKimiLayerLoader,
    InstantTensorLoadConfig,
    OfficialKimiLayerShards,
    decode_grouped_mxfp4_bf16_into,
    decode_mxfp4_bf16,
    decode_strided_grouped_mxfp4_bf16_into,
    load_checkpoint_tensor_cuda,
    load_checkpoint_tensors_cuda,
    release_layer,
)
from qsrt.io.mxfp4 import dequant


def _write_indexed_checkpoint(
    root: Path,
    tensors: dict[str, torch.Tensor],
    *,
    shard: str = "model-00001-of-00001.safetensors",
) -> Path:
    root.mkdir()
    save_file(tensors, root / shard, metadata={"format": "pt"})
    (root / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {name: shard for name in tensors}})
    )
    return root


def test_layer_shard_requires_exclusive_ownership(tmp_path: Path) -> None:
    prefix = "language_model.model.layers.4."
    checkpoint = _write_indexed_checkpoint(
        tmp_path / "checkpoint",
        {
            f"{prefix}norm.weight": torch.ones(4, dtype=torch.bfloat16),
            f"{prefix}proj.weight": torch.ones((4, 4), dtype=torch.bfloat16),
        },
    )
    index = OfficialKimiLayerShards(checkpoint)
    assert index.layer_shard(4).name == "model-00001-of-00001.safetensors"
    assert len(index.layer_names(4)) == 2

    document = json.loads((checkpoint / "model.safetensors.index.json").read_text())
    document["weight_map"]["language_model.embed_tokens.weight"] = (
        "model-00001-of-00001.safetensors"
    )
    (checkpoint / "model.safetensors.index.json").write_text(json.dumps(document))
    with pytest.raises(ValueError, match="does not exclusively own"):
        OfficialKimiLayerShards(checkpoint).layer_shard(4)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_mxfp4_cuda_decoder_is_bit_exact() -> None:
    generator = torch.Generator().manual_seed(13)
    packed = torch.randint(
        0, 256, (7, 64), dtype=torch.uint8, generator=generator
    )
    scale = torch.randint(
        112, 123, (7, 4), dtype=torch.uint8, generator=generator
    )
    expected = dequant(packed, scale).to(torch.bfloat16)
    observed = decode_mxfp4_bf16(packed.cuda(), scale.cuda()).cpu()
    assert torch.equal(expected.view(torch.int16), observed.view(torch.int16))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_grouped_mxfp4_cuda_decoder_is_bit_exact() -> None:
    generator = torch.Generator().manual_seed(29)
    packed = torch.randint(
        0, 256, (7, 19, 64), dtype=torch.uint8, generator=generator
    )
    scale = torch.randint(
        112, 123, (7, 19, 4), dtype=torch.uint8, generator=generator
    )
    expected = dequant(packed, scale).to(torch.bfloat16)
    observed = torch.empty(
        (7, 19, 128), dtype=torch.bfloat16, device="cuda:0"
    )
    decode_grouped_mxfp4_bf16_into(
        packed.cuda(),
        scale.cuda(),
        observed,
    )
    assert torch.equal(
        expected.view(torch.int16),
        observed.cpu().view(torch.int16),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_strided_grouped_mxfp4_decoder_restores_expert_order() -> None:
    experts, rows, columns = 5, 7, 128
    generator = torch.Generator().manual_seed(31)
    packed = torch.randint(
        0,
        256,
        (experts, rows, columns // 2),
        dtype=torch.uint8,
        generator=generator,
    )
    scale = torch.randint(
        112,
        123,
        (experts, rows, columns // 32),
        dtype=torch.uint8,
        generator=generator,
    )
    source_order = torch.tensor([3, 0, 4, 1, 2], dtype=torch.int32)
    packed_elements = packed[0].numel()
    scale_elements = scale[0].numel()
    source_stride = packed_elements + scale_elements + 64
    storage = torch.zeros(
        experts * source_stride,
        dtype=torch.uint8,
        device="cuda:0",
    )
    for ordinal, expert in enumerate(source_order.tolist()):
        offset = ordinal * source_stride
        storage[offset : offset + packed_elements].copy_(
            packed[expert].flatten().cuda()
        )
        storage[
            offset + packed_elements : offset + packed_elements + scale_elements
        ].copy_(scale[expert].flatten().cuda())
    first_packed = storage[:packed_elements].view(rows, columns // 2)
    first_scale = storage[
        packed_elements : packed_elements + scale_elements
    ].view(rows, columns // 32)
    observed = torch.empty(
        (experts, rows, columns),
        dtype=torch.bfloat16,
        device="cuda:0",
    )
    decode_strided_grouped_mxfp4_bf16_into(
        first_packed,
        first_scale,
        source_order.cuda(),
        observed,
        source_expert_stride=source_stride,
    )
    expected = dequant(packed, scale).to(torch.bfloat16)
    assert torch.equal(
        expected.view(torch.int16),
        observed.cpu().view(torch.int16),
    )


class _Expert(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.w1 = torch.nn.Linear(
            64, 4, bias=False, dtype=torch.bfloat16, device="meta"
        )
        self.w2 = torch.nn.Linear(
            64, 4, bias=False, dtype=torch.bfloat16, device="meta"
        )
        self.w3 = torch.nn.Linear(
            64, 4, bias=False, dtype=torch.bfloat16, device="meta"
        )


class _Moe(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.experts = torch.nn.ModuleList([_Expert(), _Expert()])


class _Layer(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.alpha = torch.nn.Parameter(
            torch.empty(3, dtype=torch.float32, device="meta")
        )
        self.block_sparse_moe = _Moe()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_bounded_loader_materializes_and_releases_layer(tmp_path: Path) -> None:
    prefix = "language_model.model.layers.1."
    generator = torch.Generator().manual_seed(17)
    tensors: dict[str, torch.Tensor] = {
        f"{prefix}alpha": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    }
    expected: dict[str, torch.Tensor] = {}
    for expert in range(2):
        for matrix in ("w1", "w2", "w3"):
            base = f"{prefix}block_sparse_moe.experts.{expert}.{matrix}"
            packed = torch.randint(
                0, 256, (4, 32), dtype=torch.uint8, generator=generator
            )
            scale = torch.randint(
                112, 123, (4, 2), dtype=torch.uint8, generator=generator
            )
            tensors[f"{base}.weight_packed"] = packed
            tensors[f"{base}.weight_scale"] = scale
            expected[f"block_sparse_moe.experts.{expert}.{matrix}.weight"] = (
                dequant(packed, scale).to(torch.bfloat16)
            )
    checkpoint = _write_indexed_checkpoint(tmp_path / "checkpoint", tensors)
    module = _Layer()
    loader = InstantTensorKimiLayerLoader(
        checkpoint,
        device="cuda:0",
        config=InstantTensorLoadConfig(
            buffer_size=1 << 20,
            chunk_size=4096,
            concurrency=1,
            io_depth=3,
            backend=Backend.AIO_BUFFERED,
        ),
    )
    stats = loader.load(module, layer=1)
    assert stats.nonexpert_parameters == 1
    assert stats.expert_matrices == 6
    assert torch.equal(module.alpha.cpu(), tensors[f"{prefix}alpha"])
    for name, value in expected.items():
        observed = dict(module.named_parameters())[name].detach().cpu()
        assert torch.equal(observed.view(torch.int16), value.view(torch.int16))

    release_layer(module)
    assert all(parameter.is_meta for parameter in module.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("prototype_only", (False, True))
def test_bounded_loader_decodes_directly_into_grouped_banks(
    tmp_path: Path,
    prototype_only: bool,
) -> None:
    prefix = "language_model.model.layers.1."
    generator = torch.Generator().manual_seed(23)
    tensors: dict[str, torch.Tensor] = {
        f"{prefix}alpha": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
    }
    expected: dict[tuple[int, str], torch.Tensor] = {}
    for expert in range(2):
        for matrix in ("w1", "w2", "w3"):
            base = f"{prefix}block_sparse_moe.experts.{expert}.{matrix}"
            packed = torch.randint(
                0, 256, (4, 32), dtype=torch.uint8, generator=generator
            )
            scale = torch.randint(
                112, 123, (4, 2), dtype=torch.uint8, generator=generator
            )
            tensors[f"{base}.weight_packed"] = packed
            tensors[f"{base}.weight_scale"] = scale
            expected[(expert, matrix)] = dequant(packed, scale).to(torch.bfloat16)
    checkpoint = _write_indexed_checkpoint(tmp_path / "checkpoint", tensors)
    module = _Layer()
    if prototype_only:
        module.block_sparse_moe.experts = torch.nn.ModuleList(
            [module.block_sparse_moe.experts[0]]
        )
    banks = {
        matrix: torch.empty(
            (2, 4, 64), dtype=torch.bfloat16, device="cuda:0"
        )
        for matrix in ("w1", "w2", "w3")
    }
    loader = InstantTensorKimiLayerLoader(
        checkpoint,
        device="cuda:0",
        config=InstantTensorLoadConfig(
            buffer_size=1 << 20,
            chunk_size=4096,
            concurrency=1,
            io_depth=3,
            backend=Backend.AIO_BUFFERED,
        ),
    )

    loader.load(module, layer=1, expert_weight_banks=banks)

    for (expert, matrix), value in expected.items():
        if expert < len(module.block_sparse_moe.experts):
            observed = getattr(module.block_sparse_moe.experts[expert], matrix).weight
            assert observed.is_meta
        assert torch.equal(
            banks[matrix][expert].detach().cpu().view(torch.int16),
            value.view(torch.int16),
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_indexed_tensor_load_owns_result_after_ring_release(tmp_path: Path) -> None:
    name = "language_model.model.embed_tokens.weight"
    expected = torch.arange(48, dtype=torch.float32).reshape(12, 4).to(torch.bfloat16)
    checkpoint = _write_indexed_checkpoint(
        tmp_path / "checkpoint",
        {
            "language_model.lm_head.weight": expected + 100,
            name: expected,
        },
    )
    observed = load_checkpoint_tensor_cuda(
        checkpoint,
        name,
        device="cuda:0",
        config=InstantTensorLoadConfig(
            buffer_size=1 << 20,
            chunk_size=4096,
            concurrency=1,
            io_depth=3,
            backend=Backend.AIO_BUFFERED,
        ),
    )
    torch.empty((1 << 20,), dtype=torch.uint8, device="cuda:0").fill_(19)
    assert torch.equal(observed.cpu(), expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_indexed_tensor_batch_load_owns_shared_shard_results(tmp_path: Path) -> None:
    tensors = {
        "language_model.model.norm.weight": torch.arange(8, dtype=torch.bfloat16),
        "language_model.lm_head.weight": torch.arange(
            24, dtype=torch.bfloat16
        ).reshape(3, 8),
    }
    checkpoint = _write_indexed_checkpoint(tmp_path / "checkpoint", tensors)
    observed = load_checkpoint_tensors_cuda(
        checkpoint,
        tuple(tensors),
        device="cuda:0",
        config=InstantTensorLoadConfig(
            buffer_size=1 << 20,
            chunk_size=4096,
            concurrency=1,
            io_depth=3,
            backend=Backend.AIO_BUFFERED,
        ),
    )
    assert tuple(observed) == tuple(tensors)
    for name, expected in tensors.items():
        assert torch.equal(observed[name].cpu(), expected)
