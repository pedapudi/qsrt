import pytest

from qsrt.kernel_audit import audit_kernel_path


def test_stock_w4a16_audit_requires_ordinary_packed_kernel() -> None:
    log = "\n".join(
        (
            "Using B12xExperts",
            "B12X MoE force-A16 enabled: using quant_mode=w4a16.",
            "target=some.module.W4A16FusedMoeKernel",
            "attrs={'weight_layout': 'packed'}",
            "repeat implementation=w4a16",
        )
    )

    report = audit_kernel_path(log, "stock-w4a16")

    assert report["pass"] is True


def test_stock_w4a16_audit_rejects_hybrid_trellis() -> None:
    log = "\n".join(
        (
            "Using B12xExperts",
            "B12X MoE force-A16 enabled: using quant_mode=w4a16.",
            "target=some.module.W4A16FusedMoeKernel",
            "attrs={'weight_layout': 'packed'}",
            "repeat implementation=w4a16",
            "quantization=qsrt_hybrid",
            "weight_layout=trellis3_t256",
        )
    )

    report = audit_kernel_path(log, "stock-w4a16")

    assert report["pass"] is False
    assert set(report["unexpected_evidence"]) == {
        "hybrid_quantization",
        "exl3_trellis_layout",
    }


def test_hybrid_audit_requires_exl3_trellis_evidence() -> None:
    log = "\n".join(
        (
            "quantization=qsrt_hybrid",
            "allocated w13_exl3_trellis",
            "target=some.module.W4A16FusedMoeKernel",
            "weight_layout=trellis3_t256",
            "repeat implementation=w4a16",
        )
    )

    assert audit_kernel_path(log, "hybrid-exl3")["pass"] is True


def test_qsrt_audit_requires_canonical_atom_reader() -> None:
    log = "\n".join(
        (
            "quantization=qsrt_hybrid",
            "Loaded QSRT layer 1 shard 0/12: 800 compressed, 96 X4T experts",
            "B12X MoE repeat check: finite=True max_abs=0 "
            "quant_mode=w4a16 implementation=w4a16",
        )
    )

    assert audit_kernel_path(log, "qsrt")["pass"] is True


def test_qsrt_audit_rejects_loader_only_evidence() -> None:
    log = "\n".join(
        (
            "quantization=qsrt_hybrid",
            "Loaded QSRT layer 1 shard 0/12: 800 compressed, 96 X4T experts",
        )
    )

    report = audit_kernel_path(log, "qsrt")

    assert report["pass"] is False
    assert set(report["missing_evidence"]) == {
        "w4a16_runtime",
        "repeat_check_w4a16",
    }


def test_kernel_audit_rejects_unknown_path() -> None:
    with pytest.raises(ValueError, match="unknown kernel path"):
        audit_kernel_path("", "other")
