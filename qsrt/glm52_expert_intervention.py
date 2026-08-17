"""Build dense expert endpoints for bounded GLM-5.2 full-model KLD tests.

The resident model remains the immutable 3.5-bpw EXL3 checkpoint.  This
module decodes selected EXL3 experts, encodes matched uniform-K3 QSRT experts
from the already-present BF16 layer window, and stores both dense FP16
endpoints.  An inference-only runtime can then add

``candidate_expert(x) - resident_exl3_expert(x)``

for routed tokens.  The subtraction preserves every unselected weight and
lets a small expert panel be evaluated without downloading or materializing
the complete BF16 checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from safetensors import safe_open
from safetensors.torch import save_file

from qsrt.correctness import sha256_file
from qsrt.exl3_loader import load_qsrt_encoder
from qsrt.exl3_reference import CODEBOOK_SQG_XOR_CHEB_T12
from qsrt.glm52_pilot import (
    PROJECTIONS,
    IndexedTensorStore,
    ProjectionSpec,
    _expert_path,
    _load_completed_expert,
    _read_json_object,
    _transform_seeds,
    atomic_write_json,
    panel_cells,
    prepare_destination,
    source_tensor_name,
)
from qsrt.glm52_real_weight_benchmark import (
    REAL_WEIGHT_CODEC_BENCHMARK_KIND,
    _r7_projection_key,
    load_frozen_real_weight_panel,
    select_frozen_panel_slice,
    validate_bounded_source_window,
    validate_r7_endpoint_layer,
)
from qsrt.ldlq import SIGMA_REG
from qsrt.qsrt_codec_pilot import encode_uniform_candidate, tensor_sha256
from qsrt.sqg_quantizer import install_sqg_quantizer


INTERVENTION_ARTIFACT_KIND = "qsrt_glm52_dense_expert_intervention_v1"
INTERVENTION_BITS = 3
TAILBITE_CONTEXT = 128


def _shared_scale_key(layer: int, projection: str) -> str:
    prefix = f"model.layers.{layer}.mlp.experts.r7_shared"
    if projection in ("gate_proj", "up_proj"):
        return f"{prefix}.gate_up_suh"
    if projection == "down_proj":
        return f"{prefix}.down_svh"
    raise ValueError(f"unsupported expert projection {projection!r}")


def _require_tensor(
    value: torch.Tensor,
    *,
    name: str,
    dtype: torch.dtype,
    shape: Sequence[int],
) -> None:
    if value.dtype != dtype or tuple(value.shape) != tuple(shape):
        raise TypeError(
            f"{name} must be {dtype} {tuple(shape)}, got "
            f"{value.dtype} {tuple(value.shape)}"
        )


@torch.no_grad()
def decode_r7_projection(
    handle: Any,
    *,
    layer: int,
    expert: int,
    spec: ProjectionSpec,
    bits: int,
    device: torch.device,
    quantizer_module: Any,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Decode one R7 MCG projection to the runtime's dense FP16 endpoint."""

    if isinstance(bits, bool) or not isinstance(bits, int) or bits not in (3, 4, 5):
        raise ValueError("the GLM-5.2 R7 endpoint uses only K3, K4, or K5")
    prefix = _r7_projection_key(layer, expert, spec.name)
    trellis = handle.get_tensor(f"{prefix}.trellis").to(device).contiguous()
    expert_scale = (
        handle.get_tensor(f"{prefix}.{spec.expert_scale}")
        .to(device)
        .contiguous()
    )
    shared_scale_name = _shared_scale_key(layer, spec.name)
    shared_scale = handle.get_tensor(shared_scale_name).to(device).contiguous()
    marker = handle.get_tensor(f"{prefix}.mcg")

    _require_tensor(
        trellis,
        name=f"{prefix}.trellis",
        dtype=torch.int16,
        shape=(
            spec.encoder_shape[0] // 16,
            spec.encoder_shape[1] // 16,
            16 * bits,
        ),
    )
    expert_length = (
        spec.encoder_shape[0]
        if spec.expert_scale == "suh"
        else spec.encoder_shape[1]
    )
    shared_length = (
        spec.encoder_shape[0]
        if spec.shared_scale == "suh"
        else spec.encoder_shape[1]
    )
    _require_tensor(
        expert_scale,
        name=f"{prefix}.{spec.expert_scale}",
        dtype=torch.float16,
        shape=(expert_length,),
    )
    _require_tensor(
        shared_scale,
        name=shared_scale_name,
        dtype=torch.float16,
        shape=(shared_length,),
    )
    if marker.dtype != torch.int32 or marker.ndim != 0:
        raise TypeError(f"{prefix}.mcg must be a scalar I32 tensor")
    marker_unsigned = int(marker.item()) & 0xFFFF_FFFF
    if marker_unsigned != 0xCBAC1FED:
        raise ValueError(f"{prefix}.mcg has an unexpected multiplier")

    suh = shared_scale if spec.shared_scale == "suh" else expert_scale
    svh = shared_scale if spec.shared_scale == "svh" else expert_scale
    regularized = torch.empty(spec.encoder_shape, dtype=torch.float16, device=device)
    quantizer_module.ext.reconstruct(regularized, trellis, bits, True, False)
    # The sealed R7 decoder promotes the reconstructed regularized values and
    # stored scale vectors to FP32 before applying either Hadamard transform.
    # Rounding these operations to FP16 changes the endpoint hash.
    decoded = quantizer_module.preapply_had_l(regularized.float(), 128)
    decoded *= suh.float().unsqueeze(1)
    decoded = quantizer_module.preapply_had_r(decoded, 128)
    decoded *= svh.float().unsqueeze(0)
    reconstruction_kn = decoded.half().contiguous()
    reconstruction = reconstruction_kn.T.contiguous().cpu()
    if tuple(reconstruction.shape) != spec.source_shape:
        raise ValueError(
            f"decoded {prefix} has shape {tuple(reconstruction.shape)}, "
            f"expected {spec.source_shape}"
        )

    trellis_bytes = trellis.numel() * trellis.element_size()
    weight_count = math.prod(spec.source_shape)
    return reconstruction, {
        "profile": "exl3_r7_mcg",
        "rate": bits,
        "trellis_bytes": trellis_bytes,
        "trellis_bpw": trellis_bytes * 8.0 / weight_count,
        "expert_scale_bytes": expert_scale.numel() * expert_scale.element_size(),
        "shared_scale_bytes": shared_scale.numel() * shared_scale.element_size(),
        "mcg_multiplier": marker_unsigned,
        "reconstruction_kn_sha256": tensor_sha256(reconstruction_kn),
    }


def _atomic_save_tensors(path: Path, tensors: Mapping[str, torch.Tensor]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        save_file(
            {name: value.detach().cpu().contiguous() for name, value in tensors.items()},
            temporary,
            metadata={"format": INTERVENTION_ARTIFACT_KIND},
        )
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _dense_expert_path(dest: Path, layer: int, expert: int) -> Path:
    return dest / "experts" / f"layer-{layer:03d}-expert-{expert:03d}.safetensors"


@torch.no_grad()
def build_expert_endpoints(
    *,
    source: IndexedTensorStore,
    endpoint_handle: Any,
    layer: int,
    expert: int,
    exl3_rates: Sequence[int],
    device: torch.device,
    quantizer_module: Any,
    manifest_sha256: str,
    dest: Path,
    expected_roundtrip_sha256: Mapping[str, str],
    permutation_new_to_old: Sequence[int],
) -> dict[str, Any]:
    """Build one selected expert's dense EXL3 and uniform-K3 QSRT endpoints."""

    rates = tuple(int(value) for value in exl3_rates)
    if len(rates) != len(PROJECTIONS) or any(rate not in (3, 4, 5) for rate in rates):
        raise ValueError("EXL3 rates must contain gate, up, and down K3/K4/K5 values")
    permutation = torch.tensor(tuple(int(value) for value in permutation_new_to_old))
    if (
        permutation.dtype != torch.int64
        or permutation.numel() != 2048
        or not torch.equal(torch.sort(permutation).values, torch.arange(2048))
    ):
        raise ValueError("R7 expert permutation must contain each coordinate 0..2047")
    inverse_permutation = torch.argsort(permutation)

    tensors: dict[str, torch.Tensor] = {}
    projections: dict[str, Any] = {}
    total_source_energy = 0.0
    total_exl3_sse = 0.0
    total_qsrt_sse = 0.0
    for spec, exl3_bits in zip(PROJECTIONS, rates, strict=True):
        source_name = source_tensor_name(layer, expert, spec.name)
        source_weight = source.get(source_name)
        if source_weight.dtype != torch.bfloat16 or tuple(source_weight.shape) != spec.source_shape:
            raise TypeError(
                f"{source_name} must be BF16 {spec.source_shape}, got "
                f"{source_weight.dtype} {tuple(source_weight.shape)}"
            )
        exl3_weight, exl3_payload = decode_r7_projection(
            endpoint_handle,
            layer=layer,
            expert=expert,
            spec=spec,
            bits=exl3_bits,
            device=device,
            quantizer_module=quantizer_module,
        )
        expected_hash = expected_roundtrip_sha256.get(spec.name)
        if (
            not isinstance(expected_hash, str)
            or exl3_payload["reconstruction_kn_sha256"] != expected_hash
        ):
            raise ValueError(
                f"decoded {spec.name} does not match the sealed R7 round-trip hash"
            )
        input_seed, output_seed = _transform_seeds(layer, spec)
        encoded = encode_uniform_candidate(
            source_weight,
            bits=INTERVENTION_BITS,
            codebook=CODEBOOK_SQG_XOR_CHEB_T12,
            device=device,
            quantizer_module=quantizer_module,
            input_sign_seed=input_seed,
            output_sign_seed=output_seed,
            scale_scope_key=(
                INTERVENTION_ARTIFACT_KIND,
                layer,
                expert,
                spec.name,
            )
            if spec.name in ("gate_proj", "up_proj")
            else None,
            g_scale_into_sv=spec.name in ("gate_proj", "up_proj"),
            sigma_reg=SIGMA_REG,
            tailbite_context=TAILBITE_CONTEXT,
            ldlq_tf32=True,
        )
        qsrt_weight = encoded.pop("reconstruction")
        source_float = source_weight.float()
        source_energy = float(torch.sum(source_float.double().square()).item())
        # R7 stores every expert in a source-equivalent permuted middle basis.
        # Gate/up rows and down columns must be restored before comparing the
        # dense endpoint with the unpermuted official source tensor.
        if spec.name in ("gate_proj", "up_proj"):
            exl3_for_source_metric = exl3_weight.index_select(0, inverse_permutation)
        else:
            exl3_for_source_metric = exl3_weight.index_select(1, inverse_permutation)
        exl3_sse = float(
            torch.sum(
                (source_float - exl3_for_source_metric.float()).double().square()
            ).item()
        )
        qsrt_sse = float(
            torch.sum((source_float - qsrt_weight.float()).double().square()).item()
        )
        tensors[f"exl3.{spec.name}"] = exl3_weight.half()
        tensors[f"qsrt_k3.{spec.name}"] = qsrt_weight.half()
        projections[spec.name] = {
            "source_tensor": source_name,
            "source_dtype": "BF16",
            "shape": list(spec.source_shape),
            "exl3": {
                "payload": exl3_payload,
                "dense_tensor_sha256": tensor_sha256(exl3_weight.half()),
                "source_relative_sse": exl3_sse / source_energy,
                "stored_coordinate_basis": "sealed_R7_permuted_middle_coordinates",
                "source_metric_basis": "official_unpermuted_middle_coordinates",
            },
            "qsrt_k3": {
                "payload": encoded["payload"],
                "dense_tensor_sha256": tensor_sha256(qsrt_weight.half()),
                "source_relative_sse": qsrt_sse / source_energy,
            },
        }
        total_source_energy += source_energy
        total_exl3_sse += exl3_sse
        total_qsrt_sse += qsrt_sse
        del (
            source_weight,
            exl3_weight,
            exl3_for_source_metric,
            qsrt_weight,
            source_float,
        )
        torch.cuda.empty_cache()

    tensor_path = _dense_expert_path(dest, layer, expert)
    _atomic_save_tensors(tensor_path, tensors)
    return {
        "kind": f"{INTERVENTION_ARTIFACT_KIND}_expert",
        "complete": True,
        "manifest_sha256": manifest_sha256,
        "layer": layer,
        "expert": expert,
        "exl3_rates": list(rates),
        "dense_endpoint_file": tensor_path.name,
        "dense_endpoint_file_bytes": tensor_path.stat().st_size,
        "dense_endpoint_file_sha256": sha256_file(tensor_path),
        "projections": projections,
        "aggregate": {
            "source_energy": total_source_energy,
            "exl3_sse": total_exl3_sse,
            "qsrt_k3_sse": total_qsrt_sse,
            "exl3_source_relative_sse": total_exl3_sse / total_source_energy,
            "qsrt_k3_source_relative_sse": total_qsrt_sse / total_source_energy,
            "qsrt_k3_over_exl3_sse": total_qsrt_sse / total_exl3_sse,
        },
    }


def build_dense_intervention_artifacts(
    *,
    source_root: Path,
    source_inventory_path: Path,
    exl3_endpoint_root: Path,
    panel_manifest_path: Path,
    dest: Path,
    layer: int,
    expert_count: int,
    panel_offset: int,
    device: torch.device,
    exllamav3_root: Path,
    resume: bool = False,
    verify_source_shard_hashes: bool = True,
    verify_exl3_shard_hash: bool = True,
) -> dict[str, Any]:
    """Build a resumable dense endpoint panel from bounded existing inputs."""

    frozen_panel = load_frozen_real_weight_panel(panel_manifest_path, layer=layer)
    experts = select_frozen_panel_slice(
        frozen_panel, offset=panel_offset, expert_count=expert_count
    )
    endpoint = validate_r7_endpoint_layer(
        exl3_endpoint_root,
        layer=layer,
        expert_count=expert_count,
        selected_experts=experts,
        verify_shard_hash=verify_exl3_shard_hash,
    )
    endpoint_rates = {
        int(expert): tuple(int(value) for value in rates)
        for expert, rates in endpoint["selected_rate_patterns"].items()
    }
    expected_rates = {expert: frozen_panel["rate_patterns"][expert] for expert in experts}
    if endpoint_rates != expected_rates:
        raise ValueError("frozen-panel rates do not match the EXL3 endpoint sidecar")
    sidecar = _read_json_object(exl3_endpoint_root / endpoint["shard"].replace(
        ".safetensors", ".json"
    ))
    roundtrip_hashes = sidecar.get("roundtrip_hashes")
    permutations = sidecar.get("permutations")
    if not isinstance(roundtrip_hashes, dict) or not isinstance(permutations, dict):
        raise TypeError("R7 sidecar must contain round-trip hashes and permutations")
    panel = {layer: experts}
    source_inventory = validate_bounded_source_window(
        source_root,
        source_inventory_path,
        panel=panel,
        verify_shard_hashes=verify_source_shard_hashes,
    )
    manifest = {
        "kind": f"{INTERVENTION_ARTIFACT_KIND}_manifest",
        "source": source_inventory,
        "exl3_endpoint": endpoint,
        "frozen_panel": {
            "path": frozen_panel["path"],
            "sha256": frozen_panel["sha256"],
            "selected_offset": panel_offset,
            "selected_count": expert_count,
        },
        "panel": {str(layer): list(experts)},
        "candidate": {
            "profile": "qsrt_sqg_e4m3",
            "codebook": CODEBOOK_SQG_XOR_CHEB_T12,
            "uniform_rate": INTERVENTION_BITS,
            "endpoint_dtype": "FP16",
        },
        "resident_endpoint_dtype": "FP16",
        "resident_coordinate_basis": (
            "sealed per-expert R7 middle-coordinate permutation; functionally "
            "equivalent to the official source expert"
        ),
        "device": str(device),
        "evidence_boundary": (
            "dense endpoints for reversible routed-expert intervention; these "
            "files alone do not measure full-model KLD or task quality"
        ),
        "model_downloads_required": False,
        "complete_bf16_checkpoint_required": False,
    }
    manifest_sha256 = prepare_destination(dest, manifest, resume=resume)
    source = IndexedTensorStore(source_root)
    shard_path = exl3_endpoint_root / endpoint["shard"]
    quantizer_module = load_qsrt_encoder(exllamav3_root)
    install_sqg_quantizer(quantizer_module)
    records: list[dict[str, Any]] = []
    with safe_open(shard_path, framework="pt", device="cpu") as endpoint_handle:
        for index, expert in enumerate(experts, start=1):
            receipt_path = _expert_path(dest, layer, expert)
            if receipt_path.is_file():
                record = _load_completed_expert(
                    receipt_path,
                    layer=layer,
                    expert=expert,
                    manifest_sha256=manifest_sha256,
                    pilot_kind=INTERVENTION_ARTIFACT_KIND,
                )
                dense_path = _dense_expert_path(dest, layer, expert)
                if (
                    not dense_path.is_file()
                    or dense_path.stat().st_size != record["dense_endpoint_file_bytes"]
                    or sha256_file(dense_path) != record["dense_endpoint_file_sha256"]
                ):
                    raise ValueError(f"dense endpoint does not match {receipt_path}")
            else:
                started = time.monotonic()
                record = build_expert_endpoints(
                    source=source,
                    endpoint_handle=endpoint_handle,
                    layer=layer,
                    expert=expert,
                    exl3_rates=endpoint_rates[expert],
                    device=device,
                    quantizer_module=quantizer_module,
                    manifest_sha256=manifest_sha256,
                    dest=dest,
                    expected_roundtrip_sha256={
                        spec.name: roundtrip_hashes[
                            _r7_projection_key(layer, expert, spec.name)
                        ]["reconstruction_sha256"]
                        for spec in PROJECTIONS
                    },
                    permutation_new_to_old=permutations[str(expert)]["new_to_old"],
                )
                record["wall_seconds"] = time.monotonic() - started
                atomic_write_json(receipt_path, record)
            records.append(record)
            print(
                f"[{index:02d}/{len(experts)}] layer {layer} expert {expert}: "
                f"QSRT-K3/EXL3 SSE={record['aggregate']['qsrt_k3_over_exl3_sse']:.6f}",
                flush=True,
            )

    report = {
        "kind": INTERVENTION_ARTIFACT_KIND,
        "status": "complete",
        "manifest_sha256": manifest_sha256,
        "layer": layer,
        "expert_count": len(records),
        "panel": manifest["panel"],
        "dense_endpoint_bytes": sum(
            int(record["dense_endpoint_file_bytes"]) for record in records
        ),
        "experts": records,
        "evidence_boundary": manifest["evidence_boundary"],
    }
    atomic_write_json(dest / "report.json", report)
    return report


def merge_dense_intervention_artifacts(
    *,
    inputs: Sequence[Path],
    dest: Path,
    panel_manifest_path: Path,
    layer: int,
) -> dict[str, Any]:
    """Merge disjoint GPU-built endpoint slices without changing tensor bytes."""

    roots = tuple(path.resolve(strict=True) for path in inputs)
    if len(roots) < 2 or len(set(roots)) != len(roots):
        raise ValueError("merge inputs must contain at least two distinct artifacts")
    manifests = [_read_json_object(root / "manifest.json") for root in roots]
    reports = [_read_json_object(root / "report.json") for root in roots]
    for root, manifest, report in zip(roots, manifests, reports, strict=True):
        manifest_sha256 = hashlib.sha256(
            json.dumps(
                manifest, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest()
        if (
            manifest.get("kind") != f"{INTERVENTION_ARTIFACT_KIND}_manifest"
            or report.get("kind") != INTERVENTION_ARTIFACT_KIND
            or report.get("status") != "complete"
            or report.get("manifest_sha256") != manifest_sha256
            or report.get("layer") != layer
        ):
            raise ValueError(f"invalid dense intervention slice {root}")
    invariant_fields = (
        "candidate",
        "resident_endpoint_dtype",
        "resident_coordinate_basis",
        "device",
        "evidence_boundary",
        "model_downloads_required",
        "complete_bf16_checkpoint_required",
    )
    for field in invariant_fields:
        if any(manifest.get(field) != manifests[0].get(field) for manifest in manifests):
            raise ValueError(f"dense intervention slices disagree on {field!r}")
    source_identity_fields = (
        "model_id",
        "revision",
        "config_sha256",
        "index_sha256",
        "source_inventory_sha256",
    )
    endpoint_identity_fields = (
        "model_id",
        "revision",
        "manifest_sha256",
        "manifest_json_sha256",
        "layer",
        "sidecar_sha256",
        "shard",
        "shard_sha256",
        "allocation_bpw",
    )

    def identity(value: Mapping[str, Any], fields: Sequence[str]) -> dict[str, Any]:
        return {field: value.get(field) for field in fields}

    source_identity = identity(manifests[0]["source"], source_identity_fields)
    uses_direct_exl3_identity = all("exl3_endpoint" in manifest for manifest in manifests)
    uses_input_artifact_identity = all(
        "input_intervention_artifact" in manifest for manifest in manifests
    )
    if uses_direct_exl3_identity == uses_input_artifact_identity:
        raise ValueError(
            "dense intervention slices must share either a direct EXL3 identity "
            "or one input intervention artifact"
        )
    endpoint_identity = (
        identity(manifests[0]["exl3_endpoint"], endpoint_identity_fields)
        if uses_direct_exl3_identity
        else dict(manifests[0]["input_intervention_artifact"])
    )
    if any(
        identity(manifest["source"], source_identity_fields) != source_identity
        for manifest in manifests
    ):
        raise ValueError("dense intervention slices disagree on source identity")
    if uses_direct_exl3_identity:
        endpoint_mismatch = any(
            identity(manifest["exl3_endpoint"], endpoint_identity_fields)
            != endpoint_identity
            for manifest in manifests
        )
    else:
        endpoint_mismatch = any(
            manifest["input_intervention_artifact"] != endpoint_identity
            for manifest in manifests
        )
    if endpoint_mismatch:
        raise ValueError("dense intervention slices disagree on endpoint identity")
    optional_common: dict[str, Any] = {}
    for field in ("activation_capture", "fit_numeric_policy"):
        present = [field in manifest for manifest in manifests]
        if any(present) and not all(present):
            raise ValueError(f"dense intervention slices disagree on {field!r} presence")
        if all(present):
            value = manifests[0][field]
            if any(manifest[field] != value for manifest in manifests):
                raise ValueError(f"dense intervention slices disagree on {field!r}")
            optional_common[field] = value

    records: list[dict[str, Any]] = []
    seen: set[int] = set()
    inputs_record: list[dict[str, Any]] = []
    for root, manifest, report in zip(roots, manifests, reports, strict=True):
        for record in report["experts"]:
            expert = int(record["expert"])
            if expert in seen:
                raise ValueError(f"dense intervention slices repeat expert {expert}")
            seen.add(expert)
            dense_path = root / "experts" / record["dense_endpoint_file"]
            if (
                not dense_path.is_file()
                or dense_path.stat().st_size != record["dense_endpoint_file_bytes"]
                or sha256_file(dense_path) != record["dense_endpoint_file_sha256"]
            ):
                raise ValueError(f"dense endpoint for expert {expert} failed closure")
            records.append(dict(record))
        inputs_record.append(
            {
                "root": str(root),
                "manifest_sha256": report["manifest_sha256"],
                "report_sha256": sha256_file(root / "report.json"),
                "experts": [int(record["expert"]) for record in report["experts"]],
            }
        )
    records.sort(key=lambda record: int(record["expert"]))
    frozen_panel = load_frozen_real_weight_panel(panel_manifest_path, layer=layer)
    expected_experts = tuple(frozen_panel["experts"][: len(records)])
    if {int(record["expert"]) for record in records} != set(expected_experts):
        raise ValueError("merged experts do not equal the frozen-panel prefix")

    manifest = {
        "kind": f"{INTERVENTION_ARTIFACT_KIND}_manifest",
        "composition": "verified_disjoint_dense_endpoint_slices",
        "source_identity": source_identity,
        (
            "exl3_endpoint_identity"
            if uses_direct_exl3_identity
            else "input_intervention_artifact"
        ): endpoint_identity,
        "input_artifacts": inputs_record,
        "frozen_panel": {
            "path": frozen_panel["path"],
            "sha256": frozen_panel["sha256"],
            "selected_offset": 0,
            "selected_count": len(records),
        },
        "panel": {str(layer): list(expected_experts)},
        **{field: manifests[0].get(field) for field in invariant_fields},
        **optional_common,
    }
    manifest_sha256 = prepare_destination(dest, manifest, resume=False)
    records_by_expert = {int(record["expert"]): record for record in records}
    for root, report in zip(roots, reports, strict=True):
        for original in report["experts"]:
            expert = int(original["expert"])
            source_path = root / "experts" / original["dense_endpoint_file"]
            destination_path = _dense_expert_path(dest, layer, expert)
            try:
                os.link(source_path, destination_path)
            except OSError:
                shutil.copyfile(source_path, destination_path)
            record = records_by_expert[expert]
            record["manifest_sha256"] = manifest_sha256
            record["dense_endpoint_file"] = destination_path.name
            atomic_write_json(_expert_path(dest, layer, expert), record)

    merged_records = [records_by_expert[expert] for expert in expected_experts]
    report = {
        "kind": INTERVENTION_ARTIFACT_KIND,
        "status": "complete",
        "composition": manifest["composition"],
        "manifest_sha256": manifest_sha256,
        "layer": layer,
        "expert_count": len(merged_records),
        "panel": manifest["panel"],
        "dense_endpoint_bytes": sum(
            int(record["dense_endpoint_file_bytes"]) for record in merged_records
        ),
        "experts": merged_records,
        "evidence_boundary": manifest["evidence_boundary"],
    }
    experiments = {value.get("experiment") for value in reports}
    if len(experiments) == 1 and None not in experiments:
        report["experiment"] = experiments.pop()
    atomic_write_json(dest / "report.json", report)
    return report


__all__ = [
    "INTERVENTION_ARTIFACT_KIND",
    "build_dense_intervention_artifacts",
    "build_expert_endpoints",
    "decode_r7_projection",
    "merge_dense_intervention_artifacts",
]
