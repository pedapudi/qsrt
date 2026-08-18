#include <torch/extension.h>

void quantize_tiles_sqg_cuda(
    at::Tensor input_tiles,
    at::Tensor output_tiles,
    at::Tensor output_indices,
    at::Tensor temp_costs,
    at::Tensor temp_edges,
    at::Tensor codebook,
    int64_t K,
    int64_t tailbite_context,
    double lut_abs_max);

void quantize_tiles_procedural_cuda(
    at::Tensor input_tiles,
    at::Tensor output_tiles,
    at::Tensor output_indices,
    at::Tensor temp_costs,
    at::Tensor temp_edges,
    int64_t K,
    int64_t codebook,
    int64_t tailbite_context);

void quantize_tiles_fp16_cuda(
    at::Tensor input_tiles,
    at::Tensor output_tiles,
    at::Tensor output_indices,
    at::Tensor temp_costs,
    at::Tensor temp_edges,
    at::Tensor codebook,
    int64_t K,
    int64_t tailbite_context);

void quantize_tiles_periodic_cuda(
    at::Tensor input_tiles,
    at::Tensor output_tiles,
    at::Tensor output_indices,
    at::Tensor temp_edges,
    at::Tensor codebooks,
    at::Tensor rates,
    int64_t tailbite_context);

void bakron_block_update_cuda(
    at::Tensor workspace,
    at::Tensor quantized,
    at::Tensor output_factor,
    at::Tensor input_factor,
    at::Tensor intermediate,
    int64_t first_diagonal,
    int64_t middle_diagonal,
    int64_t last_diagonal,
    int64_t block_size);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module)
{
    module.def(
        "quantize_tiles_sqg",
        &quantize_tiles_sqg_cuda,
        "Quantize EXL L16 tiles against a supplied E4M3 SQG table");
    module.def(
        "quantize_tiles_procedural",
        &quantize_tiles_procedural_cuda,
        "Quantize L16 tiles against an MCG or MUL1 control codebook");
    module.def(
        "quantize_tiles_fp16",
        &quantize_tiles_fp16_cuda,
        "Quantize L16 tiles against an experimental FP16 state table");
    module.def(
        "quantize_tiles_periodic",
        &quantize_tiles_periodic_cuda,
        "Quantize L16 tiles with a fixed per-symbol K2/K3/K4 schedule");
    module.def(
        "bakron_block_update",
        &bakron_block_update_cuda,
        "Apply one recursive FP32 BaKron block-band update");
}
