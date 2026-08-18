#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>
#include <torch/extension.h>

#include "exl3_compat/util.cuh"
#include "exl3_compat/quant/codebook.cuh"

// Keep only the table pointer in broadcast-friendly constant memory.  The
// labels themselves remain in ordinary read-only global memory: a warp's
// Viterbi lanes normally request different states, which would serialize a
// 64-KiB constant-memory table despite its attractive nominal footprint.
__device__ __constant__ const uint8_t* sqg_e4m3_lut;
__device__ __constant__ const half* sqg_fp16_lut;

template <>
__device__ inline half decode_3inst<4>(uint32_t state)
{
    const __nv_fp8_storage_t fp8 = sqg_e4m3_lut[state & 0xffffu];
    const __half_raw raw = __nv_cvt_fp8_to_halfraw(fp8, __NV_E4M3);
    return __ushort_as_half(raw.x);
}

template <>
__device__ inline half2 decode_3inst_2<4>(uint32_t state0, uint32_t state1)
{
    // The trellis kernel always asks for an aligned consecutive pair.  Load
    // both E4M3 labels together so a warp consumes every byte in each cache
    // sector instead of issuing separate even- and odd-byte gathers.
    const __nv_fp8x2_storage_t fp8x2 =
        reinterpret_cast<const __nv_fp8x2_storage_t*>(sqg_e4m3_lut)[state0 >> 1];
    const __half2_raw raw = __nv_cvt_fp8x2_to_halfraw2(fp8x2, __NV_E4M3);
    return __halves2half2(__ushort_as_half(raw.x), __ushort_as_half(raw.y));
}

template <>
__device__ inline half decode_3inst<5>(uint32_t state)
{
    return sqg_fp16_lut[state & 0xffffu];
}

template <>
__device__ inline half2 decode_3inst_2<5>(uint32_t state0, uint32_t state1)
{
    return reinterpret_cast<const half2*>(sqg_fp16_lut)[state0 >> 1];
}

__device__ inline half2 decode_sqg_fp8x2(__nv_fp8x2_storage_t fp8x2)
{
    const __half2_raw raw = __nv_cvt_fp8x2_to_halfraw2(fp8x2, __NV_E4M3);
    return __halves2half2(__ushort_as_half(raw.x), __ushort_as_half(raw.y));
}

__device__ inline half2 decode_sqg_k2_pair(const uint2& packed, int predecessor)
{
    const uint32_t word = predecessor < 2 ? packed.x : packed.y;
    return decode_sqg_fp8x2(
        static_cast<__nv_fp8x2_storage_t>(word >> (16 * (predecessor & 1))));
}

__device__ inline half decode_sqg_k2_scalar(uint32_t state)
{
    constexpr int edges = 65536 >> 2;
    const int predecessor = state >> 14;
    const int out_edge = state & (edges - 1);
    const __nv_fp8x2_storage_t pair =
        reinterpret_cast<const __nv_fp8x2_storage_t*>(sqg_e4m3_lut)[
            (out_edge >> 1) * 4 + predecessor];
    const __nv_fp8_storage_t fp8 =
        static_cast<__nv_fp8_storage_t>(pair >> (8 * (out_edge & 1)));
    const __half_raw raw = __nv_cvt_fp8_to_halfraw(fp8, __NV_E4M3);
    return __ushort_as_half(raw.x);
}

__device__ inline half2 decode_sqg_k3_pair(const uint4& packed, int predecessor)
{
    uint32_t word;
    switch (predecessor >> 1)
    {
        case 0: word = packed.x; break;
        case 1: word = packed.y; break;
        case 2: word = packed.z; break;
        default: word = packed.w; break;
    }
    return decode_sqg_fp8x2(
        static_cast<__nv_fp8x2_storage_t>(word >> (16 * (predecessor & 1))));
}

__device__ inline half2 decode_sqg_k4_pair(
    const uint4& packed0, const uint4& packed1, int predecessor)
{
    const uint4& packed = predecessor < 8 ? packed0 : packed1;
    const int local = predecessor & 7;
    uint32_t word;
    switch (local >> 1)
    {
        case 0: word = packed.x; break;
        case 1: word = packed.y; break;
        case 2: word = packed.z; break;
        default: word = packed.w; break;
    }
    return decode_sqg_fp8x2(
        static_cast<__nv_fp8x2_storage_t>(word >> (16 * (local & 1))));
}

__device__ inline half decode_sqg_k4_scalar(uint32_t state)
{
    constexpr int edges = 65536 >> 4;
    const int predecessor = state >> 12;
    const int out_edge = state & (edges - 1);
    const __nv_fp8x2_storage_t pair =
        reinterpret_cast<const __nv_fp8x2_storage_t*>(sqg_e4m3_lut)[
            (out_edge >> 1) * 16 + predecessor];
    const __nv_fp8_storage_t fp8 =
        static_cast<__nv_fp8_storage_t>(pair >> (8 * (out_edge & 1)));
    const __half_raw raw = __nv_cvt_fp8_to_halfraw(fp8, __NV_E4M3);
    return __ushort_as_half(raw.x);
}

__device__ inline half decode_sqg_k3_scalar(uint32_t state)
{
    constexpr int edges = 65536 >> 3;
    const int predecessor = state >> 13;
    const int out_edge = state & (edges - 1);
    const __nv_fp8x2_storage_t pair =
        reinterpret_cast<const __nv_fp8x2_storage_t*>(sqg_e4m3_lut)[
            (out_edge >> 1) * 8 + predecessor];
    const __nv_fp8_storage_t fp8 =
        static_cast<__nv_fp8_storage_t>(pair >> (8 * (out_edge & 1)));
    const __half_raw raw = __nv_cvt_fp8_to_halfraw(fp8, __NV_E4M3);
    return __ushort_as_half(raw.x);
}

#include "qsrt_quantize_tiles_kernel.cuh"
#include "qsrt_quantize_tiles_k2_kernel.cuh"

namespace
{

constexpr int BAKRON_BLOCK = 16;

__global__ __launch_bounds__(256)
void bakron_left_band_kernel(
    const float* __restrict__ workspace,
    const float* __restrict__ quantized,
    const float* __restrict__ output_factor,
    float* __restrict__ intermediate,
    int output_dimension,
    int input_dimension,
    int output_blocks,
    int input_blocks,
    int first_diagonal,
    int middle_diagonal,
    int last_diagonal,
    int diagonal_capacity,
    int64_t output_factor_batch_stride)
{
    const int batch = blockIdx.y;
    workspace += batch * output_dimension * input_dimension;
    quantized += batch * output_dimension * input_dimension;
    output_factor += batch * output_factor_batch_stride;
    intermediate += batch * output_dimension * input_dimension;
    const int band_diagonal = blockIdx.x / diagonal_capacity;
    const int position = blockIdx.x - band_diagonal * diagonal_capacity;
    const int diagonal = first_diagonal + band_diagonal;
    const int output_begin = max(0, diagonal - input_blocks + 1);
    const int output_end = min(output_blocks, diagonal + 1);
    const int output_block = output_begin + position;
    if (output_block >= output_end) return;
    const int input_block = diagonal - output_block;

    const int source_begin = max(0, first_diagonal - input_block);
    const int source_end = min(output_blocks, middle_diagonal - input_block);
    const int lane = threadIdx.x;
    const int row = lane / BAKRON_BLOCK;
    const int column = lane - row * BAKRON_BLOCK;
    __shared__ float left[BAKRON_BLOCK * BAKRON_BLOCK];
    __shared__ float error[BAKRON_BLOCK * BAKRON_BLOCK];
    float accumulator = 0.0f;

    for (int source = source_begin; source < source_end; ++source)
    {
        left[lane] = output_factor[
            (output_block * BAKRON_BLOCK + row) * output_dimension
            + source * BAKRON_BLOCK + column];
        const int error_row = source * BAKRON_BLOCK + row;
        const int error_column = input_block * BAKRON_BLOCK + column;
        const int error_offset = error_row * input_dimension + error_column;
        error[lane] = quantized[error_offset] - workspace[error_offset];
        __syncthreads();
#pragma unroll
        for (int inner = 0; inner < BAKRON_BLOCK; ++inner)
        {
            accumulator = fmaf(
                left[row * BAKRON_BLOCK + inner],
                error[inner * BAKRON_BLOCK + column],
                accumulator);
        }
        __syncthreads();
    }

    intermediate[
        (output_block * BAKRON_BLOCK + row) * input_dimension
        + input_block * BAKRON_BLOCK + column] = accumulator;
}

__global__ __launch_bounds__(256)
void bakron_right_band_kernel(
    float* __restrict__ workspace,
    const float* __restrict__ input_factor,
    const float* __restrict__ intermediate,
    int output_dimension,
    int input_dimension,
    int output_blocks,
    int input_blocks,
    int first_diagonal,
    int middle_diagonal,
    int last_diagonal,
    int diagonal_capacity,
    int64_t input_factor_batch_stride)
{
    const int batch = blockIdx.y;
    workspace += batch * output_dimension * input_dimension;
    input_factor += batch * input_factor_batch_stride;
    intermediate += batch * output_dimension * input_dimension;
    const int band_diagonal = blockIdx.x / diagonal_capacity;
    const int position = blockIdx.x - band_diagonal * diagonal_capacity;
    const int diagonal = middle_diagonal + band_diagonal;
    const int output_begin = max(0, diagonal - input_blocks + 1);
    const int output_end = min(output_blocks, diagonal + 1);
    const int output_block = output_begin + position;
    if (output_block >= output_end) return;
    const int input_block = diagonal - output_block;

    const int source_begin = max(0, first_diagonal - output_block);
    const int source_end = min(
        input_blocks,
        min(middle_diagonal, last_diagonal - output_block));
    const int lane = threadIdx.x;
    const int row = lane / BAKRON_BLOCK;
    const int column = lane - row * BAKRON_BLOCK;
    __shared__ float left[BAKRON_BLOCK * BAKRON_BLOCK];
    __shared__ float right[BAKRON_BLOCK * BAKRON_BLOCK];
    float accumulator = 0.0f;

    for (int source = source_begin; source < source_end; ++source)
    {
        left[lane] = intermediate[
            (output_block * BAKRON_BLOCK + row) * input_dimension
            + source * BAKRON_BLOCK + column];
        right[lane] = input_factor[
            (input_block * BAKRON_BLOCK + row) * input_dimension
            + source * BAKRON_BLOCK + column];
        __syncthreads();
#pragma unroll
        for (int inner = 0; inner < BAKRON_BLOCK; ++inner)
        {
            accumulator = fmaf(
                left[row * BAKRON_BLOCK + inner],
                right[column * BAKRON_BLOCK + inner],
                accumulator);
        }
        __syncthreads();
    }

    workspace[
        (output_block * BAKRON_BLOCK + row) * input_dimension
        + input_block * BAKRON_BLOCK + column] += accumulator;
}

} // namespace

namespace
{

constexpr int PERIODIC_THREADS = 1024;
constexpr int PERIODIC_MAX_STATES = 1 << 14;
constexpr int PERIODIC_TRACE_STRIDE = PERIODIC_MAX_STATES / 2;

__device__ inline half decode_periodic_label(
    const uint8_t* codebooks, int bits, uint16_t codeword)
{
    const __nv_fp8_storage_t fp8 =
        codebooks[(bits - 2) * 65536 + static_cast<int>(codeword)];
    const __half_raw raw = __nv_cvt_fp8_to_halfraw(fp8, __NV_E4M3);
    return __ushort_as_half(raw.x);
}

__device__ inline half2 decode_periodic_label_pair(
    const uint8_t* codebooks, int bits, uint16_t codeword)
{
    const __nv_fp8x2_storage_t fp8 =
        reinterpret_cast<const __nv_fp8x2_storage_t*>(
            codebooks + (bits - 2) * 65536)[codeword >> 1];
    const __half2_raw raw = __nv_cvt_fp8x2_to_halfraw2(fp8, __NV_E4M3);
    return __halves2half2(__ushort_as_half(raw.x), __ushort_as_half(raw.y));
}

__global__ __launch_bounds__(PERIODIC_THREADS, 1)
void quantize_tiles_periodic_kernel(
    const float* __restrict__ input_tiles,
    float* __restrict__ output_tiles,
    uint16_t* __restrict__ output_indices,
    uint8_t* __restrict__ temp_edges_base,
    const uint8_t* __restrict__ codebooks,
    const uint8_t* __restrict__ rates,
    int tailbite_context)
{
    extern __shared__ uint8_t shared[];
    half* input = reinterpret_cast<half*>(shared);
    half* costs0 = input + 256;
    half* costs1 = costs0 + PERIODIC_MAX_STATES;
    int* scalar = reinterpret_cast<int*>(costs1 + PERIODIC_MAX_STATES);

    const int tile = blockIdx.x;
    const int thread = threadIdx.x;
    const float* tile_input = input_tiles + tile * 256;
    float* tile_output = output_tiles + tile * 256;
    uint16_t* tile_indices = output_indices + tile * 256;
    uint8_t* trace = temp_edges_base + tile * 256 * PERIODIC_TRACE_STRIDE;
    const uint8_t* tile_rates = rates + tile * 256;

    if (thread < 256) input[thread] = __float2half_rn(tile_input[thread]);
    __syncthreads();

    auto forward = [&](int roll, int steps, int pre_state, int trace_start)
    {
        half* previous = costs0;
        half* current = costs1;
        for (int step = 0; step < steps; ++step)
        {
            const int position = (roll + step) & 255;
            const int next_position = (position + 1) & 255;
            const int current_bits = static_cast<int>(tile_rates[position]);
            const int next_bits = static_cast<int>(tile_rates[next_position]);
            const int next_states = 1 << (16 - next_bits);
            const int predecessor_count = 1 << next_bits;
            const int predecessor_shift = 16 - next_bits;
            const half2 target = __half2half2(input[position]);

            for (int next_state = 2 * thread;
                 next_state < next_states;
                 next_state += 2 * PERIODIC_THREADS)
            {
                const uint16_t first_word = static_cast<uint16_t>(next_state);
                half2 values = decode_periodic_label_pair(
                    codebooks, current_bits, first_word);
                half2 residual = __hsub2(values, target);
                int predecessor0 = next_state >> current_bits;
                int predecessor1 = (next_state + 1) >> current_bits;
                half2 best;
                if (step == 0)
                {
                    half2 local = __hmin2(
                        __hmul2(residual, residual), __half2half2(H_MAX_FINITE));
                    const half cost0 = predecessor0 == pre_state || pre_state < 0
                        ? __low2half(local) : H_INF;
                    const half cost1 = predecessor1 == pre_state || pre_state < 0
                        ? __high2half(local) : H_INF;
                    best = __halves2half2(cost0, cost1);
                }
                else
                {
                    const half prior0 = previous[predecessor0];
                    const half prior1 = previous[predecessor1];
                    const half2 prior = __halves2half2(prior0, prior1);
                    best = __heq(prior0, H_INF) && __heq(prior1, H_INF)
                        ? __half2half2(H_INF)
                        : __hmin2(
                            __hfma2(residual, residual, prior),
                            __half2half2(H_MAX_FINITE));
                    if (__heq(prior0, H_INF))
                        best = __halves2half2(H_INF, __high2half(best));
                    if (__heq(prior1, H_INF))
                        best = __halves2half2(__low2half(best), H_INF);
                }
                int decision0 = 0;
                int decision1 = 0;

                #pragma unroll
                for (int decision = 1; decision < 16; ++decision)
                {
                    if (decision >= predecessor_count) break;
                    const uint16_t word = static_cast<uint16_t>(
                        (decision << predecessor_shift) | next_state);
                    values = decode_periodic_label_pair(codebooks, current_bits, word);
                    residual = __hsub2(values, target);
                    predecessor0 = static_cast<int>(word) >> current_bits;
                    predecessor1 = (static_cast<int>(word) + 1) >> current_bits;
                    half2 candidate;
                    if (step == 0)
                    {
                        half2 local = __hmin2(
                            __hmul2(residual, residual),
                            __half2half2(H_MAX_FINITE));
                        candidate = __halves2half2(
                            predecessor0 == pre_state || pre_state < 0
                                ? __low2half(local) : H_INF,
                            predecessor1 == pre_state || pre_state < 0
                                ? __high2half(local) : H_INF);
                    }
                    else
                    {
                        const half prior0 = previous[predecessor0];
                        const half prior1 = previous[predecessor1];
                        const half2 prior = __halves2half2(prior0, prior1);
                        candidate = __hmin2(
                            __hfma2(residual, residual, prior),
                            __half2half2(H_MAX_FINITE));
                        if (__heq(prior0, H_INF))
                            candidate = __halves2half2(H_INF, __high2half(candidate));
                        if (__heq(prior1, H_INF))
                            candidate = __halves2half2(__low2half(candidate), H_INF);
                    }
                    const unsigned less = __hlt2_mask(candidate, best);
                    best = __hmin2(candidate, best);
                    if (less & 0xffffu) decision0 = decision;
                    if (less >> 16) decision1 = decision;
                }

                reinterpret_cast<half2*>(current)[next_state >> 1] = best;
                if (step >= trace_start)
                {
                    trace[position * PERIODIC_TRACE_STRIDE + (next_state >> 1)] =
                        static_cast<uint8_t>(decision0 | (decision1 << 4));
                }
            }
            __syncthreads();
            half* swap = previous;
            previous = current;
            current = swap;
        }
        if (thread == 0) scalar[1] = previous == costs0 ? 0 : 1;
        __syncthreads();
    };

    auto argmin = [&](int states)
    {
        if (thread == 0)
        {
            const half* final_costs = scalar[1] == 0 ? costs0 : costs1;
            half best = H_INF;
            int index = 0;
            for (int state = 0; state < states; ++state)
            {
                const half value = final_costs[state];
                if (__hlt(value, best))
                {
                    best = value;
                    index = state;
                }
            }
            scalar[0] = index;
        }
        __syncthreads();
        return scalar[0];
    };

    auto backward = [&](int roll, int steps, bool write, int state)
    {
        if (thread == 0)
        {
            for (int step = steps - 1; step >= 0; --step)
            {
                const int position = (roll + step) & 255;
                const int next_position = (position + 1) & 255;
                const int current_bits = static_cast<int>(tile_rates[position]);
                const int next_bits = static_cast<int>(tile_rates[next_position]);
                const uint8_t packed =
                    trace[position * PERIODIC_TRACE_STRIDE + (state >> 1)];
                const int decision = (packed >> (4 * (state & 1))) & 0x0f;
                const uint16_t codeword = static_cast<uint16_t>(
                    (decision << (16 - next_bits)) | state);
                state = static_cast<int>(codeword) >> current_bits;
                if (write)
                {
                    tile_indices[position] = codeword;
                    tile_output[position] = __half2float(
                        decode_periodic_label(codebooks, current_bits, codeword));
                }
                else if (position == 0)
                {
                    break;
                }
            }
            scalar[0] = state;
        }
        __syncthreads();
        return scalar[0];
    };

    const int primer_roll = 256 - tailbite_context;
    const int primer_steps = 2 * tailbite_context;
    forward(primer_roll, primer_steps, -1, tailbite_context);
    const int primer_end_bits = static_cast<int>(
        tile_rates[(primer_roll + primer_steps) & 255]);
    int end_state = backward(
        primer_roll,
        primer_steps,
        false,
        argmin(1 << (16 - primer_end_bits)));
    forward(0, 256, end_state, 0);
    backward(0, 256, true, end_state);
}

void launch_trellis_k2_sqg(
    const float* input,
    float* output,
    uint16_t* indices,
    uint8_t* edges,
    int tiles,
    int max_batch,
    int tailbite_context,
    cudaStream_t stream)
{
    constexpr int shared_memory =
        848 + 2 * (65536 >> 2) * static_cast<int>(sizeof(half));
    cudaFuncSetAttribute(
        quantize_tiles_k2_sqg_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_memory);
    cudaFuncSetAttribute(
        quantize_tiles_k2_sqg_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared);
    for (int start = 0; start < tiles; start += max_batch)
    {
        const int batch = min(max_batch, tiles - start);
        quantize_tiles_k2_sqg_kernel<<<
            batch, QUANTIZE_TILES_K2_SQG_NUM_THREADS, shared_memory, stream>>>(
            input + 256 * start,
            output + 256 * start,
            indices + 256 * start,
            edges,
            tailbite_context);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

template <int K, int codebook>
void launch_trellis(
    const float* input,
    float* output,
    uint16_t* indices,
    half* costs,
    uint8_t* edges,
    int tiles,
    int max_batch,
    int shared_memory,
    int tailbite_context,
    cudaStream_t stream)
{
    const int threads =
        K == 2 ? 1024 : (K == 3 ? 640 : (K == 4 ? 704 : 512));
    cudaFuncSetAttribute(
        quantize_tiles_kernel<K, codebook>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_memory);
    cudaFuncSetAttribute(
        quantize_tiles_kernel<K, codebook>,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        K == 4 ? cudaSharedmemCarveoutMaxL1 : cudaSharedmemCarveoutMaxShared);
    for (int start = 0; start < tiles; start += max_batch)
    {
        const int batch = min(max_batch, tiles - start);
        quantize_tiles_kernel<K, codebook><<<batch, threads, shared_memory, stream>>>(
            input + 256 * start,
            output + 256 * start,
            indices + 256 * start,
            costs,
            edges,
            tailbite_context);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

}  // namespace

void quantize_tiles_periodic_cuda(
    at::Tensor input_tiles,
    at::Tensor output_tiles,
    at::Tensor output_indices,
    at::Tensor temp_edges,
    at::Tensor codebooks,
    at::Tensor rates,
    int64_t tailbite_context)
{
    const at::cuda::OptionalCUDAGuard guard(input_tiles.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    TORCH_CHECK(input_tiles.is_cuda() && input_tiles.scalar_type() == at::kFloat,
                "input_tiles must be CUDA FP32");
    TORCH_CHECK(input_tiles.dim() == 2 && input_tiles.size(1) == 256,
                "input_tiles must have shape [tiles, 256]");
    TORCH_CHECK(output_tiles.sizes() == input_tiles.sizes() &&
                output_tiles.scalar_type() == at::kFloat,
                "output_tiles shape or dtype mismatch");
    TORCH_CHECK(output_indices.sizes() == input_tiles.sizes() &&
                output_indices.scalar_type() == at::kShort,
                "output_indices shape or dtype mismatch");
    TORCH_CHECK(codebooks.is_cuda() && codebooks.device() == input_tiles.device() &&
                codebooks.scalar_type() == at::kByte &&
                codebooks.dim() == 2 && codebooks.size(0) == 3 &&
                codebooks.size(1) == 65536 && codebooks.is_contiguous(),
                "codebooks must be contiguous CUDA uint8 [3, 65536]");
    TORCH_CHECK(rates.is_cuda() && rates.device() == input_tiles.device() &&
                rates.scalar_type() == at::kByte &&
                rates.sizes() == input_tiles.sizes() && rates.is_contiguous(),
                "rates must be contiguous CUDA uint8 [tiles, 256]");
    TORCH_CHECK(temp_edges.is_cuda() && temp_edges.device() == input_tiles.device() &&
                temp_edges.scalar_type() == at::kByte && temp_edges.dim() == 3 &&
                temp_edges.size(1) == 256 &&
                temp_edges.size(2) == PERIODIC_TRACE_STRIDE,
                "temp_edges must be CUDA uint8 [batch, 256, 8192]");
    TORCH_CHECK(tailbite_context >= 1 && tailbite_context <= 128,
                "tailbite_context must be in 1..128");
    TORCH_CHECK(input_tiles.numel() == 0 || temp_edges.size(0) > 0,
                "periodic quantizer needs a nonempty traceback batch");

    const int shared_memory =
        256 * static_cast<int>(sizeof(half)) +
        2 * PERIODIC_MAX_STATES * static_cast<int>(sizeof(half)) +
        2 * static_cast<int>(sizeof(int));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        quantize_tiles_periodic_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_memory));
    C10_CUDA_CHECK(cudaFuncSetAttribute(
        quantize_tiles_periodic_kernel,
        cudaFuncAttributePreferredSharedMemoryCarveout,
        cudaSharedmemCarveoutMaxShared));

    const int tiles = static_cast<int>(input_tiles.size(0));
    const int max_batch = static_cast<int>(temp_edges.size(0));
    for (int start = 0; start < tiles; start += max_batch)
    {
        const int batch = min(max_batch, tiles - start);
        quantize_tiles_periodic_kernel<<<batch, PERIODIC_THREADS, shared_memory, stream>>>(
            input_tiles.data_ptr<float>() + 256 * start,
            output_tiles.data_ptr<float>() + 256 * start,
            reinterpret_cast<uint16_t*>(output_indices.data_ptr<int16_t>()) +
                256 * start,
            temp_edges.data_ptr<uint8_t>(),
            codebooks.data_ptr<uint8_t>(),
            rates.data_ptr<uint8_t>() + 256 * start,
            static_cast<int>(tailbite_context));
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
}

void quantize_tiles_sqg_cuda(
    at::Tensor input_tiles,
    at::Tensor output_tiles,
    at::Tensor output_indices,
    at::Tensor temp_costs,
    at::Tensor temp_edges,
    at::Tensor codebook,
    int64_t K,
    int64_t tailbite_context,
    double lut_abs_max)
{
    const at::cuda::OptionalCUDAGuard guard(input_tiles.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK(input_tiles.is_cuda(), "input_tiles must be CUDA");
    TORCH_CHECK(input_tiles.scalar_type() == at::kFloat, "input_tiles must be FP32");
    TORCH_CHECK(input_tiles.dim() == 2 && input_tiles.size(1) == 256,
                "input_tiles must have shape [tiles, 256]");
    TORCH_CHECK(output_tiles.sizes() == input_tiles.sizes(),
                "output_tiles shape mismatch");
    TORCH_CHECK(output_tiles.scalar_type() == at::kFloat,
                "output_tiles must be FP32");
    TORCH_CHECK(output_indices.sizes() == input_tiles.sizes(),
                "output_indices shape mismatch");
    TORCH_CHECK(output_indices.scalar_type() == at::kShort,
                "output_indices must be int16");
    TORCH_CHECK(K >= 1 && K <= 6, "SQG validation kernel supports K1--K6");
    TORCH_CHECK(tailbite_context >= 1 && tailbite_context <= 128,
                "tailbite_context must be in 1..128");
    TORCH_CHECK(codebook.is_cuda() && codebook.device() == input_tiles.device(),
                "codebook must be on the input CUDA device");
    TORCH_CHECK(codebook.scalar_type() == at::kByte && codebook.numel() == 65536,
                "codebook must contain 65,536 uint8 E4M3 values");
    TORCH_CHECK(codebook.is_contiguous(), "codebook must be contiguous");

    const int edges_per_state = 65536 >> K;
    const int decision_entries =
        K == 2 ? edges_per_state / 4 : (K <= 4 ? edges_per_state / 2 : edges_per_state);
    TORCH_CHECK(temp_costs.scalar_type() == at::kHalf && temp_costs.dim() == 3,
                "temp_costs must be rank-3 FP16");
    TORCH_CHECK(temp_costs.size(1) == 2 && temp_costs.size(2) == edges_per_state,
                "temp_costs shape mismatch");
    TORCH_CHECK(temp_edges.scalar_type() == at::kByte && temp_edges.dim() == 3,
                "temp_edges must be rank-3 uint8");
    TORCH_CHECK(temp_edges.size(1) == 256 && temp_edges.size(2) == decision_entries,
                "temp_edges shape mismatch");

    const uint8_t* codebook_pointer = codebook.data_ptr<uint8_t>();
    C10_CUDA_CHECK(cudaMemcpyToSymbolAsync(
        sqg_e4m3_lut,
        &codebook_pointer,
        sizeof(codebook_pointer),
        0,
        cudaMemcpyHostToDevice,
        stream));

    int device = -1;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    int multiprocessors = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(
        &multiprocessors, cudaDevAttrMultiProcessorCount, device));
    const int max_batch = min(
        static_cast<int>(temp_costs.size(0)), 3 * multiprocessors);
    const int shared_memory =
        (K >= 2 ? 2 * edges_per_state * static_cast<int>(sizeof(half)) : 0) +
        512 + 64 + 128;
    const int tiles = static_cast<int>(input_tiles.size(0));
    if (!tiles) return;

    switch (K)
    {
        case 1:
            launch_trellis<1, 4>(
                input_tiles.data_ptr<float>(), output_tiles.data_ptr<float>(),
                reinterpret_cast<uint16_t*>(output_indices.data_ptr<int16_t>()),
                reinterpret_cast<half*>(temp_costs.data_ptr<at::Half>()),
                temp_edges.data_ptr<uint8_t>(), tiles, max_batch, shared_memory,
                static_cast<int>(tailbite_context), stream);
            break;
        case 2:
            // The dedicated kernel's bounded-metric arithmetic requires a
            // label magnitude bound; control tables beyond it, and tables
            // whose decoded magnitude is not even finite, use the
            // saturating generic kernel.
            if (std::isfinite(lut_abs_max) && lut_abs_max > 0.0 &&
                lut_abs_max <= QUANTIZE_TILES_K2_SQG_MAX_LABEL)
                launch_trellis_k2_sqg(
                    input_tiles.data_ptr<float>(), output_tiles.data_ptr<float>(),
                    reinterpret_cast<uint16_t*>(output_indices.data_ptr<int16_t>()),
                    temp_edges.data_ptr<uint8_t>(), tiles, max_batch,
                    static_cast<int>(tailbite_context), stream);
            else
                launch_trellis<2, 4>(
                    input_tiles.data_ptr<float>(), output_tiles.data_ptr<float>(),
                    reinterpret_cast<uint16_t*>(output_indices.data_ptr<int16_t>()),
                    reinterpret_cast<half*>(temp_costs.data_ptr<at::Half>()),
                    temp_edges.data_ptr<uint8_t>(), tiles, max_batch, shared_memory,
                    static_cast<int>(tailbite_context), stream);
            break;
        case 3:
            launch_trellis<3, 4>(
                input_tiles.data_ptr<float>(), output_tiles.data_ptr<float>(),
                reinterpret_cast<uint16_t*>(output_indices.data_ptr<int16_t>()),
                reinterpret_cast<half*>(temp_costs.data_ptr<at::Half>()),
                temp_edges.data_ptr<uint8_t>(), tiles, max_batch, shared_memory,
                static_cast<int>(tailbite_context), stream);
            break;
        case 4:
            launch_trellis<4, 4>(
                input_tiles.data_ptr<float>(), output_tiles.data_ptr<float>(),
                reinterpret_cast<uint16_t*>(output_indices.data_ptr<int16_t>()),
                reinterpret_cast<half*>(temp_costs.data_ptr<at::Half>()),
                temp_edges.data_ptr<uint8_t>(), tiles, max_batch, shared_memory,
                static_cast<int>(tailbite_context), stream);
            break;
        case 5:
            launch_trellis<5, 4>(
                input_tiles.data_ptr<float>(), output_tiles.data_ptr<float>(),
                reinterpret_cast<uint16_t*>(output_indices.data_ptr<int16_t>()),
                reinterpret_cast<half*>(temp_costs.data_ptr<at::Half>()),
                temp_edges.data_ptr<uint8_t>(), tiles, max_batch, shared_memory,
                static_cast<int>(tailbite_context), stream);
            break;
        case 6:
            launch_trellis<6, 4>(
                input_tiles.data_ptr<float>(), output_tiles.data_ptr<float>(),
                reinterpret_cast<uint16_t*>(output_indices.data_ptr<int16_t>()),
                reinterpret_cast<half*>(temp_costs.data_ptr<at::Half>()),
                temp_edges.data_ptr<uint8_t>(), tiles, max_batch, shared_memory,
                static_cast<int>(tailbite_context), stream);
            break;
    }
}

void quantize_tiles_procedural_cuda(
    at::Tensor input_tiles,
    at::Tensor output_tiles,
    at::Tensor output_indices,
    at::Tensor temp_costs,
    at::Tensor temp_edges,
    int64_t K,
    int64_t codebook,
    int64_t tailbite_context)
{
    const at::cuda::OptionalCUDAGuard guard(input_tiles.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK(input_tiles.is_cuda(), "input_tiles must be CUDA");
    TORCH_CHECK(input_tiles.scalar_type() == at::kFloat, "input_tiles must be FP32");
    TORCH_CHECK(input_tiles.dim() == 2 && input_tiles.size(1) == 256,
                "input_tiles must have shape [tiles, 256]");
    TORCH_CHECK(output_tiles.sizes() == input_tiles.sizes(),
                "output_tiles shape mismatch");
    TORCH_CHECK(output_tiles.scalar_type() == at::kFloat,
                "output_tiles must be FP32");
    TORCH_CHECK(output_indices.sizes() == input_tiles.sizes(),
                "output_indices shape mismatch");
    TORCH_CHECK(output_indices.scalar_type() == at::kShort,
                "output_indices must be int16");
    TORCH_CHECK(K >= 2 && K <= 6, "procedural control supports K2--K6");
    TORCH_CHECK(codebook == 1 || codebook == 2,
                "procedural codebook must be 1 (MCG) or 2 (MUL1)");
    TORCH_CHECK(tailbite_context >= 1 && tailbite_context <= 128,
                "tailbite_context must be in 1..128");

    const int edges_per_state = 65536 >> K;
    const int decision_entries =
        K == 2 ? edges_per_state / 4 :
            (K <= 4 ? edges_per_state / 2 : edges_per_state);
    TORCH_CHECK(temp_costs.scalar_type() == at::kHalf && temp_costs.dim() == 3,
                "temp_costs must be rank-3 FP16");
    TORCH_CHECK(temp_costs.size(1) == 2 && temp_costs.size(2) == edges_per_state,
                "temp_costs shape mismatch");
    TORCH_CHECK(temp_edges.scalar_type() == at::kByte && temp_edges.dim() == 3,
                "temp_edges must be rank-3 uint8");
    TORCH_CHECK(temp_edges.size(1) == 256 && temp_edges.size(2) == decision_entries,
                "temp_edges shape mismatch");

    int device = -1;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    int multiprocessors = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(
        &multiprocessors, cudaDevAttrMultiProcessorCount, device));
    const int max_batch = min(
        static_cast<int>(temp_costs.size(0)), 3 * multiprocessors);
    const int shared_memory =
        2 * edges_per_state * static_cast<int>(sizeof(half)) + 512 + 64 + 128;
    const int tiles = static_cast<int>(input_tiles.size(0));
    if (!tiles) return;

#define LAUNCH_PROCEDURAL(K_VALUE, CODEBOOK_VALUE) \
    launch_trellis<K_VALUE, CODEBOOK_VALUE>( \
        input_tiles.data_ptr<float>(), output_tiles.data_ptr<float>(), \
        reinterpret_cast<uint16_t*>(output_indices.data_ptr<int16_t>()), \
        reinterpret_cast<half*>(temp_costs.data_ptr<at::Half>()), \
        temp_edges.data_ptr<uint8_t>(), tiles, max_batch, shared_memory, \
        static_cast<int>(tailbite_context), stream)

    if (codebook == 1)
    {
        switch (K)
        {
            case 2: LAUNCH_PROCEDURAL(2, 1); break;
            case 3: LAUNCH_PROCEDURAL(3, 1); break;
            case 4: LAUNCH_PROCEDURAL(4, 1); break;
            case 5: LAUNCH_PROCEDURAL(5, 1); break;
            case 6: LAUNCH_PROCEDURAL(6, 1); break;
        }
    }
    else
    {
        switch (K)
        {
            case 2: LAUNCH_PROCEDURAL(2, 2); break;
            case 3: LAUNCH_PROCEDURAL(3, 2); break;
            case 4: LAUNCH_PROCEDURAL(4, 2); break;
            case 5: LAUNCH_PROCEDURAL(5, 2); break;
            case 6: LAUNCH_PROCEDURAL(6, 2); break;
        }
    }
#undef LAUNCH_PROCEDURAL
}

void quantize_tiles_fp16_cuda(
    at::Tensor input_tiles,
    at::Tensor output_tiles,
    at::Tensor output_indices,
    at::Tensor temp_costs,
    at::Tensor temp_edges,
    at::Tensor codebook,
    int64_t K,
    int64_t tailbite_context)
{
    const at::cuda::OptionalCUDAGuard guard(input_tiles.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK(input_tiles.is_cuda(), "input_tiles must be CUDA");
    TORCH_CHECK(input_tiles.scalar_type() == at::kFloat, "input_tiles must be FP32");
    TORCH_CHECK(input_tiles.dim() == 2 && input_tiles.size(1) == 256,
                "input_tiles must have shape [tiles, 256]");
    TORCH_CHECK(output_tiles.sizes() == input_tiles.sizes(),
                "output_tiles shape mismatch");
    TORCH_CHECK(output_tiles.scalar_type() == at::kFloat,
                "output_tiles must be FP32");
    TORCH_CHECK(output_indices.sizes() == input_tiles.sizes(),
                "output_indices shape mismatch");
    TORCH_CHECK(output_indices.scalar_type() == at::kShort,
                "output_indices must be int16");
    TORCH_CHECK(K >= 2 && K <= 6,
                "experimental FP16 SQG kernel supports only K2--K6");
    TORCH_CHECK(tailbite_context >= 1 && tailbite_context <= 128,
                "tailbite_context must be in 1..128");
    TORCH_CHECK(codebook.is_cuda() && codebook.device() == input_tiles.device(),
                "codebook must be on the input CUDA device");
    TORCH_CHECK(codebook.scalar_type() == at::kHalf && codebook.numel() == 65536,
                "codebook must contain 65,536 FP16 values");
    TORCH_CHECK(codebook.is_contiguous(), "codebook must be contiguous");

    const int edges_per_state = 65536 >> K;
    const int decision_entries =
        K == 2 ? edges_per_state / 4 :
        (K <= 4 ? edges_per_state / 2 : edges_per_state);
    TORCH_CHECK(temp_costs.scalar_type() == at::kHalf && temp_costs.dim() == 3,
                "temp_costs must be rank-3 FP16");
    TORCH_CHECK(temp_costs.size(1) == 2 && temp_costs.size(2) == edges_per_state,
                "temp_costs shape mismatch");
    TORCH_CHECK(temp_edges.scalar_type() == at::kByte && temp_edges.dim() == 3,
                "temp_edges must be rank-3 uint8");
    TORCH_CHECK(temp_edges.size(1) == 256 && temp_edges.size(2) == decision_entries,
                "temp_edges shape mismatch");

    const half* codebook_pointer =
        reinterpret_cast<const half*>(codebook.data_ptr<at::Half>());
    C10_CUDA_CHECK(cudaMemcpyToSymbolAsync(
        sqg_fp16_lut,
        &codebook_pointer,
        sizeof(codebook_pointer),
        0,
        cudaMemcpyHostToDevice,
        stream));

    int device = -1;
    C10_CUDA_CHECK(cudaGetDevice(&device));
    int multiprocessors = 0;
    C10_CUDA_CHECK(cudaDeviceGetAttribute(
        &multiprocessors, cudaDevAttrMultiProcessorCount, device));
    const int max_batch = min(
        static_cast<int>(temp_costs.size(0)), 3 * multiprocessors);
    const int shared_memory =
        2 * edges_per_state * static_cast<int>(sizeof(half)) + 512 + 64 + 128;
    const int tiles = static_cast<int>(input_tiles.size(0));
    if (!tiles) return;

    if (K == 2)
    {
        launch_trellis<2, 5>(
            input_tiles.data_ptr<float>(), output_tiles.data_ptr<float>(),
            reinterpret_cast<uint16_t*>(output_indices.data_ptr<int16_t>()),
            reinterpret_cast<half*>(temp_costs.data_ptr<at::Half>()),
            temp_edges.data_ptr<uint8_t>(), tiles, max_batch, shared_memory,
            static_cast<int>(tailbite_context), stream);
    }
    else if (K == 3)
    {
        launch_trellis<3, 5>(
            input_tiles.data_ptr<float>(), output_tiles.data_ptr<float>(),
            reinterpret_cast<uint16_t*>(output_indices.data_ptr<int16_t>()),
            reinterpret_cast<half*>(temp_costs.data_ptr<at::Half>()),
            temp_edges.data_ptr<uint8_t>(), tiles, max_batch, shared_memory,
            static_cast<int>(tailbite_context), stream);
    }
    else if (K == 4)
    {
        launch_trellis<4, 5>(
            input_tiles.data_ptr<float>(), output_tiles.data_ptr<float>(),
            reinterpret_cast<uint16_t*>(output_indices.data_ptr<int16_t>()),
            reinterpret_cast<half*>(temp_costs.data_ptr<at::Half>()),
            temp_edges.data_ptr<uint8_t>(), tiles, max_batch, shared_memory,
            static_cast<int>(tailbite_context), stream);
    }
    else if (K == 5)
    {
        launch_trellis<5, 5>(
            input_tiles.data_ptr<float>(), output_tiles.data_ptr<float>(),
            reinterpret_cast<uint16_t*>(output_indices.data_ptr<int16_t>()),
            reinterpret_cast<half*>(temp_costs.data_ptr<at::Half>()),
            temp_edges.data_ptr<uint8_t>(), tiles, max_batch, shared_memory,
            static_cast<int>(tailbite_context), stream);
    }
    else
    {
        launch_trellis<6, 5>(
            input_tiles.data_ptr<float>(), output_tiles.data_ptr<float>(),
            reinterpret_cast<uint16_t*>(output_indices.data_ptr<int16_t>()),
            reinterpret_cast<half*>(temp_costs.data_ptr<at::Half>()),
            temp_edges.data_ptr<uint8_t>(), tiles, max_batch, shared_memory,
            static_cast<int>(tailbite_context), stream);
    }
}

void bakron_block_update_cuda(
    at::Tensor workspace,
    at::Tensor quantized,
    at::Tensor output_factor,
    at::Tensor input_factor,
    at::Tensor intermediate,
    int64_t first_diagonal,
    int64_t middle_diagonal,
    int64_t last_diagonal,
    int64_t block_size)
{
    const at::cuda::OptionalCUDAGuard guard(workspace.device());
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    TORCH_CHECK(workspace.is_cuda(), "workspace must be CUDA");
    TORCH_CHECK(workspace.scalar_type() == at::kFloat, "workspace must be FP32");
    TORCH_CHECK((workspace.dim() == 2 || workspace.dim() == 3)
                    && workspace.is_contiguous(),
                "workspace must be a contiguous matrix or matrix batch");
    TORCH_CHECK(quantized.sizes() == workspace.sizes()
                    && quantized.scalar_type() == at::kFloat
                    && quantized.is_contiguous(),
                "quantized matrix must match workspace");
    TORCH_CHECK(intermediate.sizes() == workspace.sizes()
                    && intermediate.scalar_type() == at::kFloat
                    && intermediate.is_contiguous(),
                "intermediate matrix must match workspace");
    TORCH_CHECK(output_factor.is_cuda()
                    && output_factor.device() == workspace.device()
                    && output_factor.scalar_type() == at::kFloat
                    && (output_factor.dim() == 2 || output_factor.dim() == 3)
                    && output_factor.is_contiguous(),
                "output factor must be a contiguous CUDA FP32 matrix");
    TORCH_CHECK(input_factor.is_cuda()
                    && input_factor.device() == workspace.device()
                    && input_factor.scalar_type() == at::kFloat
                    && (input_factor.dim() == 2 || input_factor.dim() == 3)
                    && input_factor.is_contiguous(),
                "input factor must be a contiguous CUDA FP32 matrix");
    TORCH_CHECK(quantized.device() == workspace.device()
                    && intermediate.device() == workspace.device(),
                "BaKron matrices must share one CUDA device");
    TORCH_CHECK(block_size == BAKRON_BLOCK,
                "fused BaKron updates require 16x16 blocks");

    const int output_dimension = static_cast<int>(workspace.size(-2));
    const int input_dimension = static_cast<int>(workspace.size(-1));
    const int batch = workspace.dim() == 2
        ? 1 : static_cast<int>(workspace.size(0));
    TORCH_CHECK(output_dimension % BAKRON_BLOCK == 0
                    && input_dimension % BAKRON_BLOCK == 0,
                "BaKron dimensions must be divisible by 16");
    TORCH_CHECK(output_factor.size(-2) == output_dimension
                    && output_factor.size(-1) == output_dimension
                    && (output_factor.dim() == 2 || output_factor.size(0) == batch),
                "output factor shape mismatch");
    TORCH_CHECK(input_factor.size(-2) == input_dimension
                    && input_factor.size(-1) == input_dimension
                    && (input_factor.dim() == 2 || input_factor.size(0) == batch),
                "input factor shape mismatch");

    const int64_t output_factor_batch_stride = output_factor.dim() == 2
        ? 0 : output_dimension * output_dimension;
    const int64_t input_factor_batch_stride = input_factor.dim() == 2
        ? 0 : input_dimension * input_dimension;

    const int output_blocks = output_dimension / BAKRON_BLOCK;
    const int input_blocks = input_dimension / BAKRON_BLOCK;
    const int diagonals = output_blocks + input_blocks - 1;
    TORCH_CHECK(first_diagonal >= 0
                    && first_diagonal < middle_diagonal
                    && middle_diagonal < last_diagonal
                    && last_diagonal <= diagonals,
                "invalid recursive BaKron diagonal range");
    const int diagonal_capacity = min(output_blocks, input_blocks);
    const int left_grid =
        static_cast<int>(last_diagonal - first_diagonal) * diagonal_capacity;
    const int right_grid =
        static_cast<int>(last_diagonal - middle_diagonal) * diagonal_capacity;

    bakron_left_band_kernel<<<dim3(left_grid, batch), 256, 0, stream>>>(
        workspace.data_ptr<float>(),
        quantized.data_ptr<float>(),
        output_factor.data_ptr<float>(),
        intermediate.data_ptr<float>(),
        output_dimension,
        input_dimension,
        output_blocks,
        input_blocks,
        static_cast<int>(first_diagonal),
        static_cast<int>(middle_diagonal),
        static_cast<int>(last_diagonal),
        diagonal_capacity,
        output_factor_batch_stride);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    bakron_right_band_kernel<<<dim3(right_grid, batch), 256, 0, stream>>>(
        workspace.data_ptr<float>(),
        input_factor.data_ptr<float>(),
        intermediate.data_ptr<float>(),
        output_dimension,
        input_dimension,
        output_blocks,
        input_blocks,
        static_cast<int>(first_diagonal),
        static_cast<int>(middle_diagonal),
        static_cast<int>(last_diagonal),
        diagonal_capacity,
        input_factor_batch_stride);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
