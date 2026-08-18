#pragma once

#include <type_traits>

// Dedicated K2 trellis-quantization (Viterbi) kernel for table-driven SQG
// codebooks with bounded reconstruction labels.  Included by sqg_quantize.cu
// after the E4M3 decode helpers and the generic quantize_tiles_kernel; the
// generic template remains the implementation for every other rate and for
// unbounded control tables.
//
// The kernel evaluates the same tail-biting search as the generic K2 path
// but restructures the inner loop around invariants of the K2 graph:
//
// - Out-edges 4q..4q+3 share one predecessor-cost set {q + 4096k}.  One
//   group therefore needs four shared-memory cost reads for sixteen branch
//   evaluations, and its four 2-bit decisions pack natively into the same
//   traceback byte the backward pass already consumes.
// - Each thread visits a fixed group set on every step, so its slice of the
//   predecessor-major label table (one uint4 per group) is loaded into
//   registers once per block instead of once per step.
// - With |d| bounded by the launcher, |w| clamped at load, and costs
//   renormalized every step, finite path costs provably stay far from the
//   FP16 range limit: the branch metric is at most (64 + 16)^2 = 6400 and a
//   renormalized cost can exceed the running minimum by at most nine step
//   metrics, since every state is reachable from the minimum-cost state
//   within seven steps and the subtracted minimum lags by two steps.  The
//   generic kernel's per-branch saturation clamp is therefore unnecessary.
//   Unreachable states carry +inf, which the fused multiply-add propagates
//   on its own; the only explicit infinity handling left is the constrained
//   first step that injects it.
//
// The renormalization subtracts one uniform value from every state cost:
// the block-wide minimum from two steps earlier, folded in a lag-2 pipeline
// that adds no synchronization, and forced to zero while that minimum is
// still small.  A uniform shift preserves every argmin exactly, and a zero
// shift is subtracted exactly, so ordinary tiles accumulate no extra
// rounding while runaway costs are pulled back near zero well before the
// FP16 range limit.

#define QUANTIZE_TILES_K2_SQG_NUM_THREADS 1024

// Largest reconstruction-label magnitude the bounded-metric arithmetic
// admits.  The launcher must route tables that exceed it (no production SQG
// law does) to the generic kernel.
#define QUANTIZE_TILES_K2_SQG_MAX_LABEL 16.0

__global__ __launch_bounds__(QUANTIZE_TILES_K2_SQG_NUM_THREADS, 1)
void quantize_tiles_k2_sqg_kernel
(
    const float* __restrict__ input_tiles_ptr,
    float* __restrict__ output_tiles_ptr,
    uint16_t* __restrict__ output_indices_ptr,
    uint8_t* __restrict__ temp_edges_ptr,
    int tailbite_context
)
{
    constexpr int num_threads = QUANTIZE_TILES_K2_SQG_NUM_THREADS;
    constexpr int edges = 65536 >> 2;
    constexpr int groups = edges / 4;
    constexpr int groups_per_thread = groups / num_threads;
    constexpr int decisions_per_row = groups;

    extern __shared__ uint8_t shbuf[];
    uint8_t* sh = shbuf;
    half* sh_input_tile = (half*) sh; sh += 256 * sizeof(half);
    half* sh_min = (half*) sh; sh += 32 * sizeof(half);
    int* sh_idx = (int*) sh; sh += 32 * sizeof(int);
    half* sh_warp_mins = (half*) sh; sh += 2 * 32 * sizeof(half);
    half* sh_shift = (half*) sh; sh += 8 * sizeof(half);
    half* temp_costs = (half*) sh;
    half* temp_costs_inc = temp_costs + edges;

    const int tile_idx = blockIdx.x;
    const int thread = threadIdx.x;
    const int lane_id = thread & 31;
    const int warp_id = thread >> 5;
    const float* input_tile = input_tiles_ptr + 256 * tile_idx;
    float* output_tile = output_tiles_ptr + 256 * tile_idx;
    uint16_t* output_indices = output_indices_ptr + 256 * tile_idx;
    uint8_t* temp_edges = temp_edges_ptr + 256 * decisions_per_row * tile_idx;

    if (thread < 256)
    {
        // Every label satisfies |d| <= QUANTIZE_TILES_K2_SQG_MAX_LABEL, so
        // beyond +-64 the branch ordering of (d - w)^2 within a step no
        // longer depends on w; the clamp bounds the arithmetic while the
        // saturating generic kernel loses all branch information once the
        // metric itself saturates.
        const float w = input_tile[thread];
        sh_input_tile[thread] = __float2half_rn(fmaxf(-64.0f, fminf(w, 64.0f)));
    }
    if (thread == 0)
    {
        sh_shift[0] = __ushort_as_half(0);
        sh_shift[1] = __ushort_as_half(0);
    }

    // Group q covers out-edges 4q..4q+3; their sixteen predecessor-major
    // labels are the sixteen contiguous table bytes at uint4 index q.
    uint4 reg_labels[groups_per_thread];
    #pragma unroll
    for (int g = 0; g < groups_per_thread; ++g)
        reg_labels[g] =
            reinterpret_cast<const uint4*>(sqg_e4m3_lut)[thread + g * num_threads];
    __syncthreads();

    auto step = [&](int ri, int step_idx, int pre_state,
                    auto first_tag, auto constrained_tag, auto record_tag)
    {
        constexpr bool FIRST = decltype(first_tag)::value;
        constexpr bool CONSTRAINED = decltype(constrained_tag)::value;
        // Primer steps below the traceback window only need costs; skipping
        // decision selection there removes the comparison mask and branch
        // bookkeeping from a quarter of all steps at full context.
        constexpr bool RECORD = decltype(record_tag)::value;

        if constexpr (!FIRST)
        {
            // Fold the previous step's per-warp minima into the shift that
            // the step after this one will subtract.  Warp 0 works on slots
            // the concurrent writers below do not touch.  The shift stays
            // zero until costs have actually grown: subtracting zero is
            // exact, so quiet tiles pay no extra rounding, while growing
            // costs are renormalized long before the FP16 range limit
            // (bounded by 1024 + 9 * (64 + 16)^2).
            if (warp_id == 0)
            {
                half m = sh_warp_mins[((step_idx - 1) & 1) * 32 + lane_id];
                #pragma unroll
                for (int offset = 16; offset > 0; offset >>= 1)
                    m = __hmin(m, __shfl_down_sync(0xffffffff, m, offset));
                if (lane_id == 0)
                    sh_shift[(step_idx + 1) & 1] =
                        __hge(m, __float2half_rn(1024.0f))
                            ? m : __ushort_as_half(0);
            }
        }

        const half2 w2 = __half2half2(sh_input_tile[ri]);
        const half2 shift2 = __half2half2(sh_shift[step_idx & 1]);
        half local_min = H_INF;

        #pragma unroll
        for (int g = 0; g < groups_per_thread; ++g)
        {
            const int q = thread + g * num_threads;
            const uint4 lab = reg_labels[g];
            half p[4];
            if constexpr (!FIRST)
            {
                p[0] = temp_costs_inc[q];
                p[1] = temp_costs_inc[q + 1 * groups];
                p[2] = temp_costs_inc[q + 2 * groups];
                p[3] = temp_costs_inc[q + 3 * groups];
            }
            half2 m01{}, m23{};
            int k0 = 0, k1 = 0, k2 = 0, k3 = 0;
            #pragma unroll
            for (int k = 0; k < 4; ++k)
            {
                const uint32_t word01 = k < 2 ? lab.x : lab.y;
                const uint32_t word23 = k < 2 ? lab.z : lab.w;
                const half2 d01 = decode_sqg_fp8x2(
                    static_cast<__nv_fp8x2_storage_t>(word01 >> (16 * (k & 1))));
                const half2 d23 = decode_sqg_fp8x2(
                    static_cast<__nv_fp8x2_storage_t>(word23 >> (16 * (k & 1))));
                const half2 dh01 = __hsub2(d01, w2);
                const half2 dh23 = __hsub2(d23, w2);
                half2 c01, c23;
                if constexpr (FIRST)
                {
                    c01 = __hmul2(dh01, dh01);
                    c23 = __hmul2(dh23, dh23);
                    if constexpr (CONSTRAINED)
                    {
                        if (q + (k << 12) != pre_state)
                        {
                            c01 = __half2half2(H_INF);
                            c23 = __half2half2(H_INF);
                        }
                    }
                }
                else
                {
                    const half2 pk2 = __half2half2(p[k]);
                    c01 = __hfma2(dh01, dh01, pk2);
                    c23 = __hfma2(dh23, dh23, pk2);
                }
                if (k == 0)
                {
                    m01 = c01;
                    m23 = c23;
                }
                else if constexpr (RECORD)
                {
                    const unsigned less01 = __hlt2_mask(c01, m01);
                    const unsigned less23 = __hlt2_mask(c23, m23);
                    m01 = __hmin2(c01, m01);
                    m23 = __hmin2(c23, m23);
                    if (less01 & 0xffffu) k0 = k;
                    if (less01 >> 16) k1 = k;
                    if (less23 & 0xffffu) k2 = k;
                    if (less23 >> 16) k3 = k;
                }
                else
                {
                    m01 = __hmin2(c01, m01);
                    m23 = __hmin2(c23, m23);
                }
            }
            m01 = __hsub2(m01, shift2);
            m23 = __hsub2(m23, shift2);
            reinterpret_cast<uint2*>(temp_costs)[q] = uint2{
                half2_uint32(m01).as_uint32, half2_uint32(m23).as_uint32};
            local_min = __hmin(
                local_min,
                __hmin(
                    __hmin(__low2half(m01), __high2half(m01)),
                    __hmin(__low2half(m23), __high2half(m23))));
            if constexpr (RECORD)
                temp_edges[decisions_per_row * ri + q] =
                    (uint8_t) (k0 | (k1 << 2) | (k2 << 4) | (k3 << 6));
        }

        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
            local_min = __hmin(
                local_min, __shfl_down_sync(0xffffffff, local_min, offset));
        if (lane_id == 0)
            sh_warp_mins[(step_idx & 1) * 32 + warp_id] = local_min;
    };

    // The constrained pass records decisions from step 0; the primer only
    // records once inside its traceback window at trace_start.
    auto forward = [&](int roll, int steps, int pre_state, int trace_start,
                       auto constrained_tag)
    {
        constexpr bool CONSTRAINED = decltype(constrained_tag)::value;
        int ri = roll & 255;
        half* t = temp_costs;
        temp_costs = temp_costs_inc;
        temp_costs_inc = t;
        step(ri, 0, pre_state,
             std::integral_constant<bool, true>{}, constrained_tag,
             std::integral_constant<bool, CONSTRAINED>{});
        __syncthreads();
        int i = 1;
        for (; i < min(steps, trace_start); ++i)
        {
            ri = (i + roll) & 255;
            t = temp_costs;
            temp_costs = temp_costs_inc;
            temp_costs_inc = t;
            step(ri, i, -1,
                 std::integral_constant<bool, false>{},
                 std::integral_constant<bool, false>{},
                 std::integral_constant<bool, false>{});
            __syncthreads();
        }
        for (; i < steps; ++i)
        {
            ri = (i + roll) & 255;
            t = temp_costs;
            temp_costs = temp_costs_inc;
            temp_costs_inc = t;
            step(ri, i, -1,
                 std::integral_constant<bool, false>{},
                 std::integral_constant<bool, false>{},
                 std::integral_constant<bool, true>{});
            __syncthreads();
        }
    };

    // Deterministic argmin over the final step's costs, reduced warp-local
    // first.  Only thread 0 consumes the result.
    auto argmin_cost = [&]()
    {
        half local_min = H_INF;
        int local_idx = -1;
        for (int e = thread; e < edges; e += num_threads)
        {
            const half v = temp_costs[e];
            if (__hlt(v, local_min))
            {
                local_min = v;
                local_idx = e;
            }
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1)
        {
            const half other_min = __shfl_down_sync(0xffffffff, local_min, offset);
            const int other_idx = __shfl_down_sync(0xffffffff, local_idx, offset);
            if (__hlt(other_min, local_min))
            {
                local_min = other_min;
                local_idx = other_idx;
            }
        }
        if (lane_id == 0)
        {
            sh_min[warp_id] = local_min;
            sh_idx[warp_id] = local_idx;
        }
        __syncthreads();
        int result = 0;
        if (warp_id == 0)
        {
            half best = sh_min[lane_id];
            result = sh_idx[lane_id];
            #pragma unroll
            for (int offset = 16; offset > 0; offset >>= 1)
            {
                const half other_min = __shfl_down_sync(0xffffffff, best, offset);
                const int other_idx = __shfl_down_sync(0xffffffff, result, offset);
                if (__hlt(other_min, best))
                {
                    best = other_min;
                    result = other_idx;
                }
            }
        }
        __syncthreads();
        return result;
    };

    auto backward = [&](int roll, int steps, bool write, int edge)
    {
        if (thread == 0)
        {
            for (int i = steps - 1; i >= 0; --i)
            {
                const int ri = (i + roll) & 255;
                const uint8_t packed =
                    temp_edges[decisions_per_row * ri + (edge >> 2)];
                const int decision = (packed >> ((edge & 3) * 2)) & 0x03;
                const int prev_edge = (decision << 12) | (edge >> 2);
                const int encoded = (prev_edge << 2) | edge;
                edge = prev_edge;
                if (write)
                {
                    output_indices[ri] = (uint16_t) encoded;
                    output_tile[ri] = __half2float(decode_sqg_k2_scalar(encoded));
                }
                else if (ri == 0) break;
            }
            sh_idx[0] = edge;
            // Decouple the next forward pass from this one's shift history.
            sh_shift[0] = __ushort_as_half(0);
            sh_shift[1] = __ushort_as_half(0);
        }
        __syncthreads();
        return sh_idx[0];
    };

    // Infer a cyclic boundary state from symmetric context around element 0,
    // then run the complete constrained pass; see the generic kernel for the
    // scheme.  The boundary argmin here scores the full primer window.
    const int primer_roll = 256 - tailbite_context;
    const int primer_steps = 2 * tailbite_context;
    forward(primer_roll, primer_steps, -1, tailbite_context,
            std::integral_constant<bool, false>{});
    int end_state = backward(primer_roll, primer_steps, false, argmin_cost());
    forward(0, 256, end_state, 0, std::integral_constant<bool, true>{});
    backward(0, 256, true, end_state);
}
