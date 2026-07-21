/*
 * Standalone CUDA extension for the warp-per-group fast Hadamard transform (FHT)
 * ConvRot row-wise INT8 quantize kernel. Extracted from comfy-kitchen's
 * quantize_int8_rowwise_convrot_warp_kernel (comfy_kitchen/backends/cuda/ops/int8_linear.cu)
 * so it can run standalone, without a comfy-kitchen source fork, as a drop-in
 * replacement for comfy_kitchen.backends.cuda._C.quantize_int8_rowwise_convrot64
 * on the shapes it covers.
 *
 * Invariants:
 *   - Groups of 256 columns are QuaRot-style Hadamard-rotated online (offline-rotated
 *     weight side, per comfy-kitchen's ConvRot design), then row-wise int8-quantized.
 *   - A warp holds four 256-element groups entirely in registers: 32 fp32/thread/group
 *     (v[4][8], 8 registers per group per thread), no row-sized shared memory. Element
 *     e = lane + 32*slot is a group's position: bits 0-4 are the lane, bits 5-7 the slot.
 *   - convrot_warp_h4_combine computes the branchless 4-point Hadamard butterfly combine
 *     (selects among 4 output positions by d, without warp divergence).
 *   - convrot_warp_fht4 applies the transform in 4 butterfly stages over strides
 *     1, 4, 16, 64: strides 1 and 4 vary lane bits only (full-warp lane-XOR shuffles);
 *     stride 16 varies one lane bit plus a slot pairing (one shuffle, local slot swap);
 *     stride 64 varies slot bits only (pure register work, no shuffle).
 *   - Only the non-stochastic path is implemented (round-to-nearest via nearbyintf);
 *     the Python shim never routes stochastic-rounding calls here.
 *   - Quantize arithmetic (scale = max(min(abs_max, dtype_finite_max)/127, 1e-30),
 *     round-to-nearest, clamp to [-128,127]) matches comfy-kitchen's
 *     quantize_int8_rowwise_convrot64 bit-for-bit for the shapes this kernel serves
 *     (K % 1024 == 0, 1024 <= K <= 32768).
 *
 * Portions of this file derive from comfy-kitchen
 * (https://github.com/Comfy-Org/comfy-kitchen), licensed under the Apache License,
 * Version 2.0. See LICENSE and NOTICE in this repository for the full license text
 * and upstream copyright notice.
 * SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
#include <torch/extension.h>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <cfloat>
#include <cstdint>
#include <type_traits>

namespace anima_turbo {

constexpr int kThreadsPerWarp = 32;
constexpr int kConvRotGroup = 256;

template<typename T>
__device__ __forceinline__ float to_float(T val);
template<> __device__ __forceinline__ float to_float<float>(float val) { return val; }
template<> __device__ __forceinline__ float to_float<half>(half val) { return __half2float(val); }
template<> __device__ __forceinline__ float to_float<nv_bfloat16>(nv_bfloat16 val) { return __bfloat162float(val); }

template<typename T>
__device__ __forceinline__ float finite_max_for_dtype();
template<> __device__ __forceinline__ float finite_max_for_dtype<float>() { return FLT_MAX; }
template<> __device__ __forceinline__ float finite_max_for_dtype<half>() { return 65504.0f; }
template<> __device__ __forceinline__ float finite_max_for_dtype<nv_bfloat16>() { return 3.38953139e38f; }

template<typename T>
__device__ __forceinline__ float finite_absmax_for_int8_scale(float abs_max) {
    return fminf(abs_max, finite_max_for_dtype<T>());
}

template<typename T>
__device__ __forceinline__ T from_float(float val);
template<> __device__ __forceinline__ float from_float<float>(float val) { return val; }
template<> __device__ __forceinline__ half from_float<half>(float val) { return __float2half_rn(val); }
template<> __device__ __forceinline__ nv_bfloat16 from_float<nv_bfloat16>(float val) { return __float2bfloat16_rn(val); }

// Round-trips val and scale through the dtype's own finite range before dividing,
// matching stock's dequant/quant path bit-for-bit (a plain fp32 divide is not
// equivalent for half/bfloat16 inputs).
template<typename T>
__device__ __forceinline__ float quant_div_float_to_float(float val, float scale) {
    const float scale_t = to_float(from_float<T>(scale));
    return to_float(from_float<T>(to_float(from_float<T>(val)) / scale_t));
}

// Own quad-position d (0-3) plus its three partner values assemble into the
// canonical (x0,x1,x2,x3) field order the four cases below expect. Branchless:
// d varies per lane within a warp, so a branch on it would diverge the warp.
__device__ __forceinline__ float convrot_warp_h4_combine(int d, float own, float b1, float b2, float b3) {
    const float val0 = own + b1 + b2 - b3;   // d=0: (x0,x1,x2,x3)=(own,b1,b2,b3)
    const float val1 = b1 + own - b3 + b2;   // d=1: (x0,x1,x2,x3)=(b1,own,b3,b2)
    const float val2 = b2 - b3 + own + b1;   // d=2: (x0,x1,x2,x3)=(b2,b3,own,b1)
    const float val3 = -b3 + b2 + b1 + own;  // d=3: (x0,x1,x2,x3)=(b3,b2,b1,own)
    const float val = (d == 0) ? val0 : (d == 1) ? val1 : (d == 2) ? val2 : val3;
    return 0.5f * val;
}

// Four FHT stages (strides 1, 4, 16, 64) over v[4][8], the four 256-element groups
// a warp holds. e = lane + 32*slot is the position within a group.
__device__ __forceinline__ void convrot_warp_fht4(float v[4][8], int lane) {
    #pragma unroll
    for (int g = 0; g < 4; ++g) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int d = lane & 3;
            const float b1 = __shfl_xor_sync(0xffffffff, v[g][j], 1);
            const float b2 = __shfl_xor_sync(0xffffffff, v[g][j], 2);
            const float b3 = __shfl_xor_sync(0xffffffff, v[g][j], 3);
            v[g][j] = convrot_warp_h4_combine(d, v[g][j], b1, b2, b3);
        }
    }

    #pragma unroll
    for (int g = 0; g < 4; ++g) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int d = (lane >> 2) & 3;
            const float b1 = __shfl_xor_sync(0xffffffff, v[g][j], 4);
            const float b2 = __shfl_xor_sync(0xffffffff, v[g][j], 8);
            const float b3 = __shfl_xor_sync(0xffffffff, v[g][j], 12);
            v[g][j] = convrot_warp_h4_combine(d, v[g][j], b1, b2, b3);
        }
    }

    #pragma unroll
    for (int g = 0; g < 4; ++g) {
        #pragma unroll
        for (int j0 = 0; j0 < 8; j0 += 2) {
            const float old_a = v[g][j0];
            const float old_b = v[g][j0 + 1];
            const int da = (lane >> 4) & 1;
            const float a_b1 = __shfl_xor_sync(0xffffffff, old_a, 16);
            const float a_b3 = __shfl_xor_sync(0xffffffff, old_b, 16);
            v[g][j0]     = convrot_warp_h4_combine(da, old_a, a_b1, old_b, a_b3);
            v[g][j0 + 1] = convrot_warp_h4_combine(da | 2, old_b, a_b3, old_a, a_b1);
        }
    }

    #pragma unroll
    for (int g = 0; g < 4; ++g) {
        #pragma unroll
        for (int base = 0; base < 2; ++base) {
            const float e0 = v[g][base];
            const float e2 = v[g][base + 2];
            const float e4 = v[g][base + 4];
            const float e6 = v[g][base + 6];
            v[g][base]     = convrot_warp_h4_combine(0, e0, e2, e4, e6);
            v[g][base + 2] = convrot_warp_h4_combine(1, e2, e0, e6, e4);
            v[g][base + 4] = convrot_warp_h4_combine(2, e4, e6, e0, e2);
            v[g][base + 6] = convrot_warp_h4_combine(3, e6, e4, e2, e0);
        }
    }
}

// Rows per block and warps per row are template/runtime split: warps per row is
// K / 1024 (this kernel instantiation requires K % 1024 == 0), rows per block is
// fixed by the caller based on warps_per_row (see launch_convrot_warp_quantize).
template<typename InputType, int ROWS_PER_BLOCK>
__global__ void quantize_int8_rowwise_convrot_warp_kernel(
    const InputType* __restrict__ x,
    int8_t* __restrict__ q,
    float* __restrict__ scales,
    int K,
    int64_t num_rows)
{
    extern __shared__ float row_warp_max[];  // ROWS_PER_BLOCK * warps_per_row floats.

    const int warps_per_row = K >> 10;
    const int lane = threadIdx.x & 31;
    const int warp_id = static_cast<int>(threadIdx.x >> 5);
    const int local_row = warp_id / warps_per_row;
    const int warp_in_row = warp_id % warps_per_row;
    const int row = static_cast<int>(blockIdx.x) * ROWS_PER_BLOCK + local_row;
    const bool active_row = static_cast<int64_t>(row) < num_rows;
    const int64_t row_offset = static_cast<int64_t>(row) * K;
    const int group_base = warp_in_row * 4;

    float v[4][8];
    #pragma unroll
    for (int g = 0; g < 4; ++g) {
        const int group_col = (group_base + g) * kConvRotGroup;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int e = lane + 32 * j;
            v[g][j] = active_row ? to_float(x[row_offset + group_col + e]) : 0.0f;
        }
    }

    convrot_warp_fht4(v, lane);

    float local_max = 0.0f;
    #pragma unroll
    for (int g = 0; g < 4; ++g) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            local_max = fmaxf(local_max, fabsf(v[g][j]));
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, off));
    }

    float abs_max = local_max;
    if (warps_per_row > 1) {
        row_warp_max[local_row * warps_per_row + warp_in_row] = local_max;
        __syncthreads();
        abs_max = 0.0f;
        #pragma unroll 1
        for (int w = 0; w < warps_per_row; ++w) {
            abs_max = fmaxf(abs_max, row_warp_max[local_row * warps_per_row + w]);
        }
    }

    const float scale = fmaxf(
        finite_absmax_for_int8_scale<InputType>(abs_max) * (1.0f / 127.0f),
        1.0e-30f);
    if (active_row && lane == 0 && warp_in_row == 0) {
        scales[row] = scale;
    }

    if (active_row) {
        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            const int group_col = (group_base + g) * kConvRotGroup;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int e = lane + 32 * j;
                const int64_t idx = row_offset + group_col + e;
                const float scaled = quant_div_float_to_float<InputType>(v[g][j], scale);
                float quantized = nearbyintf(scaled);
                quantized = fminf(127.0f, fmaxf(-128.0f, quantized));
                q[idx] = static_cast<int8_t>(quantized);
            }
        }
    }
}

template<typename InputType>
void launch_for_rows_per_block(
    int rows_per_block,
    int64_t blocks,
    int block_threads,
    size_t smem_bytes,
    const InputType* x,
    int8_t* q,
    float* scales,
    int K,
    int64_t num_rows,
    cudaStream_t stream)
{
    if (rows_per_block == 4) {
        quantize_int8_rowwise_convrot_warp_kernel<InputType, 4>
            <<<static_cast<unsigned int>(blocks), block_threads, smem_bytes, stream>>>(
                x, q, scales, K, num_rows);
    } else if (rows_per_block == 2) {
        quantize_int8_rowwise_convrot_warp_kernel<InputType, 2>
            <<<static_cast<unsigned int>(blocks), block_threads, smem_bytes, stream>>>(
                x, q, scales, K, num_rows);
    } else {
        quantize_int8_rowwise_convrot_warp_kernel<InputType, 1>
            <<<static_cast<unsigned int>(blocks), block_threads, smem_bytes, stream>>>(
                x, q, scales, K, num_rows);
    }
}

// ---------------------------------------------------------------------------
// INT4 addition below: same namespace, reuses kConvRotGroup/kThreadsPerWarp/
// to_float/from_float/finite_max_for_dtype/convrot_warp_h4_combine/convrot_warp_fht4
// declared above unchanged. Warp-per-group design shared with the int8 kernel above;
// only the final quantize (scale = absmax/7, clamp [-7,7]) and 2-per-byte int4 pack
// stages differ. Output matches quantize_int4_rowwise_convrot64_kernel (group_size
// 256) bitwise; stock has no warp-per-group int4 kernel.
//
// Invariants:
//   - fp32 input reuses convrot_warp_h4_combine/convrot_warp_fht4 above unchanged:
//     stock's fp32 rotation path is itself plain per-element float arithmetic with
//     the same term order, verified stage-by-stage against the fp32 combine above.
//   - fp16/bf16 input do NOT reuse the fp32 path. Stock rotates fp16/bf16 rows in
//     native fp16/bf16 arithmetic (paired hadd/hsub, then a scale-by-0.5 hmul) --
//     not fp32 -- so convrot_native_h4_combine/convrot_warp_fht4_native replay that
//     same op sequence in native precision via the header-provided fp16/bf16
//     __shfl_xor_sync overloads (pure data movement, bit-preserving). val_even/
//     val_odd below are the two distinct hadd/hsub expressions stock's four (x0..x3)
//     formulas reduce to once own/b1/b2/b3 are substituted per lane role -- selected
//     by d's parity, never derived from one another (no negation/commutativity
//     shortcuts), to keep operand order identical to stock's own calls.
//   - Quantize: scale = max(min(abs_max, dtype_finite_max)/7, 1e-10), no dtype
//     round-trip on the value (stock's row_buf already holds the dtype-rounded
//     rotated value; only float32 registers need to match that here). Each element
//     rounds via __float2int_rn and clamps to [-7,7] (kInt4Max), matching
//     quantize_int4_value's non-stochastic path.
//   - Pack: byte n holds columns 2n (low nibble) and 2n+1 (high nibble), matching
//     pack_int4_pair. Columns 2n/2n+1 are adjacent lanes at the same slot (e and
//     e+1); the even lane obtains the odd lane's quantized value via one
//     __shfl_xor_sync(mask, q, 1) and writes both nibbles.
// ---------------------------------------------------------------------------

__device__ __forceinline__ int8_t pack_int4_pair_local(int lo, int hi) {
    const uint32_t packed = (static_cast<uint32_t>(lo) & 0x0Fu) | ((static_cast<uint32_t>(hi) & 0x0Fu) << 4);
    return static_cast<int8_t>(packed);
}

// Native fp16/bf16 4-point Hadamard combine (own = x_d, b1/b2/b3 = partners at
// xor 1/2/3 of d's own stride bits). val_even covers d in {0,2}, val_odd covers
// d in {1,3}; each is one of stock's four y0..y3 expressions after substituting
// own/b1/b2/b3 for x0..x3 per d's role (verified by direct substitution, not by
// algebraic identity), so operand order to hadd/hsub always matches stock's.
template<typename T>
__device__ __forceinline__ T convrot_native_h4_combine(int d, T own, T b1, T b2, T b3, T half_val) {
    const T p = __hadd(own, b1);
    const T q_even = __hsub(b2, b3);
    const T q_odd = __hsub(b3, b2);
    const T val_even = __hmul(__hadd(p, q_even), half_val);
    const T val_odd = __hmul(__hsub(p, q_odd), half_val);
    return (d & 1) ? val_odd : val_even;
}

// Mirrors convrot_warp_fht4 above stage-for-stage (same strides, same shuffle
// offsets, same d formulas) but keeps registers in native T and combines via
// convrot_native_h4_combine instead of the fp32 combine.
template<typename T>
__device__ __forceinline__ void convrot_warp_fht4_native(T v[4][8], int lane, T half_val) {
    #pragma unroll
    for (int g = 0; g < 4; ++g) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int d = lane & 3;
            const T b1 = __shfl_xor_sync(0xffffffff, v[g][j], 1);
            const T b2 = __shfl_xor_sync(0xffffffff, v[g][j], 2);
            const T b3 = __shfl_xor_sync(0xffffffff, v[g][j], 3);
            v[g][j] = convrot_native_h4_combine<T>(d, v[g][j], b1, b2, b3, half_val);
        }
    }

    #pragma unroll
    for (int g = 0; g < 4; ++g) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int d = (lane >> 2) & 3;
            const T b1 = __shfl_xor_sync(0xffffffff, v[g][j], 4);
            const T b2 = __shfl_xor_sync(0xffffffff, v[g][j], 8);
            const T b3 = __shfl_xor_sync(0xffffffff, v[g][j], 12);
            v[g][j] = convrot_native_h4_combine<T>(d, v[g][j], b1, b2, b3, half_val);
        }
    }

    #pragma unroll
    for (int g = 0; g < 4; ++g) {
        #pragma unroll
        for (int j0 = 0; j0 < 8; j0 += 2) {
            const T old_a = v[g][j0];
            const T old_b = v[g][j0 + 1];
            const int da = (lane >> 4) & 1;
            const T a_b1 = __shfl_xor_sync(0xffffffff, old_a, 16);
            const T a_b3 = __shfl_xor_sync(0xffffffff, old_b, 16);
            v[g][j0]     = convrot_native_h4_combine<T>(da,     old_a, a_b1, old_b, a_b3, half_val);
            v[g][j0 + 1] = convrot_native_h4_combine<T>(da | 2, old_b, a_b3, old_a, a_b1, half_val);
        }
    }

    #pragma unroll
    for (int g = 0; g < 4; ++g) {
        #pragma unroll
        for (int base = 0; base < 2; ++base) {
            const T e0 = v[g][base];
            const T e2 = v[g][base + 2];
            const T e4 = v[g][base + 4];
            const T e6 = v[g][base + 6];
            v[g][base]     = convrot_native_h4_combine<T>(0, e0, e2, e4, e6, half_val);
            v[g][base + 2] = convrot_native_h4_combine<T>(1, e2, e0, e6, e4, half_val);
            v[g][base + 4] = convrot_native_h4_combine<T>(2, e4, e6, e0, e2, half_val);
            v[g][base + 6] = convrot_native_h4_combine<T>(3, e6, e4, e2, e0, half_val);
        }
    }
}

// Rows per block / warps per row split identically to quantize_int8_rowwise_convrot_warp_kernel
// above (K / 1024 warps per row, this instantiation requires K % 1024 == 0). Output q is
// (num_rows, K/2) packed int4; scales is (num_rows,) float32.
template<typename InputType, int ROWS_PER_BLOCK>
__global__ void quantize_int4_rowwise_convrot_warp_kernel(
    const InputType* __restrict__ x,
    int8_t* __restrict__ q,
    float* __restrict__ scales,
    int K,
    int64_t num_rows)
{
    extern __shared__ float row_warp_max[];  // ROWS_PER_BLOCK * warps_per_row floats.

    const int warps_per_row = K >> 10;
    const int lane = threadIdx.x & 31;
    const int warp_id = static_cast<int>(threadIdx.x >> 5);
    const int local_row = warp_id / warps_per_row;
    const int warp_in_row = warp_id % warps_per_row;
    const int row = static_cast<int>(blockIdx.x) * ROWS_PER_BLOCK + local_row;
    const bool active_row = static_cast<int64_t>(row) < num_rows;
    const int64_t row_offset = static_cast<int64_t>(row) * K;
    const int64_t row_offset_half = static_cast<int64_t>(row) * (K >> 1);
    const int group_base = warp_in_row * 4;

    float local_max = 0.0f;

    if constexpr (std::is_same<InputType, float>::value) {
        float v[4][8];
        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            const int group_col = (group_base + g) * kConvRotGroup;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int e = lane + 32 * j;
                v[g][j] = active_row ? to_float(x[row_offset + group_col + e]) : 0.0f;
            }
        }

        convrot_warp_fht4(v, lane);

        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                local_max = fmaxf(local_max, fabsf(v[g][j]));
            }
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, off));
        }

        float abs_max = local_max;
        if (warps_per_row > 1) {
            row_warp_max[local_row * warps_per_row + warp_in_row] = local_max;
            __syncthreads();
            abs_max = 0.0f;
            #pragma unroll 1
            for (int w = 0; w < warps_per_row; ++w) {
                abs_max = fmaxf(abs_max, row_warp_max[local_row * warps_per_row + w]);
            }
        }

        const float scale = fmaxf(
            fminf(abs_max, finite_max_for_dtype<InputType>()) * (1.0f / 7.0f),
            1.0e-10f);
        if (active_row && lane == 0 && warp_in_row == 0) {
            scales[row] = scale;
        }
        const float inv_scale = 1.0f / scale;

        if (active_row) {
            #pragma unroll
            for (int g = 0; g < 4; ++g) {
                const int group_col = (group_base + g) * kConvRotGroup;
                const int64_t byte_group_base = row_offset_half + (group_col >> 1);
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float scaled = v[g][j] * inv_scale;
                    int qi = __float2int_rn(scaled);
                    qi = min(7, max(-7, qi));
                    const int partner = __shfl_xor_sync(0xffffffff, qi, 1);
                    if ((lane & 1) == 0) {
                        const int64_t byte_idx = byte_group_base + (lane >> 1) + 16 * j;
                        q[byte_idx] = pack_int4_pair_local(qi, partner);
                    }
                }
            }
        }
    } else {
        InputType v[4][8];
        const InputType half_val = from_float<InputType>(0.5f);
        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            const int group_col = (group_base + g) * kConvRotGroup;
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int e = lane + 32 * j;
                v[g][j] = active_row ? x[row_offset + group_col + e] : from_float<InputType>(0.0f);
            }
        }

        convrot_warp_fht4_native<InputType>(v, lane, half_val);

        #pragma unroll
        for (int g = 0; g < 4; ++g) {
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                local_max = fmaxf(local_max, fabsf(to_float(v[g][j])));
            }
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, off));
        }

        float abs_max = local_max;
        if (warps_per_row > 1) {
            row_warp_max[local_row * warps_per_row + warp_in_row] = local_max;
            __syncthreads();
            abs_max = 0.0f;
            #pragma unroll 1
            for (int w = 0; w < warps_per_row; ++w) {
                abs_max = fmaxf(abs_max, row_warp_max[local_row * warps_per_row + w]);
            }
        }

        const float scale = fmaxf(
            fminf(abs_max, finite_max_for_dtype<InputType>()) * (1.0f / 7.0f),
            1.0e-10f);
        if (active_row && lane == 0 && warp_in_row == 0) {
            scales[row] = scale;
        }
        const float inv_scale = 1.0f / scale;

        if (active_row) {
            #pragma unroll
            for (int g = 0; g < 4; ++g) {
                const int group_col = (group_base + g) * kConvRotGroup;
                const int64_t byte_group_base = row_offset_half + (group_col >> 1);
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    const float scaled = to_float(v[g][j]) * inv_scale;
                    int qi = __float2int_rn(scaled);
                    qi = min(7, max(-7, qi));
                    const int partner = __shfl_xor_sync(0xffffffff, qi, 1);
                    if ((lane & 1) == 0) {
                        const int64_t byte_idx = byte_group_base + (lane >> 1) + 16 * j;
                        q[byte_idx] = pack_int4_pair_local(qi, partner);
                    }
                }
            }
        }
    }
}

template<typename InputType>
void launch_for_rows_per_block_int4(
    int rows_per_block,
    int64_t blocks,
    int block_threads,
    size_t smem_bytes,
    const InputType* x,
    int8_t* q,
    float* scales,
    int K,
    int64_t num_rows,
    cudaStream_t stream)
{
    if (rows_per_block == 4) {
        quantize_int4_rowwise_convrot_warp_kernel<InputType, 4>
            <<<static_cast<unsigned int>(blocks), block_threads, smem_bytes, stream>>>(
                x, q, scales, K, num_rows);
    } else if (rows_per_block == 2) {
        quantize_int4_rowwise_convrot_warp_kernel<InputType, 2>
            <<<static_cast<unsigned int>(blocks), block_threads, smem_bytes, stream>>>(
                x, q, scales, K, num_rows);
    } else {
        quantize_int4_rowwise_convrot_warp_kernel<InputType, 1>
            <<<static_cast<unsigned int>(blocks), block_threads, smem_bytes, stream>>>(
                x, q, scales, K, num_rows);
    }
}

// ---------------------------------------------------------------------------
// INT4 half2/bf162 SIMD-paired addition below: fp16/bf16 only (fp32 keeps the
// scalar/native paths above unchanged -- there is no fp32 packed-arithmetic
// win to chase here). Reuses kConvRotGroup/to_float/from_float/
// finite_max_for_dtype/pack_int4_pair_local declared above unchanged.
//
// Invariant: a group's butterfly sign pattern (which of own/b1/b2/b3 feeds
// which output, and with which sign) depends only on the lane and the
// intra-group slot j -- never on which group it is. Packing the SAME (lane,
// slot) position of two adjacent groups (2k, 2k+1) into one half2/bf162 means
// both halves of every register see an IDENTICAL sequence of hadd2/hsub2/
// hmul2 ops, so the packed path reduces to the native scalar path applied
// twice in lockstep. Confirmed bit-for-bit on this hardware (SM80+ packed
// hadd2/hsub2/hmul2 use the same per-lane IEEE round-to-nearest as their
// scalar hadd/hsub/hmul counterparts): both halves of a register hold the
// same intra-group position of adjacent groups, so every butterfly applies
// identically to both.
//
// Layout: PAIRS_PER_WARP h2/bf162 registers per slot (v2[PAIRS_PER_WARP][8]),
// each covering 2 groups (512 elements) -- PAIRS_PER_WARP=4 (8 groups, 2048
// elements, 32 packed regs/thread) is the standard width (K a multiple of
// 2048); PAIRS_PER_WARP=2 (4 groups, 1024 elements, 16 packed regs/thread)
// covers K==1024, which is smaller than one standard-width warp. Other
// multiples of 512 (e.g. 1536, 3072, 5120) are not covered by this kernel --
// only K==1024 or K%2048==0 dispatch here; anything else in the general
// K%1024==0 domain still runs the scalar-native or stock path.
// ---------------------------------------------------------------------------

template<typename T> struct PairedType;
template<> struct PairedType<half> { using type = half2; };
template<> struct PairedType<nv_bfloat16> { using type = nv_bfloat162; };

__device__ __forceinline__ half low_half(half2 p) { return __low2half(p); }
__device__ __forceinline__ half high_half(half2 p) { return __high2half(p); }
__device__ __forceinline__ nv_bfloat16 low_half(nv_bfloat162 p) { return __low2bfloat16(p); }
__device__ __forceinline__ nv_bfloat16 high_half(nv_bfloat162 p) { return __high2bfloat16(p); }
__device__ __forceinline__ half2 make_pair(half lo, half hi) { return __halves2half2(lo, hi); }
__device__ __forceinline__ nv_bfloat162 make_pair(nv_bfloat16 lo, nv_bfloat16 hi) { return __halves2bfloat162(lo, hi); }

// h2/bf162 counterpart of convrot_native_h4_combine: identical operand order
// and op sequence (hadd2/hsub2 into p/q_even/q_odd, then hmul2 by half_val2),
// selected by d's parity exactly as the scalar version. PackedT is half2 or
// nv_bfloat162; __hadd2/__hsub2/__hmul2 are already overloaded for both by
// cuda_fp16.h/cuda_bf16.h, so this single template body covers both dtypes.
template<typename PackedT>
__device__ __forceinline__ PackedT convrot_native_h2_combine(int d, PackedT own, PackedT b1, PackedT b2, PackedT b3, PackedT half_val2) {
    const PackedT p = __hadd2(own, b1);
    const PackedT q_even = __hsub2(b2, b3);
    const PackedT q_odd = __hsub2(b3, b2);
    const PackedT val_even = __hmul2(__hadd2(p, q_even), half_val2);
    const PackedT val_odd = __hmul2(__hsub2(p, q_odd), half_val2);
    return (d & 1) ? val_odd : val_even;
}

// Mirrors convrot_warp_fht4_native stage-for-stage (same strides 1/4/16/64,
// same shuffle offsets, same d formulas) over G2 packed-pair "groups" instead
// of 4 scalar groups. __shfl_xor_sync has native half2/nv_bfloat162 overloads
// (cuda_fp16.hpp/cuda_bf16.hpp): one shuffle moves both halves at once.
template<typename PackedT, int G2>
__device__ __forceinline__ void convrot_warp_fht_h2(PackedT v2[G2][8], int lane, PackedT half_val2) {
    #pragma unroll
    for (int g = 0; g < G2; ++g) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int d = lane & 3;
            const PackedT b1 = __shfl_xor_sync(0xffffffff, v2[g][j], 1);
            const PackedT b2 = __shfl_xor_sync(0xffffffff, v2[g][j], 2);
            const PackedT b3 = __shfl_xor_sync(0xffffffff, v2[g][j], 3);
            v2[g][j] = convrot_native_h2_combine<PackedT>(d, v2[g][j], b1, b2, b3, half_val2);
        }
    }

    #pragma unroll
    for (int g = 0; g < G2; ++g) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int d = (lane >> 2) & 3;
            const PackedT b1 = __shfl_xor_sync(0xffffffff, v2[g][j], 4);
            const PackedT b2 = __shfl_xor_sync(0xffffffff, v2[g][j], 8);
            const PackedT b3 = __shfl_xor_sync(0xffffffff, v2[g][j], 12);
            v2[g][j] = convrot_native_h2_combine<PackedT>(d, v2[g][j], b1, b2, b3, half_val2);
        }
    }

    #pragma unroll
    for (int g = 0; g < G2; ++g) {
        #pragma unroll
        for (int j0 = 0; j0 < 8; j0 += 2) {
            const PackedT old_a = v2[g][j0];
            const PackedT old_b = v2[g][j0 + 1];
            const int da = (lane >> 4) & 1;
            const PackedT a_b1 = __shfl_xor_sync(0xffffffff, old_a, 16);
            const PackedT a_b3 = __shfl_xor_sync(0xffffffff, old_b, 16);
            v2[g][j0]     = convrot_native_h2_combine<PackedT>(da,     old_a, a_b1, old_b, a_b3, half_val2);
            v2[g][j0 + 1] = convrot_native_h2_combine<PackedT>(da | 2, old_b, a_b3, old_a, a_b1, half_val2);
        }
    }

    #pragma unroll
    for (int g = 0; g < G2; ++g) {
        #pragma unroll
        for (int base = 0; base < 2; ++base) {
            const PackedT e0 = v2[g][base];
            const PackedT e2 = v2[g][base + 2];
            const PackedT e4 = v2[g][base + 4];
            const PackedT e6 = v2[g][base + 6];
            v2[g][base]     = convrot_native_h2_combine<PackedT>(0, e0, e2, e4, e6, half_val2);
            v2[g][base + 2] = convrot_native_h2_combine<PackedT>(1, e2, e0, e6, e4, half_val2);
            v2[g][base + 4] = convrot_native_h2_combine<PackedT>(2, e4, e6, e0, e2, half_val2);
            v2[g][base + 6] = convrot_native_h2_combine<PackedT>(3, e6, e4, e2, e0, half_val2);
        }
    }
}

// warps_per_row = K / (2*PAIRS_PER_WARP*kConvRotGroup); rows_per_block chosen
// by the caller (see convrot_warp_quantize_int4_h2). Quantize/pack stage
// pulls the two groups (2k, 2k+1) back apart from each h2/bf162 register --
// they are different 128-byte output regions -- and runs the SAME per-group
// scalar quantize+pack sequence as quantize_int4_rowwise_convrot_warp_kernel
// (round via __float2int_rn, clamp [-7,7], pack via the even/odd-lane
// shuffle-and-write trick) independently for each half, so correctness
// reduces to that already-verified scalar sequence run twice.
template<typename InputType, int PAIRS_PER_WARP, int ROWS_PER_BLOCK>
__global__ void quantize_int4_rowwise_convrot_warp_h2_kernel(
    const InputType* __restrict__ x,
    int8_t* __restrict__ q,
    float* __restrict__ scales,
    int K,
    int64_t num_rows)
{
    using PackedT = typename PairedType<InputType>::type;
    constexpr int GROUPS_PER_WARP = 2 * PAIRS_PER_WARP;
    extern __shared__ float row_warp_max[];  // ROWS_PER_BLOCK * warps_per_row floats.

    const int warps_per_row = K / (GROUPS_PER_WARP * kConvRotGroup);
    const int lane = threadIdx.x & 31;
    const int warp_id = static_cast<int>(threadIdx.x >> 5);
    const int local_row = warp_id / warps_per_row;
    const int warp_in_row = warp_id % warps_per_row;
    const int row = static_cast<int>(blockIdx.x) * ROWS_PER_BLOCK + local_row;
    const bool active_row = static_cast<int64_t>(row) < num_rows;
    const int64_t row_offset = static_cast<int64_t>(row) * K;
    const int64_t row_offset_half = static_cast<int64_t>(row) * (K >> 1);
    const int group_base = warp_in_row * GROUPS_PER_WARP;

    const InputType half_scalar = from_float<InputType>(0.5f);
    const PackedT half_val2 = make_pair(half_scalar, half_scalar);

    PackedT v2[PAIRS_PER_WARP][8];
    #pragma unroll
    for (int k = 0; k < PAIRS_PER_WARP; ++k) {
        const int col_lo = (group_base + 2 * k) * kConvRotGroup;
        const int col_hi = (group_base + 2 * k + 1) * kConvRotGroup;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int e = lane + 32 * j;
            const InputType xlo = active_row ? x[row_offset + col_lo + e] : from_float<InputType>(0.0f);
            const InputType xhi = active_row ? x[row_offset + col_hi + e] : from_float<InputType>(0.0f);
            v2[k][j] = make_pair(xlo, xhi);
        }
    }

    convrot_warp_fht_h2<PackedT, PAIRS_PER_WARP>(v2, lane, half_val2);

    float local_max = 0.0f;
    #pragma unroll
    for (int k = 0; k < PAIRS_PER_WARP; ++k) {
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            local_max = fmaxf(local_max, fabsf(to_float(low_half(v2[k][j]))));
            local_max = fmaxf(local_max, fabsf(to_float(high_half(v2[k][j]))));
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        local_max = fmaxf(local_max, __shfl_xor_sync(0xffffffff, local_max, off));
    }

    float abs_max = local_max;
    if (warps_per_row > 1) {
        row_warp_max[local_row * warps_per_row + warp_in_row] = local_max;
        __syncthreads();
        abs_max = 0.0f;
        #pragma unroll 1
        for (int w = 0; w < warps_per_row; ++w) {
            abs_max = fmaxf(abs_max, row_warp_max[local_row * warps_per_row + w]);
        }
    }

    const float scale = fmaxf(
        fminf(abs_max, finite_max_for_dtype<InputType>()) * (1.0f / 7.0f),
        1.0e-10f);
    if (active_row && lane == 0 && warp_in_row == 0) {
        scales[row] = scale;
    }
    const float inv_scale = 1.0f / scale;

    if (active_row) {
        #pragma unroll
        for (int k = 0; k < PAIRS_PER_WARP; ++k) {
            const int group_lo = group_base + 2 * k;
            const int group_hi = group_lo + 1;
            const int64_t byte_base_lo = row_offset_half + group_lo * (kConvRotGroup / 2);
            const int64_t byte_base_hi = row_offset_half + group_hi * (kConvRotGroup / 2);
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                const float scaled_lo = to_float(low_half(v2[k][j])) * inv_scale;
                int qi_lo = __float2int_rn(scaled_lo);
                qi_lo = min(7, max(-7, qi_lo));
                const int partner_lo = __shfl_xor_sync(0xffffffff, qi_lo, 1);
                if ((lane & 1) == 0) {
                    q[byte_base_lo + (lane >> 1) + 16 * j] = pack_int4_pair_local(qi_lo, partner_lo);
                }

                const float scaled_hi = to_float(high_half(v2[k][j])) * inv_scale;
                int qi_hi = __float2int_rn(scaled_hi);
                qi_hi = min(7, max(-7, qi_hi));
                const int partner_hi = __shfl_xor_sync(0xffffffff, qi_hi, 1);
                if ((lane & 1) == 0) {
                    q[byte_base_hi + (lane >> 1) + 16 * j] = pack_int4_pair_local(qi_hi, partner_hi);
                }
            }
        }
    }
}

template<typename InputType, int PAIRS_PER_WARP>
void launch_int4_h2_for_rows_per_block(
    int rows_per_block,
    int64_t blocks,
    int block_threads,
    size_t smem_bytes,
    const InputType* x,
    int8_t* q,
    float* scales,
    int K,
    int64_t num_rows,
    cudaStream_t stream)
{
    if (rows_per_block == 8) {
        quantize_int4_rowwise_convrot_warp_h2_kernel<InputType, PAIRS_PER_WARP, 8>
            <<<static_cast<unsigned int>(blocks), block_threads, smem_bytes, stream>>>(
                x, q, scales, K, num_rows);
    } else if (rows_per_block == 4) {
        quantize_int4_rowwise_convrot_warp_h2_kernel<InputType, PAIRS_PER_WARP, 4>
            <<<static_cast<unsigned int>(blocks), block_threads, smem_bytes, stream>>>(
                x, q, scales, K, num_rows);
    } else if (rows_per_block == 2) {
        quantize_int4_rowwise_convrot_warp_h2_kernel<InputType, PAIRS_PER_WARP, 2>
            <<<static_cast<unsigned int>(blocks), block_threads, smem_bytes, stream>>>(
                x, q, scales, K, num_rows);
    } else {
        quantize_int4_rowwise_convrot_warp_h2_kernel<InputType, PAIRS_PER_WARP, 1>
            <<<static_cast<unsigned int>(blocks), block_threads, smem_bytes, stream>>>(
                x, q, scales, K, num_rows);
    }
}

template<typename InputType>
void launch_int4_h2(
    int pairs_per_warp,
    int rows_per_block,
    int64_t blocks,
    int block_threads,
    size_t smem_bytes,
    const InputType* x,
    int8_t* q,
    float* scales,
    int K,
    int64_t num_rows,
    cudaStream_t stream)
{
    if (pairs_per_warp == 4) {
        launch_int4_h2_for_rows_per_block<InputType, 4>(
            rows_per_block, blocks, block_threads, smem_bytes, x, q, scales, K, num_rows, stream);
    } else {
        launch_int4_h2_for_rows_per_block<InputType, 2>(
            rows_per_block, blocks, block_threads, smem_bytes, x, q, scales, K, num_rows, stream);
    }
}

} // namespace anima_turbo

// Mirrors comfy-kitchen's launch_quantize_int8_rowwise_convrot64_kernel dispatch for
// its K % 1024 == 0 warp-per-group branch exactly: warps_per_row = K / 1024,
// rows_per_block = 4 (warps_per_row==1), 2 (warps_per_row<=3), else 1; shared memory
// holds one float per (row, warp) slot only when warps_per_row > 1.
void convrot_warp_quantize(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor scales,
    int64_t stream_ptr)
{
    TORCH_CHECK(input.is_cuda() && output.is_cuda() && scales.is_cuda(),
                "convrot_warp_quantize: input/output/scales must be CUDA tensors");
    TORCH_CHECK(input.dim() == 2 && output.dim() == 2,
                "convrot_warp_quantize: input/output must be 2D");
    TORCH_CHECK(output.scalar_type() == torch::kInt8,
                "convrot_warp_quantize: output must be int8");
    TORCH_CHECK(output.sizes() == input.sizes(),
                "convrot_warp_quantize: output shape must match input shape");
    TORCH_CHECK(scales.scalar_type() == torch::kFloat32,
                "convrot_warp_quantize: scales must be float32");
    TORCH_CHECK(scales.dim() == 2 && scales.size(1) == 1 && scales.size(0) == input.size(0),
                "convrot_warp_quantize: scales must be shape (M, 1)");

    const int64_t M = input.size(0);
    const int64_t K = input.size(1);
    TORCH_CHECK(K % 1024 == 0, "convrot_warp_quantize: K must be a multiple of 1024");
    TORCH_CHECK(K >= 1024 && K <= 32768, "convrot_warp_quantize: K must be in [1024, 32768]");
    TORCH_CHECK(
        input.scalar_type() == torch::kFloat32 || input.scalar_type() == torch::kFloat16 ||
        input.scalar_type() == torch::kBFloat16,
        "convrot_warp_quantize: input dtype must be float32, float16, or bfloat16");

    const int warps_per_row = static_cast<int>(K >> 10);
    int rows_per_block;
    if (warps_per_row == 1) {
        rows_per_block = 4;
    } else if (warps_per_row <= 3) {
        rows_per_block = 2;
    } else {
        rows_per_block = 1;
    }

    const int block_threads = rows_per_block * warps_per_row * anima_turbo::kThreadsPerWarp;
    const int64_t blocks = (M + rows_per_block - 1) / rows_per_block;
    const size_t smem_per_row = static_cast<size_t>(warps_per_row) * sizeof(float);
    const size_t smem_bytes = (warps_per_row == 1) ? 0 : static_cast<size_t>(rows_per_block) * smem_per_row;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    float* scales_ptr = scales.data_ptr<float>();
    int8_t* q_ptr = output.data_ptr<int8_t>();
    const int K_i = static_cast<int>(K);

    switch (input.scalar_type()) {
        case torch::kFloat32:
            anima_turbo::launch_for_rows_per_block<float>(
                rows_per_block, blocks, block_threads, smem_bytes,
                input.data_ptr<float>(), q_ptr, scales_ptr, K_i, M, stream);
            break;
        case torch::kFloat16:
            anima_turbo::launch_for_rows_per_block<half>(
                rows_per_block, blocks, block_threads, smem_bytes,
                reinterpret_cast<const half*>(input.data_ptr<at::Half>()), q_ptr, scales_ptr, K_i, M, stream);
            break;
        case torch::kBFloat16:
            anima_turbo::launch_for_rows_per_block<nv_bfloat16>(
                rows_per_block, blocks, block_threads, smem_bytes,
                reinterpret_cast<const nv_bfloat16*>(input.data_ptr<at::BFloat16>()), q_ptr, scales_ptr, K_i, M, stream);
            break;
        default:
            TORCH_CHECK(false, "convrot_warp_quantize: unreachable dtype branch");
    }

    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "convrot_warp_quantize: kernel launch failed: ", cudaGetErrorString(err));
}

// Mirrors comfy-kitchen's launch_quantize_int4_rowwise_convrot64_kernel dispatch for its
// group_size==256, K % 1024 == 0 warp-per-group case exactly (same rows_per_block/
// warps_per_row/shared-memory sizing as convrot_warp_quantize above). Output is packed
// int4: (M, K/2) int8.
void convrot_warp_quantize_int4(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor scales,
    int64_t stream_ptr)
{
    TORCH_CHECK(input.is_cuda() && output.is_cuda() && scales.is_cuda(),
                "convrot_warp_quantize_int4: input/output/scales must be CUDA tensors");
    TORCH_CHECK(input.dim() == 2 && output.dim() == 2,
                "convrot_warp_quantize_int4: input/output must be 2D");
    TORCH_CHECK(output.scalar_type() == torch::kInt8,
                "convrot_warp_quantize_int4: output must be int8");

    const int64_t M = input.size(0);
    const int64_t K = input.size(1);
    TORCH_CHECK(K % 1024 == 0, "convrot_warp_quantize_int4: K must be a multiple of 1024");
    TORCH_CHECK(K >= 1024 && K <= 32768, "convrot_warp_quantize_int4: K must be in [1024, 32768]");
    TORCH_CHECK(
        input.scalar_type() == torch::kFloat32 || input.scalar_type() == torch::kFloat16 ||
        input.scalar_type() == torch::kBFloat16,
        "convrot_warp_quantize_int4: input dtype must be float32, float16, or bfloat16");
    TORCH_CHECK(output.size(0) == M && output.size(1) == K / 2,
                "convrot_warp_quantize_int4: output shape must be (M, K/2)");
    TORCH_CHECK(scales.scalar_type() == torch::kFloat32,
                "convrot_warp_quantize_int4: scales must be float32");
    TORCH_CHECK(scales.dim() == 2 && scales.size(1) == 1 && scales.size(0) == M,
                "convrot_warp_quantize_int4: scales must be shape (M, 1)");

    const int warps_per_row = static_cast<int>(K >> 10);
    int rows_per_block;
    if (warps_per_row == 1) {
        rows_per_block = 4;
    } else if (warps_per_row <= 3) {
        rows_per_block = 2;
    } else {
        rows_per_block = 1;
    }

    const int block_threads = rows_per_block * warps_per_row * anima_turbo::kThreadsPerWarp;
    const int64_t blocks = (M + rows_per_block - 1) / rows_per_block;
    const size_t smem_per_row = static_cast<size_t>(warps_per_row) * sizeof(float);
    const size_t smem_bytes = (warps_per_row == 1) ? 0 : static_cast<size_t>(rows_per_block) * smem_per_row;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    float* scales_ptr = scales.data_ptr<float>();
    int8_t* q_ptr = output.data_ptr<int8_t>();
    const int K_i = static_cast<int>(K);

    switch (input.scalar_type()) {
        case torch::kFloat32:
            anima_turbo::launch_for_rows_per_block_int4<float>(
                rows_per_block, blocks, block_threads, smem_bytes,
                input.data_ptr<float>(), q_ptr, scales_ptr, K_i, M, stream);
            break;
        case torch::kFloat16:
            anima_turbo::launch_for_rows_per_block_int4<half>(
                rows_per_block, blocks, block_threads, smem_bytes,
                reinterpret_cast<const half*>(input.data_ptr<at::Half>()), q_ptr, scales_ptr, K_i, M, stream);
            break;
        case torch::kBFloat16:
            anima_turbo::launch_for_rows_per_block_int4<nv_bfloat16>(
                rows_per_block, blocks, block_threads, smem_bytes,
                reinterpret_cast<const nv_bfloat16*>(input.data_ptr<at::BFloat16>()), q_ptr, scales_ptr, K_i, M, stream);
            break;
        default:
            TORCH_CHECK(false, "convrot_warp_quantize_int4: unreachable dtype branch");
    }

    cudaError_t err_int4 = cudaGetLastError();
    TORCH_CHECK(err_int4 == cudaSuccess, "convrot_warp_quantize_int4: kernel launch failed: ", cudaGetErrorString(err_int4));
}

// half2/bf162 SIMD-paired sibling of convrot_warp_quantize_int4 above, fp16/bf16
// only (fp32 has no packed-arithmetic win here; callers keep routing fp32 to
// convrot_warp_quantize_int4). Domain is narrower than the scalar kernel's
// general K%1024==0 band: K==1024, or K a multiple of 2048 (both satisfy the
// even-group-count requirement K%512==0; other K%512==0 values that are
// neither -- e.g. 1536, 3072, 5120 -- are not covered by this kernel's fixed
// 4-pair/2-pair warp width and must use the scalar-native or stock path).
// rows_per_block: the scalar kernel's warps_per_row-bucketed rule (4/2/1) measured
// best or within noise of every alternative tried at K=2048/4096/8192, but K==1024
// is a documented exception (see below) -- keep it out of that bucketing.
void convrot_warp_quantize_int4_h2(
    torch::Tensor input,
    torch::Tensor output,
    torch::Tensor scales,
    int64_t stream_ptr)
{
    TORCH_CHECK(input.is_cuda() && output.is_cuda() && scales.is_cuda(),
                "convrot_warp_quantize_int4_h2: input/output/scales must be CUDA tensors");
    TORCH_CHECK(input.dim() == 2 && output.dim() == 2,
                "convrot_warp_quantize_int4_h2: input/output must be 2D");
    TORCH_CHECK(output.scalar_type() == torch::kInt8,
                "convrot_warp_quantize_int4_h2: output must be int8");
    TORCH_CHECK(
        input.scalar_type() == torch::kFloat16 || input.scalar_type() == torch::kBFloat16,
        "convrot_warp_quantize_int4_h2: input dtype must be float16 or bfloat16 (fp32 not covered by "
        "this kernel; use convrot_warp_quantize_int4)");

    const int64_t M = input.size(0);
    const int64_t K = input.size(1);
    TORCH_CHECK(K >= 1024 && K <= 32768, "convrot_warp_quantize_int4_h2: K must be in [1024, 32768]");
    TORCH_CHECK(K == 1024 || K % 2048 == 0,
                "convrot_warp_quantize_int4_h2: K must be 1024 or a multiple of 2048");
    TORCH_CHECK(output.size(0) == M && output.size(1) == K / 2,
                "convrot_warp_quantize_int4_h2: output shape must be (M, K/2)");
    TORCH_CHECK(scales.scalar_type() == torch::kFloat32,
                "convrot_warp_quantize_int4_h2: scales must be float32");
    TORCH_CHECK(scales.dim() == 2 && scales.size(1) == 1 && scales.size(0) == M,
                "convrot_warp_quantize_int4_h2: scales must be shape (M, 1)");

    const int pairs_per_warp = (K == 1024) ? 2 : 4;
    const int groups_per_warp = 2 * pairs_per_warp;
    const int warps_per_row = static_cast<int>(K) / (groups_per_warp * anima_turbo::kConvRotGroup);
    // rows_per_block: benchmarked 1/2/4/8 at each of K=1024/2048/4096/8192 (bf16,
    // idle GPU, >=150 iters/config). K==1024 (the PAIRS_PER_WARP=2, 48-reg/thread
    // config) is the outlier: rows_per_block=1 measured 66.8us vs 72.4us at 4 (more
    // resident blocks/SM beats the bigger block there); every other tested K matched
    // or was within noise of the same warps_per_row-bucketed rule the scalar kernel
    // uses (4 at warps_per_row==1, 2 at <=3, else 1).
    int rows_per_block;
    if (K == 1024) {
        rows_per_block = 1;
    } else if (warps_per_row == 1) {
        rows_per_block = 4;
    } else if (warps_per_row <= 3) {
        rows_per_block = 2;
    } else {
        rows_per_block = 1;
    }

    const int block_threads = rows_per_block * warps_per_row * anima_turbo::kThreadsPerWarp;
    const int64_t blocks = (M + rows_per_block - 1) / rows_per_block;
    const size_t smem_per_row = static_cast<size_t>(warps_per_row) * sizeof(float);
    const size_t smem_bytes = (warps_per_row == 1) ? 0 : static_cast<size_t>(rows_per_block) * smem_per_row;

    cudaStream_t stream = reinterpret_cast<cudaStream_t>(stream_ptr);
    float* scales_ptr = scales.data_ptr<float>();
    int8_t* q_ptr = output.data_ptr<int8_t>();
    const int K_i = static_cast<int>(K);

    switch (input.scalar_type()) {
        case torch::kFloat16:
            anima_turbo::launch_int4_h2<half>(
                pairs_per_warp, rows_per_block, blocks, block_threads, smem_bytes,
                reinterpret_cast<const half*>(input.data_ptr<at::Half>()), q_ptr, scales_ptr, K_i, M, stream);
            break;
        case torch::kBFloat16:
            anima_turbo::launch_int4_h2<nv_bfloat16>(
                pairs_per_warp, rows_per_block, blocks, block_threads, smem_bytes,
                reinterpret_cast<const nv_bfloat16*>(input.data_ptr<at::BFloat16>()), q_ptr, scales_ptr, K_i, M, stream);
            break;
        default:
            TORCH_CHECK(false, "convrot_warp_quantize_int4_h2: unreachable dtype branch");
    }

    cudaError_t err_h2 = cudaGetLastError();
    TORCH_CHECK(err_h2 == cudaSuccess, "convrot_warp_quantize_int4_h2: kernel launch failed: ", cudaGetErrorString(err_h2));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("convrot_warp_quantize", &convrot_warp_quantize,
          "Warp-per-group FHT ConvRot row-wise INT8 quantize (K%1024==0, 1024<=K<=32768, fp32/fp16/bf16)");
    m.def("convrot_warp_quantize_int4", &convrot_warp_quantize_int4,
          "Warp-per-group FHT ConvRot row-wise INT4 rotate+quantize+pack (K%1024==0, 1024<=K<=32768, fp32/fp16/bf16)");
    m.def("convrot_warp_quantize_int4_h2", &convrot_warp_quantize_int4_h2,
          "half2/bf162 SIMD-paired warp-per-group-pair FHT ConvRot row-wise INT4 rotate+quantize+pack "
          "(K==1024 or K%2048==0, 1024<=K<=32768, fp16/bf16 only)");
}
