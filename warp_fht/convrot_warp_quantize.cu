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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("convrot_warp_quantize", &convrot_warp_quantize,
          "Warp-per-group FHT ConvRot row-wise INT8 quantize (K%1024==0, 1024<=K<=32768, fp32/fp16/bf16)");
}
