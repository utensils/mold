/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 Comfy Org. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * The row quantization and dequantization equations are adapted from the
 * source-pinned comfy-kitchen 0.2.26 INT8 CUDA reference. Mold keeps only the
 * two kernels surrounding cuBLASLt; ConvRot remains in the shared Candle path.
 */

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cfloat>
#include <cstdint>

constexpr int H3_INT8_THREADS = 256;

template <typename T> __device__ __forceinline__ float h3_to_float(T value);
template <> __device__ __forceinline__ float h3_to_float<float>(float value) { return value; }
template <> __device__ __forceinline__ float h3_to_float<half>(half value) { return __half2float(value); }
template <> __device__ __forceinline__ float h3_to_float<nv_bfloat16>(nv_bfloat16 value) {
    return __bfloat162float(value);
}

template <typename T> __device__ __forceinline__ T h3_from_float(float value);
template <> __device__ __forceinline__ float h3_from_float<float>(float value) { return value; }
template <> __device__ __forceinline__ half h3_from_float<half>(float value) {
    return __float2half_rn(value);
}
template <> __device__ __forceinline__ nv_bfloat16 h3_from_float<nv_bfloat16>(float value) {
    return __float2bfloat16_rn(value);
}

template <typename T> __device__ __forceinline__ float h3_finite_max();
template <> __device__ __forceinline__ float h3_finite_max<float>() { return FLT_MAX; }
template <> __device__ __forceinline__ float h3_finite_max<half>() { return 65504.0f; }
template <> __device__ __forceinline__ float h3_finite_max<nv_bfloat16>() { return 3.38953139e38f; }

__device__ __forceinline__ float h3_warp_max(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
    }
    return value;
}

template <typename T>
__device__ __forceinline__ void h3_quantize_int8_rowwise_kernel(
    const T* __restrict__ input,
    int8_t* __restrict__ quantized,
    float* __restrict__ scales,
    int columns) {
    __shared__ float warp_maxima[H3_INT8_THREADS / 32];
    __shared__ float row_maximum;

    const int row = static_cast<int>(blockIdx.x);
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int64_t row_offset = static_cast<int64_t>(row) * columns;

    float maximum = 0.0f;
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        maximum = fmaxf(maximum, fabsf(h3_to_float(input[row_offset + column])));
    }
    maximum = h3_warp_max(maximum);
    if (lane == 0) {
        warp_maxima[warp] = maximum;
    }
    __syncthreads();
    if (warp == 0) {
        maximum = lane < H3_INT8_THREADS / 32 ? warp_maxima[lane] : 0.0f;
        maximum = h3_warp_max(maximum);
        if (lane == 0) {
            row_maximum = maximum;
        }
    }
    __syncthreads();

    const float scale = fmaxf(
        fminf(row_maximum, h3_finite_max<T>()) * (1.0f / 127.0f),
        1.0e-30f);
    if (threadIdx.x == 0) {
        scales[row] = scale;
    }
    const float typed_scale = h3_to_float(h3_from_float<T>(scale));
    for (int column = threadIdx.x; column < columns; column += blockDim.x) {
        const int64_t index = row_offset + column;
        const float value = h3_to_float(h3_from_float<T>(h3_to_float(input[index]) / typed_scale));
        const float rounded = nearbyintf(value);
        quantized[index] = static_cast<int8_t>(fminf(127.0f, fmaxf(-128.0f, rounded)));
    }
}

template <typename T>
__device__ __forceinline__ void h3_dequantize_int8_linear_kernel(
    const int32_t* __restrict__ accumulator,
    const float* __restrict__ input_scales,
    const float* __restrict__ weight_scales,
    T* __restrict__ output,
    int64_t elements,
    int columns) {
    const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index >= elements) {
        return;
    }
    const int column = static_cast<int>(index % columns);
    const int row = static_cast<int>(index / columns);
    const float value = static_cast<float>(accumulator[index])
        * input_scales[row]
        * weight_scales[column];
    output[index] = h3_from_float<T>(value);
}

extern "C" __global__ void h3_quantize_int8_rowwise_f32(
    const float* input, int8_t* quantized, float* scales, int columns) {
    h3_quantize_int8_rowwise_kernel(input, quantized, scales, columns);
}

extern "C" __global__ void h3_quantize_int8_rowwise_f16(
    const half* input, int8_t* quantized, float* scales, int columns) {
    h3_quantize_int8_rowwise_kernel(input, quantized, scales, columns);
}

extern "C" __global__ void h3_quantize_int8_rowwise_bf16(
    const nv_bfloat16* input, int8_t* quantized, float* scales, int columns) {
    h3_quantize_int8_rowwise_kernel(input, quantized, scales, columns);
}

extern "C" __global__ void h3_dequantize_int8_linear_f32(
    const int32_t* accumulator, const float* input_scales, const float* weight_scales,
    float* output, int64_t elements, int columns) {
    h3_dequantize_int8_linear_kernel(
        accumulator, input_scales, weight_scales, output, elements, columns);
}

extern "C" __global__ void h3_dequantize_int8_linear_f16(
    const int32_t* accumulator, const float* input_scales, const float* weight_scales,
    half* output, int64_t elements, int columns) {
    h3_dequantize_int8_linear_kernel(
        accumulator, input_scales, weight_scales, output, elements, columns);
}

extern "C" __global__ void h3_dequantize_int8_linear_bf16(
    const int32_t* accumulator, const float* input_scales, const float* weight_scales,
    nv_bfloat16* output, int64_t elements, int columns) {
    h3_dequantize_int8_linear_kernel(
        accumulator, input_scales, weight_scales, output, elements, columns);
}
