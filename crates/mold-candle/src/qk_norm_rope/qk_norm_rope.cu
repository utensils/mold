// Fused per-head RMSNorm + interleaved RoPE + BSHD->BHSD transpose.
//
// Reproduces, element for element, the composite
//   rms_norm(x[B,S,H,D], w)             (candle-kernels reduce.cu `rmsnorm`)
//   .transpose(1, 2).contiguous()
//   .to_dtype(F32) -> rope_i(F32 cos/sin) (reduce.cu `ropei`) -> to_dtype(T)
// in ONE read and ONE write. The sum of squares uses the same per-thread
// column assignment and the same xor-butterfly warp reduction as candle's
// `rmsnorm` for rows under 1024 columns (one 32-thread warp per row), and the
// normalized value is rounded to T before the rotation exactly as the
// composite's intermediate tensor is.
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, mask, 32);
    }
    return x;
}

template <typename T>
__device__ void qk_norm_rope(const T* x, const T* weight, const float* cos, const float* sin,
                             T* dst, const uint32_t seq, const uint32_t heads,
                             const uint32_t head_dim, const float eps) {
    // One warp per (b, s, h) row; rows enumerate x in its BSHD order.
    const uint32_t row = blockIdx.x;
    const uint32_t tid = threadIdx.x;
    const uint32_t h = row % heads;
    const uint32_t s = (row / heads) % seq;
    const uint32_t b = row / (heads * seq);
    const T* src = x + static_cast<uint64_t>(row) * head_dim;

    float tmp = 0.0f;
    for (uint32_t col = tid; col < head_dim; col += 32) {
        const float xi = static_cast<float>(src[col]);
        tmp += xi * xi;
    }
    tmp = warp_reduce_sum(tmp);
    const float mean = tmp / static_cast<int>(head_dim);
    const float scale = rsqrtf(mean + eps);

    const uint32_t half = head_dim / 2;
    const float* c_row = cos + static_cast<uint64_t>(s) * half;
    const float* s_row = sin + static_cast<uint64_t>(s) * half;
    T* out = dst + ((static_cast<uint64_t>(b) * heads + h) * seq + s) * head_dim;
    for (uint32_t pair = tid; pair < half; pair += 32) {
        const uint32_t i0 = 2 * pair;
        const uint32_t i1 = i0 + 1;
        // rms_norm writes T, and to_dtype(F32) widens it back exactly.
        const float n0 = static_cast<float>(static_cast<T>(
            scale * static_cast<float>(src[i0]) * static_cast<float>(weight[i0])));
        const float n1 = static_cast<float>(static_cast<T>(
            scale * static_cast<float>(src[i1]) * static_cast<float>(weight[i1])));
        const float c = c_row[pair];
        const float sn = s_row[pair];
        // nvcc contracts candle's F32 `ropei` (`a * c - b * s`, `a * s + b * c`)
        // into exactly these fused multiply-adds; spelling them out keeps the
        // result independent of this file's own contraction choices.
        out[i0] = static_cast<T>(__fmaf_rn(n0, c, -(n1 * sn)));
        out[i1] = static_cast<T>(__fmaf_rn(n0, sn, n1 * c));
    }
}

#define QK_NORM_ROPE_OP(TYPENAME, FN_NAME)                                                  \
    extern "C" __global__ void FN_NAME(const TYPENAME* x, const TYPENAME* weight,            \
                                       const float* cos, const float* sin, TYPENAME* dst,     \
                                       const uint32_t seq, const uint32_t heads,              \
                                       const uint32_t head_dim, const float eps) {            \
        qk_norm_rope<TYPENAME>(x, weight, cos, sin, dst, seq, heads, head_dim, eps);          \
    }

QK_NORM_ROPE_OP(__nv_bfloat16, qk_norm_rope_bf16)
QK_NORM_ROPE_OP(__half, qk_norm_rope_f16)
QK_NORM_ROPE_OP(float, qk_norm_rope_f32)
