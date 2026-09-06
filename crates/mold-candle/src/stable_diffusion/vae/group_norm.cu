// PyTorch v2.5.1 CUDA GroupNorm/Welford reduction port; BSD terms and
// copyright notices in LICENSE-PYTORCH. Mold adaptation: standalone kernels,
// F16 NCHW input and F32 affine parameters, no ATen or Python dependency.
#include <cuda_fp16.h>
#include <stdint.h>

struct Moments { float mean, m2; int64_t n; float nf; };
__device__ Moments update(Moments a, float x) {
    const int64_t n = a.n + 1;
    const float delta = x - a.mean;
    const float nf = static_cast<float>(n);
    const float mean = a.mean + delta / nf;
    const float delta2 = x - mean;
    return {mean, a.m2 + delta * delta2, n, static_cast<float>(n)};
}
__device__ Moments combine(Moments a, Moments b) {
    if (a.nf == 0) return b;
    if (b.nf == 0) return a;
    const float delta = b.mean - a.mean;
    const float count = a.nf + b.nf;
    const float weight = b.nf / count;
    return {a.mean + delta * weight,
            a.m2 + b.m2 + delta * delta * a.nf * weight, -1, count};
}
__device__ Moments warp_reduce(Moments a) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        Moments b;
        b.mean = __shfl_down_sync(0xffffffff, a.mean, offset);
        b.m2 = __shfl_down_sync(0xffffffff, a.m2, offset);
        b.n = __shfl_down_sync(0xffffffff, a.n, offset);
        b.nf = __shfl_down_sync(0xffffffff, a.nf, offset);
        a = combine(a, b);
    }
    return a;
}
extern "C" __global__ void paint_group_norm_stats(
    const __half* input, __half* mean, __half* rstd, int64_t row_size, float epsilon) {
    Moments state = {0, 0, 0, 0};
    const int64_t base = static_cast<int64_t>(blockIdx.x) * row_size;
    for (int64_t j = threadIdx.x; j < row_size; j += blockDim.x)
        state = update(state, __half2float(input[base + j]));
    state = warp_reduce(state);
    if (blockDim.x > 32) {
        __shared__ Moments partial[32];
        if (threadIdx.x % 32 == 0) partial[threadIdx.x / 32] = state;
        __syncthreads();
        if (threadIdx.x < 32) {
            state = threadIdx.x < blockDim.x / 32 ? partial[threadIdx.x] : Moments{0,0,0,0};
            state = warp_reduce(state);
        }
    }
    if (threadIdx.x == 0) {
        const float variance = state.m2 / state.nf;
        const float eps = __half2float(__float2half(epsilon));
        mean[blockIdx.x] = __float2half(state.mean);
        rstd[blockIdx.x] = __float2half(rsqrtf(variance + eps));
    }
}
extern "C" __global__ void paint_group_norm_apply(
    const __half* input, const float* weight, const float* bias,
    const __half* mean, const __half* rstd, __half* output,
    int64_t count, int64_t channels, int64_t spatial, int64_t groups) {
    const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= count) return;
    const int64_t channel = (i / spatial) % channels;
    const int64_t group = (i / spatial / channels) * groups + channel / (channels / groups);
    if (spatial == 1) {
        const float normalized = (__half2float(input[i]) - __half2float(mean[group])) * __half2float(rstd[group]);
        output[i] = __float2half(normalized * weight[channel] + bias[channel]);
        return;
    }
    const float a = __half2float(rstd[group]) * weight[channel];
    const float b = -a * __half2float(mean[group]) + bias[channel];
    output[i] = __float2half(a * __half2float(input[i]) + b);
}
