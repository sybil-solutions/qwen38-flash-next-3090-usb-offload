// Opt23: the slim EXL3 decode doorbell, per-ring payload slots + fp16 x + the "no cold expert" decision.
// Replaces doorbell_publish for the EXL3 slim window (STRATA_O23_*): one block of 1024 threads, as doorbell_publish_kernel.
//   x (fp32 rows) -> x_out (fp16 when x_half, else fp32 copy), ids -> ids_out, w -> w_out  (all in this ring's slot of the mapped payload),
//   any routed expert with d_res[id] < 0 (not in VRAM; the same test as exl3_plan_kernel's miss) -> aux_out[0] = 0, else 1;
//   skip_dev (device word the window's wait kernel compares with its ring value) = skip_val when no expert is cold, else 0;
//   then the fence and the ring (seq += 1), exactly doorbell_publish's order.
#include <cuda_fp16.h>
#include <stdint.h>

extern "C" __global__ void o23_doorbell(const float* __restrict__ x, int n, const int32_t* __restrict__ ids, const float* __restrict__ w, int k,
                                        const int32_t* __restrict__ d_res, int n_expert, void* x_out, int x_half, int32_t* ids_out,
                                        float* w_out, int32_t* aux_out, uint32_t* skip_dev, uint32_t skip_val, uint32_t* seq) {
    int any_miss = 0;
    if ((int) threadIdx.x < k) {
        const int32_t id = ids[threadIdx.x];
        ids_out[threadIdx.x] = id;
        w_out[threadIdx.x] = w[threadIdx.x];
        if (id < 0 || id >= n_expert || d_res[id] < 0) any_miss = 1;
    }
    const int miss = __syncthreads_or(any_miss);
    if (x_half) {
        __half* xo = (__half*) x_out;
        for (int i = threadIdx.x; i < n; i += blockDim.x) xo[i] = __float2half_rn(x[i]);
    } else {
        float* xo = (float*) x_out;
        for (int i = threadIdx.x; i < n; i += blockDim.x) xo[i] = x[i];
    }
    if (threadIdx.x == 0) {
        if (aux_out != nullptr) *(volatile int32_t*) aux_out = miss ? 0 : 1;
        if (skip_dev != nullptr) *(volatile uint32_t*) skip_dev = miss ? 0u : skip_val;
    }
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) *(volatile uint32_t*) seq = *(volatile uint32_t*) seq + 1u;
}

// y rows as fp16 (host -> device): bo += float(yh), then the mirror is zero again (replaces add_inplace + the memset of the fp32 mirror).
extern "C" __global__ void o23_add_h(float* __restrict__ bo, __half* __restrict__ yh, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { bo[i] += __half2float(yh[i]); yh[i] = __float2half_rn(0.0f); }
}
