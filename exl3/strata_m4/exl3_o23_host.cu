// Opt23: host side of exl3_o23.fatbin (launch stub registered under the fatbin's kernel name) + the C API verify.cpp dlopens (STRATA_O23_LIB).
#include <cuda_fp16.h>
#include <cstdint>
extern "C" __global__ void o23_doorbell(const float*, int, const int32_t*, const float*, int, const int32_t*, int, void*, int, int32_t*, float*,
                                        int32_t*, uint32_t*, uint32_t, uint32_t*) {}
extern "C" int exl3o_doorbell(const float* x, int n, const int32_t* ids, const float* w, int k, const int32_t* d_res, int n_expert, void* x_out,
                              int x_half, int32_t* ids_out, float* w_out, int32_t* aux_out, uint32_t* skip_dev, uint32_t skip_val, uint32_t* seq,
                              cudaStream_t s) {
    if (k < 1 || k > 1024) return -1;
    o23_doorbell<<<1, 1024, 0, s>>>(x, n, ids, w, k, d_res, n_expert, x_out, x_half, ids_out, w_out, aux_out, skip_dev, skip_val, seq);
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}
extern "C" __global__ void o23_add_h(float*, __half*, int) {}
extern "C" int exl3o_add_h(float* bo, uint16_t* yh, int n, cudaStream_t s) {
    if (n < 1) return 0;
    o23_add_h<<<(n + 255) / 256, 256, 0, s>>>(bo, (__half*) yh, n);
    return cudaGetLastError() == cudaSuccess ? 0 : -2;
}
