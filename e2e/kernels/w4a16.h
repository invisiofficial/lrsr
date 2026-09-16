#pragma once

#include <cuda_runtime_api.h>
#include <cstdint>

#if defined(_WIN32)
#define CUTLASS_EXPORT __declspec(dllexport)
#else
#define CUTLASS_EXPORT __attribute__((visibility("default")))
#endif

extern "C" {

// Packs INT8 weights into UINT8 tensor (2 INT4 elements per byte)
CUTLASS_EXPORT int cutlass_pack_int4(const int8_t* input, uint8_t* output, int64_t elements, cudaStream_t stream);

// W4A16 General Matrix Multiplication
CUTLASS_EXPORT int cutlass_gemm(
    const void* x, const void* packed,
    void* output, int m, int n, int k, cudaStream_t stream);

// W4A16 Per-Group Matrix Multiplication
CUTLASS_EXPORT int cutlass_pgmm(
    const void* x, const void* packed, const void* group_scales,
    void* output, int m, int n, int k, cudaStream_t stream);

}
