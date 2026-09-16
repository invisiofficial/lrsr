#include "w4a16.h"
#include <cuda_fp16.h>
#include <cutlass/arch/wmma.h>
#include <cutlass/layout/matrix.h>

namespace {
using Mma = cutlass::arch::Wmma<
    cutlass::gemm::GemmShape<16, 16, 16>,
    cutlass::half_t, cutlass::layout::RowMajor,
    cutlass::half_t, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor>;

constexpr int BN = 64, BK = 64, LD = 72;

// General Matrix Multiplication Kernel
template<int BM>
__global__ void kernel_gemm(const half* __restrict__ x, const uint8_t* __restrict__ packed,
                            half* __restrict__ output, int m, int n, int k) {
    constexpr int Threads = (BM / 16) * (BN / 16) * 32;
    __shared__ __align__(32) half a[BM * LD], b[BN * LD];
    __shared__ __align__(32) float c[BM * BN];

    const int tid = threadIdx.x;
    const int wm = (tid / 32) / (BN / 16), wn = (tid / 32) % (BN / 16);
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;

    Mma::FragmentC acc;
    nvcuda::wmma::fill_fragment(acc, 0.0f);

    for (int k0 = 0; k0 < k; k0 += BK) {
        for (int i = tid; i < BM * BK / 2; i += Threads) {
            int row = i / (BK / 2), col = (i % (BK / 2)) * 2;
            half2 val = __float2half2_rn(0.0f);
            if (m0 + row < m && k0 + col < k)
                val = *reinterpret_cast<const half2*>(x + int64_t(m0 + row) * k + k0 + col);
            *reinterpret_cast<half2*>(a + row * LD + col) = val;
        }
        for (int i = tid; i < BN * BK / 8; i += Threads) {
            int col = i / (BK / 8), red = (i % (BK / 8)) * 8;
            uint32_t word = 0;
            if (n0 + col < n && k0 + red < k)
                word = *reinterpret_cast<const uint32_t*>(packed + int64_t(n0 + col) * (k / 2) + (k0 + red) / 2);
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                int q = ((word >> (j * 4)) & 15);
                b[col * LD + red + j] = __float2half_rn(float((q ^ 8) - 8));
            }
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            Mma::FragmentA fa; Mma::FragmentB fb;
            nvcuda::wmma::load_matrix_sync(fa, a + wm * 16 * LD + kk, LD);
            nvcuda::wmma::load_matrix_sync(fb, b + wn * 16 * LD + kk, LD);
            Mma{}(acc, fa, fb, acc);
        }
        __syncthreads();
    }
    nvcuda::wmma::store_matrix_sync(c + wm * 16 * BN + wn * 16, acc, BN, nvcuda::wmma::mem_row_major);
    __syncthreads();

    for (int i = tid; i < BM * BN; i += Threads) {
        int row = i / BN, col = i % BN;
        if (m0 + row < m && n0 + col < n) {
            output[int64_t(m0 + row) * n + n0 + col] = __float2half_rn(c[i]);
        }
    }
}

// Per-Group Matrix Multiplication Kernel
template<int BM>
__global__ void kernel_pgmm(const half* __restrict__ x, const uint8_t* __restrict__ packed,
                            const half* __restrict__ group_scales,
                            half* __restrict__ output, int m, int n, int k) {
    constexpr int Threads = (BM / 16) * (BN / 16) * 32;
    __shared__ __align__(32) half a[BM * LD], b[BN * LD];
    __shared__ __align__(32) float c[BM * BN];

    const int tid = threadIdx.x;
    const int wm = (tid / 32) / (BN / 16), wn = (tid / 32) % (BN / 16);
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;

    Mma::FragmentC acc;
    nvcuda::wmma::fill_fragment(acc, 0.0f);

    for (int k0 = 0; k0 < k; k0 += BK) {
        for (int i = tid; i < BM * BK / 2; i += Threads) {
            int row = i / (BK / 2), col = (i % (BK / 2)) * 2;
            half2 val = __float2half2_rn(0.0f);
            if (m0 + row < m && k0 + col < k)
                val = *reinterpret_cast<const half2*>(x + int64_t(m0 + row) * k + k0 + col);
            *reinterpret_cast<half2*>(a + row * LD + col) = val;
        }
        for (int i = tid; i < BN * BK / 8; i += Threads) {
            int col = i / (BK / 8), red = (i % (BK / 8)) * 8;
            uint32_t word = 0;
            float scale = 1.0f;
            if (n0 + col < n && k0 + red < k) {
                word = *reinterpret_cast<const uint32_t*>(packed + int64_t(n0 + col) * (k / 2) + (k0 + red) / 2);
                scale = __half2float(group_scales[int64_t((k0 + red) / 128) * n + n0 + col]);
            }
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                int q = ((word >> (j * 4)) & 15);
                b[col * LD + red + j] = __float2half_rn(float((q ^ 8) - 8) * scale);
            }
        }
        __syncthreads();
        #pragma unroll
        for (int kk = 0; kk < BK; kk += 16) {
            Mma::FragmentA fa; Mma::FragmentB fb;
            nvcuda::wmma::load_matrix_sync(fa, a + wm * 16 * LD + kk, LD);
            nvcuda::wmma::load_matrix_sync(fb, b + wn * 16 * LD + kk, LD);
            Mma{}(acc, fa, fb, acc);
        }
        __syncthreads();
    }
    nvcuda::wmma::store_matrix_sync(c + wm * 16 * BN + wn * 16, acc, BN, nvcuda::wmma::mem_row_major);
    __syncthreads();

    for (int i = tid; i < BM * BN; i += Threads) {
        int row = i / BN, col = i % BN;
        if (m0 + row < m && n0 + col < n) {
            output[int64_t(m0 + row) * n + n0 + col] = __float2half_rn(c[i]);
        }
    }
}

__global__ void pack(const int8_t* input, uint8_t* output, int64_t pairs) {
    for (int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x; i < pairs; i += int64_t(gridDim.x) * blockDim.x) {
        output[i] = (uint8_t(input[2 * i]) & 15) | ((uint8_t(input[2 * i + 1]) & 15) << 4);
    }
}

} // namespace

extern "C" {

int cutlass_pack_int4(const int8_t* input, uint8_t* output, int64_t elements, cudaStream_t stream) {
    if (!input || !output || elements <= 0 || elements % 2 != 0) return cudaErrorInvalidValue;
    int64_t blocks = ((elements / 2 - 1) / 256) + 1;
    pack<<<static_cast<unsigned>(blocks < 65535 ? blocks : 65535), 256, 0, stream>>>(input, output, elements / 2);
    return cudaGetLastError();
}

int cutlass_gemm(const void* x, const void* packed, void* output, int m, int n, int k, cudaStream_t stream) {
    dim3 grid((n + 63) / 64, (m + 31) / 32);
    if (m <= 16) {
        grid.y = (m + 15) / 16;
        kernel_gemm<16><<<grid, 128, 0, stream>>>((const half*)x, (const uint8_t*)packed, (half*)output, m, n, k);
    } else {
        kernel_gemm<32><<<grid, 256, 0, stream>>>((const half*)x, (const uint8_t*)packed, (half*)output, m, n, k);
    }
    return cudaGetLastError();
}

int cutlass_pgmm(const void* x, const void* packed, const void* group_scales, void* output, int m, int n, int k, cudaStream_t stream) {
    dim3 grid((n + 63) / 64, (m + 31) / 32);
    if (m <= 16) {
        grid.y = (m + 15) / 16;
        kernel_pgmm<16><<<grid, 128, 0, stream>>>((const half*)x, (const uint8_t*)packed, (const half*)group_scales, (half*)output, m, n, k);
    } else {
        kernel_pgmm<32><<<grid, 256, 0, stream>>>((const half*)x, (const uint8_t*)packed, (const half*)group_scales, (half*)output, m, n, k);
    }
    return cudaGetLastError();
}

}