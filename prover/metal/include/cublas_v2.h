// The part of cuBLAS that FlashProve uses (Metal backend): exact integer products on the GPU
// (prover/metal/runtime/), and one double-precision product.
#pragma once
#include <cuda_runtime.h>
typedef struct cublasContext *cublasHandle_t;
enum cublasStatus_t { CUBLAS_STATUS_SUCCESS = 0, CUBLAS_STATUS_ALLOC_FAILED = 3, CUBLAS_STATUS_NOT_SUPPORTED = 15 };
enum cublasOperation_t { CUBLAS_OP_N = 0, CUBLAS_OP_T = 1 };
enum cublasMath_t { CUBLAS_DEFAULT_MATH = 0, CUBLAS_PEDANTIC_MATH = 2 };
enum cublasComputeType_t { CUBLAS_COMPUTE_32I = 72 };
enum cublasGemmAlgo_t { CUBLAS_GEMM_DEFAULT = -1 };
enum cudaDataType_t { CUDA_R_8I = 3, CUDA_R_32I = 10 };
typedef cudaDataType_t cudaDataType;
extern "C" {
cublasStatus_t cublasCreate(cublasHandle_t *);
cublasStatus_t cublasDestroy(cublasHandle_t);
cublasStatus_t cublasSetStream(cublasHandle_t, cudaStream_t);
cublasStatus_t cublasSetWorkspace(cublasHandle_t, void *, size_t);
cublasStatus_t cublasSetMathMode(cublasHandle_t, cublasMath_t);
cublasStatus_t cublasGemmEx(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const void *, const void *, cudaDataType, int, const void *, cudaDataType, int, const void *, void *, cudaDataType, int, cublasComputeType_t, cublasGemmAlgo_t);
cublasStatus_t cublasGemmStridedBatchedEx(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const void *, const void *, cudaDataType, int, long long, const void *, cudaDataType, int, long long, const void *, void *, cudaDataType, int, long long, int, cublasComputeType_t, cublasGemmAlgo_t);
cublasStatus_t cublasDgemm(cublasHandle_t, cublasOperation_t, cublasOperation_t, int, int, int, const double *, const double *, int, const double *, int, const double *, double *, int);
}
