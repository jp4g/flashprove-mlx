// The part of cuBLASLt that FlashProve uses (Metal backend).
#pragma once
#include <cublas_v2.h>
typedef struct cublasLtContext *cublasLtHandle_t;
typedef struct cublasLtMatmulDescOpaque *cublasLtMatmulDesc_t;
typedef struct cublasLtLayoutOpaque *cublasLtMatrixLayout_t;
typedef struct cublasLtPrefOpaque *cublasLtMatmulPreference_t;
struct cublasLtMatmulAlgo_t { uint64_t data[8]; };
struct cublasLtMatmulHeuristicResult_t { cublasLtMatmulAlgo_t algo; size_t workspaceSize; cublasStatus_t state; float wavesCount; int reserved[4]; };
enum cublasLtMatmulDescAttributes_t { CUBLASLT_MATMUL_DESC_TRANSA = 3, CUBLASLT_MATMUL_DESC_TRANSB = 4 };
enum cublasLtMatrixLayoutAttribute_t { CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT = 5, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET = 6, CUBLASLT_MATRIX_LAYOUT_PLANE_OFFSET = 7 };
enum cublasLtMatmulPreferenceAttributes_t { CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES = 1, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES = 5 };
extern "C" {
cublasStatus_t cublasLtCreate(cublasLtHandle_t *);
cublasStatus_t cublasLtMatmulDescCreate(cublasLtMatmulDesc_t *, cublasComputeType_t, cudaDataType);
cublasStatus_t cublasLtMatmulDescSetAttribute(cublasLtMatmulDesc_t, cublasLtMatmulDescAttributes_t, const void *, size_t);
cublasStatus_t cublasLtMatrixLayoutCreate(cublasLtMatrixLayout_t *, cudaDataType, uint64_t, uint64_t, int64_t);
cublasStatus_t cublasLtMatrixLayoutSetAttribute(cublasLtMatrixLayout_t, cublasLtMatrixLayoutAttribute_t, const void *, size_t);
cublasStatus_t cublasLtMatmulPreferenceCreate(cublasLtMatmulPreference_t *);
cublasStatus_t cublasLtMatmulPreferenceDestroy(cublasLtMatmulPreference_t);
cublasStatus_t cublasLtMatmulPreferenceSetAttribute(cublasLtMatmulPreference_t, cublasLtMatmulPreferenceAttributes_t, const void *, size_t);
cublasStatus_t cublasLtMatmulAlgoGetHeuristic(cublasLtHandle_t, cublasLtMatmulDesc_t, cublasLtMatrixLayout_t, cublasLtMatrixLayout_t, cublasLtMatrixLayout_t, cublasLtMatrixLayout_t, cublasLtMatmulPreference_t, int, cublasLtMatmulHeuristicResult_t *, int *);
cublasStatus_t cublasLtMatmul(cublasLtHandle_t, cublasLtMatmulDesc_t, const void *, const void *, cublasLtMatrixLayout_t, const void *, cublasLtMatrixLayout_t, const void *, const void *, cublasLtMatrixLayout_t, void *, cublasLtMatrixLayout_t, const cublasLtMatmulAlgo_t *, void *, size_t, cudaStream_t);
}
