// What metal_runtime.mm shares with the programs. It compiles against the same CUDA declarations.
#pragma once
#include <cuda.h>
#include <cublas_v2.h>
#include <cublasLt.h>
int mtl_raw_alloc(void **out, size_t n);
size_t mtl_peak_bytes();
