// __pipeline_*: on the Metal backend the copy is plain, and commit and wait have nothing to do.
#pragma once
#include <cuda_runtime.h>
static __device__ __forceinline__ void __pipeline_memcpy_async(void *dst, const void *src, size_t n, size_t zfill = 0) {
  if (n == 16 && !zfill) { *(uint4 *)dst = *(const uint4 *)src; return; }
  if (n == 8 && !zfill) { *(uint2 *)dst = *(const uint2 *)src; return; }
  if (n == 4 && !zfill) { *(unsigned *)dst = *(const unsigned *)src; return; }
  for (size_t i = 0; i < n - zfill; i++) ((char *)dst)[i] = ((const char *)src)[i];
  for (size_t i = n - zfill; i < n; i++) ((char *)dst)[i] = 0;
}
static __device__ __forceinline__ void __pipeline_commit() {}
static __device__ __forceinline__ void __pipeline_wait_prior(size_t) {}
