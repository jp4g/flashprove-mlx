// float16 as FlashProve's kernels read it (Metal backend).
#pragma once
#include <cuda_runtime.h>
struct __half { unsigned short x; };
struct __half2 { __half x, y; };
#ifdef __CUDA__
static __device__ __forceinline__ float __half2float(__half h) { float f; unsigned short b = h.x; asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(b)); return f; }
static __device__ __forceinline__ float __low2float(__half2 h) { return __half2float(h.x); }
static __device__ __forceinline__ float __high2float(__half2 h) { return __half2float(h.y); }
static __device__ __forceinline__ __half __ushort_as_half(unsigned short u) { __half h; h.x = u; return h; }
#endif
