// The part of the CUDA runtime API that FlashProve's sources use, for the Metal backend
// (prover/metal/README.md). clang compiles the unchanged .cu files against this header: their device
// code to PTX, which prover/metal/ptx2msl.py turns into Metal Shading Language, their host code to
// arm64 calling prover/metal/runtime/metal_runtime.mm. Names and meanings are CUDA's.
#pragma once
#define __CUDA_METAL_SHIM__ 1
#ifndef __CUDACC__
#define __CUDACC__ 1
#endif
#define CUDA_API_PER_THREAD_DEFAULT_STREAM 1
#define CUDART_VERSION 13020
#define __CUDACC_VER_MAJOR__ 13
#define __CUDACC_VER_MINOR__ 2

#ifdef __CUDA__
#define __host__ __attribute__((host))
#define __device__ __attribute__((device))
#define __global__ __attribute__((global))
#define __shared__ __attribute__((shared))
#define __constant__ __attribute__((constant))
#define __managed__ __attribute__((managed))
#define __launch_bounds__(...) __attribute__((launch_bounds(__VA_ARGS__)))
#define __forceinline__ __inline__ __attribute__((always_inline))
#define __noinline__ __attribute__((noinline))
#define __grid_constant__
#else // the runtime itself, compiled as plain Objective-C++
#define __host__
#define __device__
#define __global__
#define __shared__
#define __constant__
#define __managed__
#define __launch_bounds__(...)
#define __forceinline__ inline
#define __noinline__
#define __grid_constant__
#endif
#define __align__(n) __attribute__((aligned(n)))
#define __restrict__ __restrict
#define __builtin_align__(n) __attribute__((aligned(n)))

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <algorithm>
#include <limits>

#include "metal_compat.h"
struct uint3 { unsigned x, y, z; };
struct dim3;
#ifdef __CUDA__
#include <__clang_cuda_builtin_vars.h>
#endif

struct dim3 {
  unsigned x, y, z;
  __host__ __device__ constexpr dim3(unsigned x_ = 1, unsigned y_ = 1, unsigned z_ = 1) : x(x_), y(y_), z(z_) {}
  __host__ __device__ constexpr dim3(uint3 v) : x(v.x), y(v.y), z(v.z) {}
};

#define DEFVEC2(N, T) struct __attribute__((aligned(2 * sizeof(T)))) N { T x, y; }; \
  __host__ __device__ inline N make_##N(T x, T y) { N v; v.x = x; v.y = y; return v; }
#define DEFVEC4(N, T, A) struct __attribute__((aligned(A))) N { T x, y, z, w; }; \
  __host__ __device__ inline N make_##N(T x, T y, T z, T w) { N v; v.x = x; v.y = y; v.z = z; v.w = w; return v; }
DEFVEC2(uint2, unsigned) DEFVEC2(int2, int) DEFVEC2(ulonglong2, unsigned long long) DEFVEC2(longlong2, long long)
DEFVEC2(ushort2, unsigned short) DEFVEC2(short2, short) DEFVEC2(float2, float) DEFVEC2(double2, double) DEFVEC2(char2, signed char) DEFVEC2(uchar2, unsigned char)
DEFVEC4(uint4, unsigned, 16) DEFVEC4(int4, int, 16) DEFVEC4(float4, float, 16) DEFVEC4(char4, signed char, 4) DEFVEC4(uchar4, unsigned char, 4)
DEFVEC4(ushort4, unsigned short, 8) DEFVEC4(short4, short, 8) DEFVEC4(ulonglong4, unsigned long long, 32) DEFVEC4(longlong4, long long, 32) 
__host__ __device__ inline uint3 make_uint3(unsigned x, unsigned y, unsigned z) { uint3 v; v.x = x; v.y = y; v.z = z; return v; }

enum cudaError { cudaSuccess = 0, cudaErrorInvalidValue = 1, cudaErrorMemoryAllocation = 2, cudaErrorNotReady = 600,
                 cudaErrorMisalignedAddress = 716, cudaErrorUnknown = 999 };
typedef cudaError cudaError_t;
typedef struct CUstream_st *cudaStream_t;
typedef struct CUevent_st *cudaEvent_t;
typedef struct CUgraph_st *cudaGraph_t;
typedef struct CUgraphExec_st *cudaGraphExec_t;
typedef struct CUmempool_st *cudaMemPool_t;
#define cudaStreamLegacy ((cudaStream_t)1)
#define cudaStreamPerThread ((cudaStream_t)2)
enum cudaMemcpyKind { cudaMemcpyHostToHost, cudaMemcpyHostToDevice, cudaMemcpyDeviceToHost, cudaMemcpyDeviceToDevice, cudaMemcpyDefault };
#define cudaStreamDefault 0u
#define cudaStreamNonBlocking 1u
#define cudaEventDefault 0u
#define cudaEventBlockingSync 1u
#define cudaEventDisableTiming 2u
#define cudaHostAllocDefault 0u
#define cudaHostAllocMapped 2u
#define cudaHostRegisterDefault 0u
#define cudaHostRegisterPortable 1u
#define cudaHostRegisterMapped 2u
#define cudaMemAttachGlobal 1u
#define cudaGraphInstantiateFlagUseNodePriority 8ull
enum cudaFuncAttribute { cudaFuncAttributeMaxDynamicSharedMemorySize = 8, cudaFuncAttributePreferredSharedMemoryCarveout = 9 };
enum cudaDeviceAttr { cudaDevAttrClockRate = 13, cudaDevAttrMultiProcessorCount = 16, cudaDevAttrMaxSharedMemoryPerBlockOptin = 97 };
enum cudaLimit { cudaLimitStackSize = 0, cudaLimitMallocHeapSize = 2 };
enum cudaMemPoolAttr { cudaMemPoolAttrReleaseThreshold = 4, cudaMemPoolAttrReservedMemCurrent = 5, cudaMemPoolAttrReservedMemHigh = 6 };
enum cudaStreamCaptureMode { cudaStreamCaptureModeGlobal, cudaStreamCaptureModeThreadLocal, cudaStreamCaptureModeRelaxed };
enum cudaStreamCaptureStatus { cudaStreamCaptureStatusNone, cudaStreamCaptureStatusActive, cudaStreamCaptureStatusInvalidated };
enum cudaLaunchAttributeID { cudaLaunchAttributeIgnore = 0, cudaLaunchAttributeCooperative = 2, cudaLaunchAttributePriority = 8 };
union cudaLaunchAttributeValue { int cooperative; int priority; char pad[64]; };
struct cudaLaunchAttribute { cudaLaunchAttributeID id; char pad[4]; cudaLaunchAttributeValue val; };
struct cudaLaunchConfig_t { dim3 gridDim; dim3 blockDim; size_t dynamicSmemBytes; cudaStream_t stream; cudaLaunchAttribute *attrs; unsigned numAttrs; };
struct cudaFuncAttributes { size_t sharedSizeBytes, constSizeBytes, localSizeBytes; int maxThreadsPerBlock, numRegs, ptxVersion, binaryVersion, cacheModeCA, maxDynamicSharedSizeBytes, preferredShmemCarveout; };
struct cudaDeviceProp {
  char name[256]; size_t totalGlobalMem, sharedMemPerBlock, sharedMemPerBlockOptin, sharedMemPerMultiprocessor; int regsPerBlock, warpSize, maxThreadsPerBlock, maxThreadsPerMultiProcessor, multiProcessorCount, major, minor, clockRate, memoryClockRate, memoryBusWidth, l2CacheSize, canMapHostMemory, unifiedAddressing, managedMemory, concurrentManagedAccess, pciBusID, pciDeviceID, pciDomainID;
  char uuid[16];
};

extern "C" {
const char *cudaGetErrorString(cudaError_t);
const char *cudaGetErrorName(cudaError_t);
cudaError_t cudaGetLastError(void);
cudaError_t cudaPeekAtLastError(void);
cudaError_t cudaGetDevice(int *);
cudaError_t cudaSetDevice(int);
cudaError_t cudaGetDeviceCount(int *);
cudaError_t cudaGetDeviceProperties(cudaDeviceProp *, int);
cudaError_t cudaDeviceGetAttribute(int *, cudaDeviceAttr, int);
cudaError_t cudaDeviceGetPCIBusId(char *, int, int);
cudaError_t cudaDeviceGetStreamPriorityRange(int *, int *);
cudaError_t cudaDeviceSynchronize(void);
cudaError_t cudaDeviceSetLimit(cudaLimit, size_t);
cudaError_t cudaDeviceGetMemPool(cudaMemPool_t *, int);
cudaError_t cudaMemPoolGetAttribute(cudaMemPool_t, cudaMemPoolAttr, void *);
cudaError_t cudaMemPoolSetAttribute(cudaMemPool_t, cudaMemPoolAttr, void *);
cudaError_t cudaMemPoolTrimTo(cudaMemPool_t, size_t);
cudaError_t cudaMallocFromPoolAsync(void **, size_t, cudaMemPool_t, cudaStream_t);
cudaError_t cudaFreeAsync(void *, cudaStream_t);
cudaError_t cudaMalloc(void **, size_t);
cudaError_t cudaMallocHost(void **, size_t);
cudaError_t cudaMallocManaged(void **, size_t, unsigned);
cudaError_t cudaMallocPitch(void **, size_t *, size_t, size_t);
cudaError_t cudaHostAlloc(void **, size_t, unsigned);
cudaError_t cudaHostRegister(void *, size_t, unsigned);
cudaError_t cudaHostUnregister(void *);
cudaError_t cudaHostGetDevicePointer(void **, void *, unsigned);
cudaError_t cudaFree(void *);
cudaError_t cudaFreeHost(void *);
cudaError_t cudaMemGetInfo(size_t *, size_t *);
cudaError_t cudaMemcpy(void *, const void *, size_t, cudaMemcpyKind);
cudaError_t cudaMemcpyAsync(void *, const void *, size_t, cudaMemcpyKind, cudaStream_t);
cudaError_t cudaMemcpy2DAsync(void *, size_t, const void *, size_t, size_t, size_t, cudaMemcpyKind, cudaStream_t);
cudaError_t cudaMemset(void *, int, size_t);
cudaError_t cudaMemsetAsync(void *, int, size_t, cudaStream_t);
cudaError_t cudaMemset2DAsync(void *, size_t, int, size_t, size_t, cudaStream_t);
cudaError_t cudaStreamCreate(cudaStream_t *);
cudaError_t cudaStreamCreateWithFlags(cudaStream_t *, unsigned);
cudaError_t cudaStreamCreateWithPriority(cudaStream_t *, unsigned, int);
cudaError_t cudaStreamDestroy(cudaStream_t);
cudaError_t cudaStreamSynchronize(cudaStream_t);
cudaError_t cudaStreamWaitEvent(cudaStream_t, cudaEvent_t, unsigned);
cudaError_t cudaStreamGetFlags(cudaStream_t, unsigned *);
cudaError_t cudaStreamIsCapturing(cudaStream_t, cudaStreamCaptureStatus *);
cudaError_t cudaStreamBeginCapture(cudaStream_t, cudaStreamCaptureMode);
cudaError_t cudaStreamEndCapture(cudaStream_t, cudaGraph_t *);
cudaError_t cudaGraphInstantiate(cudaGraphExec_t *, cudaGraph_t, unsigned long long);
cudaError_t cudaGraphLaunch(cudaGraphExec_t, cudaStream_t);
cudaError_t cudaGraphDestroy(cudaGraph_t);
cudaError_t cudaGraphExecDestroy(cudaGraphExec_t);
cudaError_t cudaEventCreate(cudaEvent_t *);
cudaError_t cudaEventCreateWithFlags(cudaEvent_t *, unsigned);
cudaError_t cudaEventDestroy(cudaEvent_t);
cudaError_t cudaEventRecord(cudaEvent_t, cudaStream_t);
cudaError_t cudaEventRecordWithFlags(cudaEvent_t, cudaStream_t, unsigned);
cudaError_t cudaEventSynchronize(cudaEvent_t);
cudaError_t cudaEventQuery(cudaEvent_t);
cudaError_t cudaEventElapsedTime(float *, cudaEvent_t, cudaEvent_t);
cudaError_t cudaFuncSetAttribute(const void *, cudaFuncAttribute, int);
cudaError_t cudaFuncGetAttributes(cudaFuncAttributes *, const void *);
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *, const void *, int, size_t);
cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *, const void *, int, size_t, unsigned);
cudaError_t cudaLaunchKernel(const void *, dim3, dim3, void **, size_t, cudaStream_t);
cudaError_t cudaLaunchCooperativeKernel(const void *, dim3, dim3, void **, size_t, cudaStream_t);
cudaError_t cudaLaunchKernelExC(const cudaLaunchConfig_t *, const void *, void **);
cudaError_t cudaGetSymbolAddress(void **, const void *);
cudaError_t cudaMemcpyToSymbol(const void *, const void *, size_t, size_t, cudaMemcpyKind);
cudaError_t cudaMemcpyFromSymbol(void *, const void *, size_t, size_t, cudaMemcpyKind);
// What clang's CUDA front end calls for <<<...>>> and for its module constructor.
unsigned __cudaPushCallConfiguration(dim3, dim3, size_t = 0, void * = nullptr);
cudaError_t __cudaPopCallConfiguration(dim3 *, dim3 *, size_t *, void *);
}
template <class T> static inline cudaError_t cudaMalloc(T **p, size_t n) { return cudaMalloc((void **)p, n); }
template <class T> static inline cudaError_t cudaMallocHost(T **p, size_t n) { return cudaMallocHost((void **)p, n); }
template <class T> static inline cudaError_t cudaMallocManaged(T **p, size_t n, unsigned f = cudaMemAttachGlobal) { return cudaMallocManaged((void **)p, n, f); }
static inline cudaError_t cudaMallocManaged(void **p, size_t n) { return cudaMallocManaged(p, n, cudaMemAttachGlobal); }
template <class T> static inline cudaError_t cudaMallocPitch(T **p, size_t *pitch, size_t w, size_t h) { return cudaMallocPitch((void **)p, pitch, w, h); }
template <class T> static inline cudaError_t cudaHostAlloc(T **p, size_t n, unsigned f) { return cudaHostAlloc((void **)p, n, f); }
template <class T, class U> static inline cudaError_t cudaHostGetDevicePointer(T **p, U *h, unsigned f) { return cudaHostGetDevicePointer((void **)p, (void *)h, f); }
template <class T> static inline cudaError_t cudaMallocFromPoolAsync(T **p, size_t n, cudaMemPool_t m, cudaStream_t s) { return cudaMallocFromPoolAsync((void **)p, n, m, s); }
template <class T> static inline cudaError_t cudaFuncSetAttribute(T *f, cudaFuncAttribute a, int v) { return cudaFuncSetAttribute((const void *)f, a, v); }
template <class T> static inline cudaError_t cudaFuncGetAttributes(cudaFuncAttributes *a, T *f) { return cudaFuncGetAttributes(a, (const void *)f); }
template <class T> static inline cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessor(int *n, T f, int b, size_t s) { return cudaOccupancyMaxActiveBlocksPerMultiprocessor(n, (const void *)f, b, s); }
template <class T> static inline cudaError_t cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(int *n, T f, int b, size_t s, unsigned g) { return cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(n, (const void *)f, b, s, g); }
template <class T> static inline cudaError_t cudaLaunchKernel(T *f, dim3 g, dim3 b, void **a, size_t s = 0, cudaStream_t st = 0) { return cudaLaunchKernel((const void *)f, g, b, a, s, st); }
template <class T> static inline cudaError_t cudaLaunchCooperativeKernel(T *f, dim3 g, dim3 b, void **a, size_t s = 0, cudaStream_t st = 0) { return cudaLaunchCooperativeKernel((const void *)f, g, b, a, s, st); }
template <class T> static inline cudaError_t cudaGetSymbolAddress(void **p, const T &s) { return cudaGetSymbolAddress(p, (const void *)&s); }
template <class T> static inline cudaError_t cudaMemcpyToSymbol(const T &s, const void *src, size_t n, size_t off = 0, cudaMemcpyKind k = cudaMemcpyHostToDevice) { return cudaMemcpyToSymbol((const void *)&s, src, n, off, k); }
template <class T> static inline cudaError_t cudaMemcpyFromSymbol(void *dst, const T &s, size_t n, size_t off = 0, cudaMemcpyKind k = cudaMemcpyDeviceToHost) { return cudaMemcpyFromSymbol(dst, (const void *)&s, n, off, k); }
static inline cudaError_t cudaGraphInstantiate(cudaGraphExec_t *e, cudaGraph_t g) { return cudaGraphInstantiate(e, g, 0); }
// cudaLaunchKernelEx: the arguments are packed as a call of a kernel.
template <class... A, class... P>
static inline cudaError_t cudaLaunchKernelEx(const cudaLaunchConfig_t *c, void (*f)(A...), P &&...p) {
  void *args[sizeof...(P) + 1] = {(void *)&p...};
  return cudaLaunchKernelExC(c, (const void *)f, args);
}
#define cudaStreamGetCaptureInfo cudaStreamGetCaptureInfo_shim
static inline cudaError_t cudaStreamGetCaptureInfo_shim(cudaStream_t s, cudaStreamCaptureStatus *st, ...) { *st = cudaStreamCaptureStatusNone; (void)s; return cudaSuccess; }

#include "metal_device_fns.h"
