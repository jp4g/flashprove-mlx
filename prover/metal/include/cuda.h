// The part of the CUDA driver API that FlashProve uses (Metal backend).
#pragma once
#include <cuda_runtime.h>
typedef int CUresult;
typedef unsigned long long CUdeviceptr;
typedef cudaEvent_t CUevent;
typedef unsigned long long CUmemGenericAllocationHandle;
#define CUDA_SUCCESS 0
enum { CU_MEM_ALLOCATION_TYPE_PINNED = 1, CU_MEM_LOCATION_TYPE_DEVICE = 1, CU_MEM_ACCESS_FLAGS_PROT_READWRITE = 3, CU_MEM_ALLOC_GRANULARITY_MINIMUM = 0 };
struct CUmemLocation { int type; int id; };
struct CUmemAllocationProp { int type; int requestedHandleTypes; CUmemLocation location; void *win32HandleMetaData; struct { unsigned char compressionType, gpuDirectRDMACapable; unsigned short usage; } allocFlags; };
struct CUmemAccessDesc { CUmemLocation location; int flags; };
extern "C" {
CUresult cuGetErrorString(CUresult, const char **);
CUresult cuEventElapsedTime(float *, CUevent, CUevent);
CUresult cuMemAddressReserve(CUdeviceptr *, size_t, size_t, CUdeviceptr, unsigned long long);
CUresult cuMemAddressFree(CUdeviceptr, size_t);
CUresult cuMemCreate(CUmemGenericAllocationHandle *, size_t, const CUmemAllocationProp *, unsigned long long);
CUresult cuMemRelease(CUmemGenericAllocationHandle);
CUresult cuMemMap(CUdeviceptr, size_t, size_t, CUmemGenericAllocationHandle, unsigned long long);
CUresult cuMemUnmap(CUdeviceptr, size_t);
CUresult cuMemSetAccess(CUdeviceptr, size_t, const CUmemAccessDesc *, size_t);
CUresult cuMemGetAllocationGranularity(size_t *, const CUmemAllocationProp *, int);
}
