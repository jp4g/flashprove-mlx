// cub::BlockReduce through shared memory. The result is valid in thread 0, as in CUB.
#pragma once
#include <cuda_runtime.h>
namespace cub {
template <class T, int TB> struct BlockReduce {
  struct TempStorage { T v[TB]; };
  TempStorage &s;
  __device__ BlockReduce(TempStorage &t) : s(t) {}
  template <class Op> __device__ __forceinline__ T Reduce(T v, Op op) {
    s.v[threadIdx.x] = v;
    __syncthreads();
    if (threadIdx.x == 0) for (int i = 1; i < TB; i++) v = op(v, s.v[i]);
    __syncthreads();
    return v;
  }
};
}
