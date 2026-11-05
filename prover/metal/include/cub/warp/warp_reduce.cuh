// cub::WarpReduce, with shuffles. The operators here are exact and associative.
#pragma once
#include <cuda_runtime.h>
namespace cub {
template <class T> struct WarpReduce {
  struct TempStorage { char unused; };
  __device__ WarpReduce(TempStorage &) {}
  template <class Op> __device__ __forceinline__ T Reduce(T v, Op op) {
    for (int d = 16; d; d >>= 1) {
      constexpr int W = (sizeof(T) + 3) / 4;
      unsigned w[W] = {};
      __builtin_memcpy(w, &v, sizeof(T));
      for (int i = 0; i < W; i++) w[i] = __shfl_down_sync(0xFFFFFFFFu, w[i], d);
      T o;
      __builtin_memcpy(&o, w, sizeof(T));
      v = op(v, o);
    }
    return v;
  }
};
}
