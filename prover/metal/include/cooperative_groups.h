// this_grid().sync() (Metal backend): an instruction of its own that ptx2msl.py lowers to a barrier
// over the grid; the runtime launches such kernels with every block resident (see README.md).
#pragma once
#include <cuda_runtime.h>
extern "C" __device__ void __mtl_grid_sync();
namespace cooperative_groups {
struct grid_group { __device__ void sync() { __mtl_grid_sync(); } };
static __device__ __forceinline__ grid_group this_grid() { return grid_group(); }
}
