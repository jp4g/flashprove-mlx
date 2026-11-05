// nvcuda::wmma for signed bytes, 16 x 16 x 16 (Metal backend): one 16 x 16 tile is spread over the
// warp, lane l holding row (or column) l / 2, entries 8 (l % 2) .. + 7.
#pragma once
#include <cuda_runtime.h>
namespace nvcuda { namespace wmma {
struct matrix_a {}; struct matrix_b {}; struct accumulator {}; struct row_major {}; struct col_major {};
enum layout_t { mem_row_major, mem_col_major };
template <class Use, int M, int N, int K, class T, class L = void> struct fragment;
template <class L> struct fragment<matrix_a, 16, 16, 16, signed char, L> { signed char x[8]; };
template <class L> struct fragment<matrix_b, 16, 16, 16, signed char, L> { signed char x[8]; };
template <> struct fragment<accumulator, 16, 16, 16, int, void> { int x[8]; };
static __device__ __forceinline__ void fill_fragment(fragment<accumulator, 16, 16, 16, int, void> &f, int v) { for (int i = 0; i < 8; i++) f.x[i] = v; }
template <class L> __device__ __forceinline__ void load_matrix_sync(fragment<matrix_a, 16, 16, 16, signed char, L> &f, const signed char *p, unsigned ld) {
  const int l = threadIdx.x & 31;
  for (int i = 0; i < 8; i++) f.x[i] = p[(size_t)(l >> 1) * ld + (l & 1) * 8 + i];
}
template <class L> __device__ __forceinline__ void load_matrix_sync(fragment<matrix_b, 16, 16, 16, signed char, L> &f, const signed char *p, unsigned ld) {
  const int l = threadIdx.x & 31;
  for (int i = 0; i < 8; i++) f.x[i] = p[(size_t)(l >> 1) * ld + (l & 1) * 8 + i];
}
template <class LA, class LB> __device__ __forceinline__ void mma_sync(fragment<accumulator, 16, 16, 16, int, void> &d, const fragment<matrix_a, 16, 16, 16, signed char, LA> &a, const fragment<matrix_b, 16, 16, 16, signed char, LB> &b, const fragment<accumulator, 16, 16, 16, int, void> &c) {
  const int l = threadIdx.x & 31, r = l >> 1, h = l & 1;
  int out[8];
  for (int i = 0; i < 8; i++) out[i] = c.x[i];
  for (int half = 0; half < 2; half++) {
    unsigned aw[2], own[2];
    __builtin_memcpy(own, a.x, 8);
    aw[0] = __shfl_sync(0xFFFFFFFFu, own[0], 2 * r + half);
    aw[1] = __shfl_sync(0xFFFFFFFFu, own[1], 2 * r + half);
    unsigned bo[2];
    __builtin_memcpy(bo, b.x, 8);
    for (int cc = 0; cc < 16; cc++) {
      const unsigned b0 = __shfl_sync(0xFFFFFFFFu, bo[0], 2 * cc + half), b1 = __shfl_sync(0xFFFFFFFFu, bo[1], 2 * cc + half);
      int s = 0;
      for (int i = 0; i < 4; i++) s += (int)(signed char)(aw[0] >> (8 * i)) * (int)(signed char)(b0 >> (8 * i)) + (int)(signed char)(aw[1] >> (8 * i)) * (int)(signed char)(b1 >> (8 * i));
      if ((cc >> 3) == h) out[cc & 7] += s;
    }
  }
  for (int i = 0; i < 8; i++) d.x[i] = out[i];
}
static __device__ __forceinline__ void store_matrix_sync(int *p, const fragment<accumulator, 16, 16, 16, int, void> &f, unsigned ld, layout_t) {
  const int l = threadIdx.x & 31;
  for (int i = 0; i < 8; i++) p[(size_t)(l >> 1) * ld + (l & 1) * 8 + i] = f.x[i];
}
}}
