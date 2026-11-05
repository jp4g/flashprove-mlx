// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
// The runtime's own kernels: fill, copy, and the exact signed-byte matrix product behind
// cublasGemmEx. C[j ldc + i] = alpha sum_p A[i lda + p] B[j ldb + p] + beta C[..], in int32 (wrapping,
// as the sums in the sources stay in range).
#include <metal_stdlib>
using namespace metal;
struct FillArgs { ulong d, n; uint v, pad; };
kernel void mtl_fill(constant FillArgs& a [[buffer(0)]], uint id [[thread_position_in_grid]], uint n [[threads_per_grid]]) {
  device uchar* d = reinterpret_cast<device uchar*>(a.d);
  const ulong head = min((16ul - (a.d & 15ul)) & 15ul, a.n);
  const ulong words = (a.n - head) / 16;
  device uint4* w = reinterpret_cast<device uint4*>(a.d + head);
  for (ulong i = id; i < words; i += n) w[i] = uint4(a.v);
  if (id == 0) {
    for (ulong i = 0; i < head; i++) d[i] = (uchar)a.v;
    for (ulong i = head + words * 16; i < a.n; i++) d[i] = (uchar)a.v;
  }
}
struct CopyArgs { ulong d, s, n; };
kernel void mtl_copy(constant CopyArgs& a [[buffer(0)]], uint id [[thread_position_in_grid]], uint n [[threads_per_grid]]) {
  device uchar* d = reinterpret_cast<device uchar*>(a.d);
  device const uchar* s = reinterpret_cast<device const uchar*>(a.s);
  if (((a.d | a.s) & 15ul) == 0) {
    const ulong words = a.n / 16;
    device uint4* dw = reinterpret_cast<device uint4*>(a.d);
    device const uint4* sw = reinterpret_cast<device const uint4*>(a.s);
    for (ulong i = id; i < words; i += n) dw[i] = sw[i];
    for (ulong i = words * 16 + id; i < a.n; i += n) d[i] = s[i];
  } else {
    for (ulong i = id; i < a.n; i += n) d[i] = s[i];
  }
}
struct GemmArgs { ulong a, b, c; long sa, sb, sc; int m, n, k, lda, ldb, ldc, alpha, beta; };
kernel void mtl_gemm_tn_i8(constant GemmArgs& g [[buffer(0)]], uint3 tg [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]]) {
  threadgroup char As[64][68], Bs[64][68];
  device const char* A = reinterpret_cast<device const char*>(g.a + (ulong)(g.sa * tg.z));
  device const char* B = reinterpret_cast<device const char*>(g.b + (ulong)(g.sb * tg.z));
  device int* C = reinterpret_cast<device int*>(g.c + (ulong)(g.sc * tg.z * 4));
  const int i0 = (int)tg.x * 64, j0 = (int)tg.y * 64, ti = (int)(t & 15u) * 4, tj = (int)(t >> 4) * 4;
  int acc[4][4] = {};
  for (int p0 = 0; p0 < g.k; p0 += 64) {
    for (int e = (int)t; e < 64 * 64; e += 256) {
      const int r = e >> 6, q = e & 63, p = p0 + q;
      As[r][q] = (i0 + r < g.m && p < g.k) ? A[(long)(i0 + r) * g.lda + p] : (char)0;
      Bs[r][q] = (j0 + r < g.n && p < g.k) ? B[(long)(j0 + r) * g.ldb + p] : (char)0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int q = 0; q < 64; q++) {
      int a[4], b[4];
      for (int x = 0; x < 4; x++) { a[x] = As[ti + x][q]; b[x] = Bs[tj + x][q]; }
      for (int x = 0; x < 4; x++)
        for (int y = 0; y < 4; y++) acc[x][y] += a[x] * b[y];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  for (int x = 0; x < 4; x++)
    for (int y = 0; y < 4; y++) {
      const int i = i0 + ti + x, j = j0 + tj + y;
      if (i < g.m && j < g.n) {
        device int* c = C + (long)j * g.ldc + i;
        *c = g.alpha * acc[x][y] + (g.beta ? g.beta * *c : 0);
      }
    }
}
