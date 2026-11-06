// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
// The witness of a feed-forward segment: the same steps as ff_build() and norm_row() (seg_ff_build.h,
// seg_ff_witness.h), which are the reference. A cell and a key are written by one thread, once, so the
// kernels of a segment need no order among them, except that the norm's rows come first.
#pragma once
struct U128 { ulong lo, hi; };
inline U128 u_of(ulong x) { return U128{x, 0}; }
inline U128 s_of(long x) { return U128{(ulong)x, x < 0 ? ~0ul : 0ul}; }
inline U128 u_add(U128 a, U128 b) { const ulong l = a.lo + b.lo; return U128{l, a.hi + b.hi + (ulong)(l < a.lo)}; }
inline U128 u_sub(U128 a, U128 b) { return U128{a.lo - b.lo, a.hi - b.hi - (ulong)(a.lo < b.lo)}; }
inline U128 u_mul(ulong a, ulong b) { return U128{a * b, mulhi(a, b)}; }
inline U128 u_shl(U128 a, uint n) {
  if (n == 0) return a;
  if (n >= 64) return U128{0, a.lo << (n - 64)};
  return U128{a.lo << n, (a.hi << n) | (a.lo >> (64 - n))};
}
inline bool u_le(U128 a, U128 b) { return a.hi < b.hi || (a.hi == b.hi && a.lo <= b.lo); }
// x / d, the quotient below 2^64.
inline ulong u_div(U128 x, ulong d) {
  ulong q = 0, r = 0;
  for (int b = 127; b >= 0; b--) {
    const ulong bit = b >= 64 ? (x.hi >> (b - 64)) & 1 : (x.lo >> b) & 1;
    r = (r << 1) | bit;
    if (r >= d) { r -= d; if (b < 64) q |= 1ul << b; }
  }
  return q;
}
struct WCtx {
  device int *m[WM_COUNT];
  device ulong *keys;
  device uchar *pad;
  const device uint *pos;
  const device ulong *kd, *md, *td;
  ulong T;
};
#define W_CONTEXT WCtx W; W.m[0] = (device int *)c0; W.m[1] = (device int *)c1; W.m[2] = (device int *)c2; W.m[3] = (device int *)c3; W.m[4] = (device int *)c4; W.m[5] = (device int *)c5; \
  W.keys = (device ulong *)keys; W.pad = (device uchar *)padf; W.pos = (const device uint *)pos; W.kd = kd; W.md = md; W.td = td; W.T = p[WP_T];
inline void w_cell(thread const WCtx &W, ulong mat, ulong ft, ulong t, int v) {
  const ulong k = W.md[3 * mat], fr = W.md[3 * mat + 1], rows = W.md[3 * mat + 2], kf = k * fr;
  W.m[mat][(ft / kf) * rows * k + ((ft % fr) * W.T + t) * k + (ft / fr) % k] = v;
}
inline void w_put(thread const WCtx &W, uint id, ulong ix, ulong j, ulong t, long v) {
  const ulong off = W.td[5 * id], nl = W.td[5 * id + 1], cnt = W.td[5 * id + 2], J = W.td[5 * id + 3], mat = W.td[5 * id + 4];
  for (ulong lb = 0; lb < nl; lb++) {
    int c;
    if (lb + 1 == nl) c = (int)v;
    else { const long x = ((v + 128) & 255) - 128; c = (int)x; v = (v - x) >> 8; }
    w_cell(W, mat, off + (lb * cnt + ix) * J + j, t, c);
  }
}
inline void w_key(thread const WCtx &W, uint seg, ulong lam, ulong x, ulong t, ulong key) {
  const ulong idx = W.kd[3 * seg] + (((lam << W.kd[3 * seg + 1]) + W.pos[W.kd[3 * seg + 2] + x]) * W.T + t);
  W.keys[idx] = key, W.pad[idx] = 0;
}
inline uint w_lift(int k) { return k < 16 ? 16 - k : 0; }
inline long w_ofs(int h) { return h == 1 ? 128 : h == 2 ? 32896 : 0; }
// In 64 bits: the remainder is small, and the keys are taken modulo 2^64.
inline long w_rem(long A, long Q, int k) { return A + (1l << (k - 1)) - (long)((ulong)Q << k); }
inline long w_RH(long A, long Q, int k, int h) { return h ? (w_rem(A, Q, k) >> 16) - w_ofs(h) : 0; }
inline ulong w_SK(long A, long Q, int k, int h, long rh) { return (ulong)w_rem(A, Q, k) * (1ul << w_lift(k)) - (h ? (ulong)(rh + w_ofs(h)) << 16 : 0ul); }
inline ulong w_HK(long rh, int k, int h) { const int top = h == 1 ? 24 : 32; return (ulong)(rh + w_ofs(h)) << (top - max(k, 16)); }
inline void w_digits(thread int *D, int at, int n, ulong x) { for (int j = 0; j < n; j++) D[at + j] = (int)(x & 255), x >>= 8; }
inline ulong w_row_mul(int j) { return j == 9 ? 4 : j == 58 ? 2 : j == 59 ? 128 : j == 63 ? 2 : j == 76 ? 2 : j == 79 ? 32 : 1; }
// norm_row() at one stream row; five limbs.
inline void w_norm_row(const device long *x, ulong d, long eps, uint M, device long *o) {
  long Wq[9];
  for (int s = 0; s < 9; s++) Wq[s] = 0;
  for (ulong j = 0; j < d; j++) {
    long lim[5], v = x[j];
    for (int l = 0; l < 5; l++) {
      if (l == 4) { lim[l] = (long)(int)v; break; }
      const long c = ((v + 128) & 255) - 128;
      lim[l] = c, v = (v - c) >> 8;
    }
    for (int p = 0; p < 5; p++) for (int q = 0; q < 5; q++) Wq[p + q] += lim[p] * lim[q];
  }
  U128 V = u_of(0);
  for (int s = 8; s >= 0; s--) V = u_add(u_shl(V, 8), s_of(Wq[s]));
  const U128 Ve = u_add(V, s_of(eps));
  ulong sig = 0;
  for (int b = 39; b >= 0; b--) { const ulong c = sig | (1ul << b); if (u_le(u_mul(c, c), Ve)) sig = c; }
  const U128 top = u_add(u_shl(u_of(1), M), u_of(sig >> 1));
  const long R = (long)u_div(top, sig);
  long c = 0, cs[10]; int vb[10];
  cs[0] = 0;
  for (int s = 0; s < 9; s++) { const long t = Wq[s] + c; vb[s] = (int)(t & 255); c = (t - vb[s]) >> 8; cs[s + 1] = c; }
  int D[99];
  for (int j = 0; j < 99; j++) D[j] = 0;
  const long a = (long)sig & ((1l << 24) - 1), b = (long)(sig >> 24), Rlo = R & ((1l << 24) - 1), Rhi = R >> 24;
  const long l1 = (long)u_sub(Ve, u_mul(sig, sig)).lo, h1 = 2 * (long)sig - l1;
  for (int s = 0; s < 9; s++) D[s] = vb[s];
  D[9] = (int)cs[9];
  for (int s = 1; s < 9; s++) w_digits(D, 10 + 6 * (s - 1), 6, (ulong)(cs[s] + (1l << 47)));
  D[59] = (int)(a & 1), D[58] = (int)((a & 255) >> 1), D[60] = (int)((a >> 8) & 255), D[61] = (int)((a >> 16) & 255);
  D[62] = (int)(b & 255), D[63] = (int)(b >> 8);
  w_digits(D, 64, 5, (ulong)l1), w_digits(D, 69, 5, (ulong)h1);
  const long ab = a * b, Blo = ab & ((1l << 23) - 1), Bhi = ab >> 23;
  w_digits(D, 74, 3, (ulong)Blo), w_digits(D, 77, 2, (ulong)Bhi);
  long Vlo = 0;
  for (int j = 5; j >= 0; j--) Vlo = Vlo * 256 + D[j];
  const long rest = a * a + (Blo << 25) + l1 - Vlo - eps;
  const long kap = rest < 0 ? -((-rest) >> 48) : rest >> 48;
  D[79] = (int)(kap + 4);
  w_digits(D, 80, 3, (ulong)Rlo), w_digits(D, 83, 3, (ulong)Rhi);
  w_digits(D, 86, 3, (ulong)(Rlo * b + Rhi * a));
  const long rr = (long)u_sub(top, u_mul((ulong)R, sig)).lo, rq = (long)sig - 1 - rr;
  w_digits(D, 89, 5, (ulong)rr), w_digits(D, 94, 5, (ulong)rq);
  o[WN_A] = a, o[WN_B] = b, o[WN_SIG] = (long)sig, o[WN_RLO] = Rlo, o[WN_RHI] = Rhi, o[WN_R] = R;
  for (int j = 0; j < 99; j++) o[WN_D + j] = D[j];
  for (int s = 0; s < 9; s++) o[WN_W + s] = Wq[s];
}
