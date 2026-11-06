// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
// The field p = 2^64 - 2^32 + 1 and its cubic extension E = F[X]/(X^3 - 2), for Metal: the same steps
// as blocks/field/gl64.h and blocks/field/field.h, so the same canonical words.
#pragma once
constant constexpr ulong GL_P = 0xFFFFFFFF00000001ul;
constant constexpr ulong GL_EPS = 0xFFFFFFFFul;
inline ulong gl_canonical(ulong x) { return x >= GL_P ? x - GL_P : x; }
inline ulong gl_add(ulong a, ulong b) { const ulong nb = GL_P - b, d = a - nb; return a < nb ? d - GL_EPS : d; }
inline ulong gl_sub(ulong a, ulong b) { const ulong d = a - b; return a < b ? d - GL_EPS : d; }
inline ulong gl_neg(ulong a) { return a ? GL_P - a : 0ul; }
inline ulong gl_reduce128(ulong hi, ulong lo) {
  const ulong hh = hi >> 32, hl = hi & GL_EPS;
  ulong t = lo - hh;
  t -= GL_EPS & (0ul - (ulong)(lo < hh));
  const ulong m = (hl << 32) - hl;
  ulong r = t + m;
  r += GL_EPS & (0ul - (ulong)(r < m));
  return gl_canonical(r);
}
inline ulong gl_mul(ulong a, ulong b) { return gl_reduce128(mulhi(a, b), a * b); }
inline ulong gl_dot3(ulong a0, ulong b0, ulong a1, ulong b1, ulong a2, ulong b2) {
  ulong hi = mulhi(a0, b0), lo = a0 * b0, c = 0;
  ulong h = mulhi(a1, b1), l = a1 * b1;
  lo += l; h += (ulong)(lo < l); hi += h; c += (ulong)(hi < h);
  h = mulhi(a2, b2); l = a2 * b2;
  lo += l; h += (ulong)(lo < l); hi += h; c += (ulong)(hi < h);
  return gl_sub(gl_reduce128(hi, lo), c << 32);
}
struct GlE { ulong c0, c1, c2; };
inline GlE e_add(GlE a, GlE b) { return {gl_add(a.c0, b.c0), gl_add(a.c1, b.c1), gl_add(a.c2, b.c2)}; }
inline GlE e_sub(GlE a, GlE b) { return {gl_sub(a.c0, b.c0), gl_sub(a.c1, b.c1), gl_sub(a.c2, b.c2)}; }
inline GlE e_scale(GlE a, ulong s) { return {gl_mul(a.c0, s), gl_mul(a.c1, s), gl_mul(a.c2, s)}; }
inline GlE e_mul(GlE a, GlE b) { // (a0 + a1 X + a2 X^2)(b0 + b1 X + b2 X^2), X^3 = 2
  const ulong t1 = gl_add(a.c1, a.c1), t2 = gl_add(a.c2, a.c2);
  return {gl_dot3(a.c0, b.c0, t1, b.c2, t2, b.c1), gl_dot3(a.c0, b.c1, a.c1, b.c0, t2, b.c2), gl_dot3(a.c0, b.c2, a.c1, b.c1, a.c2, b.c0)};
}
