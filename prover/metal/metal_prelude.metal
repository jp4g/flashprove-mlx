// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
// What every translated kernel starts with: the instructions of PTX that are more than one C
// expression. See ptx2msl.py. All of it is exact integer arithmetic, or IEEE arithmetic with the
// rounding that PTX names.
#include <metal_stdlib>
using namespace metal;
#pragma clang diagnostic ignored "-Weverything"
#pragma METAL fp math_mode(safe)
#pragma METAL fp contract(off)
#define SH_TAG (1ul << 60)
#define LOC_TAG (2ul << 60)
struct LaunchInfo {
  uint gx, gy, gz, bx, by, bz, dyn, pad;
  ulong G, trapw, gsync, locks, scratch, stride;
};
#ifdef SMEM_DEVICE
#define SMEM_SPACE device
#define SMEM_PARAM
#define SMEM_ARG
#else
#define SMEM_SPACE threadgroup
#define SMEM_PARAM , threadgroup uchar* sm_ [[threadgroup(0)]]
#define SMEM_ARG , sm_
#endif
struct Ctx {
  uint tx, ty, tz, bx, by, bz, cx, cy, cz, gx, gy, gz, lane, tidx, dyn;
  ulong G, trapw, gsync, locks;
  thread uchar* stk;
  SMEM_SPACE uchar* sm;
};
static inline void ctx_init(thread Ctx& c, constant LaunchInfo& li, uint tidx, uint3 gid, uint lane, thread uchar* stk
#ifndef SMEM_DEVICE
  , threadgroup uchar* sm
#endif
) {
  c.bx = li.bx; c.by = li.by; c.bz = li.bz; c.gx = li.gx; c.gy = li.gy; c.gz = li.gz;
  c.tx = tidx % li.bx; c.ty = (tidx / li.bx) % li.by; c.tz = tidx / (li.bx * li.by);
  c.cx = gid.x; c.cy = gid.y; c.cz = gid.z; c.lane = lane; c.tidx = tidx; c.dyn = li.dyn;
  c.G = li.G; c.trapw = li.trapw; c.gsync = li.gsync; c.locks = li.locks; c.stk = stk;
#ifdef SMEM_DEVICE
  c.sm = reinterpret_cast<device uchar*>(li.scratch + li.stride * ((ulong)gid.x + (ulong)li.gx * ((ulong)gid.y + (ulong)li.gy * gid.z)));
#else
  c.sm = sm;
#endif
}
#define SMEM(v, t, a) (*reinterpret_cast<SMEM_SPACE v t*>(c.sm + (uint)(a)))
#ifdef SMEM_DEVICE
#define BARRIER() threadgroup_barrier(mem_flags::mem_device | mem_flags::mem_threadgroup)
#else
#define BARRIER() threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device)
#endif
#define TRAP() do { *reinterpret_cast<device volatile uint*>(c.trapw) = 1u; return; } while (0)
template <class T> static inline T gld(thread Ctx& c, ulong a) {
  const ulong tag = a >> 60;
  if (tag == 1) return *reinterpret_cast<SMEM_SPACE T*>(c.sm + (uint)a);
  if (tag == 2) return *reinterpret_cast<thread T*>(c.stk + (uint)a);
  return *reinterpret_cast<device T*>(a);
}
template <class T> static inline void gst(thread Ctx& c, ulong a, T v) {
  const ulong tag = a >> 60;
  if (tag == 1) *reinterpret_cast<SMEM_SPACE T*>(c.sm + (uint)a) = v;
  else if (tag == 2) *reinterpret_cast<thread T*>(c.stk + (uint)a) = v;
  else *reinterpret_cast<device T*>(a) = v;
}
static inline uint shf_l(uint lo, uint hi, uint n, int wrap) {
  n = wrap ? (n & 31u) : min(n, 32u);
  if (n == 0) return hi;
  if (n == 32) return lo;
  return (hi << n) | (lo >> (32 - n));
}
static inline uint shf_r(uint lo, uint hi, uint n, int wrap) {
  n = wrap ? (n & 31u) : min(n, 32u);
  if (n == 0) return lo;
  if (n == 32) return hi;
  return (lo >> n) | (hi << (32 - n));
}
static inline uint prmt(uint a, uint b, uint s) {
  uint r = 0;
  for (uint i = 0; i < 4; i++) {
    const uint sel = (s >> (4 * i)) & 15u, k = sel & 7u;
    uint byte = ((k < 4 ? a >> (8 * k) : b >> (8 * (k - 4))) & 255u);
    if (sel & 8u) byte = (byte & 128u) ? 255u : 0u;
    r |= byte << (8 * i);
  }
  return r;
}
static inline uint bfe_u32(uint a, uint p, uint l) { p &= 255u; l &= 255u; if (l == 0 || p >= 32) return 0; const uint m = l >= 32 ? ~0u : (1u << l) - 1u; return (a >> p) & m; }
static inline uint bfe_s32(uint a, uint p, uint l) {
  p &= 255u; l &= 255u; if (l == 0) return 0;
  const bool sg = (a >> min(p + l - 1, 31u)) & 1u;
  if (p >= 32) return sg ? ~0u : 0u;
  uint v = a >> p; const uint w = min(l, 32 - p);
  if (w < 32) v = (v & ((1u << w) - 1u)) | (sg ? ~0u << w : 0u);
  return v;
}
static inline ulong bfe_u64(ulong a, uint p, uint l) { p &= 255u; l &= 255u; if (l == 0 || p >= 64) return 0; const ulong m = l >= 64 ? ~0ul : (1ul << l) - 1ul; return (a >> p) & m; }
static inline ulong bfe_s64(ulong a, uint p, uint l) {
  p &= 255u; l &= 255u; if (l == 0) return 0;
  const bool sg = (a >> min(p + l - 1, 63u)) & 1ul;
  if (p >= 64) return sg ? ~0ul : 0ul;
  ulong v = a >> p; const uint w = min(l, 64 - p);
  if (w < 64) v = (v & ((1ul << w) - 1ul)) | (sg ? ~0ul << w : 0ul);
  return v;
}
// cvt.*.f32 to integers: saturating, a NaN gives 0.
static inline int f2i_s32(float x) { if (x != x) return 0; if (x >= 2147483648.0f) return 2147483647; if (x <= -2147483648.0f) return (-2147483647 - 1); return (int)x; }
static inline uint f2i_u32(float x) { if (x != x || x <= 0.0f) return 0u; if (x >= 4294967296.0f) return 4294967295u; return (uint)x; }
static inline long f2i_s64(float x) { if (x != x) return 0; if (x >= 9223372036854775808.0f) return 9223372036854775807l; if (x <= -9223372036854775808.0f) return (-9223372036854775807l - 1); return (long)x; }
static inline ulong f2i_u64(float x) { if (x != x || x <= 0.0f) return 0ul; if (x >= 18446744073709551616.0f) return ~0ul; return (ulong)x; }
static inline short f2i_s16(float x) { return (short)clamp(f2i_s32(x), -32768, 32767); }
static inline ushort f2i_u16(float x) { return (ushort)min(f2i_u32(x), 65535u); }
// shfl.sync, by the PTX ISA's rule: c holds the segment mask and the clamp.
#define SHFL_LANES(c_, maxl, minl) const uint seg_ = ((c_) >> 8) & 31u, cl_ = (c_) & 31u, minl = c.lane & seg_, maxl = minl | (cl_ & ~seg_)
static inline uint shfl_idx(thread Ctx& c, uint a, uint b, uint cc) { SHFL_LANES(cc, mx, mn); uint j = mn | (b & ~seg_ & 31u); if (j > mx) j = c.lane; return simd_shuffle(a, (ushort)j); }
static inline uint shfl_bfly(thread Ctx& c, uint a, uint b, uint cc) { SHFL_LANES(cc, mx, mn); uint j = c.lane ^ b; if (j > mx) j = c.lane; return simd_shuffle(a, (ushort)j); }
static inline uint shfl_down(thread Ctx& c, uint a, uint b, uint cc) { SHFL_LANES(cc, mx, mn); uint j = c.lane + b; if (j > mx) j = c.lane; return simd_shuffle(a, (ushort)j); }
static inline uint shfl_up(thread Ctx& c, uint a, uint b, uint cc) { SHFL_LANES(cc, mx, mn); int j = (int)c.lane - (int)b; if (j < (int)mx) j = (int)c.lane; return simd_shuffle(a, (ushort)j); }
// Atomics. Metal has 32-bit ones in both spaces; a 64-bit one is two halves (add, logic) or goes
// through a lock chosen by the address.
#define AG(T) reinterpret_cast<device T*>
#define AS(T) reinterpret_cast<SMEM_SPACE T*>
#define MO memory_order_relaxed
#define ATOM32(kind, tc, CT, AT, FN)                                                                         \
  static inline uint atom_##kind##_32_##tc##_global(thread Ctx& c, ulong a, uint v) { return (uint)FN(AG(AT)(a), (CT)v, MO); } \
  static inline uint atom_##kind##_32_##tc##_shared(thread Ctx& c, ulong a, uint v) { return (uint)FN(AS(AT)(c.sm + (uint)a), (CT)v, MO); } \
  static inline uint atom_##kind##_32_##tc##_gen(thread Ctx& c, ulong a, uint v) { return (a >> 60) == 1 ? atom_##kind##_32_##tc##_shared(c, a, v) : atom_##kind##_32_##tc##_global(c, a, v); }
#define ATOM32_ALL(kind, FN) ATOM32(kind, b, uint, atomic_uint, FN) ATOM32(kind, u, uint, atomic_uint, FN) ATOM32(kind, s, int, atomic_int, FN)
ATOM32_ALL(add, atomic_fetch_add_explicit)
ATOM32_ALL(or, atomic_fetch_or_explicit)
ATOM32_ALL(and, atomic_fetch_and_explicit)
ATOM32_ALL(xor, atomic_fetch_xor_explicit)
ATOM32_ALL(exch, atomic_exchange_explicit)
ATOM32_ALL(min, atomic_fetch_min_explicit)
ATOM32_ALL(max, atomic_fetch_max_explicit)
static inline uint atom_cas_32_b_global(thread Ctx& c, ulong a, uint e, uint v) {
  uint x = e;
  while (!atomic_compare_exchange_weak_explicit(AG(atomic_uint)(a), &x, v, MO, MO) && x == e) {}
  return x;
}
static inline uint atom_cas_32_b_shared(thread Ctx& c, ulong a, uint e, uint v) {
  uint x = e;
  while (!atomic_compare_exchange_weak_explicit(AS(atomic_uint)(c.sm + (uint)a), &x, v, MO, MO) && x == e) {}
  return x;
}
static inline uint atom_cas_32_b_gen(thread Ctx& c, ulong a, uint e, uint v) { return (a >> 60) == 1 ? atom_cas_32_b_shared(c, a, e, v) : atom_cas_32_b_global(c, a, e, v); }
#define atom_cas_32_u_global atom_cas_32_b_global
#define atom_cas_32_u_shared atom_cas_32_b_shared
#define atom_cas_32_s_global atom_cas_32_b_global
#define atom_cas_32_s_shared atom_cas_32_b_shared
static inline ulong add64_global(ulong a, ulong v) {
  device atomic_uint* p = AG(atomic_uint)(a);
  const uint lo = (uint)v;
  const uint o = atomic_fetch_add_explicit(p, lo, MO);
  const uint carry = (uint)(((ulong)o + lo) >> 32);
  const uint h = atomic_fetch_add_explicit(p + 1, (uint)(v >> 32) + carry, MO);
  return ((ulong)h << 32) | o;
}
static inline ulong atom_add_64_u_global(thread Ctx& c, ulong a, ulong v) { return add64_global(a, v); }
static inline ulong atom_add_64_s_global(thread Ctx& c, ulong a, ulong v) { return add64_global(a, v); }
static inline ulong atom_add_64_b_global(thread Ctx& c, ulong a, ulong v) { return add64_global(a, v); }
static inline ulong atom_add_64_u_shared(thread Ctx& c, ulong a, ulong v) {
  SMEM_SPACE atomic_uint* p = AS(atomic_uint)(c.sm + (uint)a);
  const uint lo = (uint)v;
  const uint o = atomic_fetch_add_explicit(p, lo, MO);
  const uint carry = (uint)(((ulong)o + lo) >> 32);
  const uint h = atomic_fetch_add_explicit(p + 1, (uint)(v >> 32) + carry, MO);
  return ((ulong)h << 32) | o;
}
static inline ulong atom_add_64_u_gen(thread Ctx& c, ulong a, ulong v) { return (a >> 60) == 1 ? atom_add_64_u_shared(c, a, v) : add64_global(a, v); }
#define atom_add_64_s_gen atom_add_64_u_gen
#define atom_add_64_b_gen atom_add_64_u_gen
#define atom_add_64_s_shared atom_add_64_u_shared
#define atom_add_64_b_shared atom_add_64_u_shared
#define BIT64(kind, FN)                                                                                       \
  static inline ulong atom_##kind##_64_b_global(thread Ctx& c, ulong a, ulong v) {                            \
    device atomic_uint* p = AG(atomic_uint)(a);                                                               \
    const uint lo = FN(p, (uint)v, MO), hi = FN(p + 1, (uint)(v >> 32), MO);                                  \
    return ((ulong)hi << 32) | lo; }                                                                          \
  static inline ulong atom_##kind##_64_b_gen(thread Ctx& c, ulong a, ulong v) { return atom_##kind##_64_b_global(c, a, v); }
BIT64(or, atomic_fetch_or_explicit)
BIT64(and, atomic_fetch_and_explicit)
BIT64(xor, atomic_fetch_xor_explicit)
// The locks: one word each, 4096 of them, in the runtime's memory. The lanes of a SIMD group take
// turns: a lane that spun while another lane of its group held the lock could wait for ever, as the
// group's lanes run in step.
#define LOCKED(a, BODY)                                                                                       \
  ulong old_ = 0;                                                                                             \
  device atomic_uint* l_ = AG(atomic_uint)(c.locks) + (((a) >> 3) & 4095ul);                                  \
  const ulong active_ = (ulong)simd_active_threads_mask();                                                    \
  for (uint who_ = 0; who_ < 32; who_++) {                                                                    \
    if (((active_ >> who_) & 1ul) && who_ == c.lane) {                                                        \
      uint e_ = 0;                                                                                            \
      while (!atomic_compare_exchange_weak_explicit(l_, &e_, 1u, MO, MO)) e_ = 0;                             \
      atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);                  \
      device volatile ulong* p_ = reinterpret_cast<device volatile ulong*>(a);                                \
      old_ = *p_; BODY;                                                                                       \
      atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);                  \
      atomic_store_explicit(l_, 0u, MO);                                                                      \
    }                                                                                                         \
  }                                                                                                           \
  return old_;
static inline ulong atom_cas_64_b_global(thread Ctx& c, ulong a, ulong e, ulong v) { LOCKED(a, if (old_ == e) *p_ = v) }
static inline ulong atom_exch_64_b_global(thread Ctx& c, ulong a, ulong v) { LOCKED(a, *p_ = v) }
// Metal's 64-bit min and max return nothing; the value here is a plain read before them. No kernel
// uses it (the translator's tests of the sources: grep atomicMax in prover/).
static inline ulong atom_max_64_u_global(thread Ctx& c, ulong a, ulong v) { const ulong o = *reinterpret_cast<device volatile ulong*>(a); atomic_max_explicit(AG(atomic_ulong)(a), v, MO); return o; }
static inline ulong atom_min_64_u_global(thread Ctx& c, ulong a, ulong v) { const ulong o = *reinterpret_cast<device volatile ulong*>(a); atomic_min_explicit(AG(atomic_ulong)(a), v, MO); return o; }
static inline ulong atom_max_64_s_global(thread Ctx& c, ulong a, ulong v) { LOCKED(a, if ((long)v > (long)old_) *p_ = v) }
static inline ulong atom_min_64_s_global(thread Ctx& c, ulong a, ulong v) { LOCKED(a, if ((long)v < (long)old_) *p_ = v) }
#define atom_cas_64_u_global atom_cas_64_b_global
#define atom_exch_64_u_global atom_exch_64_b_global
#define atom_cas_64_b_gen atom_cas_64_b_global
#define atom_cas_64_u_gen atom_cas_64_b_global
#define atom_exch_64_b_gen atom_exch_64_b_global
#define atom_max_64_u_gen atom_max_64_u_global
#define atom_min_64_u_gen atom_min_64_u_global
#define atom_max_64_s_gen atom_max_64_s_global
#define atom_min_64_s_gen atom_min_64_s_global
#ifdef NEED_BARRED
static inline uint bar_red_popc(thread Ctx& c, bool p) {
  SMEM_SPACE atomic_uint* w = AS(atomic_uint)(c.sm + BARRED_OFF);
  BARRIER();
  if (c.tidx == 0) atomic_store_explicit(w, 0u, MO);
  BARRIER();
  if (p) atomic_fetch_add_explicit(w, 1u, MO);
  BARRIER();
  const uint r = atomic_load_explicit(w, MO);
  BARRIER();
  return r;
}
#endif
#ifdef NEED_GRIDSYNC
// this_grid().sync(): the runtime has kept the grid small enough that its blocks are all resident.
static inline void grid_sync(thread Ctx& c) {
  BARRIER();
  if (c.tidx == 0) {
    device atomic_uint* w = AG(atomic_uint)(c.gsync);
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
    const uint g = atomic_load_explicit(w + 1, MO), all = c.gx * c.gy * c.gz;
    if (atomic_fetch_add_explicit(w, 1u, MO) == all - 1) {
      atomic_store_explicit(w, 0u, MO);
      atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
      atomic_fetch_add_explicit(w + 1, 1u, MO);
    } else {
      while (atomic_load_explicit(w + 1, MO) == g) {}
    }
    atomic_thread_fence(mem_flags::mem_device, memory_order_seq_cst, thread_scope_device);
  }
  BARRIER();
}
#endif
#ifdef NEED_MMA
// mma.sync.aligned.m16n8k32 on bytes, with the PTX ISA's fragments: lane = 4 g + t; a0, a2 hold row
// g, a1, a3 row g + 8; bytes k = 4 t .. and 16 + 4 t ..; b0, b1 column g; d0, d1 row g, columns
// 2 t, 2 t + 1, d2, d3 row g + 8.
template <class TA, class TB> static inline int dot4(uint a, uint b) {
  int s = 0;
  for (uint i = 0; i < 4; i++) s += (int)(TA)(a >> (8 * i)) * (int)(TB)(b >> (8 * i));
  return s;
}
template <class TA, class TB> static inline uint4 mma_bytes(thread Ctx& c, uint4 a, uint2 b, uint4 cc) {
  const uint g = c.lane >> 2, t = c.lane & 3u;
  int4 d = as_type<int4>(cc);
  for (uint q = 0; q < 4; q++) {
    const ushort la = (ushort)(4 * g + q), lb0 = (ushort)(8 * t + q), lb1 = (ushort)(8 * t + 4 + q);
    const uint a0 = simd_shuffle(a.x, la), a1 = simd_shuffle(a.y, la), a2 = simd_shuffle(a.z, la), a3 = simd_shuffle(a.w, la);
    const uint p0 = simd_shuffle(b.x, lb0), p1 = simd_shuffle(b.y, lb0), q0 = simd_shuffle(b.x, lb1), q1 = simd_shuffle(b.y, lb1);
    d.x += dot4<TA, TB>(a0, p0) + dot4<TA, TB>(a2, p1);
    d.y += dot4<TA, TB>(a0, q0) + dot4<TA, TB>(a2, q1);
    d.z += dot4<TA, TB>(a1, p0) + dot4<TA, TB>(a3, p1);
    d.w += dot4<TA, TB>(a1, q0) + dot4<TA, TB>(a3, q1);
  }
  return as_type<uint4>(d);
}
static inline uint4 mma_m16n8k32_s8s8(thread Ctx& c, uint4 a, uint2 b, uint4 d) { return mma_bytes<char, char>(c, a, b, d); }
static inline uint4 mma_m16n8k32_u8s8(thread Ctx& c, uint4 a, uint2 b, uint4 d) { return mma_bytes<uchar, char>(c, a, b, d); }
static inline uint4 mma_m16n8k32_s8u8(thread Ctx& c, uint4 a, uint2 b, uint4 d) { return mma_bytes<char, uchar>(c, a, b, d); }
static inline uint4 mma_m16n8k32_u8u8(thread Ctx& c, uint4 a, uint2 b, uint4 d) { return mma_bytes<uchar, uchar>(c, a, b, d); }
// ldmatrix: lanes 8 i .. 8 i + 7 name the rows of matrix i; a lane receives row lane / 4, bytes
// 4 (lane % 4) .. + 3, of each.
static inline uint ldmatrix_part(thread Ctx& c, uint addr, uint i) {
  const uint a = simd_shuffle(addr, (ushort)(8 * i + (c.lane >> 2)));
  return *reinterpret_cast<SMEM_SPACE uint*>(c.sm + a + 4 * (c.lane & 3u));
}
#endif
#ifdef NEED_F64
// float64 in integers: Metal has no double. The structure is that of J. Hauser's SoftFloat 3
// (BSD licence). mode: 0 nearest even, 1 up, 2 down, 3 toward zero. sig in sf_round: bit 62 is the
// leading one, the low ten bits are rounded off, and any lost bit is kept in bit 0.
#define SF_S(x) ((bool)((x) >> 63))
#define SF_E(x) ((int)(((x) >> 52) & 0x7FF))
#define SF_M(x) ((x) & 0xFFFFFFFFFFFFFul)
#define SF_NAN 0x7FF8000000000000ul
static inline ulong sf_neg(ulong a) { return a ^ (1ul << 63); }
static inline bool sf_isnan(ulong a) { return (a & ~(1ul << 63)) > 0x7FF0000000000000ul; }
static inline ulong sf_jam(ulong s, uint d) { return d < 63 ? (s >> d) | (ulong)((s << ((64 - d) & 63)) != 0 && d != 0) : (ulong)(s != 0); }
static inline ulong sf_round(bool s, int e, ulong sig, int mode) {
  const uint inc = mode == 0 ? 0x200u : ((mode == 1 && !s) || (mode == 2 && s)) ? 0x3FFu : 0u;
  uint rb = (uint)sig & 0x3FFu;
  if ((uint)e >= 0x7FDu) {
    if (e < 0) { sig = sf_jam(sig, (uint)(-e)); e = 0; rb = (uint)sig & 0x3FFu; }
    else if (e > 0x7FD || sig + inc >= (1ul << 63))
      return ((ulong)s << 63) | (inc ? 0x7FF0000000000000ul : 0x7FEFFFFFFFFFFFFFul);
  }
  sig = (sig + inc) >> 10;
  if (mode == 0 && rb == 0x200u) sig &= ~1ul;
  if (!sig) e = 0;
  return ((ulong)s << 63) + ((ulong)e << 52) + sig;
}
static inline ulong sf_norm_round(bool s, int e, ulong sig, int mode) {
  if (!sig) return (ulong)s << 63;
  const int sh = clz(sig) - 1;
  if (sh >= 0) { e -= sh; sig <<= sh; } else { e += 1; sig = (sig >> 1) | (sig & 1ul); }
  return sf_round(s, e, sig, mode);
}
static inline ulong sf_add(ulong a, ulong b, int mode) {
  if (sf_isnan(a) || sf_isnan(b)) return SF_NAN;
  const bool sa = SF_S(a), sb = SF_S(b);
  int ea = SF_E(a), eb = SF_E(b);
  if (ea == 0x7FF) return (eb == 0x7FF && sa != sb) ? SF_NAN : a;
  if (eb == 0x7FF) return b;
  ulong ma = SF_M(a), mb = SF_M(b);
  if (ea) ma |= 1ul << 52; else ea = 1;
  if (eb) mb |= 1ul << 52; else eb = 1;
  ma <<= 9; mb <<= 9; // leading one at bit 61
  bool sr = sa;
  if (ea < eb || (ea == eb && ma < mb)) { const ulong t = ma; ma = mb; mb = t; const int u = ea; ea = eb; eb = u; sr = sb; }
  mb = sf_jam(mb, (uint)(ea - eb));
  const ulong r = (sa == sb) ? ma + mb : ma - mb;
  if (!r) return (sa == sb) ? a : (mode == 2 ? (1ul << 63) : 0ul);
  return sf_norm_round(sr, ea, r, mode);
}
static inline ulong sf_mul(ulong a, ulong b, int mode) {
  if (sf_isnan(a) || sf_isnan(b)) return SF_NAN;
  const bool s = SF_S(a) != SF_S(b);
  int ea = SF_E(a), eb = SF_E(b);
  ulong ma = SF_M(a), mb = SF_M(b);
  if (ea == 0x7FF || eb == 0x7FF) {
    if ((ea == 0 && !ma) || (eb == 0 && !mb)) return SF_NAN;
    return ((ulong)s << 63) | 0x7FF0000000000000ul;
  }
  if (ea) ma |= 1ul << 52; else { if (!ma) return (ulong)s << 63; const int sh = clz(ma) - 11; ma <<= sh; ea = 1 - sh; }
  if (eb) mb |= 1ul << 52; else { if (!mb) return (ulong)s << 63; const int sh = clz(mb) - 11; mb <<= sh; eb = 1 - sh; }
  ma <<= 10; mb <<= 11; // leading ones at 62 and 63
  const ulong hi = mulhi(ma, mb), lo = ma * mb;
  ulong sig = hi | (ulong)(lo != 0);
  int e = ea + eb - 0x3FF;
  if (sig < (1ul << 62)) { sig <<= 1; e -= 1; }
  return sf_round(s, e, sig, mode);
}
static inline ulong sf_div(ulong a, ulong b, int mode) {
  if (sf_isnan(a) || sf_isnan(b)) return SF_NAN;
  const bool s = SF_S(a) != SF_S(b);
  int ea = SF_E(a), eb = SF_E(b);
  ulong ma = SF_M(a), mb = SF_M(b);
  if (ea == 0x7FF) return eb == 0x7FF ? SF_NAN : ((ulong)s << 63) | 0x7FF0000000000000ul;
  if (eb == 0x7FF) return (ulong)s << 63;
  if (eb == 0 && !mb) return (ea == 0 && !ma) ? SF_NAN : ((ulong)s << 63) | 0x7FF0000000000000ul;
  if (ea == 0 && !ma) return (ulong)s << 63;
  if (ea) ma |= 1ul << 52; else { const int sh = clz(ma) - 11; ma <<= sh; ea = 1 - sh; }
  if (eb) mb |= 1ul << 52; else { const int sh = clz(mb) - 11; mb <<= sh; eb = 1 - sh; }
  int e = ea - eb + 0x3FF;
  if (ma < mb) { ma <<= 1; e -= 1; } // now ma / mb is in [1, 2)
  ulong rem = ma, q = 0;
  for (int i = 0; i < 63; i++) { // 63 bits of the quotient: the leading one is at bit 62
    q <<= 1;
    if (rem >= mb) { rem -= mb; q |= 1ul; }
    rem <<= 1;
  }
  return sf_round(s, e - 1, q | (ulong)(rem != 0), mode);
}
static inline ulong sf_sqrt(ulong a, int mode) {
  if (sf_isnan(a)) return SF_NAN;
  const bool s = SF_S(a);
  int e = SF_E(a);
  ulong m = SF_M(a);
  if (!e && !m) return a;
  if (s) return SF_NAN;
  if (e == 0x7FF) return a;
  if (e) m |= 1ul << 52; else { const int sh = clz(m) - 11; m <<= sh; e = 1 - sh; }
  int ex = e - 1075; // the value is m 2^ex
  if (ex & 1) { m <<= 1; ex -= 1; }
  const int sh = clz(m) & ~1; // X = m 2^sh is in [2^62, 2^64), and sh is even
  ulong X = m << sh, root = 0, rem = 0;
  const int ex2 = ex - sh;
  for (int i = 0; i < 56; i++) { // root = floor(sqrt(X) 2^24), which has 56 bits
    const uint two = i < 32 ? (uint)(X >> 62) : 0u;
    X <<= 2;
    rem = (rem << 2) | two;
    const ulong t = (root << 2) | 1ul;
    root <<= 1;
    if (rem >= t) { rem -= t; root |= 1ul; }
  }
  return sf_round(false, 31 + ex2 / 2 + 1023 - 1, (root << 7) | (ulong)(rem != 0), mode);
}
static inline ulong sf_from_u64s(bool s, ulong v, int mode) { return (v >> 63) ? sf_norm_round(s, 1085, (v >> 1) | (v & 1ul), mode) : sf_norm_round(s, 1084, v, mode); }
static inline ulong sf_from_u64(ulong v, int mode) { return sf_from_u64s(false, v, mode); }
static inline ulong sf_from_i64(long v, int mode) { const bool s = v < 0; return sf_from_u64s(s, s ? 0ul - (ulong)v : (ulong)v, mode); }
static inline ulong sf_from_f32(uint f) {
  const ulong s = (ulong)(f >> 31) << 63; const int e = (f >> 23) & 255; uint m = f & 0x7FFFFFu;
  if (e == 255) return s | 0x7FF0000000000000ul | ((ulong)m << 29);
  if (e == 0) { if (!m) return s; const int sh = clz(m) - 8; m = (m << sh) & 0x7FFFFFu; return s | ((ulong)(897 - sh) << 52) | ((ulong)m << 29); }
  return s | ((ulong)(e + 896) << 52) | ((ulong)m << 29);
}
static inline uint sf_to_f32(ulong a, int mode) {
  const uint s = (uint)(a >> 63) << 31; const int e = SF_E(a); ulong m = SF_M(a);
  if (e == 0x7FF) return s | 0x7F800000u | (m ? 0x400000u : 0u);
  if (!e && !m) return s;
  int ee = e - 896; // of a float with leading one at bit 52
  m |= 1ul << 52;
  uint shift = 29;
  if (ee <= 0) { shift += (uint)(1 - ee); ee = 0; }
  if (shift > 60) shift = 60;
  const ulong kept = m >> shift, lost = m & ((1ul << shift) - 1ul), hf = 1ul << (shift - 1);
  ulong r = kept;
  const bool sg = s != 0;
  if (mode == 0) { if (lost > hf || (lost == hf && (kept & 1ul))) r++; }
  else if ((mode == 1 && !sg) || (mode == 2 && sg)) { if (lost) r++; }
  ulong out = (ee ? ((ulong)(ee - 1) << 23) : 0ul) + r;
  if (out >= 0x7F800000ul) return s | 0x7F800000u;
  return s | (uint)out;
}
static inline ulong sf_trunc(ulong a) {
  const int e = SF_E(a) - 1023;
  if (e >= 52) return a;
  if (e < 0) return a & (1ul << 63);
  return a & ~((1ul << (52 - e)) - 1ul);
}
static inline bool sf_lt(ulong a, ulong b) {
  if (sf_isnan(a) || sf_isnan(b)) return false;
  const bool sa = SF_S(a), sb = SF_S(b);
  if (sa != sb) return sa && ((a | b) << 1) != 0;
  return a != b && (sa != (a < b));
}
static inline bool sf_eq(ulong a, ulong b) { if (sf_isnan(a) || sf_isnan(b)) return false; return a == b || ((a | b) << 1) == 0; }
static inline ulong sf_floor(ulong a) { const ulong t = sf_trunc(a); return (SF_S(a) && !sf_eq(t, a)) ? sf_add(t, 0xBFF0000000000000ul, 0) : t; }
static inline ulong sf_ceil(ulong a) { const ulong t = sf_trunc(a); return (!SF_S(a) && !sf_eq(t, a)) ? sf_add(t, 0x3FF0000000000000ul, 0) : t; }
static inline ulong sf_rint(ulong a) { // add and subtract 2^52, by the sign
  const int e = SF_E(a) - 1023;
  if (e >= 52) return a;
  const ulong big = SF_S(a) ? 0xC330000000000000ul : 0x4330000000000000ul;
  return sf_add(sf_add(a, big, 0), sf_neg(big), 0) | (a & (1ul << 63));
}
static inline ulong sf_to_u64(ulong a, int mode) {
  if (sf_isnan(a)) return 0;
  const ulong r = mode == 0 ? sf_rint(a) : mode == 2 ? sf_floor(a) : mode == 1 ? sf_ceil(a) : sf_trunc(a);
  if (SF_S(r)) return 0;
  const int e = SF_E(r) - 1023;
  if (e < 0) return 0;
  if (e >= 64) return ~0ul;
  const ulong m = SF_M(r) | (1ul << 52);
  return e >= 52 ? m << (e - 52) : m >> (52 - e);
}
static inline long sf_to_i64(ulong a, int mode) {
  if (sf_isnan(a)) return 0;
  const ulong r = mode == 0 ? sf_rint(a) : mode == 2 ? sf_floor(a) : mode == 1 ? sf_ceil(a) : sf_trunc(a);
  const int e = SF_E(r) - 1023;
  if (e < 0) return 0;
  if (e >= 63) return SF_S(r) ? (-9223372036854775807l - 1) : 9223372036854775807l;
  const ulong m = SF_M(r) | (1ul << 52), u = e >= 52 ? m << (e - 52) : m >> (52 - e);
  return SF_S(r) ? -(long)u : (long)u;
}
static inline bool sf_cmp_eq(ulong a, ulong b) { return sf_eq(a, b); }
static inline bool sf_cmp_ne(ulong a, ulong b) { return !sf_isnan(a) && !sf_isnan(b) && !sf_eq(a, b); }
static inline bool sf_cmp_lt(ulong a, ulong b) { return sf_lt(a, b); }
static inline bool sf_cmp_le(ulong a, ulong b) { return sf_lt(a, b) || sf_eq(a, b); }
static inline bool sf_cmp_gt(ulong a, ulong b) { return sf_lt(b, a); }
static inline bool sf_cmp_ge(ulong a, ulong b) { return sf_lt(b, a) || sf_eq(a, b); }
static inline bool sf_cmp_equ(ulong a, ulong b) { return sf_isnan(a) || sf_isnan(b) || sf_eq(a, b); }
static inline bool sf_cmp_neu(ulong a, ulong b) { return !sf_eq(a, b); }
static inline bool sf_cmp_ltu(ulong a, ulong b) { return !sf_cmp_ge(a, b); }
static inline bool sf_cmp_leu(ulong a, ulong b) { return !sf_lt(b, a); }
static inline bool sf_cmp_gtu(ulong a, ulong b) { return !sf_cmp_le(a, b); }
static inline bool sf_cmp_geu(ulong a, ulong b) { return !sf_lt(a, b); }
static inline bool sf_cmp_num(ulong a, ulong b) { return !sf_isnan(a) && !sf_isnan(b); }
static inline bool sf_cmp_nan(ulong a, ulong b) { return sf_isnan(a) || sf_isnan(b); }
#endif
