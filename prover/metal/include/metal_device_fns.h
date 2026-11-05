// CUDA's device-side functions, expressed with clang's NVPTX builtins: they reach the PTX, and
// prover/metal/ptx2msl.py then gives each instruction its Metal meaning.
#pragma once
#include <type_traits>
#ifdef __CUDA__
#define DF static __device__ __forceinline__
DF unsigned __umulhi(unsigned a, unsigned b) { return __nvvm_mulhi_ui(a, b); }
DF unsigned long long __umul64hi(unsigned long long a, unsigned long long b) { return __nvvm_mulhi_ull(a, b); }
DF int __mulhi(int a, int b) { return __nvvm_mulhi_i(a, b); }
DF long long __mul64hi(long long a, long long b) { return __nvvm_mulhi_ll(a, b); }
DF int __clz(int a) { int r; asm("clz.b32 %0, %1;" : "=r"(r) : "r"(a)); return r; }
DF int __clzll(long long a) { int r; asm("clz.b64 %0, %1;" : "=r"(r) : "l"(a)); return r; }
DF int __popc(unsigned a) { int r; asm("popc.b32 %0, %1;" : "=r"(r) : "r"(a)); return r; }
DF int __popcll(unsigned long long a) { int r; asm("popc.b64 %0, %1;" : "=r"(r) : "l"(a)); return r; }
DF int __ffs(int a) { return a ? 32 - __clz(a & -a) : 0; }
DF int __ffsll(long long a) { return a ? 64 - __clzll(a & -a) : 0; }
DF unsigned __brev(unsigned a) { unsigned r; asm("brev.b32 %0, %1;" : "=r"(r) : "r"(a)); return r; }
DF unsigned long long __brevll(unsigned long long a) { unsigned long long r; asm("brev.b64 %0, %1;" : "=l"(r) : "l"(a)); return r; }
DF unsigned __byte_perm(unsigned a, unsigned b, unsigned s) { return __nvvm_prmt(a, b, s); }
DF unsigned __funnelshift_l(unsigned lo, unsigned hi, unsigned s) { unsigned long long v = ((unsigned long long)hi << 32) | lo; return (unsigned)((v << (s & 31)) >> 32); }
DF unsigned __funnelshift_r(unsigned lo, unsigned hi, unsigned s) { unsigned long long v = ((unsigned long long)hi << 32) | lo; return (unsigned)(v >> (s & 31)); }
DF void __syncwarp(unsigned m = 0xFFFFFFFFu) { __nvvm_bar_warp_sync(m); }
DF int __syncthreads_or(int p) { return __nvvm_bar0_or(p); }
DF int __syncthreads_and(int p) { return __nvvm_bar0_and(p); }
DF int __syncthreads_count(int p) { return __nvvm_bar0_popc(p); }
DF void __threadfence() { __nvvm_membar_gl(); }
DF void __threadfence_block() { __nvvm_membar_cta(); }
DF void __threadfence_system() { __nvvm_membar_sys(); }
DF void __trap() { __builtin_trap(); }
DF unsigned __ballot_sync(unsigned m, int p) { return __nvvm_vote_ballot_sync(m, p); }
DF int __any_sync(unsigned m, int p) { return __nvvm_vote_any_sync(m, p); }
DF int __all_sync(unsigned m, int p) { return __nvvm_vote_all_sync(m, p); }
DF unsigned __activemask() { return __nvvm_activemask(); }
DF void __nanosleep(unsigned) {}
DF void abort() { __builtin_trap(); }
DF void *memcpy(void *d, const void *s, size_t n) { return __builtin_memcpy(d, s, n); }
DF void *memset(void *d, int c, size_t n) { return __builtin_memset(d, c, n); }
DF unsigned long long __cvta_generic_to_shared(const void *p) { unsigned long long r; asm("cvta.to.shared.u64 %0, %1;" : "=l"(r) : "l"(p)); return r; }
DF long long clock64() { return 0; }
template <class T> DF T __mtl_bitcast_from(const unsigned *w) { T t; __builtin_memcpy(&t, w, sizeof(T)); return t; }
#define MTL_SHFL(NAME, BUILTIN, CLAMP)                                                                            \
  template <class T> DF T NAME(unsigned m, T v, int s, int width = 32) {                                          \
    static_assert(sizeof(T) == 4 || sizeof(T) == 8 || sizeof(T) == 1 || sizeof(T) == 2, "shuffle: 1, 2, 4 or 8 bytes"); \
    const int c = CLAMP;                                                                                          \
    if constexpr (sizeof(T) == 8) {                                                                               \
      unsigned w[2]; __builtin_memcpy(w, &v, 8);                                                                  \
      w[0] = BUILTIN(m, w[0], s, c); w[1] = BUILTIN(m, w[1], s, c);                                               \
      return __mtl_bitcast_from<T>(w);                                                                            \
    } else {                                                                                                      \
      unsigned w = 0; __builtin_memcpy(&w, &v, sizeof(T));                                                        \
      w = BUILTIN(m, w, s, c);                                                                                    \
      return __mtl_bitcast_from<T>(&w);                                                                           \
    }                                                                                                             \
  }
MTL_SHFL(__shfl_sync, __nvvm_shfl_sync_idx_i32, ((32 - width) << 8) | 0x1f)
MTL_SHFL(__shfl_xor_sync, __nvvm_shfl_sync_bfly_i32, ((32 - width) << 8) | 0x1f)
MTL_SHFL(__shfl_down_sync, __nvvm_shfl_sync_down_i32, ((32 - width) << 8) | 0x1f)
MTL_SHFL(__shfl_up_sync, __nvvm_shfl_sync_up_i32, ((32 - width) << 8))
template <class T> DF T __ldg(const T *p) { return *p; }
template <class T> DF T __ldcg(const T *p) { return *p; }
template <class T> DF T __ldcs(const T *p) { return *p; }
template <class T> DF T __ldca(const T *p) { return *p; }
template <class T> DF T __ldcv(const T *p) { return *p; }
template <class T> DF T __ldlu(const T *p) { return *p; }
template <class T> DF void __stcs(T *p, T v) { *p = v; }
template <class T> DF void __stcg(T *p, T v) { *p = v; }
template <class T> DF void __stwt(T *p, T v) { *p = v; }
template <class T> DF void __stwb(T *p, T v) { *p = v; }
// Integer atomics.
DF int atomicAdd(int *p, int v) { return __nvvm_atom_add_gen_i(p, v); }
DF unsigned atomicAdd(unsigned *p, unsigned v) { return (unsigned)__nvvm_atom_add_gen_i((int *)p, (int)v); }
DF unsigned long long atomicAdd(unsigned long long *p, unsigned long long v) { return (unsigned long long)__nvvm_atom_add_gen_ll((long long *)p, (long long)v); }
DF unsigned atomicSub(unsigned *p, unsigned v) { return atomicAdd(p, 0u - v); }
DF int atomicSub(int *p, int v) { return atomicAdd(p, -v); }
DF int atomicExch(int *p, int v) { return __nvvm_atom_xchg_gen_i(p, v); }
DF unsigned atomicExch(unsigned *p, unsigned v) { return (unsigned)__nvvm_atom_xchg_gen_i((int *)p, (int)v); }
DF unsigned long long atomicExch(unsigned long long *p, unsigned long long v) { return (unsigned long long)__nvvm_atom_xchg_gen_ll((long long *)p, (long long)v); }
DF int atomicCAS(int *p, int c, int v) { return __nvvm_atom_cas_gen_i(p, c, v); }
DF unsigned atomicCAS(unsigned *p, unsigned c, unsigned v) { return (unsigned)__nvvm_atom_cas_gen_i((int *)p, (int)c, (int)v); }
DF unsigned long long atomicCAS(unsigned long long *p, unsigned long long c, unsigned long long v) { return (unsigned long long)__nvvm_atom_cas_gen_ll((long long *)p, (long long)c, (long long)v); }
DF int atomicMax(int *p, int v) { return __nvvm_atom_max_gen_i(p, v); }
DF unsigned atomicMax(unsigned *p, unsigned v) { return __nvvm_atom_max_gen_ui(p, v); }
DF unsigned long long atomicMax(unsigned long long *p, unsigned long long v) { return __nvvm_atom_max_gen_ull(p, v); }
DF long long atomicMax(long long *p, long long v) { return __nvvm_atom_max_gen_ll(p, v); }
DF int atomicMin(int *p, int v) { return __nvvm_atom_min_gen_i(p, v); }
DF unsigned atomicMin(unsigned *p, unsigned v) { return __nvvm_atom_min_gen_ui(p, v); }
DF unsigned long long atomicMin(unsigned long long *p, unsigned long long v) { return __nvvm_atom_min_gen_ull(p, v); }
DF long long atomicMin(long long *p, long long v) { return __nvvm_atom_min_gen_ll(p, v); }
DF int atomicOr(int *p, int v) { return __nvvm_atom_or_gen_i(p, v); }
DF unsigned atomicOr(unsigned *p, unsigned v) { return (unsigned)__nvvm_atom_or_gen_i((int *)p, (int)v); }
DF unsigned long long atomicOr(unsigned long long *p, unsigned long long v) { return (unsigned long long)__nvvm_atom_or_gen_ll((long long *)p, (long long)v); }
DF int atomicAnd(int *p, int v) { return __nvvm_atom_and_gen_i(p, v); }
DF unsigned atomicAnd(unsigned *p, unsigned v) { return (unsigned)__nvvm_atom_and_gen_i((int *)p, (int)v); }
DF unsigned atomicXor(unsigned *p, unsigned v) { return (unsigned)__nvvm_atom_xor_gen_i((int *)p, (int)v); }
// Floating point with a named rounding.
DF float __fmul_rn(float a, float b) { return __nvvm_mul_rn_f(a, b); }
DF float __fadd_rn(float a, float b) { return __nvvm_add_rn_f(a, b); }
DF float __fsub_rn(float a, float b) { return __nvvm_add_rn_f(a, -b); }
DF int __float2int_rn(float a) { return __nvvm_f2i_rn(a); }
DF unsigned __float_as_uint(float a) { unsigned u; __builtin_memcpy(&u, &a, 4); return u; }
DF float __uint_as_float(unsigned a) { float u; __builtin_memcpy(&u, &a, 4); return u; }
DF int __float_as_int(float a) { int u; __builtin_memcpy(&u, &a, 4); return u; }
DF float __int_as_float(int a) { float u; __builtin_memcpy(&u, &a, 4); return u; }
DF double __dadd_ru(double a, double b) { return __nvvm_add_rp_d(a, b); }
DF double __dmul_ru(double a, double b) { return __nvvm_mul_rp_d(a, b); }
DF double __ddiv_ru(double a, double b) { return __nvvm_div_rp_d(a, b); }
DF double __dsqrt_ru(double a) { return __nvvm_sqrt_rp_d(a); }
DF double __ll2double_ru(long long a) { return __nvvm_ll2d_rp(a); }
DF long long __double2ll_rn(double a) { return __nvvm_d2ll_rn(a); }
DF int abs(int a) { return a < 0 ? -a : a; }
DF double fmin(double a, double b) { return a != a ? b : b != b ? a : (a < b ? a : b); }
DF double fmax(double a, double b) { return a != a ? b : b != b ? a : (a > b ? a : b); }
DF double rint(double a) { double r; asm("cvt.rni.f64.f64 %0, %1;" : "=d"(r) : "d"(a)); return r; }
DF double atomicAdd(double *p, double v) {
  unsigned long long *q = (unsigned long long *)p, old = *q, was;
  do {
    was = old;
    double x; __builtin_memcpy(&x, &was, 8); x += v;
    unsigned long long w; __builtin_memcpy(&w, &x, 8);
    old = atomicCAS(q, was, w);
  } while (old != was);
  double r; __builtin_memcpy(&r, &old, 8); return r;
}
// libm on the device. Each is exact: none rounds where the C function does not.
DF float fabsf(float a) { return __uint_as_float(__float_as_uint(a) & 0x7FFFFFFFu); }
DF double fabs(double a) { unsigned long long u; __builtin_memcpy(&u, &a, 8); u &= ~(1ull << 63); __builtin_memcpy(&a, &u, 8); return a; }
DF long long llabs(long long a) { return a < 0 ? -a : a; }
DF bool isfinite(float a) { return (__float_as_uint(a) & 0x7F800000u) != 0x7F800000u; }
DF bool isfinite(double a) { unsigned long long u; __builtin_memcpy(&u, &a, 8); return ((u >> 52) & 0x7FF) != 0x7FF; }
DF double sqrt(double a) { return __nvvm_sqrt_rn_d(a); }
DF double floor(double a) { return __nvvm_floor_d(a); }
DF double fmod(double a, double b) { // only as fold() calls it: a whole number below 2^53 in size, b = 2
  const double h = floor(a / b); return a - h * b; }
DF float ldexpf(float a, int e) { // for the arguments of the weights' rules: steps by exact powers of two
  while (e > 100) { a = __fmul_rn(a, 1.2676506e30f); e -= 100; }
  while (e < -100) { a = __fmul_rn(a, 7.888609e-31f); e += 100; }
  const float s = e >= -126 ? __uint_as_float((unsigned)(e + 127) << 23) : __uint_as_float(1u << (e + 149));
  return __fmul_rn(a, s); }
DF double ldexp(double a, int e) {
  while (e > 1000) { a *= 1.0715086071862673e301; e -= 1000; }
  while (e < -1000) { a *= 9.332636185032189e-302; e += 1000; }
  unsigned long long u = (unsigned long long)(e + 1023) << 52; double s; __builtin_memcpy(&s, &u, 8);
  return a * s; }
#undef DF
#endif
// min and max, which CUDA declares for host and device.
#ifdef __CUDA__
#define MM(T) static __host__ __device__ __forceinline__ T min(T a, T b) { return a < b ? a : b; } \
              static __host__ __device__ __forceinline__ T max(T a, T b) { return a < b ? b : a; }
MM(int) MM(unsigned) MM(long long) MM(unsigned long long) MM(long) MM(unsigned long) MM(float) MM(double)
#undef MM
#endif
