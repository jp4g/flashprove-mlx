// Float64 on the GPU (Metal has none: ptx2msl.py lowers each operation to integers) against the CPU's
// IEEE arithmetic, at the rounding modes that the prover's kernels name. Exit 0: every bit agrees.
#include <cfenv>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <random>
#include <vector>
__global__ void k_ops(const double *a, const double *b, double *o, long long *n, int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const double x = a[i], y = b[i];
  double *r = o + (size_t)i * 12;
  r[0] = x + y; r[1] = x - y; r[2] = x * y; r[3] = x / y; r[4] = sqrt(fabs(x));
  r[5] = __dadd_ru(x, y); r[6] = __dmul_ru(x, y); r[7] = __ddiv_ru(x, y); r[8] = __dsqrt_ru(fabs(x));
  r[9] = rint(x); r[10] = floor(x); r[11] = __ll2double_ru((long long)(x));
  n[i] = (x < y) | ((x == y) << 1) | ((x >= y) << 2) | ((x != x) << 3);
}
__global__ void k_conv(const long long *a, const unsigned long long *b, const float *f, double *o, unsigned *of, long long *ol, int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  o[3 * i] = (double)a[i]; o[3 * i + 1] = (double)b[i]; o[3 * i + 2] = (double)f[i];
  const float g = (float)o[3 * i + 2];
  memcpy(&of[i], &g, 4);
  ol[i] = (long long)(double)f[i];
}
#define NI __attribute__((noinline))
static NI double h_add(double a, double b) { return a + b; }
static NI double h_mul(double a, double b) { return a * b; }
static NI double h_div(double a, double b) { return a / b; }
static NI double h_sqrt(double a) { return sqrt(a); }
static NI double h_cvt(long long a) { return (double)a; }
static double bits(uint64_t u) { double d; memcpy(&d, &u, 8); return d; }
static uint64_t ubits(double d) { uint64_t u; memcpy(&u, &d, 8); return u; }
static bool same(double a, double b) { return ubits(a) == ubits(b) || (a != a && b != b); }
int main() {
  const int N = 1 << 16;
  std::mt19937_64 g(12345);
  std::vector<double> a(N), b(N);
  for (int i = 0; i < N; i++) {
    switch (i % 5) {
    case 0: a[i] = (double)(int64_t)g() / 3.0; b[i] = (double)(int64_t)g() / 7.0; break;
    case 1: a[i] = bits(g()); b[i] = bits(g()); break;
    case 2: a[i] = (double)(int)(g() % 2001) - 1000; b[i] = (double)(int)(g() % 2001) - 1000; break;
    case 3: a[i] = ldexp((double)(g() >> 11), -(int)(g() % 80)); b[i] = a[i] * (1 + ldexp(1.0, -40)); break;
    default: a[i] = bits(g() & 0x800FFFFFFFFFFFFFull); b[i] = bits(0x3FF0000000000000ull | (g() >> 12)); break; // subnormals
    }
    if (a[i] != a[i] || fabs(a[i]) > 1e300) a[i] = 1.5;
    if (b[i] != b[i] || fabs(b[i]) > 1e300) b[i] = -2.25;
  }
  double *da, *db, *dr; long long *dn;
  cudaMalloc(&da, N * 8); cudaMalloc(&db, N * 8); cudaMalloc(&dr, (size_t)N * 12 * 8); cudaMalloc(&dn, N * 8);
  cudaMemcpy(da, a.data(), N * 8, cudaMemcpyHostToDevice); cudaMemcpy(db, b.data(), N * 8, cudaMemcpyHostToDevice);
  k_ops<<<N / 256, 256>>>(da, db, dr, dn, N);
  std::vector<double> r((size_t)N * 12); std::vector<long long> nn(N);
  cudaMemcpy(r.data(), dr, r.size() * 8, cudaMemcpyDeviceToHost); cudaMemcpy(nn.data(), dn, N * 8, cudaMemcpyDeviceToHost);
  long bad[13] = {};
  const char *names[13] = {"add", "sub", "mul", "div", "sqrt", "add up", "mul up", "div up", "sqrt up", "rint", "floor", "i64 up", "compare"};
  for (int i = 0; i < N; i++) {
    const volatile double x = a[i], y = b[i];
    double e[12];
    fesetround(FE_TONEAREST);
    e[0] = x + y; e[1] = x - y; e[2] = x * y; e[3] = x / y; e[4] = sqrt(fabs(x)); e[9] = rint(x); e[10] = floor(x);
    const long long li = fabs(x) < 9e18 ? (long long)x : 0;
    fesetround(FE_UPWARD);
    e[5] = h_add(x, y); e[6] = h_mul(x, y); e[7] = h_div(x, y); e[8] = h_sqrt(fabs(x)); e[11] = h_cvt(li);
    fesetround(FE_TONEAREST);
    for (int k = 0; k < 12; k++) {
      if (k == 11 && fabs(x) >= 9e18) continue;
      if (!same(e[k], r[(size_t)i * 12 + k])) { if (bad[k]++ < 4) printf("%s: x %a y %a cpu %a gpu %a\n", names[k], (double)x, (double)y, e[k], r[(size_t)i * 12 + k]); }
    }
    const long long c = (x < y) | ((x == y) << 1) | ((x >= y) << 2) | ((x != x) << 3);
    if (c != nn[i]) bad[12]++;
  }
  std::vector<long long> ia(N); std::vector<unsigned long long> ib(N); std::vector<float> fa(N);
  for (int i = 0; i < N; i++) { ia[i] = (long long)g() >> (g() % 60); ib[i] = g() >> (g() % 60); fa[i] = (float)((double)(int64_t)g() / 1e6) * (i % 3 ? 1.f : 1e-30f); }
  long long *la; unsigned long long *lb; float *lf; double *lo; unsigned *lof; long long *lol;
  cudaMalloc(&la, N * 8); cudaMalloc(&lb, N * 8); cudaMalloc(&lf, N * 4); cudaMalloc(&lo, (size_t)N * 24); cudaMalloc(&lof, N * 4); cudaMalloc(&lol, N * 8);
  cudaMemcpy(la, ia.data(), N * 8, cudaMemcpyHostToDevice); cudaMemcpy(lb, ib.data(), N * 8, cudaMemcpyHostToDevice); cudaMemcpy(lf, fa.data(), N * 4, cudaMemcpyHostToDevice);
  k_conv<<<N / 256, 256>>>(la, lb, lf, lo, lof, lol, N);
  std::vector<double> co((size_t)N * 3); std::vector<unsigned> cf(N); std::vector<long long> cl(N);
  cudaMemcpy(co.data(), lo, co.size() * 8, cudaMemcpyDeviceToHost); cudaMemcpy(cf.data(), lof, N * 4, cudaMemcpyDeviceToHost); cudaMemcpy(cl.data(), lol, N * 8, cudaMemcpyDeviceToHost);
  long badc = 0;
  for (int i = 0; i < N; i++) {
    unsigned fb; memcpy(&fb, &fa[i], 4);
    const bool ok = same(co[3 * i], (double)ia[i]) && same(co[3 * i + 1], (double)ib[i]) && same(co[3 * i + 2], (double)fa[i]) && cf[i] == fb && cl[i] == (long long)(double)fa[i];
    if (!ok && badc++ < 4) printf("conversion: i64 %lld u64 %llu f %a: gpu %a %a %a %x %lld\n", ia[i], ib[i], fa[i], co[3 * i], co[3 * i + 1], co[3 * i + 2], cf[i], cl[i]);
  }
  long total = badc;
  for (int k = 0; k < 13; k++) { if (bad[k]) printf("%s: %ld of %d differ\n", names[k], bad[k], N); total += bad[k]; }
  printf("float64_test: %s (%d inputs, 13 operations, conversions)\n", total ? "FAILED" : "all bits agree", N);
  return total != 0;
}
