// F and E on the GPU, through MLX, against blocks/field on the CPU. Exit 0: every word is equal.
#include <cstdio>
#include <random>
#include <vector>
#include "field/field.h"
#include "mlx_kernels.h"
using namespace blk;
namespace mx = mlx::core;
static mx::array up(const std::vector<u64> &v) { return mx::array(reinterpret_cast<const uint64_t *>(v.data()), {(int)v.size()}, mx::uint64); }
int main() {
  const int N = 1 << 20;
  std::mt19937_64 g(7);
  std::vector<u64> a(N), b(N);
  for (int i = 0; i < N; i++) {
    a[i] = GL64::canonical(i % 7 == 0 ? GL64::P - 1 - (g() & 15) : g());
    b[i] = GL64::canonical(i % 11 == 0 ? (g() & 15) : g());
  }
  long bad = 0;
  const auto A = up(a), B = up(b);
  struct { const char *n; u64 (*f)(u64, u64); } ops[] = {{"add", GL64::add}, {"sub", GL64::sub}, {"mul", GL64::mul}};
  for (auto &o : ops) {
    auto r = fpmlx::f_binary(o.n, A, B);
    mx::eval(r);
    const uint64_t *d = r.data<uint64_t>();
    long w = 0;
    for (int i = 0; i < N; i++) w += d[i] != o.f(a[i], b[i]);
    printf("F %s: %ld of %d differ\n", o.n, w, N);
    bad += w;
  }
  const int M = N / 3;
  auto r = fpmlx::e_mul(mx::reshape(mx::slice(A, {0}, {3 * M}), {M, 3}), mx::reshape(mx::slice(B, {0}, {3 * M}), {M, 3}));
  mx::eval(r);
  const uint64_t *d = r.data<uint64_t>();
  long w = 0;
  for (int i = 0; i < M; i++) {
    E x, y;
    for (int k = 0; k < 3; k++) x.c[k].v = a[3 * i + k], y.c[k].v = b[3 * i + k];
    const E z = x * y;
    for (int k = 0; k < 3; k++) w += d[3 * i + k] != z.c[k].v;
  }
  printf("E mul: %ld of %d words differ\n", w, 3 * M);
  bad += w;
  printf("field_test: %s\n", bad ? "FAILED" : "all words agree");
  return bad != 0;
}
