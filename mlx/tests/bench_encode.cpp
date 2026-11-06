// Times encode_rows: usage bench_encode <rows> <log_dim> <log_n>
#include <chrono>
#include <cstdio>
#include <vector>
#include "code/code.h"
#include "mlx_kernels.h"
using namespace blk;
int main(int c, char **v) {
  const size_t rows = atoi(v[1]); const unsigned ld = atoi(v[2]), ln = atoi(v[3]);
  std::vector<u64> co(rows << ld, 5);
  auto tw = transform_table(ln);
  for (int i = 0; i < 2; i++) {
    auto t0 = std::chrono::steady_clock::now();
    fpmlx::Buf b = fpmlx::encode_rows(co.data(), rows, ld, ln, tw->data());
    printf("rows %zu: %.3f s\n", rows, std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
  }
}
