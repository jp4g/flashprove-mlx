// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
// MetalCoins against blk::StreamCoins: elements and bytes of every draw equal, in a mixed order of short
// and long draws, odd lengths among them, and a refused word redrawn the same way.
#include <cstdio>
#include <vector>
#include "metal_coins.h"
using namespace blk;
int main() {
  CoinKey k;
  for (int i = 0; i < 8; i++) k.w[i] = 0x1234567u * (i + 1);
  StreamCoins a(k, CoinContext{});
  fpmlx::MetalCoins b(k, CoinContext{});
  int bad = 0, draws = 0;
  for (size_t n : {size_t(5), size_t(1) << 14, (size_t(1) << 17) + 3, size_t(100), size_t(1000003)}) {
    std::vector<F> x(n), y(n);
    a.fill(x.data(), n), b.fill(y.data(), n);
    draws++;
    if (memcmp(x.data(), y.data(), n * 8)) bad++, printf("DIFFERS: %zu elements\n", n);
  }
  for (size_t n : {size_t(7), size_t(1) << 17, (size_t(1) << 20) + 5, (size_t(1) << 21) + 1}) {
    std::vector<u8> x(n), y(n);
    a.bytes(x.data(), n), b.bytes(y.data(), n);
    draws++;
    if (memcmp(x.data(), y.data(), n)) bad++, printf("DIFFERS: %zu bytes\n", n);
  }
  printf("%d of %d draws equal\n", draws - bad, draws);
  return bad != 0;
}
