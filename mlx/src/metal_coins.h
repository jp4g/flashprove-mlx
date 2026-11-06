// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
// blk::StreamCoins with long draws made on the GPU: the same stream numbers, the same values. A refused
// word, about one in 2^32, is drawn again on the host by coin_admit(). A wrong generator would give
// proofs that verify and hide nothing, so each long draw is compared at three places with the host's
// block, and a difference ends the process.
#pragma once
#include "coins/coins.h"
#include "mlx_kernels.h"
namespace fpmlx {
using namespace blk;
class MetalCoins : public StreamCoins {
public:
  using StreamCoins::StreamCoins;
  void fill(F *out, size_t n) override {
    if (n < LONG || in_forked_child()) return StreamCoins::fill(out, n);
    const u32 s = open(true, n);
    Buf w = chacha_words(key().w, s, n);
    check(w, s, n);
    const u64 *p = w.data();
    for (size_t i = 0; i < n; i++)
      if (!F_from_word(p[i], out[i])) out[i] = coin_admit(p[i], key(), s, i);
  }
  void bytes(u8 *out, size_t n) override {
    if (n < 8 * LONG || in_forked_child()) return StreamCoins::bytes(out, n);
    const u32 s = open(false, n);
    const size_t count = (n + 7) / 8;
    Buf w = chacha_words(key().w, s, count);
    check(w, s, count);
    memcpy(out, w.data(), n); // words, lowest byte first
  }
private:
  static constexpr size_t LONG = size_t(1) << 14;
  void check(const Buf &w, u32 s, size_t n) const {
    for (size_t i : {size_t(0), n / 2, n - 1})
      BLK_REQUIRE(w.data()[i] == coin_word(key(), s, i), "coins: the GPU's generator differs from the host's");
  }
};
}
