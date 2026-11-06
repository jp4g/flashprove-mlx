// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
// A secret seed from which a commitment's random entries are derived, one child key for each use, so
// that fp_prove can rebuild the leaf table that fp_commit made.
#pragma once
#include <cstdio>
#include <fcntl.h>
#include <memory>
#include <unistd.h>
#include "coins/run_coins.h"
#include "metal_coins.h"
namespace fpmlx {
using namespace blk;
enum : u32 { COINS_ANCHOR = 101, COINS_KEY = 102, COINS_LEAF = 103, COINS_LINK = 104, COINS_REQUEST = 105 };
inline CoinKey load_or_make_seed(const char *path) {
  u8 b[32];
  int fd = open(path, O_RDONLY);
  if (fd >= 0) {
    if (read(fd, b, 32) != 32) { fprintf(stderr, "seed file %s is too short\n", path); exit(2); }
  } else {
    fresh32(b);
    fd = open(path, O_WRONLY | O_CREAT | O_EXCL, 0600);
    if (fd < 0 || write(fd, b, 32) != 32) { fprintf(stderr, "cannot write the seed file %s\n", path); exit(2); }
  }
  close(fd);
  CoinKey k = coin_key_from_bytes(b);
  wipe(b, sizeof b);
  return k;
}
inline std::unique_ptr<StreamCoins> seeded_coins(const CoinKey &seed, u32 use) {
  CoinKey child = seed_child(seed, use, 0);
  auto c = std::make_unique<MetalCoins>(child, CoinContext{});
  wipe(child);
  return c;
}
}
