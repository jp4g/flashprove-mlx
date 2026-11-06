// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
// Fills a feed-forward segment's matrices, keys and norm rows on Metal. ff_build()'s own loops are the
// reference, and ff_keys_test compares the two, cell by cell and key by key.
#pragma once
#include "seg_ff_witness.h"
#include "../kernels/witness_ids.h"
namespace fpmlx {
// A table for a kernel: at least 8 words, so that MLX gives it the device address space.
inline mx::array w_table(std::vector<u64> v) { v.resize(std::max<size_t>(v.size(), 8), 0); return mx::array(reinterpret_cast<const uint64_t *>(v.data()), {(int)v.size()}, mx::uint64); }
inline mx::array w_buffer(size_t bytes, mx::Dtype t, int fill) {
  mx::array a(mx::allocator::malloc(bytes), {(int)(bytes / mx::size_of(t))}, t);
  memset(a.data<uint8_t>(), fill, bytes);
  return a;
}
struct WitnessRun {
  std::vector<mx::array> common; // c0 .. c5, keys, padf, pos, kd, md, td
  std::vector<std::string> names{"c0", "c1", "c2", "c3", "c4", "c5", "keys", "padf", "pos", "kd", "md", "td"};
  std::vector<mx::array> pending;
  void launch(const std::string &name, const std::string &body, std::vector<std::pair<std::string, mx::array>> in, const mx::array &p, size_t x, size_t y, mx::Shape out = {1}) {
    std::vector<mx::array> a = common;
    std::vector<std::string> n = names;
    for (auto &[k, v] : in) n.push_back(k), a.push_back(v);
    n.push_back("p"), a.push_back(p);
    pending.push_back(run_kernel(name, body, a, n, {out}, x, y)[0]);
  }
};
}
