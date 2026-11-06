// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
// Tensors written out on the GPU against Tensor::add_to(): products, choice variables on low and high
// bits, dense factors shared by several Tensors, pairs (1, 1), overlapping Tensors.
#include <cstdio>
#include <random>
#include "mlx_kernels.h"
using namespace blk;
static std::mt19937_64 g(11);
static E rnd() { E x; for (int k = 0; k < 3; k++) x.c[k] = from_u64(g()); return x; }
int main() {
  int bad = 0, runs = 0;
  for (unsigned vars : {8u, 9u, 12u, 15u, 17u, 20u})
    for (int shape = 0; shape < 6; shape++) {
      std::vector<Tensor> ts;
      auto vals = std::make_shared<std::vector<E>>(size_t(1) << std::min(vars, 12u));
      for (E &x : *vals) x = rnd();
      const int count = 1 + (int)(g() % 40);
      for (int i = 0; i < count; i++) {
        Tensor t;
        t.scalar = rnd();
        const unsigned nd = (shape & 1) ? (unsigned)(g() % std::min(vars, 8u)) : 0;
        const unsigned plain = vars - nd;
        for (unsigned b = 0; b < plain; b++) {
          const unsigned k = (unsigned)(g() % 6);
          const bool low = shape >= 2 ? b < 6 : b >= plain - 3;
          if (low || k == 0) { if (g() & 1) t.push(E::one(), E::zero()); else t.push(E::zero(), E::one()); }
          else if (k == 1) t.push(E::one(), E::one());
          else t.push(rnd(), rnd());
        }
        if (nd) { t.push_dense(nd); t.set_values(vals, g() % (vals->size() - (size_t(1) << nd) + 1)); }
        ts.push_back(t);
      }
      std::vector<E> want(size_t(1) << vars, E::zero());
      std::vector<E> room;
      for (const Tensor &t : ts) t.add_to(want.data(), room);
      const fpmlx::Buf got = fpmlx::tensors_written(ts, vars);
      const u64 *w = got.data();
      size_t diff = 0;
      for (size_t j = 0; j < want.size(); j++) for (int k = 0; k < 3; k++) diff += w[3 * j + k] != want[j].c[k].v;
      runs++;
      if (diff) bad++, printf("DIFFERS: vars %u shape %d: %zu words\n", vars, shape, diff);
    }
  printf("%d of %d forms equal\n", runs - bad, runs);
  return bad != 0;
}
