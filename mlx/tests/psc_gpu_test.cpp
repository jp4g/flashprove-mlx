// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
// The rounds of a sumcheck on the GPU against the same rounds on the CPU: every sent word, challenge
// and end value must be equal, over summands, bit orders, plain and equality rounds, and the handover
// from GPU to CPU in the middle of a run.
#include <cstdio>
#include <random>
#include "psc_prover.h"
using namespace fpmlx;
struct Mock {
  std::mt19937_64 g;
  std::vector<E> sent;
  explicit Mock(u64 seed) : g(seed) {}
  E rnd() { E x; for (int k = 0; k < 3; k++) x.c[k] = from_u64(g()); return x; }
  void send(const E &x) { sent.push_back(x); }
  void send_hiding_sum(const E &x) { sent.push_back(x); }
  E epsilon() { return rnd(); }
  E round() { return rnd(); }
  E y() { return rnd(); }
  void values() {}
  E public_part(const E &x) { return x; }
};
static bool same(const E &a, const E &b) { for (int k = 0; k < 3; k++) if (a.c[k].v != b.c[k].v) return false; return true; }
static bool same(const std::vector<E> &a, const std::vector<E> &b) {
  if (a.size() != b.size()) return false;
  for (size_t i = 0; i < a.size(); i++) if (!same(a[i], b[i])) return false;
  return true;
}
static PscJob make(u64 seed, int n, int maps, const std::vector<std::vector<unsigned>> &mono, int order, int plain_every, unsigned shown, unsigned deg) {
  std::mt19937_64 g(seed);
  auto rnd = [&] { E x; for (int k = 0; k < 3; k++) x.c[k] = from_u64(g()); return x; };
  PscJob j;
  j.maps.assign(maps, std::vector<E>(size_t(1) << n));
  for (auto &m : j.maps) for (E &x : m) x = rnd();
  for (const auto &m : mono) j.summand.push_back({rnd(), m});
  j.shown = shown, j.degree_y = shown ? 3 : 0;
  std::vector<unsigned> o;
  for (int i = 0; i < n; i++) o.push_back(i);
  if (order == 0) std::reverse(o.begin(), o.end());
  if (order == 2) std::shuffle(o.begin(), o.end(), g);
  for (int l = 0; l < n; l++) {
    const bool plain = plain_every && l % plain_every == 0;
    j.vars.push_back({plain ? Weight<E>::plain(deg) : Weight<E>::equality(rnd(), deg), o[l]});
  }
  j.start_value = rnd(), j.start_public = rnd();
  return j;
}
static std::vector<std::vector<unsigned>> many() {
  std::vector<std::vector<unsigned>> m;
  for (unsigned i = 0; i < 12; i++) m.push_back({i, 12 + i, 24 + i, 36 + i, 48 + i, 60 + i % 10});
  return m;
}
int main() {
  struct Case { int n, maps; std::vector<std::vector<unsigned>> mono; unsigned deg; };
  const std::vector<Case> cases = {
      {12, 2, {{0, 1}}, 3},
      {13, 1, {{0, 0}, {0}}, 3},
      {12, 2, {{}, {0, 1}}, 3},
      {14, 3, {{0, 1}, {2}, {}}, 3},
      {11, 4, {{0, 0, 1}, {2, 3}, {1, 3}}, 4},
      {15, 3, {{0, 1, 2}}, 4},
      {10, 5, {{0, 1}, {3}}, 3}, // maps 2 and 4 are read by nobody, and still bound
      {10, 70, many(), 7},       // too long to unroll: the monomials are data
  };
  int bad = 0, runs = 0;
  std::vector<E> mask(4096);
  { std::mt19937_64 g(9); for (E &x : mask) for (int k = 0; k < 3; k++) x.c[k] = from_u64(g()); }
  for (size_t c = 0; c < cases.size(); c++)
    for (int order = 0; order < 3; order++)
      for (int plain = 0; plain < 3; plain++)
        for (unsigned shown : {0u, 1u})
          for (size_t min : {size_t(512), size_t(2048)}) {
            const Case &k = cases[c];
            const u64 seed = 1000 * c + 100 * order + 10 * plain + shown;
            PscResult r[2]; std::vector<E> sent[2];
            for (int gpu = 0; gpu < 2; gpu++) {
              psc_gpu_min() = gpu ? min : 0;
              Mock ch(seed);
              u32 slot = 1;
              r[gpu] = psc_run(ch, mask.data(), slot, make(seed, k.n, k.maps, k.mono, order, plain, shown, k.deg));
              sent[gpu] = ch.sent;
            }
            runs++;
            if (!same(sent[0], sent[1]) || !same(r[0].values, r[1].values) || !same(r[0].challenges, r[1].challenges)) {
              bad++;
              printf("DIFFERS: case %zu order %d plain %d shown %u from %zu\n", c, order, plain, shown, min);
            }
          }
  printf("%d of %d runs equal\n", runs - bad, runs);
  return bad != 0;
}
