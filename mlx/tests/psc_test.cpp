// The masked sumcheck against the claim of docs/format/11-1-machinery.md: the verifier's computation of
// cur, then Left(C; S0; K) = 0, on random data, for equality, plain and mixed rounds, with and without Y.
#include <cstdio>
#include <random>
#include "psc_prover.h"
using namespace fpmlx;
static std::mt19937_64 g(5);
static E rnd() { E x; for (int k = 0; k < 3; k++) x.c[k] = from_u64(g()); return x; }
struct Mock {
  std::vector<E> sent; E hs; std::vector<E> drawn;
  void send(const E &x) { sent.push_back(x); }
  void send_hiding_sum(const E &x) { hs = x; }
  E draw() { E r = rnd(); drawn.push_back(r); return r; }
  E epsilon() { return draw(); }
  E round() { return draw(); }
  E y() { return draw(); }
  void values() {}
  E public_part(const E &x) { return x; }
};
static int run(int n, int plain_every, int shown_n, bool squares, int order_rev) {
  // Summand: m0 * m1 (+ m2 kept) or with shown maps.
  const size_t N = size_t(1) << n;
  PscJob j;
  const unsigned maps = 3;
  j.maps.assign(maps, std::vector<E>(N));
  for (auto &m : j.maps) for (E &x : m) x = rnd();
  j.shown = shown_n; j.degree_y = shown_n ? 2 : 0;
  j.summand = {{rnd(), {0, 1}}, {rnd(), {2}}, {rnd(), {}}};
  std::vector<unsigned> order;
  for (int i = 0; i < n; i++) order.push_back(order_rev ? i : n - 1 - i);
  std::vector<E> z(n);
  for (int l = 0; l < n; l++) {
    z[l] = rnd();
    const bool plain = plain_every && l % plain_every == 0;
    j.vars.push_back({plain ? Weight<E>::plain(3) : Weight<E>::equality(z[l], 3), order[l]});
  }
  // The true sum: phi applied to H.
  E total = E::zero();
  for (size_t i = 0; i < N; i++) {
    E w = E::one();
    for (int l = 0; l < n; l++) { const bool one = (i >> order[l]) & 1; const E hi = j.vars[l].w.moment[1], lo = j.vars[l].w.moment[0] - hi; w = w * (one ? hi : lo); }
    E h = E::zero();
    for (auto &m : j.summand) { E t = m.coeff; for (unsigned k : m.maps) t = t * j.maps[k][i]; h = h + t; }
    total = total + w * h;
  }
  std::vector<E> mask(64);
  for (E &x : mask) x = rnd();
  j.start_value = total;
  j.start_public = rnd(); // pub(C): another value than C, as with masked scalars
  Mock ch; u32 slot = 3;
  const auto vars = j.vars; const auto summand = j.summand;
  const E Cpub = j.start_public, Cval = j.start_value; const unsigned shown = j.shown; const unsigned dY = j.degree_y;
  PscResult r = psc_run(ch, mask.data(), slot, j);
  // The verifier.
  std::vector<Weight<E>> all; for (auto &v : vars) all.push_back(v.w);
  if (shown) all.push_back(Weight<E>::at_zero(dY + 1));
  E cur = ch.hs + r.eps * Cpub;
  size_t at = 0, di = 1;
  std::vector<E> gam;
  for (size_t l = 0; l < all.size(); l++) {
    std::vector<E> q(all[l].d);
    for (unsigned t = 0; t < all[l].d; t++) if (t != all[l].fixed) q[t] = ch.sent[at++];
    all[l].complete(q, cur);
    const E c = ch.drawn[di++];
    cur = poly_at(q, c);
    gam.push_back(c);
  }
  // Left = <form, mask> + eps PP (pub C - C) + eps K + eps S0 - cur.
  const std::vector<E> form = hp_form(all, gam);
  E left = E::zero();
  for (size_t i = 0; i < form.size(); i++) left = left + form[i] * mask[3 + i];
  E PP = E::one();
  for (size_t t = 0; t < all.size(); t++) PP = PP * poly_at(all[t].psi, gam[t]);
  left = left + r.eps * PP * (Cpub - Cval);
  std::vector<E> val = r.values;
  for (unsigned i = 0; i < shown; i++) val[i] = r.end_values[i];
  E S = E::zero();
  for (auto &m : summand) { E t = m.coeff; for (unsigned k : m.maps) t = t * val[k]; S = S + t; }
  left = left + r.eps * S - cur;
  const bool ok = left == E::zero();
  printf("n %d plain/%d shown %d order %d: %s\n", n, plain_every, shown_n, order_rev, ok ? "claim holds" : "CLAIM FAILS");
  return !ok;
}
int main() {
  int bad = 0;
  for (int n : {0, 1, 3, 6}) for (int p : {0, 1, 2}) for (int s : {0, 1, 2}) for (int o : {0, 1}) { if (n == 0 && !s) continue; bad += run(n, p, s, false, o); }
  printf("psc_test: %s\n", bad ? "FAILED" : "every claim holds");
  return bad != 0;
}
