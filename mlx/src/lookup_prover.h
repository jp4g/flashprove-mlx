// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
// The prover's side of a lookup by product trees (docs/format/11-1-machinery.md, "A lookup"; the
// verifier's side is verifier/src/vf_lookup_argument.h). All calls of a host's list run as one lookup.
#pragma once
#include <algorithm>
#include <unordered_map>
#include "psc_prover.h"
namespace fpmlx {
struct LookupCall {
  unsigned n = 0, k = 0;         // 2^n keys, a table of 2^k rows
  std::vector<u64> table;        // row values; empty: the range table, row t is t
  u64 t0 = 0;                    // the padding row
  std::vector<u64> keys;         // 2^n values; at a padding position, the value of row t0
  std::vector<char> padding;     // 2^n flags
  u64 row(u64 t) const { return table.empty() ? t : table[t]; }
};
struct LookupSizes {
  unsigned dT = 0, kT = 0, NC = 0, cK = 0, NT = 0, rK = 0, v = 0, dL = 0;
  std::vector<unsigned> iota, tree0, call_of; // by list index; call_of: by tree
  std::vector<u64> npad;                      // by tree
};
inline unsigned clog2u(u64 x) { unsigned l = 0; while ((u64(1) << l) < x) l++; return l; }
inline LookupSizes lookup_sizes(const std::vector<LookupCall> &c, unsigned lidb) {
  LookupSizes s;
  s.NC = (unsigned)c.size();
  s.cK = clog2u(s.NC);
  s.dT = lidb;
  for (const LookupCall &x : c) s.dT = std::min(s.dT, x.n), s.kT = std::max(s.kT, x.k);
  s.iota.resize(s.NC);
  u64 nt = 0;
  for (unsigned i = 0; i < s.NC; i++) s.iota[i] = c[i].n - s.dT, nt += u64(1) << s.iota[i];
  s.NT = (unsigned)nt;
  s.rK = clog2u(nt);
  s.v = s.dT + s.rK, s.dL = s.kT + s.cK;
  std::vector<unsigned> order(s.NC);
  for (unsigned i = 0; i < s.NC; i++) order[i] = i;
  std::stable_sort(order.begin(), order.end(), [&](unsigned a, unsigned b) { return s.iota[a] > s.iota[b]; });
  s.tree0.resize(s.NC), s.call_of.resize(s.NT), s.npad.assign(s.NT, 0);
  unsigned y = 0;
  for (unsigned i : order) {
    s.tree0[i] = y;
    for (u64 q = 0; q < (u64(1) << s.iota[i]); q++) s.call_of[y++] = i;
  }
  for (unsigned i = 0; i < s.NC; i++)
    for (u64 x = 0; x < (u64(1) << c[i].n); x++)
      if (c[i].padding[x]) s.npad[s.tree0[i] + (x >> s.dT)]++;
  return s;
}
// m(t, c): each key that is not padding, counted at its row of the call's table. False: a key is in no row.
inline bool lookup_multiplicities(const std::vector<LookupCall> &calls, const LookupSizes &s, std::vector<std::vector<u32>> &m) {
  m.assign(s.NC, std::vector<u32>(size_t(1) << s.kT, 0));
  for (unsigned c = 0; c < s.NC; c++) {
    const LookupCall &L = calls[c];
    if (L.table.empty()) { // the range table: a key is its row
      for (u64 x = 0; x < (u64(1) << L.n); x++) {
        if (L.padding[x]) continue;
        if (L.keys[x] >> L.k) return false;
        m[c][L.keys[x]]++;
      }
      continue;
    }
    std::unordered_map<u64, u32> row;
    for (u64 t = 0; t < (u64(1) << L.k); t++) row.emplace(L.row(t), (u32)t);
    for (u64 x = 0; x < (u64(1) << L.n); x++) {
      if (L.padding[x]) continue;
      auto it = row.find(L.keys[x]);
      if (it == row.end()) return false;
      m[c][it->second]++;
    }
  }
  return true;
}
// What the host needs afterwards.
struct LookupOut {
  E theta = E::zero(), alpha = E::zero();
  std::vector<E> zx, zy, zb[2]; // the key side's point; the two points of the planes
  E Cend = E::zero();           // hidden value of the key side's leaves at (zx | zy)
  E leaves_eps = E::zero();
  std::vector<E> zK;
  std::vector<E> e;                // by call: the weight of its trees at zy
  std::vector<std::vector<E>> zk;  // by call: its key point
};
// Ch adds to the sumcheck's channel: shift(), alpha(), point(round, n), line().
template <class Ch> LookupOut lookup_run(Ch &ch, const E *mask, u32 &slot, const std::vector<LookupCall> &calls, const LookupSizes &s, const std::vector<std::vector<u32>> &mult) {
  LookupOut o;
  if (s.NC > 1) o.theta = ch.shift();
  o.alpha = ch.alpha();
  const E &alpha = o.alpha, &theta = o.theta;
  auto keyed = [&](unsigned c, u64 value) { return E::from(from_u64(value)) + E::from(from_u64(c)) * theta; };
  const u64 nv = u64(1) << s.v;
  // The key side's levels, on the GPU: level m has 2^m vertices. leaves: (x | y).
  std::vector<mx::array> K;
  const E pad0 = alpha - keyed(0, calls[0].row(calls[0].t0));
  {
    LeafPlan L;
    L.rK = s.rK, L.dT = s.dT, L.v = s.v, L.NT = s.NT, L.alpha = alpha, L.theta = theta, L.pad0 = pad0;
    std::vector<u64> first(s.NC);
    std::vector<u64> keys;
    std::vector<char> padding;
    for (unsigned c = 0; c < s.NC; c++) {
      first[c] = keys.size();
      keys.insert(keys.end(), calls[c].keys.begin(), calls[c].keys.end());
      padding.insert(padding.end(), calls[c].padding.begin(), calls[c].padding.end());
      L.pad_value.push_back(calls[c].row(calls[c].t0));
    }
    // Tree y holds the call's keys from (y - tree0) 2^dT on, x being the key's low bits.
    for (u64 y = 0; y < s.NT; y++) {
      const unsigned c = s.call_of[y];
      L.tree_call.push_back(c), L.tree_first.push_back(first[c] + ((y - s.tree0[c]) << s.dT));
    }
    K.assign(s.v + 1, mx::array(0));
    K[s.v] = lookup_leaves(L, keys, padding);
    for (int m = (int)s.v - 1; m >= 0; m--) K[m] = product_level(K[m + 1]);
  }
  const E K00 = e_fetch(K[0])[0];
  // The table side: leaves (t | c | j).
  const unsigned nT = 5 + s.dL;
  std::vector<mx::array> Tt;
  std::shared_ptr<mx::array> bits_rows;
  {
    std::vector<u64> rows((size_t)s.NC << s.kT);
    std::vector<u32> counts((size_t)s.NC << s.kT);
    for (unsigned c = 0; c < s.NC; c++)
      for (u64 t = 0; t < (u64(1) << s.kT); t++) rows[(c << s.kT) + t] = calls[c].row(t & ((u64(1) << calls[c].k) - 1)), counts[(c << s.kT) + t] = mult[c][t];
    TableSide side = lookup_table_side(s.kT, s.cK, s.NC, rows, counts, alpha, theta);
    bits_rows = std::make_shared<mx::array>(side.bits_rows);
    Tt.assign(nT + 1, mx::array(0));
    Tt[nT] = side.leaves;
    for (int m = (int)nT - 1; m >= 5; m--) Tt[m] = product_level(Tt[m + 1]);
  }
  const std::vector<E> G = e_fetch(Tt[5]);
  // a_pad.
  E apad = E::one();
  {
    const u64 empty = ((u64(1) << s.rK) - s.NT) << s.dT;
    apad = pow(pad0, empty);
    for (unsigned y = 0; y < s.NT; y++) apad = apad * pow(alpha - keyed(s.call_of[y], calls[s.call_of[y]].row(calls[s.call_of[y]].t0)), s.npad[y]);
  }
  // Step 3: the tie. P_31 = G_31; P_s = P_(s+1)^2 G_s.
  std::vector<E> P(32);
  P[31] = G[31];
  for (int q = 30; q >= 0; q--) P[q] = P[q + 1] * P[q + 1] * G[q];
  E shownP = E::zero();     // shown value of the number of P_s, for the next step's pub(C)
  E shownZ = E::zero();
  std::vector<E> shownG(32);
  for (int q = 0; q <= 30; q++) {
    PscJob j;
    j.shown = q == 0 ? 3 : 2;
    j.degree_y = 3;
    if (q == 0) {
      j.maps = {{K00}, {P[1]}, {G[0]}};
      j.summand = {{E::one(), {0}}, {E::zero() - apad, {1, 1, 2}}};
      j.start_value = E::zero(), j.start_public = E::zero();
    } else {
      j.maps = {{P[q + 1]}, {G[q]}};
      j.summand = {{E::one(), {0, 0, 1}}};
      j.start_value = P[q], j.start_public = shownP;
    }
    const PscResult r = psc_run(ch, mask, slot, j);
    if (q == 0) shownZ = r.end_values[0], shownP = r.end_values[1], shownG[0] = r.end_values[2];
    else shownP = r.end_values[0], shownG[q] = r.end_values[1];
  }
  shownG[31] = shownP;
  // Step 4.
  std::vector<E> zT = ch.point_join(5);
  std::reverse(zT.begin(), zT.end());
  E CT = E::zero(), CTpub = E::zero();
  for (unsigned jn = 0; jn < 32; jn++) {
    E w = E::one();
    for (unsigned q = 0; q < 5; q++) w = w * (((jn >> (4 - q)) & 1) ? zT[q] : E::one() - zT[q]);
    CT = CT + w * G[jn], CTpub = CTpub + w * shownG[jn];
  }
  // A product level, the maps being the two halves of the level below.
  auto level = [&](const mx::array &below, unsigned m, std::vector<E> &z, E &C, E &Cpub, const std::vector<unsigned> &order) {
    PscJob j;
    j.device = std::make_shared<mx::array>(below), j.device_maps = 2;
    j.shown = 2, j.degree_y = 2;
    j.summand = {{E::one(), {0, 1}}};
    for (unsigned l = 0; l < m; l++) j.vars.push_back({Weight<E>::equality(z[m - 1 - order[l]], 3), order[l]});
    j.start_value = C, j.start_public = Cpub;
    const PscResult r = psc_run(ch, mask, slot, j);
    const E xi = ch.line();
    std::vector<E> next(m + 1);
    next[0] = xi;
    for (unsigned l = 0; l < m; l++) next[1 + (m - 1 - order[l])] = r.challenges[l];
    z = std::move(next);
    C = (E::one() - xi) * r.values[0] + xi * r.values[1];
    Cpub = (E::one() - xi) * r.end_values[0] + xi * r.end_values[1];
  };
  auto default_order = [](unsigned m) { std::vector<unsigned> o; for (unsigned b = m; b-- > 0;) o.push_back(b); return o; };
  // Step 5.
  std::vector<E> zK;
  E CK = K00, CKpub = shownZ;
  for (unsigned m = 0; m < s.v; m++) {
    std::vector<unsigned> order = default_order(m);
    if (m == s.v - 1) {
      order.clear();
      for (unsigned b : {2u, 1u, 0u, 5u, 4u, 3u, 8u, 7u, 6u}) order.push_back(s.rK + b);
      for (int b = (int)m - 1; b >= (int)s.rK + 9; b--) order.push_back((unsigned)b);
      for (int b = (int)s.rK - 1; b >= 0; b--) order.push_back((unsigned)b);
    }
    level(K[m + 1], m, zK, CK, CKpub, order);
  }
  o.zx.assign(zK.begin(), zK.begin() + s.dT);
  o.zy.assign(zK.begin() + s.dT, zK.end());
  o.zK = zK;
  o.Cend = CK;
  o.e.assign(s.NC, E::one()), o.zk.resize(s.NC);
  for (unsigned c = 0; c < s.NC; c++) {
    const unsigned hi = s.rK - s.iota[c];
    const u64 pre = (u64)s.tree0[c] >> s.iota[c];
    for (unsigned i = 0; i < hi; i++) o.e[c] = o.e[c] * (((pre >> (hi - 1 - i)) & 1) ? o.zy[i] : E::one() - o.zy[i]);
    o.zk[c].assign(o.zy.begin() + hi, o.zy.end());
    o.zk[c].insert(o.zk[c].end(), o.zx.begin(), o.zx.end());
  }
  // Step 6.
  for (unsigned m = 5; m < nT; m++) level(Tt[m + 1], m, zT, CT, CTpub, default_order(m));
  // Step 7.
  {
    PscJob j;
    j.device = bits_rows, j.device_maps = 2;
    j.summand = {{E::one(), {}}, {E::one(), {0, 1}}};
    for (unsigned l = 0; l < nT; l++) j.vars.push_back({Weight<E>::equality(zT[l], l < s.dL ? 3 : 2), nT - 1 - l});
    j.start_value = CT, j.start_public = CTpub;
    const PscResult r = psc_run(ch, mask, slot, j);
    o.zb[0] = r.challenges;
    o.leaves_eps = r.eps;
  }
  // Steps 8 and 9.
  {
    std::vector<E> zq = ch.point_zero(nT);
    std::reverse(zq.begin(), zq.end());
    PscJob j;
    j.device = std::make_shared<mx::array>(mx::slice(*bits_rows, {0}, {(int)(bits_rows->size() / 2)}));
    j.device_maps = 1;
    j.shown = 1, j.degree_y = 2;
    j.summand = {{E::one(), {0, 0}}, {E::zero() - E::one(), {0}}};
    for (unsigned l = 0; l < nT; l++) j.vars.push_back({Weight<E>::equality(zq[l], 3), nT - 1 - l});
    const PscResult r = psc_run(ch, mask, slot, j);
    o.zb[1] = r.challenges;
  }
  return o;
}
}
