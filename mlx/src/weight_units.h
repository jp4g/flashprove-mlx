// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
// The units of the committed matrix (docs/format/11-5-weights-in-the-table.md), the claims that a
// segment makes on them, and the two vectors over the columns that the merge sums.
#pragma once
#include <map>
#include "companion_link.h"
#include "integer_forward.h"
#include "seg_common.h"
namespace fpmlx {
// Weights as integers, by name, read once.
class WeightInts {
public:
  explicit WeightInts(const Tensors &w) : w_(w) {}
  const std::vector<int32_t> &at(const std::string &n) const {
    auto it = c_.find(n);
    if (it != c_.end()) return it->second;
    const mx::array a = mx::astype(w_.at(n), mx::int32);
    mx::eval(a);
    return c_[n] = std::vector<int32_t>(a.data<int32_t>(), a.data<int32_t>() + a.size());
  }
private:
  const Tensors &w_;
  mutable std::map<std::string, std::vector<int32_t>> c_;
};
enum class Cls { LOW, HIGH, A };
struct UnitGroup { u64 M = 0; Cls cls = Cls::LOW; unsigned cb0 = 0, t = 0, z = 0; u64 lam = 1, o = 1; };
struct Unit { unsigned X = 0, l = 0, nd = 2, ncb = 0; u64 off = 0, rows = 0; std::vector<UnitGroup> g; };
struct WeightFactor { std::vector<E> HV, pt, AV; };
struct WeightClaim { u32 ui = 0; std::vector<WeightFactor> f; };
inline std::vector<WeightClaim> read_claims(const std::vector<u8> &bytes) {
  RecordReader r(bytes);
  auto ve = [&] { std::vector<E> v(r.u()); for (E &x : v) { u64 w[3]; w[0] = r.u(), w[1] = r.u(), w[2] = r.u(); x = E{{F{w[0]}, F{w[1]}, F{w[2]}}}; } return v; };
  std::vector<WeightClaim> c(r.u());
  for (WeightClaim &x : c) {
    x.ui = (u32)r.u();
    x.f.resize(r.u());
    for (WeightFactor &f : x.f) f.HV = ve(), f.pt = ve(), f.AV = ve();
  }
  return c;
}
struct Units {
  unsigned lk = 0;
  std::vector<Unit> u;
  void add(unsigned X, unsigned l, unsigned nd, std::vector<std::pair<u64, Cls>> groups, u64 &off) {
    Unit x;
    x.X = X, x.l = l, x.nd = nd, x.off = off;
    unsigned left = lk, cb = 0;
    u64 rows = nd;
    for (auto &[M, c] : groups) {
      UnitGroup q;
      q.M = M, q.cls = c, q.cb0 = cb;
      if (c == Cls::A) {
        const unsigned b = clog2u(M);
        q.t = b <= left ? b : 0, q.lam = q.t ? 1 : M;
      } else {
        while (!((M >> q.z) & 1)) q.z++;
        q.o = M >> q.z, q.t = std::min(q.z, left), q.lam = M >> q.t;
      }
      left -= q.t, cb += q.t, rows *= q.lam;
      x.g.push_back(q);
    }
    x.ncb = cb, x.rows = rows;
    off += rows;
    u.push_back(std::move(x));
  }
  static Units of(const ModelShape &p, unsigned lk) {
    Units r;
    r.lk = lk;
    u64 off = 0;
    const u64 d = p.d, dh = p.dh, dkv = p.dkv, dq = p.dq, f = p.f, fe = p.fe, NE = p.NE, V = p.V;
    for (unsigned l = 0; l < (unsigned)p.L; l++) {
      r.add(0, l, 2, {{64, Cls::A}, {dh, Cls::A}, {d, Cls::LOW}}, off);
      r.add(1, l, 2, {{d, Cls::LOW}, {dkv, Cls::HIGH}}, off);
      r.add(2, l, 2, {{dq, Cls::LOW}, {d, Cls::HIGH}}, off);
      if (p.layers[l].experts) {
        r.add(5, l, 2, {{d, Cls::LOW}, {fe, Cls::HIGH}}, off);
        r.add(6, l, 2, {{d, Cls::LOW}, {fe, Cls::HIGH}}, off);
        r.add(7, l, 2, {{d, Cls::LOW}, {NE, Cls::LOW}, {fe, Cls::HIGH}}, off);
        r.add(8, l, 2, {{d, Cls::LOW}, {NE, Cls::LOW}, {fe, Cls::HIGH}}, off);
        r.add(9, l, 2, {{fe, Cls::LOW}, {d, Cls::HIGH}}, off);
        r.add(10, l, 2, {{fe, Cls::LOW}, {NE, Cls::LOW}, {d, Cls::HIGH}}, off);
        r.add(11, l, 2, {{d, Cls::LOW}, {NE, Cls::HIGH}}, off);
      } else {
        r.add(3, l, 2, {{d, Cls::LOW}, {f, Cls::HIGH}}, off);
        r.add(4, l, 2, {{d, Cls::LOW}, {f, Cls::HIGH}}, off);
        r.add(9, l, 2, {{f, Cls::LOW}, {d, Cls::HIGH}}, off);
      }
    }
    r.add(12, (unsigned)p.L, 4, {{V, Cls::A}, {d, Cls::HIGH}}, off);
    r.add(13, (unsigned)p.L, 2, {{V, Cls::A}, {d, Cls::LOW}}, off);
    return r;
  }
  // The claim as the merge sums it: over the columns c, Ea[c] Ba[c], with Ea the claim's weight of the
  // column and Ba the sum over the unit's rows of the row's weight times the cell.
  void claim_maps(const WeightClaim &cl, const int8_t *M, u64 matrix_cells, u64 columns, std::vector<E> &Ea, std::vector<E> &Ba) const {
    const Unit &x = u[cl.ui];
    if (cl.f.size() != x.g.size()) throw std::runtime_error("a claim with another number of factors");
    Ea.assign(columns, E::one());
    for (u64 c = 0; c < columns; c++) {
      E w = E::one();
      for (size_t i = 0; i < x.g.size(); i++) {
        const UnitGroup &q = x.g[i];
        const WeightFactor &f = cl.f[i];
        const u64 part = (c >> q.cb0) & ((1ull << q.t) - 1);
        if (q.cls == Cls::A) {
          if (q.t) w = w * (part < q.M ? f.AV[part] : E::zero());
          continue;
        }
        for (unsigned b = 0; b < q.t; b++) {
          // Bit b of the taken bits: bit b of lo (low), bit z - t + b (high); pt[0] is lo's highest bit.
          const E &z = f.pt[q.cls == Cls::LOW ? q.z - 1 - b : q.t - 1 - b];
          w = w * (((part >> b) & 1) ? z : E::one() - z);
        }
      }
      if (x.ncb < lk && (c >> x.ncb)) w = E::zero();
      Ea[c] = w;
    }
    // The rows' weights, group by group.
    std::vector<std::vector<E>> r(x.g.size());
    for (size_t i = 0; i < x.g.size(); i++) {
      const UnitGroup &q = x.g[i];
      const WeightFactor &f = cl.f[i];
      r[i].assign(q.lam, E::one());
      if (q.cls == Cls::A) {
        if (!q.t) for (u64 v = 0; v < q.lam; v++) r[i][v] = f.AV[v];
        continue;
      }
      std::vector<E> pl;
      if (q.cls == Cls::LOW) pl.assign(f.pt.begin(), f.pt.begin() + (q.z - q.t));
      else pl.assign(f.pt.begin() + q.t, f.pt.end());
      const std::vector<E> e = eq_all(pl);
      const u64 nlo = 1ull << (q.z - q.t);
      for (u64 v = 0; v < q.lam; v++) r[i][v] = f.HV[v / nlo] * e[v % nlo];
    }
    u64 per = 1;
    for (const UnitGroup &q : x.g) per *= q.lam;
    // The rows' weights in the order of the matrix, then g^T M on the GPU.
    std::vector<E> g((size_t)per * x.nd);
    E scale = E::one();
    for (unsigned qd = 0; qd < x.nd; qd++, scale = scale * E::from(from_u64(256)))
      for (u64 y = 0; y < per; y++) {
        E w = scale;
        u64 rest = y;
        for (size_t i = 0; i < x.g.size(); i++) { w = w * r[i][rest % x.g[i].lam]; rest /= x.g[i].lam; }
        g[y + qd * per] = w;
      }
    Ba = e_download(matrix_left(M, matrix_cells, x.off, g.size(), columns, g));
  }
  static E from_i64_e(long long v) { return E::from(from_i64(v)); }
};
}
