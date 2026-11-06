// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
// The sub-proof of the output head (docs/format/11-4-output-head.md): steps 1 to 12, on a channel like
// Chan that also has public_part(). The frame makes the arrays' messages between the moments:
//   the frame's REPLY: y;  moment 0: M0 .. (arrays0());  moment0();
//   moment 1: U3 (u3());   draw(), then digits();
//   moment 2: ZZ, BZ;      moment2().
#pragma once
#include "seg_head_build.h"
namespace fpmlx {
struct HeadProof {
  HeadWitness &W;
  const std::vector<int32_t> &Wh; // Wh[v * d + k]
  // The points and elements of 11.4, after moment0().
  LookupOut lk;
  std::vector<E> zt, zl, pV, rW, rV, rN, rH, rs; // rs[i]: bit i of the feature
  E gam = E::zero(), om = E::zero();
  std::vector<E> Ws, P;                          // hidden values of steps 3 and 5
  E ClV = E::zero(), ClN = E::zero(), ClH = E::zero();
  PscResult s4, s6, s7, s8;
  // After moment2().
  LookupOut lkZ;
  E x = E::zero(), chi = E::zero();
  PscResult merge;                               // merge.end_values[0] = fv', merge.beta, merge.challenges = rm
  Smallness small;
  HeadProof(HeadWitness &w, const std::vector<int32_t> &wh) : W(w), Wh(wh) {
    for (auto &p : W.hf.part) small.mats.push_back(&p);
    small.mats.push_back(&W.hd.part[0]), small.mats.push_back(&W.hr.part[0]), small.mats.push_back(&W.bH), small.mats.push_back(&W.xbL.part[0]);
    small.tau = W.s.tau, small.lidb = W.s.lidb;
  }
  // Moment 0's arrays in the order of the layout: M0, .. (hf, hd, hr, bH). xbL is the last group's xbO.
  std::vector<ArrayData> arrays0() const {
    std::vector<ArrayData> a;
    for (auto &p : W.hf.part) a.push_back(array_of(p));
    a.push_back(array_of(W.hd.part[0])), a.push_back(array_of(W.hr.part[0])), a.push_back(array_of(W.bH));
    return a;
  }
  // The key point's column coordinates of call c.
  std::vector<E> zj(int c) const {
    const unsigned iota = W.sizes.iota[c], h = W.sizes.rK - iota;
    std::vector<E> z(lk.zy.begin() + h, lk.zy.end());
    z.insert(z.end(), lk.zx.begin(), lk.zx.end());
    z.resize(W.lJ[c]);
    return z;
  }
  // ew_sg.
  std::vector<E> ew(const std::string &seg) const {
    const int c = W.call_of(seg);
    const std::vector<E> w = eq_all(zj(c));
    const SegPlace &p = W.place[c].at(seg);
    std::vector<E> o(p.n);
    for (u64 x = 0; x < p.n; x++) o[x] = w[p.pos(x)];
    return o;
  }
  std::vector<E> phiC() const { const std::vector<E> e = eq_all(zt); std::vector<E> o(W.s.T, E::zero()); for (int t = 0; t < W.s.T; t++) if (W.co[t]) o[t] = e[t]; return o; }
  template <class Ch> void moment0(Ch &ch, const E *mask, u32 &slot) {
    const HeadShape2 &s = W.s;
    const int T = s.T, lt = s.lt, d = s.d, lPd = s.lPd, Pd = 1 << lPd;
    const E half = inv(E::from(from_u64(2))), scale = power_of(half, lPd);
    auto F_ = [](I64 v) { return E::from(from_i64(v)); };
    auto eqv = [&](const std::vector<E> &z, unsigned top) { std::vector<PscVar> v; for (size_t l = 0; l < z.size(); l++) v.push_back({Weight<E>::equality(z[l], 3), top - (unsigned)l}); return v; };
    // Step 1.
    lk = lookup_run(ch, mask, slot, W.calls, W.sizes, W.mult);
    zt.assign(lk.zx.end() - lt, lk.zx.end());
    zl.assign(lk.zx.end() - s.la, lk.zx.end() - lt);
    // Step 2.
    pV = ch.point_zero(lt);
    const std::vector<E> ePV = eq_all(pV);
    // Step 3.
    Ws.assign(s.NS, E::zero());
    for (int i = 0; i < s.NS; i++) {
      for (int t = 0; t < T; t++) Ws[i] = Ws[i] + ePV[t] * F_((I64)W.rows[t].W[i]);
      ch.send(Ws[i] + mask[slot + i]);
    }
    E C4 = E::zero(), C4p = E::zero();
    const u32 j3 = slot;
    slot += s.NS;
    gam = ch.batch(s.NS);
    { E g = E::one(); for (int i = 0; i < s.NS; i++) C4 = C4 + g * Ws[i], C4p = C4p + g * (Ws[i] + mask[j3 + i]), g = g * gam; }
    // Step 4.
    {
      std::vector<E> XG((size_t)Pd * T, E::zero());
      for (int k = 0; k < d; k++)
        for (int t = 0; t < T; t++) {
          E a = E::zero(), g = E::one();
          for (int lb = 0; lb < s.XL; lb++) a = a + g * F_(W.xbL.cell((long long)lb * d + k, t)), g = g * gam;
          XG[(size_t)k * T + t] = a;
        }
      PscJob j;
      std::vector<E> z(lPd, half);
      z.insert(z.end(), pV.begin(), pV.end());
      j.vars = eqv(z, lPd + lt - 1);
      j.maps = {XG, XG};
      j.summand = {{E::one(), {0, 1}}};
      j.shown = 2, j.degree_y = 2, j.start_value = scale * C4, j.start_public = scale * C4p;
      s4 = psc_run(ch, mask, slot, std::move(j));
      rW = s4.challenges;
      ch.line();
    }
    // Step 5.
    std::vector<E> a(T), b(T), Rlo(T), Rhi(T);
    for (int t = 0; t < T; t++) { const NormRow &r = W.rows[t]; a[t] = F_(r.a), b[t] = F_(r.b), Rlo[t] = F_(r.Rlo), Rhi[t] = F_(r.Rhi); }
    P.assign(s.NPr, E::zero());
    for (int t = 0; t < T; t++) {
      const E pr[7] = {a[t] * a[t], a[t] * b[t], b[t] * b[t], Rlo[t] * a[t], Rlo[t] * b[t], Rhi[t] * a[t], Rhi[t] * b[t]};
      for (int i = 0; i < s.NPr; i++) P[i] = P[i] + ePV[t] * pr[i];
    }
    for (int i = 0; i < s.NPr; i++) ch.send(P[i] + mask[slot + i]);
    slot += s.NPr;
    om = ch.batch(s.NPr);
    // Step 6.
    {
      ClV = E::zero();
      E o = E::one();
      std::vector<E> pw;
      for (int i = 0; i < s.NPr; i++) ClV = ClV + o * P[i], pw.push_back(o), o = o * om;
      const E pub = ClV + mask[slot];
      ch.send(pub);
      slot += 1;
      PscJob j;
      j.vars = eqv(pV, lt - 1);
      j.maps = {a, b, Rlo, Rhi};
      j.summand = {{pw[0], {0, 0}}, {pw[1], {0, 1}}, {pw[2], {1, 1}}, {pw[3], {2, 0}}, {pw[4], {2, 1}}, {pw[5], {3, 0}}, {pw[6], {3, 1}}};
      j.shown = 4, j.degree_y = 2, j.start_value = ClV, j.start_public = pub;
      s6 = psc_run(ch, mask, slot, std::move(j));
      rV = s6.challenges;
    }
    const std::vector<E> eZT = eq_all(zt);
    // Step 7.
    {
      const std::vector<E> eNF = ew("NF");
      std::vector<E> Rt(T), SX(T, E::zero());
      ClN = E::zero();
      for (int t = 0; t < T; t++) {
        Rt[t] = F_(W.rows[t].R);
        for (int k = 0; k < d; k++) SX[t] = SX[t] + eNF[k] * F_(W.str[(size_t)t * d + k]);
        ClN = ClN + eZT[t] * Rt[t] * SX[t];
      }
      const E pub = ClN + mask[slot];
      ch.send(pub);
      slot += 1;
      PscJob j;
      j.vars = eqv(zt, lt - 1);
      j.maps = {Rt, SX};
      j.summand = {{E::one(), {0, 1}}};
      j.shown = 2, j.degree_y = 2, j.start_value = ClN, j.start_public = pub;
      s7 = psc_run(ch, mask, slot, std::move(j));
      rN = s7.challenges;
    }
    // Step 8.
    {
      const std::vector<E> e0 = ew("HD0");
      std::vector<E> G(d, E::zero());
      E wx = E::zero();
      for (int v = 0; v < s.V; v++) wx = wx + e0[v];
      psc_parallel(d, [&](size_t lo, size_t hi) { for (size_t k = lo; k < hi; k++) for (int v = 0; v < s.V; v++) G[k] = G[k] + e0[v] * F_(Wh[(size_t)v * d + k]); });
      std::vector<E> act((size_t)Pd * T, E::zero()), UH((size_t)Pd * T, E::zero());
      ClH = E::zero();
      for (int k = 0; k < d; k++)
        for (int t = 0; t < T; t++) {
          const size_t i = (size_t)k * T + t;
          act[i] = F_(W.act[(size_t)t * d + k]);
          if (W.on[t]) UH[i] = wx * F_(Wh[(size_t)W.win[t] * d + k]) - G[k];
          ClH = ClH + eZT[t] * act[i] * UH[i];
        }
      ClH = scale * ClH;
      const E pub = ClH + mask[slot];
      ch.send(pub);
      slot += 1;
      PscJob j;
      j.vars = eqv(zt, lt - 1);
      for (int i = 0; i < lPd; i++) j.vars.push_back({Weight<E>::equality(half, 3), (unsigned)(lt + lPd - 1 - i)});
      j.maps = {act, UH};
      j.summand = {{E::one(), {0, 1}}};
      j.shown = 2, j.degree_y = 2, j.start_value = ClH, j.start_public = pub;
      s8 = psc_run(ch, mask, slot, std::move(j));
      rH.assign(s8.challenges.begin(), s8.challenges.begin() + lt);
      rs.assign(lPd, E::zero());
      for (int i = 0; i < lPd; i++) rs[lPd - 1 - i] = s8.challenges[lt + i];
    }
  }
  // U3's parts, in 11.4's order. Their sum is what moment 1 commits to.
  std::vector<E> u3() {
    const HeadShape2 &s = W.s;
    const int NHf = s.NHf, T = s.T;
    const unsigned mHd = NHf, mHr = NHf + 1, mB = NHf + 2, mX = NHf + 3;
    auto cat = [](std::vector<E> a, const std::vector<E> &b) { a.insert(a.end(), b.begin(), b.end()); return a; };
    const std::vector<E> pc = phiC();
    std::vector<E> rho(T, E::zero()), pc2;
    for (int t = 0; t < T; t++) if (W.co[t]) rho[t] = E::one();
    for (int j = 0; j < s.S2; j++) pc2.insert(pc2.end(), pc.begin(), pc.end());
    small.parts.clear();
    const std::vector<E> p1 = eq_all(cat(zl, zt));
    for (int i = 0; i < NHf; i++) small.parts.push_back({(unsigned)i, p1});
    for (int i = 0; i < NHf; i++) small.parts.push_back({(unsigned)i, pc2});
    small.parts.push_back({mX, eq_all(std::vector<E>(rW.end() - s.la, rW.end()))});
    small.parts.push_back({mX, eq_all(cat(zl, rN))});
    small.parts.push_back({mHd, eq_all(zt)});
    small.parts.push_back({mHd, eq_all(rH)});
    for (const std::vector<E> *z : {&zt, &pV, &rV, &rN}) small.parts.push_back({mHr, eq_all(*z)});
    small.parts.push_back({mHr, pc});
    small.parts.push_back({mHr, rho});
    const LookupSizes &z = W.sizes;
    const int lowT = s.lmbH - 5;
    for (int e = 0; e < 2; e++) {
      std::vector<E> pt(lk.zb[e].begin() + (z.kT - lowT), lk.zb[e].begin() + z.kT);
      pt.insert(pt.end(), lk.zb[e].end() - 5, lk.zb[e].end());
      small.parts.push_back({mB, eq_all(pt)});
    }
    std::vector<E> pp(1ull << s.lmbH);
    for (u64 i = 0; i < pp.size(); i++) pp[i] = E::from(from_u64(1ull << (i & 31)));
    small.parts.push_back({mB, pp});
    return small.u3();
  }
  // Step 9.
  void draw(ProverRounds &r) { small.draw_theta(r); }
  bool digits(ArrayData &ZZ, ArrayData &BZ) { return small.digits(ZZ, BZ); }
  // The factors of the weight claim on unit 13 (11.5), over the columns.
  void weight_factors(std::vector<E> &Ec, std::vector<E> &Bc) const {
    const HeadShape2 &s = W.s;
    const int d = s.d, lk_ = s.lkW, tA = (int)clog2u(s.V), left = lk_ - tA;
    int z = 0;
    while (((d >> z) & 1) == 0) z++;
    const int t = std::min(z, left), ncb = tA + t, o = d >> z;
    const std::vector<E> e0 = ew("HD0"), eH = eq_all(rH);
    E wx = E::zero(), sR = E::zero();
    std::vector<E> psiH(s.V, E::zero());
    for (int v = 0; v < s.V; v++) wx = wx + e0[v];
    for (int q = 0; q < s.T; q++) if (W.on[q]) psiH[W.win[q]] = psiH[W.win[q]] + eH[q], sR = sR + eH[q];
    std::vector<E> AV(s.V);
    for (int v = 0; v < s.V; v++) AV[v] = wx * psiH[v] - sR * e0[v];
    std::vector<E> ph, pl; // (rs_(lPd-1), .., rs_z); (rs_(z-1), .., rs_t)
    for (int i = s.lPd - 1; i >= z; i--) ph.push_back(rs[i]);
    for (int i = z - 1; i >= t; i--) pl.push_back(rs[i]);
    const std::vector<E> HV = eq_all(ph), PL = eq_all(pl);
    Ec.assign(1ull << lk_, E::zero()), Bc.assign(1ull << lk_, E::zero());
    for (u64 c = 0; c < (1ull << ncb); c++) {
      const u64 v = c & ((1ull << tA) - 1), kc = c >> tA;
      if (v >= (u64)s.V) continue;
      E e = AV[v];
      for (int i = 0; i < t; i++) e = e * (((kc >> i) & 1) ? rs[i] : E::one() - rs[i]);
      Ec[c] = e;
      E b = E::zero();
      for (int hi = 0; hi < o; hi++)
        for (u64 lo = 0; lo < PL.size(); lo++) {
          const u64 k = kc + (lo << t) + ((u64)hi << z);
          b = b + HV[hi] * PL[lo] * E::from(from_i64(Wh[(size_t)v * d + k]));
        }
      Bc[c] = b;
    }
  }
  // Step 11. cm: the segment's entry of the index-mask message.
  template <class Ch> void merge_run(Ch &ch, const E *mask, u32 &slot, const E &cm) {
    x = ch.batch(1);
    PscJob j;
    std::vector<E> Ec, Bc;
    weight_factors(Ec, Bc);
    const int n = W.s.lkW;
    for (int l = 0; l < n; l++) j.vars.push_back({Weight<E>::plain(3), (unsigned)(n - 1 - l)});
    j.maps = {std::move(Ec), std::move(Bc)};
    j.summand = {{E::one(), {0, 1}}};
    j.ends = [](const std::vector<E> &, const std::vector<E> &b) { return std::vector<E>{b[0] * b[1]}; };
    j.at_end = [](const std::vector<E> &v) { return v[0]; };
    j.end_masks = {cm};
    j.shown = 1, j.degree_y = 1, j.start_value = s8.values[1], j.start_public = s8.end_values[1];
    merge = psc_run(ch, mask, slot, std::move(j));
  }
  // Steps 10 to 12. rows: NCm + Q, or 0 if the frame gives the rows otherwise.
  template <class Ch> void moment2(Ch &ch, const E *mask, u32 &slot, const E &cm, u64 rows) {
    lkZ = small.run_lookup(ch, mask, slot);
    merge_run(ch, mask, slot, cm);
    chi = rows ? ch.batch(rows) : ch.batch_open();
  }
};
}
