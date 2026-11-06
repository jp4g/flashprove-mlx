// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
// The prover's rounds of an attention segment A_g after its first commit point, and the parts of U3
// (docs/format/11-3-attention-segment.md, rows 3 to 15 of "The rounds of the segment", "The parts of
// `U3`"). The verifier's side is verifier/src/vf_attention_check.h. All in tables on the host.
//
// Ch: as psc_run() and lookup_run(), and batch(rows), point_zero(n). attn_own() tests each sum it
// starts from against the tables and throws if they differ, so a wrong table stops the run here.
#pragma once
#include "seg_attn_build.h"
namespace fpmlx {
inline E attn_e(I64 v) { return E::from(from_i64(v)); }
inline std::vector<E> attn_powers(const E &x, size_t n) {
  std::vector<E> p(n, E::one());
  for (size_t i = 1; i < n; i++) p[i] = p[i - 1] * x;
  return p;
}
inline std::vector<E> attn_slice(const std::vector<E> &v, size_t first, size_t n) { return std::vector<E>(v.begin() + first, v.begin() + first + n); }
// A piece with its weight wp.
struct AttnPW {
  u64 lam = 0, a = 0;
  E wp = E::zero();
};
// A unit of KFD: an inner point and what its sumcheck leaves.
struct AttnUnit {
  size_t call = 0;
  int seg = 0;
  std::vector<AttnPW> pw;
  std::vector<E> Z, zg, zP, zs, zI, zi, zj, WI;
  std::vector<E> hidden;  // PV_0 .. PV_4, MV, SV
  PscResult sc;
  u64 tM = 0, tQ = 0, tK = 0;
};
// A unit of KR.
struct AttnPlanes {
  size_t call = 0;
  int seg = 0;
  std::vector<AttnPW> pw;
  std::vector<E> Z, hidden;
};
// What the run has fixed: challenges, true values, and the parts' first entries in U3.
struct AttnRun {
  LookupOut g1;
  std::vector<AttnPlanes> kr;
  std::vector<AttnUnit> kf;
  std::vector<AttnPW> pwW, pwX, pwH;
  std::vector<E> ZW, ZX, ZH, ptM, ptD;
  E ACC = E::zero(), ZO = E::zero(), ZC = E::zero(), RH = E::zero(), RHP = E::zero(), RH2 = E::zero(), RHP2 = E::zero(), gam = E::zero();
  std::vector<E> IB, M6, om, om4, om5, tg, ti, tk, tbg, tbi;
  PscResult scE, vpm, vpd, scF, scd, scH;
  u64 tA = 0, tVM = 0, tVD = 0, hW = 0, hX = 0, hE = 0, hT = 0, hPi = 0, hR2 = 0, tO = 0, qO = 0, qVM = 0, qVD = 0, qF = 0, qKx = 0, tB[2] = {0, 0}, tP = 0, nW = 0;
};
inline E attn_sum3(const std::vector<std::vector<E>> &AB, const std::vector<E> &W, const std::vector<E> &coef) {
  E s = E::zero();
  for (size_t b = 0; b < coef.size(); b++) {
    E t = E::zero();
    for (size_t x = 0; x < W.size(); x++) t = t + W[x] * AB[2 * b][x] * AB[2 * b + 1][x];
    s = s + coef[b] * t;
  }
  return s;
}
// SC3: start = the sum over b of coef_b, over the cube of W A_b B_b; AB = A_0, B_0, A_1, ..
template <class Ch> PscResult attn_sc3(Ch &ch, const E *mask, u32 &slot, unsigned nv, std::vector<std::vector<E>> AB, std::vector<E> W, const std::vector<E> &coef, const E &start, const E &start_public, const char *name) {
  if (self_checks() && !(attn_sum3(AB, W, coef) == start)) throw std::runtime_error(std::string("attention: the sum of ") + name + " is not its start");
  PscJob j;
  const unsigned nb = (unsigned)coef.size();
  for (unsigned l = 0; l < nv; l++) j.vars.push_back({Weight<E>::plain(4), nv - 1 - l});
  j.maps = std::move(AB);
  j.maps.push_back(std::move(W));
  for (unsigned b = 0; b < nb; b++) j.summand.push_back({coef[b], {2 * b, 2 * b + 1, 2 * nb}});
  j.shown = 2 * nb, j.degree_y = 2, j.start_value = start, j.start_public = start_public;
  return psc_run(ch, mask, slot, std::move(j));
}
// Rows 4 to 15, after the lookup of row 3. Fills sm.mats and sm.parts; U3 is then sm.u3().
template <class Ch> void attn_own(Ch &ch, const E *mask, u32 &slot, const AttnGroup &A, FFGroup &ff, Smallness &sm, AttnRun &R) {
  const AShape &s = A.s;
  const int lt = s.lt, lB = s.lB, lgk = s.lgk, ldh = s.ldh, lnT = s.lnT;
  const u64 T = s.T, Bt = s.Bt, nT = s.nT, hP = s.hP, G = s.G, HA = s.HA, H = s.H, dh = s.dh, dq = s.dq, dkv = s.dkv, Lam = s.Lam, NP = s.NP;
  const LookupOut &g1 = R.g1;
  // Sends a hidden value alone; returns what was shown.
  auto hide = [&](const E &v) { const E x = v + mask[slot]; ch.send(x); slot++; return x; };
  auto zk_of = [&](size_t q) {
    const unsigned hi = A.sizes.rK - A.sizes.iota[q];
    std::vector<E> z(g1.zy.begin() + hi, g1.zy.end());
    z.insert(z.end(), g1.zx.begin(), g1.zx.end());
    return z;
  };
  auto pieces = [&](size_t q, int seg) {
    const std::vector<E> zk = zk_of(q);
    std::vector<AttnPW> v;
    for (const APiece &pc : A.ac[q].pcs)
      if (pc.seg == seg) {
        AttnPW x;
        x.lam = pc.lam, x.a = pc.a, x.wp = eq_at(attn_slice(zk, 0, zk.size() - pc.b), pc.off >> pc.b);
        v.push_back(x);
      }
    std::stable_sort(v.begin(), v.end(), [](const AttnPW &x, const AttnPW &y) { return x.lam != y.lam ? x.lam < y.lam : x.a < y.a; });
    return v;
  };
  auto inner = [&](size_t q, int seg) { const std::vector<E> zk = zk_of(q); return attn_slice(zk, zk.size() - s.segB(seg), s.segB(seg)); };
  auto plane_at = [&](const std::vector<AttnPW> &pw, bool isD, u64 x, const std::vector<E> &Z) {
    const std::vector<E> e = eq_all(Z);
    E r = E::zero();
    for (const AttnPW &w : pw) {
      E t = E::zero();
      for (u64 n = 0; n < e.size(); n++) if (const int32_t c = A.plane(x, w.lam, w.a, isD, n)) t = t + e[n] * attn_e(c);
      r = r + w.wp * t;
    }
    return r;
  };
  // Row 4: the units of KR.
  for (size_t q = 0; q < A.ac.size(); q++)
    if (A.ac[q].kind == "KR")
      for (int seg : {AS_RM, AS_RD}) {
        if (!A.ac[q].holds(seg)) continue;
        AttnPlanes p;
        p.call = q, p.seg = seg, p.Z = inner(q, seg), p.pw = pieces(q, seg);
        for (u64 x = 0; x < 3; x++) { p.hidden.push_back(plane_at(p.pw, seg == AS_RD, x, p.Z)); hide(p.hidden.back()); }
        R.kr.push_back(std::move(p));
      }
  // The tile's rows of q and of kx at a point of lB coordinates: [(hk or head) nT + tile] dh + ch.
  auto q_rows = [&](u64 lam, u64 a, const std::vector<E> &el) {
    std::vector<E> o(G * nT * dh, E::zero());
    for (u64 hk = 0; hk < G; hk++)
      for (u64 I = 0; I < nT; I++)
        for (u64 x = 0; x < Bt; x++)
          for (u64 c = 0; c < dh; c++) o[(hk * nT + I) * dh + c] = o[(hk * nT + I) * dh + c] + el[x] * attn_e(A.Q[lam][(I * Bt + x) * dq + (hk * HA + a) * dh + c]);
    return o;
  };
  auto k_rows = [&](u64 lam, const std::vector<E> &el) {
    std::vector<E> o(G * nT * dh, E::zero());
    for (u64 hk = 0; hk < G; hk++)
      for (u64 J = 0; J < nT; J++)
        for (u64 x = 0; x < Bt; x++)
          for (u64 c = 0; c < dh; c++) o[(hk * nT + J) * dh + c] = o[(hk * nT + J) * dh + c] + el[x] * attn_e(A.K[lam][(J * Bt + x) * dkv + hk * dh + c]);
    return o;
  };
  // Row 4: the units of KFD.
  for (size_t q = 0; q < A.ac.size(); q++)
    if (A.ac[q].kind == "KFD")
      for (int seg : {AS_FM, AS_FD}) {
        if (!A.ac[q].holds(seg)) continue;
        AttnUnit k;
        const bool isD = seg == AS_FD;
        k.call = q, k.seg = seg, k.pw = pieces(q, seg), k.Z = inner(q, seg);
        size_t at = 0;
        auto cut = [&](size_t n) { auto v = attn_slice(k.Z, at, n); at += n; return v; };
        k.zg = cut(lgk);
        if (isD) k.zI = cut(lnT);
        else k.zP = cut(lnT - 1), k.zs = cut(lnT);
        k.zi = cut(lB), k.zj = cut(lB);
        const std::vector<E> eZ = eq_all(k.Z);
        for (u64 x = 0; x < 5; x++) k.hidden.push_back(plane_at(k.pw, isD, x, k.Z));
        E MV = E::zero(), SV = E::zero();
        for (const AttnPW &w : k.pw)
          for (u64 hk = 0; hk < G; hk++) {
            const u64 bi = w.lam * H + hk * HA + w.a;
            E mv = E::zero(), sv = E::zero();
            auto one = [&](u64 n, u64 i, u64 j) { mv = mv + eZ[n] * attn_e(A.Mx[bi][i]), sv = sv + eZ[n] * attn_e(A.sc[bi][i * T + j]); };
            if (isD) {
              for (u64 I = 0; I < nT; I++)
                for (u64 il = 0; il < Bt; il++)
                  for (u64 jl = 0; jl < Bt; jl++) one(((hk * nT + I) * Bt + il) * Bt + jl, I * Bt + il, I * Bt + jl);
            } else {
              for (u64 P = 0; P < hP; P++)
                for (u64 sg = 0; sg + 1 < nT; sg++) {
                  u64 I, J;
                  s.cover(P, sg, I, J);
                  for (u64 il = 0; il < Bt; il++)
                    for (u64 jl = 0; jl < Bt; jl++) one((((hk * hP + P) * nT + sg) * Bt + il) * Bt + jl, I * Bt + il, J * Bt + jl);
                }
            }
            MV = MV + w.wp * mv, SV = SV + w.wp * sv;
          }
        k.hidden.push_back(MV), k.hidden.push_back(SV);
        E SVs = E::zero();
        for (const E &v : k.hidden) SVs = hide(v);
        // WI at the inner point.
        k.WI.assign(nT, E::zero());
        if (isD) k.WI = eq_all(k.zI);
        else {
          const std::vector<E> eP = eq_all(k.zP), es = eq_all(k.zs);
          for (u64 P = 0; P < hP; P++)
            for (u64 sg = 0; sg + 1 < nT; sg++) {
              u64 I, J;
              s.cover(P, sg, I, J);
              k.WI[I] = k.WI[I] + eP[P] * es[sg];
            }
        }
        // The sumcheck's tables.
        const unsigned nv = isD ? lgk + lnT + ldh : lgk + 2 * lnT - 1 + ldh;
        const u64 N = 1ull << nv;
        std::vector<std::vector<E>> AB;
        std::vector<E> W(N, E::zero()), coef;
        const std::vector<E> eg = eq_all(k.zg), ei = eq_all(k.zi), ej = eq_all(k.zj);
        const std::vector<E> eI = isD ? eq_all(k.zI) : std::vector<E>(), eP = isD ? std::vector<E>() : eq_all(k.zP), es = isD ? std::vector<E>() : eq_all(k.zs);
        for (const AttnPW &w : k.pw) {
          const std::vector<E> qa = q_rows(w.lam, w.a, ei), ka = k_rows(w.lam, ej);
          std::vector<E> Am(N, E::zero()), Bm(N, E::zero());
          for (u64 hk = 0; hk < G; hk++) {
            if (isD) {
              for (u64 I = 0; I < nT; I++)
                for (u64 c = 0; c < dh; c++) {
                  const u64 x = (hk * nT + I) * dh + c;
                  Am[x] = qa[x], Bm[x] = ka[x], W[x] = eg[hk] * eI[I];
                }
            } else {
              for (u64 P = 0; P < hP; P++)
                for (u64 sg = 0; sg + 1 < nT; sg++) {
                  u64 I, J;
                  s.cover(P, sg, I, J);
                  for (u64 c = 0; c < dh; c++) {
                    const u64 x = ((hk * hP + P) * nT + sg) * dh + c;
                    Am[x] = qa[(hk * nT + I) * dh + c], Bm[x] = ka[(hk * nT + J) * dh + c], W[x] = eg[hk] * eP[P] * es[sg];
                  }
                }
            }
          }
          AB.push_back(std::move(Am)), AB.push_back(std::move(Bm)), coef.push_back(w.wp);
        }
        k.sc = attn_sc3(ch, mask, slot, nv, std::move(AB), std::move(W), coef, SV, SVs, isD ? "K (FD)" : "K (FM)");
        R.kf.push_back(std::move(k));
      }
  // Row 5.
  size_t cWL = 0, cIX = 0, cWH = 0;
  for (size_t q = 0; q < A.ac.size(); q++) {
    if (A.ac[q].holds(AS_WL)) cWL = q;
    if (A.ac[q].holds(AS_IX)) cIX = q;
    if (A.ac[q].holds(AS_WH)) cWH = q;
  }
  R.pwW = pieces(cWL, AS_WL), R.pwX = pieces(cIX, AS_IX), R.pwH = pieces(cWH, AS_WH);
  R.ZW = inner(cWL, AS_WL), R.ZX = inner(cIX, AS_IX), R.ZH = inner(cWH, AS_WH);
  const std::vector<E> zgW = attn_slice(R.ZW, 1, lgk), ziW = attn_slice(R.ZW, 1 + lgk, lt), zcW = attn_slice(R.ZW, 1 + lgk + lt, ldh);
  const std::vector<E> zgX = attn_slice(R.ZX, 0, lgk), ziX = attn_slice(R.ZX, lgk, lt);
  const std::vector<E> zgH = attn_slice(R.ZH, 0, lgk), ziH = attn_slice(R.ZH, lgk, lt), zcH = attn_slice(R.ZH, lgk + lt, ldh);
  const std::vector<E> zihi = attn_slice(ziW, 0, lnT), zilo = attn_slice(ziW, lnT, lB);
  const std::vector<E> egW = eq_all(zgW), eiW = eq_all(ziW), ecW = eq_all(zcW), egX = eq_all(zgX), eiX = eq_all(ziX), egH = eq_all(zgH), eiH = eq_all(ziH), ecH = eq_all(zcH);
  for (const AttnPW &w : R.pwW)
    for (u64 hk = 0; hk < G; hk++)
      for (u64 i = 0; i < T; i++) {
        const u64 h = hk * HA + w.a, bi = w.lam * H + h;
        const E f = w.wp * egW[hk] * eiW[i];
        R.ZC = R.ZC + f * attn_e(A.Zd[bi][i]);
        for (u64 c = 0; c < dh; c++) {
          const E t = f * ecW[c];
          R.ACC = R.ACC + t * attn_e(A.acc[bi][i * dh + c]);
          R.ZO = R.ZO + t * attn_e(A.Zd[bi][i] * A.O[w.lam][i * dq + h * dh + c]);
          R.RH = R.RH + t * attn_e(A.rh(w.lam, h, i, c)), R.RHP = R.RHP + t * attn_e(A.rhp(w.lam, h, i, c));
        }
      }
  R.IB.assign(lt, E::zero());
  for (const AttnPW &w : R.pwX)
    for (u64 hk = 0; hk < G; hk++)
      for (u64 i = 0; i < T; i++)
        for (int k = 0; k < lt; k++)
          if (A.ibit(w.lam, hk * HA + w.a, i, k)) R.IB[k] = R.IB[k] + w.wp * egX[hk] * eiX[i];
  for (const AttnPW &w : R.pwH)
    for (u64 hk = 0; hk < G; hk++)
      for (u64 i = 0; i < T; i++)
        for (u64 c = 0; c < dh; c++) {
          const E t = w.wp * egH[hk] * eiH[i] * ecH[c];
          R.RH2 = R.RH2 + t * attn_e(A.rh(w.lam, hk * HA + w.a, i, c)), R.RHP2 = R.RHP2 + t * attn_e(A.rhp(w.lam, hk * HA + w.a, i, c));
        }
  hide(R.ACC);
  const E ZOs = hide(R.ZO);
  hide(R.ZC), hide(R.RH), hide(R.RHP);
  for (const E &v : R.IB) hide(v);
  hide(R.RH2), hide(R.RHP2);
  std::vector<E> coefW;
  for (const AttnPW &w : R.pwW) coefW.push_back(w.wp);
  {
    const unsigned nv = lgk + lt;
    std::vector<std::vector<E>> AB;
    std::vector<E> W(1ull << nv);
    for (u64 hk = 0; hk < G; hk++) for (u64 i = 0; i < T; i++) W[(hk << lt) + i] = egW[hk] * eiW[i];
    for (const AttnPW &w : R.pwW) {
      std::vector<E> Am(1ull << nv), Bm(1ull << nv, E::zero());
      for (u64 hk = 0; hk < G; hk++)
        for (u64 i = 0; i < T; i++) {
          const u64 h = hk * HA + w.a, x = (hk << lt) + i;
          Am[x] = attn_e(A.Zd[w.lam * H + h][i]);
          for (u64 c = 0; c < dh; c++) Bm[x] = Bm[x] + ecW[c] * attn_e(A.O[w.lam][i * dq + h * dh + c]);
        }
      AB.push_back(std::move(Am)), AB.push_back(std::move(Bm));
    }
    R.scE = attn_sc3(ch, mask, slot, nv, std::move(AB), std::move(W), coefW, R.ZO, ZOs, "SE");
  }
  // Rows 6 to 8.
  R.gam = ch.batch(2);
  const std::vector<E> eIh = eq_all(zihi), eil = eq_all(zilo);
  // v at zcW: [lam][token * G + hk].
  std::vector<std::vector<E>> vz(Lam, std::vector<E>(T * G, E::zero()));
  for (u64 lam = 0; lam < Lam; lam++)
    for (u64 j = 0; j < T; j++)
      for (u64 hk = 0; hk < G; hk++)
        for (u64 c = 0; c < dh; c++) vz[lam][j * G + hk] = vz[lam][j * G + hk] + ecW[c] * attn_e(A.Vv[lam][j * dkv + hk * dh + c]);
  const unsigned nvM = lgk + 2 * lnT - 1 + lB, nvD = lgk + lnT + lB;
  std::vector<std::vector<E>> ABm, ABd;
  std::vector<E> Wm(1ull << nvM, E::zero()), Wd(1ull << nvD, E::zero());
  for (const AttnPW &w : R.pwW) {
    std::vector<E> Am(1ull << nvM, E::zero()), Bm(1ull << nvM, R.gam), Ad(1ull << nvD, E::zero()), Bd(1ull << nvD, R.gam);
    for (u64 hk = 0; hk < G; hk++) {
      for (u64 P = 0; P < hP; P++)
        for (u64 sg = 0; sg < nT; sg++) {
          u64 I = 0, J = 0;
          const bool tile = sg + 1 < nT;
          if (tile) s.cover(P, sg, I, J);
          for (u64 jl = 0; jl < Bt; jl++) {
            const u64 x = ((hk * hP + P) * nT + sg) * Bt + jl;
            for (u64 il = 0; il < Bt; il++) Am[x] = Am[x] + eil[il] * attn_e(A.ew(w.lam, w.a, false, (((hk * hP + P) * nT + sg) * Bt + il) * Bt + jl));
            if (tile) Bm[x] = R.gam + vz[w.lam][(J * Bt + jl) * G + hk], Wm[x] = egW[hk] * eIh[I];
          }
        }
      for (u64 I = 0; I < nT; I++)
        for (u64 jl = 0; jl < Bt; jl++) {
          const u64 x = (hk * nT + I) * Bt + jl;
          for (u64 il = 0; il < Bt; il++) Ad[x] = Ad[x] + eil[il] * attn_e(A.ew(w.lam, w.a, true, ((hk * nT + I) * Bt + il) * Bt + jl));
          Bd[x] = R.gam + vz[w.lam][(I * Bt + jl) * G + hk], Wd[x] = egW[hk] * eIh[I];
        }
    }
    ABm.push_back(std::move(Am)), ABm.push_back(std::move(Bm)), ABd.push_back(std::move(Ad)), ABd.push_back(std::move(Bd));
  }
  const E CL0 = attn_sum3(ABm, Wm, coefW), CL1 = attn_sum3(ABd, Wd, coefW);
  if (!(CL0 + CL1 == R.ACC + R.gam * R.ZC)) throw std::runtime_error("attention: the value product is not ACC + gam ZC");
  const E CL0s = hide(CL0), CL1s = hide(CL1);
  R.vpm = attn_sc3(ch, mask, slot, nvM, std::move(ABm), std::move(Wm), coefW, CL0, CL0s, "VM");
  R.vpd = attn_sc3(ch, mask, slot, nvD, std::move(ABd), std::move(Wd), coefW, CL1, CL1s, "VD");
  // Rows 9 to 11.
  const std::vector<E> pts = ch.point_zero(2 * lgk + 2 * lt + 4);
  R.tg = attn_slice(pts, 0, lgk), R.ti = attn_slice(pts, lgk, lt), R.tk = attn_slice(pts, lgk + lt, 4), R.tbg = attn_slice(pts, lgk + lt + 4, lgk), R.tbi = attn_slice(pts, 2 * lgk + lt + 4, lt);
  const std::vector<E> etg = eq_all(R.tg), eti = eq_all(R.ti);
  std::vector<E> M6s;
  for (u64 b = 0; b < NP; b++) {
    const u64 lam = b / HA, a = b % HA;
    E m6 = E::zero();
    for (u64 hk = 0; hk < G; hk++)
      for (u64 i = 0; i < T; i++) m6 = m6 + etg[hk] * eti[i] * attn_e(A.Mx[lam * H + hk * HA + a][i]);
    R.M6.push_back(m6);
  }
  for (const E &v : R.M6) M6s.push_back(hide(v));
  R.om = attn_powers(ch.batch(NP), NP);
  std::vector<E> ri, rg, rc;
  {
    const unsigned nv = lt + lgk + ldh;
    std::vector<std::vector<E>> AB;
    std::vector<E> W(1ull << nv);
    E st = E::zero(), stp = E::zero();
    for (u64 b = 0; b < NP; b++) st = st + R.om[b] * R.M6[b], stp = stp + R.om[b] * M6s[b];
    for (u64 i = 0; i < T; i++) for (u64 hk = 0; hk < G; hk++) for (u64 c = 0; c < dh; c++) W[(i * G + hk) * dh + c] = etg[hk] * eti[i];
    for (u64 b = 0; b < NP; b++) {
      const u64 lam = b / HA, a = b % HA;
      std::vector<E> Am(1ull << nv), Bm(1ull << nv);
      for (u64 i = 0; i < T; i++)
        for (u64 hk = 0; hk < G; hk++) {
          const u64 h = hk * HA + a, j = (u64)A.ib[lam * H + h][i];
          for (u64 c = 0; c < dh; c++) {
            const u64 x = (i * G + hk) * dh + c;
            Am[x] = attn_e(A.Q[lam][i * dq + h * dh + c]), Bm[x] = attn_e(A.K[lam][j * dkv + hk * dh + c]);
          }
        }
      AB.push_back(std::move(Am)), AB.push_back(std::move(Bm));
    }
    R.scF = attn_sc3(ch, mask, slot, nv, std::move(AB), std::move(W), R.om, st, stp, "SF");
    ri = attn_slice(R.scF.challenges, 0, lt), rg = attn_slice(R.scF.challenges, lt, lgk), rc = attn_slice(R.scF.challenges, lt + lgk, ldh);
  }
  // Rows 12, 13: SD. Maps: KC_lam, IBP_(b,k) (shown); then eq(riF, i) eq(rgF, hk), y_k, and the
  // factors eq1(ibit_k, y_k), which the rounds multiply. The last are multilinear, as ibit_k and y_k
  // have no variable in common; at the end the summand is on the shown values, so at_end.
  R.om4 = attn_powers(ch.batch(NP), NP);
  {
    const unsigned nv = 2 * lt + lgk, shown = (unsigned)(Lam + NP * lt), iEq = shown, iY = iEq + 1, iG = iY + lt;
    const u64 N = 1ull << nv;
    PscJob j;
    j.maps.assign(iG + NP * lt, std::vector<E>(N, E::zero()));
    const std::vector<E> eri = eq_all(ri), erg = eq_all(rg), erc = eq_all(rc);
    auto ix = [&](u64 jj, u64 hk, u64 i) { return (jj * G + hk) * T + i; };
    for (u64 jj = 0; jj < T; jj++)
      for (u64 hk = 0; hk < G; hk++) {
        std::vector<E> kc(Lam, E::zero());
        for (u64 lam = 0; lam < Lam; lam++)
          for (u64 c = 0; c < dh; c++) kc[lam] = kc[lam] + erc[c] * attn_e(A.K[lam][jj * dkv + hk * dh + c]);
        for (u64 i = 0; i < T; i++) {
          const u64 x = ix(jj, hk, i);
          for (u64 lam = 0; lam < Lam; lam++) j.maps[lam][x] = kc[lam];
          j.maps[iEq][x] = eri[i] * erg[hk];
          for (int k = 0; k < lt; k++) j.maps[iY + k][x] = attn_e((jj >> k) & 1);
          for (u64 b = 0; b < NP; b++)
            for (int k = 0; k < lt; k++) {
              const I64 bit = A.ibit(b / HA, hk * HA + b % HA, i, k);
              j.maps[Lam + b * lt + k][x] = attn_e(bit);
              j.maps[iG + b * lt + k][x] = attn_e(bit == (I64)((jj >> k) & 1));
            }
        }
      }
    for (u64 b = 0; b < NP; b++) {
      Monomial m;
      m.coeff = R.om4[b];
      m.maps = {iEq, (unsigned)(b / HA)};
      for (int k = 0; k < lt; k++) m.maps.push_back(iG + (unsigned)(b * lt + k));
      j.summand.push_back(std::move(m));
    }
    for (unsigned l = 0; l < nv; l++) {
      const unsigned d = l < (unsigned)lt ? 3 : l < (unsigned)(lt + lgk) ? lt + 3 : lt + 2;
      j.vars.push_back({Weight<E>::plain(d), nv - 1 - l});
    }
    j.shown = shown, j.degree_y = lt + 1;
    for (u64 b = 0; b < NP; b++)
      j.start_value = j.start_value + R.om4[b] * R.scF.values[2 * b + 1], j.start_public = j.start_public + R.om4[b] * R.scF.end_values[2 * b + 1];
    const u64 NPc = NP, HAc = HA, Lamc = Lam;
    const std::vector<E> om4 = R.om4;
    j.at_end = [=](const std::vector<E> &v) {
      E t = E::zero();
      for (u64 b = 0; b < NPc; b++) {
        E p = om4[b] * v[b / HAc];
        for (int k = 0; k < lt; k++) { const E y = v[iY + k]; p = p * (v[Lamc + b * lt + k] * (y + y - E::one()) + (E::one() - y)); }
        t = t + p;
      }
      return v[iEq] * t;
    };
    if (self_checks()) {
      E sum = E::zero();
      for (u64 x = 0; x < N; x++) {
        E t = E::zero();
        for (const Monomial &m : j.summand) { E p = m.coeff; for (unsigned q : m.maps) p = p * j.maps[q][x]; t = t + p; }
        sum = sum + t;
      }
      if (!(sum == j.start_value)) throw std::runtime_error("attention: the sum of SD is not its start");
    }
    R.scd = psc_run(ch, mask, slot, std::move(j));
  }
  const std::vector<E> pj = attn_slice(R.scd.challenges, 0, lt), pi = attn_slice(R.scd.challenges, lt + lgk, lt);
  // Rows 14, 15: HB.
  R.om5 = attn_powers(ch.batch(NP), NP);
  std::vector<E> ri2;
  {
    const unsigned nv = lt + lgk + 4;
    std::vector<std::vector<E>> AB;
    std::vector<E> W(1ull << nv);
    const std::vector<E> ei = eq_all(R.tbi), eg = eq_all(R.tbg), ek = eq_all(R.tk);
    for (u64 i = 0; i < T; i++) for (u64 hk = 0; hk < G; hk++) for (u64 k = 0; k < 16; k++) W[(i * G + hk) * 16 + k] = ei[i] * eg[hk] * ek[k];
    for (u64 b = 0; b < NP; b++) {
      std::vector<E> Am(1ull << nv, E::zero()), Bm(1ull << nv);
      for (u64 i = 0; i < T; i++)
        for (u64 hk = 0; hk < G; hk++)
          for (u64 k = 0; k < 16; k++) {
            const u64 x = (i * G + hk) * 16 + k;
            if ((int)k < lt) Am[x] = attn_e(A.ibit(b / HA, hk * HA + b % HA, i, (int)k));
            Bm[x] = Am[x] - E::one();
          }
      AB.push_back(std::move(Am)), AB.push_back(std::move(Bm));
    }
    R.scH = attn_sc3(ch, mask, slot, nv, std::move(AB), std::move(W), R.om5, E::zero(), E::zero(), "HB");
    ri2 = attn_slice(R.scH.challenges, 0, lt);
  }
  // The parts of U3.
  sm.mats.clear(), sm.parts.clear();
  for (const CellMatrix2 &M : A.as) sm.mats.push_back(&M);
  const unsigned iAH = (unsigned)s.NAs, iAO = iAH + 1, iBA = iAH + 2, iQA = iAH + 3;
  sm.mats.push_back(&A.ah.part[0]), sm.mats.push_back(&A.ao.part[0]), sm.mats.push_back(&A.bA), sm.mats.push_back(&ff.qa.part[0]);
  sm.tau = (unsigned)s.tau, sm.lidb = (unsigned)s.lidb;
  u64 p = 0;
  auto add = [&](unsigned M, std::vector<E> phi) {
    const u64 at = p;
    p += sm.mats[M]->k * (sm.mats[M]->m / phi.size());
    sm.parts.push_back({M, std::move(phi)});
    return at;
  };
  auto add_as = [&](const std::vector<E> &phi) { const u64 at = p; for (u64 q = 0; q < s.NAs; q++) add((unsigned)q, phi); return at; };
  auto last_eq = [&](const std::vector<E> &z, size_t n) { return eq_all(attn_slice(z, z.size() - n, n)); };
  auto tok = [&](const std::vector<E> &hi, const std::vector<E> &lo) {
    const std::vector<E> el = eq_all(lo);
    std::vector<E> v(T);
    for (u64 I = 0; I < nT; I++) for (u64 x = 0; x < Bt; x++) v[I * Bt + x] = hi[I] * el[x];
    return v;
  };
  auto lng = [&](const std::vector<E> &zc, const std::vector<E> &tv) {
    const std::vector<E> lo = last_eq(zc, s.ls);
    std::vector<E> v(lo.size() * tv.size());
    for (size_t c = 0; c < lo.size(); c++) for (size_t t = 0; t < tv.size(); t++) v[c * tv.size() + t] = lo[c] * tv[t];
    return v;
  };
  const std::vector<E> &rM = R.vpm.challenges, &rD = R.vpd.challenges;
  R.ptM = attn_slice(rM, 0, lgk + 2 * lnT - 1), R.ptD = attn_slice(rD, 0, lgk + lnT);
  R.ptM.insert(R.ptM.end(), zilo.begin(), zilo.end()), R.ptD.insert(R.ptD.end(), zilo.begin(), zilo.end());
  { const auto t = attn_slice(rM, rM.size() - lB, lB); R.ptM.insert(R.ptM.end(), t.begin(), t.end()); }
  { const auto t = attn_slice(rD, rD.size() - lB, lB); R.ptD.insert(R.ptD.end(), t.begin(), t.end()); }
  R.tA = add_as(last_eq(g1.zx, s.lmS));
  R.tVM = add_as(last_eq(R.ptM, s.lmS)), R.tVD = add_as(last_eq(R.ptD, s.lmS));
  for (AttnUnit &k : R.kf) k.tM = add(iAH, tok(k.WI, k.zi));
  R.hW = add(iAH, eiW), R.hX = add(iAH, eiX), R.hE = add(iAH, eq_all(attn_slice(R.scE.challenges, lgk, lt))), R.hT = add(iAH, eti), R.hPi = add(iAH, eq_all(pi)), R.hR2 = add(iAH, eq_all(ri2));
  { std::vector<E> ek = eiW; R.tO = add(iAO, lng(zcW, ek)); }
  // WI and WJ at the point (rP, rs) of r, from its coordinate lgk on.
  auto wij = [&](const std::vector<E> &r, std::vector<E> &WI, std::vector<E> &WJ) {
    const std::vector<E> eP = eq_all(attn_slice(r, lgk, lnT - 1)), es = eq_all(attn_slice(r, lgk + lnT - 1, lnT));
    WI.assign(nT, E::zero()), WJ.assign(nT, E::zero());
    for (u64 P = 0; P < hP; P++)
      for (u64 sg = 0; sg + 1 < nT; sg++) {
        u64 I, J;
        s.cover(P, sg, I, J);
        WI[I] = WI[I] + eP[P] * es[sg], WJ[J] = WJ[J] + eP[P] * es[sg];
      }
  };
  for (AttnUnit &k : R.kf) {
    const std::vector<E> &r = k.sc.challenges;
    const std::vector<E> rcK = attn_slice(r, r.size() - ldh, ldh);
    std::vector<E> WIr, WJr;
    if (k.seg == AS_FM) wij(r, WIr, WJr);
    else WIr = WJr = eq_all(attn_slice(r, lgk, lnT));
    k.tQ = add(iQA, lng(rcK, tok(WIr, k.zi)));
    k.tK = add(iQA, lng(rcK, tok(WJr, k.zj)));
  }
  R.qO = add(iQA, lng(zcW, eq_all(attn_slice(R.scE.challenges, lgk, lt))));
  {
    std::vector<E> WI, WJ;
    wij(rM, WI, WJ);
    R.qVM = add(iQA, lng(zcW, tok(WJ, attn_slice(rM, rM.size() - lB, lB))));
  }
  R.qVD = add(iQA, lng(zcW, tok(eq_all(attn_slice(rD, lgk, lnT)), attn_slice(rD, rD.size() - lB, lB))));
  R.qF = add(iQA, lng(rc, eq_all(ri))), R.qKx = add(iQA, lng(rc, eq_all(pj)));
  const unsigned kT = A.sizes.kT, cK = A.sizes.cK, lm = s.lmbA;
  for (int e = 0; e < 2; e++) {
    std::vector<E> z = attn_slice(g1.zb[e], kT + 5 - lm, lm - 5);
    const auto t = attn_slice(g1.zb[e], kT + cK, 5);
    z.insert(z.end(), t.begin(), t.end());
    R.tB[e] = add(iBA, eq_all(z));
  }
  {
    std::vector<E> ph(1ull << lm);
    for (u64 r = 0; r < ph.size(); r++) ph[r] = attn_e(1ll << (r & 31));
    R.tP = add(iBA, std::move(ph));
  }
  R.nW = p;
}
// Rows 3 to 15.
template <class Ch> void attn_rounds(Ch &ch, const E *mask, u32 &slot, const AttnGroup &A, FFGroup &ff, Smallness &sm, AttnRun &R) {
  R.g1 = lookup_run(ch, mask, slot, A.calls, A.sizes, A.mult);
  attn_own(ch, mask, slot, A, ff, sm, R);
}
// The segment's own arrays, in the order of its list: as (parts), ah, ao, bA.
inline std::vector<ArrayData> attn_arrays(const AttnGroup &A) {
  std::vector<ArrayData> v;
  for (const CellMatrix2 &M : A.as) v.push_back(array_of(M));
  v.push_back(array_of(A.ah.part[0])), v.push_back(array_of(A.ao.part[0])), v.push_back(array_of(A.bA));
  return v;
}
}
