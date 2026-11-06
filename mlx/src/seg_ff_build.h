// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
// Fills the matrices of the feed-forward segment of group g, and forms the keys of its lookup.
#pragma once
#include "seg_ff_witness.h"
#include "ff_witness_gpu.h"
namespace fpmlx {
struct FFGroup {
  FFShape s;
  int g = 0;
  FeatMat af, ad, ar, qa, xbI, xbO;
  std::map<std::string, Tensor2> tf; // tensors by name
  std::vector<LookupCall> calls;
  std::vector<std::map<std::string, SegPlace>> place; // by call
  std::vector<int> lJ;
  std::vector<std::vector<std::array<std::vector<E>, 1>>> dummy;
  // by lam, n (first or second norm), token
  std::vector<std::vector<NormRow>> rows; // rows[i][t]
};
inline void lay_tensors(FFGroup &G, const std::vector<std::tuple<std::string, int, int, int>> &list) {
  int off = 0;
  for (auto &[name, nl, cnt, J] : list) {
    Tensor2 t;
    t.off = off, t.nl = nl, t.cnt = cnt, t.J = J;
    G.tf[name] = t;
    off += nl * cnt * J;
  }
}
inline long long features(const std::vector<std::tuple<std::string, int, int, int>> &list) {
  long long n = 0;
  for (auto &[name, nl, cnt, J] : list) n += (long long)nl * cnt * J;
  return n;
}
struct PutTo {
  FeatMat &m;
  const Tensor2 &t;
  void put(int ix, int j, int tok, I64 v) { int32_t c[8]; limbs_of(v, t.nl, c); for (int lb = 0; lb < t.nl; lb++) m.cell(t.feat(lb, ix, j), tok) = c[lb]; }
};
inline void ff_fill_gpu(FFGroup &R, const Policy &pol, const ModelShape &m, const Values &V, int g);
inline FFGroup ff_build(const Policy &pol, const ModelShape &m, const Values &V, const CallTables &tabs, int g, bool on_gpu = true) {
  FFGroup R;
  R.s = ff_shape(pol, m);
  R.g = g;
  const FFShape &s = R.s;
  const int Lam = s.Lam, LamM = s.LamM, T = s.T, d = s.d, f = s.f, fe = s.fe, NE = s.NE;
  // The tensors of af, in the order of the table.
  std::vector<std::tuple<std::string, int, int, int>> A, D, Rr, Q;
  if (s.gcl) A.push_back({"g0", 1, Lam, f}), A.push_back({"gc", 1, Lam, f});
  else A.push_back({"gi", 2, Lam, f});
  A.insert(A.end(), {{"sl", 2, Lam, f}, {"up", 2, Lam, f}, {"rg", s.hgD, Lam, f}, {"ru", s.huD, Lam, f}, {"xs", s.XL, 2 * Lam - 1, d}, {"ro", s.ho, Lam, d}, {"rd", s.hdD, Lam, d},
                     {"aw", 2, Lam, d}, {"ra", 2, Lam, d}, {"mw", 2, Lam, d}, {"rm", 2, Lam, d}, {"qkw", 2, Lam, s.nqk}, {"rqk", s.hq, Lam, s.nqk}, {"vw", 2, Lam, s.dkv}, {"rv", s.hv, Lam, s.dkv}});
  if (s.gcl) A.push_back({"gh", 2, Lam, f});
  D = {{"an", 2, Lam, d}, {"mn", 2, Lam, d}};
  if (NE) D.push_back({"xr", 2, LamM, d});
  if (NE) Rr = {{"lr", 2, LamM, NE}, {"rl", s.hr, LamM, NE}, {"sel", 1, LamM, NE}, {"sc", 3, LamM, 1}};
  Rr.push_back({"row", 1, 2 * Lam, s.NR});
  if (s.rcl) Rr.push_back({"ly", 2, LamM, NE}), Rr.push_back({"l3", 1, LamM, NE});
  Q = {{"q", 2, Lam, s.dq}, {"kx", 2, Lam, s.dkv}, {"v", 2, Lam, s.dkv}, {"o", 2, Lam, s.dq}};
  R.af.init(s.NAf, s.lmaf, features(A), s.lt);
  R.ad.init(s.NAd, s.lmad, features(D), s.lt);
  R.ar.init(1, s.lmar, features(Rr), s.lt);
  R.qa.init(1, s.lmq, features(Q), s.lt);
  R.xbI.init(1, s.lmx, (long long)s.XL * d, s.lt);
  R.xbO.init(1, s.lmx, (long long)s.XL * d, s.lt);
  lay_tensors(R, A);
  { FFGroup t; lay_tensors(t, D); for (auto &x : t.tf) R.tf[x.first] = x.second; }
  { FFGroup t; lay_tensors(t, Rr); for (auto &x : t.tf) R.tf[x.first] = x.second; }
  { FFGroup t; lay_tensors(t, Q); for (auto &x : t.tf) R.tf[x.first] = x.second; }
  auto AF = [&](const char *n) { return PutTo{R.af, R.tf.at(n)}; };
  auto AD = [&](const char *n) { return PutTo{R.ad, R.tf.at(n)}; };
  auto AR = [&](const char *n) { return PutTo{R.ar, R.tf.at(n)}; };
  auto QA = [&](const char *n) { return PutTo{R.qa, R.tf.at(n)}; };
  auto L = [&](const char *n, int lam) { return std::string(n) + "." + std::to_string(g * Lam + lam); };
  auto str = [&](int sg, int t, int j) -> I64 { // the stream sg of the group
    const int lam = sg / 2;
    if (sg == 2 * Lam) return (g + 1) * Lam == s.L ? V.get("x_last", t, j) : V.get("x." + std::to_string((g + 1) * Lam), t, j);
    return V.get(L(sg % 2 ? "x_mid" : "x", lam), t, j);
  };
  const auto &ks = m.layers;
  // The calls' segments and their columns.
  auto seg_len = [&](const std::string &n) -> u64 {
    static const std::map<std::string, int> none;
    if (n == "G" || n == "U" || n == "UT" || n == "GH" || n == "UH" || n == "F" || n == "G0" || n == "CG") return f;
    if (n == "QK" || n == "QKH" || n == "CQK") return s.nqk;
    if (n == "N1" || n == "N1H" || n == "N2" || n == "N2H" || n == "O" || n == "D" || n == "XR" || n == "OH" || n == "DH" || n == "CN1" || n == "CN2") return d;
    if (n == "V" || n == "VH" || n == "CV") return s.dkv;
    if (n == "LR" || n == "LT" || n == "DD" || n == "LH" || n == "SEL" || n == "L3" || n == "CLR") return NE;
    if (n == "ROW") return 2 * s.NR;
    if (n == "MR") return 1;
    throw std::runtime_error("no such segment " + n);
  };
  auto routed = [](const std::string &n) { return n == "XR" || n == "LR" || n == "LT" || n == "DD" || n == "LH" || n == "SEL" || n == "MR" || n == "L3" || n == "CLR"; };
  R.calls.resize(pol.call.size()), R.place.resize(pol.call.size()), R.lJ.resize(pol.call.size());
  for (size_t c = 0; c < pol.call.size(); c++) {
    struct Piece { std::string seg; u64 first, size; };
    std::vector<Piece> pieces;
    for (const std::string &n : pol.call[c].segs) {
      const u64 len = seg_len(n);
      u64 first = 0;
      for (int b = 62; b >= 0; b--)
        if ((len >> b) & 1) pieces.push_back({n, first, 1ull << b}), first += 1ull << b;
      R.place[c][n].n = len, R.place[c][n].routed = routed(n);
    }
    std::stable_sort(pieces.begin(), pieces.end(), [](const Piece &a, const Piece &b) { return a.size > b.size; });
    u64 col = 0;
    for (const Piece &p : pieces) {
      R.place[c][p.seg].pieces.push_back({p.first, p.size}), R.place[c][p.seg].col.push_back(col);
      col += p.size;
    }
    int lJ = clog2u(col);
    lJ = std::max(lJ, s.nmin - s.ll - s.lt);
    R.lJ[c] = lJ;
    LookupCall &K = R.calls[c];
    K.n = s.ll + lJ + s.lt;
    const std::string &kind = pol.call[c].kind;
    K.k = kind == "R8" ? 8 : 16;
    if (kind == "FC") K.table = tabs.FC, K.t0 = 32768;
    else if (kind == "FL") K.table = tabs.FL, K.t0 = 32768;
    else if (kind == "FS") K.table = tabs.FS, K.t0 = 32768;
    K.keys.assign(1ull << K.n, 0), K.padding.assign(1ull << K.n, 1);
  }
  if (on_gpu) { ff_fill_gpu(R, pol, m, V, g); return R; }
  auto key_put = [&](const std::string &seg, int lam, u64 x, int t, u64 key) {
    for (size_t c = 0; c < pol.call.size(); c++) {
      auto it = R.place[c].find(seg);
      if (it == R.place[c].end()) continue;
      const u64 idx = (((u64)lam << R.lJ[c]) + it->second.pos(x)) * T + t;
      R.calls[c].keys[idx] = key, R.calls[c].padding[idx] = 0;
      return;
    }
    throw std::runtime_error("no call holds " + seg);
  };
  auto has = [&](const std::string &seg) { for (auto &m2 : R.place) if (m2.count(seg)) return true; return false; };
  auto SK = [&](__int128 A, I64 Q, int k, int h, I64 rh) { return (u64)((((A + ((__int128)1 << (k - 1)) - ((__int128)Q << k)) << lift(k))) - (h ? ((__int128)(rh + ofs_of(h)) << 16) : 0)); };
  auto HK = [&](I64 rh, int k, int h) { const int top = h == 1 ? 24 : 32; return (u64)((rh + ofs_of(h)) << (top - std::max(k, 16))); };
  auto RH = [&](__int128 A, I64 Q, int k, int h) -> I64 { const __int128 rem = A + ((__int128)1 << (k - 1)) - ((__int128)Q << k); return h ? (I64)(rem >> 16) - ofs_of(h) : 0; };
  // Streams.
  for (int t = 0; t < T; t++) {
    for (int j = 0; j < d; j++) {
      PutTo{R.xbI, Tensor2{0, s.XL, 1, d}}.put(0, j, t, str(0, t, j));
      PutTo{R.xbO, Tensor2{0, s.XL, 1, d}}.put(0, j, t, str(2 * Lam, t, j));
      for (int ix = 0; ix < 2 * Lam - 1; ix++) AF("xs").put(ix, j, t, str(ix + 1, t, j));
    }
  }
  // Norms.
  R.rows.assign(2 * Lam, std::vector<NormRow>(T));
  for (int i = 0; i < 2 * Lam; i++) {
    const int lam = i / 2, n = i % 2;
    const LayerScales &k = ks[g * Lam + lam];
    const int M = n ? k.M2 : k.M1, K = n ? k.K2 : k.K1;
    const I64 eps = n ? m.eps2 : m.eps1;
    for (int t = 0; t < T; t++) {
      std::vector<int32_t> lim((size_t)d * s.XL);
      for (int j = 0; j < d; j++) limbs_of(str(i, t, j), s.XL, &lim[(size_t)j * s.XL]);
      NormRow r = norm_row(s.XL, lim, d, eps, M);
      for (int j = 0; j < s.NR; j++) {
        AR("row").m.cell(R.tf.at("row").feat(0, i, j), t) = r.D[j] - 128;
        key_put("ROW", lam, (u64)n * s.NR + j, t, row_mul(j) * r.D[j]);
      }
      for (int j = 0; j < d; j++) {
        const char *wn = n ? "mw" : "aw", *rn = n ? "rm" : "ra", *cn = n ? "mn" : "an";
        const __int128 Aval = (__int128)str(i, t, j) * r.R;
        const I64 wv = V.get(L(n ? "m_w" : "a_w", lam), t, j), cv = V.get(L(n ? "m" : "a", lam), t, j);
        const I64 rh = RH(Aval, wv, K, 2);
        AF(wn).put(lam, j, t, wv), AF(rn).put(lam, j, t, rh), AD(cn).put(lam, j, t, cv);
        key_put(n ? "N2" : "N1", lam, j, t, SK(Aval, wv, K, 2, rh));
        key_put(n ? "N2H" : "N1H", lam, j, t, HK(rh, K, 2));
        key_put(n ? "CN2" : "CN1", lam, j, t, (u64)(wv + 32768) + ((u64)(cv + 2048) << 28));
      }
      R.rows[i][t] = std::move(r);
    }
  }
  for (int lam = 0; lam < Lam; lam++) {
    const LayerScales &k = ks[g * Lam + lam];
    const bool ex = NE && (lam & 1);
    const int lamM = (lam - 1) / 2;
    for (int t = 0; t < T; t++) {
      for (int x = 0; x < s.nqk; x++) {
        const bool isq = x < s.dq;
        const int kk = isq ? k.kq : k.kk;
        const I64 w = isq ? V.get(L("qw", lam), t, x) : V.get(L("kw", lam), t, x - s.dq), cl = isq ? V.get(L("q", lam), t, x) : V.get(L("k", lam), t, x - s.dq);
        const __int128 Aq = isq ? V.get(L("sum_q", lam), t, x) : V.get(L("sum_k", lam), t, x - s.dq);
        const I64 rh = RH(Aq, w, kk, s.hq);
        AF("qkw").put(lam, x, t, w);
        if (s.hq) AF("rqk").put(lam, x, t, rh);
        if (isq) QA("q").put(lam, x, t, cl); else QA("kx").put(lam, x - s.dq, t, cl);
        key_put("QK", lam, x, t, SK(Aq, w, kk, s.hq, rh));
        if (s.hq) key_put("QKH", lam, x, t, HK(rh, kk, s.hq));
        key_put("CQK", lam, x, t, (u64)(w + 32768) + ((u64)(cl + 2048) << 28));
      }
      for (int x = 0; x < s.dkv; x++) {
        const I64 w = V.get(L("vw", lam), t, x), cl = V.get(L("v", lam), t, x);
        const __int128 Av = V.get(L("sum_v", lam), t, x);
        const I64 rh = RH(Av, w, k.kv, s.hv);
        AF("vw").put(lam, x, t, w);
        if (s.hv) AF("rv").put(lam, x, t, rh);
        QA("v").put(lam, x, t, cl);
        key_put("V", lam, x, t, SK(Av, w, k.kv, s.hv, rh));
        if (s.hv) key_put("VH", lam, x, t, HK(rh, k.kv, s.hv));
        key_put("CV", lam, x, t, (u64)(w + 32768) + ((u64)(cl + 2048) << 28));
      }
      for (int x = 0; x < s.dq; x++) QA("o").put(lam, x, t, V.get(L("o", lam), t, x));
      for (int j = 0; j < d; j++) {
        const I64 Q1 = str(2 * lam + 1, t, j) - str(2 * lam, t, j), Q2 = str(2 * lam + 2, t, j) - str(2 * lam + 1, t, j);
        const __int128 Ao = V.get(L("sum_o", lam), t, j), Ad = V.get(L("sum_d", lam), t, j);
        const I64 r1 = RH(Ao, Q1, k.ko, s.ho), r2 = RH(Ad, Q2, k.kd, s.hdD);
        if (s.ho) AF("ro").put(lam, j, t, r1);
        if (s.hdD) AF("rd").put(lam, j, t, r2);
        key_put("O", lam, j, t, SK(Ao, Q1, k.ko, s.ho, r1));
        key_put("D", lam, j, t, SK(Ad, Q2, k.kd, s.hdD, r2));
        if (s.ho) key_put("OH", lam, j, t, HK(r1, k.ko, s.ho));
        if (s.hdD) key_put("DH", lam, j, t, HK(r2, k.kd, s.hdD));
      }
      for (int x = 0; x < f; x++) {
        const bool rt = ex && x >= fe;
        const int kg = rt ? k.kgR : k.kg, ku = rt ? k.kuR : k.ku;
        const I64 gw = V.get(L("gw", lam), t, x), up = V.get(L("u", lam), t, x), sl = V.get(L("s", lam), t, x);
        const __int128 Ag = V.get(L("sum_g", lam), t, x), Au = V.get(L("sum_u", lam), t, x);
        const I64 rg = RH(Ag, gw, kg, s.hgD), ru = RH(Au, up, ku, s.huD);
        AF("sl").put(lam, x, t, sl), AF("up").put(lam, x, t, up);
        if (s.hgD) AF("rg").put(lam, x, t, rg);
        if (s.huD) AF("ru").put(lam, x, t, ru);
        key_put("G", lam, x, t, SK(Ag, gw, kg, s.hgD, rg));
        key_put("U", lam, x, t, SK(Au, up, ku, s.huD, ru));
        if (s.hgD) key_put("GH", lam, x, t, HK(rg, kg, s.hgD));
        if (s.huD) key_put("UH", lam, x, t, HK(ru, ku, s.huD));
        key_put("UT", lam, x, t, (u64)(up + 32768));
        I64 arg = gw;
        if (s.gcl) {
          const I64 g0 = ((gw + 128) & 255) - 128, gh = (gw - g0) >> 8, gc = std::min<I64>(std::max<I64>(gh, -127), 127);
          arg = g0 + 256 * gc;
          AF("g0").put(lam, x, t, g0), AF("gc").put(lam, x, t, gc), AF("gh").put(lam, x, t, gh);
          key_put("G0", lam, x, t, (u64)(g0 + 128));
          key_put("CG", lam, x, t, (u64)(gh + 49152) + ((u64)(gc + 128) << 28) + (1ull << 40));
        } else {
          AF("gi").put(lam, x, t, gw);
        }
        key_put("F", lam, x, t, (u64)(arg + 32768) + ((u64)(sl + 4096) << 28));
      }
      if (ex) {
        const I64 chosen = V.get(L("e_star", lam), t, 0), sc = V.get(L("sc", lam), t, 0);
        I64 mr = 0;
        for (int e = 0; e < NE; e++) {
          const I64 lr = V.get(L("lr", lam), t, e);
          const I64 sel = e == chosen;
          if (sel) mr = lr;
          AR("lr").put(lamM, e, t, lr), AR("sel").put(lamM, e, t, sel);
          key_put("SEL", lam, e, t, 128 * sel);
          key_put("LT", lam, e, t, (u64)(2 * (lr + 16384)));
        }
        AR("sc").put(lamM, 0, t, sc);
        for (int e = 0; e < NE; e++) {
          const I64 lr = V.get(L("lr", lam), t, e);
          const __int128 Ar = V.get(L("sum_r", lam), t, e);
          // lw from lr: with the clamp lw is 8 ly + l3 of the unclamped logit.
          const I64 lw = (I64)((Ar + ((__int128)1 << (k.kr - 1))) >> k.kr);
          const I64 rl = RH(Ar, lw, k.kr, s.hr);
          if (s.hr) { AR("rl").put(lamM, e, t, rl); key_put("LH", lam, e, t, HK(rl, k.kr, s.hr)); }
          key_put("LR", lam, e, t, SK(Ar, lw, k.kr, s.hr, rl));
          if (s.rcl) {
            const I64 ly = lw >> 3, l3 = lw & 7;
            AR("ly").put(lamM, e, t, ly), AR("l3").put(lamM, e, t, l3);
            key_put("L3", lam, e, t, 32 * l3);
            key_put("CLR", lam, e, t, (u64)(ly + 32768) + ((u64)(lr - l3) << 25) + (1ull << 39));
          }
          I64 after = 0;
          for (int e2 = e + 1; e2 < NE; e2++) after += e2 == chosen;
          key_put("DD", lam, e, t, (u64)(mr - lr - after));
        }
        key_put("MR", lam, 0, t, s.gcl ? (u64)(mr + 16384) + ((u64)sc << 17) : (u64)(mr + 32768) + ((u64)sc << 17));
        for (int j = 0; j < d; j++) {
          const I64 mn = V.get(L("m", lam), t, j), xr = V.get(L("xr", lam), t, j);
          AD("xr").put(lamM, j, t, xr);
          key_put("XR", lam, j, t, (u64)(sc * mn + 32768 - 65536 * xr));
        }
      }
    }
  }
  return R;
}
// The kernels' bodies: the host's loops of ff_build(), one thread for each (x, token).
inline void ff_fill_gpu(FFGroup &R, const Policy &pol, const ModelShape &m, const Values &V, int g) {
  const FFShape &s = R.s;
  const int Lam = s.Lam, T = s.T, d = s.d, NE = s.NE;
  // Matrices.
  FeatMat *mats[WM_COUNT] = {&R.af, &R.ad, &R.ar, &R.qa, &R.xbI, &R.xbO};
  std::vector<u64> md;
  WitnessRun W;
  for (FeatMat *f : mats) {
    md.push_back(f->k), md.push_back(f->fr), md.push_back(f->part[0].m);
    W.common.push_back(w_buffer(f->part.size() * f->part[0].cell.size() * 4, mx::int32, 0));
  }
  // Tensors.
  std::vector<u64> td(5 * WT_COUNT, 0);
  auto tensor = [&](int id, const char *n, int mat) {
    auto it = R.tf.find(n);
    if (it == R.tf.end()) return;
    const Tensor2 &t = it->second;
    u64 *x = &td[5 * id];
    x[0] = t.off, x[1] = t.nl, x[2] = t.cnt, x[3] = t.J, x[4] = mat;
  };
  const std::pair<const char *, int> AFt[] = {{"gi", WT_GI}, {"g0", WT_G0}, {"gc", WT_GC}, {"gh", WT_GH}, {"sl", WT_SL}, {"up", WT_UP}, {"rg", WT_RG}, {"ru", WT_RU}, {"xs", WT_XS}, {"ro", WT_RO}, {"rd", WT_RD},
                                              {"aw", WT_AW}, {"ra", WT_RA}, {"mw", WT_MW}, {"rm", WT_RM}, {"qkw", WT_QKW}, {"rqk", WT_RQK}, {"vw", WT_VW}, {"rv", WT_RV}};
  for (auto &[n, id] : AFt) tensor(id, n, WM_AF);
  tensor(WT_AN, "an", WM_AD), tensor(WT_MN, "mn", WM_AD), tensor(WT_XR, "xr", WM_AD);
  const std::pair<const char *, int> ARt[] = {{"lr", WT_LR}, {"rl", WT_RL}, {"sel", WT_SEL}, {"sc", WT_SC}, {"row", WT_ROW}, {"ly", WT_LY}, {"l3", WT_L3}};
  for (auto &[n, id] : ARt) tensor(id, n, WM_AR);
  tensor(WT_Q, "q", WM_QA), tensor(WT_KX, "kx", WM_QA), tensor(WT_V, "v", WM_QA), tensor(WT_O, "o", WM_QA);
  { u64 *x = &td[5 * WT_XBI]; x[0] = 0, x[1] = s.XL, x[2] = 1, x[3] = d, x[4] = WM_XI; }
  { u64 *x = &td[5 * WT_XBO]; x[0] = 0, x[1] = s.XL, x[2] = 1, x[3] = d, x[4] = WM_XO; }
  // Keys.
  std::vector<u64> base(R.calls.size());
  u64 nkeys = 0;
  for (size_t c = 0; c < R.calls.size(); c++) base[c] = nkeys, nkeys += u64(1) << R.calls[c].n;
  static const char *SEG[WS_COUNT] = {"ROW", "N1", "N2", "N1H", "N2H", "CN1", "CN2", "QK", "QKH", "CQK", "V", "VH", "CV", "O", "D", "OH", "DH", "G", "U", "GH", "UH", "UT",
                                      "G0", "CG", "F", "SEL", "LT", "LH", "LR", "L3", "CLR", "DD", "MR", "XR"};
  std::vector<u64> kd(3 * WS_COUNT, 0);
  std::vector<uint32_t> pos;
  for (int q = 0; q < WS_COUNT; q++)
    for (size_t c = 0; c < R.calls.size(); c++) {
      auto it = R.place[c].find(SEG[q]);
      if (it == R.place[c].end()) continue;
      kd[3 * q] = base[c], kd[3 * q + 1] = R.lJ[c], kd[3 * q + 2] = pos.size();
      for (u64 x = 0; x < it->second.n; x++) pos.push_back((uint32_t)it->second.pos(x));
      break;
    }
  pos.resize(std::max<size_t>(pos.size(), 16), 0);
  W.common.push_back(w_buffer(nkeys * 8, mx::uint64, 0));
  W.common.push_back(w_buffer(nkeys, mx::uint8, 1));
  W.common.push_back(mx::array(pos.data(), {(int)pos.size()}, mx::uint32));
  W.common.push_back(w_table(kd)), W.common.push_back(w_table(md)), W.common.push_back(w_table(td));
  auto L = [&](const char *n, int lam) { return std::string(n) + "." + std::to_string(g * Lam + lam); };
  auto at = [&](const std::string &n) { return V.g.at(n); };
  // The streams, one after another.
  std::vector<mx::array> st;
  for (int sg = 0; sg <= 2 * Lam; sg++) {
    if (sg == 2 * Lam) st.push_back(at((g + 1) * Lam == s.L ? std::string("x_last") : "x." + std::to_string((g + 1) * Lam)));
    else st.push_back(at(L(sg % 2 ? "x_mid" : "x", sg / 2)));
    st.back() = mx::reshape(st.back(), {-1});
  }
  const mx::array S = mx::concatenate(st, 0);
  std::vector<u64> P(std::max<int>(WP_COUNT, 8), 0);
  P[WP_T] = T, P[WP_D] = d, P[WP_F] = s.f, P[WP_FE] = s.fe, P[WP_NE] = NE, P[WP_NQK] = s.nqk, P[WP_DQ] = s.dq, P[WP_DKV] = s.dkv, P[WP_XL] = s.XL, P[WP_NR] = s.NR;
  P[WP_HQ] = s.hq, P[WP_HV] = s.hv, P[WP_HO] = s.ho, P[WP_HG] = s.hgD, P[WP_HU] = s.huD, P[WP_HD] = s.hdD, P[WP_HR] = s.hr, P[WP_GCL] = s.gcl, P[WP_RCL] = s.rcl, P[WP_LAM] = Lam;
  W.launch("fp_w_stream", R"(W_CONTEXT
    ulong j = thread_position_in_grid.x, t = thread_position_in_grid.y, T = p[WP_T], d = p[WP_D], lam2 = 2 * p[WP_LAM];
    w_put(W, WT_XBI, 0, j, t, S[t * d + j]);
    w_put(W, WT_XBO, 0, j, t, S[(lam2 * T + t) * d + j]);
    for (ulong ix = 0; ix + 1 < lam2; ix++) w_put(W, WT_XS, ix, j, t, S[((ix + 1) * T + t) * d + j]);)", {{"S", S}}, w_table(P), d, T);
  // Norm rows.
  std::vector<u64> nrm;
  for (int i = 0; i < 2 * Lam; i++) {
    const LayerScales &k = m.layers[g * Lam + i / 2];
    nrm.push_back(i % 2 ? k.M2 : k.M1), nrm.push_back((u64)(i % 2 ? m.eps2 : m.eps1));
  }
  W.launch("fp_w_norm", R"(W_CONTEXT
    ulong t = thread_position_in_grid.x, i = thread_position_in_grid.y, T = p[WP_T], d = p[WP_D], NR = p[WP_NR];
    device long *o = (device long *)out + (i * T + t) * WN_STRIDE;
    w_norm_row(S + (i * T + t) * d, d, (long)nm[2 * i + 1], (uint)nm[2 * i], o);
    ulong lam = i / 2, n = i % 2;
    for (ulong j = 0; j < NR; j++) {
      const long D = o[WN_D + j];
      w_cell(W, WM_AR, W.td[5 * WT_ROW] + i * NR + j, t, (int)(D - 128));
      w_key(W, WS_ROW, lam, n * NR + j, t, w_row_mul((int)j) * (ulong)D);
    })", {{"S", S}, {"nm", w_table(nrm)}}, w_table(P), T, 2 * Lam, {(int)(2 * Lam * T * WN_STRIDE)});
  const mx::array N = W.pending.back();
  for (int i = 0; i < 2 * Lam; i++) {
    const int lam = i / 2, n = i % 2;
    P[WP_I] = i, P[WP_KN] = n ? m.layers[g * Lam + lam].K2 : m.layers[g * Lam + lam].K1, P[WP_LAMM] = lam;
    W.launch("fp_w_normcol", R"(W_CONTEXT
      ulong j = thread_position_in_grid.x, t = thread_position_in_grid.y, T = p[WP_T], d = p[WP_D], i = p[WP_I], lam = p[WP_LAMM], n = i % 2;
      int K = (int)p[WP_KN];
      long x = S[(i * T + t) * d + j], R = N[(i * T + t) * WN_STRIDE + WN_R], wv = wa[t * d + j], cv = ca[t * d + j];
      long A = (long)((ulong)x * (ulong)R), rh = w_RH(A, wv, K, 2);
      w_put(W, n ? WT_MW : WT_AW, lam, j, t, wv); w_put(W, n ? WT_RM : WT_RA, lam, j, t, rh); w_put(W, n ? WT_MN : WT_AN, lam, j, t, cv);
      w_key(W, n ? WS_N2 : WS_N1, lam, j, t, w_SK(A, wv, K, 2, rh));
      w_key(W, n ? WS_N2H : WS_N1H, lam, j, t, w_HK(rh, K, 2));
      w_key(W, n ? WS_CN2 : WS_CN1, lam, j, t, (ulong)(wv + 32768) + ((ulong)(cv + 2048) << 28));)",
             {{"S", S}, {"N", N}, {"wa", at(L(n ? "m_w" : "a_w", lam))}, {"ca", at(L(n ? "m" : "a", lam))}}, w_table(P), d, T);
  }
  for (int lam = 0; lam < Lam; lam++) {
    const LayerScales &k = m.layers[g * Lam + lam];
    const bool ex = NE && (lam & 1);
    P[WP_LAMM] = lam, P[WP_EX] = ex, P[WP_KQ] = k.kq, P[WP_KK] = k.kk, P[WP_KV] = k.kv, P[WP_KO] = k.ko, P[WP_KD] = k.kd, P[WP_KG] = k.kg, P[WP_KU] = k.ku, P[WP_KGR] = k.kgR, P[WP_KUR] = k.kuR, P[WP_KR] = k.kr;
    const mx::array Pl = w_table(P);
    W.launch("fp_w_qk", R"(W_CONTEXT
      ulong x = thread_position_in_grid.x, t = thread_position_in_grid.y, dq = p[WP_DQ], dkv = p[WP_DKV], lam = p[WP_LAMM]; int hq = (int)p[WP_HQ];
      bool isq = x < dq; int kk = (int)(isq ? p[WP_KQ] : p[WP_KK]);
      long w = isq ? qw[t * dq + x] : kw[t * dkv + x - dq], cl = isq ? qv[t * dq + x] : kv[t * dkv + x - dq], A = isq ? sq[t * dq + x] : sk[t * dkv + x - dq];
      long rh = w_RH(A, w, kk, hq);
      w_put(W, WT_QKW, lam, x, t, w);
      if (hq) w_put(W, WT_RQK, lam, x, t, rh);
      if (isq) w_put(W, WT_Q, lam, x, t, cl); else w_put(W, WT_KX, lam, x - dq, t, cl);
      w_key(W, WS_QK, lam, x, t, w_SK(A, w, kk, hq, rh));
      if (hq) w_key(W, WS_QKH, lam, x, t, w_HK(rh, kk, hq));
      w_key(W, WS_CQK, lam, x, t, (ulong)(w + 32768) + ((ulong)(cl + 2048) << 28));)",
             {{"qw", at(L("qw", lam))}, {"kw", at(L("kw", lam))}, {"qv", at(L("q", lam))}, {"kv", at(L("k", lam))}, {"sq", at(L("sum_q", lam))}, {"sk", at(L("sum_k", lam))}}, Pl, s.nqk, T);
    W.launch("fp_w_v", R"(W_CONTEXT
      ulong x = thread_position_in_grid.x, t = thread_position_in_grid.y, n = p[WP_DKV], lam = p[WP_LAMM]; int h = (int)p[WP_HV], kv = (int)p[WP_KV];
      long w = vw[t * n + x], cl = vv[t * n + x], A = sv[t * n + x], rh = w_RH(A, w, kv, h);
      w_put(W, WT_VW, lam, x, t, w);
      if (h) w_put(W, WT_RV, lam, x, t, rh);
      w_put(W, WT_V, lam, x, t, cl);
      w_key(W, WS_V, lam, x, t, w_SK(A, w, kv, h, rh));
      if (h) w_key(W, WS_VH, lam, x, t, w_HK(rh, kv, h));
      w_key(W, WS_CV, lam, x, t, (ulong)(w + 32768) + ((ulong)(cl + 2048) << 28));)",
             {{"vw", at(L("vw", lam))}, {"vv", at(L("v", lam))}, {"sv", at(L("sum_v", lam))}}, Pl, s.dkv, T);
    W.launch("fp_w_o", R"(W_CONTEXT
      ulong x = thread_position_in_grid.x, t = thread_position_in_grid.y, n = p[WP_DQ], lam = p[WP_LAMM];
      w_put(W, WT_O, lam, x, t, ov[t * n + x]);)", {{"ov", at(L("o", lam))}}, Pl, s.dq, T);
    W.launch("fp_w_d", R"(W_CONTEXT
      ulong j = thread_position_in_grid.x, t = thread_position_in_grid.y, T = p[WP_T], d = p[WP_D], lam = p[WP_LAMM]; int ho = (int)p[WP_HO], hd = (int)p[WP_HD], ko = (int)p[WP_KO], kd2 = (int)p[WP_KD];
      long s0 = S[((2 * lam) * T + t) * d + j], s1 = S[((2 * lam + 1) * T + t) * d + j], s2 = S[((2 * lam + 2) * T + t) * d + j], Q1 = s1 - s0, Q2 = s2 - s1;
      long Ao = so[t * d + j], Ad = sd[t * d + j], r1 = w_RH(Ao, Q1, ko, ho), r2 = w_RH(Ad, Q2, kd2, hd);
      if (ho) w_put(W, WT_RO, lam, j, t, r1);
      if (hd) w_put(W, WT_RD, lam, j, t, r2);
      w_key(W, WS_O, lam, j, t, w_SK(Ao, Q1, ko, ho, r1));
      w_key(W, WS_D, lam, j, t, w_SK(Ad, Q2, kd2, hd, r2));
      if (ho) w_key(W, WS_OH, lam, j, t, w_HK(r1, ko, ho));
      if (hd) w_key(W, WS_DH, lam, j, t, w_HK(r2, kd2, hd));)", {{"S", S}, {"so", at(L("sum_o", lam))}, {"sd", at(L("sum_d", lam))}}, Pl, d, T);
    W.launch("fp_w_f", R"(W_CONTEXT
      ulong x = thread_position_in_grid.x, t = thread_position_in_grid.y, f = p[WP_F], lam = p[WP_LAMM]; int hg = (int)p[WP_HG], hu = (int)p[WP_HU];
      bool rt = p[WP_EX] && x >= p[WP_FE]; int kg = (int)(rt ? p[WP_KGR] : p[WP_KG]), ku = (int)(rt ? p[WP_KUR] : p[WP_KU]);
      long gv = gw[t * f + x], up = uu[t * f + x], sl = ss[t * f + x], Ag = sg[t * f + x], Au = su[t * f + x], rg = w_RH(Ag, gv, kg, hg), ru = w_RH(Au, up, ku, hu);
      w_put(W, WT_SL, lam, x, t, sl); w_put(W, WT_UP, lam, x, t, up);
      if (hg) w_put(W, WT_RG, lam, x, t, rg);
      if (hu) w_put(W, WT_RU, lam, x, t, ru);
      w_key(W, WS_G, lam, x, t, w_SK(Ag, gv, kg, hg, rg));
      w_key(W, WS_U, lam, x, t, w_SK(Au, up, ku, hu, ru));
      if (hg) w_key(W, WS_GH, lam, x, t, w_HK(rg, kg, hg));
      if (hu) w_key(W, WS_UH, lam, x, t, w_HK(ru, ku, hu));
      w_key(W, WS_UT, lam, x, t, (ulong)(up + 32768));
      long arg = gv;
      if (p[WP_GCL]) {
        long g0 = ((gv + 128) & 255) - 128, gh = (gv - g0) >> 8, gc = min(max(gh, -127l), 127l);
        arg = g0 + 256 * gc;
        w_put(W, WT_G0, lam, x, t, g0); w_put(W, WT_GC, lam, x, t, gc); w_put(W, WT_GH, lam, x, t, gh);
        w_key(W, WS_G0, lam, x, t, (ulong)(g0 + 128));
        w_key(W, WS_CG, lam, x, t, (ulong)(gh + 49152) + ((ulong)(gc + 128) << 28) + (1ul << 40));
      } else w_put(W, WT_GI, lam, x, t, gv);
      w_key(W, WS_F, lam, x, t, (ulong)(arg + 32768) + ((ulong)(sl + 4096) << 28));)",
             {{"gw", at(L("gw", lam))}, {"uu", at(L("u", lam))}, {"ss", at(L("s", lam))}, {"sg", at(L("sum_g", lam))}, {"su", at(L("sum_u", lam))}}, Pl, s.f, T);
    if (ex) {
      P[WP_LAMM] = lam, P[WP_I] = (lam - 1) / 2;
      const mx::array Pe = w_table(P);
      W.launch("fp_w_route", R"(W_CONTEXT
        ulong t = thread_position_in_grid.x, NE = p[WP_NE], lam = p[WP_LAMM], lm = p[WP_I]; int hr = (int)p[WP_HR], kr = (int)p[WP_KR];
        long chosen = es[t], sc = scv[t], mr = 0;
        for (ulong e = 0; e < NE; e++) {
          long lr = lrv[t * NE + e], sel = (long)e == chosen;
          if (sel) mr = lr;
          w_put(W, WT_LR, lm, e, t, lr); w_put(W, WT_SEL, lm, e, t, sel);
          w_key(W, WS_SEL, lam, e, t, (ulong)(128 * sel));
          w_key(W, WS_LT, lam, e, t, (ulong)(2 * (lr + 16384)));
        }
        w_put(W, WT_SC, lm, 0, t, sc);
        for (ulong e = 0; e < NE; e++) {
          long lr = lrv[t * NE + e], Ar = sr[t * NE + e], lw = (Ar + (1l << (kr - 1))) >> kr, rl = w_RH(Ar, lw, kr, hr);
          if (hr) { w_put(W, WT_RL, lm, e, t, rl); w_key(W, WS_LH, lam, e, t, w_HK(rl, kr, hr)); }
          w_key(W, WS_LR, lam, e, t, w_SK(Ar, lw, kr, hr, rl));
          if (p[WP_RCL]) {
            long ly = lw >> 3, l3 = lw & 7;
            w_put(W, WT_LY, lm, e, t, ly); w_put(W, WT_L3, lm, e, t, l3);
            w_key(W, WS_L3, lam, e, t, (ulong)(32 * l3));
            w_key(W, WS_CLR, lam, e, t, (ulong)(ly + 32768) + ((ulong)(lr - l3) << 25) + (1ul << 39));
          }
          long after = 0;
          for (ulong e2 = e + 1; e2 < NE; e2++) after += (long)e2 == chosen;
          w_key(W, WS_DD, lam, e, t, (ulong)(mr - lr - after));
        }
        w_key(W, WS_MR, lam, 0, t, p[WP_GCL] ? (ulong)(mr + 16384) + ((ulong)sc << 17) : (ulong)(mr + 32768) + ((ulong)sc << 17));)",
               {{"es", at(L("e_star", lam))}, {"scv", at(L("sc", lam))}, {"lrv", at(L("lr", lam))}, {"sr", at(L("sum_r", lam))}}, Pe, T, 1);
      W.launch("fp_w_xr", R"(W_CONTEXT
        ulong j = thread_position_in_grid.x, t = thread_position_in_grid.y, d = p[WP_D], lam = p[WP_LAMM], lm = p[WP_I];
        long mn = mv[t * d + j], xv = xr[t * d + j], sc = scv[t];
        w_put(W, WT_XR, lm, j, t, xv);
        w_key(W, WS_XR, lam, j, t, (ulong)(sc * mn + 32768 - 65536 * xv));)",
               {{"mv", at(L("m", lam))}, {"xr", at(L("xr", lam))}, {"scv", at(L("sc", lam))}}, Pe, d, T);
    }
  }
  mx::eval(W.pending);
  // Back into the group's host structures, which the proof reads.
  for (int q = 0; q < WM_COUNT; q++) {
    const int32_t *c = W.common[q].data<int32_t>();
    for (CellMatrix2 &part : mats[q]->part) { memcpy(part.cell.data(), c, part.cell.size() * 4); c += part.cell.size(); }
  }
  const uint64_t *keys = W.common[WM_COUNT].data<uint64_t>();
  const uint8_t *pad = W.common[WM_COUNT + 1].data<uint8_t>();
  for (size_t c = 0; c < R.calls.size(); c++) {
    memcpy(R.calls[c].keys.data(), keys + base[c], R.calls[c].keys.size() * 8);
    memcpy(R.calls[c].padding.data(), pad + base[c], R.calls[c].padding.size());
    // A padding position holds the value of row t0 only in lookup_leaves(), not here: as ff_build().
  }
  R.rows.assign(2 * Lam, std::vector<NormRow>(T));
  const int64_t *n = N.data<int64_t>();
  for (int i = 0; i < 2 * Lam; i++)
    for (int t = 0; t < T; t++) {
      const int64_t *o = n + ((size_t)i * T + t) * WN_STRIDE;
      NormRow &r = R.rows[i][t];
      r.a = o[WN_A], r.b = o[WN_B], r.sig = o[WN_SIG], r.Rlo = o[WN_RLO], r.Rhi = o[WN_RHI], r.R = o[WN_R];
      r.D.assign(o + WN_D, o + WN_D + 99);
      r.W.assign(o + WN_W, o + WN_W + 9);
    }
}
}
