// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
// The witness of the output head (docs/format/11-4-output-head.md, second half): the matrices of cells,
// each position's row with its 21 further digits, and the keys of the lookup of step 1. The honest
// rule for a position's cost is the one of prover/head/forward.cuh (k_head_price, k_head_budget), which
// the format leaves open. Five limbs only (NR = 99), the step table only.
#pragma once
#include "seg_ff_witness.h"
#include "ff_witness_gpu.h"
namespace fpmlx {
// The request of 11.4: T tokens, then r and e, as T + 2 words; y: the next token.
struct HeadRequest {
  std::vector<u32> tokens;
  u32 r = 0, e = 0, y = 0;
  static HeadRequest read(const std::string &bytes, unsigned T) {
    if (bytes.size() != 4ull * (T + 2)) throw std::runtime_error("request: not T + 2 words");
    HeadRequest q;
    q.tokens.resize(T);
    memcpy(q.tokens.data(), bytes.data(), 4ull * T);
    memcpy(&q.r, bytes.data() + 4ull * T, 4);
    memcpy(&q.e, bytes.data() + 4ull * T + 4, 4);
    return q;
  }
};
struct HeadShape2 {
  int lt, T, d, V, la, ls, S2, lPd, JV, JVa, XL, NR, NRH, NS, NPr, NO;
  int lmhf, NHf, lmhd, lmhr, lmbH, lmx, nmin, MF, KF, sh, sT, F, lkW, tau, lidb;
  long long C, theta, cap;
  long long offE() const { return 8421505; }
};
inline HeadShape2 head_shape(const Policy &p, const ModelShape &m) {
  HeadShape2 s{};
  s.lt = (int)p.get("lt"), s.T = 1 << s.lt, s.d = (int)p.get("d"), s.V = (int)p.get("V"), s.la = (int)p.get("la"), s.ls = s.la - s.lt, s.S2 = 1 << s.ls;
  s.lPd = (int)clog2u(s.d), s.JV = (s.V + s.S2 - 1) / s.S2 * s.S2, s.JVa = 1 << (clog2u(s.JV) - 1);
  s.XL = (int)p.get("xl"), s.NR = 99, s.NRH = s.NR + 21, s.NS = 2 * s.XL - 1, s.NPr = 7, s.NO = 4;
  if (s.XL != 5) throw std::runtime_error("head: five limbs only");
  s.lmhf = (int)p.get("lmhf"), s.NHf = (int)p.get("NHf"), s.lmhd = (int)p.get("lmhd"), s.lmhr = (int)p.get("lmhr"), s.lmbH = (int)p.get("lmbH"), s.lmx = (int)p.get("lmx");
  s.nmin = p.v.count("nminH") ? (int)p.v.at("nminH") : (int)p.get("nmin");
  s.MF = m.MF, s.KF = m.KF, s.sh = (int)p.get("sh"), s.sT = (int)p.get("sT"), s.F = (int)p.get("F");
  if (s.sT + 4 != s.sh) throw std::runtime_error("head: the step table only");
  s.lkW = (int)p.get("table_w"), s.tau = (int)p.get("tau"), s.lidb = (int)p.get("lidb");
  s.C = p.get("C"), s.theta = p.get("theta"), s.cap = p.v.at("cap");
  return s;
}
// The table of exponentials with N = 484.
inline std::vector<u32> head_ex_table(int F) {
  std::vector<u32> EX(65536);
  unsigned __int128 G = (unsigned __int128)1 << (F + 17);
  for (u32 &x : EX) {
    x = (u32)((G + 131071) >> 17);
    G = (G * 484 + 0xFFFFFFFFull) >> 32;
  }
  return EX;
}
struct HeadWitness {
  HeadShape2 s;
  HeadRequest q;
  FeatMat hf, hd, hr, xbL; // xbL: the cells of the last group's xbO
  CellMatrix2 bH;
  std::vector<LookupCall> calls;
  std::vector<std::map<std::string, SegPlace>> place; // by call
  std::vector<int> lJ;
  LookupSizes sizes;
  std::vector<std::vector<u32>> mult;
  std::vector<NormRow> rows;                  // by position
  std::vector<I64> act, str;                  // [t * d + k]
  std::vector<I64> lam;                       // [t * V + v]
  std::vector<u32> win;                       // by position; 0 where on_t = 0
  std::vector<std::array<I64, 6>> hidden;     // gc, gb, gu, gk, gz, gs by position
  std::vector<u32> EX;
  std::vector<char> on, co;
  // False: the witness is no honest one, and the proof would be refused.
  bool norm_ok = true, margins_ok = true, bounds_ok = true, keys_ok = false;
  std::vector<I64> price;
  int call_of(const std::string &seg) const { for (size_t c = 0; c < place.size(); c++) if (place[c].count(seg)) return (int)c; throw std::runtime_error("no call holds " + seg); }
};
inline u64 head_seg_len(const HeadShape2 &s, const std::string &n) {
  if (n == "NF" || n == "NFH" || n == "CNF") return s.d;
  if (n == "HD0" || n == "HD1" || n == "HD2") return s.JV;
  if (n == "ROW") return s.NRH;
  if (n == "HE0") return s.JVa;
  if (n == "HE1") return s.JV - s.JVa;
  throw std::runtime_error("no such segment " + n);
}
// y = the lowest index of the largest logit at T - 1 if e = T, else 0.
inline u32 head_next_token(const Values &V, const HeadRequest &q, unsigned T) { return q.e == T ? (u32)V.get("first_choice", T - 1, 0) : 0; }
inline HeadWitness head_finish(HeadWitness &R);
inline void head_fill_gpu(HeadWitness &R, const ModelShape &m, const Values &V);
inline HeadWitness head_build(const Policy &pol, const ModelShape &m, const Values &V, const CallTables &tabs, const HeadRequest &q, bool on_gpu = true) {
  HeadWitness R;
  R.s = head_shape(pol, m), R.q = q;
  const HeadShape2 &s = R.s;
  const int T = s.T, d = s.d, Vn = s.V, JV = s.JV, JVa = s.JVa;
  R.EX = head_ex_table(s.F);
  R.hf.init(s.NHf, s.lmhf, 4ll * d + 6ll * JV, s.lt);
  R.hd.init(1, s.lmhd, 2ll * d, s.lt);
  R.hr.init(1, s.lmhr, s.NRH, s.lt);
  R.xbL.init(1, s.lmx, (long long)s.XL * d, s.lt);
  // The calls: pieces, columns, tables.
  const size_t NC = pol.callH.size();
  R.calls.resize(NC), R.place.resize(NC), R.lJ.resize(NC);
  std::vector<u64> FE(65536);
  for (u64 u = 0; u < 65536; u++) FE[u] = u + ((u64)R.EX[u] << 16);
  for (size_t c = 0; c < NC; c++) {
    struct Piece { std::string seg; u64 first, size; };
    std::vector<Piece> pieces;
    for (const std::string &n : pol.callH[c].segs) {
      const u64 len = head_seg_len(s, n);
      u64 first = 0;
      for (int b = 62; b >= 0; b--)
        if ((len >> b) & 1) pieces.push_back({n, first, 1ull << b}), first += 1ull << b;
      R.place[c][n].n = len;
    }
    std::stable_sort(pieces.begin(), pieces.end(), [](const Piece &a, const Piece &b) { return a.size > b.size; });
    u64 col = 0;
    for (const Piece &p : pieces) {
      R.place[c][p.seg].pieces.push_back({p.first, p.size}), R.place[c][p.seg].col.push_back(col);
      col += p.size;
    }
    R.lJ[c] = std::max<int>((int)clog2u(col), s.nmin - s.lt);
    LookupCall &K = R.calls[c];
    K.n = R.lJ[c] + s.lt;
    const std::string &kind = pol.callH[c].kind;
    K.k = kind == "R8" ? 8 : 16;
    if (kind == "FC") K.table = tabs.FC, K.t0 = 32768;
    else if (kind == "FE") K.table = FE, K.t0 = 0;
    K.keys.assign(1ull << K.n, K.row(K.t0)), K.padding.assign(1ull << K.n, 1);
  }
  std::map<std::string, int> callOf;
  for (size_t c = 0; c < NC; c++) for (auto &x : R.place[c]) callOf[x.first] = (int)c;
  auto key_put = [&](const std::string &seg, u64 x, int t, u64 key, bool padding = false) {
    const int c = callOf.at(seg);
    const u64 idx = R.place[c].at(seg).pos(x) * T + t;
    R.calls[c].keys[idx] = key, R.calls[c].padding[idx] = padding;
  };
  auto put = [&](FeatMat &M, int f0, int stride, int nl, int t, I64 v) { int32_t c[8]; limbs_of(v, nl, c); for (int lb = 0; lb < nl; lb++) M.cell(f0 + (long long)lb * stride, t) = c[lb]; };
  // Positions.
  R.on.assign(T, 0), R.co.assign(T, 0), R.win.assign(T, 0);
  for (int t = 0; t < T; t++) {
    R.on[t] = (u32)t >= q.r && (u32)t < q.e, R.co[t] = R.on[t] && t + 1 < T;
    if (R.on[t]) R.win[t] = t + 1 < T ? q.tokens[t + 1] : q.y;
    if (R.win[t] >= (u32)Vn) throw std::runtime_error("head: a token is not below V");
  }
  if (on_gpu) { head_fill_gpu(R, m, V); return head_finish(R); }
  R.str.resize((size_t)T * d), R.act.resize((size_t)T * d), R.lam.resize((size_t)T * Vn), R.rows.resize(T);
  // The final norm: stream, row, wide output, remainder, activations.
  std::vector<int32_t> lim((size_t)d * s.XL);
  for (int t = 0; t < T; t++) {
    for (int k = 0; k < d; k++) {
      const I64 x = V.get("x_last", t, k);
      R.str[(size_t)t * d + k] = x;
      limbs_of(x, s.XL, &lim[(size_t)k * s.XL]);
      for (int lb = 0; lb < s.XL; lb++) R.xbL.cell((long long)lb * d + k, t) = lim[(size_t)k * s.XL + lb];
    }
    NormRow r = norm_row(s.XL, lim, d, m.epsF, s.MF);
    if (r.R != V.get("aF_Rc", t, 0) || r.sig != V.get("aF_sigma", t, 0)) R.norm_ok = false;
    for (int k = 0; k < d; k++) {
      const I64 aw = V.get("aF_w", t, k), act = V.get("aF", t, k);
      const __int128 rem = (__int128)R.str[(size_t)t * d + k] * r.R + ((__int128)1 << (s.KF - 1)) - ((__int128)aw << s.KF);
      if (rem < 0 || rem >= ((__int128)1 << s.KF)) R.norm_ok = false;
      const I64 rh = (I64)(rem >> 16), rl = (I64)(rem & 65535);
      R.act[(size_t)t * d + k] = act;
      put(R.hf, k, d, 2, t, aw), put(R.hf, 2 * d + k, d, 2, t, rh - 32896), put(R.hd, k, d, 2, t, act);
      key_put("NF", k, t, (u64)rl), key_put("NFH", k, t, (u64)rh << (32 - s.KF));
      key_put("CNF", k, t, (u64)(aw + 32768) + ((u64)(act + 2048) << 28));
    }
    R.rows[t] = std::move(r);
    for (int v = 0; v < Vn; v++) R.lam[(size_t)t * Vn + v] = V.get("lam", t, v);
  }
  // Each position's cost: k_head_price.
  const I64 BIG = 1ll << 37, most = (1ll << 40) - 1;
  R.price.assign(T, 0);
  std::vector<I64> spare(T, 0);
  for (int t = 0; t < T; t++) {
    if (!R.co[t]) continue;
    const I64 *lam = &R.lam[(size_t)t * Vn];
    const u32 w = R.win[t];
    I64 gap = 0;
    for (int v = 0; v < Vn; v++) gap = std::max<I64>(gap, lam[v] - lam[w] + (v < (int)w));
    std::vector<I64> at(Vn);
    for (int v = 0; v < Vn; v++) at[v] = (lam[w] + gap + (1ll << s.sh) - lam[v] - (v < (int)w)) >> s.sh;
    auto sum = [&](I64 j) { u64 a = 0; for (int v = 0; v < Vn; v++) a += R.EX[std::min<I64>(at[v] + j, 65535)]; return a; };
    I64 low = 0, high = 32768;
    while (low < high) {
      const I64 mid = (low + high) / 2;
      if (sum(mid) <= 1ull << s.F) high = mid; else low = mid + 1;
    }
    R.price[t] = gap + ((1 + low) << s.sh);
    spare[t] = (I64)((1ull << s.F) - sum(low));
  }
  // The six hidden numbers: k_head_budget.
  const int r0 = (int)q.r, end = std::min<int>((int)q.e, T - 1);
  I64 spent = 0, clipped = 0;
  for (int t = r0; t < end; t++) spent += std::min<I64>(R.price[t], s.C), clipped += R.price[t] > s.C;
  const I64 RC = std::max(end - r0, 0);
  I64 left = s.theta * RC - spent, rows_left = (RC ? s.cap : 0) - clipped;
  if (left < 0 || rows_left < 0) R.bounds_ok = false;
  left = std::max<I64>(left, 0), rows_left = std::max<I64>(rows_left, 0);
  R.hidden.assign(T, {0, 0, 0, 0, 0, 0});
  std::vector<I64> lift(T, 0);
  static const int dg0[6] = {0, 5, 6, 11, 16, 18}, dgn[6] = {5, 1, 5, 5, 2, 3};
  for (int t = 0; t < T; t++) {
    const bool counted = R.co[t], over = counted && R.price[t] > s.C;
    const I64 c = counted && !over ? R.price[t] : 0;
    auto &h = R.hidden[t];
    h[0] = c, h[1] = over, h[2] = counted ? std::min<I64>(std::max<I64>(left - (I64)(t - r0) * most, 0), most) : 0;
    h[3] = counted ? s.C - c : 0, h[4] = t == r0 && counted ? rows_left : 0;
    h[5] = !counted ? 0 : over ? (1ll << s.F) - Vn : spare[t];
    lift[t] = c + (over ? BIG : 0);
    std::vector<int> &D = R.rows[t].D;
    D.resize(s.NRH, 0);
    for (int i = 0; i < 6; i++) put_digits(D, s.NR + dg0[i], dgn[i], (unsigned __int128)h[i]);
    for (int j = 0; j < s.NRH; j++) {
      R.hr.cell(j, t) = D[j] - 128;
      key_put("ROW", j, t, row_mul(j) * (u64)D[j]);
    }
  }
  // Margins, digits, table values.
  const int oD1 = 4 * d, oD2 = 4 * d + 2 * JV, oE = 4 * d + 3 * JV;
  const I64 reach = 65537ll << (s.sh + 15);
  for (int t = 0; t < T; t++)
    for (int v = 0; v < JV; v++) {
      I64 K0 = 0, K1 = 0, K2 = 0;
      const bool real = v < Vn && R.on[t], padding = v >= Vn;
      if (real) {
        const I64 *lam = &R.lam[(size_t)t * Vn];
        const u32 w = R.win[t];
        I64 D = lam[w] + lift[t] - lam[v] - (v < (int)w);
        if (D < 0 || D >= reach) R.margins_ok = false, D = 0;
        const I64 mm = D >> s.sh;
        K2 = D & ((1ll << s.sh) - 1);
        K1 = mm < 32768 ? mm : 32768 + (mm & 32767);
        K0 = mm < 32768 ? 0 : (mm >> 15) - 1;
      }
      const I64 ex = R.EX[K1];
      put(R.hf, oD1 + v, JV, 2, t, real ? K1 - 32896 : 0);
      put(R.hf, oD2 + v, JV, 1, t, real ? K2 - 128 : 0);
      put(R.hf, oE + v, JV, 3, t, ex - s.offE());
      key_put("HD0", v, t, (u64)K0, padding), key_put("HD1", v, t, (u64)K1, padding), key_put("HD2", v, t, (u64)K2, padding);
      const u64 he = (u64)K1 + ((u64)ex << 16);
      if (v < JVa) key_put("HE0", v, t, he, padding); else key_put("HE1", v - JVa, t, he, padding);
    }
  return head_finish(R);
}
inline HeadWitness head_finish(HeadWitness &R) {
  const HeadShape2 &s = R.s;
  // The lookup's sizes, multiplicities and planes.
  R.sizes = lookup_sizes(R.calls, s.lidb);
  R.keys_ok = lookup_multiplicities(R.calls, R.sizes, R.mult);
  const LookupSizes &z = R.sizes;
  const int wd = (int)z.kT + 5 - s.lmbH, lowT = s.lmbH - 5;
  R.bH = CellMatrix2(1ull << s.lmbH, (u64)z.NC << wd);
  for (unsigned c = 0; c < z.NC; c++)
    for (u64 t = 0; t < (1ull << z.kT); t++)
      for (int j = 0; j < 32; j++)
        if ((R.mult[c][t] >> j) & 1) R.bH.at((t & ((1ull << lowT) - 1)) * 32 + j, ((u64)c << wd) + (t >> lowT)) = 1;
  return std::move(R);
}
// The six hidden numbers of k_head_budget, from the prices; they are a few words for each position.
inline std::vector<I64> head_hidden(HeadWitness &R, const std::vector<I64> &spare) {
  const HeadShape2 &s = R.s;
  const int T = s.T, r0 = (int)R.q.r, end = std::min<int>((int)R.q.e, T - 1);
  const I64 BIG = 1ll << 37, most = (1ll << 40) - 1;
  I64 spent = 0, clipped = 0;
  for (int t = r0; t < end; t++) spent += std::min<I64>(R.price[t], s.C), clipped += R.price[t] > s.C;
  const I64 RC = std::max(end - r0, 0);
  I64 left = s.theta * RC - spent, rows_left = (RC ? s.cap : 0) - clipped;
  if (left < 0 || rows_left < 0) R.bounds_ok = false;
  left = std::max<I64>(left, 0), rows_left = std::max<I64>(rows_left, 0);
  R.hidden.assign(T, {0, 0, 0, 0, 0, 0});
  std::vector<I64> lift(T, 0);
  static const int dg0[6] = {0, 5, 6, 11, 16, 18}, dgn[6] = {5, 1, 5, 5, 2, 3};
  for (int t = 0; t < T; t++) {
    const bool counted = R.co[t], over = counted && R.price[t] > s.C;
    const I64 c = counted && !over ? R.price[t] : 0;
    auto &h = R.hidden[t];
    h[0] = c, h[1] = over, h[2] = counted ? std::min<I64>(std::max<I64>(left - (I64)(t - r0) * most, 0), most) : 0;
    h[3] = counted ? s.C - c : 0, h[4] = t == r0 && counted ? rows_left : 0;
    h[5] = !counted ? 0 : over ? (1ll << s.F) - s.V : spare[t];
    lift[t] = c + (over ? BIG : 0);
    std::vector<int> &D = R.rows[t].D;
    D.resize(s.NRH, 0);
    for (int i = 0; i < 6; i++) put_digits(D, s.NR + dg0[i], dgn[i], (unsigned __int128)h[i]);
  }
  return lift;
}
inline void head_fill_gpu(HeadWitness &R, const ModelShape &m, const Values &V) {
  const HeadShape2 &s = R.s;
  const int T = s.T, d = s.d, Vn = s.V, JV = s.JV, JVa = s.JVa;
  std::vector<u64> base(R.calls.size());
  u64 nk = 0;
  for (size_t c = 0; c < R.calls.size(); c++) base[c] = nk, nk += u64(1) << R.calls[c].n;
  mx::array keys = w_buffer(nk * 8, mx::uint64, 0), pad = w_buffer(nk, mx::uint8, 1);
  for (size_t c = 0; c < R.calls.size(); c++) memcpy(keys.data<uint64_t>() + base[c], R.calls[c].keys.data(), R.calls[c].keys.size() * 8);
  static const char *SEG[HS_COUNT] = {"NF", "NFH", "CNF", "HD0", "HD1", "HD2", "HE0", "HE1", "ROW"};
  std::vector<u64> kd(2 * HS_COUNT, 0);
  std::vector<uint32_t> pos;
  for (int q = 0; q < HS_COUNT; q++) {
    const int c = R.call_of(SEG[q]);
    const SegPlace &pl = R.place[c].at(SEG[q]);
    kd[2 * q] = base[c], kd[2 * q + 1] = pos.size();
    for (u64 x = 0; x < pl.n; x++) pos.push_back((uint32_t)pl.pos(x));
  }
  FeatMat *mats[4] = {&R.hf, &R.hd, &R.hr, &R.xbL};
  mx::array cells[4] = {mx::array(0), mx::array(0), mx::array(0), mx::array(0)};
  for (int q = 0; q < 4; q++) cells[q] = w_buffer(mats[q]->part.size() * mats[q]->part[0].cell.size() * 4, mx::int32, 0);
  mx::array flags = w_buffer(64, mx::uint64, 0);
  std::vector<u64> P(HP_COUNT, 0);
  P[HP_T] = T, P[HP_D] = d, P[HP_V] = Vn, P[HP_JV] = JV, P[HP_JVA] = JVa, P[HP_KF] = s.KF, P[HP_MF] = s.MF, P[HP_EPS] = (u64)m.epsF, P[HP_SH] = s.sh, P[HP_F] = s.F, P[HP_NRH] = s.NRH;
  P[HP_HFK] = R.hf.k, P[HP_HFFR] = R.hf.fr, P[HP_HFROWS] = R.hf.part[0].m, P[HP_HDK] = R.hd.k, P[HP_HDFR] = R.hd.fr, P[HP_HRK] = R.hr.k, P[HP_HRFR] = R.hr.fr, P[HP_XK] = R.xbL.k, P[HP_XFR] = R.xbL.fr;
  P[HP_OD1] = 4 * d, P[HP_OD2] = 4 * d + 2 * JV, P[HP_OE] = 4 * d + 3 * JV, P[HP_OFFE] = (u64)s.offE(), P[HP_REACH] = (u64)(65537ll << (s.sh + 15));
  const mx::array Pa = w_table(P), kdA = w_table(kd), posA(pos.data(), {(int)pos.size()}, mx::uint32), x = mx::reshape(V.g.at("x_last"), {-1});
  std::vector<u64> ex(R.EX.begin(), R.EX.end()), win(R.win.begin(), R.win.end()), co(R.co.begin(), R.co.end()), on(R.on.begin(), R.on.end());
  const mx::array exA = w_table(ex), winA = w_table(win), coA = w_table(co), onA = w_table(on), lamA = V.g.at("lam");
  const std::vector<std::string> common{"c0", "c1", "c2", "c3", "keys", "padf", "kd", "pos", "fl"};
  auto with = [&](std::vector<mx::array> more, std::vector<std::string> names) {
    std::vector<mx::array> a{cells[0], cells[1], cells[2], cells[3], keys, pad, kdA, posA, flags};
    a.insert(a.end(), more.begin(), more.end());
    std::vector<std::string> n = common;
    n.insert(n.end(), names.begin(), names.end());
    return std::make_pair(a, n);
  };
  auto [a1, n1] = with({x, Pa}, {"S", "p"});
  const mx::array N = run_kernel("fp_h_norm", R"(
    ulong t = thread_position_in_grid.x, d = p[HP_D];
    w_norm_row(S + t * d, d, (long)p[HP_EPS], (uint)p[HP_MF], (device long *)out + t * WN_STRIDE);)", a1, n1, {{(int)(T * WN_STRIDE)}}, T)[0];
  auto [a2, n2] = with({x, N, V.g.at("aF_w"), V.g.at("aF"), Pa}, {"S", "N", "aw", "ac", "p"});
  const mx::array cols = run_kernel("fp_h_cols", R"(
    ulong k = thread_position_in_grid.x, t = thread_position_in_grid.y, T = p[HP_T], d = p[HP_D]; uint KF = (uint)p[HP_KF];
    device int *hf = (device int *)c0, *hd = (device int *)c1, *xb = (device int *)c3; device ulong *KY = (device ulong *)keys; device uchar *PD = (device uchar *)padf; device ulong *F = (device ulong *)fl;
    long xv = S[t * d + k], R = N[t * WN_STRIDE + WN_R], wv = aw[t * d + k], act = ac[t * d + k];
    f_put(xb, p[HP_XK], p[HP_XFR], T * p[HP_XFR], T, k, d, 5, t, xv);
    U128 rem = U128{(ulong)xv * (ulong)R, (ulong)mulhi(xv, R)};
    rem = u_sub(u_add(rem, u_shl(u_of(1), KF - 1)), u_shl(s_of(wv), KF));
    if (rem.hi != 0 || (rem.lo >> KF) != 0) F[0] = 1;
    long rh = (long)((rem.lo >> 16) | (rem.hi << 48)), rl = (long)(rem.lo & 65535);
    ulong hk = p[HP_HFK], hfr = p[HP_HFFR], hrows = p[HP_HFROWS];
    f_put(hf, hk, hfr, hrows, T, k, d, 2, t, wv);
    f_put(hf, hk, hfr, hrows, T, 2 * d + k, d, 2, t, rh - 32896);
    f_put(hd, p[HP_HDK], p[HP_HDFR], T * p[HP_HDFR], T, k, d, 2, t, act);
    h_key(KY, PD, kd, (const device uint *)pos, HS_NF, k, t, T, (ulong)rl, false);
    h_key(KY, PD, kd, (const device uint *)pos, HS_NFH, k, t, T, (ulong)rh << (32 - KF), false);
    h_key(KY, PD, kd, (const device uint *)pos, HS_CNF, k, t, T, (ulong)(wv + 32768) + ((ulong)(act + 2048) << 28), false);)", a2, n2, {{1}}, d, T)[0];
  auto [a3, n3] = with({lamA, winA, coA, exA, Pa}, {"lam", "win", "co", "ex", "p"});
  const mx::array prices = run_kernel("fp_h_price", R"(
    ulong t = thread_position_in_grid.x, Vn = p[HP_V]; uint sh = (uint)p[HP_SH], F = (uint)p[HP_F];
    device long *o = (device long *)out;
    o[2 * t] = 0, o[2 * t + 1] = 0;
    if (!co[t]) return;
    const device long *L = lam + t * Vn;
    long w = (long)win[t], gap = 0;
    for (long v = 0; v < (long)Vn; v++) gap = max(gap, L[v] - L[w] + (v < w ? 1l : 0l));
    long low = 0, high = 32768;
    while (low < high) {
      long mid = (low + high) / 2; ulong a = 0;
      for (long v = 0; v < (long)Vn; v++) { long at = (L[w] + gap + (1l << sh) - L[v] - (v < w ? 1l : 0l)) >> sh; a += ex[min(at + mid, 65535l)]; }
      if (a <= 1ul << F) high = mid; else low = mid + 1;
    }
    ulong a = 0;
    for (long v = 0; v < (long)Vn; v++) { long at = (L[w] + gap + (1l << sh) - L[v] - (v < w ? 1l : 0l)) >> sh; a += ex[min(at + low, 65535l)]; }
    o[2 * t] = gap + ((1 + low) << sh), o[2 * t + 1] = (long)((1ul << F) - a);)", a3, n3, {{2 * T}}, T)[0];
  mx::eval({N, cols, prices});
  if (flags.data<uint64_t>()[0]) R.norm_ok = false;
  R.rows.resize(T);
  const int64_t *n = N.data<int64_t>();
  for (int t = 0; t < T; t++) {
    const int64_t *o = n + (size_t)t * WN_STRIDE;
    NormRow &r = R.rows[t];
    r.a = o[WN_A], r.b = o[WN_B], r.sig = o[WN_SIG], r.Rlo = o[WN_RLO], r.Rhi = o[WN_RHI], r.R = o[WN_R];
    r.D.assign(o + WN_D, o + WN_D + 99), r.W.assign(o + WN_W, o + WN_W + 9);
    if (r.R != V.get("aF_Rc", t, 0) || r.sig != V.get("aF_sigma", t, 0)) R.norm_ok = false;
  }
  R.price.assign(T, 0);
  std::vector<I64> spare(T, 0);
  for (int t = 0; t < T; t++) R.price[t] = prices.data<int64_t>()[2 * t], spare[t] = prices.data<int64_t>()[2 * t + 1];
  const std::vector<I64> lift = head_hidden(R, spare);
  std::vector<u64> liftw(lift.begin(), lift.end()), D;
  for (int t = 0; t < T; t++) D.insert(D.end(), R.rows[t].D.begin(), R.rows[t].D.end());
  auto [a4, n4] = with({w_table(D), Pa}, {"Dg", "p"});
  const mx::array rowsk = run_kernel("fp_h_rows", R"(
    ulong j = thread_position_in_grid.x, t = thread_position_in_grid.y, T = p[HP_T], NRH = p[HP_NRH];
    long D = (long)Dg[t * NRH + j];
    f_cell((device int *)c2, p[HP_HRK], p[HP_HRFR], T * p[HP_HRFR], T, j, t, (int)(D - 128));
    h_key((device ulong *)keys, (device uchar *)padf, kd, (const device uint *)pos, HS_ROW, j, t, T, w_row_mul((int)j) * (ulong)D, false);)", a4, n4, {{1}}, s.NRH, T)[0];
  auto [a5, n5] = with({lamA, winA, onA, exA, w_table(liftw), Pa}, {"lam", "win", "on_", "ex", "lf", "p"});
  const mx::array marg = run_kernel("fp_h_margins", R"(
    ulong v = thread_position_in_grid.x, t = thread_position_in_grid.y, T = p[HP_T], Vn = p[HP_V], JV = p[HP_JV], JVa = p[HP_JVA]; uint sh = (uint)p[HP_SH];
    device ulong *KY = (device ulong *)keys; device uchar *PD = (device uchar *)padf; device ulong *F = (device ulong *)fl; const device uint *PS = (const device uint *)pos;
    long K0 = 0, K1 = 0, K2 = 0; bool real = v < Vn && on_[t] != 0, padding = v >= Vn;
    if (real) {
      const device long *L = lam + t * Vn; long w = (long)win[t];
      long D = L[w] + (long)lf[t] - L[v] - ((long)v < w ? 1l : 0l);
      if (D < 0 || D >= (long)p[HP_REACH]) { F[1] = 1; D = 0; }
      long mm = D >> sh;
      K2 = D & ((1l << sh) - 1);
      K1 = mm < 32768 ? mm : 32768 + (mm & 32767);
      K0 = mm < 32768 ? 0 : (mm >> 15) - 1;
    }
    long e = (long)ex[K1];
    ulong hk = p[HP_HFK], hfr = p[HP_HFFR], hrows = p[HP_HFROWS]; device int *hf = (device int *)c0;
    f_put(hf, hk, hfr, hrows, T, p[HP_OD1] + v, JV, 2, t, real ? K1 - 32896 : 0);
    f_put(hf, hk, hfr, hrows, T, p[HP_OD2] + v, JV, 1, t, real ? K2 - 128 : 0);
    f_put(hf, hk, hfr, hrows, T, p[HP_OE] + v, JV, 3, t, e - (long)p[HP_OFFE]);
    h_key(KY, PD, kd, PS, HS_HD0, v, t, T, (ulong)K0, padding); h_key(KY, PD, kd, PS, HS_HD1, v, t, T, (ulong)K1, padding); h_key(KY, PD, kd, PS, HS_HD2, v, t, T, (ulong)K2, padding);
    ulong he = (ulong)K1 + ((ulong)e << 16);
    if (v < JVa) h_key(KY, PD, kd, PS, HS_HE0, v, t, T, he, padding); else h_key(KY, PD, kd, PS, HS_HE1, v - JVa, t, T, he, padding);)", a5, n5, {{1}}, JV, T)[0];
  mx::eval({rowsk, marg});
  if (flags.data<uint64_t>()[1]) R.margins_ok = false;
  for (int q = 0; q < 4; q++) {
    const int32_t *c = cells[q].data<int32_t>();
    for (CellMatrix2 &part : mats[q]->part) { memcpy(part.cell.data(), c, part.cell.size() * 4); c += part.cell.size(); }
  }
  for (size_t c = 0; c < R.calls.size(); c++) {
    memcpy(R.calls[c].keys.data(), keys.data<uint64_t>() + base[c], R.calls[c].keys.size() * 8);
    memcpy(R.calls[c].padding.data(), pad.data<uint8_t>() + base[c], R.calls[c].padding.size());
  }
  R.str = V.at("x_last"), R.act = V.at("aF"), R.lam = V.at("lam");
}
}
