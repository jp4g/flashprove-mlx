// Copyright 2026 Anthropic PBC
// SPDX-License-Identifier: Apache-2.0
// The witness of an attention segment A_g (docs/format/11-3-attention-segment.md): the matrices as, ah,
// ao, bA, and the keys of the calls of the lines `callA`. The verifier's side of the shape is
// verifier/src/vf_attention_shape.h. q, kx, v and o are read from the cells of qa, which the
// feed-forward segment of the group has filled (seg_ff_build.h).
#pragma once
#include <cmath>
#include "seg_ff_build.h"
namespace fpmlx {
enum { AS_RM, AS_RD, AS_FM, AS_FD, AS_WL, AS_WH, AS_IX };
struct APiece {
  int seg = 0;
  u64 lam = 0, a = 0;
  unsigned b = 0;
  u64 off = 0;
};
// A call of the expanded list.
struct ACallInfo {
  std::string name, kind;
  unsigned sh = 0;
  std::vector<int> segs;
  std::vector<APiece> pcs;
  u64 tot = 0;
  bool holds(int s) const { return std::find(segs.begin(), segs.end(), s) != segs.end(); }
};
struct AShape {
  int lt = 0, lB = 0, lnT = 0, lgk = 0, ldh = 0, ll = 0, la = 0, ls = 0, nM = 0, nD = 0, lmS = 0, lmbA = 0, nmin = 0, kFD = 0, S = 0, rx = 0;
  u64 T = 0, Bt = 0, nT = 0, hP = 0, H = 0, G = 0, HA = 0, dh = 0, dq = 0, dkv = 0, Lam = 0, NP = 0, S2 = 0, NAs = 0, wcH = 0, wcO = 0, wcQ = 0;
  u64 tau = 0, lidb = 0, u0 = 0, Smax = 0, kAs = 0, kPart = 0, featH = 0, featO = 0, oZr = 0, oIb = 0, oRhp = 0;
  std::vector<u64> Tt;
  std::vector<unsigned> sh; // by lam
  struct ShD { u64 C = 0, rho = 0, nHD = 0, Om = 0; };
  ShD shd(unsigned s) const {
    ShD d;
    d.C = u0 << s, d.rho = 65536 - (1ull << s), d.nHD = (4 * Smax + 1 + d.rho - 1) / d.rho, d.Om = 2 * Smax + d.C;
    return d;
  }
  unsigned segB(int s) const {
    switch (s) {
    case AS_RM: case AS_FM: return nM;
    case AS_RD: case AS_FD: return nD;
    case AS_WL: return 1 + lgk + lt + ldh;
    case AS_WH: return lgk + lt + ldh;
    default: return lgk + lt;
    }
  }
  // The tile (I, J), J < I, of the address (P, sg), sg < nT - 1.
  void cover(u64 P, u64 sg, u64 &I, u64 &J) const {
    if (sg < P) I = P, J = sg;
    else I = nT - 1 - P, J = sg - P;
  }
};
inline int ilog2_exact(u64 x) {
  const int l = (int)clog2u(x);
  if ((1ull << l) != x) throw std::runtime_error("attention: not a power of 2");
  return l;
}
inline int attn_seg_of(const std::string &n) {
  static const std::map<std::string, int> m{{"RM", AS_RM}, {"RD", AS_RD}, {"FM", AS_FM}, {"FD", AS_FD}, {"WL", AS_WL}, {"WH", AS_WH}, {"IX", AS_IX}};
  return m.at(n);
}
inline AShape attn_shape(const Policy &p, const ModelShape &m, int g) {
  AShape s;
  s.lt = (int)p.get("lt"), s.lB = (int)p.get("lB"), s.ll = (int)p.get("ll"), s.la = (int)p.get("la"), s.lmS = (int)p.get("lmS"), s.lmbA = (int)p.get("lmbA");
  s.nmin = (int)p.get("nmin"), s.kFD = (int)p.get("kFD"), s.S = (int)p.get("S"), s.rx = (int)p.get("rx");
  s.H = (u64)p.get("H"), s.G = (u64)p.get("G"), s.dh = (u64)p.get("dh"), s.NAs = (u64)p.get("NAs"), s.tau = (u64)p.get("tau"), s.lidb = (u64)p.get("lidb");
  s.T = 1ull << s.lt, s.Bt = 1ull << s.lB, s.lnT = s.lt - s.lB, s.nT = 1ull << s.lnT, s.hP = s.nT / 2;
  s.lgk = ilog2_exact(s.G), s.ldh = ilog2_exact(s.dh), s.HA = s.H / s.G, s.Lam = 1ull << s.ll, s.NP = s.Lam * s.HA;
  s.dq = s.H * s.dh, s.dkv = s.G * s.dh, s.ls = s.la - s.lt, s.S2 = 1ull << s.ls;
  s.nM = s.lgk + 2 * s.lt - 1, s.nD = s.lgk + s.lt + s.lB;
  s.wcH = 1ull << (p.get("lmh") - s.lt), s.wcO = 1ull << (p.get("lmo") - s.lt), s.wcQ = 1ull << (p.get("lmq") - s.lt);
  for (u64 u = 0;; u++) {
    const double v = nearbyint(ldexp(1.0, s.S) * exp(-(double)u / ldexp(1.0, s.rx)));
    if (v == 0) { s.u0 = u; break; }
    s.Tt.push_back((u64)v);
  }
  s.Smax = s.dh * 2047 * 2047;
  for (u64 l = 0; l < s.Lam; l++) s.sh.push_back((unsigned)m.layers[(size_t)g * s.Lam + l].sh);
  const u64 cells = 5 * s.NP * ((1ull << s.nM) + (1ull << s.nD)), unit = s.NAs << s.lmS;
  s.kAs = (cells + unit - 1) / unit * s.NAs, s.kPart = s.kAs / s.NAs;
  s.oZr = 4 * s.Lam * s.H, s.oIb = 8 * s.Lam * s.H, s.featH = s.oIb + s.Lam * s.lt * s.H;
  s.oRhp = s.Lam * s.dq, s.featO = 2 * s.Lam * s.dq;
  return s;
}
struct AttnGroup {
  AShape s;
  int g = 0;
  std::vector<CellMatrix2> as;  // the NAs parts
  FeatMat ah, ao;
  CellMatrix2 bA;
  std::vector<ACallInfo> ac;    // the expanded list
  std::vector<LookupCall> calls;
  LookupSizes sizes;
  std::vector<std::vector<u32>> mult;
  bool keys_ok = false;
  // Values by layer: q, kx, v, o as in qa ([lam][token * width + entry]).
  std::vector<std::vector<I64>> Q, K, Vv, O;
  // By (lam H + h): Mx, Zd, ib at token i; sc at (i, j); acc and rem at (i, ch).
  std::vector<std::vector<I64>> Mx, Zd, ib, sc, acc, rem;
  // The cell of plane x of the pair n of (lam, a).
  int32_t &plane(u64 x, u64 lam, u64 a, bool isD, u64 n) {
    const u64 N = isD ? ((5 * s.NP) << s.nM) + (((x * s.Lam + lam) * s.HA + a) << s.nD) + n : (((x * s.Lam + lam) * s.HA + a) << s.nM) + n;
    const u64 col = N >> s.lmS;
    return as[col / s.kPart].at(N & ((1ull << s.lmS) - 1), col % s.kPart);
  }
  int32_t plane(u64 x, u64 lam, u64 a, bool isD, u64 n) const { return const_cast<AttnGroup *>(this)->plane(x, lam, a, isD, n); }
  I64 ew(u64 lam, u64 a, bool isD, u64 n) const { return (I64)plane(3, lam, a, isD, n) + 128 + 256 * (I64)plane(4, lam, a, isD, n); }
  // rh and rhp of (lam, h) at (i, ch): the cells of ao.
  I64 rh(u64 lam, u64 h, u64 i, u64 ch) const { const I64 r = rem[lam * s.H + h][i * s.dh + ch]; return (r >> 16) - 128; }
  I64 rhp(u64 lam, u64 h, u64 i, u64 ch) const { const u64 b = lam * s.H + h; const I64 r = 2 * Zd[b][i] - 1 - rem[b][i * s.dh + ch]; return (r >> 16) - 128; }
  // Bit k of ib(i).
  I64 ibit(u64 lam, u64 h, u64 i, int k) const { return (ib[lam * s.H + h][i] >> k) & 1; }
};
// The expanded list, its pieces and its tables.
inline void attn_calls(const Policy &p, AttnGroup &R) {
  const AShape &s = R.s;
  std::vector<unsigned> shs(s.sh);
  std::sort(shs.begin(), shs.end());
  shs.erase(std::unique(shs.begin(), shs.end()), shs.end());
  const bool one = shs.size() == 1;
  for (const auto &l : p.callA) {
    std::vector<int> segs;
    for (const std::string &n : l.segs) segs.push_back(attn_seg_of(n));
    auto mk = [&](const std::string &name, unsigned sh) { ACallInfo c; c.name = name, c.kind = l.kind, c.sh = sh, c.segs = segs; R.ac.push_back(c); };
    if (l.kind[0] == 'K') for (unsigned sh : shs) mk(l.name + (one ? std::string() : std::to_string(sh)), sh);
    else mk(l.name, 0);
  }
  R.calls.resize(R.ac.size());
  for (size_t q = 0; q < R.ac.size(); q++) {
    ACallInfo &c = R.ac[q];
    const bool by_sh = c.kind[0] == 'K';
    for (int seg : c.segs)
      for (u64 lam = 0; lam < s.Lam; lam++) {
        if (by_sh && s.sh[lam] != c.sh) continue;
        for (u64 a = 0; a < s.HA; a++) { APiece x; x.seg = seg, x.lam = lam, x.a = a, x.b = s.segB(seg); c.pcs.push_back(x); }
      }
    std::stable_sort(c.pcs.begin(), c.pcs.end(), [](const APiece &x, const APiece &y) { return x.b > y.b; });
    u64 off = 0;
    for (APiece &x : c.pcs) x.off = off, off += 1ull << x.b;
    c.tot = off;
    LookupCall &K = R.calls[q];
    K.n = std::max<unsigned>(clog2u(off), (unsigned)s.nmin);
    if (c.kind == "KR" || c.kind == "WH") {
      K.k = 16, K.table.resize(65536);
      for (u64 t = 0; t < 65536; t++) K.table[t] = (t & 255) + ((t >> 8) << 20) + (c.kind == "KR" ? (u64)(t >= (1ull << c.sh)) << 40 : 0);
    } else if (c.kind == "R16") {
      K.k = 16;
    } else {
      const AShape::ShD d = s.shd(c.sh);
      K.k = (unsigned)s.kFD, K.t0 = s.u0, K.table.resize(1ull << K.k);
      for (u64 t = 0; t < K.table.size(); t++)
        K.table[t] = t < s.u0 ? (t << c.sh) + (s.Tt[t] << 33)
                   : t < s.u0 + d.nHD ? d.C - (1ull << c.sh) + d.rho * (t - s.u0) + (1ull << 32)
                                      : d.C - (1ull << c.sh) + d.rho * (t - s.u0 - d.nHD) + d.rho / 2 + (1ull << 32);
    }
    K.keys.assign(1ull << K.n, K.row(K.t0)), K.padding.assign(1ull << K.n, 1);
  }
}
// The lookup's sizes, multiplicities and planes, from the keys.
inline AttnGroup attn_finish(AttnGroup &R) {
  const AShape &s = R.s;
  R.sizes = lookup_sizes(R.calls, (unsigned)s.lidb);
  R.keys_ok = lookup_multiplicities(R.calls, R.sizes, R.mult);
  const unsigned kT = R.sizes.kT;
  R.bA = CellMatrix2(1ull << s.lmbA, (u64)R.sizes.NC << (kT + 5 - s.lmbA));
  if (R.keys_ok)
    for (unsigned c = 0; c < R.sizes.NC; c++)
      for (u64 t = 0; t < (1ull << kT); t++)
        for (u64 j = 0; j < 32; j++)
          if ((R.mult[c][t] >> j) & 1) {
            const u64 S = (((u64)c << kT) + t) * 32 + j;
            R.bA.at(S & ((1ull << s.lmbA) - 1), S >> s.lmbA) = 1;
          }
  return std::move(R);
}
inline void attn_fill_gpu(AttnGroup &R, const Values &V, const std::array<std::map<u64, std::pair<size_t, u64>>, 7> &at);
// V given: the values come from the forward pass and the witness is made on Metal. Without it: on the host,
// from the cells of qa; attn_keys_test compares the two.
inline AttnGroup attn_build(const Policy &pol, const ModelShape &m, FFGroup &ff, int g, const Values *fromV = nullptr) {
  AttnGroup R;
  R.s = attn_shape(pol, m, g);
  R.g = g;
  const AShape &s = R.s;
  const u64 T = s.T, Bt = s.Bt, nT = s.nT, hP = s.hP, H = s.H, HA = s.HA, dh = s.dh, dq = s.dq, dkv = s.dkv, Lam = s.Lam;
  attn_calls(pol, R);
  R.as.assign(s.NAs, CellMatrix2(1ull << s.lmS, s.kPart));
  R.ah.init(1, (int)pol.get("lmh"), (long long)s.featH, s.lt);
  R.ao.init(1, (int)pol.get("lmo"), (long long)s.featO, s.lt);
  // Where each piece's keys go: by segment and by (lam, a).
  std::array<std::map<u64, std::pair<size_t, u64>>, 7> at;
  for (size_t q = 0; q < R.ac.size(); q++)
    for (const APiece &x : R.ac[q].pcs) at[x.seg][x.lam * HA + x.a] = {q, x.off};
  auto key = [&](int seg, u64 lam, u64 a, u64 n, u64 v) {
    const auto &[q, off] = at[seg].at(lam * HA + a);
    R.calls[q].keys[off + n] = v, R.calls[q].padding[off + n] = 0;
  };
  if (fromV) { attn_fill_gpu(R, *fromV, at); return attn_finish(R); }
  // Values, from the cells of qa.
  auto read = [&](const char *name, u64 lam, u64 f, u64 t) {
    const Tensor2 &X = ff.tf.at(name);
    return (I64)ff.qa.cell(X.feat(0, (int)lam, (int)f), (int)t) + 256 * (I64)ff.qa.cell(X.feat(1, (int)lam, (int)f), (int)t);
  };
  R.Q.assign(Lam, std::vector<I64>(T * dq)), R.O = R.Q;
  R.K.assign(Lam, std::vector<I64>(T * dkv)), R.Vv = R.K;
  for (u64 lam = 0; lam < Lam; lam++)
    for (u64 t = 0; t < T; t++) {
      for (u64 f = 0; f < dq; f++) R.Q[lam][t * dq + f] = read("q", lam, f, t), R.O[lam][t * dq + f] = read("o", lam, f, t);
      for (u64 f = 0; f < dkv; f++) R.K[lam][t * dkv + f] = read("kx", lam, f, t), R.Vv[lam][t * dkv + f] = read("v", lam, f, t);
    }
  const size_t NH = Lam * H;
  R.Mx.assign(NH, std::vector<I64>(T)), R.Zd = R.Mx, R.ib = R.Mx;
  R.sc.assign(NH, std::vector<I64>(T * T));
  R.acc.assign(NH, std::vector<I64>(T * dh)), R.rem = R.acc;
  const Tensor2 tMx{0, 4, (int)Lam, (int)H}, tZd{(int)s.oZr, 4, (int)Lam, (int)H}, tIb{(int)s.oIb, 1, (int)Lam, s.lt * (int)H};
  const Tensor2 tRh{0, 1, (int)Lam, (int)dq}, tRhp{(int)s.oRhp, 1, (int)Lam, (int)dq};
  for (u64 lam = 0; lam < Lam; lam++) {
    const unsigned sh = s.sh[lam];
    const AShape::ShD D = s.shd(sh);
    // The pair's five cells and its keys.
    auto pair = [&](u64 a, bool isD, u64 n, I64 Df) {
      u64 rm, fl, ew;
      if (Df < (I64)D.C) rm = (u64)Df & ((1ull << sh) - 1), fl = 0, ew = s.Tt[(u64)Df >> sh];
      else rm = (1ull << sh) + ((u64)Df - D.C) % D.rho, fl = 1, ew = 0;
      R.plane(0, lam, a, isD, n) = (int32_t)(rm & 255) - 128;
      R.plane(1, lam, a, isD, n) = (int32_t)(rm >> 8) - 128;
      R.plane(2, lam, a, isD, n) = (int32_t)fl;
      R.plane(3, lam, a, isD, n) = (int32_t)(ew & 255) - 128;
      R.plane(4, lam, a, isD, n) = (int32_t)(ew >> 8);
      key(isD ? AS_RD : AS_RM, lam, a, n, (rm & 255) + ((rm >> 8) << 20) + (fl << 40));
      key(isD ? AS_FD : AS_FM, lam, a, n, (u64)(Df - (I64)rm) + (fl << 32) + (ew << 33));
    };
    for (u64 h = 0; h < H; h++) {
      const u64 hk = h / HA, a = h % HA, bi = lam * H + h;
      std::vector<I64> &Mx = R.Mx[bi], &Zd = R.Zd[bi], &ib = R.ib[bi], &sc = R.sc[bi], &acc = R.acc[bi], &rem = R.rem[bi];
      for (u64 i = 0; i < T; i++) {
        for (u64 j = 0; j < T; j++) {
          I64 t = 0;
          for (u64 c = 0; c < dh; c++) t += R.Q[lam][i * dq + h * dh + c] * R.K[lam][j * dkv + hk * dh + c];
          sc[i * T + j] = t;
        }
        u64 best = 0;
        for (u64 j = 1; j <= i; j++) if (sc[i * T + j] > sc[i * T + best]) best = j;
        Mx[i] = sc[i * T + best], ib[i] = (I64)best;
        for (u64 j = 0; j <= i; j++) {
          const I64 Df = Mx[i] - sc[i * T + j];
          if (Df >= (I64)D.C) continue;
          const I64 e = (I64)s.Tt[(u64)Df >> sh];
          Zd[i] += e;
          for (u64 c = 0; c < dh; c++) acc[i * dh + c] += e * R.Vv[lam][j * dkv + hk * dh + c];
        }
        int32_t c4[4];
        limbs_of(Mx[i], 4, c4);
        for (int l = 0; l < 4; l++) R.ah.cell(tMx.feat(l, (int)lam, (int)h), (int)i) = c4[l];
        limbs_of(Zd[i], 4, c4);
        for (int l = 0; l < 4; l++) R.ah.cell(tZd.feat(l, (int)lam, (int)h), (int)i) = c4[l];
        for (int k = 0; k < s.lt; k++) R.ah.cell(tIb.feat(0, (int)lam, k * (int)H + (int)h), (int)i) = (int32_t)((best >> k) & 1);
        key(AS_IX, lam, a, (hk << s.lt) + i, (u64)(i - best) << (16 - s.lt));
        for (u64 c = 0; c < dh; c++) {
          const I64 r = 2 * acc[i * dh + c] + Zd[i] - 2 * Zd[i] * R.O[lam][i * dq + h * dh + c], r2 = 2 * Zd[i] - 1 - r;
          rem[i * dh + c] = r;
          const I64 rhv = (r >> 16) - 128, rpv = (r2 >> 16) - 128;
          R.ao.cell(tRh.feat(0, (int)lam, (int)(h * dh + c)), (int)i) = (int32_t)rhv;
          R.ao.cell(tRhp.feat(0, (int)lam, (int)(h * dh + c)), (int)i) = (int32_t)rpv;
          const u64 ix = ((hk << s.lt) + i) * dh + c;
          key(AS_WL, lam, a, ix, (u64)(r - 65536 * (rhv + 128)));
          key(AS_WL, lam, a, (s.G << (s.lt + s.ldh)) + ix, (u64)(r2 - 65536 * (rpv + 128)));
          key(AS_WH, lam, a, ix, (u64)(rhv + 128) + ((u64)(rpv + 128) << 20));
        }
      }
      // Region M.
      for (u64 P = 0; P < hP; P++)
        for (u64 sg = 0; sg < nT; sg++) {
          u64 I = 0, J = 0;
          if (sg + 1 < nT) s.cover(P, sg, I, J);
          for (u64 il = 0; il < Bt; il++)
            for (u64 jl = 0; jl < Bt; jl++) {
              const u64 n = ((((hk * hP + P) * nT + sg) * Bt + il) * Bt) + jl;
              const u64 i = I * Bt + il, j = J * Bt + jl;
              pair(a, false, n, sg + 1 < nT ? Mx[i] - sc[i * T + j] : (I64)D.C);
            }
        }
      // Region D.
      for (u64 I = 0; I < nT; I++)
        for (u64 il = 0; il < Bt; il++)
          for (u64 jl = 0; jl < Bt; jl++) {
            const u64 n = (((hk * nT + I) * Bt + il) * Bt) + jl, i = I * Bt + il, j = I * Bt + jl;
            pair(a, true, n, Mx[i] - sc[i * T + j] + (jl > il ? (I64)D.Om : 0));
          }
    }
  }
  return attn_finish(R);
}
inline void attn_fill_gpu(AttnGroup &R, const Values &V, const std::array<std::map<u64, std::pair<size_t, u64>>, 7> &at) {
  const AShape &s = R.s;
  const u64 T = s.T, Bt = s.Bt, nT = s.nT, hP = s.hP, H = s.H, HA = s.HA, dh = s.dh, dq = s.dq, dkv = s.dkv, Lam = s.Lam, NH = Lam * H;
  // Keys: their padding's values are already in place.
  std::vector<u64> base(R.calls.size());
  u64 nk = 0;
  for (size_t c = 0; c < R.calls.size(); c++) base[c] = nk, nk += u64(1) << R.calls[c].n;
  mx::array keys = w_buffer(nk * 8, mx::uint64, 0), pad = w_buffer(nk, mx::uint8, 1);
  for (size_t c = 0; c < R.calls.size(); c++) memcpy(keys.data<uint64_t>() + base[c], R.calls[c].keys.data(), R.calls[c].keys.size() * 8);
  const u64 nseg = Lam * HA;
  std::vector<u64> kb(7 * nseg, 0);
  for (int q = 0; q < 7; q++) for (auto &[where, x] : at[q]) kb[q * nseg + where] = base[x.first] + x.second;
  mx::array as = w_buffer(s.NAs * R.as[0].cell.size() * 4, mx::int32, 0);
  mx::array ah = w_buffer(R.ah.part[0].cell.size() * 4, mx::int32, 0), ao = w_buffer(R.ao.part[0].cell.size() * 4, mx::int32, 0);
  mx::array sc = w_buffer(NH * T * T * 8, mx::int64, 0), acc = w_buffer(NH * T * dh * 8, mx::int64, 0), rem = w_buffer(NH * T * dh * 8, mx::int64, 0), mzi = w_buffer(3 * NH * T * 8, mx::int64, 0);
  std::vector<u64> P(AP_COUNT, 0);
  P[AP_T] = T, P[AP_BT] = Bt, P[AP_NT] = nT, P[AP_HP] = hP, P[AP_H] = H, P[AP_HA] = HA, P[AP_DH] = dh, P[AP_DQ] = dq, P[AP_DKV] = dkv, P[AP_LAMS] = Lam, P[AP_G] = s.G, P[AP_LT] = s.lt, P[AP_LDH] = s.ldh;
  P[AP_NM] = s.nM, P[AP_ND] = s.nD, P[AP_NP] = s.NP, P[AP_LMS] = s.lmS, P[AP_KPART] = s.kPart, P[AP_NSEG] = nseg, P[AP_AHK] = R.ah.k, P[AP_AHFR] = R.ah.fr, P[AP_AOK] = R.ao.k, P[AP_AOFR] = R.ao.fr;
  P[AP_OZR] = s.oZr, P[AP_OIB] = s.oIb, P[AP_ORHP] = s.oRhp, P[AP_U0] = s.u0;
  const mx::array kbA = w_table(kb), ttA = w_table(s.Tt);
  std::vector<mx::array> done;
  auto L = [&](const char *n, u64 lam) { return std::string(n) + "." + std::to_string((u64)R.g * Lam + lam); };
  const std::vector<std::string> names1{"as_", "ah", "ao", "keys", "padf", "kb", "tt", "sc", "acc", "rem", "mzi", "Q", "K", "Vv", "O", "p"};
  for (u64 lam = 0; lam < Lam; lam++) {
    const AShape::ShD D = s.shd(s.sh[lam]);
    P[AP_LAM] = lam, P[AP_SH] = s.sh[lam], P[AP_C] = D.C, P[AP_RHO] = D.rho, P[AP_OM] = D.Om;
    const mx::array Pa = w_table(P);
    const mx::array one = run_kernel("fp_a_rows", R"(
      ulong i = thread_position_in_grid.x, h = thread_position_in_grid.y, T = p[AP_T], H = p[AP_H], HA = p[AP_HA], dh = p[AP_DH], dq = p[AP_DQ], dkv = p[AP_DKV], lam = p[AP_LAM], Lam = p[AP_LAMS], lt = p[AP_LT], sh = p[AP_SH], C = p[AP_C], G = p[AP_G], ldh = p[AP_LDH];
      ulong hk = h / HA, a = h % HA, bi = lam * H + h;
      device long *scr = (device long *)sc + (bi * T + i) * T, *ac = (device long *)acc + (bi * T + i) * dh, *re = (device long *)rem + (bi * T + i) * dh, *m3 = (device long *)mzi;
      device int *AH = (device int *)ah, *AO = (device int *)ao; device ulong *KY = (device ulong *)keys; device uchar *PD = (device uchar *)padf;
      for (ulong j = 0; j < T; j++) { long t = 0; for (ulong c = 0; c < dh; c++) t += Q[i * dq + h * dh + c] * K[j * dkv + hk * dh + c]; scr[j] = t; }
      ulong best = 0;
      for (ulong j = 1; j <= i; j++) if (scr[j] > scr[best]) best = j;
      long Mx = scr[best], Zd = 0;
      for (ulong j = 0; j <= i; j++) {
        long Df = Mx - scr[j];
        if (Df >= (long)C) continue;
        long e = (long)tt[(ulong)Df >> sh];
        Zd += e;
        for (ulong c = 0; c < dh; c++) ac[c] += e * Vv[j * dkv + hk * dh + c];
      }
      m3[(0 * Lam * H + bi) * T + i] = Mx, m3[(1 * Lam * H + bi) * T + i] = Zd, m3[(2 * Lam * H + bi) * T + i] = (long)best;
      ulong rows = T * p[AP_AHFR], k = p[AP_AHK], fr = p[AP_AHFR];
      f_put(AH, k, fr, rows, T, lam * H + h, Lam * H, 4, i, Mx);
      f_put(AH, k, fr, rows, T, p[AP_OZR] + lam * H + h, Lam * H, 4, i, Zd);
      for (ulong b = 0; b < lt; b++) f_cell(AH, k, fr, rows, T, p[AP_OIB] + lam * lt * H + b * H + h, i, (int)((best >> b) & 1));
      a_key(KY, PD, kb, p, A_IX, lam, a, (hk << lt) + i, (ulong)(i - best) << (16 - lt));
      ulong k2 = p[AP_AOK], fr2 = p[AP_AOFR], rows2 = T * fr2;
      for (ulong c = 0; c < dh; c++) {
        long r = 2 * ac[c] + Zd - 2 * Zd * O[i * dq + h * dh + c], r2 = 2 * Zd - 1 - r, rhv = (r >> 16) - 128, rpv = (r2 >> 16) - 128;
        re[c] = r;
        f_cell(AO, k2, fr2, rows2, T, lam * dq + h * dh + c, i, (int)rhv);
        f_cell(AO, k2, fr2, rows2, T, p[AP_ORHP] + lam * dq + h * dh + c, i, (int)rpv);
        ulong ix = ((hk << lt) + i) * dh + c;
        a_key(KY, PD, kb, p, A_WL, lam, a, ix, (ulong)(r - 65536 * (rhv + 128)));
        a_key(KY, PD, kb, p, A_WL, lam, a, (G << (lt + ldh)) + ix, (ulong)(r2 - 65536 * (rpv + 128)));
        a_key(KY, PD, kb, p, A_WH, lam, a, ix, (ulong)(rhv + 128) + ((ulong)(rpv + 128) << 20));
      })", {as, ah, ao, keys, pad, kbA, ttA, sc, acc, rem, mzi, V.g.at(L("q", lam)), V.g.at(L("k", lam)), V.g.at(L("v", lam)), V.g.at(L("o", lam)), Pa}, names1, {{1}}, T, H)[0];
    done.push_back(one);
    const std::vector<std::string> names2{"as_", "keys", "padf", "kb", "tt", "sc", "mzi", "dep", "p"};
    done.push_back(run_kernel("fp_a_regionM", R"(
      ulong x = thread_position_in_grid.x, h = thread_position_in_grid.y, T = p[AP_T], Bt = p[AP_BT], nT = p[AP_NT], hP = p[AP_HP], H = p[AP_H], HA = p[AP_HA], lam = p[AP_LAM], Lam = p[AP_LAMS];
      ulong hk = h / HA, a = h % HA, bi = lam * H + h, jl = x % Bt, il = (x / Bt) % Bt, sg = (x / (Bt * Bt)) % nT, P = x / (Bt * Bt * nT);
      ulong I = 0, J = 0;
      if (sg + 1 < nT) { if (sg < P) I = P, J = sg; else I = nT - 1 - P, J = sg - P; }
      ulong n = ((((hk * hP + P) * nT + sg) * Bt + il) * Bt) + jl, i = I * Bt + il, j = J * Bt + jl;
      long Df = sg + 1 < nT ? mzi[bi * T + i] - sc[(bi * T + i) * T + j] : (long)p[AP_C];
      a_pair((device int *)as_, (device ulong *)keys, (device uchar *)padf, kb, tt, p, lam, a, false, n, Df);)",
      {as, keys, pad, kbA, ttA, sc, mzi, one, Pa}, names2, {{1}}, hP * nT * Bt * Bt, H)[0]);
    done.push_back(run_kernel("fp_a_regionD", R"(
      ulong x = thread_position_in_grid.x, h = thread_position_in_grid.y, T = p[AP_T], Bt = p[AP_BT], nT = p[AP_NT], H = p[AP_H], HA = p[AP_HA], lam = p[AP_LAM];
      ulong hk = h / HA, a = h % HA, bi = lam * H + h, jl = x % Bt, il = (x / Bt) % Bt, I = x / (Bt * Bt);
      ulong n = (((hk * nT + I) * Bt + il) * Bt) + jl, i = I * Bt + il, j = I * Bt + jl;
      long Df = mzi[bi * T + i] - sc[(bi * T + i) * T + j] + (jl > il ? (long)p[AP_OM] : 0);
      a_pair((device int *)as_, (device ulong *)keys, (device uchar *)padf, kb, tt, p, lam, a, true, n, Df);)",
      {as, keys, pad, kbA, ttA, sc, mzi, one, Pa}, names2, {{1}}, nT * Bt * Bt, H)[0]);
  }
  mx::eval(done);
  { const int32_t *c = as.data<int32_t>(); for (CellMatrix2 &x : R.as) memcpy(x.cell.data(), c, x.cell.size() * 4), c += x.cell.size(); }
  memcpy(R.ah.part[0].cell.data(), ah.data<int32_t>(), R.ah.part[0].cell.size() * 4);
  memcpy(R.ao.part[0].cell.data(), ao.data<int32_t>(), R.ao.part[0].cell.size() * 4);
  for (size_t c = 0; c < R.calls.size(); c++) {
    memcpy(R.calls[c].keys.data(), keys.data<uint64_t>() + base[c], R.calls[c].keys.size() * 8);
    memcpy(R.calls[c].padding.data(), pad.data<uint8_t>() + base[c], R.calls[c].padding.size());
  }
  R.Q.resize(Lam), R.K.resize(Lam), R.Vv.resize(Lam), R.O.resize(Lam);
  for (u64 lam = 0; lam < Lam; lam++) R.Q[lam] = V.at(L("q", lam)), R.K[lam] = V.at(L("k", lam)), R.Vv[lam] = V.at(L("v", lam)), R.O[lam] = V.at(L("o", lam));
  R.Mx.resize(NH), R.Zd.resize(NH), R.ib.resize(NH), R.sc.resize(NH), R.acc.resize(NH), R.rem.resize(NH);
  const int64_t *m3 = mzi.data<int64_t>();
  for (u64 b = 0; b < NH; b++) {
    R.Mx[b].assign(m3 + b * T, m3 + (b + 1) * T);
    R.Zd[b].assign(m3 + (NH + b) * T, m3 + (NH + b + 1) * T);
    R.ib[b].assign(m3 + (2 * NH + b) * T, m3 + (2 * NH + b + 1) * T);
    R.sc[b].assign(sc.data<int64_t>() + b * T * T, sc.data<int64_t>() + (b + 1) * T * T);
    R.acc[b].assign(acc.data<int64_t>() + b * T * dh, acc.data<int64_t>() + (b + 1) * T * dh);
    R.rem[b].assign(rem.data<int64_t>() + b * T * dh, rem.data<int64_t>() + (b + 1) * T * dh);
  }
}
}
